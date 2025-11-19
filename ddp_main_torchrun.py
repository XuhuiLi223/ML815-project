import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from utils import define_model, load_resized_data, get_training_metrics, print_metrics, reset_peak_memory
from argument import get_args
from task import get_task_criterion, get_task_eval_fn, print_task_metrics, adapt_outputs_for_task


def prepare():
    args = get_args(gpu_default="2,3")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


def get_ddp_generator(seed=3407):
    local_rank = int(os.environ["LOCAL_RANK"])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def train(model, train_dloader, criterion, optimizer, scaler, task_type='classification'):
    model.train()
    samples_processed = 0
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        # Adapt outputs and labels for different task types
        adapted_outputs, adapted_labels = adapt_outputs_for_task(outputs, labels, task_type)
        loss = criterion(adapted_outputs, adapted_labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()  ###
        scaler.step(optimizer)  ###
        scaler.update()  ###
        samples_processed += images.size(0)
    return samples_processed


def test(model, test_dloader, task_type):
    """Test model using task-specific evaluation function"""
    local_rank = int(os.environ["LOCAL_RANK"])
    eval_fn = get_task_eval_fn(task_type)
    metrics = eval_fn(model, test_dloader, device='cuda', is_distributed=True)
    if local_rank == 0:
        print_task_metrics(metrics, task_type, prefix="Test ")


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # Rank 0 loads first to avoid concurrent downloads
    if local_rank == 0:
        train_dataset, val_loader = load_resized_data(args)
        nclass = train_dataset.nclass
        # Broadcast nclass to all ranks before other ranks load
        nclass_tensor = torch.tensor(nclass, dtype=torch.int).cuda()
        dist.broadcast(nclass_tensor, 0)
    else:
        # Other ranks wait for nclass from rank 0
        nclass_tensor = torch.tensor(0, dtype=torch.int).cuda()
        dist.broadcast(nclass_tensor, 0)
        # Now load dataset (nclass is already known from broadcast)
        train_dataset, val_loader = load_resized_data(args)
    
    nclass = nclass_tensor.item()
    
    # Reset peak memory before training (only on rank 0)
    if local_rank == 0:
        reset_peak_memory()
    
    model = define_model(args, nclass).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )  
    # Get task-specific criterion
    criterion = get_task_criterion(args.task, num_classes=nclass, device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    scaler = GradScaler()  ### Used for mixed precision training
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset
    )  ### Sampler specifically for DDP
    g = get_ddp_generator()  
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  ### The shuffle operation is performed by the sampler
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        generator=g,
    )  ### 添加额外的 generator
    
    # Create test sampler and dataloader
    test_dataset = val_loader.dataset
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, shuffle=False
    )  ### Sampler specifically for DDP
    test_dloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=test_sampler,
    )
    
    # Track training metrics
    total_samples = 0
    if local_rank == 0:
        training_start_time = time.time()
    
    for epoch in range(args.epochs):
        if local_rank == 0:  ### avoid redundant printing for each process
            print(f"begin training of epoch {epoch + 1}/{args.epochs}")
        train_dloader.sampler.set_epoch(epoch)  ### Set the epoch for the sampler
        samples = train(model, train_dloader, criterion, optimizer, scaler, args.task)
        # Sum samples across all ranks
        samples_tensor = torch.tensor(samples, dtype=torch.long).cuda()
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            total_samples += samples_tensor.item()
    
    if local_rank == 0:
        training_time = time.time() - training_start_time
        print(f"begin testing")
    
    test(model, test_dloader, args.task)
    if local_rank == 0:  ### avoid redundant saving
        torch.save(
            {"model": model.state_dict(), "scaler": scaler.state_dict()},
            "ddp_checkpoint.pt",
        )
        
        # Calculate and print metrics
        num_gpus = dist.get_world_size()
        input_shape = (1, args.nch, args.size, args.size)
        # Get the underlying model (unwrap DDP)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        metrics = get_training_metrics(
            model=unwrapped_model,
            samples_processed=total_samples,
            time_elapsed=training_time,
            input_shape=input_shape,
            num_gpus=num_gpus,
            device='cuda'
        )
        print_metrics(metrics, prefix="Training ")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    args = prepare()
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(f"\ntime elapsed: {time_elapsed:.2f} seconds")
