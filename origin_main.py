import time
import os
import torch
import torch.nn as nn
from utils import define_model, load_resized_data, get_training_metrics, print_metrics, reset_peak_memory
from argument import get_args
from task import get_task_criterion, get_task_eval_fn, print_task_metrics, adapt_outputs_for_task


def prepare():
    args = get_args(gpu_default="0")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


def train(model, train_dloader, criterion, optimizer, task_type='classification'):
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
        loss.backward()
        optimizer.step()
        samples_processed += images.size(0)
    return samples_processed


def test(model, test_dloader, task_type):
    """Test model using task-specific evaluation function"""
    eval_fn = get_task_eval_fn(task_type)
    metrics = eval_fn(model, test_dloader, device='cuda', is_distributed=False)
    print_task_metrics(metrics, task_type, prefix="Test ")


def main(args):
    # Reset peak memory before training
    reset_peak_memory()
    
    # Load datasets
    train_dataset, val_loader = load_resized_data(args)
    nclass = train_dataset.nclass
    
    # Get model
    model = define_model(args, nclass).cuda()
    
    # Get task-specific criterion
    criterion = get_task_criterion(args.task, num_classes=nclass, device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    # Create train dataloader
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Use val_loader as test_dloader
    test_dloader = val_loader
    
    # Track training metrics
    total_samples = 0
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"begin training of epoch {epoch + 1}/{args.epochs}")
        samples = train(model, train_dloader, criterion, optimizer, args.task)
        total_samples += samples
    
    training_time = time.time() - training_start_time
    
    print(f"begin testing")
    test(model, test_dloader, args.task)
    torch.save({"model": model.state_dict()}, "origin_checkpoint.pt")
    
    # Calculate and print metrics
    input_shape = (1, args.nch, args.size, args.size)
    metrics = get_training_metrics(
        model=model.module if hasattr(model, 'module') else model,
        samples_processed=total_samples,
        time_elapsed=training_time,
        input_shape=input_shape,
        num_gpus=1,
        device='cuda'
    )
    print_metrics(metrics, prefix="Training ")


if __name__ == "__main__":
    args = prepare()
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    print(f"\ntime elapsed: {time_elapsed:.2f} seconds")
