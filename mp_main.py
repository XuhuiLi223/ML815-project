# mp_main.py
# -*- coding: utf-8 -*-
"""
Model Parallel (Tensor/ Intra-layer) training entry for ML815-project.

- Differentiates from pipeline/hybrid parallel: we split Conv2d/Linear along
  the OUT channel/feature dimension across multiple GPUs within the SAME layer.
- Does NOT alter your existing model/forward code; wrapping is done module-wise.
- Robust dataset/class detection, automatic fixes for common pitfalls:
  * First Conv2d in_channels (MNIST→CIFAR10: 1→3)
  * Last Linear in_features (28x28→32x32: 1568↔2048)
  * Last Linear out_features = num_classes
"""

import os
import time
from typing import Tuple, Sequence

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Repo-local modules
import argument   # expected: get_args() or parse_args()
import utils      # expected: load_resized_data(args) or load_data(args); define_model(args, ...)
import task       # expected: get_task_criterion(args), adapt_outputs_for_task(...)

# The tensor-parallel wrapper file you added
from mp_tensor_parallel import wrap_model_tensor_parallel


# ------------------------------
# Utilities
# ------------------------------
def parse_devices(gpu_arg: str) -> Tuple[torch.device, Sequence[torch.device]]:
    ids = [s.strip() for s in str(gpu_arg).split(",") if s.strip() != ""]
    if not torch.cuda.is_available() or len(ids) == 0:
        return torch.device("cpu"), [torch.device("cpu")]
    devs = [torch.device(f"cuda:{int(i)}") for i in ids]
    return devs[0], devs


@torch.no_grad()
def infer_input_shape_and_channels(train_loader) -> Tuple[int, Tuple[int, int]]:
    """
    Infer nch and (H,W) from the first batch.
    Works for common vision tasks where inputs are tensors like [B,C,H,W].
    """
    data = next(iter(train_loader))
    x = data[0] if isinstance(data, (list, tuple)) else data
    nch = int(x.shape[1]) if x.dim() >= 2 else 1
    if x.dim() >= 4:
        H, W = int(x.shape[-2]), int(x.shape[-1])
    else:
        H, W = -1, -1
    return nch, (H, W)


def infer_num_classes(train_loader, dataset_name: str, task_type: str) -> int:
    """
    Robustly infer number of classes.
    Priority: dataset.num_classes / len(dataset.classes) / len(dataset.class_to_idx)
    Fallback by dataset name; last resort: 10.
    """
    if task_type == "regression":
        return 1

    ds = getattr(train_loader, "dataset", None)
    # Common attributes
    for attr in ["num_classes"]:
        if hasattr(ds, attr):
            try:
                v = int(getattr(ds, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    if hasattr(ds, "classes"):
        try:
            return int(len(ds.classes))
        except Exception:
            pass
    if hasattr(ds, "class_to_idx"):
        try:
            return int(len(ds.class_to_idx))
        except Exception:
            pass

    name = (dataset_name or "").lower()
    if name in ["mnist", "fashionmnist", "fmnist", "cifar10"]:
        return 10
    if name in ["imagenet", "imagenet1k", "ilsvrc2012"]:
        return 1000
    return 10


def get_dataloaders(args):
    """
    Prefer utils.load_resized_data(args); fallback to utils.load_data(args).
    """
    if hasattr(utils, "load_resized_data"):
        return utils.load_resized_data(args)
    if hasattr(utils, "load_data"):
        return utils.load_data(args)
    raise RuntimeError("Neither utils.load_resized_data(args) nor utils.load_data(args) is available.")


def build_model_with_fallbacks(args, nch: int, num_classes: int) -> nn.Module:
    """
    Try several signatures for utils.define_model; final fallback is a tiny ConvNet.
    """
    # Try define_model(args)
    try:
        return utils.define_model(args,num_classes)
    except TypeError:
        pass
    except Exception as e:
        print(f"[warn] define_model(args) failed: {e}")

    # Try variants
    variants = [
        lambda: utils.define_model(args=args, nch=nch, num_classes=num_classes),
        lambda: utils.define_model(getattr(args, "net_type", "convnet"),
                                   getattr(args, "norm_type", "instance"),
                                   getattr(args, "width", 1.0),
                                   nch, num_classes, getattr(args, "size", -1)),
        lambda: utils.define_model(net_type=getattr(args, "net_type", "convnet"),
                                   norm_type=getattr(args, "norm_type", "instance"),
                                   width=getattr(args, "width", 1.0),
                                   nch=nch, num_classes=num_classes, size=getattr(args, "size", -1)),
    ]
    for f in variants:
        try:
            return f()
        except TypeError:
            continue
        except Exception as e:
            print(f"[warn] define_model variant failed: {e}")

    # Last resort: minimal ConvNet to avoid crash
    print("[warn] Fallback to a tiny ConvNet; consider fixing define_model signature for your models.")
    return nn.Sequential(
        nn.Conv2d(nch, 64, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


# ------------------------------
# Structural auto-fixes
# ------------------------------
def ensure_first_conv_in_channels(model: nn.Module, nch: int, device: torch.device) -> nn.Module:
    """
    Adjust the FIRST Conv2d's in_channels to 'nch' when mismatched (common MNIST→CIFAR10 pitfall).
    Only handles groups==1 cases; otherwise no-op.
    """
    first_name, first_conv = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            first_name, first_conv = name, m
            break
    if first_conv is None or first_conv.in_channels == nch:
        return model
    if first_conv.groups != 1:
        print(f"[mp][warn] First Conv2d has groups={first_conv.groups}, skip auto-fix.")
        return model

    new_conv = nn.Conv2d(
        in_channels=nch,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=1,
        bias=first_conv.bias is not None,
    ).to(device)

    with torch.no_grad():
        w_old = first_conv.weight.data.to(device)  # [out, in_old, k, k]
        in_old = w_old.shape[1]
        if nch > in_old:
            rep = max(1, nch // in_old)
            w_new = w_old.repeat(1, rep, 1, 1)
            if w_new.shape[1] != nch:
                w_new = w_new[:, :nch, :, :]
            w_new = w_new * (in_old / float(nch))
        elif nch < in_old:
            if in_old % nch == 0:
                factor = in_old // nch
                w_new = w_old.view(w_old.shape[0], nch, factor, *w_old.shape[2:]).mean(dim=2)
            else:
                w_new = w_old[:, :nch, :, :]
        else:
            w_new = w_old
        new_conv.weight.copy_(w_new)
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias.data.to(device))

    parent = model
    path = first_name.split(".")
    for p in path[:-1]:
        parent = getattr(parent, p)
    setattr(parent, path[-1], new_conv)
    print(f"[mp] Adjusted first Conv2d in_channels: {first_conv.in_channels} -> {nch}")
    return model


class _ShapeCatcher(nn.Module):
    """
    Helper used in an earlier approach; kept for completeness but not used now.
    """
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = int(out_features)
        self.in_features_seen = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.in_features_seen = x.shape[1]
        return x.new_zeros(x.shape[0], self.out_features)


def ensure_last_linear_in_features_via_sample(model: nn.Module,
                                              train_loader,
                                              device: torch.device,
                                              num_classes: int) -> nn.Module:
    """
    Detect the REAL flattened feature dimension before the last Linear using a small batch,
    and rebuild the last Linear if its in_features mismatches (e.g., 1568 vs 2048).
    """
    last_name, last_fc = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_name, last_fc = name, m
    if last_fc is None:
        return model

    # Get one mini-batch of inputs
    data = next(iter(train_loader))
    x = data[0] if isinstance(data, (list, tuple)) else data
    x = x.to(device, non_blocking=True)

    # Temporarily bypass the last fc to get features before it
    parent = model
    path = last_name.split(".")
    for p in path[:-1]:
        parent = getattr(parent, p)
    original_fc = getattr(parent, path[-1])
    setattr(parent, path[-1], nn.Identity().to(device))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        feats = model(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        in_feat_actual = int(feats.shape[1])
    model.train(was_training)

    # If dimension already matches and out_features equals num_classes, restore and return
    if in_feat_actual == original_fc.in_features and original_fc.out_features == num_classes:
        setattr(parent, path[-1], original_fc)
        return model

    # Build a new fc layer and copy overlapping weights/bias
    new_fc = nn.Linear(in_feat_actual, num_classes, bias=(original_fc.bias is not None)).to(device)
    with torch.no_grad():
        min_in = min(in_feat_actual, original_fc.in_features)
        min_out = min(num_classes, original_fc.out_features)
        new_fc.weight[:min_out, :min_in].copy_(original_fc.weight[:min_out, :min_in].to(device))
        if original_fc.bias is not None:
            new_fc.bias[:min_out].copy_(original_fc.bias[:min_out].to(device))
    setattr(parent, path[-1], new_fc)

    print(f"[mp] Adjusted last Linear in_features: {original_fc.in_features} -> {in_feat_actual}, "
          f"out_features: {original_fc.out_features} -> {num_classes}")
    return model


def ensure_last_linear_out(model: nn.Module, num_classes: int, device: torch.device) -> nn.Module:
    """
    Ensure the final Linear has out_features == num_classes.
    """
    last_name, last_fc = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_name, last_fc = name, m
    if last_fc is None or last_fc.out_features == num_classes:
        return model

    new_fc = nn.Linear(last_fc.in_features, num_classes, bias=(last_fc.bias is not None)).to(device)
    with torch.no_grad():
        of = last_fc.out_features
        if num_classes <= of:
            new_fc.weight[:num_classes].copy_(last_fc.weight[:num_classes].to(device))
            if last_fc.bias is not None:
                new_fc.bias[:num_classes].copy_(last_fc.bias[:num_classes].to(device))
        else:
            new_fc.weight[:of].copy_(last_fc.weight.data.to(device))
            if last_fc.bias is not None:
                new_fc.bias[:of].copy_(last_fc.bias.data.to(device))

    parent = model
    path = last_name.split(".")
    for p in path[:-1]:
        parent = getattr(parent, p)
    setattr(parent, path[-1], new_fc)
    print(f"[mp] Adjusted last Linear out_features: {last_fc.out_features} -> {num_classes}")
    return model



def evaluate_classification(model, dataloader, device='cuda', is_distributed=False):
    """
    Evaluate classification model
    
    Returns:
        Dictionary with accuracy and other metrics
    """
    model.eval()
    size = torch.tensor(0.0).to(device)
    correct = torch.tensor(0.0).to(device)
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            size += images.size(0)
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    
    # if is_distributed:
    #     import torch.distributed as dist
    #     dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    #     dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    
    acc = (correct / size).item() if size > 0 else 0.0
    return {'accuracy': acc, 'correct': correct.item(), 'total': size.item()}

def evaluate_detection(model, dataloader, device='cuda', is_distributed=False):
    """
    Evaluate detection model (placeholder - should implement mAP, IoU, etc.)
    
    Returns:
        Dictionary with detection metrics
    """
    # Placeholder implementation
    # In practice, this should compute mAP, IoU, etc.
    model.eval()
    total_samples = torch.tensor(0.0).to(device)
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            total_samples += images.size(0)
    
    # if is_distributed:
    #     import torch.distributed as dist
    #     dist.reduce(total_samples, 0, op=dist.ReduceOp.SUM)
    
    # Placeholder metrics
    return {'mAP': 0.0, 'samples': total_samples.item()}

def evaluate_segmentation(model, dataloader, device='cuda', is_distributed=False):
    """
    Evaluate segmentation model
    
    Note: Since current models are designed for classification (output shape: batch_size, num_classes),
    we treat the entire image as a single "pixel" for segmentation evaluation.
    For true segmentation, models should output (batch_size, num_classes, H, W) and labels should be (batch_size, H, W).
    
    Returns:
        Dictionary with segmentation metrics (mIoU, pixel_acc)
    """
    model.eval()
    
    num_classes = None
    total_correct = torch.tensor(0.0).to(device)
    total_pixels = torch.tensor(0.0).to(device)
    
    # For mIoU calculation: intersection and union per class
    class_intersection = {}  # class_id -> intersection count
    class_union = {}  # class_id -> union count
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            
            # Get predictions
            if outputs.dim() > 1:
                preds = outputs.argmax(dim=1)  # (batch_size,)
            else:
                preds = outputs.argmax(dim=0).unsqueeze(0)
            
            # Pixel accuracy: for classification models, this is just classification accuracy
            # Treat each image as one pixel
            batch_size = images.size(0)
            correct = (preds == labels).sum().float()
            total_correct += correct
            total_pixels += batch_size
            
            # Compute IoU per class
            # For each class, compute intersection and union
            unique_classes = torch.unique(torch.cat([preds, labels]))
            
            for cls in unique_classes:
                cls = cls.item()
                if cls not in class_intersection:
                    class_intersection[cls] = torch.tensor(0.0).to(device)
                    class_union[cls] = torch.tensor(0.0).to(device)
                
                # Intersection: pixels where both pred and label are this class
                intersection = ((preds == cls) & (labels == cls)).sum().float()
                # Union: pixels where either pred or label is this class
                union = ((preds == cls) | (labels == cls)).sum().float()
                
                class_intersection[cls] += intersection
                class_union[cls] += union
    
    # Aggregate across all processes if distributed
    # if is_distributed:
    #     import torch.distributed as dist
        
    #     # Reduce basic metrics
    #     dist.reduce(total_correct, 0, op=dist.ReduceOp.SUM)
    #     dist.reduce(total_pixels, 0, op=dist.ReduceOp.SUM)
        
    #     # For class-wise metrics, we need to handle different class sets across processes
    #     # Simple approach: reduce only classes that exist in this process
    #     # Note: This assumes all processes see all classes eventually, or we use a fixed num_classes
    #     # For now, we'll reduce what we have and let rank 0 aggregate
    #     for cls in list(class_intersection.keys()):
    #         dist.reduce(class_intersection[cls], 0, op=dist.ReduceOp.SUM)
    #         dist.reduce(class_union[cls], 0, op=dist.ReduceOp.SUM)
    
    # Calculate pixel accuracy
    pixel_acc = (total_correct / total_pixels).item() if total_pixels > 0 else 0.0
    
    # Calculate mIoU (mean Intersection over Union)
    ious = []
    for cls in sorted(class_intersection.keys()):  # Sort for consistency
        intersection = class_intersection[cls].item()
        union = class_union[cls].item()
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    mIoU = sum(ious) / len(ious) if len(ious) > 0 else 0.0
    
    return {
        'mIoU': mIoU,
        'pixel_acc': pixel_acc,
        'samples': total_pixels.item()
    }

def evaluate_regression(model, dataloader, device='cuda', is_distributed=False):
    """
    Evaluate regression model
    
    Returns:
        Dictionary with MSE, MAE, and other regression metrics
    """
    model.eval()
    total_mse = torch.tensor(0.0).to(device)
    total_mae = torch.tensor(0.0).to(device)
    size = torch.tensor(0.0).to(device)
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            # Adapt outputs and labels for regression task
            adapted_outputs, adapted_labels = task.adapt_outputs_for_task(outputs, labels, 'regression')
            
            mse = F.mse_loss(adapted_outputs, adapted_labels, reduction='sum')
            mae = F.l1_loss(adapted_outputs, adapted_labels, reduction='sum')
            total_mse += mse
            total_mae += mae
            size += adapted_labels.size(0)
    
    # if is_distributed:
    #     import torch.distributed as dist
    #     dist.reduce(total_mse, 0, op=dist.ReduceOp.SUM)
    #     dist.reduce(total_mae, 0, op=dist.ReduceOp.SUM)
    #     dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    
    mse = (total_mse / size).item() if size > 0 else 0.0
    mae = (total_mae / size).item() if size > 0 else 0.0
    
    return {'mse': mse, 'mae': mae, 'rmse': mse ** 0.5, 'samples': size.item()}

def get_task_eval_fn(task_type):
    """
    Get evaluation function based on task type
    
    Args:
        task_type: 'classification', 'detection', 'segmentation', or 'regression'
    
    Returns:
        Evaluation function that takes (model, dataloader, device, is_distributed) and returns metrics
    """
    if task_type == 'classification':
        return evaluate_classification
    elif task_type == 'detection':
        return evaluate_detection
    elif task_type == 'segmentation':
        return evaluate_segmentation
    elif task_type == 'regression':
        return evaluate_regression
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def test(model, test_dloader, task_type):
    """Test model using task-specific evaluation function"""
    eval_fn = get_task_eval_fn(task_type)
    metrics = eval_fn(model, test_dloader, device='cuda', is_distributed=True)
    task.print_task_metrics(metrics, task_type, prefix="Test ")


# ------------------------------
# Main
# ------------------------------
def main():
    # ---- Parse args (robust to different repo variants) ----
    try:
        get_args = getattr(argument, "get_args", None) or getattr(argument, "parse_args", None)
        args = get_args()
    except Exception:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--gpu", type=str, default="0,1")
        p.add_argument("--dataset", type=str, default="mnist")
        p.add_argument("--data_dir", type=str, default="./data")
        p.add_argument("--net_type", type=str, default="convnet")
        p.add_argument("--norm_type", type=str, default="instance")
        p.add_argument("--task", type=str, default="classification")
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("-b", "--batch_size", type=int, default=64)
        p.add_argument("--width", type=float, default=1.0)
        p.add_argument("--size", type=int, default=-1)
        p.add_argument("--nch", type=int, default=-1)
        p.add_argument("--lr", type=float, default=0.01)
        args = p.parse_args()

    primary_device, devices = parse_devices(getattr(args, "gpu", "0,1"))
    print(f"[mp] devices = {devices}, primary = {primary_device}")

    # ---- Data ----
    loaders = get_dataloaders(args)
    # Accept (train, test, ...) or a single loader
    train_loader, test_loader = None, None
    if isinstance(loaders, (tuple, list)):
        # cand = [x for x in loaders if hasattr(x, "dataset") and hasattr(x, "__iter__")]
        # if len(cand) >= 1:
        #     train_loader = cand[0]
        # if len(cand) >= 2:
        #     test_loader = cand[1]
        train_dataset, test_loader = loaders
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    else:
        if hasattr(loaders, "dataset"):
            train_loader = loaders
    if train_loader is None:
        raise RuntimeError("No valid train DataLoader returned by utils.load_resized_data/load_data.")

    # ---- Infer nch / size / num_classes ----
    inferred_nch, (H, W) = infer_input_shape_and_channels(train_loader)
    nch = inferred_nch if getattr(args, "nch", -1) in (-1, None) else int(args.nch)
    num_classes = infer_num_classes(train_loader, getattr(args, "dataset", ""), getattr(args, "task", "classification"))
    print(f"[mp] inferred: nch={nch}, size=({H},{W}), num_classes={num_classes}")

    # ---- Write back to args so utils.define_model can use them ----
    for k in ["nch", "in_channels", "input_channels", "channels"]:
        setattr(args, k, int(nch))
    for k in ["num_classes", "n_classes", "nclass", "classes", "out_dim"]:
        setattr(args, k, int(num_classes))
    if H > 0 and W > 0:
        # Provide both tuple/int style keys so different code paths can pick what they want
        if not hasattr(args, "img_hw"):
            setattr(args, "img_hw", (int(H), int(W)))
        for key in ["size", "img_size", "image_size", "input_size"]:
            v = getattr(args, key, None)
            if v in (None, -1, 0):
                setattr(args, key, max(int(H), int(W)))

    # ---- Build model ----
    model = build_model_with_fallbacks(args, nch=nch, num_classes=num_classes)
    model.to(primary_device)

    # ---- Structural fixes BEFORE wrapping parallel ----
    # 1) First conv in_channels (MNIST→CIFAR10)
    model = ensure_first_conv_in_channels(model, nch=nch, device=primary_device)

    # 2) Last Linear in_features via sample (28x28→32x32, e.g., 1568↔2048)
    if getattr(args, "task", "classification") == "classification":
        model = ensure_last_linear_in_features_via_sample(model, train_loader, primary_device, num_classes)
        # 3) Ensure classifier head size
        model = ensure_last_linear_out(model, num_classes=num_classes, device=primary_device)

    # ---- Tensor model-parallel wrapping ----
    # Lower threshold to include small FC layers for tiny convnets; adjust as you wish.
    model = wrap_model_tensor_parallel(
        model, devices=devices, output_device=primary_device,
        min_params=20_000,  # was 100_000; 20k lets the final Linear be split (e.g., 2048x10)
        wrap_conv=True, wrap_linear=True
    )

    # ---- Task: criterion & output adapter ----
    try:
        criterion = task.get_task_criterion(args)
    except Exception:
        criterion = nn.MSELoss() if getattr(args, "task", "classification") == "regression" else nn.CrossEntropyLoss()

    adapt_outputs_for_task = getattr(task, "adapt_outputs_for_task", None)
    if adapt_outputs_for_task is None:
        def adapt_outputs_for_task(y, t, args):
            # For regression, flatten predictions; for classification, return as-is
            if getattr(args, "task", "classification") == "regression" and y.dim() > 2:
                y = torch.flatten(y, 1)
            return y, t

    # ---- Optimizer / AMP ----
    lr = float(getattr(args, "lr", 0.01))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=(primary_device.type == "cuda"))

    # ---- Train loop ----
    epochs = int(getattr(args, "epochs", 3))
    bs = int(getattr(args, "batch_size", 64))
    print(f"[mp] start training: epochs={epochs}, batch_size={bs}")

    training_start_time = time.time()
    total_samples = 0
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        total, correct = 0, 0

        for i, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None
            total_samples += len(x)

            x = x.to(primary_device, non_blocking=True)
            if y is not None:
                y = y.to(primary_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(primary_device.type == "cuda")):
                out = model(x)
                out, y_adapt = adapt_outputs_for_task(out, y, args)

                task_type = getattr(args, "task", "classification")
                if task_type == "classification":
                    loss = criterion(out, y_adapt)
                elif task_type == "regression":
                    loss = criterion(out.squeeze(), y_adapt.float().squeeze())
                else:
                    loss = criterion(out, y_adapt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())

            # Simple accuracy for classification
            if getattr(args, "task", "classification") == "classification" and y is not None:
                with torch.no_grad():
                    # support one-hot or label indices
                    yt = y
                    if yt.dim() > 1 and yt.size(-1) > 1:
                        yt = yt.argmax(dim=-1)
                    pred = out.argmax(dim=1)
                    total += yt.numel()
                    correct += (pred == yt).sum().item()

        dt = time.time() - t0
        msg = f"[mp][epoch {epoch+1}/{epochs}] loss={running_loss/(i+1):.4f}, time={dt:.2f}s"
        if getattr(args, "task", "classification") == "classification" and total > 0:
            acc = 100.0 * correct / total
            msg += f", acc@1={acc:.2f}%"
        print(msg)

    print("[mp] training done.")
    training_time = time.time() - training_start_time
    print(f"begin testing")
    
    test(model, test_loader, args.task)
    torch.save({"model": model.state_dict()}, f"checkpoint/model_parallel_{args.dataset}_{args.net_type}_checkpoint.pt")
    
    # Calculate and print metrics
    input_shape = (1, args.nch, args.size, args.size)
    metrics = utils.get_training_metrics(
        model=model.module if hasattr(model, 'module') else model,
        samples_processed=total_samples,
        time_elapsed=training_time,
        input_shape=input_shape,
        num_gpus=1,
        device='cuda'
    )
    utils.print_metrics(metrics, prefix="Training ")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_elapsed = time.time() - time_start
    print(f"\ntime elapsed: {time_elapsed:.2f} seconds")
