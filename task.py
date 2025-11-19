import torch
import torch.nn as nn
import torch.nn.functional as F


def get_task_criterion(task_type, num_classes=None, device='cuda'):
    """
    Get loss function (criterion) based on task type
    
    Args:
        task_type: 'classification', 'detection', 'segmentation', or 'regression'
        num_classes: Number of classes (for classification/segmentation)
        device: Device to place criterion on
    
    Returns:
        Loss function (criterion)
    """
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'detection':
        criterion = nn.CrossEntropyLoss()  # Placeholder, should be combined with bbox loss
    elif task_type == 'segmentation':
        criterion = nn.CrossEntropyLoss()  # For semantic segmentation
    elif task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return criterion.to(device)


def adapt_outputs_for_task(outputs, labels, task_type):
    """
    Adapt model outputs and labels for different task types
    
    Args:
        outputs: Model outputs (typically (batch_size, num_classes) for classification models)
        labels: Ground truth labels (typically (batch_size,) with class indices)
        task_type: Task type string
    
    Returns:
        Tuple of (adapted_outputs, adapted_labels)
    """
    if task_type == 'regression':
        if outputs.dim() > 1:
            adapted_outputs = outputs[:, 0] 
        else:
            adapted_outputs = outputs
        
        adapted_labels = labels.float().to(dtype=adapted_outputs.dtype)
        
        return adapted_outputs, adapted_labels
    else:
        return outputs, labels


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
    
    if is_distributed:
        import torch.distributed as dist
        dist.reduce(size, 0, op=dist.ReduceOp.SUM)
        dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    
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
    
    if is_distributed:
        import torch.distributed as dist
        dist.reduce(total_samples, 0, op=dist.ReduceOp.SUM)
    
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
    if is_distributed:
        import torch.distributed as dist
        
        # Reduce basic metrics
        dist.reduce(total_correct, 0, op=dist.ReduceOp.SUM)
        dist.reduce(total_pixels, 0, op=dist.ReduceOp.SUM)
        
        # For class-wise metrics, we need to handle different class sets across processes
        # Simple approach: reduce only classes that exist in this process
        # Note: This assumes all processes see all classes eventually, or we use a fixed num_classes
        # For now, we'll reduce what we have and let rank 0 aggregate
        for cls in list(class_intersection.keys()):
            dist.reduce(class_intersection[cls], 0, op=dist.ReduceOp.SUM)
            dist.reduce(class_union[cls], 0, op=dist.ReduceOp.SUM)
    
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
            adapted_outputs, adapted_labels = adapt_outputs_for_task(outputs, labels, 'regression')
            
            mse = F.mse_loss(adapted_outputs, adapted_labels, reduction='sum')
            mae = F.l1_loss(adapted_outputs, adapted_labels, reduction='sum')
            total_mse += mse
            total_mae += mae
            size += adapted_labels.size(0)
    
    if is_distributed:
        import torch.distributed as dist
        dist.reduce(total_mse, 0, op=dist.ReduceOp.SUM)
        dist.reduce(total_mae, 0, op=dist.ReduceOp.SUM)
        dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    
    mse = (total_mse / size).item() if size > 0 else 0.0
    mae = (total_mae / size).item() if size > 0 else 0.0
    
    return {'mse': mse, 'mae': mae, 'rmse': mse ** 0.5, 'samples': size.item()}


def print_task_metrics(metrics, task_type, prefix=""):
    """
    Print task-specific metrics
    
    Args:
        metrics: Dictionary of metrics from evaluation function
        task_type: Task type string
        prefix: Prefix string for printing
    """
    if task_type == 'classification':
        print(f"{prefix}Accuracy: {metrics['accuracy']:.2%}")
    elif task_type == 'detection':
        print(f"{prefix}mAP: {metrics['mAP']:.4f}")
    elif task_type == 'segmentation':
        print(f"{prefix}mIoU: {metrics['mIoU']:.4f}, Pixel Accuracy: {metrics['pixel_acc']:.2%}")
    elif task_type == 'regression':
        print(f"{prefix}MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    else:
        print(f"{prefix}Metrics: {metrics}")

