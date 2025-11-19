import argparse
import os


def get_args(gpu_default="0"):
    """
    Args:
        gpu_default: Default GPU string (e.g., "0" for single GPU, "0,1" for multi-GPU)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=gpu_default, help="GPU IDs to use (comma-separated)")
    parser.add_argument(
        "-e",
        "--epochs",
        default=3,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        metavar="N",
        help="number of batchsize",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "imagenet"],
        help="dataset to use",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="convnet",
        choices=["resnet18", "resnet32", "convnet", "resnet-ap", "resnet_ap"],
        help="model type to use",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="root directory for datasets",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=-1,
        help="image size (default: auto based on dataset)",
    )
    parser.add_argument(
        "--nch",
        type=int,
        default=-1,
        help="number of channels (default: auto based on dataset)",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="instance",
        choices=["batch", "instance"],
        help="normalization type",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=1.0,
        help="width multiplier for ResNet-AP",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "detection", "segmentation", "regression"],
        help="task type: classification, detection, segmentation, or regression",
    )
    args = parser.parse_args()
    
    # Set default size and nch based on dataset
    if args.size == -1:
        if args.dataset == 'mnist':
            args.size = 28
        elif args.dataset == 'cifar10':
            args.size = 32
        elif args.dataset == 'imagenet':
            args.size = 224
    
    if args.nch == -1:
        if args.dataset == 'mnist':
            args.nch = 1
        else:
            args.nch = 3
    
    return args

