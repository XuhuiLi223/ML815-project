# mp_tensor_parallel.py
# -*- coding: utf-8 -*-

from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _partition(n: int, k: int) -> List[int]:
    """把 n 尽量均匀地分成 k 份"""
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]


class TensorParallelConv2d(nn.Module):
    """
    把同一个 Conv2d 的 out_channels 维在多张 GPU 上切分并行计算，
    结果在 output_device 上 concat（dim=1）。
    """
    def __init__(self, conv: nn.Conv2d, devices: Sequence[torch.device], output_device=None):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)
        self.devices = [torch.device(d) for d in devices]
        self.out_device = torch.device(self.devices[0] if output_device is None else output_device)

        self.groups = conv.groups
        self.in_channels = conv.in_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.out_channels = conv.out_channels
        self.bias_flag = conv.bias is not None

        k = len(self.devices)
        splits = _partition(self.out_channels, k)

        # groups 安全性：每块 out_channels 必须可被 groups 整除；否则退化到单卡
        # 亦或 out_channels < k 时也退化
        if any((s % self.groups) != 0 for s in splits) or self.out_channels < k:
            self.parallel = False
            self.single = nn.Conv2d(
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.padding, self.dilation, self.groups, bias=self.bias_flag
            ).to(self.out_device)
            with torch.no_grad():
                self.single.weight.copy_(conv.weight.data.to(self.out_device))
                if self.bias_flag:
                    self.single.bias.copy_(conv.bias.data.to(self.out_device))
        else:
            self.parallel = True
            self.shards = nn.ModuleList()
            w = conv.weight.data
            b = conv.bias.data if self.bias_flag else None
            start = 0
            for i, s in enumerate(splits):
                sh = nn.Conv2d(
                    self.in_channels, s, self.kernel_size,
                    self.stride, self.padding, self.dilation, self.groups, bias=self.bias_flag
                ).to(self.devices[i])
                with torch.no_grad():
                    sh.weight.copy_(w[start:start+s].to(self.devices[i]))
                    if self.bias_flag:
                        sh.bias.copy_(b[start:start+s].to(self.devices[i]))
                self.shards.append(sh)
                start += s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.parallel:
            return self.single(x.to(self.out_device, non_blocking=True))
        outs = []
        for dev, sh in zip(self.devices, self.shards):
            y = sh(x.to(dev, non_blocking=True))
            outs.append(y.to(self.out_device, non_blocking=True))
        return torch.cat(outs, dim=1)


class TensorParallelLinear(nn.Module):
    """
    把 Linear 的 out_features 维在多张 GPU 上切分并行计算，
    结果在 output_device 上 concat（dim=1）。
    """
    def __init__(self, linear: nn.Linear, devices: Sequence[torch.device], output_device=None):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.devices = [torch.device(d) for d in devices]
        self.out_device = torch.device(self.devices[0] if output_device is None else output_device)

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias_flag = linear.bias is not None

        k = len(self.devices)
        splits = _partition(self.out_features, k)

        # 如果 out_features < k，退化到单卡
        if self.out_features < k:
            self.parallel = False
            self.single = nn.Linear(self.in_features, self.out_features, bias=self.bias_flag).to(self.out_device)
            with torch.no_grad():
                self.single.weight.copy_(linear.weight.data.to(self.out_device))
                if self.bias_flag:
                    self.single.bias.copy_(linear.bias.data.to(self.out_device))
        else:
            self.parallel = True
            self.shards = nn.ModuleList()
            w = linear.weight.data
            b = linear.bias.data if self.bias_flag else None
            start = 0
            for i, s in enumerate(splits):
                sh = nn.Linear(self.in_features, s, bias=self.bias_flag).to(self.devices[i])
                with torch.no_grad():
                    sh.weight.copy_(w[start:start+s].to(self.devices[i]))
                    if self.bias_flag:
                        sh.bias.copy_(b[start:start+s].to(self.devices[i]))
                self.shards.append(sh)
                start += s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.parallel:
            return self.single(x.to(self.out_device, non_blocking=True))
        outs = []
        for dev, sh in zip(self.devices, self.shards):
            y = sh(x.to(dev, non_blocking=True))
            outs.append(y.to(self.out_device, non_blocking=True))
        return torch.cat(outs, dim=1)


def wrap_model_tensor_parallel(model: nn.Module,
                               devices: Sequence[torch.device],
                               output_device=None,
                               min_params: int = 100_000,
                               wrap_conv: bool = True,
                               wrap_linear: bool = True) -> nn.Module:
    """
    递归替换大算子（Conv2d/Linear）为张量并行版本；小层保持单卡，避免通信开销过大。
    注意：请在 wrap 之前把整模先 move 到 output_device，再调用本函数。
    """
    if output_device is None:
        output_device = devices[0]
    devices = [torch.device(d) for d in devices]
    output_device = torch.device(output_device)

    def _maybe_wrap(module: nn.Module) -> nn.Module:
        # 重要：避免对已包裹的模块再次包裹
        if isinstance(module, (TensorParallelConv2d, TensorParallelLinear)):
            return module

        # 先递归处理子模块，再决定是否替换当前模块
        for name, child in list(module.named_children()):
            wrapped = _maybe_wrap(child)
            if wrapped is not child:
                setattr(module, name, wrapped)

        # 按需替换当前模块
        if wrap_conv and isinstance(module, nn.Conv2d) and module.weight.numel() >= min_params and len(devices) > 1:
            return TensorParallelConv2d(module, devices, output_device=output_device)

        if wrap_linear and isinstance(module, nn.Linear) and module.weight.numel() >= min_params and len(devices) > 1:
            return TensorParallelLinear(module, devices, output_device=output_device)

        return module

    return _maybe_wrap(model)
