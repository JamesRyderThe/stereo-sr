from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, learnable: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.to(x.dtype)).to(orig_dtype)


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, learnable: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        weight = self.weight.to(x.dtype).view(1, -1, 1, 1)
        return (x * weight).to(orig_dtype)


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None and self.bias is not None:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x
