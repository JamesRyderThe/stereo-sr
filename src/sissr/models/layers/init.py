from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

TRUNC_STD: float = 0.02
KAIMING_A: float = math.sqrt(5.0)


def _weight(module: nn.Module) -> nn.Parameter:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter):
        raise AttributeError(f"{module.__class__.__name__} has no trainable weight")
    return weight


def _zero_bias(module: nn.Module) -> None:
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        nn.init.zeros_(bias)


def _trunc_normal(module: nn.Module, std: float = TRUNC_STD) -> None:
    nn.init.trunc_normal_(_weight(module), mean=0.0, std=std)
    _zero_bias(module)


def _fan_in(weight: torch.Tensor) -> int:
    if weight.ndim < 2:
        raise ValueError("fan_in requires at least 2D weight")
    receptive = 1
    for s in weight.shape[2:]:
        receptive *= s
    return weight.shape[1] * receptive


def _kaiming_conv(module: nn.Module) -> None:
    weight = _weight(module)
    nn.init.kaiming_uniform_(weight, a=KAIMING_A, mode="fan_in", nonlinearity="leaky_relu")
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        bound = 1.0 / math.sqrt(float(_fan_in(weight)))
        nn.init.uniform_(bias, -bound, bound)


def _orthogonal_1x1(module: nn.Module) -> None:
    weight = _weight(module)
    if weight.shape[2:] != (1, 1):
        raise ValueError("orthogonal init requires 1x1 conv")
    out_c, in_c = weight.shape[:2]
    with torch.no_grad():
        nn.init.orthogonal_(weight.view(out_c, in_c))
    _zero_bias(module)


def zero_weights(module: nn.Module) -> None:
    nn.init.zeros_(_weight(module))
    _zero_bias(module)


@dataclass(frozen=True)
class InitPolicy:
    attn_in: Callable[[nn.Module], None] = _trunc_normal
    attn_out: Callable[[nn.Module], None] = _trunc_normal
    mlp_in: Callable[[nn.Module], None] = _trunc_normal
    mlp_out: Callable[[nn.Module], None] = _trunc_normal
    linear: Callable[[nn.Module], None] = _trunc_normal
    conv: Callable[[nn.Module], None] = _kaiming_conv
    skip_conv: Callable[[nn.Module], None] = _orthogonal_1x1


DEFAULT_POLICY = InitPolicy()


def init_attn_in(module: nn.Module, policy: InitPolicy = DEFAULT_POLICY) -> None:
    policy.attn_in(module)


def init_attn_out(module: nn.Module, policy: InitPolicy = DEFAULT_POLICY) -> None:
    policy.attn_out(module)


def init_mlp_in(module: nn.Module, policy: InitPolicy = DEFAULT_POLICY) -> None:
    policy.mlp_in(module)


def init_mlp_out(module: nn.Module, policy: InitPolicy = DEFAULT_POLICY) -> None:
    policy.mlp_out(module)
