from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sissr.models.layers.init import DEFAULT_POLICY, InitPolicy, init_mlp_in, init_mlp_out


class SwiGLUActivation(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size(self.dim)
        if size % 2:
            raise ValueError(f"SwiGLU requires even split along dim {self.dim}, got {size}")
        gate, value = torch.chunk(x, 2, dim=self.dim)
        return F.silu(gate) * value


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        policy: InitPolicy = DEFAULT_POLICY,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.activation = SwiGLUActivation(dim=-1)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        init_mlp_in(self.up_proj, policy)
        init_mlp_out(self.down_proj, policy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(self.activation(self.up_proj(x)))
        return out
