from __future__ import annotations

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        if x.ndim == 3:
            return x * gamma.view(1, 1, -1)
        if x.ndim == 4:
            return x * gamma.view(1, -1, 1, 1)
        return x * gamma
