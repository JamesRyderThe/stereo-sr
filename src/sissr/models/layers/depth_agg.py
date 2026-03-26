from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from sissr.models.layers.norm import RMSNorm


class DepthAggregator(nn.Module):
    def __init__(
        self, dim: int, num_sublayers: int, *, max_sources: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if max_sources < 1:
            raise ValueError(f"max_sources must be positive, got {max_sources}")
        self.max_sources = max_sources
        self.queries = nn.Parameter(torch.zeros(num_sublayers, dim))
        self.source_norm = RMSNorm(dim, eps=eps, learnable=False)
        self.register_buffer("_source_indices", torch.arange(max_sources), persistent=False)
        self._source_indices: torch.Tensor

    def forward(
        self,
        sources: torch.Tensor,
        num_sources: int,
        sublayer_idx: int,
    ) -> torch.Tensor:
        if num_sources < 1 or num_sources > self.max_sources:
            raise ValueError(f"num_sources must be in [1, {self.max_sources}], got {num_sources}")
        if sources.shape[0] != self.max_sources:
            raise ValueError(
                "sources.shape[0] "
                f"({sources.shape[0]}) must equal "
                f"max_sources ({self.max_sources})"
            )
        q = self.queries[sublayer_idx]
        normed = self.source_norm(sources)
        logits = torch.einsum("nbsd, d -> nbs", normed, q)
        mask = self._source_indices >= num_sources
        logits = logits.masked_fill(mask[:, None, None], float("-inf"))
        weights = logits.softmax(dim=0)
        return torch.einsum("nbsd, nbs -> bsd", sources, weights)


def _pad_sources(sources: torch.Tensor, max_sources: int) -> torch.Tensor:
    pad_n = max_sources - sources.shape[0]
    if pad_n < 0:
        raise ValueError(
            f"sources.shape[0] ({sources.shape[0]}) exceeds max_sources ({max_sources})"
        )
    if pad_n > 0:
        sources = F.pad(sources, (0, 0, 0, 0, 0, 0, 0, pad_n))
    return sources


def _run_aggregate(
    agg: DepthAggregator,
    blocks: tuple[torch.Tensor, ...],
    partial: torch.Tensor | None,
    sublayer_idx: int,
) -> torch.Tensor:
    if partial is not None:
        sources = torch.stack([*blocks, partial])
        n = len(blocks) + 1
    else:
        sources = torch.stack(list(blocks))
        n = len(blocks)
    sources = _pad_sources(sources, agg.max_sources)
    result: torch.Tensor = agg(sources, n, sublayer_idx)
    return result


def _run_finalize(
    agg: DepthAggregator,
    blocks: tuple[torch.Tensor, ...],
    sublayer_idx: int,
) -> torch.Tensor:
    sources = _pad_sources(torch.stack(list(blocks)), agg.max_sources)
    result: torch.Tensor = agg(sources, len(blocks), sublayer_idx)
    return result


@dataclass(eq=False)
class DepthState:
    _agg: DepthAggregator
    _blocks: list[torch.Tensor] = field(default_factory=list)
    _sublayer_idx: int = 0
    partial: torch.Tensor | None = None

    @classmethod
    def create(cls, agg: DepthAggregator, initial: torch.Tensor) -> DepthState:
        return cls(_agg=agg, partial=initial)

    def aggregate(self) -> torch.Tensor:
        idx = self._sublayer_idx
        self._sublayer_idx += 1
        blocks = tuple(self._blocks)
        result: torch.Tensor = grad_checkpoint(
            _run_aggregate, self._agg, blocks, self.partial, idx, use_reentrant=False
        )
        return result

    def accumulate(self, delta: torch.Tensor) -> None:
        if self.partial is None:
            self.partial = delta
        else:
            self.partial = self.partial + delta

    def commit_boundary(self) -> None:
        if self.partial is None:
            raise RuntimeError("commit_boundary called with no partial state")
        self._blocks.append(self.partial)
        self.partial = None

    def finalize(self, *, skip_first: bool = False) -> torch.Tensor:
        if self.partial is None:
            raise RuntimeError("finalize called with no partial state")
        self._blocks.append(self.partial)
        self.partial = None
        start_idx = 1 if skip_first else 0
        final_blocks = tuple(self._blocks[start_idx:])
        if not final_blocks:
            raise RuntimeError("finalize called with no logical block sources")
        idx = self._sublayer_idx
        self._sublayer_idx += 1
        result: torch.Tensor = grad_checkpoint(
            _run_finalize, self._agg, final_blocks, idx, use_reentrant=False
        )
        return result
