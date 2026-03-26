from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sissr.models.layers.embedding import RotarySpec
from sissr.models.layers.init import (
    DEFAULT_POLICY,
    InitPolicy,
    init_attn_in,
    init_attn_out,
    zero_weights,
)
from sissr.models.layers.norm import RMSNorm


@dataclass
class QKVProjection:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _apply_rope(
    tensor: torch.Tensor,
    *,
    spec: RotarySpec,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    if spec.rope is None:
        return tensor
    if height is not None:
        target_height = height
    elif spec.grid is not None:
        target_height = spec.grid.height
    else:
        target_height = None
    if width is not None:
        target_width = width
    elif spec.grid is not None:
        target_width = spec.grid.width
    else:
        target_width = None
    if target_height is None or target_width is None:
        raise ValueError("RotarySpec requires explicit spatial dimensions")
    out: torch.Tensor = spec.rope(tensor, height=target_height, width=target_width)
    return out


class AttentionCore:
    def _setup_attention(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = True,
        gate: bool = True,
        residual_v: bool = False,
        xsa: bool = False,
        policy: InitPolicy = DEFAULT_POLICY,
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.dropout = dropout
        self.use_gate = gate
        self.use_residual_v = residual_v
        self.use_xsa = xsa

        kv_dim = num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(embed_dim, embed_dim + 2 * kv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        init_attn_in(self.qkv_proj, policy)
        init_attn_out(self.out_proj, policy)

        if gate:
            self.gate_proj = nn.Linear(embed_dim, self.head_dim, bias=True)
            zero_weights(self.gate_proj)

        if residual_v:
            self.v_residual_lambda = nn.Parameter(torch.tensor(0.5))

        self.q_norm: nn.Module = (
            RMSNorm(self.head_dim, eps=qk_norm_eps, learnable=False) if qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            RMSNorm(self.head_dim, eps=qk_norm_eps, learnable=False) if qk_norm else nn.Identity()
        )

    def _project_qkv(self, x: torch.Tensor) -> QKVProjection:
        qkv = self.qkv_proj(x)
        kv_dim = self.num_kv_heads * self.head_dim
        q_raw, k_raw, v_raw = qkv.split([self.embed_dim, kv_dim, kv_dim], dim=-1)
        q = rearrange(q_raw, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k_raw, "b s (h d) -> b s h d", h=self.num_kv_heads)
        v = rearrange(v_raw, "b s (h d) -> b s h d", h=self.num_kv_heads)
        return QKVProjection(q=q, k=k, v=v)

    def _normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)

    def _blend_residual_v(self, v: torch.Tensor, v0: torch.Tensor | None) -> torch.Tensor:
        if not self.use_residual_v or v0 is None:
            return v
        if v0.shape != v.shape:
            raise ValueError(f"v0 shape {v0.shape} must match v shape {v.shape}")
        lam = self.v_residual_lambda
        return lam * v + (1.0 - lam) * v0

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.kv_repeat == 1:
            return tensor
        return tensor.repeat_interleave(self.kv_repeat, dim=2)

    def _apply_gate(self, attn_out: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if not self.use_gate:
            return attn_out
        gate = torch.sigmoid(self.gate_proj(hidden))
        return attn_out * rearrange(gate, "b s d -> b 1 s d")

    def _apply_xsa(self, attn_out: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not self.use_xsa:
            return attn_out
        vn = F.normalize(v, dim=-1)
        return attn_out - (attn_out * vn).sum(dim=-1, keepdim=True) * vn

    def _output_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.out_proj(rearrange(attn_out, "b h s d -> b s (h d)"))
        return out


class Attention(AttentionCore, nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = True,
        gate: bool = True,
        residual_v: bool = False,
        xsa: bool = False,
        policy: InitPolicy = DEFAULT_POLICY,
    ) -> None:
        super().__init__()
        self._setup_attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            dropout=dropout,
            bias=bias,
            gate=gate,
            residual_v=residual_v,
            xsa=xsa,
            policy=policy,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rotary: RotarySpec | None = None,
        v0: torch.Tensor | None = None,
        return_value: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        proj = self._project_qkv(hidden_states)
        q, k = self._normalize_qk(proj.q, proj.k)
        v = proj.v

        if rotary is not None:
            q = _apply_rope(q, spec=rotary)
            k = _apply_rope(k, spec=rotary)

        v = self._blend_residual_v(v, v0)

        q_t = rearrange(q, "b s h d -> b h s d")
        k_t = rearrange(self._expand_kv(k), "b s h d -> b h s d")
        v_t = rearrange(self._expand_kv(v), "b s h d -> b h s d")

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_out = self._apply_xsa(attn_out, v_t)
        attn_out = self._apply_gate(attn_out, hidden_states)
        result = self._output_proj(attn_out)

        if self.use_residual_v or return_value:
            return result, v
        return result
