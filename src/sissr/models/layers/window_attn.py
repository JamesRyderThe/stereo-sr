from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sissr.models.enums import StereoDirection
from sissr.models.layers.attn import AttentionCore, _apply_rope
from sissr.models.layers.embedding import (
    EpipolarRoPE,
    RectifiedDisparityRoPE,
    RotarySpec,
    StereoGeometry,
)
from sissr.models.layers.init import DEFAULT_POLICY, InitPolicy, init_attn_in, init_attn_out
from sissr.models.layers.norm import RMSNorm

WindowSize = tuple[int, int]
ShiftSize = tuple[int, int]


def _format_window_size(window_size: WindowSize) -> str:
    return f"({window_size[0]}, {window_size[1]})"


def _square_window(window_size: int) -> WindowSize:
    return (window_size, window_size)


def window_partition(x: torch.Tensor, window_size: WindowSize) -> torch.Tensor:
    _, _, height, width = x.shape
    window_h, window_w = window_size
    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            "spatial dims "
            f"({height}, {width}) must divide by window_size {_format_window_size(window_size)}"
        )
    return rearrange(
        x,
        "b c (nh wh) (nw ww) -> (b nh nw) (wh ww) c",
        wh=window_h,
        ww=window_w,
    )


def window_merge(
    x: torch.Tensor, window_size: WindowSize, height: int, width: int, batch: int
) -> torch.Tensor:
    window_h, window_w = window_size
    nh = height // window_h
    nw = width // window_w
    return rearrange(
        x,
        "(b nh nw) (wh ww) c -> b c (nh wh) (nw ww)",
        b=batch,
        nh=nh,
        nw=nw,
        wh=window_h,
        ww=window_w,
    )


def cyclic_shift(x: torch.Tensor, amount: ShiftSize) -> torch.Tensor:
    return torch.roll(x, shifts=(-amount[0], -amount[1]), dims=(2, 3))


def cyclic_unshift(x: torch.Tensor, amount: ShiftSize) -> torch.Tensor:
    return torch.roll(x, shifts=(amount[0], amount[1]), dims=(2, 3))


def _axis_slices(window_size: int, shift: int) -> tuple[slice, ...]:
    if shift == 0:
        return (slice(0, None),)
    return (
        slice(0, -window_size),
        slice(-window_size, -shift),
        slice(-shift, None),
    )


def build_shift_mask(
    height: int,
    width: int,
    window_size: WindowSize,
    shift: ShiftSize,
    device: torch.device,
) -> torch.Tensor:
    window_h, window_w = window_size
    shift_h, shift_w = shift
    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            "spatial "
            f"({height}, {width}) not divisible by window_size {_format_window_size(window_size)}"
        )
    region_ids = torch.zeros((1, height, width, 1), device=device, dtype=torch.int32)
    h_slices = _axis_slices(window_h, shift_h)
    w_slices = _axis_slices(window_w, shift_w)
    for count, (hs, ws) in enumerate(product(h_slices, w_slices)):
        region_ids[:, hs, ws, :] = count
    region_ids = rearrange(
        region_ids,
        "b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c",
        wh=window_h,
        ww=window_w,
    ).squeeze(-1)
    return region_ids.unsqueeze(-1) != region_ids.unsqueeze(-2)


def _build_cross_shift_mask(
    height: int,
    width: int,
    window_size: WindowSize,
    shift: ShiftSize,
    num_contexts: int,
    device: torch.device,
) -> torch.Tensor:
    base = build_shift_mask(height, width, window_size, shift, device=device)
    num_windows, q_seq, _ = base.shape
    expanded = base.unsqueeze(-1).expand(num_windows, q_seq, q_seq, num_contexts)
    return expanded.reshape(num_windows, q_seq, q_seq * num_contexts)


class WindowedAttention(AttentionCore, nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
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
        self.window_size = window_size
        self.window_shape = _square_window(window_size)
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

    def _make_shift_mask(
        self,
        height: int,
        width: int,
        shift_amount: ShiftSize,
        num_batch_windows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        bool_mask = build_shift_mask(height, width, self.window_shape, shift_amount, device=device)
        num_windows = bool_mask.shape[0]
        repeats = num_batch_windows // num_windows
        bool_mask = bool_mask.unsqueeze(0).expand(repeats, -1, -1, -1)
        bool_mask = rearrange(bool_mask, "b nw s1 s2 -> (b nw) s1 s2")
        additive = torch.zeros_like(bool_mask, dtype=dtype)
        additive.masked_fill_(bool_mask, torch.finfo(dtype).min)
        return additive.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        shift: bool = False,
        mask: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
        return_value: bool = False,
        rotary: RotarySpec | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        window_h, window_w = self.window_shape
        if height % window_h != 0 or width % window_w != 0:
            raise ValueError(
                "spatial dims "
                f"({height}, {width}) must divide by window_size "
                f"{_format_window_size(self.window_shape)}"
            )

        shift_amount = (window_h // 2, window_w // 2) if shift else (0, 0)
        if shift:
            x = cyclic_shift(x, shift_amount)

        windows = window_partition(x, self.window_shape)
        proj = self._project_qkv(windows)
        q, k = self._normalize_qk(proj.q, proj.k)
        v = proj.v

        if rotary is not None:
            q = _apply_rope(q, spec=rotary, height=window_h, width=window_w)
            k = _apply_rope(k, spec=rotary, height=window_h, width=window_w)

        v = self._blend_residual_v(v, v0)

        attn_mask: torch.Tensor | None = None
        if shift:
            attn_mask = self._make_shift_mask(
                height, width, shift_amount, windows.shape[0], dtype=q.dtype, device=q.device
            )
        if mask is not None:
            attn_mask = mask if attn_mask is None else attn_mask + mask

        q_t = rearrange(q, "b s h d -> b h s d")
        k_t = rearrange(self._expand_kv(k), "b s h d -> b h s d")
        v_t = rearrange(self._expand_kv(v), "b s h d -> b h s d")

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_out = self._apply_xsa(attn_out, v_t)
        attn_out = self._apply_gate(attn_out, windows)
        out = self._output_proj(attn_out)
        out = window_merge(out, self.window_shape, height, width, batch)

        if shift:
            out = cyclic_unshift(out, shift_amount)

        if self.use_residual_v or return_value:
            return out, v
        return out


class WindowedCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: WindowSize,
        *,
        kv_dim: int | None = None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = True,
        policy: InitPolicy = DEFAULT_POLICY,
        epipolar_rope: EpipolarRoPE | None = None,
        rect_disp_rope: RectifiedDisparityRoPE | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if epipolar_rope is not None and rect_disp_rope is not None:
            raise ValueError("cannot use both epipolar_rope and rect_disp_rope")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.epipolar_rope = epipolar_rope
        self.rect_disp_rope = rect_disp_rope
        kv_dim = kv_dim if kv_dim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        init_attn_in(self.q_proj, policy)
        init_attn_in(self.kv_proj, policy)
        init_attn_out(self.out_proj, policy)

        self.q_norm: nn.Module = (
            RMSNorm(self.head_dim, eps=qk_norm_eps, learnable=False) if qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            RMSNorm(self.head_dim, eps=qk_norm_eps, learnable=False) if qk_norm else nn.Identity()
        )

    def _partition_geometry(
        self,
        geometry: StereoGeometry,
        shift_amount: ShiftSize,
        shift: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disp = geometry.disparity
        sig = geometry.sigma
        if shift:
            disp = cyclic_shift(disp, shift_amount)
            sig = cyclic_shift(sig, shift_amount)
        disp_w = window_partition(disp, self.window_size).squeeze(-1)
        sig_w = window_partition(sig, self.window_size).squeeze(-1)
        return disp_w, sig_w

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | Sequence[torch.Tensor],
        *,
        shift: bool = False,
        mask: torch.Tensor | None = None,
        rotary: RotarySpec | None = None,
        direction: StereoDirection = StereoDirection.LEFT_TO_RIGHT,
        context_geometry: StereoGeometry | None = None,
    ) -> torch.Tensor:
        batch, _, height, width = x.shape
        full_width = self.window_size[1] == -1
        window_h = self.window_size[0]
        window_w = width if full_width else self.window_size[1]
        runtime_window: WindowSize = (window_h, window_w)
        if height % window_h != 0 or width % window_w != 0:
            raise ValueError(
                "spatial dims "
                f"({height}, {width}) must divide by window_size "
                f"{_format_window_size(runtime_window)}"
            )

        contexts = [context] if isinstance(context, torch.Tensor) else list(context)
        num_contexts = len(contexts)
        shift_w = 0 if full_width else window_w // 2
        shift_amount = (window_h // 2, shift_w) if shift else (0, 0)

        ctx_disp: torch.Tensor | None = None
        ctx_sigma: torch.Tensor | None = None
        if context_geometry is not None and self.rect_disp_rope is not None:
            ctx_disp, ctx_sigma = self._partition_geometry(context_geometry, shift_amount, shift)

        if shift:
            x = cyclic_shift(x, shift_amount)
            contexts = [cyclic_shift(ctx, shift_amount) for ctx in contexts]

        q_windows = window_partition(x, runtime_window)
        ctx_windows = [window_partition(ctx, runtime_window) for ctx in contexts]

        q = rearrange(self.q_proj(q_windows), "bw s (h d) -> bw s h d", h=self.num_heads)
        q = self.q_norm(q)
        if self.epipolar_rope is not None:
            q = self.epipolar_rope.apply_q(q, height=window_h, width=window_w)
        elif self.rect_disp_rope is not None:
            q = self.rect_disp_rope.apply_q(q, height=window_h, width=window_w)
        elif rotary is not None:
            q = _apply_rope(q, spec=rotary, height=window_h, width=window_w)

        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for ctx_window in ctx_windows:
            kv = self.kv_proj(ctx_window)
            k, v = kv.chunk(2, dim=-1)
            k = rearrange(k, "bw s (h d) -> bw s h d", h=self.num_heads)
            v = rearrange(v, "bw s (h d) -> bw s h d", h=self.num_heads)
            k = self.k_norm(k)
            if self.epipolar_rope is not None:
                mu, sigma = self.epipolar_rope.predict_offset(ctx_window)
                k = self.epipolar_rope.apply_k(
                    k,
                    height=window_h,
                    width=window_w,
                    mu=mu,
                    sigma=sigma,
                    direction=direction,
                )
            elif (
                self.rect_disp_rope is not None and ctx_disp is not None and ctx_sigma is not None
            ):
                k = self.rect_disp_rope.apply_k(
                    k,
                    height=window_h,
                    width=window_w,
                    disparity=ctx_disp,
                    sigma=ctx_sigma,
                    direction=direction,
                )
            elif rotary is not None:
                k = _apply_rope(k, spec=rotary, height=window_h, width=window_w)
            ks.append(k)
            vs.append(v)
        k = torch.cat(ks, dim=1)
        v = torch.cat(vs, dim=1)

        q_t = rearrange(q, "bw s h d -> bw h s d")
        k_t = rearrange(k, "bw s h d -> bw h s d")
        v_t = rearrange(v, "bw s h d -> bw h s d")

        attn_mask: torch.Tensor | None = None
        if shift:
            bool_mask = _build_cross_shift_mask(
                height, width, runtime_window, shift_amount, num_contexts, device=q.device
            )
            num_windows = bool_mask.shape[0]
            repeats = q_windows.shape[0] // num_windows
            bool_mask = bool_mask.unsqueeze(0).expand(repeats, -1, -1, -1)
            bool_mask = rearrange(bool_mask, "b nw s1 s2 -> (b nw) s1 s2")
            attn_mask = torch.zeros_like(bool_mask, dtype=q.dtype)
            attn_mask.masked_fill_(bool_mask, torch.finfo(q.dtype).min)
            attn_mask = attn_mask.unsqueeze(1)

        if mask is not None:
            attn_mask = mask if attn_mask is None else attn_mask + mask

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = self.out_proj(rearrange(attn_out, "bw h s d -> bw s (h d)"))
        out = window_merge(out, runtime_window, height, width, batch)

        if shift:
            out = cyclic_unshift(out, shift_amount)

        return out
