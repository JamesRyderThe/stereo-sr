from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from sissr.models.enums import ResidualStrategy
from sissr.models.layers.depth_agg import DepthAggregator, DepthState
from sissr.models.layers.embedding import (
    AxisConfig,
    RotarySpec,
    SpatialRoPE,
    VisionGrid,
)
from sissr.models.layers.layerscale import LayerScale
from sissr.models.layers.mlp import SwiGLU
from sissr.models.layers.norm import ChannelRMSNorm, RMSNorm
from sissr.models.layers.window_attn import WindowedAttention, WindowedCrossAttention


def _to_tokens(x: Tensor) -> Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_map(x: Tensor, *, height: int, width: int) -> Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


@dataclass(eq=False)
class StereoDepthPair:
    left: DepthState
    right: DepthState

    @classmethod
    def create(
        cls,
        left_agg: DepthAggregator,
        right_agg: DepthAggregator,
        left_tokens: Tensor,
        right_tokens: Tensor,
    ) -> StereoDepthPair:
        return cls(
            left=DepthState.create(left_agg, left_tokens),
            right=DepthState.create(right_agg, right_tokens),
        )

    def commit_boundary(self) -> None:
        self.left.commit_boundary()
        self.right.commit_boundary()

    def finalize(self, *, skip_first: bool = False) -> tuple[Tensor, Tensor]:
        return self.left.finalize(skip_first=skip_first), self.right.finalize(
            skip_first=skip_first
        )


class _StereoBlockBase(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        cross_window_size: tuple[int, int],
        mlp_hidden_dim: int,
        *,
        shift: bool,
        num_kv_heads: int | None,
        residual_v: bool,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.shift = shift
        head_dim = dim // num_heads
        self.window_rope = RotarySpec.spatial(
            SpatialRoPE(AxisConfig.from_head_dim(head_dim)),
            VisionGrid(height=window_size, width=window_size),
        )

        self.self_norm = ChannelRMSNorm(dim, learnable=True)
        self.self_attn = WindowedAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            qk_norm=True,
            gate=True,
            residual_v=residual_v,
            xsa=True,
        )
        self.self_scale = LayerScale(dim, init_value=layer_scale_init)

        self.cross_norm = ChannelRMSNorm(dim, learnable=True)
        self.cross_attn = WindowedCrossAttention(
            embed_dim=dim,
            num_heads=num_heads,
            window_size=cross_window_size,
        )

        self.mlp_norm = RMSNorm(dim, learnable=True)
        self.mlp = SwiGLU(dim=dim, hidden_dim=mlp_hidden_dim)
        self.mlp_scale = LayerScale(dim, init_value=layer_scale_init)


class StereoBlock(_StereoBlockBase):
    def forward(
        self,
        pair: StereoDepthPair,
        *,
        height: int,
        width: int,
        left_v0: Tensor | None,
        right_v0: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        left_v, right_v = self._self_attn_sublayer(
            pair, height=height, width=width, left_v0=left_v0, right_v0=right_v0
        )
        self._cross_attn_sublayer(pair, height=height, width=width)
        self._mlp_sublayer(pair)
        return left_v, right_v

    def _self_attn_sublayer(
        self,
        pair: StereoDepthPair,
        *,
        height: int,
        width: int,
        left_v0: Tensor | None,
        right_v0: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        both_h = self.self_norm(
            _to_map(
                torch.cat([pair.left.aggregate(), pair.right.aggregate()], dim=0),
                height=height,
                width=width,
            )
        )
        v0: Tensor | None = None
        if left_v0 is not None and right_v0 is not None:
            v0 = torch.cat([left_v0, right_v0], dim=0)
        both_out, both_v = self.self_attn(
            both_h, shift=self.shift, rotary=self.window_rope, v0=v0, return_value=True
        )
        both_scaled = _to_tokens(self.self_scale(both_out))
        left_scaled, right_scaled = both_scaled.chunk(2, dim=0)
        pair.left.accumulate(left_scaled)
        pair.right.accumulate(right_scaled)

        left_v, right_v = both_v.chunk(2, dim=0)
        return left_v, right_v

    def _cross_attn_sublayer(self, pair: StereoDepthPair, *, height: int, width: int) -> None:
        both_normed = self.cross_norm(
            _to_map(
                torch.cat([pair.left.aggregate(), pair.right.aggregate()], dim=0),
                height=height,
                width=width,
            )
        )
        left_normed, right_normed = both_normed.chunk(2, dim=0)

        left_delta, right_delta = self.cross_attn(left_normed, right_normed, shift=self.shift)

        pair.left.accumulate(_to_tokens(left_delta))
        pair.right.accumulate(_to_tokens(right_delta))

    def _mlp_sublayer(self, pair: StereoDepthPair) -> None:
        both = torch.cat([pair.left.aggregate(), pair.right.aggregate()], dim=0)
        both_out = self.mlp_scale(self.mlp(self.mlp_norm(both)))
        left_out, right_out = both_out.chunk(2, dim=0)
        pair.left.accumulate(left_out)
        pair.right.accumulate(right_out)


class StandardStereoBlock(_StereoBlockBase):
    def forward(
        self,
        left: Tensor,
        right: Tensor,
        *,
        height: int,
        width: int,
        left_v0: Tensor | None,
        right_v0: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        both_map = self.self_norm(
            _to_map(torch.cat([left, right], dim=0), height=height, width=width)
        )
        v0: Tensor | None = None
        if left_v0 is not None and right_v0 is not None:
            v0 = torch.cat([left_v0, right_v0], dim=0)
        both_sa, both_v = self.self_attn(
            both_map, shift=self.shift, rotary=self.window_rope, v0=v0, return_value=True
        )
        both_scaled = _to_tokens(self.self_scale(both_sa))
        left_scaled, right_scaled = both_scaled.chunk(2, dim=0)
        left = left + left_scaled
        right = right + right_scaled
        left_v, right_v = both_v.chunk(2, dim=0)

        both_normed = self.cross_norm(
            _to_map(torch.cat([left, right], dim=0), height=height, width=width)
        )
        left_normed, right_normed = both_normed.chunk(2, dim=0)

        left_delta, right_delta = self.cross_attn(left_normed, right_normed, shift=self.shift)
        left = left + _to_tokens(left_delta)
        right = right + _to_tokens(right_delta)

        both_mlp = self.mlp_scale(self.mlp(self.mlp_norm(torch.cat([left, right], dim=0))))
        left_mlp, right_mlp = both_mlp.chunk(2, dim=0)
        left = left + left_mlp
        right = right + right_mlp

        return left, right, left_v, right_v


def _build_stereo_blocks(
    block_cls: type[_StereoBlockBase],
    dim: int,
    num_heads: int,
    num_blocks: int,
    window_size: int,
    cross_window_size: tuple[int, int],
    mlp_hidden_dim: int,
    *,
    num_kv_heads: int | None,
    layer_scale_init: float,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            block_cls(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                cross_window_size=cross_window_size,
                mlp_hidden_dim=mlp_hidden_dim,
                shift=idx % 2 == 1,
                num_kv_heads=num_kv_heads,
                residual_v=idx > 0,
                layer_scale_init=layer_scale_init,
            )
            for idx in range(num_blocks)
        ]
    )


class StandardStereoBody(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_blocks: int,
        window_size: int,
        cross_window_size: tuple[int, int],
        mlp_hidden_dim: int,
        *,
        num_kv_heads: int | None,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.blocks = _build_stereo_blocks(
            StandardStereoBlock,
            dim,
            num_heads,
            num_blocks,
            window_size,
            cross_window_size,
            mlp_hidden_dim,
            num_kv_heads=num_kv_heads,
            layer_scale_init=layer_scale_init,
        )

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        _, _, height, width = left.shape
        left_t = _to_tokens(left)
        right_t = _to_tokens(right)
        left_v0: Tensor | None = None
        right_v0: Tensor | None = None

        for block in self.blocks:
            left_t, right_t, left_v, right_v = block(
                left_t, right_t, height=height, width=width, left_v0=left_v0, right_v0=right_v0
            )
            if left_v0 is None:
                left_v0 = left_v
            if right_v0 is None:
                right_v0 = right_v

        return (
            _to_map(left_t, height=height, width=width),
            _to_map(right_t, height=height, width=width),
        )


class StereoBody(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_blocks: int,
        window_size: int,
        cross_window_size: tuple[int, int],
        mlp_hidden_dim: int,
        *,
        num_kv_heads: int | None,
        layer_scale_init: float,
        blocks_per_group: int,
    ) -> None:
        super().__init__()
        if blocks_per_group < 1:
            raise ValueError(f"blocks_per_group must be positive, got {blocks_per_group}")
        self.blocks_per_group = blocks_per_group
        self.blocks = _build_stereo_blocks(
            StereoBlock,
            dim,
            num_heads,
            num_blocks,
            window_size,
            cross_window_size,
            mlp_hidden_dim,
            num_kv_heads=num_kv_heads,
            layer_scale_init=layer_scale_init,
        )
        num_groups = math.ceil(num_blocks / blocks_per_group)
        max_sources = num_groups + 1
        self.left_depth = DepthAggregator(dim, 3 * num_blocks + 1, max_sources=max_sources)
        self.right_depth = DepthAggregator(dim, 3 * num_blocks + 1, max_sources=max_sources)

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        _, _, height, width = left.shape
        pair = StereoDepthPair.create(
            self.left_depth, self.right_depth, _to_tokens(left), _to_tokens(right)
        )
        pair.commit_boundary()

        left_v0: Tensor | None = None
        right_v0: Tensor | None = None

        for idx, block in enumerate(self.blocks):
            left_v, right_v = block(
                pair, height=height, width=width, left_v0=left_v0, right_v0=right_v0
            )
            if left_v0 is None:
                left_v0 = left_v
            if right_v0 is None:
                right_v0 = right_v
            if idx < len(self.blocks) - 1 and (idx + 1) % self.blocks_per_group == 0:
                pair.commit_boundary()

        left_out, right_out = pair.finalize(skip_first=True)
        return (
            _to_map(left_out, height=height, width=width),
            _to_map(right_out, height=height, width=width),
        )


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        modules: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            modules.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super().__init__(*modules)


class StereoSRModel(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_blocks: int = 36,
        window_size: int = 16,
        cross_window_size: tuple[int, int] = (4, -1),
        residual_strategy: ResidualStrategy = ResidualStrategy.DEPTH_AGG,
        mlp_hidden_dim: int = 384,
        num_kv_heads: int | None = None,
        blocks_per_group: int | None = None,
        upscale: int = 4,
        img_range: float = 1.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        if blocks_per_group is None:
            blocks_per_group = 8 if residual_strategy == ResidualStrategy.DEPTH_AGG else 1
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for rotary embeddings")
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        cross_h, cross_w = cross_window_size
        if cross_h < 2:
            raise ValueError("cross_window_size height must be at least 2")
        if cross_w < 2 and cross_w != -1:
            raise ValueError("cross_window_size width must be at least 2 (or -1 for full width)")
        if cross_h % 2 != 0:
            raise ValueError("cross_window_size height must be even for shifted windows")
        if cross_w != -1 and cross_w % 2 != 0:
            raise ValueError("cross_window_size width must be even for shifted windows (or -1)")
        if blocks_per_group < 1:
            raise ValueError(f"blocks_per_group must be positive, got {blocks_per_group}")
        if residual_strategy != ResidualStrategy.DEPTH_AGG and blocks_per_group != 1:
            raise ValueError("blocks_per_group requires residual_strategy='depth_agg'")

        self.img_range = img_range
        self.upscale = upscale
        self.window_size = window_size
        self.cross_window_size = cross_window_size
        pad_w = window_size if cross_w == -1 else math.lcm(window_size, cross_w)
        self.pad_window_size = (math.lcm(window_size, cross_h), pad_w)
        mean = torch.tensor((0.4488, 0.4371, 0.4040), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.mean: Tensor

        self.stem = nn.Conv2d(3, embed_dim, 3, 1, 1)
        if residual_strategy == ResidualStrategy.DEPTH_AGG:
            self.body: StereoBody | StandardStereoBody = StereoBody(
                dim=embed_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                window_size=window_size,
                cross_window_size=cross_window_size,
                mlp_hidden_dim=mlp_hidden_dim,
                num_kv_heads=num_kv_heads,
                layer_scale_init=layer_scale_init,
                blocks_per_group=blocks_per_group,
            )
        else:
            self.body = StandardStereoBody(
                dim=embed_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                window_size=window_size,
                cross_window_size=cross_window_size,
                mlp_hidden_dim=mlp_hidden_dim,
                num_kv_heads=num_kv_heads,
                layer_scale_init=layer_scale_init,
            )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.pre_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.upsample = Upsample(upscale, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)

    def _restore_view(self, x: Tensor) -> Tensor:
        x = self.pre_upsample(x)
        x = self.upsample(x)
        result: Tensor = self.out(x)
        return result

    def _pad_to_window(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        pad_h = (self.pad_window_size[0] - h % self.pad_window_size[0]) % self.pad_window_size[0]
        pad_w = (self.pad_window_size[1] - w % self.pad_window_size[1]) % self.pad_window_size[1]
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != 6:
            raise ValueError(f"StereoSRModel expects 6-channel stereo input, got {x.shape[1]}")

        left = x[:, :3]
        right = x[:, 3:]
        orig_h, orig_w = left.shape[2], left.shape[3]
        left = self._pad_to_window(left)
        right = self._pad_to_window(right)
        mean = self.mean.to(dtype=x.dtype, device=x.device)

        left = (left - mean) * self.img_range
        right = (right - mean) * self.img_range

        both_shallow = self.stem(torch.cat([left, right], dim=0))
        left_shallow, right_shallow = both_shallow.chunk(2, dim=0)

        left_body, right_body = self.body(left_shallow, right_shallow)

        both_after = self.conv_after_body(torch.cat([left_body, right_body], dim=0))
        left_after, right_after = both_after.chunk(2, dim=0)
        both = (
            self._restore_view(
                torch.cat([left_after + left_shallow, right_after + right_shallow], dim=0)
            )
            / self.img_range
            + mean
        )
        left, right = both.chunk(2, dim=0)
        out = torch.cat([left, right], dim=1)
        return out[:, :, : orig_h * self.upscale, : orig_w * self.upscale]
