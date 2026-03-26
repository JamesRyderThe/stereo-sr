from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from sissr.models.enums import CrossPosEncoding, ModelName, ResiConnection, ResidualStrategy


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"

    def build(self, params: object, **kwargs: object) -> torch.optim.Optimizer:
        cls = {self.ADAM: torch.optim.Adam, self.ADAMW: torch.optim.AdamW}[self]
        return cls(params, **kwargs)  # type: ignore[no-any-return]


class Scheduler(str, Enum):
    COSINE = "cosine"
    MULTISTEP = "multistep"

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_iters: int = 0,
        eta_min: float = 0.0,
        milestones: list[int] | None = None,
    ) -> CosineAnnealingLR | MultiStepLR:
        if self == self.COSINE:
            return CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=eta_min)
        return MultiStepLR(optimizer, milestones=milestones or [], gamma=0.5)


class CompileMode(str, Enum):
    OFF = "off"
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        if self == self.OFF:
            return model
        return torch.compile(model, mode=self.value)  # type: ignore[return-value]


class Precision(str, Enum):
    BF16 = "bf16"
    FP32 = "fp32"

    @property
    def accelerate_value(self) -> str:
        return "no" if self == self.FP32 else self.value


class BatchMixMode(str, Enum):
    OFF = "off"
    MIXUP = "mixup"


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BaseModelConfig(FrozenConfigModel):
    embed_dim: int = 180
    num_heads: Annotated[int, Field(ge=2)] = 6
    num_blocks: int = 13
    window_size: Annotated[int, Field(ge=1)] = 16
    upscale: int = 4
    img_range: float = 1.0

    @model_validator(mode="after")
    def _check_common_heads(self) -> BaseModelConfig:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        return self


class DiffSSRModelConfig(BaseModelConfig):
    name: Literal[ModelName.DIFFSSR] = ModelName.DIFFSSR
    block_depth: int = 3
    mlp_ratio: float = 2.0
    resi_connection: ResiConnection = ResiConnection.ONE_CONV

    @model_validator(mode="after")
    def _check_diffssr_heads(self) -> DiffSSRModelConfig:
        if self.num_heads % 2 != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be even (MultiheadDiffAttn halves head count)"
            )
        return self

    def build(self) -> torch.nn.Module:
        from sissr.baselines.diffssr.model import DIFFSSR

        return DIFFSSR(
            embed_dim=self.embed_dim,
            depths=(self.block_depth,) * self.num_blocks,
            num_heads=(self.num_heads,) * self.num_blocks,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            upscale=self.upscale,
            img_range=self.img_range,
            resi_connection=self.resi_connection,
        )


class StereoSRModelConfig(BaseModelConfig):
    name: Literal[ModelName.STEREO_SR] = ModelName.STEREO_SR
    embed_dim: int = 192
    num_blocks: int = 36
    mlp_hidden_dim: int = 384
    cross_window_size: tuple[int, int] = (8, 32)
    cross_pos_encoding: CrossPosEncoding = CrossPosEncoding.WINDOW_ROPE
    residual_strategy: ResidualStrategy = ResidualStrategy.DEPTH_AGG
    num_kv_heads: int | None = None
    blocks_per_group: Annotated[int, Field(ge=1)] | None = None
    layer_scale_init: float = 1e-5

    @model_validator(mode="after")
    def _check_stereo_sr_heads(self) -> StereoSRModelConfig:
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads "
                f"({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        head_dim = self.embed_dim // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"stereo_sr head_dim ({head_dim}) must be even for rotary embeddings")
        cross_h, cross_w = self.cross_window_size
        if cross_h < 2:
            raise ValueError("cross_window_size height must be at least 2")
        if cross_w < 2 and cross_w != -1:
            raise ValueError("cross_window_size width must be at least 2 (or -1 for full width)")
        if cross_h % 2 != 0:
            raise ValueError("cross_window_size height must be even for shifted windows")
        if cross_w != -1 and cross_w % 2 != 0:
            raise ValueError("cross_window_size width must be even for shifted windows (or -1)")
        if cross_w == -1 and self.cross_pos_encoding != CrossPosEncoding.NONE:
            raise ValueError(
                "full-width cross-attention (cross_window_size width = -1) "
                "requires cross_pos_encoding = none"
            )
        if (
            self.blocks_per_group not in (None, 1)
            and self.residual_strategy != ResidualStrategy.DEPTH_AGG
        ):
            raise ValueError("blocks_per_group requires residual_strategy='depth_agg'")
        return self

    def build(self) -> torch.nn.Module:
        from sissr.models.stereo_sr import StereoSRModel

        blocks_per_group = self.blocks_per_group
        if blocks_per_group is None:
            blocks_per_group = 8 if self.residual_strategy == ResidualStrategy.DEPTH_AGG else 1

        return StereoSRModel(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            window_size=self.window_size,
            cross_window_size=self.cross_window_size,
            cross_pos_encoding=self.cross_pos_encoding,
            residual_strategy=self.residual_strategy,
            mlp_hidden_dim=self.mlp_hidden_dim,
            num_kv_heads=self.num_kv_heads,
            blocks_per_group=blocks_per_group,
            upscale=self.upscale,
            img_range=self.img_range,
            layer_scale_init=self.layer_scale_init,
        )


ModelConfig: TypeAlias = Annotated[
    DiffSSRModelConfig | StereoSRModelConfig,
    Field(discriminator="name"),
]


class AugmentConfig(FrozenConfigModel):
    hflip_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    vflip_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    channel_shuffle_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    horizontal_shift_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    horizontal_shift_max_px: Annotated[int, Field(ge=0)] = 8
    batch_mix_mode: BatchMixMode = BatchMixMode.MIXUP
    batch_mix_alpha: Annotated[float, Field(gt=0.0)] = 1.0


class DataConfig(FrozenConfigModel):
    dataset_root: str = "datasets/StereoSR"
    test_datasets: list[str] = ["flickr1024", "kitti2012", "kitti2015", "middlebury"]
    scale: Literal[2, 4] = 4
    patch_size_lr: tuple[int, int] = (32, 96)
    batch_size: int = 4
    num_workers: Annotated[int, Field(ge=0)] = 4
    prefetch_factor: Annotated[int, Field(ge=1)] = 2
    augment: AugmentConfig = AugmentConfig()
    use_geometry: bool = False
    geometry_dir: str = "geometry"


class TrainConfig(FrozenConfigModel):
    optimizer: Optimizer = Optimizer.ADAM
    lr: float = 2e-4
    total_iters: int = 800_000
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    scheduler: Scheduler = Scheduler.COSINE
    scheduler_milestones: list[int] = [400_000, 600_000]
    gradient_accumulation_steps: Annotated[int, Field(ge=1)] = 1
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    compile: CompileMode = CompileMode.DEFAULT
    precision: Precision = Precision.BF16
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 50_000
    max_checkpoints: Annotated[int, Field(ge=1)] = 3
    eta_min: Annotated[float, Field(ge=0)] = 1e-7
    log_interval: int = 100
    val_interval: int = 10_000
    seed: int = 42
    resume_from: str = ""
    wandb_project: str = "james-sissr"
    wandb_run_name: str = ""


class LossConfig(FrozenConfigModel):
    charbonnier_weight: Annotated[float, Field(ge=0)] = 1.0
    charbonnier_eps: float = 1e-12
    warp_consistency_weight: Annotated[float, Field(ge=0)] = 0.0
    disparity_preservation_weight: Annotated[float, Field(ge=0)] = 0.0
    disparity_loss_interval: Annotated[int, Field(ge=1)] = 1


class ExperimentConfig(FrozenConfigModel):
    model: ModelConfig = Field(default_factory=DiffSSRModelConfig)
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    loss: LossConfig = LossConfig()
    experiment_name: str = "unnamed"
    output_dir: str = "outputs"

    @model_validator(mode="before")
    @classmethod
    def _default_model_name(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw_model = data.get("model")
        if not isinstance(raw_model, dict) or "name" in raw_model:
            return data
        return {**data, "model": {"name": ModelName.DIFFSSR, **raw_model}}

    @model_validator(mode="after")
    def _check_cross_field(self) -> ExperimentConfig:
        if self.data.scale != self.model.upscale:
            raise ValueError(
                f"data.scale ({self.data.scale}) != model.upscale ({self.model.upscale})"
            )
        if self.data.use_geometry and self.data.augment.batch_mix_mode != BatchMixMode.OFF:
            raise ValueError("batch_mix_mode must be 'off' when use_geometry is true")
        ph, pw = self.data.patch_size_lr
        ws = self.model.window_size
        if isinstance(self.model, DiffSSRModelConfig):
            if ph % ws != 0 or pw % ws != 0:
                raise ValueError(
                    f"patch_size_lr ({ph}, {pw}) must be divisible by window_size ({ws})"
                )
            if ph % 4 != 0:
                raise ValueError(
                    f"patch_size_lr height ({ph}) must be divisible by 4 (SSCAM window height)"
                )
            return self
        cross_h, cross_w = self.model.cross_window_size
        req_h = math.lcm(ws, cross_h)
        req_w = ws if cross_w == -1 else math.lcm(ws, cross_w)
        if ph % req_h != 0 or pw % req_w != 0:
            raise ValueError(
                "patch_size_lr "
                f"({ph}, {pw}) must be divisible by combined stereo windows ({req_h}, {req_w})"
            )
        return self
