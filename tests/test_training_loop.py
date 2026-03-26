from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

from sissr.configs.schema import (
    AugmentConfig,
    BatchMixMode,
    CompileMode,
    DataConfig,
    DiffSSRModelConfig,
    ExperimentConfig,
    LossConfig,
    Precision,
    StereoSRModelConfig,
    TrainConfig,
)
from sissr.data.datasets import StereoSRBatch
from sissr.models.enums import CrossPosEncoding, ResidualStrategy
from sissr.train.trainer import Trainer, _batch_stream, _finalize_eval_metrics, build_model


class _CPUAccelerator(Accelerator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, cpu=True, **kwargs)


class _ReduceStub:
    def __init__(self, reduced: torch.Tensor, *, is_main_process: bool) -> None:
        self.device = torch.device("cpu")
        self.is_main_process = is_main_process
        self._reduced = reduced

    def reduce(
        self,
        tensor: torch.Tensor,
        reduction: str = "sum",
        scale: float = 1.0,
    ) -> torch.Tensor:
        assert reduction == "sum"
        assert scale == 1.0
        assert tensor.dtype == torch.float64
        return self._reduced


def _tiny_model_config() -> DiffSSRModelConfig:
    return DiffSSRModelConfig(
        embed_dim=12,
        num_heads=2,
        num_blocks=1,
        block_depth=1,
        window_size=4,
        mlp_ratio=2.0,
        upscale=4,
    )


def _disabled_augment(*, batch_mix_mode: BatchMixMode = BatchMixMode.OFF) -> AugmentConfig:
    return AugmentConfig(
        hflip_prob=0.0,
        vflip_prob=0.0,
        channel_shuffle_prob=0.0,
        horizontal_shift_prob=0.0,
        batch_mix_mode=batch_mix_mode,
    )


def _tiny_dataset(tmp_path: Path, *, test_datasets: tuple[str, ...] = ("flickr1024",)) -> Path:
    hr_h, hr_w = 16, 16
    lr_h, lr_w = 4, 4
    split_dirs = [(tmp_path / "train" / "hr", tmp_path / "train" / "lr_x4")]
    split_dirs.extend(
        (
            tmp_path / "test" / dataset_name / "hr",
            tmp_path / "test" / dataset_name / "lr_x4",
        )
        for dataset_name in test_datasets
    )
    for split_dir_pair in split_dirs:
        for d in split_dir_pair:
            d.mkdir(parents=True, exist_ok=True)

    for stem in ("0001", "0002"):
        for suffix in ("_L.png", "_R.png"):
            Image.fromarray(np.random.randint(0, 256, (hr_h, hr_w, 3), dtype=np.uint8)).save(
                tmp_path / "train" / "hr" / f"{stem}{suffix}"
            )
            Image.fromarray(np.random.randint(0, 256, (lr_h, lr_w, 3), dtype=np.uint8)).save(
                tmp_path / "train" / "lr_x4" / f"{stem}{suffix}"
            )

    for dataset_name in test_datasets:
        for suffix in ("_L.png", "_R.png"):
            Image.fromarray(np.random.randint(0, 256, (hr_h, hr_w, 3), dtype=np.uint8)).save(
                tmp_path / "test" / dataset_name / "hr" / f"0001{suffix}"
            )
            Image.fromarray(np.random.randint(0, 256, (lr_h, lr_w, 3), dtype=np.uint8)).save(
                tmp_path / "test" / dataset_name / "lr_x4" / f"0001{suffix}"
            )

    return tmp_path


def _tiny_config(dataset_root: Path, ckpt_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model=_tiny_model_config(),
        data=DataConfig(
            dataset_root=str(dataset_root),
            test_datasets=["flickr1024"],
            patch_size_lr=(4, 4),
            batch_size=2,
            num_workers=0,
            augment=_disabled_augment(),
        ),
        train=TrainConfig(
            total_iters=4,
            checkpoint_interval=2,
            val_interval=4,
            log_interval=1,
            wandb_project="",
            compile=CompileMode.OFF,
            precision=Precision.FP32,
            seed=42,
            checkpoint_dir=str(ckpt_dir),
        ),
        loss=LossConfig(charbonnier_eps=1e-12),
    )


@pytest.fixture(autouse=True)
def _force_cpu_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sissr.train.trainer.Accelerator", _CPUAccelerator)


def test_smoke_params_update_and_loss_finite(tmp_path: Path) -> None:
    dataset_root = _tiny_dataset(tmp_path)
    config = _tiny_config(dataset_root, tmp_path / "ckpts")
    trainer = Trainer(config)

    params_before = {n: p.clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    trainer.run()
    params_after = dict(trainer.model.named_parameters())

    changed = sum(
        1 for n in params_before if not torch.equal(params_before[n], params_after[n].data)
    )
    assert changed > 0
    assert trainer.global_step == 4
    assert (tmp_path / "ckpts" / "latest").exists()


def test_resume_restores_step(tmp_path: Path) -> None:
    dataset_root = _tiny_dataset(tmp_path)
    ckpt_dir = tmp_path / "ckpts"
    config = _tiny_config(dataset_root, ckpt_dir)

    Trainer(config).run()
    assert (ckpt_dir / "latest").exists()

    config_resume = ExperimentConfig(
        **{
            **config.model_dump(),
            "train": {**config.train.model_dump(), "total_iters": 8, "resume_from": "latest"},
        }
    )
    trainer2 = Trainer(config_resume)
    assert trainer2.global_step == 4
    trainer2.run()
    assert trainer2.global_step == 8


def test_build_model_from_config() -> None:
    model = build_model(_tiny_model_config())
    assert sum(p.numel() for p in model.parameters()) > 0
    x = torch.randn(1, 6, 4, 4)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 16, 16)


@pytest.mark.parametrize(
    "cross_pos_encoding",
    [CrossPosEncoding.NONE, CrossPosEncoding.WINDOW_ROPE],
)
@pytest.mark.parametrize(
    "residual_strategy",
    [ResidualStrategy.DEPTH_AGG, ResidualStrategy.STANDARD],
)
def test_build_stereo_model_from_config(
    cross_pos_encoding: CrossPosEncoding, residual_strategy: ResidualStrategy
) -> None:
    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=1,
        window_size=4,
        cross_window_size=(4, 8),
        cross_pos_encoding=cross_pos_encoding,
        residual_strategy=residual_strategy,
        mlp_hidden_dim=64,
        upscale=4,
    )
    model = build_model(config)

    x = torch.randn(1, 6, 8, 12)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 6, 32, 48)


def test_batch_stream_yields_indefinitely(synthetic_dataset_dir: Path) -> None:
    from torch.utils.data import DataLoader

    from sissr.data.datasets import StereoSRTrainDataset

    ds = StereoSRTrainDataset(
        str(synthetic_dataset_dir),
        scale=4,
        patch_h=32,
        patch_w=96,
        augment=_disabled_augment(),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    stream = _batch_stream(loader)
    for _ in range(len(ds) * 3):
        batch = next(stream)
        assert "lr" in batch


def test_scheduler_integration() -> None:
    model = build_model(_tiny_model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] < 1e-6


def test_finalize_eval_metrics_uses_reduced_totals_on_non_main_rank() -> None:
    reducer = _ReduceStub(
        torch.tensor([42.0, 1.2, 2.0], dtype=torch.float64),
        is_main_process=False,
    )

    avg_psnr, avg_ssim = _finalize_eval_metrics(reducer, 0.0, 0.0, 0)

    assert avg_psnr == pytest.approx(21.0)
    assert avg_ssim == pytest.approx(0.6)


def test_finalize_eval_metrics_returns_zero_when_all_ranks_have_no_samples() -> None:
    reducer = _ReduceStub(
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        is_main_process=True,
    )

    avg_psnr, avg_ssim = _finalize_eval_metrics(reducer, 0.0, 0.0, 0)

    assert avg_psnr == 0.0
    assert avg_ssim == 0.0


def test_evaluate_logs_mean_across_all_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _tiny_dataset(tmp_path, test_datasets=("flickr1024", "kitti2012"))
    config = ExperimentConfig(
        model=_tiny_model_config(),
        data=DataConfig(
            dataset_root=str(dataset_root),
            test_datasets=["flickr1024", "kitti2012"],
            patch_size_lr=(4, 4),
            batch_size=2,
            num_workers=0,
            augment=_disabled_augment(),
        ),
        train=TrainConfig(
            total_iters=1,
            checkpoint_interval=1,
            val_interval=1,
            log_interval=1,
            wandb_project="",
            compile=CompileMode.OFF,
            precision=Precision.FP32,
            seed=42,
            checkpoint_dir=str(tmp_path / "ckpts"),
        ),
        loss=LossConfig(charbonnier_eps=1e-12),
    )
    trainer = Trainer(config)
    trainer.global_step = 7

    captured: list[dict[str, float]] = []

    def fake_log(metrics: dict[str, float], step: int) -> None:
        assert step == 7
        captured.append(metrics)

    values = iter([(20.0, 0.5), (30.0, 0.7)])
    monkeypatch.setattr(trainer.accelerator, "log", fake_log)
    monkeypatch.setattr(trainer, "_evaluate_dataset", lambda loader: next(values))

    mean_psnr = trainer._evaluate()

    assert mean_psnr == pytest.approx(25.0)
    assert captured == [
        {
            "val/psnr_flickr1024": 20.0,
            "val/psnr_kitti2012": 30.0,
            "val/ssim_flickr1024": 0.5,
            "val/ssim_kitti2012": 0.7,
            "val/psnr_mean": 25.0,
            "val/ssim_mean": 0.6,
        }
    ]


def test_apply_batch_mix_off_is_identity(tmp_path: Path) -> None:
    dataset_root = _tiny_dataset(tmp_path)
    config = ExperimentConfig(
        model=_tiny_model_config(),
        data=DataConfig(
            dataset_root=str(dataset_root),
            test_datasets=["flickr1024"],
            patch_size_lr=(4, 4),
            batch_size=2,
            num_workers=0,
            augment=_disabled_augment(batch_mix_mode=BatchMixMode.OFF),
        ),
        train=TrainConfig(
            total_iters=1,
            checkpoint_interval=1,
            val_interval=1,
            log_interval=1,
            wandb_project="",
            compile=CompileMode.OFF,
            precision=Precision.FP32,
            seed=42,
            checkpoint_dir=str(tmp_path / "ckpts"),
        ),
        loss=LossConfig(charbonnier_eps=1e-12),
    )
    trainer = Trainer(config)
    batch: StereoSRBatch = {
        "lr": torch.tensor([[[[1.0]]], [[[3.0]]]]),
        "gt": torch.tensor([[[[10.0]]], [[[30.0]]]]),
        "disp_left": torch.tensor([[[1.0]], [[2.0]]]),
        "disp_right": torch.tensor([[[3.0]], [[4.0]]]),
        "conf_left": torch.tensor([[[0.1]], [[0.2]]]),
        "conf_right": torch.tensor([[[0.3]], [[0.4]]]),
    }

    mixed = trainer._apply_batch_mix(batch)

    assert mixed is batch


def test_apply_batch_mix_interpolates_lr_and_gt_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _tiny_dataset(tmp_path)
    config = ExperimentConfig(
        model=_tiny_model_config(),
        data=DataConfig(
            dataset_root=str(dataset_root),
            test_datasets=["flickr1024"],
            patch_size_lr=(4, 4),
            batch_size=2,
            num_workers=0,
            augment=_disabled_augment(batch_mix_mode=BatchMixMode.MIXUP),
        ),
        train=TrainConfig(
            total_iters=1,
            checkpoint_interval=1,
            val_interval=1,
            log_interval=1,
            wandb_project="",
            compile=CompileMode.OFF,
            precision=Precision.FP32,
            seed=42,
            checkpoint_dir=str(tmp_path / "ckpts"),
        ),
        loss=LossConfig(charbonnier_eps=1e-12),
    )
    trainer = Trainer(config)
    batch: StereoSRBatch = {
        "lr": torch.tensor([[[[1.0]]], [[[3.0]]]]),
        "gt": torch.tensor([[[[10.0]]], [[[30.0]]]]),
        "disp_left": torch.tensor([[[1.0]], [[2.0]]]),
        "disp_right": torch.tensor([[[3.0]], [[4.0]]]),
        "conf_left": torch.tensor([[[0.1]], [[0.2]]]),
        "conf_right": torch.tensor([[[0.3]], [[0.4]]]),
    }
    monkeypatch.setattr(
        "sissr.train.trainer.torch.distributions.Beta.sample",
        lambda self: torch.tensor(0.25),
    )
    monkeypatch.setattr(
        "sissr.train.trainer.torch.randperm",
        lambda n, device=None: torch.tensor([1, 0], device=device),
    )

    mixed = trainer._apply_batch_mix(batch)

    assert torch.equal(mixed["lr"], torch.tensor([[[[2.5]]], [[[1.5]]]]))
    assert torch.equal(mixed["gt"], torch.tensor([[[[25.0]]], [[[15.0]]]]))
    assert torch.equal(mixed["disp_left"], batch["disp_left"])
    assert torch.equal(mixed["disp_right"], batch["disp_right"])
    assert torch.equal(mixed["conf_left"], batch["conf_left"])
    assert torch.equal(mixed["conf_right"], batch["conf_right"])
