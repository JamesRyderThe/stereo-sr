from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from sissr.configs.schema import DataConfig, ExperimentConfig, TrainConfig
from sissr.data.loading import build_test_loader, build_train_loader, seed_worker


def _make_config(dataset_root: Path, *, num_workers: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            dataset_root=str(dataset_root),
            num_workers=num_workers,
            prefetch_factor=3,
            batch_size=2,
        ),
        train=TrainConfig(seed=123),
    )


def test_build_train_loader_propagates_prefetch_factor(
    synthetic_dataset_dir: Path,
) -> None:
    loader = build_train_loader(_make_config(synthetic_dataset_dir))
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True


def test_build_train_loader_accepts_sampler(synthetic_dataset_dir: Path) -> None:
    sampler = SequentialSampler(range(2))
    loader = build_train_loader(_make_config(synthetic_dataset_dir), sampler=sampler)
    assert loader.sampler is sampler
    assert loader.batch_sampler.sampler is sampler


def test_build_test_loader_accepts_sampler(synthetic_dataset_dir: Path) -> None:
    sampler = SequentialSampler(range(1))
    loader = build_test_loader(_make_config(synthetic_dataset_dir), "flickr1024", sampler=sampler)
    assert loader.sampler is sampler


def test_build_train_loader_zero_workers(synthetic_dataset_dir: Path) -> None:
    loader = build_train_loader(_make_config(synthetic_dataset_dir, num_workers=0))
    assert loader.prefetch_factor is None
    assert loader.persistent_workers is False
    assert loader.num_workers == 0


def test_seed_worker_is_deterministic() -> None:
    torch.manual_seed(123)
    seed_worker(0)
    first = (
        int(torch.randint(0, 1_000_000, (1,)).item()),
        float(np.random.rand()),
        random.randint(0, 1_000_000),
    )

    torch.manual_seed(123)
    seed_worker(0)
    second = (
        int(torch.randint(0, 1_000_000, (1,)).item()),
        float(np.random.rand()),
        random.randint(0, 1_000_000),
    )

    torch.manual_seed(124)
    seed_worker(0)
    third = (
        int(torch.randint(0, 1_000_000, (1,)).item()),
        float(np.random.rand()),
        random.randint(0, 1_000_000),
    )

    assert first == second
    assert first != third
