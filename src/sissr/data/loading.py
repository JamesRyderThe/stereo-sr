from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from sissr.configs.schema import ExperimentConfig
from sissr.data.datasets import (
    StereoSRBatch,
    StereoSRTestBatch,
    StereoSRTestDataset,
    StereoSRTrainDataset,
)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_train_loader(
    config: ExperimentConfig,
    *,
    sampler: Sampler[int] | None = None,
) -> DataLoader[StereoSRBatch]:
    patch_h, patch_w = config.data.patch_size_lr
    dataset = StereoSRTrainDataset(
        root=config.data.dataset_root,
        scale=config.data.scale,
        patch_h=patch_h,
        patch_w=patch_w,
        augment=config.data.augment,
        use_geometry=config.data.use_geometry,
        geometry_dir=config.data.geometry_dir,
    )
    prefetch_factor = config.data.prefetch_factor if config.data.num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=_make_generator(config.train.seed),
        prefetch_factor=prefetch_factor,
        persistent_workers=config.data.num_workers > 0,
    )


def build_test_loader(
    config: ExperimentConfig,
    dataset_name: str,
    *,
    sampler: Sampler[int] | None = None,
) -> DataLoader[StereoSRTestBatch]:
    dataset = StereoSRTestDataset(
        root=config.data.dataset_root,
        dataset_name=dataset_name,
        scale=config.data.scale,
        use_geometry=config.data.use_geometry,
        geometry_dir=config.data.geometry_dir,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=0)
