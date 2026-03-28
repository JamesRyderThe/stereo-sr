from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch.utils.data import DataLoader

from sissr.configs.schema import BatchMixMode, ExperimentConfig, ModelConfig
from sissr.data.datasets import StereoSRBatch, StereoSRTestBatch
from sissr.data.loading import build_test_loader, build_train_loader
from sissr.eval.metrics import compute_stereo_metrics
from sissr.losses.charbonnier import stereo_charbonnier_loss
from sissr.train.checkpoint import CheckpointManager
from sissr.train.console import Console
from sissr.train.ema import ExponentialMovingAverage

log = logging.getLogger(__name__)


def build_model(config: ModelConfig) -> torch.nn.Module:
    return config.build()


def _batch_stream(loader: DataLoader[StereoSRBatch]) -> Iterator[StereoSRBatch]:
    while True:
        yield from loader


class _EvalMetricReducer(Protocol):
    device: torch.device
    is_main_process: bool

    def reduce(self, tensor: Tensor, reduction: str = "sum", scale: float = 1.0) -> Tensor: ...


def _finalize_eval_metrics(
    reducer: _EvalMetricReducer,
    psnr_sum: float,
    ssim_sum: float,
    count: int,
) -> tuple[float, float]:
    totals = torch.tensor(
        [psnr_sum, ssim_sum, float(count)],
        device=reducer.device,
        dtype=torch.float64,
    )
    reduced = reducer.reduce(totals, reduction="sum")
    total_count = int(reduced[2].item())
    if total_count == 0:
        if reducer.is_main_process:
            log.warning("Evaluation produced zero samples")
        return 0.0, 0.0
    return (
        float(reduced[0].item()) / total_count,
        float(reduced[1].item()) / total_count,
    )


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.global_step = 0

        self.accelerator = Accelerator(
            mixed_precision=config.train.precision.accelerate_value,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            log_with="wandb" if config.train.wandb_project else None,
        )
        self.console = Console(enabled=self.accelerator.is_main_process)

        set_seed(config.train.seed)

        unwrapped: torch.nn.Module = build_model(config.model)

        optimizer = config.train.optimizer.build(
            unwrapped.parameters(),
            lr=config.train.lr,
            betas=config.train.betas,
            weight_decay=config.train.weight_decay,
        )

        scheduler = config.train.scheduler.build(
            optimizer,
            total_iters=config.train.total_iters,
            eta_min=config.train.eta_min,
            milestones=config.train.scheduler_milestones,
        )

        compiled = config.train.compile.apply(unwrapped)
        self._eval_model = compiled
        if config.train.compile.value != "off":
            self.console.print(
                f"[cyan]torch.compile enabled (mode={config.train.compile.value}). "
                "First step will be slow while kernels compile...[/]"
            )

        self.ema = ExponentialMovingAverage(unwrapped.parameters(), decay=config.train.ema_decay)
        self.train_loader = build_train_loader(config)
        self.test_loaders = {
            dataset_name: build_test_loader(config, dataset_name)
            for dataset_name in config.data.test_datasets
        }

        self.accelerator.register_for_checkpointing(self.ema)
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            compiled, optimizer, self.train_loader, scheduler
        )
        self.ema.to(device=self.accelerator.device)

        self.ckpt_mgr = CheckpointManager(
            config.train.checkpoint_dir, config.train.max_checkpoints
        )
        self._resume()
        self._init_trackers()

        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            self.console.config_summary(
                model_name=config.model.name,
                params=total_params,
                total_iters=config.train.total_iters,
                batch_size=config.data.batch_size,
                accum_steps=config.train.gradient_accumulation_steps,
                num_gpus=self.accelerator.num_processes,
                compile_mode=config.train.compile.value,
                precision=config.train.precision.value,
            )

    def _resume(self) -> None:
        self._wandb_run_id: str | None = None
        if not self.config.train.resume_from:
            return
        resume_path = self.ckpt_mgr.resolve(self.config.train.resume_from)
        if resume_path is not None:
            self.accelerator.load_state(str(resume_path))
            self.global_step = self.ckpt_mgr.load_step(resume_path)
            self._wandb_run_id = self.ckpt_mgr.load_wandb_run_id(resume_path)
            log.info("Resumed from step %d at %s", self.global_step, resume_path)

    def _init_trackers(self) -> None:
        if not self.config.train.wandb_project:
            return
        wandb_kwargs: dict[str, object] = {
            "name": self.config.train.wandb_run_name or self.config.experiment_name,
        }
        if self._wandb_run_id is not None:
            wandb_kwargs["id"] = self._wandb_run_id
            wandb_kwargs["resume"] = "allow"
        self.accelerator.init_trackers(
            project_name=self.config.train.wandb_project,
            config=self.config.model_dump(),
            init_kwargs={"wandb": wandb_kwargs},
        )

    def _get_wandb_run_id(self) -> str | None:
        if not self.config.train.wandb_project:
            return None
        if not self.accelerator.is_main_process:
            return None
        try:
            tracker = self.accelerator.get_tracker("wandb")
            return tracker.tracker.id  # type: ignore[no-any-return]
        except (ValueError, AttributeError):
            return None

    def _should_log(self) -> bool:
        return self.global_step % self.config.train.log_interval == 0

    def _should_evaluate(self) -> bool:
        return self.global_step % self.config.train.val_interval == 0

    def _should_checkpoint(self) -> bool:
        return self.global_step % self.config.train.checkpoint_interval == 0

    def _is_done(self) -> bool:
        return self.global_step >= self.config.train.total_iters

    def _apply_batch_mix(self, batch: StereoSRBatch) -> StereoSRBatch:
        if self.config.data.augment.batch_mix_mode == BatchMixMode.OFF:
            return batch
        alpha = float(
            torch.distributions.Beta(
                self.config.data.augment.batch_mix_alpha,
                self.config.data.augment.batch_mix_alpha,
            ).sample()
        )
        indices = torch.randperm(batch["lr"].shape[0], device=batch["lr"].device)
        return StereoSRBatch(
            lr=alpha * batch["lr"] + (1 - alpha) * batch["lr"][indices],
            gt=alpha * batch["gt"] + (1 - alpha) * batch["gt"][indices],
            disp_left=batch["disp_left"],
            disp_right=batch["disp_right"],
            conf_left=batch["conf_left"],
            conf_right=batch["conf_right"],
        )

    def _train_step(self, batch: StereoSRBatch) -> Tensor:
        batch = self._apply_batch_mix(batch)
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                sr = self.model(batch["lr"])
                loss = stereo_charbonnier_loss(sr, batch["gt"], self.config.loss.charbonnier_eps)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.train.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.accelerator.sync_gradients:
            self.scheduler.step()
        return loss

    def _evaluate(self) -> float:
        self._eval_model.eval()
        results: dict[str, tuple[float, float]] = {}
        with self.ema.average_parameters(), torch.no_grad():
            for dataset_name, loader in self.test_loaders.items():
                results[dataset_name] = self._evaluate_dataset(loader)
        self._eval_model.train()

        mean_psnr = 0.0
        mean_ssim = 0.0
        if results:
            mean_psnr = sum(psnr for psnr, _ in results.values()) / len(results)
            mean_ssim = sum(ssim for _, ssim in results.values()) / len(results)
        if self.accelerator.is_main_process:
            metrics = {
                f"val/psnr_{dataset_name}": psnr for dataset_name, (psnr, _) in results.items()
            }
            metrics.update(
                {f"val/ssim_{dataset_name}": ssim for dataset_name, (_, ssim) in results.items()}
            )
            metrics["val/psnr_mean"] = mean_psnr
            metrics["val/ssim_mean"] = mean_ssim
            self.accelerator.log(metrics, step=self.global_step)
            summary = " ".join(
                f"{dataset_name}: {psnr:.2f}/{ssim:.4f}"
                for dataset_name, (psnr, ssim) in results.items()
            )
            log.info(
                "step %d | val mean=%.2f/%.4f | %s",
                self.global_step,
                mean_psnr,
                mean_ssim,
                summary,
            )

        return mean_psnr

    def _evaluate_dataset(self, loader: DataLoader[StereoSRTestBatch]) -> tuple[float, float]:
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0

        for batch_index, batch in enumerate(loader):
            if batch_index % self.accelerator.num_processes != self.accelerator.process_index:
                continue
            lr = batch["lr"].to(self.accelerator.device)
            gt = batch["gt"].to(self.accelerator.device)
            with self.accelerator.autocast():
                sr = self._eval_model(lr)
            for i in range(sr.shape[0]):
                metrics = compute_stereo_metrics(sr[i], gt[i])
                psnr_sum += metrics["psnr_stereo"]
                ssim_sum += metrics["ssim_stereo"]
                count += 1
        return _finalize_eval_metrics(self.accelerator, psnr_sum, ssim_sum, count)

    def run(self) -> None:
        best_psnr = 0.0

        with self.console.training_progress(
            self.config.train.total_iters, self.global_step
        ) as progress:
            for batch in _batch_stream(self.train_loader):
                loss = self._train_step(batch)

                if self.accelerator.sync_gradients:
                    self.ema.update()
                    self.global_step += 1
                    progress.update(self.global_step, loss.item(), self.scheduler.get_last_lr()[0])

                    if self._should_log():
                        self.accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": self.scheduler.get_last_lr()[0],
                            },
                            step=self.global_step,
                        )

                    if self._should_evaluate():
                        val_psnr = self._evaluate()
                        best_psnr = max(best_psnr, val_psnr)
                        progress.set_val_metrics(val_psnr, best_psnr)
                        self.ckpt_mgr.save(
                            self.accelerator,
                            self.global_step,
                            metric=val_psnr,
                            wandb_run_id=self._get_wandb_run_id(),
                        )
                    elif self._should_checkpoint():
                        self.ckpt_mgr.save(
                            self.accelerator,
                            self.global_step,
                            wandb_run_id=self._get_wandb_run_id(),
                        )

                    if self._is_done():
                        break

        if self.config.train.wandb_project:
            self.accelerator.end_training()


def train(config: ExperimentConfig) -> None:
    Trainer(config).run()
