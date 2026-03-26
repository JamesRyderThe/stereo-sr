from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections import deque
from pathlib import Path
from typing import TypedDict

from accelerate import Accelerator

log = logging.getLogger(__name__)


class _CheckpointMeta(TypedDict):
    step: int
    metric: float | None
    wandb_run_id: str | None


def _upload_to_r2(local_dir: Path, prefix: str) -> None:
    endpoint = os.environ.get("R2_ACCOUNT_ID", "")
    access = os.environ.get("R2_ACCESS_KEY", "")
    secret = os.environ.get("R2_SECRET_KEY", "")
    if not all([endpoint, access, secret]):
        return

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{endpoint}.r2.cloudflarestorage.com",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    bucket = "super-res"

    paginator = client.get_paginator("list_objects_v2")
    to_delete: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            to_delete.append({"Key": obj["Key"]})
    if to_delete:
        for i in range(0, len(to_delete), 1000):
            client.delete_objects(Bucket=bucket, Delete={"Objects": to_delete[i : i + 1000]})

    for file_path in sorted(local_dir.rglob("*")):
        if file_path.is_file():
            key = f"{prefix}/{file_path.relative_to(local_dir)}"
            client.upload_file(str(file_path), bucket, key)

    client.put_object(Bucket=bucket, Key=f"{prefix}/_COMPLETE", Body=b"")
    log.info("Uploaded checkpoint to r2://%s/%s", bucket, prefix)


class CheckpointManager:
    def __init__(self, base_dir: str, max_checkpoints: int = 3) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._best_metric = float("-inf")
        self._periodic: deque[Path] = deque()
        self._upload_thread: threading.Thread | None = None

    def save(
        self,
        accelerator: Accelerator,
        step: int,
        metric: float | None = None,
        wandb_run_id: str | None = None,
    ) -> None:
        accelerator.wait_for_everyone()

        step_name = f"step_{step:08d}"
        final_dir = self._base_dir / step_name
        tmp_dir = self._base_dir / f".tmp_{step_name}"

        accelerator.save_state(str(tmp_dir))

        if accelerator.is_main_process:
            meta: _CheckpointMeta = {"step": step, "metric": metric, "wandb_run_id": wandb_run_id}
            (tmp_dir / "metadata.json").write_text(json.dumps(meta))
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.rename(str(tmp_dir), str(final_dir))

            self._update_symlink("latest", final_dir)
            self._upload_async(final_dir, "checkpoints/latest")

            if metric is not None and metric > self._best_metric:
                self._best_metric = metric
                self._update_symlink("best", final_dir)
                self._upload_async(final_dir, "checkpoints/best")
                log.info("New best metric: %.4f at step %d", metric, step)

            self._periodic.append(final_dir)
            while len(self._periodic) > self._max_checkpoints:
                old = self._periodic.popleft()
                if old.exists() and not self._is_symlink_target(old):
                    shutil.rmtree(old)

        accelerator.wait_for_everyone()

    def _upload_async(self, local_dir: Path, prefix: str) -> None:
        if self._upload_thread is not None and self._upload_thread.is_alive():
            self._upload_thread.join(timeout=300)
        self._upload_thread = threading.Thread(
            target=_upload_to_r2, args=(local_dir, prefix), daemon=True
        )
        self._upload_thread.start()

    def find_latest(self) -> Path | None:
        link = self._base_dir / "latest"
        if not link.is_symlink():
            return None
        resolved = link.resolve()
        return resolved if resolved.exists() else None

    def resolve(self, name: str) -> Path:
        return self._base_dir / name

    def load_step(self, path: Path) -> int:
        with open(path / "metadata.json") as f:
            meta: _CheckpointMeta = json.load(f)
        step = meta["step"]
        if not isinstance(step, int):
            raise ValueError(f"Expected int step, got {type(step).__name__}")
        return step

    def load_wandb_run_id(self, path: Path) -> str | None:
        with open(path / "metadata.json") as f:
            meta: _CheckpointMeta = json.load(f)
        return meta.get("wandb_run_id")

    def _update_symlink(self, name: str, target: Path) -> None:
        link_path = self._base_dir / name
        tmp_link = self._base_dir / f".tmp_{name}_{os.getpid()}"
        os.symlink(target, tmp_link)
        os.replace(tmp_link, link_path)

    def _is_symlink_target(self, path: Path) -> bool:
        for name in ("latest", "best"):
            link = self._base_dir / name
            if link.is_symlink() and link.resolve() == path.resolve():
                return True
        return False
