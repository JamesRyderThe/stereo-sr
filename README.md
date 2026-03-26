# stereo-scope

Stereo image super-resolution with geometry-aware cross-view transfer.

The idea: stereo SR models shouldn't learn cross-view correspondence from scratch on tiny datasets. A frozen stereo foundation model already knows where things are — use that to decide where cross-view borrowing is safe.

## What's here

Custom stereo SR backbone. Parallel L/R paths, windowed self-attention, rectangular cross-attention aligned to epipolar geometry, geometry-gated transfer via predicted disparity + confidence. Depth aggregation residual strategy based on [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi team). Epipolar RoPE. `torch.compile`, DDP, bf16. 180+ tests, `mypy --strict`, Pydantic configs.

Active research — architecture and training are evolving.

## Results

| Method | Scale | Flickr1024 | KITTI 2012 | KITTI 2015 | Middlebury |
|--------|-------|-----------|-----------|-----------|-----------|
| NAFSSR-L | x4 | 24.17 | 27.12 | 26.96 | 30.30 |
| DIFFSSR | x4 | 24.47 | 27.26 | 26.98 | 30.65 |
| **Ours** | x4 | — | — | — | — |

Evaluation matches iPASSR protocol exactly (RGB, 7x7 uniform window SSIM, no border crop).

## Usage

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

Credentials in `.env`:

```
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
WANDB_API_KEY=...
```

Training:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m sissr.train --config configs/stereo_sr_baseline.yaml
```

Development:

```bash
make full     # format + lint + test
```

## License

Research use only.
