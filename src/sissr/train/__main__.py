from __future__ import annotations

import argparse
import logging
import warnings

import torch
from dotenv import load_dotenv

from sissr.configs.loading import load_config
from sissr.train.trainer import train


def main() -> None:
    load_dotenv()
    warnings.filterwarnings("ignore", message="Grad strides do not match")
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    train(config)


if __name__ == "__main__":
    main()
