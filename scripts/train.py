#!/usr/bin/env python3
"""Thin training CLI; implementation lives in fm_receiver.training."""

import argparse
from dataclasses import replace
from pathlib import Path

from fm_receiver.runtime import prepare_cache, resolve_device


def main():
    root = Path(__file__).resolve().parents[1]
    prepare_cache(root)
    from fm_receiver.config import load_config
    from fm_receiver.training.engine import TrainOptions, train
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("outer", "decoder", "joint"), required=True)
    parser.add_argument("--config", type=Path, default=root / "config/b0.toml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--outer-checkpoint", type=Path)
    parser.add_argument("--inner-checkpoint", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--validation-frames", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=8.)
    args = parser.parse_args()
    from fm_receiver.paths import data_path
    data_path(args.output)
    config = load_config(args.config)
    device = resolve_device(args.device)
    config = replace(config, runtime=replace(config.runtime, device=device))
    options = TrainOptions(args.stage, args.seed,
                           args.steps or (1000 if args.overfit else 5000 if args.stage == "joint" else 20000),
                           args.batch_size, args.microbatch, args.eval_interval,
                           args.validation_frames, args.patience, args.max_hours, args.overfit,
                           1e-4 if args.stage == "joint" else 1e-3)
    result = train(config, device, options, args.output, root, args.resume,
                   args.outer_checkpoint, args.inner_checkpoint)
    return 0 if result["status"] in ("completed", "early_stopped") else 2


if __name__ == "__main__":
    raise SystemExit(main())
