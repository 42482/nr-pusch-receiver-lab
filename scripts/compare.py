#!/usr/bin/env python3
import argparse
from pathlib import Path
from fm_receiver.runtime import prepare_cache, resolve_device


def main():
    root = Path(__file__).resolve().parents[1]
    prepare_cache(root)
    from fm_receiver.paths import data_path
    from fm_receiver.config import load_config
    from fm_receiver.evaluation.comparison import compare
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "config/b0.toml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-tbs", type=int, default=256)
    parser.add_argument("--max-tbs", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 < args.min_tbs <= args.max_tbs:
        parser.error("Invalid comparison sample counts")
    compare(load_config(args.config), resolve_device(args.device), data_path(args.output), root,
            args.outer_checkpoint, args.joint_checkpoint, args.batch_size,
            args.min_tbs, args.max_tbs, args.resume)


if __name__ == "__main__":
    main()
