"""Thin experiment entrypoints; all PHY and evaluation logic lives in modules."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

from .config import load_config, ExperimentConfig
from .runtime import prepare_cache, resolve_device, environment_snapshot, seed_all

ROOT = Path(__file__).resolve().parents[2]


def _parser(kind):
    parser = argparse.ArgumentParser(description="Reproducible first-round NR PUSCH baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "b0.toml")
    parser.add_argument("--device", help="cpu, cuda:N or auto; defaults to config (cuda:0)")
    parser.add_argument("--seed", type=int, help="Override the evaluation payload/noise seed")
    parser.add_argument("--output", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--channel", choices=("awgn", "tdl-a"))
    if kind == "evaluate":
        parser.add_argument("--scenario", choices=("all", "hello", "passthrough", "awgn", "ofdm-loopback"),
                            default="all")
        parser.add_argument("--max-tbs", type=int)
        parser.add_argument("--min-tbs", type=int)
        parser.add_argument("--passthrough-tbs", type=int)
        parser.add_argument("--compare-run", type=Path, help="Require identical inputs/counts to a prior run")
    else:
        parser.add_argument("--num-frames", type=int)
        parser.add_argument("--snr-db", type=float)
    return parser


def _overrides(config, args, kind):
    runtime = replace(config.runtime, device=args.device or config.runtime.device,
                      seed=config.runtime.seed if args.seed is None else args.seed)
    evaluation = config.evaluation
    if args.batch_size is not None:
        evaluation = replace(evaluation, batch_size=args.batch_size)
    samples = config.samples
    if kind == "evaluate":
        for name in ("max_tbs", "min_tbs", "passthrough_tbs"):
            value = getattr(args, name)
            if value is not None:
                evaluation = replace(evaluation, **{name: value})
    else:
        samples = replace(samples, seed=samples.seed if args.seed is None else args.seed,
                          num_frames=samples.num_frames if args.num_frames is None else args.num_frames,
                          snr_db=samples.snr_db if args.snr_db is None else args.snr_db)
    channel = replace(config.channel, kind=args.channel or config.channel.kind)
    return replace(config, runtime=runtime, evaluation=evaluation, samples=samples,
                   channel=channel).validate()


def _compare_run(run, previous):
    previous = previous.resolve()
    if json.loads((previous / "requested_config.json").read_text()) != json.loads(
            (run.path / "requested_config.json").read_text()):
        raise ValueError("Comparison runs must use identical effective requested configurations")
    old_env = json.loads((previous / "environment.json").read_text())
    new_env = json.loads((run.path / "environment.json").read_text())
    if old_env["device"] != new_env["device"]:
        raise ValueError("Exact input/count replay requires the same execution device")
    if json.loads((previous / "source_hashes.json").read_text()) != json.loads(
            (run.path / "source_hashes.json").read_text()):
        raise ValueError("Comparison runs must use identical source code")
    old = json.loads((previous / "metrics.json").read_text())
    new = json.loads((run.path / "metrics.json").read_text())
    keys = ("scenario", "snr_db", "seed", "bit_errors", "total_bits", "block_errors",
            "total_blocks", "crc_failures", "input_sha256")
    def select(rows):
        return [{k: row.get(k) for k in keys} for row in rows]
    if select(old) != select(new):
        raise AssertionError("Reproduced input hashes or error counts differ")
    run.log(f"Reproducibility comparison passed: {previous}")


def _run(kind, argv=None):
    args = _parser(kind).parse_args(argv)
    prepare_cache(ROOT)
    run = None
    try:
        config_path = args.config.resolve()
        config = _overrides(load_config(config_path), args, kind)
        if args.output:
            from .paths import data_path
            data_path(args.output)
        from .evaluation.artifacts import RunArtifacts, write_json
        run = RunArtifacts(ROOT, args.output, config_path, config, vars(args), kind)
        run.log(f"Run directory: {run.path}")
        device = resolve_device(config.runtime.device)
        run.save_environment(environment_snapshot(device))
        seed_all(config.runtime.seed, device, config.runtime.cpu_threads)
        import torch
        with torch.inference_mode():
            if kind == "evaluate":
                from .evaluation.runner import evaluate
                summary = evaluate(config, device, run, args.scenario)
                if args.scenario == "all":
                    from .data.export import export_samples
                    summary["checks"]["sample_export"] = export_samples(
                        config, device, run.path / "samples", run.log)
                if args.compare_run:
                    _compare_run(run, args.compare_run)
                    summary["checks"]["reproducibility_comparison"] = True
                write_json(run.path / "checks.json", summary["checks"])
                summary.pop("rows", None)
            else:
                from .data.export import export_samples
                summary = {"checks": export_samples(config, device, run.path / "samples", run.log)}
        baseline = ExperimentConfig()
        full_profile = (config.channel == baseline.channel and
                        config.system == baseline.system and config.dmrs == baseline.dmrs
                        and config.receiver == baseline.receiver and config.evaluation == baseline.evaluation
                        and config.samples == baseline.samples and config.runtime.seed == baseline.runtime.seed)
        accepted = (kind == "evaluate" and args.scenario == "all" and device.startswith("cuda")
                    and full_profile and run.manifest["dependency_check_passed"])
        run.finish("completed", device=device, formal_profile=full_profile,
                   profile_id="b0-" + config.channel.kind,
                   executed_on_gpu=device.startswith("cuda"),
                   gpu_validation_passed=accepted, milestone_accepted=accepted,
                   **summary)
        run.log(f"Completed on {device}; full GPU milestone accepted={accepted}")
        return 0
    except Exception as exc:
        if run is not None:
            run.log(traceback.format_exc())
            run.finish("failed", error=str(exc), milestone_accepted=False,
                       environment=environment_snapshot())
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def evaluate_main(argv=None):
    return _run("evaluate", argv)


def export_main(argv=None):
    return _run("export", argv)
