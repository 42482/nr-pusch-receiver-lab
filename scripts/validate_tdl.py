#!/usr/bin/env python3
"""TDL diagnostics and full baseline sweep, with explicit profile acceptance."""

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from fm_receiver.runtime import prepare_cache, seed_all, resolve_device


def main():
    root = Path(__file__).resolve().parents[1]
    prepare_cache(root)
    import torch
    from fm_receiver.config import load_config
    from fm_receiver.e2e_model import E2EModel
    from fm_receiver.data.schema import ObservationBatch
    from fm_receiver.sionna_adapter import soft_detect, CodingAdapter
    from fm_receiver.training.engine import oracle_check
    from fm_receiver.evaluation.metrics import ErrorCounts
    from fm_receiver.evaluation.artifacts import write_json
    from fm_receiver.paths import data_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = data_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    seed_all(42, device)
    config = load_config(root / "config/b0.toml")
    config = replace(config, channel=replace(config.channel, kind="tdl-a"))
    model = E2EModel(config, device, 42)
    try:
        with torch.inference_mode():
            batch = model.generate(64, "tdl-a", 6.)
            clean = batch.observation.y - batch.noise
            obs = ObservationBatch(batch.observation.frame_ids, clean,
                                   torch.tensor(1e-6, device=device), batch.observation.layout_id)
            llr, _, _ = soft_detect(model.components, obs, batch.h_true)
            bits, crc = model.components.receiver.decoder(llr)
            torch.testing.assert_close(bits, batch.labels.tb_bits, rtol=0, atol=0)
            assert bool(crc.all())
            if config.channel.normalize:
                energy = batch.h_true.abs().square().mean((-2, -1))
                torch.testing.assert_close(energy, torch.ones_like(energy), rtol=1e-5, atol=1e-5)
            counts = ErrorCounts()
            counts.update(batch.labels.tb_bits, model.receiver(batch.observation), batch.noise)
            counts.validate_noise(10 ** (-.6))
            codec = CodingAdapter(model.components.adapter.encoder, model.components.receiver.decoder)
            checks = {"perfect_csi_noiseless": True, "normalization": True, "noise_power": True,
                      "channel": model.channel.diagnostics, "codec": oracle_check(model, codec)}
            write_json(output / "diagnostics.json", checks)
        code = subprocess.run([sys.executable, "-I", str(root / "scripts/evaluate.py"),
                               "--device", device, "--config", str(root / "config/b0_tdl.toml"),
                               "--output", str(output / "baseline")]).returncode
        if code:
            raise RuntimeError("TDL baseline sweep failed")
        write_json(output / "manifest.json", {"status": "completed", "profile_id": "b0-tdl-a",
                                               "verification_passed": True, "device": device,
                                               "gpu_accepted": device.startswith("cuda")})
    except BaseException as exc:
        write_json(output / "manifest.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
