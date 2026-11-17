#!/usr/bin/env python3
"""Synthetic IQ contract demo; this script never opens RF hardware."""

import argparse
from pathlib import Path
import numpy as np
import torch
from fm_receiver.runtime import prepare_cache, seed_all


def main():
    root = Path(__file__).resolve().parents[1]
    prepare_cache(root)
    from fm_receiver.config import load_config
    from fm_receiver.e2e_model import E2EModel
    from fm_receiver.hardware.capture import export_waveform, write_capture, load_observations
    from fm_receiver.evaluation.artifacts import write_json
    from fm_receiver.paths import data_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = data_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(44, "cpu")
    model = E2EModel(load_config(root / "config/b0.toml"), "cpu", 44)
    batch = model.generate(8, "passthrough")
    iq, spec = export_waveform(model, batch, out / "transmit")
    padding = 19
    cfo = 321.
    padded = np.pad(iq, ((padding, 0), (0, 0)))
    padded *= np.exp(2j*np.pi*cfo*np.arange(len(padded))/spec.logical_rate_hz)[:, None]
    ids = batch.observation.frame_ids.numpy()
    starts = padding + np.arange(len(ids)) * spec.frame_samples
    metadata = {"actual_rx_rate_hz": spec.logical_rate_hz, "noise_variance_grid": 1e-6,
                "frequency_offset_hz": cfo, "sync_source": "synthetic", "synthetic": True,
                "noise_source": "synthetic receiver numerical floor"}
    write_capture(out / "capture.h5", padded, spec, metadata, ids, starts,
                  events=[{"kind": "gap", "start_sample": int(starts[3]), "num_samples": 10}])
    observation, quality = load_observations(out / "capture.h5", spec)
    expected = batch.tx_grid[observation.frame_ids]
    torch.testing.assert_close(observation.y, expected, rtol=1e-5, atol=1e-5)
    result = model.receiver(observation)
    torch.testing.assert_close(result.tb_hat, batch.labels.tb_bits[observation.frame_ids], rtol=0, atol=0)
    assert bool(result.tb_crc.all())
    write_json(out / "checks.json", {"status": "passed", "accepted_frames": 7,
                                    "grid_match": True, "tb_crc": True, "quality": quality,
                                    "hardware_tested": False})
    print(f"Offline IQ/offset/CFO/gap/FFT/CRC checks passed: {out}")


if __name__ == "__main__":
    main()
