"""Paired B0/B1/B2 evaluation; every receiver sees exactly the same frames."""

from pathlib import Path
import hashlib
import json
import time
import torch

from .artifacts import write_json
from .metrics import ErrorCounts
from ..e2e_model import E2EModel
from ..runtime import seed_all, environment_snapshot
from ..models.outer import OuterTransformer
from ..models.gecct import GroupedECCT
from ..receivers.neural import NeuralReceiver
from ..sionna_adapter import CodingAdapter
from ..training.checkpoint import load_checkpoint, source_hashes


def read_networks(path, model, codec, device, root=None):
    value = load_checkpoint(path, device, model.components.adapter.layout_id, codec.layout_id)
    if root is not None and value["source_hashes"] != source_hashes(root):
        raise ValueError("Checkpoint source differs from current source")
    outer, inner = None, None
    if value["outer"] is not None:
        outer = OuterTransformer(model.components.adapter).to(device)
        outer.load_state_dict(value["outer"])
        outer.eval()
    if value["inner"] is not None:
        inner = GroupedECCT(codec).to(device)
        inner.load_state_dict(value["inner"])
        inner.eval()
    return outer, inner


@torch.inference_mode()
def compare(config, device, output, root, outer_path, joint_path, batch_size=16,
            min_tbs=256, max_tbs=4096, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    identity = {"source_hashes": source_hashes(root), "config": config.to_dict(),
                "checkpoints": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                                for k, v in {"outer": outer_path, "joint": joint_path}.items()},
                "device": device, "batch_size": batch_size, "min_tbs": min_tbs,
                "max_tbs": max_tbs}
    if resume and (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Comparison restart requires identical inputs/source/checkpoints")
    manifest = {"status": "running", "kind": "paired_comparison", "identity": identity,
                "restarted_from_beginning": resume}
    write_json(output / "manifest.json", manifest)
    write_json(output / "environment.json", environment_snapshot(device))
    seed_all(70000000, device)
    model = E2EModel(config, device, 70000000)
    codec = CodingAdapter(model.components.adapter.encoder, model.components.receiver.decoder)
    outer, _ = read_networks(outer_path, model, codec, device, root)
    joint_outer, inner = read_networks(joint_path, model, codec, device, root)
    if outer is None or joint_outer is None or inner is None:
        raise ValueError("B1 outer and B2 joint checkpoints are required")
    receivers = {"b0": model.receiver, "b1": NeuralReceiver(model.components, outer),
                 "b2": NeuralReceiver(model.components, joint_outer, codec, inner)}
    rows = []
    try:
        for channel_index, channel in enumerate(("awgn", "tdl-a")):
            for index, snr in enumerate(range(-4, 13, 2)):
                seed = 70000000 + channel_index * 10000 + index
                seed_all(seed, device)
                model.payload_rng.manual_seed(seed + 1)
                model.next_frame_id = 0
                counts = {name: ErrorCounts() for name in receivers}
                elapsed = dict.fromkeys(receivers, 0.)
                digest = hashlib.sha256()
                total = 0
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                while total < max_tbs:
                    size = min(batch_size, max_tbs - total)
                    batch = model.generate(size, channel, float(snr))
                    digest.update(batch.observation.y.cpu().numpy().tobytes())
                    digest.update(batch.labels.tb_bits.cpu().numpy().tobytes())
                    for name, receiver in receivers.items():
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        started = time.monotonic()
                        result = receiver(batch.observation)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        elapsed[name] += time.monotonic() - started
                        counts[name].update(batch.labels.tb_bits, result, batch.noise)
                    total += size
                    if total >= min_tbs and all(v.block_errors >= 100 for v in counts.values()):
                        break
                for name, counter in counts.items():
                    rows.append({"receiver": name, "channel": channel, "snr_db": snr,
                                 "input_sha256": digest.hexdigest(), "seed": seed,
                                 "seconds": elapsed[name],
                                 "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                                 if device.startswith("cuda") else None, **counter.to_dict()})
                write_json(output / "metrics.json", rows)
                print(f"Paired {channel} {snr} dB, {total} TB: " + ", ".join(
                    f"{k} BLER={v.block_errors/v.total_blocks:.5g}" for k, v in counts.items()), flush=True)
        manifest.update(status="completed", paired_observations=True,
                        performance_claim="Measured counts only; improvement is not assumed")
        import csv
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, channel in zip(axes, ("awgn", "tdl-a")):
            for name in receivers:
                points = [r for r in rows if r["channel"] == channel and r["receiver"] == name]
                ax.semilogy([p["snr_db"] for p in points],
                            [max(p["bler"], 1/p["total_blocks"]) for p in points], "o-", label=name)
                zero = [p for p in points if p["block_errors"] == 0]
                ax.scatter([p["snr_db"] for p in zero], [1/p["total_blocks"] for p in zero],
                           marker="v", label=f"{name}: zero observed at 1/N" if zero else None)
            ax.set(title=channel, xlabel="Data-symbol Es/N0 (dB)", ylabel="TB BLER")
            ax.grid(True, which="both", alpha=.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "bler.png", dpi=180)
        plt.close(fig)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest
