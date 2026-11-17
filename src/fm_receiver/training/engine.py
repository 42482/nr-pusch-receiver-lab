"""B1, grouped decoder and joint training on disjoint deterministic data streams."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import random
import signal
import time
import torch
from torch import nn
from torch.nn import functional as F

from ..e2e_model import E2EModel
from ..models.outer import OuterTransformer
from ..models.gecct import GroupedECCT
from ..runtime import seed_all, environment_snapshot
from ..sionna_adapter import CodingAdapter, soft_detect
from ..evaluation.artifacts import write_json
from ..evaluation.metrics import ErrorCounts
from .checkpoint import (atomic_save, load_checkpoint, random_state, restore_random,
                         source_hashes)


@dataclass(frozen=True)
class TrainOptions:
    stage: str = "outer"
    seed: int = 42
    steps: int = 20000
    batch_size: int = 16
    microbatch: int = 4
    eval_interval: int = 250
    validation_frames: int = 4096
    patience: int = 8
    max_hours: float = 8.
    overfit: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def validate(self):
        if self.stage not in ("outer", "decoder", "joint"):
            raise ValueError("stage must be outer, decoder or joint")
        if min(self.steps, self.batch_size, self.microbatch, self.eval_interval,
               self.validation_frames, self.patience) <= 0 or self.seed < 0:
            raise ValueError("Training counts must be positive and seed nonnegative")
        if self.batch_size % self.microbatch:
            raise ValueError("microbatch must divide effective batch size")
        if self.overfit and 64 % self.batch_size:
            raise ValueError("Overfit effective batch size must divide the fixed 64-frame pool")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.max_hours, self.learning_rate)) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid training budget/optimizer settings")
        return self


class Networks(nn.Module):
    def __init__(self, components, codec, stage):
        super().__init__()
        self.outer = OuterTransformer(components.adapter) if stage != "decoder" else None
        self.inner = GroupedECCT(codec) if stage != "outer" else None


def build_training(config, device, options):
    seed_all(options.seed, device, config.runtime.cpu_threads)
    model = E2EModel(config, device, options.seed)
    codec = CodingAdapter(model.components.adapter.encoder, model.components.receiver.decoder)
    networks = Networks(model.components, codec, options.stage).to(device)
    return model, codec, networks


def sampled_batch(model, options, step, offset, size, validation=False):
    # Seeds depend on data coordinates, not mutable process history/checkpoint time.
    from ..data.schema import ObservationBatch, SimulatedBatch, TxLabels
    index = offset if validation else step * options.batch_size + offset
    if options.overfit:
        index %= 64
    group, within = divmod(index, options.batch_size)
    if within + size > options.batch_size:
        raise ValueError("A microbatch cannot cross an effective-batch data boundary")
    key = (validation, options.overfit, group, options.stage, options.seed, options.batch_size)
    if not hasattr(model, "_sample_cache"):
        model._sample_cache = {}
    if key not in model._sample_cache:
        if not validation and not options.overfit:
            model._sample_cache = {k: v for k, v in model._sample_cache.items() if k[0] or k[1]}
        model._sample_cache[key] = _generate_group(model, options, group, validation)
    batch = model._sample_cache[key]
    part = slice(within, within + size)
    obs = batch.observation
    return SimulatedBatch(ObservationBatch(obs.frame_ids[part], obs.y[part], obs.no, obs.layout_id),
                          TxLabels(batch.labels.tb_bits[part], batch.labels.mapper_bits[part]),
                          batch.tx_grid[part], batch.noise[part],
                          batch.h_true[part] if batch.h_true is not None else None)


def _generate_group(model, options, group, validation):
    if options.overfit:
        data_seed = 100000 + group * options.batch_size
        scenario, snr = "awgn", 2.
    else:
        base = 50000000 if validation else 1000000 + options.seed * 100000
        data_seed = base + group * options.batch_size
        rng = random.Random(data_seed)
        scenario = "awgn" if options.stage == "decoder" or rng.random() < .5 else "tdl-a"
        snr = rng.uniform(-4., 12.)
    seed_all(data_seed, model.device, model.config.runtime.cpu_threads)
    model.payload_rng.manual_seed(data_seed + 1)
    model.next_frame_id = group * options.batch_size
    with torch.no_grad():
        return model.generate(options.batch_size, scenario, snr)


def optimizer_step(model, codec, networks, optimizer, options, step, microbatch):
    """Retry only forward/backward OOMs; effective data/batch stays unchanged."""
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            for offset in range(0, options.batch_size, microbatch):
                batch = sampled_batch(model, options, step, offset, microbatch)
                loss, _, _, _ = loss_and_output(model, codec, networks, batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                (loss * microbatch / options.batch_size).backward()
            break
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if microbatch == 1:
                raise
            microbatch //= 2
            while options.batch_size % microbatch:
                microbatch -= 1
            torch.cuda.empty_cache()
            print(f"CUDA OOM: retrying unchanged effective batch with microbatch={microbatch}", flush=True)
    for name in ("outer", "inner"):
        module = getattr(networks, name)
        if module is not None:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or not any(
                    bool((g != 0).any()) for g in grads):
                raise FloatingPointError(f"Missing/nonfinite gradient in {name}")
    nn.utils.clip_grad_norm_(networks.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return microbatch


def loss_and_output(model, codec, networks, batch, decode=False):
    obs = batch.observation
    with torch.no_grad():
        if networks.outer is None:
            llr, h, _ = soft_detect(model.components, obs)
        else:
            h, _ = model.components.receiver.channel_estimator(obs.y, obs.no)
    if networks.outer is not None:
        llr = networks.outer(obs, h)
    bit_loss = F.binary_cross_entropy_with_logits(llr, batch.labels.mapper_bits)
    loss = bit_loss if networks.outer is not None else llr.new_zeros(())
    output = None
    if networks.inner is not None:
        code_llr = codec.to_code_llr(llr)
        target_code = codec.code_labels(batch.labels.tb_bits)
        target_error = ((code_llr > 0).float() != target_code).float()
        decoded, error_logits = networks.inner(code_llr, codec)
        error_loss = F.binary_cross_entropy_with_logits(
            error_logits[:, ~codec.known_zero], target_error[:, ~codec.known_zero])
        loss = loss + error_loss
        if decode:
            output = codec.to_tb(decoded)
    elif decode:
        output = model.components.receiver.decoder(llr)
    mapper_errors = int(((llr > 0) != batch.labels.mapper_bits.bool()).sum())
    return loss, bit_loss, mapper_errors, output


@torch.no_grad()
def validate(model, codec, networks, options):
    from ..data.schema import ReceiverResult
    networks.eval()
    counts = ErrorCounts()
    total_loss = total_bce = errors = bits = 0
    total = 64 if options.overfit else options.validation_frames
    for offset in range(0, total, options.microbatch):
        size = min(options.microbatch, total - offset)
        batch = sampled_batch(model, options, 0, offset, size, validation=True)
        loss, bce, error, output = loss_and_output(model, codec, networks, batch, decode=True)
        counts.update(batch.labels.tb_bits, ReceiverResult(*output))
        total_loss += float(loss) * size
        total_bce += float(bce) * size
        errors += error
        bits += batch.labels.mapper_bits.numel()
    networks.train()
    return {"loss": total_loss / total, "mapper_bce": total_bce / total,
            "mapper_ber": errors / bits, **counts.to_dict()}


def oracle_check(model, codec):
    """Verify parity, scrambling, puncture handling and TB/CRC recovery."""
    with torch.no_grad():
        batch = model.generate(4, "passthrough")
        code = codec.code_labels(batch.labels.tb_bits)
        if not bool((codec.syndrome(code) == 0).all()):
            raise AssertionError("Legal codewords fail parity checks")
        llr = (2 * batch.labels.mapper_bits - 1) * 20
        observed = codec.to_code_llr(llr)
        torch.testing.assert_close((observed[:, :codec.sent] > 0).float(),
                                   code[:, :codec.sent], rtol=0, atol=0)
        bits, crc = codec.to_tb((2 * code - 1) * 20)
        torch.testing.assert_close(bits, batch.labels.tb_bits, rtol=0, atol=0)
        if not bool(crc.all()):
            raise AssertionError("Oracle NR recovery CRC failure")
    return {"parity_zero": True, "mapper_correspondence": True,
            "oracle_tb_crc": True, **codec.metadata()}


def train(config, device, options, output, root, resume=False, outer_checkpoint=None,
          inner_checkpoint=None):
    options.validate()
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Training output exists; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    model, codec, networks = build_training(config, device, options)
    hashes = source_hashes(root)
    write_json(output / "codec_checks.json", oracle_check(model, codec))
    write_json(output / "environment.json", environment_snapshot(device))
    for path, key in ((outer_checkpoint, "outer"), (inner_checkpoint, "inner")):
        if path:
            checkpoint = load_checkpoint(path, device, model.components.adapter.layout_id,
                                         codec.layout_id)
            if checkpoint["source_hashes"] != hashes:
                raise ValueError("Pretrained checkpoint source differs; use a consistent frozen release")
            module = getattr(networks, key)
            if module is None or checkpoint.get(key) is None:
                raise ValueError("Checkpoint does not contain requested network")
            module.load_state_dict(checkpoint[key])
    if options.stage == "joint" and not resume and not (outer_checkpoint and inner_checkpoint):
        raise ValueError("Joint training requires both pretrained checkpoints")
    optimizer = torch.optim.AdamW(networks.parameters(), lr=options.learning_rate,
                                  weight_decay=options.weight_decay)
    start_step, best_score, stale, previous_elapsed = 0, None, 0, 0.
    microbatch = options.microbatch
    signature = {"stage": options.stage, "seed": options.seed, "overfit": options.overfit,
                 "batch_size": options.batch_size,
                 "validation_frames": options.validation_frames,
                 "eval_interval": options.eval_interval,
                 "learning_rate": options.learning_rate, "weight_decay": options.weight_decay,
                 "config": config.to_dict()}
    if resume:
        checkpoint = load_checkpoint(output / "last.pt", device,
                                     model.components.adapter.layout_id, codec.layout_id)
        if checkpoint["signature"] != signature or checkpoint["source_hashes"] != hashes:
            raise ValueError("Resume requires identical data/optimizer configuration and source")
        for key in ("outer", "inner"):
            if getattr(networks, key) is not None:
                getattr(networks, key).load_state_dict(checkpoint[key])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_random(checkpoint["random_state"])
        start_step, best_score, stale = checkpoint["step"], checkpoint["best_score"], checkpoint["stale"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        microbatch = min(options.microbatch, checkpoint.get("microbatch", options.microbatch))
    status = {"status": "running", "stage": options.stage, "seed": options.seed,
              "options": asdict(options), "config": config.to_dict(), "source_hashes": hashes,
              "layout_id": model.components.adapter.layout_id, "codec": codec.metadata(),
              "device": device, "formal_training": options.steps >= (5000 if options.stage == "joint" else 20000)
              and options.validation_frames == 4096 and not options.overfit}
    write_json(output / "manifest.json", status)
    stopped = []
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stopped.append(signum))
    started = time.monotonic()
    metrics = None
    step = start_step

    def save():
        value = {"format_version": 1, "stage": options.stage, "signature": signature,
                 "layout_id": model.components.adapter.layout_id, "codec_layout_id": codec.layout_id,
                 "source_hashes": hashes, "step": step, "best_score": best_score, "stale": stale,
                 "microbatch": microbatch,
                 "optimizer": optimizer.state_dict(), "random_state": random_state(),
                 "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                 "outer": networks.outer.state_dict() if networks.outer is not None else None,
                 "inner": networks.inner.state_dict() if networks.inner is not None else None}
        atomic_save(value, output / "last.pt")

    try:
        while step < options.steps:
            if stopped or previous_elapsed + time.monotonic() - started >= options.max_hours * 3600:
                status["status"] = "interrupted" if stopped else "budget_exhausted"
                break
            networks.train()
            microbatch = optimizer_step(model, codec, networks, optimizer, options, step, microbatch)
            step += 1
            if step % options.eval_interval == 0 or step == options.steps:
                while True:
                    try:
                        metrics = validate(model, codec, networks, replace(options, microbatch=microbatch))
                        break
                    except torch.cuda.OutOfMemoryError:
                        if microbatch == 1:
                            raise
                        microbatch //= 2
                        while options.batch_size % microbatch:
                            microbatch -= 1
                        torch.cuda.empty_cache()
                score = ((metrics["mapper_bce"],) if options.stage == "outer" else
                         (metrics["bler"], metrics["loss"]))
                improved = best_score is None or score < tuple(best_score)
                if improved:
                    best_score, stale = score, 0
                else:
                    stale += 1
                save()
                if improved:
                    atomic_save(torch.load(output / "last.pt", map_location="cpu", weights_only=False),
                                output / "best.pt")
                row = {"step": step, "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                       **metrics}
                with (output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"{options.stage} seed={options.seed} step={step}: "
                      f"loss={metrics['loss']:.6g}, mapper BER={metrics['mapper_ber']:.6g}, "
                      f"TB BLER={metrics['bler']:.6g}", flush=True)
                status.update(step=step, validation=metrics, microbatch=microbatch)
                write_json(output / "manifest.json", status)
                if options.overfit and metrics["block_errors"] == 0 and (
                        options.stage == "decoder" or metrics["mapper_ber"] <= 1e-3):
                    status["status"] = "completed"
                    break
                if stale >= options.patience and not options.overfit:
                    status["status"] = "early_stopped"
                    break
        else:
            status["status"] = "completed"
        save()
        if options.overfit:
            passed = metrics is not None and metrics["block_errors"] == 0 and (
                options.stage == "decoder" or metrics["mapper_ber"] <= 1e-3)
            status["overfit_passed"] = passed
            if not passed and status["status"] == "completed":
                status["status"] = "overfit_failed"
        status.update(step=step, elapsed_seconds=previous_elapsed + time.monotonic() - started)
        write_json(output / "manifest.json", status)
        return status
    except BaseException as exc:
        save()
        status.update(status="failed", step=step, error=str(exc))
        write_json(output / "manifest.json", status)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
