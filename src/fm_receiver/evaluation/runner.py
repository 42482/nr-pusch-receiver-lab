"""One evaluation path for B0 and later receivers using the same observations."""

import csv
import hashlib
import torch
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.channel import AWGN
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator

from .artifacts import plot_awgn, write_json
from .metrics import ErrorCounts
from ..channels import esn0_to_no
from ..data.schema import ObservationBatch, ReceiverResult
from ..e2e_model import E2EModel
from ..runtime import seed_all


def hello_world(config, device):
    seed_all(config.runtime.seed, device, config.runtime.cpu_threads)
    pc = PUSCHConfig()
    tx = PUSCHTransmitter(pc, device=device)
    rx = PUSCHReceiver(tx, return_tb_crc_status=True, device=device)
    x, bits = tx(16)
    y = AWGN(device=device)(x, .1)
    decoded, crc = rx(y, .1)
    counts = ErrorCounts()
    counts.update(bits, ReceiverResult(decoded, crc), y - x)
    return {"scenario": "hello", "noise_variance": .1, "mcs_index": int(pc.tb.mcs_index),
            "tx_shape": list(x.shape), "tb_shape": list(bits.shape), **counts.to_dict()}


def compare_default(model, batch, result):
    decoded, crc = model.components.default_receiver(batch.observation.y, batch.observation.no)
    torch.testing.assert_close(decoded, result.tb_hat, rtol=0, atol=0)
    torch.testing.assert_close(crc, result.tb_crc, rtol=0, atol=0)
    return True


def ofdm_loopback(model, batch):
    rg = model.components.adapter.resource_grid
    mod = OFDMModulator(rg.cyclic_prefix_length, precision="single", device=model.device)
    demod = OFDMDemodulator(rg.fft_size, l_min=0, cyclic_prefix_length=rg.cyclic_prefix_length,
                            precision="single", device=model.device)
    iq = mod(batch.tx_grid)
    grid = demod(iq)
    torch.testing.assert_close(grid, batch.tx_grid, rtol=1e-5, atol=1e-6)
    observation = ObservationBatch(batch.observation.frame_ids, grid, batch.observation.no,
                                   batch.observation.layout_id)
    decoded = model.receiver(observation)
    torch.testing.assert_close(decoded.tb_hat, batch.labels.tb_bits, rtol=0, atol=0)
    if not bool(decoded.tb_crc.all()):
        raise AssertionError("OFDM loopback CRC failure")
    return {"passed": True, "iq_shape": list(iq.shape),
            "max_grid_difference": float((grid - batch.tx_grid).abs().max())}


def check_reproducibility(config, device):
    """Regenerate the same noisy inputs, then compare the decoded results."""
    def once():
        seed_all(config.runtime.seed, device, config.runtime.cpu_threads)
        model = E2EModel(config, device, config.runtime.seed)
        batch = model.generate(3, config.channel.kind, 2.)
        result = model.receiver(batch.observation)
        return (batch.labels.tb_bits, batch.observation.y, result.tb_hat, result.tb_crc)
    first, second = once(), once()
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    return True


def check_cpu_gpu(config, device, batch, gpu_result):
    if not device.startswith("cuda"):
        return {"status": "pending", "reason": "This run explicitly selected CPU"}
    seed_all(config.runtime.seed, "cpu", config.runtime.cpu_threads)
    cpu_model = E2EModel(config, "cpu", config.runtime.seed)
    observation = ObservationBatch(batch.observation.frame_ids, batch.observation.y.cpu(),
                                   batch.observation.no.cpu(), batch.observation.layout_id)
    cpu_result = cpu_model.receiver(observation)
    torch.testing.assert_close(cpu_result.tb_hat, gpu_result.tb_hat.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(cpu_result.tb_crc, gpu_result.tb_crc.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(cpu_result.h_hat, gpu_result.h_hat.cpu(), rtol=1e-5, atol=1e-6)
    seed_all(config.runtime.seed, device, config.runtime.cpu_threads)
    return {"status": "passed", "same_observations": True}


def _save_rows(run, rows):
    write_json(run.path / "metrics.json", rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (run.path / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config, device, run, scenario="all"):
    rows, checks = [], {}
    if scenario in ("all", "hello"):
        result = hello_world(config, device)
        write_json(run.path / "hello_world.json", result)
        checks["hello_world"] = True
        run.log(f"Hello World: BER={result['ber']:.6g}, CRC failures={result['crc_failures']}")
    seed_all(config.runtime.seed, device, config.runtime.cpu_threads)
    model = E2EModel(config, device, config.runtime.seed)
    resolved = model.components.adapter.metadata()
    write_json(run.path / "resolved_config.json", {"requested": config.to_dict(), "generated": resolved})
    run.log(f"Resolved TB={resolved['tb_size']}, coded bits={resolved['num_coded_bits']}, "
            f"data RE={resolved['num_data_re']}, device={device}")
    if scenario == "hello":
        return {"checks": checks, "rows": rows}
    if scenario in ("all", "passthrough", "ofdm-loopback"):
        counts = ErrorCounts()
        first = True
        while counts.total_blocks < config.evaluation.passthrough_tbs:
            size = min(config.evaluation.batch_size,
                       config.evaluation.passthrough_tbs - counts.total_blocks)
            batch = model.generate(size, "passthrough")
            result = model.receiver(batch.observation)
            counts.update(batch.labels.tb_bits, result, batch.noise)
            if first:
                checks["passthrough_default_receiver"] = compare_default(model, batch, result)
                checks["mapper_alignment"] = model.components.adapter.validate_mapping(
                    batch.labels.mapper_bits, batch.tx_grid)
                checks["ofdm_loopback"] = ofdm_loopback(model, batch)
                checks["cpu_gpu_consistency"] = check_cpu_gpu(config, device, batch, result)
                first = False
        if counts.bit_errors or counts.block_errors or counts.crc_failures:
            raise AssertionError(f"Noiseless passthrough failed: {counts.to_dict()}")
        rows.append({"scenario": "passthrough", "snr_db": None,
                     "noise_variance": 0., "receiver_noise_variance": config.evaluation.receiver_noise_floor,
                     "seed": config.runtime.seed, **counts.to_dict()})
        checks["passthrough"] = True
        _save_rows(run, rows)
        run.log(f"Passthrough: {counts.total_blocks} TB, zero errors and all CRCs passed")
    if scenario in ("all", "awgn"):
        for index, snr in enumerate(config.evaluation.snr_db):
            seed = config.runtime.seed + 1000 + index
            seed_all(seed, device, config.runtime.cpu_threads)
            model = E2EModel(config, device, seed)
            counts = ErrorCounts()
            digest = hashlib.sha256()
            checked_default = False
            while counts.total_blocks < config.evaluation.max_tbs:
                size = min(config.evaluation.batch_size,
                           config.evaluation.max_tbs - counts.total_blocks)
                batch = model.generate(size, config.channel.kind, snr)
                result = model.receiver(batch.observation)
                if not checked_default:
                    compare_default(model, batch, result)
                    checked_default = True
                counts.update(batch.labels.tb_bits, result, batch.noise)
                digest.update(batch.labels.tb_bits.cpu().numpy().tobytes())
                digest.update(batch.observation.y.cpu().numpy().tobytes())
                if (counts.total_blocks >= config.evaluation.min_tbs
                        and counts.block_errors >= config.evaluation.target_block_errors):
                    break
            no = esn0_to_no(snr)
            counts.validate_noise(no)
            rows.append({"scenario": config.channel.kind, "snr_db": snr, "snr_definition": "data-symbol Es/N0",
                         "noise_variance": no, "receiver_noise_variance": no, "seed": seed,
                         "batch_size": config.evaluation.batch_size, "input_sha256": digest.hexdigest(),
                         "default_receiver_match": True, "noise_power_check": True, **counts.to_dict()})
            _save_rows(run, rows)
            run.log(f"{config.channel.kind.upper()} Es/N0={snr:g} dB: BER={rows[-1]['ber']:.6g}, "
                    f"BLER={rows[-1]['bler']:.6g}, TB={counts.total_blocks}, CRC fail={counts.crc_failures}")
        points = [r for r in rows if r["scenario"] == config.channel.kind]
        checks["awgn_trend"] = (len(points) >= 2 and points[0]["bit_errors"] > 0
                                 and points[-1]["ber"] < points[0]["ber"]
                                 and points[-1]["bler"] < points[0]["bler"])
        system = config.system
        plot_awgn(rows, run.path / "ber_bler.png",
                  f"B0: {system.n_size_bwp} PRB / QPSK / MCS table {system.mcs_table}, index {system.mcs_index}")
        if not checks["awgn_trend"]:
            raise AssertionError("AWGN sweep does not demonstrate an error-rate decrease")
    checks["seed_reproducibility"] = check_reproducibility(config, device)
    write_json(run.path / "checks.json", checks)
    return {"checks": checks, "rows": rows}
