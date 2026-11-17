"""Small, self-describing receiver datasets with an executable replay check."""

import numpy as np
import torch
from .schema import ObservationBatch
from ..e2e_model import E2EModel
from ..evaluation.artifacts import write_json
from ..runtime import seed_all


def export_samples(config, device, directory, log=print):
    directory.mkdir(parents=True, exist_ok=False)
    seed_all(config.samples.seed, device, config.runtime.cpu_threads)
    model = E2EModel(config, device, config.samples.seed)
    adapter = model.components.adapter
    chunks = {key: [] for key in ("frame_ids", "y", "no", "h_hat_ls", "err_var",
                                   "tb_bits", "mapper_bits", "tx_grid", "tb_hat", "tb_crc", "llr")}
    generated = 0
    pilot_grid = None
    while generated < config.samples.num_frames:
        size = min(config.evaluation.batch_size, config.samples.num_frames - generated)
        batch = model.generate(size, config.channel.kind, config.samples.snr_db)
        result = model.receiver(batch.observation)
        adapter.validate_mapping(batch.labels.mapper_bits, batch.tx_grid)
        values = {"frame_ids": batch.observation.frame_ids,
                  "y": batch.observation.y, "no": batch.observation.no.expand(size),
                  "h_hat_ls": result.h_hat, "err_var": result.err_var.expand_as(result.h_hat.real),
                  "tb_bits": batch.labels.tb_bits, "mapper_bits": batch.labels.mapper_bits,
                  "tx_grid": batch.tx_grid, "tb_hat": result.tb_hat, "tb_crc": result.tb_crc,
                  "llr": result.llr}
        for key, value in values.items():
            array = value.detach().cpu().numpy()
            if key in ("tb_bits", "mapper_bits", "tb_hat"):
                array = array.astype(np.uint8)
            chunks[key].append(array)
        if pilot_grid is None:
            pilot_grid = np.where(~adapter.data_mask.numpy(), batch.tx_grid[0, 0, 0].cpu().numpy(), 0)
        generated += size
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    arrays.update(data_mask=adapter.data_mask.numpy(),
                  data_re_indices=adapter.data_re_indices.numpy(),
                  resource_type_grid=adapter.type_grid.numpy(), pilot_grid=pilot_grid)
    np.savez_compressed(directory / "samples.npz", **arrays)
    metadata = {"schema_version": 1, "num_frames": generated, "seed": config.samples.seed,
                "channel": config.channel.kind,
                "snr_db": config.samples.snr_db, "snr_definition": "data-symbol Es/N0",
                "device": device, "layout": adapter.metadata(),
                "fields": {key: {"shape": list(v.shape), "dtype": str(v.dtype)} for key, v in arrays.items()},
                "axes": {"y": ["frame", "rx", "rx_antenna", "ofdm_symbol", "subcarrier"],
                         "tx_grid": ["frame", "tx", "tx_antenna", "ofdm_symbol", "subcarrier"],
                         "tb_bits": ["frame", "tx", "information_bit"],
                         "mapper_bits": ["frame", "tx", "mapper_bit"],
                         "llr": ["frame", "tx", "mapper_bit"],
                         "h_hat_ls": ["frame", "rx", "rx_antenna", "tx", "stream", "ofdm_symbol", "subcarrier"],
                         "data_re_indices": ["data_symbol", "(ofdm_symbol, subcarrier)"]},
                "note": "Mapper labels include scrambling. TB labels are for post-decoding BER/BLER. "
                        "Pilot values and reserved zeros are represented by pilot_grid/resource_type_grid. "
                        "This small export is for interface/overfitting checks, not an independent test set."}
    write_json(directory / "samples.json", metadata)
    with np.load(directory / "samples.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["frame_ids"], np.arange(generated)):
            raise AssertionError("Sample frame IDs are missing or duplicated")
        ids = saved["data_re_indices"]
        if not np.array_equal(np.argwhere(saved["data_mask"]), ids):
            raise AssertionError("Stored data coordinates and mask disagree")
        for start in range(0, generated, config.evaluation.batch_size):
            stop = min(start + config.evaluation.batch_size, generated)
            y = torch.from_numpy(saved["y"][start:stop]).to(device)
            no = torch.from_numpy(saved["no"][start:stop]).to(device)[:, None, None]
            observation = ObservationBatch(torch.from_numpy(saved["frame_ids"][start:stop]),
                                            y, no, adapter.layout_id)
            replay = model.receiver(observation)
            np.testing.assert_array_equal(replay.tb_hat.cpu().numpy(), saved["tb_hat"][start:stop])
            np.testing.assert_array_equal(replay.tb_crc.cpu().numpy(), saved["tb_crc"][start:stop])
            adapter.validate_mapping(torch.from_numpy(saved["mapper_bits"][start:stop]).to(
                device=device, dtype=torch.float32), torch.from_numpy(saved["tx_grid"][start:stop]).to(device))
    checks = {"num_frames": generated, "mapping_alignment": True, "saved_observation_replay": True,
              "frame_ids_unique": True}
    write_json(directory / "checks.json", checks)
    log(f"Exported and replayed {generated} aligned frames: {directory}")
    return checks
