from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from fm_receiver.config import ExperimentConfig
from fm_receiver.data.export import export_samples
from fm_receiver.data.schema import ObservationBatch
from fm_receiver.e2e_model import E2EModel
from fm_receiver.evaluation.metrics import ErrorCounts
from fm_receiver.evaluation.runner import compare_default, ofdm_loopback
from fm_receiver.runtime import seed_all, prepare_cache

prepare_cache(Path(__file__).resolve().parents[1])


class PuschTests(unittest.TestCase):
    def setUp(self):
        seed_all(42, "cpu")
        self.config = ExperimentConfig()
        self.model = E2EModel(self.config, "cpu", 42)

    @torch.inference_mode()
    def test_passthrough_mapper_labels_and_llr_sign(self):
        batch = self.model.generate(16, "passthrough")
        result = self.model.receiver(batch.observation)
        torch.testing.assert_close(result.tb_hat, batch.labels.tb_bits, rtol=0, atol=0)
        self.assertTrue(bool(result.tb_crc.all()))
        self.assertEqual(tuple(batch.tx_grid.shape), (16, 1, 1, 14, 48))
        self.assertEqual(self.model.components.adapter.metadata()["dmrs_symbol_indices"], [2])
        self.assertTrue(self.model.components.adapter.validate_mapping(batch.labels.mapper_bits, batch.tx_grid))
        torch.testing.assert_close((result.llr > 0).float(), batch.labels.mapper_bits, rtol=0, atol=0)
        self.assertEqual(batch.labels.mapper_bits.shape[-1],
                         2 * int(self.model.components.adapter.data_mask.sum()))
        self.assertTrue(compare_default(self.model, batch, result))

    @torch.inference_mode()
    def test_awgn_power_and_default_receiver(self):
        batch = self.model.generate(16, "awgn", 0.)
        result = self.model.receiver(batch.observation)
        self.assertTrue(compare_default(self.model, batch, result))
        counts = ErrorCounts()
        counts.update(batch.labels.tb_bits, result, batch.noise)
        self.assertTrue(counts.validate_noise(1.))
        self.assertGreater(counts.block_errors, 0)

    @torch.inference_mode()
    def test_ofdm_loopback(self):
        batch = self.model.generate(3, "passthrough")
        self.assertTrue(ofdm_loopback(self.model, batch)["passed"])

    @torch.inference_mode()
    def test_seed_reproduces_payload_noise_and_decoded_output(self):
        a = self.model.generate(3, "awgn", 2.)
        ar = self.model.receiver(a.observation)
        seed_all(42, "cpu")
        model = E2EModel(self.config, "cpu", 42)
        b = model.generate(3, "awgn", 2.)
        br = model.receiver(b.observation)
        for x, y in ((a.labels.tb_bits, b.labels.tb_bits), (a.noise, b.noise),
                     (a.observation.y, b.observation.y), (ar.tb_hat, br.tb_hat), (ar.tb_crc, br.tb_crc)):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    @torch.inference_mode()
    def test_saved_samples_replay_with_partial_final_batch(self):
        config = replace(self.config, samples=replace(self.config.samples, num_frames=5),
                         evaluation=replace(self.config.evaluation, batch_size=3))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "samples"
            checks = export_samples(config, "cpu", out, log=lambda message: None)
            self.assertTrue(checks["saved_observation_replay"])
            with np.load(out / "samples.npz", allow_pickle=False) as data:
                self.assertEqual(data["y"].shape[0], 5)
                self.assertEqual(data["h_hat_ls"].shape, data["err_var"].shape)
                np.testing.assert_array_equal(data["frame_ids"], np.arange(5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is not exposed in this execution session")
    @torch.inference_mode()
    def test_same_observations_cpu_and_gpu(self):
        batch = self.model.generate(3, "passthrough")
        cpu = self.model.receiver(batch.observation)
        seed_all(42, "cuda:0")
        gpu_model = E2EModel(self.config, "cuda:0", 42)
        obs = ObservationBatch(batch.observation.frame_ids, batch.observation.y.cuda(),
                               batch.observation.no.cuda(), batch.observation.layout_id)
        gpu = gpu_model.receiver(obs)
        torch.testing.assert_close(cpu.tb_hat, gpu.tb_hat.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(cpu.tb_crc, gpu.tb_crc.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(cpu.h_hat, gpu.h_hat.cpu(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
