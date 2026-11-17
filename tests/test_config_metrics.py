from pathlib import Path
import tempfile
import unittest

import torch
from fm_receiver.config import ExperimentConfig, load_config
from fm_receiver.data.schema import ReceiverResult
from fm_receiver.evaluation.metrics import ErrorCounts
from fm_receiver.runtime import resolve_device
from unittest.mock import patch


class ConfigurationTests(unittest.TestCase):
    def test_committed_config_matches_frozen_defaults(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(load_config(root / "config/b0.toml"), ExperimentConfig())

    def test_unknown_and_mistyped_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for text in ('[system]\nmcs_indx=9\n', '[evaluation]\nbatch_size=true\n',
                         '[evaluation]\nmax_tbs=1\n', '[system]\nmcs_index=14\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_explicit_cuda_does_not_silently_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda:0")
            self.assertEqual(resolve_device("auto"), "cpu")


class MetricsTests(unittest.TestCase):
    def test_crc_and_actual_block_errors_are_counted_independently(self):
        truth = torch.tensor([[[0., 1., 0.]], [[1., 0., 1.]]])
        decoded = torch.tensor([[[1., 1., 1.]], [[1., 0., 1.]]])
        result = ReceiverResult(decoded, torch.tensor([[True], [False]]))
        counts = ErrorCounts()
        counts.update(truth, result)
        self.assertEqual((counts.bit_errors, counts.total_bits, counts.block_errors,
                          counts.total_blocks, counts.crc_failures), (2, 6, 1, 2, 1))
        self.assertEqual(counts.to_dict()["bler"], .5)

    def test_incorrect_noise_scale_is_detected(self):
        counts = ErrorCounts(noise_energy=200., noise_real_energy=100., noise_imag_energy=100.,
                             noise_samples=100000)
        with self.assertRaises(AssertionError):
            counts.validate_noise(1.)


if __name__ == "__main__":
    unittest.main()
