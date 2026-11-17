"""Cross-layer invariants for channel, neural codec, recovery and capture ingress."""

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch

from fm_receiver.config import ExperimentConfig
from fm_receiver.runtime import prepare_cache, seed_all

from fm_receiver.e2e_model import E2EModel
from fm_receiver.training.engine import (TrainOptions, build_training, oracle_check,
                                        loss_and_output, sampled_batch, train)
from fm_receiver.hardware.capture import export_waveform, write_capture, load_observations
from fm_receiver.orchestration.pipeline import job_graph, job_argv, preflight, prepare_workers
from fm_receiver.orchestration.transport import Worker, Transport
from fm_receiver.orchestration.worker import run_job
from fm_receiver.sionna_adapter import soft_detect
from fm_receiver.paths import data_path

ROOT = Path(__file__).resolve().parents[1]
prepare_cache(ROOT)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        seed_all(42, "cpu")
        self.config = replace(ExperimentConfig(), runtime=replace(ExperimentConfig().runtime, device="cpu"))

    def test_tdl_power_perfect_csi_and_repeat(self):
        def once():
            seed_all(42, "cpu")
            m = E2EModel(self.config, "cpu", 42)
            batch = m.generate(8, "tdl-a", 6.)
            return m, batch
        m, a = once()
        _, b = once()
        torch.testing.assert_close(a.h_true, b.h_true, rtol=0, atol=0)
        torch.testing.assert_close(a.observation.y, b.observation.y, rtol=0, atol=0)
        energy = a.h_true.abs().square().mean((-2, -1))
        torch.testing.assert_close(energy, torch.ones_like(energy), rtol=1e-5, atol=1e-5)
        obs = replace(a.observation, y=a.observation.y-a.noise,
                      no=torch.tensor(1e-6))
        llr, _, _ = soft_detect(m.components, obs, a.h_true)
        bits, crc = m.components.receiver.decoder(llr)
        torch.testing.assert_close(bits, a.labels.tb_bits, rtol=0, atol=0)
        self.assertTrue(bool(crc.all()))
        self.assertLess(m.channel.diagnostics["max_delay_s"], m.channel.diagnostics["cp_duration_s"])

    def test_codec_oracle_erasures_and_joint_gradient(self):
        m, codec, networks = build_training(self.config, "cpu", TrainOptions(stage="joint"))
        self.assertTrue(oracle_check(m, codec)["oracle_tb_crc"])
        batch = m.generate(2, "awgn", 4.)
        llr = codec.to_code_llr((2*batch.labels.mapper_bits-1)*10)
        self.assertTrue(bool((llr[:, codec.sent:codec.sent+codec.punctured] == 0).all()))
        self.assertTrue(bool((llr[:, codec.known_zero] < 0).all()))
        loss, _, _, _ = loss_and_output(m, codec, networks, batch)
        loss.backward()
        for module in (networks.outer, networks.inner):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()
                                   if p.grad is not None), 0)
        # A flip prediction can recover an initially-zero punctured LLR.
        decoded, _ = networks.inner(llr, codec)
        self.assertTrue(bool((decoded[:, codec.sent:codec.sent+codec.punctured] != 0).any()))

    def test_microbatch_partition_preserves_training_inputs(self):
        m, _, _ = build_training(self.config, "cpu", TrainOptions())
        options = TrainOptions(batch_size=4, microbatch=2)
        full = sampled_batch(m, options, 2, 0, 4)
        parts = [sampled_batch(m, options, 2, i, 1) for i in range(4)]
        torch.testing.assert_close(torch.cat([b.observation.y for b in parts]), full.observation.y,
                                   rtol=0, atol=0)
        torch.testing.assert_close(torch.cat([b.labels.tb_bits for b in parts]), full.labels.tb_bits,
                                   rtol=0, atol=0)

    def test_checkpoint_resume_matches_uninterrupted_optimizer(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            directory = Path(directory)
            options = TrainOptions(steps=2, batch_size=2, microbatch=1, eval_interval=1,
                                   validation_frames=2)
            train(self.config, "cpu", options, directory / "continuous", ROOT)
            train(self.config, "cpu", replace(options, steps=1), directory / "resumed", ROOT)
            train(self.config, "cpu", options, directory / "resumed", ROOT, resume=True)
            a = torch.load(directory / "continuous/last.pt", weights_only=False)
            b = torch.load(directory / "resumed/last.pt", weights_only=False)
            self.assertEqual(a["step"], b["step"])
            for key in a["outer"]:
                torch.testing.assert_close(a["outer"][key], b["outer"][key], rtol=0, atol=0)

    def test_capture_cfo_gap_and_truth_free_replay(self):
        m = E2EModel(self.config, "cpu", 42)
        batch = m.generate(3, "passthrough")
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            directory = Path(directory)
            iq, spec = export_waveform(m, batch, directory / "tx")
            cfo, padding = 123., 13
            iq = np.pad(iq, ((padding, 0), (0, 0)))
            iq *= np.exp(2j*np.pi*cfo*np.arange(len(iq))/spec.logical_rate_hz)[:, None]
            starts = padding + np.arange(3)*spec.frame_samples
            path = directory / "capture.h5"
            write_capture(path, iq, spec, {"actual_rx_rate_hz": spec.logical_rate_hz,
                "noise_variance_grid": 1e-6, "frequency_offset_hz": cfo, "sync_source": "synthetic"},
                np.arange(3), starts, [{"kind": "overflow", "start_sample": int(starts[1]),
                                       "num_samples": 1}])
            # Deleting all labels cannot affect inference.
            (directory / "tx/labels.npz").unlink()
            observation, quality = load_observations(path, spec)
            self.assertEqual(observation.frame_ids.tolist(), [0, 2])
            self.assertEqual(len(quality["rejected"]), 1)
            torch.testing.assert_close(observation.y, batch.tx_grid[[0, 2]], rtol=1e-5, atol=1e-5)
            result = m.receiver(observation)
            torch.testing.assert_close(result.tb_hat, batch.labels.tb_bits[[0, 2]], rtol=0, atol=0)
            self.assertTrue(bool(result.tb_crc.all()))
            with self.assertRaises(ValueError):
                load_observations(path, replace(spec, layout_id="wrong"))

    def test_pipeline_dependency_and_worker_exit_contract(self):
        graph = job_graph()
        seen = set()
        while len(seen) < len(graph):
            available = {name for name, job in graph.items() if name not in seen and
                         set(job["deps"]) <= seen}
            self.assertTrue(available, "Pipeline dependency cycle")
            seen |= available
        self.assertIn("overfit_decoder_42", graph["train_decoder_42"]["deps"])
        self.assertIn("train_outer_42", graph["joint_42"]["deps"])
        args = job_argv(ROOT, "test", "joint_42", graph["joint_42"], 1.)
        self.assertIn("--inner-checkpoint", args)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            import sys
            directory = Path(directory)
            spec = {"name": "test", "root": str(ROOT), "output": str(directory / "job"),
                    "gpu_uuid": None, "max_hours": .01,
                    "argv": [sys.executable, "-I", "-c", "print('worker-ok')"]}
            (directory / "spec.json").write_text(json.dumps(spec))
            self.assertEqual(run_job(directory / "spec.json"), 0)
            result = json.loads((directory / "job/job_status.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertIn("worker-ok", (directory / "job/job.log").read_text())
        with self.assertRaises(ValueError):
            data_path("/home/jiayl/forbidden-output")

    def test_capture_resampling_retains_frame_alignment_and_crc(self):
        from scipy.signal import resample_poly
        m = E2EModel(self.config, "cpu", 42)
        batch = m.generate(4, "passthrough")
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            directory = Path(directory)
            iq, spec = export_waveform(m, batch, directory / "tx")
            logical = np.pad(iq, ((48, 48), (0, 0)))
            for up, down in [(2, 1), (4, 3), (32, 3)]:
                with self.subTest(rate_ratio=(up, down)):
                    raw = resample_poly(logical, up, down, axis=0).astype(np.complex64)
                    starts = np.rint((48 + np.arange(4) * spec.frame_samples) * up / down).astype(np.int64)
                    path = directory / f"rx-{up}-{down}.h5"
                    write_capture(path, raw, spec,
                                  {"actual_rx_rate_hz": spec.logical_rate_hz * up / down,
                                   "noise_variance_grid": 1e-6, "sync_source": "synthetic"},
                                  np.arange(4), starts)
                    obs, quality = load_observations(path, spec)
                    self.assertEqual(obs.frame_ids.tolist(), [0, 1, 2, 3])
                    self.assertEqual(quality["resampling"], [down, up])
                    result = m.receiver(obs)
                    torch.testing.assert_close(result.tb_hat, batch.labels.tb_bits, rtol=0, atol=0)
                    self.assertTrue(bool(result.tb_crc.all()))

    def test_capture_rejects_bad_indices_and_retains_generated_events(self):
        from fm_receiver.hardware.capture import WaveformSpec
        spec = WaveformSpec("example", 720000., 48, (4,) * 14, 15000.)
        iq = np.zeros((spec.frame_samples * 2, 1), dtype=np.complex64)
        metadata = {"actual_rx_rate_hz": 720000., "noise_variance_grid": 1., "sync_source": "provided"}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            path = Path(directory) / "rx.h5"
            for ids, starts in [([.5], [0]), ([0, 0], [0, 728]), ([0, 1], [728, 0])]:
                with self.assertRaises(ValueError):
                    write_capture(path, iq, spec, metadata, ids, starts)
            with self.assertRaises(ValueError):
                write_capture(path, iq, spec, {**metadata, "actual_rx_rate_hz": 480000.}, [0], [0])
            events = ({"kind": "gap", "start_sample": 0, "num_samples": 1} for _ in range(1))
            write_capture(path, iq, spec, metadata, [0, 1], [0, 728], events)
            obs, quality = load_observations(path, spec)
            self.assertEqual(obs.frame_ids.tolist(), [1])
            self.assertEqual(quality["rejected"][0]["frame_id"], 0)

    def test_preflight_reports_capacity_and_driver_failures_without_crashing(self):
        import subprocess
        transport = Mock()
        transport.worker = Worker("worker", str(ROOT), expected_gpus=3)
        transport.command.side_effect = [subprocess.CompletedProcess([], 0, "amax\n", ""),
                                        subprocess.CompletedProcess([], 0, "GPU-one, 0, 22000, RTX 3090\n", "")]
        result = preflight(transport, check_python=False)
        self.assertFalse(result["ready"])
        self.assertEqual(result["failed_stage"], "gpu_capacity")
        transport.command.side_effect = [subprocess.CompletedProcess([], 0, "amax\n", ""),
                                        subprocess.CalledProcessError(9, [], output="Driver unavailable", stderr="")]
        result = preflight(transport, check_python=False)
        self.assertEqual(result["failed_stage"], "gpu_inventory")
        self.assertIn("Driver unavailable", result["detail"])

    def test_prepare_gate_does_not_launch_jobs(self):
        worker = Worker("controller", str(ROOT))
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            output = Path(directory) / "prepare"
            from fm_receiver.training.checkpoint import source_hashes
            result = Mock(stdout=json.dumps(source_hashes(ROOT)))
            with patch("fm_receiver.orchestration.pipeline.Transport") as transport, patch(
                    "fm_receiver.orchestration.pipeline.preflight", return_value={"ready": True}):
                transport.return_value.command.return_value = result
                self.assertEqual(prepare_workers([worker], output), 0)
                transport.return_value.deploy.assert_not_called()
                for call in transport.return_value.command.call_args_list:
                    self.assertNotIn("screen", call.args[0])
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["status"], "ready")
            self.assertFalse(state["training_started"])

    def test_connect_reuses_an_authenticated_master(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as directory:
            transport = Transport(Path(directory), Worker("remote", str(ROOT), host="example.invalid"))
            with patch("fm_receiver.orchestration.transport.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0)) as run:
                transport.connect()
                self.assertEqual(run.call_count, 1)
                self.assertIn("check", run.call_args.args[0])
