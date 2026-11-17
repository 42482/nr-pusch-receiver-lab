"""Regression checks for evaluation restarts and detached controller startup."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from fm_receiver.config import ExperimentConfig
from fm_receiver.evaluation.comparison import compare
from fm_receiver.orchestration.pipeline import main as pipeline_main
from fm_receiver.runtime import prepare_cache, seed_all
from fm_receiver.training.checkpoint import source_hashes
from fm_receiver.training.engine import TrainOptions, build_training

ROOT = Path(__file__).resolve().parents[1]
prepare_cache(ROOT)


class RecoveryTests(unittest.TestCase):
    def test_comparison_restart_reproduces_inputs_and_rejects_changed_checkpoints(self):
        seed_all(42, "cpu")
        config = ExperimentConfig()
        config = replace(config, runtime=replace(config.runtime, device="cpu"))
        model, codec, networks = build_training(config, "cpu", TrainOptions(stage="joint"))
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as temporary:
            directory = Path(temporary)
            checkpoint = directory / "joint.pt"
            torch.save({"format_version": 1, "layout_id": model.components.adapter.layout_id,
                        "codec_layout_id": codec.layout_id, "source_hashes": source_hashes(ROOT),
                        "outer": networks.outer.state_dict(), "inner": networks.inner.state_dict()},
                       checkpoint)
            output = directory / "comparison"
            with redirect_stdout(io.StringIO()):
                compare(config, "cpu", output, ROOT, checkpoint, checkpoint,
                        batch_size=2, min_tbs=2, max_tbs=2)
                original = json.loads((output / "metrics.json").read_text())
                compare(config, "cpu", output, ROOT, checkpoint, checkpoint,
                        batch_size=2, min_tbs=2, max_tbs=2, resume=True)
            resumed = json.loads((output / "metrics.json").read_text())
            self.assertEqual(len(resumed), 54)
            fields = ("receiver", "channel", "snr_db", "input_sha256", "bit_errors", "total_bits",
                      "block_errors", "total_blocks", "crc_failures")
            self.assertEqual([{key: row[key] for key in fields} for row in original],
                             [{key: row[key] for key in fields} for row in resumed])
            self.assertTrue(json.loads((output / "manifest.json").read_text())["restarted_from_beginning"])
            # A changed file must fail identity validation before loading weights or overwriting results.
            previous_manifest = (output / "manifest.json").read_bytes()
            with checkpoint.open("ab") as stream:
                stream.write(b"changed-checkpoint")
            with self.assertRaisesRegex(ValueError, "identical inputs"):
                compare(config, "cpu", output, ROOT, checkpoint, checkpoint,
                        batch_size=2, min_tbs=2, max_tbs=2, resume=True)
            self.assertEqual((output / "manifest.json").read_bytes(), previous_manifest)

    def test_resume_requires_a_new_controller_heartbeat(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache/tmp") as temporary:
            directory = Path(temporary)
            initial = {"status": "failed", "jobs": {}, "controller_pid": 12345,
                       "heartbeat_epoch": 1, "started_epoch": 1, "max_hours": 1}
            for started in (False, True):
                with self.subTest(new_controller_started=started):
                    (directory / "state.json").write_text(json.dumps(initial))

                    def launch(command, check):
                        self.assertIn("_run", command)
                        if started:
                            state = json.loads((directory / "state.json").read_text())
                            state.update(controller_pid=67890, heartbeat_epoch=2)
                            (directory / "state.json").write_text(json.dumps(state))
                        return subprocess.CompletedProcess(command, 0)

                    with patch("fm_receiver.orchestration.pipeline.workers_from_file", return_value=[]), \
                         patch("fm_receiver.orchestration.pipeline.subprocess.run", side_effect=launch), \
                         patch("fm_receiver.orchestration.pipeline.time.sleep"), \
                         redirect_stdout(io.StringIO()):
                        code = pipeline_main(["resume", "--run", str(directory), "--max-hours", "1"])
                    state = json.loads((directory / "state.json").read_text())
                    if started:
                        self.assertEqual(code, 0)
                        self.assertEqual(state["controller_pid"], 67890)
                    else:
                        self.assertEqual(code, 2)
                        self.assertEqual(state["status"], "launch_unconfirmed")
                        self.assertNotIn("controller_pid", state)
