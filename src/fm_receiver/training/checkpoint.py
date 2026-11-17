"""Atomic, provenance-checked experiment checkpoints."""

import hashlib
from pathlib import Path
import os
import random
import numpy as np
import torch


def source_hashes(root):
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*"))
    files += [root / name for name in ("pyproject.toml", "environment.yml",
                                      "requirements.lock.txt", "conda-linux-64.lock")]
    files += [p for p in sorted((root / "config").glob("*")) if p.name != "workers.local.toml"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_checkpoint(path, device, layout_id, codec_id):
    # Only load checkpoints produced by this project from trusted experiment storage.
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if value["layout_id"] != layout_id or value["codec_layout_id"] != codec_id:
        raise ValueError("Checkpoint resource/codec layout mismatch")
    return value
