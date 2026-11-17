"""Device selection and reproducibility; called before constructing PHY blocks."""

import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import sys


def prepare_cache(root: Path):
    for variable, relative in {
        "TMPDIR": "tmp", "PIP_CACHE_DIR": "pip", "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch",
        "CUDA_CACHE_PATH": "cuda", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton", "NUMBA_CACHE_DIR": "numba",
        "CONDA_PKGS_DIRS": "conda/pkgs",
    }.items():
        path = root / ".cache" / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["MPLBACKEND"] = "Agg"


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu, cuda:N and auto devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this execution session. Run in a GPU-enabled "
                               "SSH terminal, or explicitly select --device cpu for CPU validation.")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} does not exist")
        requested = f"cuda:{index}"
    else:
        requested = "cpu"
    x = torch.tensor([1 + 1j], dtype=torch.complex64, device=requested)
    if not torch.allclose(x * x.conj(), torch.tensor([2 + 0j], device=requested)):
        raise RuntimeError("Complex device arithmetic failed")
    return requested


def seed_all(seed: int, device: str, cpu_threads: int = 4):
    import numpy as np
    import torch
    import sionna.phy
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    sionna.phy.config.device = device
    sionna.phy.config.precision = "single"
    sionna.phy.config.seed = seed


def environment_snapshot(device: str | None = None):
    packages = {}
    for name in ("fm-receiver", "torch", "sionna-no-rt", "sionna", "numpy", "scipy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    snapshot = {"python": sys.version, "executable": sys.executable,
                "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                "platform": platform.platform(), "packages": packages, "device": device}
    try:
        import torch
        snapshot.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda)
        if device and device.startswith("cuda"):
            props = torch.cuda.get_device_properties(device)
            snapshot.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory)
    except ImportError:
        pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        snapshot["nvidia_smi"] = (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        snapshot["nvidia_smi"] = str(exc)
    return snapshot
