"""A detached job owns its GPU lock and records exit/timeout status."""

from pathlib import Path
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time

from ..paths import data_path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run_job(spec_path):
    spec_path = data_path(spec_path)
    spec = json.loads(spec_path.read_text())
    root = data_path(spec["root"])
    output = data_path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    if spec.get("restart") and "--resume" not in spec["argv"] and "--output" in spec["argv"]:
        previous = data_path(spec["argv"][spec["argv"].index("--output") + 1])
        if previous.exists():
            if previous.parent != output:
                raise ValueError("Cannot archive an output outside the owned job directory")
            previous.rename(previous.with_name(previous.name + f".attempt-{time.time_ns()}"))
    status_path = output / "job_status.json"
    lock_dir = root / ".cache/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    resource = spec.get("gpu_uuid") or "cpu-" + spec["name"]
    with (lock_dir / (resource + ".lock")).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            atomic_json(status_path, {"status": "failed", "error": "Resource already locked"})
            return 2
        os.environ["PYTHONNOUSERSITE"] = "1"
        if spec.get("gpu_uuid"):
            os.environ["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        if spec.get("gpu_uuid"):
            probe = subprocess.run([sys.executable, "-I", "-c",
                                    "import torch; assert torch.cuda.is_available(); "
                                    "assert torch.cuda.device_count()==1"],
                                   env=env, capture_output=True, text=True)
            if probe.returncode:
                atomic_json(status_path, {"status": "failed", "error": probe.stderr})
                return 2
        status = {"status": "running", "host": socket.gethostname(),
                  "gpu_uuid": spec.get("gpu_uuid"), "name": spec["name"],
                  "started_epoch": time.time(), "worker_pid": os.getpid()}
        stop = []
        old_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop.append(signum))
        with (output / "job.log").open("a") as log:
            child = subprocess.Popen(spec["argv"], cwd=root, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["pid"] = child.pid
            atomic_json(status_path, status)
            reason = None
            while child.poll() is None:
                if stop or time.time() - status["started_epoch"] >= spec["max_hours"] * 3600:
                    reason = "interrupted" if stop else "budget_exhausted"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                status["heartbeat_epoch"] = time.time()
                atomic_json(status_path, status)
                time.sleep(2)
        status.update(status=reason or ("completed" if child.returncode == 0 else "failed"),
                      exit_code=child.returncode, finished_epoch=time.time())
        atomic_json(status_path, status)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        return 0 if status["status"] == "completed" else 2
