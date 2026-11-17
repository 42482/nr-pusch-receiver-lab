"""Dependency-driven multi-host job scheduling under GNU Screen."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import tomllib

from ..paths import data_path
from ..runtime import prepare_cache
from .transport import Worker, Transport
from .worker import atomic_json, run_job

ROOT = Path(__file__).resolve().parents[3]


def workers_from_file(path):
    if path.exists():
        with path.open("rb") as stream:
            workers = [Worker(**item) for item in tomllib.load(stream)["workers"]]
    else:
        workers = [Worker(socket.gethostname(), str(ROOT))]
    for worker in workers:
        data_path(worker.root)
        if not worker.name or not 0 < worker.port <= 65535 or worker.expected_gpus <= 0:
            raise ValueError("Invalid worker configuration")
    if len({w.name for w in workers}) != len(workers):
        raise ValueError("Worker names must be unique")
    return workers


def preflight(transport, check_python=True):
    worker = transport.worker
    report = {"worker": asdict(worker), "ready": False, "gpus": []}
    stage = "connection"
    try:
        host = transport.command(["hostname"]).stdout.strip()
        report["hostname"] = host
        stage = "gpu_inventory"
        raw = transport.command(["nvidia-smi", "--query-gpu=uuid,index,memory.free,name",
                                 "--format=csv,noheader,nounits"]).stdout
        for line in raw.splitlines():
            uuid, index, free, name = [s.strip() for s in line.split(",", 3)]
            report["gpus"].append({"uuid": uuid, "index": int(index),
                                   "free_mib": int(free), "name": name})
        if check_python:
            stage = "project_environment"
            python = str(Path(worker.root) / ".conda/fm-receiver/bin/python")
            expected = str(Path(worker.root) / ".conda/fm-receiver")
            code = ("import json,sys,torch; from pathlib import Path; "
                    f"assert Path(sys.prefix).resolve()==Path({expected!r}).resolve(); "
                    "assert sys.prefix==sys.base_prefix; assert torch.cuda.is_available(); "
                    "assert torch.__version__=='2.11.0+cu128'; "
                    "print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
                    "'executable':sys.executable,'torch':torch.__version__,"
                    "'cuda_runtime':torch.version.cuda}))")
            report["environment"] = json.loads(transport.command([python, "-I", "-c", code]).stdout)
        stage = "gpu_capacity"
        if len(report["gpus"]) < worker.expected_gpus:
            raise RuntimeError("Fewer GPUs than expected; review worker configuration")
        report["gpus"] = report["gpus"][:worker.expected_gpus]
        report["ready"] = True
        report["environment_checked"] = check_python
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
        report["error"] = str(exc)
        report["failed_stage"] = stage
        if isinstance(exc, subprocess.CalledProcessError):
            report["detail"] = ((exc.stdout or "") + (exc.stderr or ""))[-2000:]
    return report


def prepare_workers(workers, output):
    """Deployment/environment gate only: never submits training jobs."""
    output = data_path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "preparing", "training_started": False, "workers": {}}
    atomic_json(output / "state.json", state)
    from ..training.checkpoint import source_hashes
    expected_hashes = source_hashes(ROOT)
    for worker in workers:
        transport = Transport(ROOT, worker)
        state["workers"][worker.name] = {"status": "checking" if worker.local else "deploying",
                                        "log": str(output / f"{worker.name}-setup.log")}
        atomic_json(output / "state.json", state)
        print(f"[{worker.name}] environment preparation; logs: {output}", flush=True)
        try:
            if not worker.local:
                transport.connect()
                transport.deploy(output / f"{worker.name}-setup.log")
            report = preflight(transport, check_python=True)
            if report["ready"]:
                code = ("import json; from pathlib import Path; "
                        "from fm_receiver.training.checkpoint import source_hashes; "
                        f"print(json.dumps(source_hashes(Path({worker.root!r}))))")
                actual = json.loads(transport.command([
                    str(Path(worker.root) / ".conda/fm-receiver/bin/python"), "-I", "-c", code]).stdout)
                report["source_matches"] = actual == expected_hashes
                if not report["source_matches"]:
                    report.update(ready=False, failed_stage="source_identity")
            state["workers"][worker.name].update(report)
            state["workers"][worker.name]["status"] = "ready" if report["ready"] else "failed"
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            state["workers"][worker.name].update(status="failed", ready=False, error=str(exc))
        atomic_json(output / "state.json", state)
    state["status"] = "ready" if all(v.get("ready") for v in state["workers"].values()) else "failed"
    atomic_json(output / "state.json", state)
    print(json.dumps(state, indent=2), flush=True)
    return 0 if state["status"] == "ready" else 2


def job_graph():
    jobs = {"offline": {"kind": "offline", "deps": [], "gpu": False},
            "tdl": {"kind": "tdl", "deps": [], "gpu": True}}
    for seed in (42, 43):
        for stage in ("outer", "decoder"):
            small, full = f"overfit_{stage}_{seed}", f"train_{stage}_{seed}"
            jobs[small] = {"kind": "train", "stage": stage, "seed": seed,
                           "overfit": True, "deps": ["tdl"], "gpu": True}
            jobs[full] = {"kind": "train", "stage": stage, "seed": seed,
                          "overfit": False, "deps": [small], "gpu": True}
        joint = f"joint_{seed}"
        jobs[joint] = {"kind": "train", "stage": "joint", "seed": seed, "overfit": False,
                       "deps": [f"train_outer_{seed}", f"train_decoder_{seed}"], "gpu": True}
        jobs[f"compare_{seed}"] = {"kind": "compare", "seed": seed,
                                   "deps": [joint], "gpu": True}
    return {name: {**value, "status": "pending"} for name, value in jobs.items()}


def job_argv(root, run_id, name, job, remaining_hours, smoke=False):
    root = Path(root)
    base = root / "results/pipelines" / run_id / "jobs"
    output = base / name / "output"
    python = str(root / ".conda/fm-receiver/bin/python")
    common = [python, "-I"]
    if job["kind"] == "offline":
        return common + [str(root / "scripts/x410_offline.py"), "--output", str(output)]
    if job["kind"] == "tdl":
        return common + [str(root / "scripts/validate_tdl.py"), "--output", str(output),
                         "--device", "cuda:0"]
    if job["kind"] == "compare":
        seed = job["seed"]
        args = common + [str(root / "scripts/compare.py"), "--output", str(output),
                         "--outer-checkpoint", str(base / f"train_outer_{seed}/output/best.pt"),
                         "--joint-checkpoint", str(base / f"joint_{seed}/output/best.pt")]
        if job.get("restart"):
            args += ["--resume"]
        if smoke:
            args += ["--min-tbs", "4", "--max-tbs", "4", "--batch-size", "2"]
        return args
    args = common + [str(root / "scripts/train.py"), "--stage", job["stage"],
                     "--seed", str(job["seed"]), "--output", str(output),
                     "--max-hours", str(min(8., remaining_hours))]
    if job.get("overfit") and not smoke:
        args += ["--overfit", "--eval-interval", "50", "--steps", "4000"]
    if job["stage"] == "joint":
        args += ["--outer-checkpoint", str(base / f"train_outer_{job['seed']}/output/best.pt"),
                 "--inner-checkpoint", str(base / f"train_decoder_{job['seed']}/output/best.pt")]
    if job.get("restart"):
        args += ["--resume"]
    if smoke:
        args += ["--steps", "2", "--batch-size", "2", "--microbatch", "2",
                 "--validation-frames", "4", "--eval-interval", "2"]
    return args


def start_job(transport, run_dir, name, job, gpu, remaining, smoke=False):
    w = transport.worker
    remote_dir = Path(w.root) / "results/pipelines" / run_dir.name / "jobs" / name
    local_dir = run_dir / "jobs" / name
    local_dir.mkdir(parents=True, exist_ok=True)
    transport.command(["mkdir", "-p", str(remote_dir)])
    for dependency in job["deps"]:
        if job["kind"] in ("compare", "train") and not job.get("overfit"):
            source = run_dir / "jobs" / dependency
            if source.exists() and not w.local:
                transport.command(["mkdir", "-p", str(remote_dir.parent / dependency)])
                transport.copy(str(source) + "/", str(remote_dir.parent / dependency) + "/")
    # Paired evaluation also needs the independent B1 checkpoint, not only joint weights.
    if job["kind"] == "compare" and not w.local:
        dep = f"train_outer_{job['seed']}"
        transport.command(["mkdir", "-p", str(remote_dir.parent / dep)])
        transport.copy(str(run_dir / "jobs" / dep) + "/", str(remote_dir.parent / dep) + "/")
    spec = {"name": name, "root": w.root, "output": str(remote_dir),
            "restart": bool(job.get("attempt", 0)),
            "gpu_uuid": gpu, "max_hours": min(8., remaining),
            "argv": job_argv(w.root, run_dir.name, name, job, remaining, smoke)}
    atomic_json(local_dir / "job_spec.json", spec)
    if not w.local:
        transport.copy(local_dir / "job_spec.json", remote_dir / "job_spec.json")
    socket_dir = str(Path(w.root) / ".cache/screen")
    transport.command(["mkdir", "-p", socket_dir])
    transport.command(["chmod", "700", socket_dir])
    env = ["env", f"SCREENDIR={socket_dir}", f"TMPDIR={Path(w.root) / '.cache/tmp'}"]
    session = f"fx-{run_dir.name[-12:]}-{name}"[:55]
    transport.command([*env, "screen", "-c", str(Path(w.root) / "config/screenrc"),
                       "-ln", "-L", "-Logfile", str(remote_dir / "screen.log"), "-dmS", session,
                       str(Path(w.root) / ".conda/fm-receiver/bin/python"), "-I",
                       str(Path(w.root) / "scripts/pipeline.py"), "_job", "--run",
                       str(remote_dir / "job_spec.json")])
    job.update(status="running", worker=w.name, gpu_uuid=gpu, session=session,
               started_epoch=time.time())


def controller(run_dir):
    prepare_cache(ROOT)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text())
    lock_file = (run_dir / "controller.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state["controller_pid"] = os.getpid()
    state["status"] = "preflighting"
    atomic_json(state_path, state)
    transports = {w["name"]: Transport(ROOT, Worker(**w)) for w in state["workers"]}
    state["preflight"] = {}
    ready = {}
    for name, transport in transports.items():
        try:
            if not transport.worker.local:
                transport.deploy()
            report = preflight(transport)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            report = {"ready": False, "error": str(exc)}
        state["preflight"][name] = report
        if report["ready"]:
            ready[name] = transport
    # Verify deployments contain the same frozen source before starting jobs.
    from ..training.checkpoint import source_hashes
    if state["source_hashes"] != source_hashes(ROOT):
        raise RuntimeError("Pipeline source changed since launch")
    for name, transport in list(ready.items()):
        if transport.worker.local:
            continue
        code = ("import json; from pathlib import Path; "
                "from fm_receiver.training.checkpoint import source_hashes; "
                f"print(json.dumps(source_hashes(Path({transport.worker.root!r}))))")
        hashes = json.loads(transport.command([
            str(Path(transport.worker.root) / ".conda/fm-receiver/bin/python"), "-I", "-c", code]).stdout)
        if hashes != state["source_hashes"]:
            state["preflight"][name].update(ready=False, error="Deployed source mismatch")
            del ready[name]
    seen_gpus = set()
    for name, report in state["preflight"].items():
        if report.get("ready"):
            ids = {gpu["uuid"] for gpu in report["gpus"]}
            if seen_gpus & ids:
                report.update(ready=False, error="GPU UUID already advertised by another worker")
                ready.pop(name, None)
            else:
                seen_gpus |= ids
    state["status"] = "running"
    atomic_json(state_path, state)
    while True:
        remaining = state["max_hours"] - (time.time() - state["started_epoch"]) / 3600
        if remaining <= 0:
            state["status"] = "budget_exhausted"
            break
        for name, job in state["jobs"].items():
            if job["status"] not in ("running", "unknown"):
                continue
            transport = transports[job["worker"]]
            remote_dir = Path(transport.worker.root) / "results/pipelines" / run_dir.name / "jobs" / name
            try:
                value = json.loads(transport.command(["cat", str(remote_dir / "job_status.json")]).stdout)
            except (OSError, subprocess.SubprocessError, ValueError):
                if time.time() - job["started_epoch"] > 60:
                    job["status"] = "unknown"
                    job["error"] = "No worker heartbeat; inspect Screen before resuming"
                continue
            if value["status"] != "running":
                if not transport.worker.local:
                    transport.copy(str(remote_dir) + "/", str(run_dir / "jobs" / name) + "/", pull=True)
                job.update(status=value["status"], exit_code=value.get("exit_code"))
            else:
                job["status"] = "running"
        used = {(j.get("worker"), j.get("gpu_uuid")) for j in state["jobs"].values()
                if j["status"] in ("running", "unknown")}
        launched = False
        for name, job in state["jobs"].items():
            if job["status"] != "pending":
                continue
            deps = [state["jobs"][d]["status"] for d in job["deps"]]
            if any(s in ("failed", "overfit_failed", "blocked_dependency") for s in deps):
                job["status"] = "blocked_dependency"
                continue
            if any(s != "completed" for s in deps):
                continue
            candidates = []
            if not job["gpu"]:
                candidates = [(t, None) for t in transports.values() if t.worker.local]
            else:
                for worker_name, transport in ready.items():
                    candidates += [(transport, g["uuid"]) for g in state["preflight"][worker_name]["gpus"]
                                   if g["free_mib"] >= 8000 and (worker_name, g["uuid"]) not in used]
            # Prefer the three-card worker for neural jobs; keep local card available for diagnostics.
            candidates.sort(key=lambda pair: pair[0].worker.local if job["kind"] == "train" else
                            not pair[0].worker.local)
            if not candidates:
                continue
            transport, gpu = candidates[0]
            try:
                start_job(transport, run_dir, name, job, gpu, remaining, state.get("smoke", False))
                used.add((transport.worker.name, gpu))
                launched = True
            except (OSError, subprocess.SubprocessError) as exc:
                job.update(status="failed", error=str(exc))
        state["heartbeat_epoch"] = time.time()
        atomic_json(state_path, state)
        statuses = [j["status"] for j in state["jobs"].values()]
        if all(s == "completed" for s in statuses):
            state["status"] = "diagnostic_completed" if state.get("smoke") else "completed"
            break
        if not launched and not any(s == "running" for s in statuses):
            state["status"] = "blocked" if any(s in ("pending", "unknown") for s in statuses) else "failed"
            break
        time.sleep(5)
    atomic_json(state_path, state)
    print(json.dumps({"status": state["status"], "run": str(run_dir)}, indent=2), flush=True)
    return 0 if state["status"] in ("completed", "diagnostic_completed") else 2


def main(argv=None):
    prepare_cache(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "resume", "connect", "preflight",
                                           "prepare", "_run", "_job"))
    parser.add_argument("--workers", type=Path, default=ROOT / "config/workers.local.toml")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--max-hours", type=float, default=24.)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "_job":
        return run_job(args.run)
    if args.action == "_run":
        run = data_path(args.run)
        try:
            return controller(run)
        except BaseException as exc:
            state = json.loads((run / "state.json").read_text())
            state.update(status="controller_failed", error=str(exc))
            atomic_json(run / "state.json", state)
            raise
    workers = workers_from_file(args.workers)
    if args.action == "prepare":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return prepare_workers(workers, args.run or ROOT / "results/worker_prepare" / stamp)
    if args.action in ("connect", "preflight"):
        reports = []
        for worker in workers:
            transport = Transport(ROOT, worker)
            if args.action == "connect":
                transport.connect()
            else:
                reports.append(preflight(transport, check_python=worker.local))
        if reports:
            print(json.dumps(reports, indent=2))
            return 0 if all(r["ready"] for r in reports) else 2
        return 0
    if args.action == "status":
        if args.run:
            path = data_path(args.run)
        else:
            paths = sorted((ROOT / "results/pipelines").glob("*/state.json"))
            if not paths:
                print("No pipeline has been started.")
                return 0
            path = paths[-1].parent
        print((path / "state.json").read_text())
        return 0
    if args.action == "start":
        if not math.isfinite(args.max_hours) or args.max_hours <= 0:
            parser.error("--max-hours must be positive")
        for worker in workers:
            if not worker.local:
                # Authentication is interactive before detaching; SSH handles the password.
                Transport(ROOT, worker).connect()
        from ..training.checkpoint import source_hashes
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run = data_path(args.run or ROOT / "results/pipelines" / stamp)
        run.mkdir(parents=True, exist_ok=False)
        state = {"status": "starting", "started_epoch": time.time(), "max_hours": args.max_hours,
                 "workers": [asdict(w) for w in workers], "jobs": job_graph(),
                 "source_hashes": source_hashes(ROOT), "smoke": args.smoke}
    else:
        if args.run is None:
            parser.error("resume requires --run")
        run = data_path(args.run)
        with (run / "controller.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Controller is still active; use status instead of resume")
        state = json.loads((run / "state.json").read_text())
        for job in state["jobs"].values():
            if job["status"] in ("failed", "interrupted", "budget_exhausted", "blocked_dependency"):
                job["status"] = "pending"
                job["attempt"] = job.get("attempt", 0) + 1
        for name, job in state["jobs"].items():
            checkpoint = run / "jobs" / name / "output/last.pt"
            job["restart"] = checkpoint.exists() if job["kind"] == "train" else (
                job["kind"] == "compare" and (run / "jobs" / name / "output/manifest.json").exists())
        state["started_epoch"] = time.time()
        state["max_hours"] = args.max_hours
        state.pop("controller_pid", None)
        state.pop("heartbeat_epoch", None)
    socket_dir = ROOT / ".cache/screen"
    socket_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    socket_dir.chmod(0o700)
    os.environ["SCREENDIR"] = str(socket_dir)
    session = "fmrx-" + run.name
    state["session"] = session
    atomic_json(run / "state.json", state)
    command = ["screen", "-c", str(ROOT / "config/screenrc"), "-ln", "-L", "-Logfile",
               str(run / "controller.log"), "-dmS", session, sys.executable, "-I",
               str(ROOT / "scripts/pipeline.py"), "_run", "--run", str(run)]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        state.update(status="launch_failed", error=str(exc))
        atomic_json(run / "state.json", state)
        raise
    confirmed = False
    for _ in range(50):
        try:
            current = json.loads((run / "state.json").read_text())
            confirmed = "controller_pid" in current
        except (OSError, ValueError):
            confirmed = False
        if confirmed:
            break
        time.sleep(.1)
    if not confirmed:
        state.update(status="launch_unconfirmed", error="Screen returned without a controller heartbeat. "
                     "Inspect Screen/socket/PTY permissions from an ordinary SSH terminal.")
        atomic_json(run / "state.json", state)
        print(json.dumps(state, indent=2))
        return 2
    print(f"Pipeline submitted: {run}\nSession: {session}\n"
          f"Inspect: SCREENDIR={shlex.quote(str(socket_dir))} screen -r {shlex.quote(session)}\n"
          "Submission does not imply GPU preflight or training acceptance; check status.")
    return 0
