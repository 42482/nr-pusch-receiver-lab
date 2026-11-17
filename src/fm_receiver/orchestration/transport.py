"""System SSH with persistent connections; credentials are never serialized."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import shlex
import subprocess
import time

from ..paths import data_path


@dataclass
class Worker:
    name: str
    root: str
    host: str = "local"
    user: str = "jiayl"
    port: int = 22
    expected_gpus: int = 1

    @property
    def local(self):
        return self.host == "local"


class Transport:
    def __init__(self, project_root, worker):
        self.root, self.worker = Path(project_root), worker
        self.ssh_dir = self.root / ".cache/ssh"
        self.ssh_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ssh_dir.chmod(0o700)
        key = hashlib.sha256(f"{worker.user}@{worker.host}:{worker.port}".encode()).hexdigest()[:16]
        self.control = self.ssh_dir / ("c-" + key)

    def ssh(self, interactive=False):
        w = self.worker
        return ["ssh", "-p", str(w.port), "-o", "ConnectTimeout=10",
                "-o", "ControlMaster=auto", "-o", "ControlPersist=24h",
                "-o", f"ControlPath={self.control}",
                "-o", f"UserKnownHostsFile={self.ssh_dir / 'known_hosts'}",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"BatchMode={'no' if interactive else 'yes'}"]

    def connect(self):
        if not self.worker.local:
            check = subprocess.run([*self.ssh(), "-O", "check",
                                    f"{self.worker.user}@{self.worker.host}"],
                                   capture_output=True, text=True)
            if check.returncode == 0:
                return
            subprocess.run([*self.ssh(interactive=True), "-MNf",
                            f"{self.worker.user}@{self.worker.host}"], check=True)

    def command(self, argv, timeout=30):
        if self.worker.local:
            return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True)
        return subprocess.run([*self.ssh(), f"{self.worker.user}@{self.worker.host}",
                               shlex.join([str(x) for x in argv])],
                              capture_output=True, text=True, timeout=timeout, check=True)

    def shell(self, command, timeout=30):
        return self.command(["bash", "-c", command], timeout)

    def copy(self, source, destination, pull=False, excludes=()):
        args = ["rsync", "-a", "--exclude=*.tmp"]
        for exclude in excludes:
            args.append("--exclude=" + exclude)
        if self.worker.local:
            if Path(source).resolve() == Path(destination).resolve():
                return
        else:
            args += ["-e", shlex.join(self.ssh())]
            remote = f"{self.worker.user}@{self.worker.host}:"
            if pull:
                source = remote + shlex.quote(str(source))
            else:
                destination = remote + shlex.quote(str(destination))
        subprocess.run([*args, str(source), str(destination)], check=True, timeout=1800)

    def deploy(self, log_path=None):
        if self.worker.local:
            return
        data_path(self.worker.root)
        target = shlex.quote(self.worker.root)
        self.shell(f"if [ -d {target} ] && [ ! -f {target}/.fmrx-worker ]; then "
                   "echo 'Refusing to overwrite an unmanaged worker directory' >&2; exit 2; fi; "
                   f"mkdir -p {target}; touch {target}/.fmrx-worker")
        self.copy(str(self.root) + "/", self.worker.root + "/", excludes=(
            "/.git/", "/.conda/", "/.venv/", "/.cache/", "/results/", "/weights/",
            "/data/", "/datasets/", "/.aws/", "/.codex/", "/.agents/", ".env*",
            "/config/workers.local.toml", "__pycache__/", "*.egg-info/"))
        log_path = data_path(log_path or self.root / "results/worker_setup" /
                             f"{self.worker.name}-{time.time_ns()}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Only the manager executable is discovered here; the project prefix is fresh.
        manager_paths = [str(Path(self.worker.root).parent / name / "bin")
                         for name in ("anaconda3", "miniconda3", "miniforge3")]
        command = (f"set -o pipefail; cd {target}; export PATH=" +
                   shlex.quote(":".join(manager_paths)) +
                   ":\"$HOME/anaconda3/bin:$HOME/miniconda3/bin:$HOME/miniforge3/bin:$PATH\"; "
                   "export CONDA_PKGS_DIRS=" +
                   shlex.quote(str(Path(self.worker.root) / ".cache/conda/pkgs")) + "; "
                   "bash scripts/setup_env.sh 2>&1 | tee worker-setup.log")
        argv = ["bash", "-c", command]
        with log_path.open("w") as stream:
            result = subprocess.run([*self.ssh(), f"{self.worker.user}@{self.worker.host}",
                                     shlex.join(argv)], stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            raise RuntimeError(f"Worker environment setup failed; inspect {log_path}")
