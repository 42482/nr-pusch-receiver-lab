from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=json_default, allow_nan=False) + "\n")


class RunArtifacts:
    def __init__(self, root, output, config_path, config, arguments, kind):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = Path(output).resolve() if output else root / "results" / f"{stamp}_{kind}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = {"status": "running", "kind": kind, "arguments": arguments,
                         "started_utc": stamp, "config_path": str(config_path),
                         "command": shlex.join([sys.executable, *sys.argv]),
                         "project_root": str(root)}
        self.log_path = self.path / "run.log"
        shutil.copy2(config_path, self.path / "config.toml")
        write_json(self.path / "requested_config.json", config.to_dict())
        files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*"))
        environment_dir = self.path / "environment"
        environment_dir.mkdir()
        for name in ("pyproject.toml", "environment.yml", "conda-linux-64.lock", "requirements.lock.txt"):
            source = root / name
            if source.is_file():
                shutil.copy2(source, environment_dir / name)
                files.append(source)
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files if p.is_file()}
        write_json(self.path / "source_hashes.json", hashes)
        self.write_manifest()

    def write_manifest(self):
        write_json(self.path / "manifest.json", self.manifest)

    def log(self, message):
        print(message, flush=True)
        with self.log_path.open("a") as file:
            file.write(message + "\n")

    def save_environment(self, snapshot):
        write_json(self.path / "environment.json", snapshot)
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError("Could not record dependency versions: " + result.stderr)
        (self.path / "requirements.freeze.txt").write_text(result.stdout)
        check = subprocess.run([sys.executable, "-m", "pip", "check"],
                               text=True, capture_output=True, timeout=30)
        self.manifest["dependency_check_passed"] = check.returncode == 0
        (self.path / "dependency_check.txt").write_text(check.stdout + check.stderr)
        self.write_manifest()

    def finish(self, status, **values):
        self.manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **values)
        self.write_manifest()


def plot_awgn(rows, path, title):
    import matplotlib.pyplot as plt
    points = [r for r in rows if r["scenario"] in ("awgn", "tdl-a")]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, total, color in (("ber", "total_bits", "C0"), ("bler", "total_blocks", "C1")):
        nonzero = [p for p in points if p[key] > 0]
        if nonzero:
            ax.semilogy([p["snr_db"] for p in nonzero], [p[key] for p in nonzero],
                        "o-", color=color, label=key.upper())
        zero = [p for p in points if p[key] == 0]
        if zero:
            # Display resolution markers rather than implying a positive estimate
            # or independent post-LDPC bit errors for a confidence interval.
            bounds = [1 / p[total] for p in zero]
            ax.semilogy([p["snr_db"] for p in zero], bounds, "v", color=color,
                        label=key.upper() + ": 0 observed (marker at 1/N)")
    ax.set(xlabel="Data-symbol Es/N0 (dB)", ylabel="Error rate", title=title)
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
