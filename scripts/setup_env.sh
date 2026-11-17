#!/usr/bin/env bash
# Create an independent Conda environment from committed package definitions.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_prefix="$task_root/.conda/fm-receiver"
task_wheelhouse=""
task_offline=false
task_conda_spec="$task_root/conda-linux-64.lock"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix|--wheelhouse|--conda-lock)
            [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --prefix) task_prefix="$(realpath -m -- "$2")" ;;
                --wheelhouse) task_wheelhouse="$(realpath -m -- "$2")" ;;
                --conda-lock) task_conda_spec="$(realpath -m -- "$2")" ;;
            esac
            shift 2 ;;
        --offline) task_offline=true; shift ;;
        --help|-h)
            printf 'Usage: bash scripts/setup_env.sh [--prefix PATH] [--offline --wheelhouse PATH] [--conda-lock FILE]\n'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || {
    printf 'This locked profile supports Linux x86_64 only.\n' >&2; exit 2;
}
command -v conda >/dev/null || { printf 'Install and initialize Conda first.\n' >&2; exit 2; }
[[ -f "$task_conda_spec" ]] || { printf 'Missing Conda lock: %s\n' "$task_conda_spec" >&2; exit 2; }
if $task_offline && [[ -z "$task_wheelhouse" ]]; then
    printf -- '--offline requires a complete --wheelhouse directory.\n' >&2; exit 2
fi
cd "$task_root"
mkdir -p .cache/{tmp,pip,matplotlib,xdg,torch,conda/pkgs} .conda
export TMPDIR="$task_root/.cache/tmp" PIP_CACHE_DIR="$task_root/.cache/pip"
export MPLCONFIGDIR="$task_root/.cache/matplotlib" XDG_CACHE_HOME="$task_root/.cache/xdg"
export TORCH_HOME="$task_root/.cache/torch" PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$task_root/.cache/conda/pkgs}"
export CONDA_ENVS_PATH="$task_root/.conda"
# Prefix activation works without registering environments in ~/.conda.
export CONDA_REGISTER_ENVS=false CONDA_REPORT_ERRORS=false
unset PYTHONPATH PYTHONHOME

task_marker="$task_prefix/.fm-receiver-environment"
task_fingerprint="$(cat "$task_root/conda-linux-64.lock" "$task_root/requirements.lock.txt" | sha256sum | cut -d ' ' -f 1)"
if [[ -e "$task_prefix" ]]; then
    [[ -f "$task_prefix/conda-meta/history" && -f "$task_marker" ]] || {
        printf 'Refusing to reuse an unmanaged environment: %s\nChoose a new --prefix.\n' "$task_prefix" >&2
        exit 2
    }
    [[ "$(cat "$task_marker")" == "$task_fingerprint" ]] || {
        printf 'Environment locks changed. Build a fresh environment using a new --prefix.\n' >&2
        exit 2
    }
else
    task_conda_flags=()
    $task_offline && task_conda_flags+=(--offline)
    conda create --prefix "$task_prefix" --file "$task_conda_spec" --copy \
        --no-default-packages --yes "${task_conda_flags[@]}"
    printf '%s\n' "$task_fingerprint" > "$task_marker"
fi
task_python="$task_prefix/bin/python"
"$task_python" -I - "$task_prefix" <<'PY'
from pathlib import Path
import sys
prefix = Path(sys.argv[1]).resolve()
assert sys.version_info[:3] == (3, 11, 15), sys.version
assert Path(sys.prefix).resolve() == prefix
assert Path(sys.base_prefix).resolve() == prefix, 'Borrowed/venv interpreter detected'
assert Path(sys.executable).resolve().is_relative_to(prefix), sys.executable
assert not (prefix / 'pyvenv.cfg').exists()
print('Independent Conda Python:', sys.executable)
PY
task_pip_flags=()
if [[ -n "$task_wheelhouse" ]]; then
    task_pip_flags+=(--no-index --find-links "$task_wheelhouse")
fi
"$task_python" -I -m pip install --disable-pip-version-check "${task_pip_flags[@]}" \
    -r "$task_root/requirements.lock.txt"
"$task_python" -I -m pip install --disable-pip-version-check --no-deps --no-build-isolation -e "$task_root"
"$task_python" -I -m pip check
conda env config vars set --prefix "$task_prefix" PYTHONNOUSERSITE=1 >/dev/null
"$task_python" -I - "$task_root" <<'PY'
from pathlib import Path
import importlib.metadata
import json
import sys
import torch
import sionna.phy
import h5py
root = Path(sys.argv[1])
x = torch.tensor([1 + 1j], dtype=torch.complex64)
assert (x * x.conj()).item() == 2
report = {
    'python': sys.version, 'executable': sys.executable,
    'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'sionna': importlib.metadata.version('sionna-no-rt'), 'h5py': h5py.__version__,
    'cuda_available': torch.cuda.is_available(),
}
assert report['torch'] == '2.11.0+cu128'
assert report['sionna'] == '2.2.0'
(root / '.cache/environment-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
printf '\nEnvironment ready. Activate with:\n  conda activate %q\n' "$task_prefix"
