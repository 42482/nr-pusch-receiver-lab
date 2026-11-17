#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
mkdir -p .cache/{tmp,xdg,pip,matplotlib,torch,cuda,triton,inductor,screen,ssh}
chmod 700 .cache/screen .cache/ssh
export TMPDIR="$task_root/.cache/tmp" XDG_CACHE_HOME="$task_root/.cache/xdg"
export PIP_CACHE_DIR="$task_root/.cache/pip" MPLCONFIGDIR="$task_root/.cache/matplotlib"
export TORCH_HOME="$task_root/.cache/torch" CUDA_CACHE_PATH="$task_root/.cache/cuda"
export TRITON_CACHE_DIR="$task_root/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$task_root/.cache/inductor"
export SCREENDIR="$task_root/.cache/screen" PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec "$task_root/.conda/fm-receiver/bin/python" -I "$task_root/scripts/pipeline.py" "$@"
