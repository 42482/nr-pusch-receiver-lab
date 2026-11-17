#!/usr/bin/env bash
# Deliberate lock updates; setup_env.sh leaves tracked definitions unchanged.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_prefix="${1:-$task_root/.conda/fm-receiver}"
task_python="$task_prefix/bin/python"
[[ -f "$task_prefix/conda-meta/history" && -f "$task_prefix/.fm-receiver-environment" ]] || {
    printf 'Expected a Conda prefix created by this project.\n' >&2; exit 2;
}
mkdir -p "$task_root/.cache"
export PIP_CACHE_DIR="$task_root/.cache/pip" XDG_CACHE_HOME="$task_root/.cache/xdg"
"$task_python" -I -m pip check
conda list --prefix "$task_prefix" --explicit --sha256 > "$task_root/.cache/conda-base.export"
"$task_python" -I -m pip freeze --all --exclude-editable > "$task_root/.cache/pip.export"
"$task_python" -I - "$task_root" "$task_prefix" <<'PY'
from pathlib import Path
import importlib.metadata
import json
import re
import sys
root = Path(sys.argv[1])
prefix = Path(sys.argv[2])
base = (root / '.cache/conda-base.export').read_text()
packages = (root / '.cache/pip.export').read_text().splitlines()
assert '@EXPLICIT' in base
public = {line.rsplit('#', 1)[-1]: line for line in
          (root / 'conda-linux-64.lock').read_text().splitlines() if line.startswith('https://')}
lines = []
for line in base.splitlines():
    if line and not line.startswith(('#', '@')):
        if line.startswith('file:'):
            line = public.get(line.rsplit('#', 1)[-1], line)
        assert line.startswith('https://')
        assert len(line.rsplit('#', 1)[-1]) == 64, 'Expected SHA256 package lock'
    lines.append(line)
base = '\n'.join(lines) + '\n'
normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
conda_names = {normalize(json.loads(p.read_text())['name']) for p in (prefix / 'conda-meta').glob('*.json')}
portable = []
for line in packages:
    if ' @ ' in line:
        name, _ = line.split(' @ ', 1)
        assert normalize(name) in conda_names, 'Nonportable direct pip dependency: ' + name
        line = name + '==' + importlib.metadata.version(name)
    portable.append(line)
packages = portable
assert all(re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s]+', line) for line in packages), 'Nonportable pip reference'
header = ('# Linux x86_64 / CPython 3.11.15 / PyTorch CUDA 12.8.\n'
          '# Regenerate intentionally with scripts/export_env.sh.\n'
          '--index-url https://pypi.org/simple\n'
          '--extra-index-url https://download.pytorch.org/whl/cu128\n\n')
(root / 'conda-linux-64.lock').write_text(base)
(root / 'requirements.lock.txt').write_text(header + '\n'.join(packages) + '\n')
print('Updated locks. Review the diff, rebuild a fresh environment, and rerun checks before committing.')
PY
