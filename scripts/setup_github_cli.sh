#!/usr/bin/env bash
# Install a checksum-verified GitHub CLI under this checkout, without sudo.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$task_root/.cache/tools" "$task_root/.cache/tmp"
export TMPDIR="$task_root/.cache/tmp"
python3 - "$task_root" <<'PY'
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import urllib.request

root = Path(sys.argv[1])
version = "2.104.0"
asset = f"gh_{version}_linux_amd64.tar.gz"
base = f"https://github.com/cli/cli/releases/download/v{version}/"
def download(name):
    request = urllib.request.Request(base + name, headers={"User-Agent": "nr-pusch-receiver-lab"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()
checksums = download(f"gh_{version}_checksums.txt").decode()
expected = next(line.split()[0] for line in checksums.splitlines() if line.split()[-1] == asset)
archive = download(asset)
if hashlib.sha256(archive).hexdigest() != expected:
    raise RuntimeError("GitHub CLI checksum mismatch")
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
    content = stream.extractfile(f"gh_{version}_linux_amd64/bin/gh").read()
binary = root / ".cache/tools/gh"
binary.write_bytes(content)
binary.chmod(0o755)
print(f"GitHub CLI {version}: {binary}")
PY
