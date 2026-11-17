"""All experiment-generated storage stays on the project data volume."""

from pathlib import Path
import os

DATA_ROOT = Path(os.environ.get("FM_RECEIVER_DATA_ROOT",
                               Path(__file__).resolve().parents[3])).resolve()


def data_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(DATA_ROOT):
        raise ValueError(f"Generated files must be under {DATA_ROOT}: {path}")
    return path
