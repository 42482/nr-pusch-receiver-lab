"""Initialize project-local caches before importing PHY-dependent test modules."""
from pathlib import Path
from fm_receiver.runtime import prepare_cache

prepare_cache(Path(__file__).resolve().parents[1])
