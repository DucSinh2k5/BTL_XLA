"""Make existing flat research modules available when a builder is run directly."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for relative in ("", "src/captioning", "src/training", "tools/notebook_builders"):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)

from project_paths import PROJECT_ROOT, project_path  # noqa: E402
