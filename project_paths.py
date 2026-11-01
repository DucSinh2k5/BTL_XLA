"""Locate reorganized project files without retaining duplicate legacy files."""
from __future__ import annotations

import fnmatch
import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _moves():
    moves = json.loads((PROJECT_ROOT / "manifests/migration_plan.json").read_text(encoding="utf-8"))
    aliases = json.loads((PROJECT_ROOT / "manifests/path_aliases.json").read_text(encoding="utf-8"))
    return sorted(moves + aliases, key=lambda item: len(item["old"]), reverse=True)


def project_path(legacy_relative_path):
    """Translate a recorded XLA-relative path to its current TACS location."""
    requested = Path(legacy_relative_path)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("Expected a project-relative path without parent traversal")
    name = requested.as_posix()
    for move in _moves():
        if name == move["old"] or name.startswith(move["old"] + "/"):
            name = move["new"] + name[len(move["old"]):]
            break
    result = (PROJECT_ROOT / name).resolve()
    if not result.is_relative_to(PROJECT_ROOT):
        raise ValueError("Project path escaped the project root")
    return result


def project_glob(legacy_pattern):
    """Find recorded files by their original relative name, independent of layout."""
    records = json.loads((PROJECT_ROOT / "manifests/migration_files.json").read_text(encoding="utf-8"))
    return [PROJECT_ROOT / row["new"] for row in records if fnmatch.fnmatchcase(row["old"], legacy_pattern)]
