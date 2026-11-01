"""Verify that migration preserved every inventoried artifact and notebook."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import project_path  # noqa: E402


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    files = json.loads((ROOT / "manifests/migration_files.json").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT / "manifests/notebook_catalog.json").read_text(encoding="utf-8"))
    exceptions = []
    unchanged = []
    errors = []
    for row in files:
        path = ROOT / row["new"]
        if not path.is_file():
            errors.append("Missing: " + row["new"])
            continue
        if project_path(row["old"]) != path.resolve():
            errors.append("Bad path translation: " + row["old"])
        if digest(path) == row["sha256_before"]:
            unchanged.append(row["new"])
        elif (row["new"].startswith(("tools/notebook_builders/", "tests/"))
              or row["new"] in {"src/captioning/dtd_caption_generation_cells.py",
                                "data/captions/dtd/grounded_controls/caption_methods_manifest.json"}):
            exceptions.append(row["new"])
        else:
            errors.append("Unexpected content change: " + row["new"])
    for row in catalog:
        path = ROOT / row["path"]
        if not path.is_file() or digest(path) != row["sha256"]:
            errors.append("Notebook source/output changed: " + row["path"])
    on_disk = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts}
    registered = {row["path"] for row in catalog}
    if on_disk != registered:
        errors.append(f"Notebook catalog mismatch: unlisted={on_disk-registered}, absent={registered-on_disk}")
    manifest = json.loads((ROOT / "data/captions/dtd/grounded_controls/caption_methods_manifest.json").read_text(encoding="utf-8"))
    for name in manifest["created_training_notebooks"].values():
        if not (ROOT / name).is_file():
            errors.append("Broken caption notebook manifest link: " + name)
    status = {"files_checked": len(files), "unchanged_files": len(unchanged),
              "locator_updates": exceptions, "notebooks_unchanged": len(catalog), "errors": errors}
    (ROOT / "manifests/migration_validation.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
