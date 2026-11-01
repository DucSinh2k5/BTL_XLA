import hashlib
import json

import pytest

from project_paths import PROJECT_ROOT, project_glob, project_path


def test_catalog_is_complete_and_all_notebook_bytes_are_preserved():
    catalog = json.loads((PROJECT_ROOT / "manifests/notebook_catalog.json").read_text(encoding="utf-8"))
    listed = {PROJECT_ROOT / row["path"] for row in catalog}
    assert listed == set(PROJECT_ROOT.rglob("*.ipynb"))
    assert len(listed) == 78
    for row in catalog:
        assert hashlib.sha256((PROJECT_ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_legacy_lookup_supports_files_and_directory_descendants():
    assert project_path("mtacs-train-captions-vitl16.ipynb").parent == (
        PROJECT_ROOT / "notebooks/02_improvements/query_text/sun397")
    assert project_path("output_ablation/analysis/saved_output_evidence.json").is_file()
    assert project_path("all_4_captions").is_dir()
    assert len(project_glob("mtacs-train-vitl16-dtd-ablation-*.ipynb")) == 8


@pytest.mark.parametrize("name", ["../outside", str(PROJECT_ROOT / "README.md")])
def test_legacy_lookup_refuses_paths_outside_project(name):
    with pytest.raises(ValueError):
        project_path(name)


def test_notebook_placement_uses_configuration_not_misleading_original_name():
    catalog = json.loads((PROJECT_ROOT / "manifests/notebook_catalog.json").read_text(encoding="utf-8"))
    records = {row["old_path"]: row for row in catalog}
    assert records["tacs-chuanpaper-sun-b16 (2).ipynb"]["dataset"] == "stanford_cars"
    assert records["SUN/table2/ban-cuoi-frozen-dino-car.ipynb"]["method"] == "tacs"
    assert records["SUN/table2/baseline_no_tacs_cars.ipynb"]["dataset"] == "sun397"


def test_caption_manifest_points_to_separate_method_notebooks():
    folder = PROJECT_ROOT / "data/captions/dtd/grounded_controls"
    manifest = json.loads((folder / "caption_methods_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["created_training_notebooks"]) == 4
    for name in manifest["created_training_notebooks"].values():
        path = PROJECT_ROOT / name
        assert path.is_file() and path.suffix == ".ipynb"
        assert "caption_ablation" in path.parts
    assert not list(folder.glob("*.ipynb"))
