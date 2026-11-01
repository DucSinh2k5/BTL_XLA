
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import copy
import hashlib
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
import torch

import build_dtd_caption_method01_prefix_notebook as builder
from test_dtd_grounded_caption import actual_view_runtime


@lru_cache(maxsize=1)
def catalog():
    return builder.inspect_caption_files()


@lru_cache(maxsize=1)
def code_cells():
    return {index: builder.source_of(cell) for index, cell in enumerate(builder.build_notebook(catalog())["cells"])}


@pytest.fixture
def caption_runtime(tmp_path):
    expected = copy.deepcopy(catalog()[0])
    source = builder.FOLDER / expected["filename"]
    rows = pd.read_csv(source).fillna("")
    frames = {split: rows.loc[rows.split.eq(split), ["image_id"]].copy() for split in ("train", "val", "test")}
    ns = {"os": os, "Path": Path, "hashlib": hashlib, "shutil": shutil,
          "GENERATE_CAPTIONS_IF_MISSING": False, "CAPTION_INPUT_POLICY": "dtd_lexical_v2"}

    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    ns["write_json"] = write_json
    exec(builder.PREFIX_INPUT_HELPERS, ns)
    return ns, source, expected, frames, tmp_path


def test_four_uploaded_csvs_are_identified_by_contents_not_just_names():
    methods = catalog()
    assert [item["generation_method"] for item in methods] == [
        "prefix_control", "first_valid_control", "single_grounded_control", "multibeam_grounded"]
    assert [item["stored_valid_total"] for item in methods] == [4749, 5640, 5640, 5640]
    assert len({item["captioner_fingerprint"] for item in methods}) == 1
    assert all(item["rows"] == 5640 and item["policy"] == "dtd_lexical_v2" for item in methods)


def test_notebook_matches_builder_is_self_contained_and_keeps_original_models_and_training():
    before = {path: path.read_bytes() for path in [builder.SOURCE, *builder.FOLDER.glob("*.csv")]}
    notebook = builder.build_notebook(catalog())
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert all(path.read_bytes() == content for path, content in before.items())
    original = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    for index in (11, 12, 14, 15, 19, 21):
        assert builder.source_of(notebook["cells"][index]) == builder.source_of(original["cells"][index])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = builder.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert "from dtd_" not in source and "import build_" not in source
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["caption_method"] == "prefix_control" and not metadata["caption_generation"]
    assert metadata["experiment"] == "full_both" and metadata["default_epochs"] == 50
    code = code_cells()
    assert 'DTD_CAPTION_01_PREFIX_EPOCHS", "50"' in code[4]
    assert "caption_01_florence_prefix_vitl16" in code[4]
    normalized = code[4].replace("DTD_CAPTION_01_PREFIX_OUTPUT", "DTD_FULL_BOTH_OUTPUT").replace(
        "caption_01_florence_prefix_vitl16", "ablation_05_full_both_vitl16").replace(
        "DTD_CAPTION_01_PREFIX_EPOCHS", "DTD_FULL_BOTH_EPOCHS")
    assert normalized == builder.source_of(original["cells"][4])
    assert "ensure_dtd_caption_csv(" not in code[9]
    assert '"caption_method": CAPTION_BINDING' in code[17]
    assert "caption_01_florence_prefix_results.csv" in code[23]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]


def test_configuration_ignores_unrelated_prior_caption_and_output_variables(monkeypatch):
    monkeypatch.setenv("DTD_CAPTIONS_INPUT_PATH", "/wrong/other_method.csv")
    monkeypatch.setenv("DTD_CAPTION_INPUT_POLICY", "legacy_strict")
    monkeypatch.setenv("DTD_FULL_BOTH_SEED", "123")
    monkeypatch.delenv("DTD_PREFIX_CAPTIONS_INPUT_PATH", raising=False)
    monkeypatch.delenv("DTD_CAPTION_01_PREFIX_SEED", raising=False)
    monkeypatch.delenv("DTD_SPLIT", raising=False)
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["CAPTIONS_INPUT_PATH"] == "" and ns["TRAIN_SEEDS"] == [42]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    selected = ns["SELECTED_ABLATIONS"]
    assert len(selected) == 1 and all(selected[0][key] for key in ("query_fusion", "candidate_fusion", "semantic_selector"))
    monkeypatch.setenv("DTD_SPLIT", "2")
    with pytest.raises(ValueError, match="split 1"):
        exec(code_cells()[2], {"os": os})


def test_exact_prefix_csv_binds_and_is_copied_without_touching_the_original(caption_runtime):
    ns, source, expected, frames, folder = caption_runtime
    before = source.read_bytes()
    cached = folder / "shared/captions.csv"
    result, binding = ns["stage_method_caption_csv"](str(source), cached, frames, expected)
    assert result == cached and cached.read_bytes() == before == source.read_bytes()
    assert binding["method"] == "prefix_control" and binding["rows"] == 5640
    assert ns["verify_method_caption_csv"](cached, frames, expected) == binding
    saved = json.loads((cached.parent / "caption_source.json").read_text(encoding="utf-8"))
    assert saved["generation_needed"] is False


def test_automatic_discovery_accepts_only_exact_filename_and_hash(caption_runtime):
    ns, source, expected, _, folder = caption_runtime
    root = folder / "kaggle/input/datasets/owner/all-4-captions"
    root.mkdir(parents=True)
    destination = root / expected["filename"]
    wrong = builder.FOLDER / catalog()[1]["filename"]
    shutil.copy2(wrong, destination)
    with pytest.raises(FileNotFoundError, match="regeneration"):
        ns["resolve_method_caption_csv"]("", expected, roots=[folder / "kaggle/input"])
    shutil.copy2(source, destination)
    assert ns["resolve_method_caption_csv"]("", expected, roots=[folder / "kaggle/input"]) == destination.resolve()
    with pytest.raises(FileNotFoundError, match="CAPTIONS_INPUT_PATH"):
        ns["resolve_method_caption_csv"](folder / "missing.csv", expected, local_candidates=[source])


def test_wrong_method_is_rejected_even_if_the_expected_hash_is_changed(caption_runtime):
    ns, _, expected, frames, _ = caption_runtime
    other = builder.FOLDER / catalog()[1]["filename"]
    with pytest.raises(ValueError, match="Wrong or modified"):
        ns["verify_method_caption_csv"](other, frames, expected)
    expected["sha256"] = catalog()[1]["sha256"]
    with pytest.raises(ValueError, match="generation_method"):
        ns["verify_method_caption_csv"](other, frames, expected)


def test_wrong_split_assignment_is_rejected(caption_runtime):
    ns, source, expected, frames, _ = caption_runtime
    image_id = frames["train"].iloc[0].image_id
    frames["train"] = frames["train"].iloc[1:].copy()
    frames["val"] = pd.concat([frames["val"], pd.DataFrame({"image_id": [image_id]})], ignore_index=True)
    with pytest.raises(ValueError, match="official DTD"):
        ns["verify_method_caption_csv"](source, frames, expected)


def test_stale_output_is_rejected_without_overwriting_cached_or_source_csv(caption_runtime):
    ns, source, expected, frames, folder = caption_runtime
    cached = folder / "shared/captions.csv"
    cached.parent.mkdir()
    other = builder.FOLDER / catalog()[2]["filename"]
    shutil.copy2(other, cached)
    before = cached.read_bytes()
    with pytest.raises(ValueError, match="another caption CSV"):
        ns["stage_method_caption_csv"](str(source), cached, frames, expected)
    assert cached.read_bytes() == before
    assert source.read_bytes() != before


@pytest.mark.parametrize("change", [{"GENERATE_CAPTIONS_IF_MISSING": True}, {"CAPTION_INPUT_POLICY": "legacy_strict"}])
def test_generation_and_policy_changes_are_not_silently_allowed(caption_runtime, change):
    ns, source, expected, frames, folder = caption_runtime
    ns.update(change)
    with pytest.raises(ValueError, match="precomputed captions"):
        ns["stage_method_caption_csv"](str(source), folder / "shared/captions.csv", frames, expected)


def test_real_uploaded_prefix_csv_flows_through_existing_audit_and_both_text_banks(caption_runtime):
    ns, source, expected, frames, folder = caption_runtime
    runtime = actual_view_runtime()
    ns.update(runtime)
    rows = pd.read_csv(source).fillna("")
    class_names = sorted({image_id.split("/")[0] for image_id in rows.image_id})

    def cache(views, clip_asset, output):
        view = views["screened"]
        return {"embedding_dim": 512, "fingerprint": "synthetic-CPU-test-only",
                "views": {"screened": {**view, "embeddings": torch.zeros(len(view["image_ids"]), 512)}}}

    ns.update(OUT_PUT_DIR=folder, PROJECT_DIR=folder, train_df=frames["train"], val_df=frames["val"],
        test_df=frames["test"], candidate_df=frames["train"].iloc[:376], CAPTIONS_INPUT_PATH=str(source),
        CAPTION_METHOD_EXPECTED=expected, CAPTION_SEMANTIC_THRESHOLD=.3, CLIP_MODEL_LOCAL_PATH="unused-test-asset",
        DEVICE=torch.device("cpu"), class_names=class_names, pd=pd, torch=torch,
        load_ablation_captions=lambda path, splits: pd.read_csv(path).fillna(""),
        caption_filter_diagnostics=lambda *args: None, resolve_clip_path=lambda *args: "unused-test-asset",
        encode_view_cache=cache, display=lambda *args: None)
    exec(code_cells()[9], ns)
    assert ns["CAPTION_BINDING"]["method"] == "prefix_control"
    assert int(ns["caption_audit"].screened_valid.sum()) == 4749
    assert (folder / "caption_method.json").is_file()
    spec = {"name": "full_both", "view": "screened", "q_transform": "aligned", "c_transform": "aligned",
            "query_fusion": True, "candidate_fusion": True, "semantic_selector": True}
    banks = ns["make_ablation_banks"](spec, 42)
    assert len(banks["c"]["source_ids"]) == 376 and len(banks["q"]["source_ids"]) == 5640
    assert set(banks["c"]["source_ids"]) <= set(frames["train"].image_id)
    assert not set(banks["c"]["source_ids"]) & set(frames["test"].image_id)
    assert not banks["q"]["valid"].all()
    assert (folder / "shared/caption_role_coverage.csv").is_file()
