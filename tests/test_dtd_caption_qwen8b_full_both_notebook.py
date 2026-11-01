"""Qwen8B CSV binding and unchanged notebook-05 training, tested without GPU models."""
import ast
import copy
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import build_dtd_caption_qwen8b_full_both_notebook as builder
import dtd_qwen_ablation_input as binding
import dtd_qwen_caption as qwen


TEXT = "rough gray grains cover pale uneven surface"


@lru_cache(maxsize=1)
def code_cells():
    return {index: builder.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


@pytest.fixture
def caption_input(tmp_path):
    records = [{"image_id": f"banded/{role}_{index}.jpg", "split": role}
               for role in ("train", "val", "test") for index in range(2)]
    rows = []
    for index, record in enumerate(records):
        text = "striped red fabric covers rough dark stone" if index % 2 else TEXT
        row = qwen.generation_row(record, text, "qwen3_vl_8b_instruct", "a" * 64, ["banded", "striped"])
        rows.append(qwen.audited_row(row, .95, "banded", "b" * 64, .30))
    path = tmp_path / "input/captions.csv"
    qwen.atomic_csv(path, rows, qwen.COLUMNS)
    frames = {role: pd.DataFrame([row for row in records if row["split"] == role]) for role in ("train", "val", "test")}
    return path, rows, frames, tmp_path


def stage(source, destination, frames, **kwargs):
    return binding.stage_qwen8b_caption_csv(source, destination, frames, ["banded", "striped"],
        kwargs.pop("policy", "dtd_lexical_v2"), kwargs.pop("generate", False), **kwargs)


def test_builder_preserves_original_and_matches_saved_notebook():
    before = builder.SOURCE.read_bytes()
    notebook = builder.build_notebook()
    assert builder.SOURCE.read_bytes() == before
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    original = json.loads(before)
    assert len(notebook["cells"]) == 24
    for index in (11, 12, 14, 15, 19, 21):
        assert builder.source_of(notebook["cells"][index]) == builder.source_of(original["cells"][index])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = builder.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert "from dtd_" not in source
            assert "import build_" not in source
    settings = code_cells()[4].replace("DTD_QWEN8B_FULL_BOTH_OUTPUT", "DTD_FULL_BOTH_OUTPUT").replace(
        "caption_05_qwen3vl_8b_vitl16", "ablation_05_full_both_vitl16").replace(
        "DTD_QWEN8B_FULL_BOTH_EPOCHS", "DTD_FULL_BOTH_EPOCHS")
    assert settings == builder.source_of(original["cells"][4])
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "full_both" and metadata["caption_method"] == "qwen3_vl_8b_instruct"
    assert not metadata["caption_generation"] and not metadata["captioner_finetuning"]
    assert metadata["default_epochs"] == 50 and metadata["default_caption_input_policy"] == "dtd_lexical_v2"
    assert '"caption_method": CAPTION_BINDING' in code_cells()[17]
    assert "caption_05_qwen3vl_8b_results.csv" in code_cells()[23]


def test_config_defaults_ignore_other_caption_experiment_variables(monkeypatch):
    for name in ("DTD_QWEN8B_FULL_BOTH_SEED", "DTD_QWEN8B_CAPTIONS_INPUT_PATH", "DTD_QWEN8B_CAPTIONS_SHA256", "DTD_SPLIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DTD_FULL_BOTH_SEED", "123")
    monkeypatch.setenv("DTD_CAPTIONS_INPUT_PATH", "wrong-florence.csv")
    monkeypatch.setenv("DTD_CAPTION_INPUT_POLICY", "legacy_strict")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["TRAIN_SEEDS"] == [42] and ns["CAPTIONS_INPUT_PATH"] == ""
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    assert not ns["EARLY_STOPPING"] and ns["RUN_FINAL_TEST"]
    spec = ns["SELECTED_ABLATIONS"]
    assert len(spec) == 1 and spec[0]["name"] == "full_both"
    assert all(spec[0][key] for key in ("query_fusion", "candidate_fusion", "semantic_selector"))
    monkeypatch.setenv("DTD_SPLIT", "2")
    with pytest.raises(ValueError, match="split 1"):
        exec(code_cells()[2], {"os": os})


def test_no_caption_generation_or_package_installation_code_remains():
    source = "\n".join(source for index, source in code_cells().items()
                       if builder.build_notebook()["cells"][index]["cell_type"] == "code")
    for forbidden in ("def load_florence", "def florence_checkpoint", "def run_local_dtd_caption_generation",
                      "CAPTION_WORKER_SOURCE", "Qwen3VLForConditionalGeneration", "pip install", "subprocess.Popen"):
        assert forbidden not in source
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[1]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[19]
    assert "ensure_dtd_caption_csv(" not in code_cells()[9]


def test_binding_matches_qwen_generator_metadata():
    assert binding.QWEN8B_EXPECTED["model"] == qwen.MODEL_IDS["qwen3_vl_8b_instruct"]
    assert binding.QWEN8B_EXPECTED["prompt_version"] == qwen.METHOD_VERSION
    assert binding.QWEN8B_REQUIRED.issubset(qwen.COLUMNS)


def test_uploaded_caption_csv_is_hash_locked_and_never_modified(caption_input):
    source, _, frames, folder = caption_input
    before = source.read_bytes()
    cached, details = stage(source, folder / "output/shared/captions.csv", frames)
    assert source.read_bytes() == cached.read_bytes() == before
    assert details["method"] == "qwen3_vl_8b_instruct" and details["rows"] == 6
    assert details["stored_valid_by_split"] == {"train": 1, "val": 1, "test": 1}
    assert details["sha256"] == qwen.file_sha256(source)
    provenance = json.loads((cached.parent / "caption_source.json").read_text())
    assert provenance["generation_needed"] is False
    assert stage(source, cached, frames)[1] == details


def test_automatic_discovery_uses_metadata_not_filename_and_collapses_identical_copies(caption_input):
    source, _, frames, folder = caption_input
    renamed = folder / "kaggle/input/my-captions/captions (8).csv"
    renamed.parent.mkdir(parents=True)
    renamed.write_bytes(source.read_bytes())
    found, details = binding.resolve_qwen8b_caption_csv("", frames, ["banded", "striped"], roots=[folder / "kaggle"])
    assert found == renamed.resolve() and details["rows"] == 6
    found, _ = binding.resolve_qwen8b_caption_csv("", frames, ["banded", "striped"], local_candidates=[source], roots=[folder / "kaggle"])
    assert found == source.resolve()


def test_multiple_different_qwen_csvs_require_explicit_choice(caption_input):
    source, rows, frames, folder = caption_input
    other = folder / "input/captions_new.csv"
    changed = copy.deepcopy(rows)
    changed[0]["source_long_caption"] = "different original response"
    qwen.atomic_csv(other, changed, qwen.COLUMNS)
    with pytest.raises(ValueError, match="Multiple different"):
        binding.resolve_qwen8b_caption_csv("", frames, ["banded", "striped"], roots=[folder / "input"])
    assert binding.resolve_qwen8b_caption_csv(source, frames, ["banded", "striped"])[0] == source.resolve()


@pytest.mark.parametrize("column,value", [
    ("generation_method", "qwen3_vl_32b_instruct"),
    ("vlm_model", "microsoft/Florence-2-base"),
    ("semantic_audit_status", "pending"),
    ("caption_input_policy", "legacy_strict"),
    ("prompt_version", "another-prompt"),
    ("captioner_fingerprint", ""),
    ("audit_fingerprint", ""),
    ("caption_valid", False),
    ("raw_caption", "a longer mismatched original raw caption"),
    ("semantic_leakage_score", "nan"),
])
def test_wrong_or_incomplete_metadata_and_masks_are_rejected(caption_input, column, value):
    source, rows, frames, _ = caption_input
    changed = copy.deepcopy(rows)
    changed[0][column] = value
    qwen.atomic_csv(source, changed, qwen.COLUMNS)
    with pytest.raises(ValueError):
        binding.verify_qwen8b_caption_csv(source, frames, ["banded", "striped"])


def test_partial_duplicate_and_wrong_split_csvs_are_rejected(caption_input):
    source, rows, frames, _ = caption_input
    for invalid in (rows[:-1], rows + rows[:1], [dict(row, split="test") for row in rows]):
        qwen.atomic_csv(source, invalid, qwen.COLUMNS)
        with pytest.raises(ValueError, match="official DTD"):
            binding.verify_qwen8b_caption_csv(source, frames, ["banded", "striped"])


def test_optional_expected_hash_is_checked_without_inventing_one(caption_input):
    source, _, frames, _ = caption_input
    expected = qwen.file_sha256(source)
    assert binding.verify_qwen8b_caption_csv(source, frames, ["banded", "striped"], expected)["sha256"] == expected
    with pytest.raises(ValueError, match="SHA256"):
        binding.verify_qwen8b_caption_csv(source, frames, ["banded", "striped"], "f" * 64)


def test_stale_output_or_provenance_is_rejected_without_overwrite(caption_input):
    source, rows, frames, folder = caption_input
    cached = folder / "output/shared/captions.csv"
    cached.parent.mkdir(parents=True)
    cached.write_text("old Florence data", encoding="utf-8")
    before = cached.read_bytes()
    with pytest.raises(ValueError, match="another caption CSV"):
        stage(source, cached, frames)
    assert cached.read_bytes() == before
    cached.unlink()
    stage(source, cached, frames)
    lock = cached.parent / "caption_source.json"
    previous = json.loads(lock.read_text())
    previous["sha256"] = "wrong"
    lock.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="Changed Qwen8B"):
        stage(source, cached, frames)


@pytest.mark.parametrize("options", [{"policy": "legacy_strict"}, {"generate": True}])
def test_regeneration_and_policy_changes_cannot_silently_change_the_experiment(caption_input, options):
    source, _, frames, folder = caption_input
    with pytest.raises(ValueError, match="precomputed Qwen8B"):
        stage(source, folder / "output/captions.csv", frames, **options)


def test_csv_flows_through_original_policy_into_both_query_and_candidate_banks(caption_input):
    source, _, frames, folder = caption_input
    ns = {"pd": pd, "np": np, "torch": torch, "Path": Path, "os": os}
    exec(code_cells()[8], ns)

    def write_json(path, value):
        Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def cache(views, clip_asset, destination):
        view = views["screened"]
        return {"embedding_dim": 512, "fingerprint": "synthetic-CPU-test-only", "views": {"screened": {
            **view, "embeddings": torch.zeros(len(view["image_ids"]), 512)}}}

    ids = frames["train"].image_id.tolist()
    ns.update(OUT_PUT_DIR=folder, PROJECT_DIR=folder, train_df=frames["train"], val_df=frames["val"],
        test_df=frames["test"], candidate_df=frames["train"], CAPTIONS_INPUT_PATH=str(source),
        CAPTION_INPUT_POLICY="dtd_lexical_v2", GENERATE_CAPTIONS_IF_MISSING=False, EXPECTED_CAPTIONS_SHA256="",
        CAPTION_SEMANTIC_THRESHOLD=.3, CLIP_MODEL_LOCAL_PATH="unused-test-asset", DEVICE=torch.device("cpu"),
        class_names=["banded", "striped"], write_json=write_json, caption_filter_diagnostics=lambda *args: None,
        resolve_clip_path=lambda *args: "unused-test-asset", encode_view_cache=cache, display=lambda *args: None)
    exec(code_cells()[9], ns)
    assert ns["CAPTION_BINDING"]["method"] == "qwen3_vl_8b_instruct"
    assert ns["caption_audit"].screened_valid.tolist() == [True, False, True, False, True, False]
    assert ns["caption_audit"].semantic_score_matches_input.all()
    assert ns["caption_audit"].semantic_leakage_detected.all()
    banks = ns["make_ablation_banks"]({"name": "full_both", "view": "screened", "q_transform": "aligned",
        "c_transform": "aligned", "query_fusion": True, "candidate_fusion": True, "semantic_selector": True}, 42)
    assert banks["c"]["source_ids"] == ids
    assert len(banks["q"]["source_ids"]) == 6
    assert banks["q"]["valid"].tolist() == [True, False, True, False, True, False]
    assert banks["c"]["valid"].tolist() == [True, False]
    assert not set(banks["c"]["source_ids"]) & set(frames["test"].image_id)
    assert (folder / "caption_method.json").is_file()
    assert (folder / "shared/caption_model_inputs.csv").is_file()
    assert (folder / "shared/caption_role_coverage.csv").is_file()
