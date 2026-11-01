"""Paired 32B caption binding without changing the original or Qwen8B classifier."""
import ast
import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import build_dtd_caption_qwen8b_full_both_notebook as previous
import build_dtd_caption_qwen32b_full_both_notebook as builder
import dtd_qwen_caption as qwen


TEXT = "rough gray grains cover pale uneven surface"


@lru_cache(maxsize=1)
def code_cells():
    return {index: builder.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


@pytest.fixture
def runtime():
    ns = {"np": np, "pd": pd, "torch": torch, "Path": Path, "os": os}
    exec(code_cells()[8], ns)
    return ns


@pytest.fixture
def captions(tmp_path):
    records = [{"image_id": f"banded/{role}_{index}.jpg", "split": role}
               for role in ("train", "val", "test") for index in range(2)]
    frames = {role: pd.DataFrame([row for row in records if row["split"] == role]) for role in ("train", "val", "test")}
    rows = []
    for index, record in enumerate(records):
        text = "striped red fabric covers rough dark stone" if index % 2 else TEXT
        generated = qwen.generation_row(record, text, "qwen3_vl_32b_instruct", "c" * 64, ["banded", "striped"])
        rows.append(qwen.audited_row(generated, .95, "banded", "d" * 64, .30))
    source = tmp_path / "input32/captions.csv"
    qwen.atomic_csv(source, rows, qwen.COLUMNS)
    return source, rows, frames, tmp_path


def test_notebook_is_fresh_standalone_and_preserves_original_and_8b_files():
    protected = {path: path.read_bytes() for path in (builder.SOURCE, previous.TARGET)}
    notebook = builder.build_notebook()
    assert all(path.read_bytes() == content for path, content in protected.items())
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert len(notebook["cells"]) == 24
    base = previous.build_notebook()
    original = json.loads(protected[builder.SOURCE])
    for index in (11, 12, 14, 15, 19, 21):
        assert builder.source_of(notebook["cells"][index]) == builder.source_of(base["cells"][index])
        assert builder.source_of(notebook["cells"][index]) == builder.source_of(original["cells"][index])
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-caption-qwen32b-")
        if cell["cell_type"] == "code":
            source = builder.source_of(cell)
            ast.parse(source)
            assert not cell["outputs"] and cell["execution_count"] is None
            for forbidden in ("from dtd_", "import build_", "def load_florence", "Qwen3VLForConditionalGeneration",
                              "CAPTION_WORKER_SOURCE", "pip install", "subprocess.Popen", "QWEN8B_", "qwen8b_"):
                assert forbidden not in source
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "full_both" and metadata["caption_method"] == "qwen3_vl_32b_instruct"
    assert not metadata["caption_generation"] and not metadata["captioner_finetuning"]
    assert metadata["default_epochs"] == 50 and metadata["default_caption_input_policy"] == "dtd_lexical_v2"


def test_settings_only_change_artifact_and_environment_names():
    original = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    normalized = code_cells()[4].replace("DTD_QWEN32B_FULL_BOTH_OUTPUT", "DTD_FULL_BOTH_OUTPUT").replace(
        "caption_06_qwen3vl_32b_vitl16", "ablation_05_full_both_vitl16").replace(
        "DTD_QWEN32B_FULL_BOTH_EPOCHS", "DTD_FULL_BOTH_EPOCHS")
    assert normalized == builder.source_of(original["cells"][4])
    assert 'DTD_QWEN32B_FULL_BOTH_EPOCHS", "50"' in code_cells()[4]
    assert "CONTEXT_TOP_K = 1" in code_cells()[4]
    assert "caption_06_qwen3vl_32b_results.csv" in code_cells()[23]
    assert "caption_06_qwen3vl_32b_report.md" in code_cells()[23]
    assert '"schema": "dtd_qwen3vl32b_full_both_v1"' in code_cells()[17]
    assert '"caption_method": CAPTION_BINDING' in code_cells()[17]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[1]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[19]


def test_helper_checksum_is_recomputed_for_32b_and_differs_from_8b():
    ns = {}
    assignments = [node for node in ast.parse(code_cells()[6]).body if isinstance(node, ast.Assign)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "checksums", "exec"), ns)
    helpers = "".join(code_cells()[index] for index in (8, 9, 11, 12, 14, 15))
    assert ns["HELPER_SOURCE_SHA256"] == hashlib.sha256(helpers.encode()).hexdigest()
    assert ns["SOURCE_NOTEBOOK_SHA256"] == hashlib.sha256(builder.SOURCE.read_bytes()).hexdigest()
    assert ns["HELPER_SOURCE_SHA256"] not in builder.source_of(previous.build_notebook()["cells"][6])


def test_32b_config_does_not_adopt_8b_inputs_seeds_or_caches(monkeypatch):
    for name in ("DTD_QWEN32B_FULL_BOTH_SEED", "DTD_QWEN32B_CAPTIONS_INPUT_PATH", "DTD_QWEN32B_CAPTIONS_SHA256", "DTD_SPLIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DTD_QWEN8B_FULL_BOTH_SEED", "43")
    monkeypatch.setenv("DTD_QWEN8B_CAPTIONS_INPUT_PATH", "/wrong/8b.csv")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["TRAIN_SEEDS"] == [42] and ns["CAPTIONS_INPUT_PATH"] == ""
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    assert not ns["EARLY_STOPPING"]
    spec = ns["SELECTED_ABLATIONS"]
    assert len(spec) == 1 and spec[0]["name"] == "full_both"
    assert all(spec[0][key] for key in ("query_fusion", "candidate_fusion", "semantic_selector"))
    monkeypatch.setenv("DTD_SPLIT", "2")
    with pytest.raises(ValueError, match="split 1"):
        exec(code_cells()[2], {"os": os})


def test_32b_csv_is_accepted_and_source_preserved(runtime, captions):
    source, _, frames, folder = captions
    before = source.read_bytes()
    cached, details = runtime["stage_qwen32b_caption_csv"](source, folder / "output/shared/captions.csv",
        frames, ["banded", "striped"], "dtd_lexical_v2", False)
    assert before == source.read_bytes() == cached.read_bytes()
    assert details["method"] == "qwen3_vl_32b_instruct" and details["vlm_model"] == qwen.MODEL_IDS["qwen3_vl_32b_instruct"]
    assert details["prompt_version"] == qwen.METHOD_VERSION and details["rows"] == 6
    assert details["stored_valid_by_split"] == {"train": 1, "val": 1, "test": 1}
    assert json.loads((cached.parent / "caption_source.json").read_text())["generation_needed"] is False


def test_8b_csv_is_rejected_and_auto_discovery_selects_32b_when_both_attached(runtime, captions):
    source, rows, frames, folder = captions
    other = folder / "input8/captions.csv"
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(generation_method="qwen3_vl_8b_instruct", vlm_model="Qwen/Qwen3-VL-8B-Instruct")
    qwen.atomic_csv(other, changed, qwen.COLUMNS)
    with pytest.raises(ValueError, match="not Florence/8B/partial"):
        runtime["verify_qwen32b_caption_csv"](other, frames, ["banded", "striped"])
    found, details = runtime["resolve_qwen32b_caption_csv"]("", frames, ["banded", "striped"], roots=[folder])
    assert found == source.resolve() and details["method"] == "qwen3_vl_32b_instruct"


@pytest.mark.parametrize("kind", ["partial", "duplicate", "wrong_split", "pending_audit", "wrong_hash", "bad_mask"])
def test_incomplete_wrong_or_modified_32b_exports_are_rejected(runtime, captions, kind):
    source, rows, frames, _ = captions
    changed = copy.deepcopy(rows)
    if kind == "partial":
        changed.pop()
    elif kind == "duplicate":
        changed.append(changed[0])
    elif kind == "wrong_split":
        changed[0]["split"] = "test"
    elif kind == "pending_audit":
        changed[0]["semantic_audit_status"] = "pending"
    elif kind == "bad_mask":
        changed[0]["caption_valid"] = False
    qwen.atomic_csv(source, changed, qwen.COLUMNS)
    with pytest.raises(ValueError):
        runtime["verify_qwen32b_caption_csv"](source, frames, ["banded", "striped"], "a" * 64 if kind == "wrong_hash" else "")


def test_multiple_32b_runs_require_explicit_selection(runtime, captions):
    source, rows, frames, folder = captions
    rows[0]["source_long_caption"] = "another response"
    qwen.atomic_csv(folder / "input_new/captions.csv", rows, qwen.COLUMNS)
    with pytest.raises(ValueError, match="Multiple different Qwen32B"):
        runtime["resolve_qwen32b_caption_csv"]("", frames, ["banded", "striped"], roots=[folder])


def test_stale_cache_cannot_be_overwritten(runtime, captions):
    source, _, frames, folder = captions
    cached = folder / "output/shared/captions.csv"
    cached.parent.mkdir(parents=True)
    cached.write_text("previous 8B output", encoding="utf-8")
    before = cached.read_bytes()
    with pytest.raises(ValueError, match="another caption CSV"):
        runtime["stage_qwen32b_caption_csv"](source, cached, frames, ["banded", "striped"], "dtd_lexical_v2", False)
    assert cached.read_bytes() == before


@pytest.mark.parametrize("policy,generate", [("legacy_strict", False), ("dtd_lexical_v2", True)])
def test_regeneration_and_policy_changes_are_disabled(runtime, captions, policy, generate):
    source, _, frames, folder = captions
    with pytest.raises(ValueError, match="precomputed Qwen32B"):
        runtime["stage_qwen32b_caption_csv"](source, folder / "output/captions.csv", frames, ["banded", "striped"], policy, generate)


def test_32b_text_and_natural_masks_reach_both_roles_with_original_policy(runtime, captions):
    source, _, frames, folder = captions

    def write_json(path, value):
        Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def cache(views, clip_asset, destination):
        view = views["screened"]
        return {"embedding_dim": 512, "fingerprint": "synthetic-CPU-test-only", "views": {"screened": {
            **view, "embeddings": torch.zeros(len(view["image_ids"]), 512)}}}

    runtime.update(OUT_PUT_DIR=folder, PROJECT_DIR=folder, train_df=frames["train"], val_df=frames["val"],
        test_df=frames["test"], candidate_df=frames["train"], CAPTIONS_INPUT_PATH=str(source),
        CAPTION_INPUT_POLICY="dtd_lexical_v2", GENERATE_CAPTIONS_IF_MISSING=False, EXPECTED_CAPTIONS_SHA256="",
        CAPTION_SEMANTIC_THRESHOLD=.3, CLIP_MODEL_LOCAL_PATH="unused", DEVICE=torch.device("cpu"),
        class_names=["banded", "striped"], write_json=write_json, caption_filter_diagnostics=lambda *args: None,
        resolve_clip_path=lambda *args: "unused", encode_view_cache=cache, display=lambda *args: None)
    exec(code_cells()[9], runtime)
    assert runtime["CAPTION_BINDING"]["method"] == "qwen3_vl_32b_instruct"
    assert runtime["caption_audit"].screened_valid.tolist() == [True, False, True, False, True, False]
    assert runtime["caption_audit"].semantic_leakage_detected.all()
    assert runtime["caption_audit"].semantic_score_matches_input.all()
    banks = runtime["make_ablation_banks"]({"name": "full_both", "view": "screened", "q_transform": "aligned",
        "c_transform": "aligned", "query_fusion": True, "candidate_fusion": True, "semantic_selector": True}, 42)
    assert banks["q"]["valid"].tolist() == [True, False, True, False, True, False]
    assert banks["c"]["source_ids"] == frames["train"].image_id.tolist()
    assert banks["c"]["valid"].tolist() == [True, False]
    assert not set(banks["c"]["source_ids"]) & set(frames["test"].image_id)
    assert (folder / "shared/caption_model_inputs.csv").is_file()
    assert (folder / "caption_method.json").is_file()
