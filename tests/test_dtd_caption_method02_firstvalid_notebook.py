
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import copy
import json
import os
from functools import lru_cache

import pandas as pd
import pytest
import torch

import build_dtd_caption_method01_prefix_notebook as prefix
import build_dtd_caption_method02_firstvalid_notebook as builder
from test_dtd_caption_method01_prefix_notebook import caption_runtime, catalog
from test_dtd_grounded_caption import actual_view_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: prefix.source_of(cell) for index, cell in enumerate(builder.build_notebook(catalog())["cells"])}


@pytest.fixture
def firstvalid_runtime(caption_runtime):
    ns, _, _, frames, folder = caption_runtime
    selected = copy.deepcopy(catalog()[1])
    nodes = [node for node in ast.parse(code_cells()[8]).body if isinstance(node, ast.FunctionDef)
             and node.name in {"verify_method_caption_csv", "resolve_method_caption_csv", "stage_method_caption_csv"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "firstvalid_input", "exec"), ns)
    return ns, prefix.FOLDER / selected["filename"], selected, frames, folder


def test_notebook_is_reproducible_and_does_not_modify_prefix_or_any_original_csv():
    protected = [prefix.SOURCE, prefix.TARGET, *prefix.FOLDER.glob("*.csv")]
    before = {path: path.read_bytes() for path in protected}
    notebook = builder.build_notebook(catalog())
    assert notebook == json.loads(builder.TARGET.read_text(encoding="utf-8"))
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(notebook["cells"]) == 24
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = prefix.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert "from dtd_" not in source and "import build_" not in source
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["caption_method"] == "first_valid_control"
    assert metadata["caption_csv_sha256"] == catalog()[1]["sha256"]
    assert metadata["caption_display_name"] == "Florence-FirstValid"
    assert metadata["experiment"] == "full_both" and metadata["default_epochs"] == 50
    assert not metadata["caption_generation"]


def test_model_losses_audit_and_train_test_logic_are_identical_to_method_one():
    previous = {index: prefix.source_of(cell) for index, cell in enumerate(prefix.build_notebook(catalog())["cells"])}
    code = code_cells()
    for index in (9, 11, 12, 14, 15, 17, 19, 21):
        assert code[index] == previous[index]
    normalized = code[4].replace("DTD_CAPTION_02_FIRST_VALID_OUTPUT", "DTD_CAPTION_01_PREFIX_OUTPUT").replace(
        "caption_02_florence_firstvalid_vitl16", "caption_01_florence_prefix_vitl16").replace(
        "DTD_CAPTION_02_FIRST_VALID_EPOCHS", "DTD_CAPTION_01_PREFIX_EPOCHS")
    assert normalized == previous[4]
    assert 'DTD_CAPTION_02_FIRST_VALID_EPOCHS", "50"' in code[4]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert "caption_02_florence_firstvalid_results.csv" in code[23]
    assert "caption_01_florence_prefix" not in "\n".join(code[index] for index in (2, 4, 23))


def test_method_specific_config_ignores_prefix_and_historical_input_environment(monkeypatch):
    monkeypatch.setenv("DTD_PREFIX_CAPTIONS_INPUT_PATH", "/wrong/prefix.csv")
    monkeypatch.setenv("DTD_CAPTIONS_INPUT_PATH", "/wrong/history.csv")
    monkeypatch.setenv("DTD_CAPTION_01_PREFIX_SEED", "123")
    monkeypatch.delenv("DTD_FIRST_VALID_CAPTIONS_INPUT_PATH", raising=False)
    monkeypatch.delenv("DTD_CAPTION_02_FIRST_VALID_SEED", raising=False)
    monkeypatch.delenv("DTD_SPLIT", raising=False)
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["CAPTIONS_INPUT_PATH"] == "" and ns["TRAIN_SEEDS"] == [42]
    assert ns["CAPTION_METHOD_EXPECTED"] == catalog()[1]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    selected = ns["SELECTED_ABLATIONS"][0]
    assert selected["name"] == "full_both" and selected["query_fusion"] and selected["candidate_fusion"] and selected["semantic_selector"]


def test_correct_firstvalid_csv_binds_and_prefix_is_rejected(firstvalid_runtime):
    ns, source, expected, frames, folder = firstvalid_runtime
    original = source.read_bytes()
    _, binding = ns["stage_method_caption_csv"](str(source), folder / "shared/captions.csv", frames, expected)
    assert binding["method"] == "first_valid_control" and binding["rows"] == 5640
    assert source.read_bytes() == original
    other = prefix.FOLDER / catalog()[0]["filename"]
    with pytest.raises(ValueError, match="Florence-FirstValid"):
        ns["verify_method_caption_csv"](other, frames, expected)
    altered = {**expected, "sha256": catalog()[0]["sha256"]}
    with pytest.raises(ValueError, match="generation_method"):
        ns["verify_method_caption_csv"](other, frames, altered)


def test_discovery_finds_only_the_second_csv_and_stale_prefix_cache_is_not_overwritten(firstvalid_runtime):
    ns, source, expected, frames, folder = firstvalid_runtime
    source_prefix = prefix.FOLDER / catalog()[0]["filename"]
    found = ns["resolve_method_caption_csv"]("", expected, roots=[prefix.FOLDER])
    assert found == source.resolve()
    cached = folder / "shared/captions.csv"
    cached.parent.mkdir()
    cached.write_bytes(source_prefix.read_bytes())
    before = cached.read_bytes()
    with pytest.raises(ValueError, match="another caption CSV"):
        ns["stage_method_caption_csv"](str(source), cached, frames, expected)
    assert cached.read_bytes() == before


def test_real_firstvalid_csv_passes_the_existing_audit_and_both_banks(firstvalid_runtime):
    ns, source, expected, frames, folder = firstvalid_runtime
    ns.update(actual_view_runtime())
    rows = pd.read_csv(source).fillna("")
    class_names = sorted({image_id.split("/")[0] for image_id in rows.image_id})

    def cache(views, clip_asset, output):
        view = views["screened"]
        return {"embedding_dim": 512, "fingerprint": "synthetic-firstvalid-test-only",
                "views": {"screened": {**view, "embeddings": torch.zeros(len(view["image_ids"]), 512)}}}

    ns.update(OUT_PUT_DIR=folder, PROJECT_DIR=folder, train_df=frames["train"], val_df=frames["val"],
        test_df=frames["test"], candidate_df=frames["train"].iloc[:376], CAPTIONS_INPUT_PATH=str(source),
        CAPTION_METHOD_EXPECTED=expected, CAPTION_SEMANTIC_THRESHOLD=.3, CLIP_MODEL_LOCAL_PATH="unused-test-asset",
        DEVICE=torch.device("cpu"), class_names=class_names, pd=pd, torch=torch,
        load_ablation_captions=lambda path, splits: pd.read_csv(path).fillna(""),
        caption_filter_diagnostics=lambda *args: None, resolve_clip_path=lambda *args: "unused-test-asset",
        encode_view_cache=cache, display=lambda *args: None)
    exec(code_cells()[9], ns)
    assert ns["CAPTION_BINDING"]["method"] == "first_valid_control"
    assert int(ns["caption_audit"].screened_valid.sum()) == 5640
    assert not rows.final_caption.equals(pd.read_csv(prefix.FOLDER / catalog()[0]["filename"]).final_caption)
    spec = {"name": "full_both", "view": "screened", "q_transform": "aligned", "c_transform": "aligned",
            "query_fusion": True, "candidate_fusion": True, "semantic_selector": True}
    banks = ns["make_ablation_banks"](spec, 42)
    assert len(banks["c"]["source_ids"]) == 376 and len(banks["q"]["source_ids"]) == 5640
    assert banks["q"]["valid"].all() and banks["c"]["valid"].all()
    assert set(banks["c"]["source_ids"]) <= set(frames["train"].image_id)
    assert not set(banks["c"]["source_ids"]) & set(frames["test"].image_id)


def test_manifest_lists_both_created_notebooks_and_does_not_claim_training_metrics():
    manifest = json.loads((prefix.FOLDER / "caption_methods_manifest.json").read_text(encoding="utf-8"))
    created = manifest["created_training_notebooks"]
    assert created["prefix_control"] == prefix.TARGET.relative_to(PROJECT_ROOT).as_posix()
    assert created["first_valid_control"] == builder.TARGET.relative_to(PROJECT_ROOT).as_posix()
    assert manifest["no_training_executed"]
