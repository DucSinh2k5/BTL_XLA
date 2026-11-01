
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
import build_dtd_caption_method02_firstvalid_notebook as firstvalid
import build_dtd_caption_method03_singlegrounded_notebook as builder
from test_dtd_caption_method01_prefix_notebook import caption_runtime, catalog
from test_dtd_grounded_caption import actual_view_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: prefix.source_of(cell) for index, cell in enumerate(builder.build_notebook(catalog())["cells"])}


@pytest.fixture
def single_runtime(caption_runtime):
    ns, _, _, frames, folder = caption_runtime
    selected = copy.deepcopy(catalog()[2])
    nodes = [node for node in ast.parse(code_cells()[8]).body if isinstance(node, ast.FunctionDef)
             and node.name in {"verify_method_caption_csv", "resolve_method_caption_csv", "stage_method_caption_csv"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "single_grounded_input", "exec"), ns)
    return ns, prefix.FOLDER / selected["filename"], selected, frames, folder


def test_reproducible_self_contained_method_three_preserves_previous_notebooks_and_csvs():
    protected = [prefix.SOURCE, prefix.TARGET, firstvalid.TARGET, *prefix.FOLDER.glob("*.csv")]
    before = {path: path.read_bytes() for path in protected}
    notebook = builder.build_notebook(catalog())
    assert notebook == json.loads(builder.TARGET.read_text(encoding="utf-8"))
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(notebook["cells"]) == 24
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-caption-singlegrounded-")
        if cell["cell_type"] == "code":
            source = prefix.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert "from dtd_" not in source and "import build_" not in source
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["caption_method"] == "single_grounded_control"
    assert metadata["caption_csv_sha256"] == catalog()[2]["sha256"]
    assert metadata["caption_display_name"] == "Florence-SingleGrounded"
    assert metadata["experiment"] == "full_both" and metadata["default_epochs"] == 50
    assert not metadata["caption_generation"]


def test_model_fusion_losses_train_test_and_caption_banks_match_firstvalid():
    previous = {index: prefix.source_of(cell) for index, cell in enumerate(firstvalid.build_notebook(catalog())["cells"])}
    code = code_cells()
    for index in (9, 11, 12, 14, 15, 17, 19, 21):
        assert code[index] == previous[index]
    normalized = code[4].replace("DTD_CAPTION_03_SINGLE_GROUNDED_OUTPUT", "DTD_CAPTION_02_FIRST_VALID_OUTPUT").replace(
        "caption_03_florence_singlegrounded_vitl16", "caption_02_florence_firstvalid_vitl16").replace(
        "DTD_CAPTION_03_SINGLE_GROUNDED_EPOCHS", "DTD_CAPTION_02_FIRST_VALID_EPOCHS")
    assert normalized == previous[4]
    assert 'DTD_CAPTION_03_SINGLE_GROUNDED_EPOCHS", "50"' in code[4]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert "caption_03_florence_singlegrounded_results.csv" in code[23]
    assert "caption_02_florence_firstvalid" not in "\n".join(code[index] for index in (2, 4, 23))


def test_config_uses_own_seed_path_policy_and_both_text_roles(monkeypatch):
    for name, value in (("DTD_FIRST_VALID_CAPTIONS_INPUT_PATH", "/wrong/first.csv"),
                        ("DTD_CAPTION_02_FIRST_VALID_SEED", "123"), ("DTD_CAPTION_INPUT_POLICY", "legacy_strict")):
        monkeypatch.setenv(name, value)
    for name in ("DTD_SINGLE_GROUNDED_CAPTIONS_INPUT_PATH", "DTD_CAPTION_03_SINGLE_GROUNDED_SEED", "DTD_SPLIT"):
        monkeypatch.delenv(name, raising=False)
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["CAPTIONS_INPUT_PATH"] == "" and ns["TRAIN_SEEDS"] == [42]
    assert ns["CAPTION_METHOD_EXPECTED"] == catalog()[2]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    selected = ns["SELECTED_ABLATIONS"][0]
    assert selected["name"] == "full_both" and all(selected[key] for key in ("query_fusion", "candidate_fusion", "semantic_selector"))


def test_correct_single_csv_binds_while_firstvalid_is_rejected(single_runtime):
    ns, source, expected, frames, folder = single_runtime
    original = source.read_bytes()
    _, binding = ns["stage_method_caption_csv"](str(source), folder / "shared/captions.csv", frames, expected)
    assert binding["method"] == "single_grounded_control" and binding["rows"] == 5640
    assert source.read_bytes() == original
    other = prefix.FOLDER / catalog()[1]["filename"]
    with pytest.raises(ValueError, match="Florence-SingleGrounded"):
        ns["verify_method_caption_csv"](other, frames, expected)
    altered = {**expected, "sha256": catalog()[1]["sha256"]}
    with pytest.raises(ValueError, match="generation_method"):
        ns["verify_method_caption_csv"](other, frames, altered)


def test_discovery_and_stale_cache_guard_do_not_fall_back_to_another_method(single_runtime):
    ns, source, expected, frames, folder = single_runtime
    found = ns["resolve_method_caption_csv"]("", expected, roots=[prefix.FOLDER])
    assert found == source.resolve()
    cached = folder / "shared/captions.csv"
    cached.parent.mkdir()
    cached.write_bytes((prefix.FOLDER / catalog()[1]["filename"]).read_bytes())
    before = cached.read_bytes()
    with pytest.raises(ValueError, match="another caption CSV"):
        ns["stage_method_caption_csv"](str(source), cached, frames, expected)
    assert cached.read_bytes() == before


def test_uploaded_control_really_uses_same_greedy_sources_and_coverage_with_image_ranking():
    first = pd.read_csv(prefix.FOLDER / catalog()[1]["filename"]).fillna("")
    single = pd.read_csv(prefix.FOLDER / catalog()[2]["filename"]).fillna("")
    assert first.image_id.equals(single.image_id) and first.split.equals(single.split)
    assert first.source_long_caption.equals(single.source_long_caption)
    assert single.source_hypothesis.eq(0).all() and first.caption_valid.equals(single.caption_valid)
    assert first.caption_valid.all() and single.final_caption.ne(first.final_caption).any()
    # Ranking is optimized by this control, not an independent quality assessment.
    assert (single.image_text_similarity >= first.image_text_similarity - 1e-6).all()


def test_real_single_csv_flows_through_existing_audit_and_both_banks(single_runtime):
    ns, source, expected, frames, folder = single_runtime
    ns.update(actual_view_runtime())
    rows = pd.read_csv(source).fillna("")
    class_names = sorted({image_id.split("/")[0] for image_id in rows.image_id})

    def cache(views, clip_asset, output):
        view = views["screened"]
        return {"embedding_dim": 512, "fingerprint": "synthetic-single-grounded-test-only",
                "views": {"screened": {**view, "embeddings": torch.zeros(len(view["image_ids"]), 512)}}}

    ns.update(OUT_PUT_DIR=folder, PROJECT_DIR=folder, train_df=frames["train"], val_df=frames["val"],
        test_df=frames["test"], candidate_df=frames["train"].iloc[:376], CAPTIONS_INPUT_PATH=str(source),
        CAPTION_METHOD_EXPECTED=expected, CAPTION_SEMANTIC_THRESHOLD=.3, CLIP_MODEL_LOCAL_PATH="unused-test-asset",
        DEVICE=torch.device("cpu"), class_names=class_names, pd=pd, torch=torch,
        load_ablation_captions=lambda path, splits: pd.read_csv(path).fillna(""),
        caption_filter_diagnostics=lambda *args: None, resolve_clip_path=lambda *args: "unused-test-asset",
        encode_view_cache=cache, display=lambda *args: None)
    exec(code_cells()[9], ns)
    assert ns["CAPTION_BINDING"]["method"] == "single_grounded_control"
    assert int(ns["caption_audit"].screened_valid.sum()) == 5640
    spec = {"name": "full_both", "view": "screened", "q_transform": "aligned", "c_transform": "aligned",
            "query_fusion": True, "candidate_fusion": True, "semantic_selector": True}
    banks = ns["make_ablation_banks"](spec, 42)
    assert len(banks["c"]["source_ids"]) == 376 and len(banks["q"]["source_ids"]) == 5640
    assert banks["q"]["valid"].all() and banks["c"]["valid"].all()
    assert set(banks["c"]["source_ids"]) <= set(frames["train"].image_id)
    assert not set(banks["c"]["source_ids"]) & set(frames["test"].image_id)


def test_manifest_retains_all_three_created_notebooks():
    manifest = json.loads((prefix.FOLDER / "caption_methods_manifest.json").read_text(encoding="utf-8"))
    created = manifest["created_training_notebooks"]
    assert created["prefix_control"] == prefix.TARGET.relative_to(PROJECT_ROOT).as_posix()
    assert created["first_valid_control"] == firstvalid.TARGET.relative_to(PROJECT_ROOT).as_posix()
    assert created["single_grounded_control"] == builder.TARGET.relative_to(PROJECT_ROOT).as_posix()
    assert manifest["no_training_executed"]
