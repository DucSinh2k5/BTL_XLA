
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
import build_dtd_caption_method03_singlegrounded_notebook as single
import build_dtd_caption_method04_multigrounded_notebook as builder
from test_dtd_caption_method01_prefix_notebook import caption_runtime, catalog
from test_dtd_grounded_caption import actual_view_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: prefix.source_of(cell) for index, cell in enumerate(builder.build_notebook(catalog())["cells"])}


@pytest.fixture
def multi_runtime(caption_runtime):
    ns, _, _, frames, folder = caption_runtime
    selected = copy.deepcopy(catalog()[3])
    nodes = [node for node in ast.parse(code_cells()[8]).body if isinstance(node, ast.FunctionDef)
             and node.name in {"verify_method_caption_csv", "resolve_method_caption_csv", "stage_method_caption_csv"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "multigrounded_input", "exec"), ns)
    return ns, prefix.FOLDER / selected["filename"], selected, frames, folder


def test_reproducible_self_contained_method_four_preserves_all_previous_notebooks_and_csvs():
    protected = [prefix.SOURCE, prefix.TARGET, firstvalid.TARGET, single.TARGET, *prefix.FOLDER.glob("*.csv")]
    before = {path: path.read_bytes() for path in protected}
    notebook = builder.build_notebook(catalog())
    assert notebook == json.loads(builder.TARGET.read_text(encoding="utf-8"))
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(notebook["cells"]) == 24
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-caption-multigrounded-")
        if cell["cell_type"] == "code":
            source = prefix.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert "from dtd_" not in source and "import build_" not in source
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["caption_method"] == "multibeam_grounded"
    assert metadata["caption_csv_sha256"] == catalog()[3]["sha256"]
    assert metadata["caption_display_name"] == "Florence-MultiGrounded"
    assert metadata["experiment"] == "full_both" and metadata["default_epochs"] == 50
    assert not metadata["caption_generation"]


def test_core_model_losses_banks_train_test_and_k_one_remain_unchanged():
    previous = {index: prefix.source_of(cell) for index, cell in enumerate(single.build_notebook(catalog())["cells"])}
    code = code_cells()
    for index in (9, 11, 12, 14, 15, 17, 19, 21):
        assert code[index] == previous[index]
    normalized = code[4].replace("DTD_CAPTION_04_MULTI_GROUNDED_OUTPUT", "DTD_CAPTION_03_SINGLE_GROUNDED_OUTPUT").replace(
        "caption_04_florence_multigrounded_vitl16", "caption_03_florence_singlegrounded_vitl16").replace(
        "DTD_CAPTION_04_MULTI_GROUNDED_EPOCHS", "DTD_CAPTION_03_SINGLE_GROUNDED_EPOCHS")
    assert normalized == previous[4]
    assert 'DTD_CAPTION_04_MULTI_GROUNDED_EPOCHS", "50"' in code[4]
    assert "CONTEXT_TOP_K = 1" in code[4]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert "caption_04_florence_multigrounded_results.csv" in code[23]
    assert "caption_03_florence_singlegrounded" not in "\n".join(code[index] for index in (2, 4, 23))


def test_config_ignores_prior_method_specific_paths_and_seeds(monkeypatch):
    monkeypatch.setenv("DTD_SINGLE_GROUNDED_CAPTIONS_INPUT_PATH", "/wrong/single.csv")
    monkeypatch.setenv("DTD_CAPTION_03_SINGLE_GROUNDED_SEED", "123")
    monkeypatch.setenv("DTD_CAPTION_INPUT_POLICY", "legacy_strict")
    for name in ("DTD_MULTI_GROUNDED_CAPTIONS_INPUT_PATH", "DTD_CAPTION_04_MULTI_GROUNDED_SEED", "DTD_SPLIT"):
        monkeypatch.delenv(name, raising=False)
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["CAPTIONS_INPUT_PATH"] == "" and ns["TRAIN_SEEDS"] == [42]
    assert ns["CAPTION_METHOD_EXPECTED"] == catalog()[3]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    selected = ns["SELECTED_ABLATIONS"][0]
    assert selected["name"] == "full_both" and all(selected[key] for key in ("query_fusion", "candidate_fusion", "semantic_selector"))


def test_correct_multi_csv_binds_and_single_method_is_rejected(multi_runtime):
    ns, source, expected, frames, folder = multi_runtime
    original = source.read_bytes()
    _, binding = ns["stage_method_caption_csv"](str(source), folder / "shared/captions.csv", frames, expected)
    assert binding["method"] == "multibeam_grounded" and binding["rows"] == 5640
    assert source.read_bytes() == original
    other = prefix.FOLDER / catalog()[2]["filename"]
    with pytest.raises(ValueError, match="Florence-MultiGrounded"):
        ns["verify_method_caption_csv"](other, frames, expected)
    altered = {**expected, "sha256": catalog()[2]["sha256"]}
    with pytest.raises(ValueError, match="generation_method"):
        ns["verify_method_caption_csv"](other, frames, altered)


def test_discovery_and_cache_guard_keep_the_fourth_method_isolated(multi_runtime):
    ns, source, expected, frames, folder = multi_runtime
    found = ns["resolve_method_caption_csv"]("", expected, roots=[prefix.FOLDER])
    assert found == source.resolve()
    cached = folder / "shared/captions.csv"
    cached.parent.mkdir()
    cached.write_bytes((prefix.FOLDER / catalog()[2]["filename"]).read_bytes())
    before = cached.read_bytes()
    with pytest.raises(ValueError, match="another caption CSV"):
        ns["stage_method_caption_csv"](str(source), cached, frames, expected)
    assert cached.read_bytes() == before


def test_uploaded_multi_csv_uses_added_hypotheses_with_the_same_coverage_and_image_ranking():
    single_rows = pd.read_csv(prefix.FOLDER / catalog()[2]["filename"]).fillna("")
    multi = pd.read_csv(prefix.FOLDER / catalog()[3]["filename"]).fillna("")
    assert multi.image_id.equals(single_rows.image_id) and multi.split.equals(single_rows.split)
    assert multi.caption_valid.equals(single_rows.caption_valid) and multi.caption_valid.all()
    assert multi.source_hypothesis.between(0, 3).all() and multi.source_hypothesis.gt(0).any()
    assert multi.final_caption.ne(single_rows.final_caption).any()
    # Higher optimized ranking scores do not establish improved independent quality or F1.
    assert (multi.image_text_similarity >= single_rows.image_text_similarity - 1e-6).all()


def test_real_multi_csv_passes_the_existing_audit_and_both_text_banks(multi_runtime):
    ns, source, expected, frames, folder = multi_runtime
    ns.update(actual_view_runtime())
    rows = pd.read_csv(source).fillna("")
    class_names = sorted({image_id.split("/")[0] for image_id in rows.image_id})

    def cache(views, clip_asset, output):
        view = views["screened"]
        return {"embedding_dim": 512, "fingerprint": "synthetic-multigrounded-test-only",
                "views": {"screened": {**view, "embeddings": torch.zeros(len(view["image_ids"]), 512)}}}

    ns.update(OUT_PUT_DIR=folder, PROJECT_DIR=folder, train_df=frames["train"], val_df=frames["val"],
        test_df=frames["test"], candidate_df=frames["train"].iloc[:376], CAPTIONS_INPUT_PATH=str(source),
        CAPTION_METHOD_EXPECTED=expected, CAPTION_SEMANTIC_THRESHOLD=.3, CLIP_MODEL_LOCAL_PATH="unused-test-asset",
        DEVICE=torch.device("cpu"), class_names=class_names, pd=pd, torch=torch,
        load_ablation_captions=lambda path, splits: pd.read_csv(path).fillna(""),
        caption_filter_diagnostics=lambda *args: None, resolve_clip_path=lambda *args: "unused-test-asset",
        encode_view_cache=cache, display=lambda *args: None)
    exec(code_cells()[9], ns)
    assert ns["CAPTION_BINDING"]["method"] == "multibeam_grounded"
    assert int(ns["caption_audit"].screened_valid.sum()) == 5640
    spec = {"name": "full_both", "view": "screened", "q_transform": "aligned", "c_transform": "aligned",
            "query_fusion": True, "candidate_fusion": True, "semantic_selector": True}
    banks = ns["make_ablation_banks"](spec, 42)
    assert len(banks["c"]["source_ids"]) == 376 and len(banks["q"]["source_ids"]) == 5640
    assert banks["q"]["valid"].all() and banks["c"]["valid"].all()
    assert set(banks["c"]["source_ids"]) <= set(frames["train"].image_id)
    assert not set(banks["c"]["source_ids"]) & set(frames["test"].image_id)


def test_manifest_contains_all_four_created_notebooks():
    manifest = json.loads((prefix.FOLDER / "caption_methods_manifest.json").read_text(encoding="utf-8"))
    created = manifest["created_training_notebooks"]
    for method, target in (("prefix_control", prefix.TARGET), ("first_valid_control", firstvalid.TARGET),
                           ("single_grounded_control", single.TARGET), ("multibeam_grounded", builder.TARGET)):
        assert created[method] == target.relative_to(PROJECT_ROOT).as_posix()
        assert target.is_file()
    assert manifest["no_training_executed"]
