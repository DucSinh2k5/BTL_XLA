
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import json
import os
import sys
from functools import lru_cache
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import build_mtacs_dtd_candidate_only_notebook as builder
import build_mtacs_dtd_query_only_notebook as query
import build_mtacs_dtd_visual_only_notebook as visual
import mtacs_dtd_ablation_cells as original
from test_dtd_query_only_notebook import audited_rows, runtime as query_runtime
from test_dtd_text_ablation import spec, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: visual.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = query_runtime(toy_runtime)
    code = code_cells()
    exec(code[2], ns)
    nodes = [node for node in ast.parse(code[9]).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "candidate_banks", "exec"), ns)
    exec(code[14], ns)
    ns.update({"CLIP_EMBED_DIM": 4, "PROTOCOL_FINGERPRINT": "candidate-test"})
    return ns


def test_notebook_is_single_candidate_only_run_and_keeps_existing_artifacts_unchanged(monkeypatch):
    before = query.TARGET.read_bytes()
    notebook = builder.build_notebook()
    assert query.TARGET.read_bytes() == before
    assert len(notebook["cells"]) == 24
    all_code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = visual.source_of(cell)
            ast.parse(text)
            assert cell["execution_count"] is None and cell["outputs"] == []
            all_code.append(text)
    ast.parse("\n".join(all_code))
    assert 'DTD_CANDIDATE_ONLY_EPOCHS", "50"' in "\n".join(all_code)
    assert "_ablation_03_candidate_only_vitl16" in code_cells()[4]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[1]
    assert "validation_interventions" not in "\n".join(all_code)
    for required in ("load_florence", "generate_batch", "ensure_dtd_caption_csv", "--no-index", "--target"):
        assert required in "\n".join(all_code)
    assert notebook["metadata"]["dtd_ablation"]["caption_generation"]
    monkeypatch.setenv("DTD_ABLATIONS", "query_only,full_both")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert len(ns["SELECTED_ABLATIONS"]) == 1
    mode = ns["SELECTED_ABLATIONS"][0]
    assert mode["name"] == "candidate_only" and mode["candidate_fusion"]
    assert not mode["query_fusion"] and not mode["semantic_selector"]
    assert ns["TRAIN_SEEDS"] == [42] and not ns["EARLY_STOPPING"]


def test_architecture_losses_and_augmentation_are_unchanged():
    code = code_cells()
    second = query.build_notebook()
    for index in (4, 11, 12):
        if index == 4:
            source = visual.source_of(second["cells"][index])
            for name in ("train_transform", "eval_transform"):
                node = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
                assert ast.get_source_segment(source, node) in code[index]
        else:
            assert code[index].strip() == visual.source_of(second["cells"][index]).strip()
    names = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu"]
    assert visual.definitions(code[14], names) == visual.definitions(visual.source_of(second["cells"][14]), names)


def test_candidate_vectors_are_reordered_to_pool_and_query_inputs_stay_zero(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    source = ns["TEXT_CACHE"]["views"]["screened"]
    order = torch.tensor([2, 0, 1])
    ns["TEXT_CACHE"] = {"views": {"screened": {
        "image_ids": [source["image_ids"][int(i)] for i in order],
        "texts": [source["texts"][int(i)] for i in order],
        "embeddings": source["embeddings"][order], "valid": source["valid"][order],
    }}}
    banks = ns["make_ablation_banks"](mode, 42)
    assert banks["c"]["source_ids"] == ns["CANDIDATE_IDS"]
    torch.testing.assert_close(banks["c"]["embeddings"], source["embeddings"][:3])
    assert len(banks["q"]["source_ids"]) == 18
    assert not banks["q"]["valid"].any() and banks["q"]["embeddings"].count_nonzero() == 0
    assert set(banks["q"]["texts"]) == {""}
    with pytest.raises(ValueError, match="candidate_only only"):
        ns["make_ablation_banks"](mode, 42, {"query_fusion": True})
    ns["TEXT_CACHE"]["views"]["screened"]["image_ids"][0] = "outside/pool"
    with pytest.raises(ValueError, match="missing"):
        ns["make_ablation_banks"](mode, 42)


def test_context_text_affects_fusion_and_gradients_but_never_visual_selection(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](mode, 42)
    network = ns["create_ablation_model"](42).eval()
    images = torch.randn(2, 3, 2, 2)
    q_text, q_valid = ns["batch_caption_inputs"](banks, ["train/0", "train/1"])
    c_text, c_valid = banks["c"]["embeddings"][:2], banks["c"]["valid"][:2]
    feature_bank = ns["refresh_ablation_bank"](network)
    excluded = torch.tensor([0, 1])
    selected = network.select_context(images, feature_bank, q_text, q_valid, banks, excluded)
    changed_banks = {**banks, "c": {**banks["c"], "embeddings": -banks["c"]["embeddings"]}}
    other = network.select_context(images, feature_bank, q_text + 99, ~q_valid, changed_banks, excluded)
    torch.testing.assert_close(selected["indices"], other["indices"])
    assert (selected["indices"] != excluded).all()
    before = network.forward_ablated(images, images, q_text, q_valid, c_text, c_valid, mode)
    changed = network.forward_ablated(images, images, q_text, q_valid, -c_text, c_valid, mode)
    ignored = network.forward_ablated(images, images, q_text + 99, ~q_valid, c_text, c_valid, mode)
    assert not torch.allclose(before["logits"], changed["logits"])
    torch.testing.assert_close(before["logits"], ignored["logits"])
    assert before["query_gate"].eq(0).all() and before["candidate_gate"].gt(0).all()
    before["logits"].sum().backward()
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert network.text_projection[0].weight.grad.abs().sum() == 0


def test_only_valid_training_pool_captions_are_encoded_and_cache_is_reused(toy_runtime, tmp_path, monkeypatch):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    for i, image_id in enumerate(ns["CANDIDATE_IDS"]):
        rows.loc[rows.image_id.eq(image_id), "final_caption"] = f"Pale rough material with tiny dark marks{i}"
    rows.loc[~rows.image_id.isin(ns["CANDIDATE_IDS"]), "final_caption"] = "Outside pool texture with scattered white dots"
    rows.loc[rows.image_id.eq(ns["CANDIDATE_IDS"][1]), "caption_valid"] = False
    csv = tmp_path / "captions.csv"
    rows.to_csv(csv, index=False)
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "config.json").write_text(json.dumps({"model_type": "clip", "projection_dim": 512}))
    (clip / "model.safetensors").write_bytes(b"fake weights")
    calls = {"models": 0, "texts": []}

    class Tokens(dict):
        def to(self, device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            return cls()

        def __call__(self, texts, **kwargs):
            calls["texts"].extend(texts)
            return Tokens(input_ids=torch.ones((len(texts), 1), dtype=torch.long))

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=77, projection_dim=512)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            calls["models"] += 1
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids):
            return SimpleNamespace(text_embeds=torch.ones(len(input_ids), 512))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPTokenizer=Tokenizer, CLIPTextModelWithProjection=Encoder))
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1,
               "CAPTIONS_INPUT_PATH": str(csv), "CLIP_MODEL_LOCAL_PATH": str(clip)})
    exec(code_cells()[9], ns)
    view = ns["TEXT_CACHE"]["views"]["screened"]
    assert view["image_ids"] == ns["CANDIDATE_IDS"]
    assert view["embeddings"].shape == (3, 512)
    assert not view["valid"][1] and view["embeddings"][1].eq(0).all()
    assert len(calls["texts"]) == 2
    assert all("Outside pool" not in text for text in calls["texts"])
    assert len(ns["caption_audit"]) == 3 and set(ns["caption_audit"].split) == {"train"}
    assert int(ns["audit_summary"].loc["train", "count"]) == 3
    exec(code_cells()[9], ns)
    assert calls["models"] == 1
    assert ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)["q"]["embeddings"].count_nonzero() == 0


def test_invalid_pool_is_rejected_even_when_nonpool_training_captions_are_valid(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows.loc[rows.image_id.isin(ns["CANDIDATE_IDS"]), "caption_valid"] = False
    csv = tmp_path / "captions.csv"
    rows.to_csv(csv, index=False)
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CAPTIONS_INPUT_PATH": str(csv)})
    with pytest.raises(ValueError, match="No eligible candidate-pool captions"):
        exec(code_cells()[9], ns)


def test_candidate_run_train_resume_test_and_report(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    code = code_cells()
    mode = ns["SELECTED_ABLATIONS"][0]
    exec(code[19], ns)
    run_dir = ns["RUNS_DIR"] / "candidate_only" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"candidate_only"}
    assignment = pd.read_csv(run_dir / "candidate_text_assignments.csv")
    assert len(assignment[assignment.role.eq("q")]) == 18
    assert len(assignment[assignment.role.eq("c")]) == 3
    assert not assignment.loc[assignment.role.eq("q"), "caption_available"].any()
    timestamp = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](mode, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == timestamp
    exec(code[21], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update({"plt": plt, "audit_summary": pd.DataFrame({"valid_rate": [1.0]}, index=["train"])})
    exec(code[23], ns)
    result = pd.read_csv(ns["OUT_PUT_DIR"] / "candidate_only_results.csv")
    assert set(result.split) == {"val", "test"}
    assert result.mean_query_gate.eq(0).all() and result.mean_candidate_gate.gt(0).all()
    report = (ns["OUT_PUT_DIR"] / "candidate_only_report.md").read_text()
    assert "## Candidate-pool caption coverage" in report
    plt.close("all")


def test_candidate_only_matches_the_original_suite_training(toy_runtime):
    original_mode = spec("candidate_only", query_fusion=False, semantic_selector=False)
    original_dir = original.train_one_configuration(original_mode, 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "candidate_runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
