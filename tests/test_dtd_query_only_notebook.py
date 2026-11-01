
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

import build_mtacs_dtd_query_only_notebook as builder
import build_mtacs_dtd_visual_only_notebook as visual
import mtacs_dtd_ablation_cells as original
from test_dtd_text_ablation import ToyBase, ToyVisual, spec, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: visual.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    code = code_cells()
    ns = dict(vars(original))
    ns.update({"os": os, "display": lambda value: None})
    exec(compile(code[2], "query_config", "exec"), ns)
    exec(compile(code[6], "utilities", "exec"), ns)
    exec(compile(code[8], "text_helpers", "exec"), ns)
    nodes = [node for node in ast.parse(code[9]).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "query_banks", "exec"), ns)
    names = {"make_ablation_model_class", "candidate_images_at", "run_forward"}
    nodes = [node for node in ast.parse(code[12]).body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "model_helpers", "exec"), ns)
    ns["AblationModel"] = ns["make_ablation_model_class"](ToyBase, ToyVisual)
    exec(compile(code[14], "training", "exec"), ns)
    exec(compile(code[15], "evaluation", "exec"), ns)
    ns.update({"CLIP_EMBED_DIM": 4, "LR_SCHEDULE_EPOCHS": ns["EPOCHS"],
               "PROTOCOL_FINGERPRINT": "query-test", "candidate_df": ns["train_df"].iloc[:3].copy()})
    return ns


def audited_rows(frames):
    rows = []
    for split, frame in frames.items():
        for image_id in frame.image_id:
            rows.append({"image_id": image_id, "split": split,
                         "final_caption": "Pale rough material with tiny dark marks",
                         "caption_valid": True, "leakage_detected": False,
                         "semantic_leakage_detected": False, "semantic_leakage_score": 0.10})
    return pd.DataFrame(rows)


def test_only_query_mode_is_selected_with_offline_caption_generation(monkeypatch):
    notebook = builder.build_notebook()
    assert len(notebook["cells"]) == 24
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = visual.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            code.append(source)
    all_code = "\n".join(code)
    ast.parse(all_code)
    for forbidden in ("validation_interventions", "pip install --index-url"):
        assert forbidden not in all_code
    for required in ("load_florence", "generate_batch", "ensure_dtd_caption_csv", "--no-index", "--target"):
        assert required in all_code
    assert notebook["metadata"]["dtd_ablation"]["caption_generation"]
    assert "CLIPTextModelWithProjection" in all_code and "local_files_only=True" in all_code
    assert 'DTD_QUERY_ONLY_EPOCHS", "50"' in all_code
    assert "EARLY_STOPPING = False" in all_code
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[0]
    monkeypatch.setenv("DTD_ABLATIONS", "visual_only,full_both")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert len(ns["SELECTED_ABLATIONS"]) == 1
    mode = ns["SELECTED_ABLATIONS"][0]
    assert mode["name"] == "query_only" and mode["query_fusion"]
    assert not mode["candidate_fusion"] and not mode["semantic_selector"]


def test_visual_architecture_training_losses_and_input_transforms_match_notebook_01():
    first = visual.build_notebook()
    code = code_cells()
    assert code[11].strip() == visual.source_of(first["cells"][9]).strip()
    assert code[12].strip() == visual.source_of(first["cells"][10]).strip()
    functions = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu"]
    assert visual.definitions(code[14], functions) == visual.definitions(visual.source_of(first["cells"][12]), functions)
    settings = visual.source_of(first["cells"][4])
    for name in ("train_transform", "eval_transform"):
        statement = next(ast.get_source_segment(settings, node) for node in ast.parse(settings).body
                         if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
        assert statement in code[4]
    assert "_ablation_02_query_only_vitl16" in code[4]
    assert "CANDIDATE_POOL_FRACTION = 0.20" in code[4]
    assert "CAPTION_SEMANTIC_THRESHOLD = 0.30" in code[2]


def test_csv_discovery_rejects_wrong_dataset_and_keeps_explicit_cache_consistent(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    frames = ns["SPLIT_FRAMES"]
    wrong = tmp_path / "a_sun397_captions.csv"
    rows = audited_rows(frames)
    rows.assign(image_id=[f"sun397/{i}.jpg" for i in range(len(rows))]).to_csv(wrong, index=False)
    correct = tmp_path / "b_dtd_captions.csv"
    rows.to_csv(correct, index=False)
    cached = tmp_path / "cache" / "captions.csv"
    assert ns["find_dtd_caption_csv"](frames, "", cached, [tmp_path]) == correct
    with pytest.raises(ValueError, match="incomplete"):
        ns["find_dtd_caption_csv"](frames, str(wrong), cached, [tmp_path])
    cached.parent.mkdir()
    rows.assign(final_caption="Fine dark patches over pale rough material").to_csv(cached, index=False)
    with pytest.raises(ValueError, match="differ"):
        ns["find_dtd_caption_csv"](frames, str(correct), cached, [])
    assert ns["find_dtd_caption_csv"](frames, "", cached, []) == cached


def test_query_vectors_affect_fusion_but_not_visual_selection_and_candidates_remain_zero(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](mode, 42)
    assert banks["q"]["valid"].any()
    assert not banks["c"]["valid"].any()
    assert banks["c"]["embeddings"].count_nonzero() == 0
    with pytest.raises(ValueError, match="query_only only"):
        ns["make_ablation_banks"](mode, 42, {"semantic_selector": True})
    network = ns["create_ablation_model"](42).eval()
    images = torch.randn(2, 3, 2, 2)
    text, valid = ns["batch_caption_inputs"](banks, ["train/0", "train/1"])
    feature_bank = ns["refresh_ablation_bank"](network)
    excluded = torch.tensor([0, 1])
    first = network.select_context(images, feature_bank, text, valid, banks, excluded)
    second = network.select_context(images, feature_bank, text * -100, valid, banks, excluded)
    torch.testing.assert_close(first["indices"], second["indices"])
    assert (first["indices"] != excluded).all()
    zero = torch.zeros_like(text)
    absent = torch.zeros_like(valid)
    before = network.forward_ablated(images, images, text, valid, zero, absent, mode)
    after = network.forward_ablated(images, images, -text, valid, zero, absent, mode)
    assert not torch.allclose(before["logits"], after["logits"])
    assert before["query_gate"].gt(0).all() and before["candidate_gate"].eq(0).all()
    before["logits"].sum().backward()
    assert network.text_projection[0].weight.grad.abs().sum() > 0
    assert network.context_text_projection[0].weight.grad.abs().sum() == 0


def test_caption_stage_reuses_csv_encodes_locally_and_resumes_cache(toy_runtime, tmp_path, monkeypatch):
    ns = runtime(toy_runtime)
    csv = tmp_path / "input_captions.csv"
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows.loc[1, "caption_valid"] = False
    rows.to_csv(csv, index=False)
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "config.json").write_text(json.dumps({"model_type": "clip", "projection_dim": 512}))
    (clip / "model.safetensors").write_bytes(b"fake test weights")
    calls = []

    class Tokens(dict):
        def to(self, device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            return cls()

        def __call__(self, texts, **kwargs):
            return Tokens(input_ids=torch.ones((len(texts), 1), dtype=torch.long))

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=77, projection_dim=512)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            calls.append(path)
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
    assert len(calls) == 1
    view = ns["TEXT_CACHE"]["views"]["screened"]
    assert view["embeddings"].shape == (18, 512)
    assert not view["valid"][1] and view["embeddings"][1].eq(0).all()
    assert set(ns["TEXT_CACHE"]["views"]) == {"screened"}
    exec(code_cells()[9], ns)
    assert len(calls) == 1
    assert ns["CAPTION_CSV"].read_bytes() == csv.read_bytes()
    assert not any("Florence" in call for call in calls)


def test_no_eligible_training_captions_stops_before_loading_clip(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    csv = tmp_path / "bad_captions.csv"
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows.loc[rows.split.eq("train"), "caption_valid"] = False
    rows.to_csv(csv, index=False)
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CAPTIONS_INPUT_PATH": str(csv)})
    with pytest.raises(ValueError, match="No eligible training captions"):
        exec(code_cells()[9], ns)


def test_query_run_train_resume_test_and_report(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    code = code_cells()
    exec(code[19], ns)
    run_dir = ns["RUNS_DIR"] / "query_only" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"query_only"}
    assert (run_dir / "query_text_manifest.json").is_file()
    before = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](mode, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == before
    exec(code[21], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update({"plt": plt, "audit_summary": pd.DataFrame({"valid_rate": [0.9]}, index=["train"])})
    exec(code[23], ns)
    table = pd.read_csv(ns["OUT_PUT_DIR"] / "query_only_results.csv")
    assert set(table.split) == {"val", "test"}
    assert table.mean_query_gate.gt(0).all() and table.mean_candidate_gate.eq(0).all()
    assert "## Screened caption coverage" in (ns["OUT_PUT_DIR"] / "query_only_report.md").read_text()
    plt.close("all")


def test_query_only_matches_original_training_despite_masked_candidate_vectors(toy_runtime):
    mode = spec("query_only", candidate_fusion=False, semantic_selector=False)
    original_dir = original.train_one_configuration(mode, 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "separate_runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
