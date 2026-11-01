
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import json
import os
from functools import lru_cache

import pandas as pd
import pytest
import torch

import build_mtacs_dtd_visual_only_notebook as builder
import mtacs_dtd_ablation_cells as original
from test_dtd_text_ablation import ToyBase, ToyVisual, spec, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: builder.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    code = code_cells()
    namespace = dict(vars(original))
    namespace.update({"os": os, "display": lambda value: None})
    exec(compile(code[2], "visual_only_config", "exec"), namespace)
    exec(compile(code[6], "utilities", "exec"), namespace)
    namespace["CLIP_EMBED_DIM"] = 4
    namespace["candidate_df"] = namespace["train_df"].iloc[:3].copy()
    exec(compile(code[7], "masked_inputs", "exec"), namespace)
    # Keep synthetic image loaders, but exercise the notebook model/selection/forward functions.
    names = {"make_ablation_model_class", "candidate_images_at", "run_forward"}
    nodes = [node for node in ast.parse(code[10]).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "model_helpers", "exec"), namespace)
    namespace["AblationModel"] = namespace["make_ablation_model_class"](ToyBase, ToyVisual)
    exec(compile(code[12], "training", "exec"), namespace)
    exec(compile(code[13], "evaluation", "exec"), namespace)
    namespace.update({"LR_SCHEDULE_EPOCHS": namespace["EPOCHS"], "PROTOCOL_FINGERPRINT": "visual-test"})
    return namespace


def test_generated_notebook_is_standalone_visual_only_without_caption_models():
    notebook = builder.build_notebook()
    assert len(notebook["cells"]) == 22
    assert len({cell["id"] for cell in notebook["cells"]}) == 22
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = builder.source_of(cell)
            ast.parse(text)
            assert cell["execution_count"] is None and cell["outputs"] == []
            code.append(text)
    all_code = "\n".join(code)
    ast.parse(all_code)
    for name in ("load_florence", "AutoProcessor", "CLIPModel", "load_semantic_auditor",
                 "generate_batch", "validation_interventions", "caption_probes", "CAPTION_CSV"):
        assert name not in all_code
    assert "import transformers" not in all_code
    assert 'DTD_VISUAL_ONLY_EPOCHS", "50"' in all_code
    assert 'EPOCHS"' not in builder.CONFIG  # Configuration selection cannot include the other seven modes.
    assert 'ABLATION_NAMES = ["visual_only"]' in all_code
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[0]
    assert "source='local'" in all_code
    assert "DTD_VISUAL_ONLY_OUTPUT" in all_code
    assert "_ablation_01_visual_only_vitl16" in all_code
    assert builder.TARGET.name != builder.SOURCE.name


def test_original_visual_architecture_losses_and_preprocessing_are_retained():
    source = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    code = code_cells()
    assert code[9].strip() == builder.source_of(source["cells"][13]).strip()
    assert code[10].strip() == builder.source_of(source["cells"][14]).strip()
    original_training = builder.source_of(source["cells"][15])
    names = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu"]
    assert builder.definitions(code[12], names) == builder.definitions(original_training, names)
    settings = builder.source_of(source["cells"][5])
    for name in ("train_transform", "eval_transform"):
        before = next(ast.get_source_segment(settings, node) for node in ast.parse(settings).body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
        assert before in code[4]
    assert "NUM_CLASSES = 47" in code[4]
    assert "CANDIDATE_POOL_FRACTION = 0.20" in code[4]
    assert "CONTEXT_TOP_K = 1" in code[4]


def test_placeholder_text_is_masked_and_cannot_enable_a_text_mode(toy_runtime):
    ns = runtime(toy_runtime)
    configuration = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](configuration, 42)
    for role in ("q", "c"):
        assert not banks[role]["valid"].any()
        assert torch.count_nonzero(banks[role]["embeddings"]) == 0
        assert set(banks[role]["texts"]) == {""}
    assert banks["c"]["source_ids"] == ns["CANDIDATE_IDS"]
    with pytest.raises(ValueError, match="visual_only only"):
        ns["make_ablation_banks"]({**configuration, "query_fusion": True}, 42)

    image = torch.randn(2, 3, 2, 2)
    network = ns["create_ablation_model"](42).eval()
    text = torch.randn(2, 4)
    with torch.inference_mode():
        original_output = network.forward_ablated(image, image, text, torch.ones(2, dtype=torch.bool),
                                                   text * 2, torch.ones(2, dtype=torch.bool), configuration)
        empty_output = network.forward_ablated(image, image, torch.zeros_like(text), torch.zeros(2, dtype=torch.bool),
                                                torch.zeros_like(text), torch.zeros(2, dtype=torch.bool), configuration)
    torch.testing.assert_close(original_output["logits"], empty_output["logits"])
    assert empty_output["query_gate"].sum() == empty_output["candidate_gate"].sum() == 0


def test_single_configuration_trains_resumes_and_reports_without_caption_files(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    configuration = ns["SELECTED_ABLATIONS"][0]
    code = code_cells()
    with pytest.raises(RuntimeError, match="Complete every"):
        ns["evaluate_locked_test"]()
    exec(compile(code[17], "train_cell", "exec"), ns)
    run_dir = ns["RUNS_DIR"] / "visual_only" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"visual_only"}
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "disabled_text_manifest.json").is_file()
    assert not list(ns["OUT_PUT_DIR"].rglob("captions.csv"))
    timestamp = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](configuration, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == timestamp
    exec(compile(code[19], "test_cell", "exec"), ns)
    assert len(ns["summarize_ablation"]("test")) == 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns["plt"] = plt
    exec(compile(code[21], "report_cell", "exec"), ns)
    table = pd.read_csv(ns["OUT_PUT_DIR"] / "visual_only_results.csv")
    assert len(table) == 2
    assert set(table["split"]) == {"val", "test"}
    assert set(table["experiment"]) == {"visual_only"}
    assert table["mean_query_gate"].eq(0).all() and table["mean_candidate_gate"].eq(0).all()
    report = (ns["OUT_PUT_DIR"] / "visual_only_report.md").read_text(encoding="utf-8")
    assert "\n## Metrics\n" in report
    assert "| visual_only |" in report
    plt.close("all")


def test_fresh_report_is_safe_before_training(toy_runtime):
    ns = runtime(toy_runtime)
    exec(compile(code_cells()[19], "test_cell", "exec"), ns)
    exec(compile(code_cells()[21], "report_cell", "exec"), ns)
    assert not (ns["OUT_PUT_DIR"] / "locked_test_plan.json").exists()
    assert pd.read_csv(ns["OUT_PUT_DIR"] / "visual_only_results.csv").empty


def test_masked_zero_training_matches_the_suites_visual_only(toy_runtime):
    configuration = spec("visual_only", query_fusion=False, candidate_fusion=False, semantic_selector=False)
    original_dir = original.train_one_configuration(configuration, 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "standalone_runs"
    ns["LR_SCHEDULE_EPOCHS"] = original.EPOCHS
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
