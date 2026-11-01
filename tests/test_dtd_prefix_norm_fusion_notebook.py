import ast
import hashlib
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch

import build_dtd_prefix_norm_fusion_notebook as builder
from test_dtd_prefix_candidate_baseline_notebook import baseline_runtime
from test_dtd_text_ablation import ToyBase, ToyVisual, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {i: builder.baseline.prefix.source_of(cell) for i, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = baseline_runtime(toy_runtime)
    exec(code_cells()[2], ns)
    nodes = [node for node in ast.parse(code_cells()[12]).body
             if isinstance(node, ast.FunctionDef) and node.name == "make_norm_calibrated_model_class"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "norm_model", "exec"), ns)
    ns["AblationModel"] = ns["make_norm_calibrated_model_class"](ToyBase, ToyVisual)
    exec(code_cells()[15], ns)
    ns.update(CLIP_EMBED_DIM=4, PROTOCOL_FINGERPRINT="norm-fusion-test", BOOTSTRAP_REPEATS=0)
    ns["make_norm_validation_loader"] = lambda seed: ns["make_ablation_loaders"](seed)["val"]
    return ns


def test_notebook_is_frozen_csv_norm_control_and_preserves_previous_artifacts():
    paths = [builder.baseline.TARGET, builder.baseline.PROJECT_ROOT / "dtd-architecture-study-01.ipynb",
             builder.baseline.MANIFEST, builder.baseline.MANIFEST.parent / builder.baseline.expected_prefix()["filename"]]
    before = {path: path.read_bytes() for path in paths}
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(notebook["cells"]) == 26 and len({c["id"] for c in notebook["cells"]}) == 26
    baseline = builder.baseline.build_notebook()
    for index in (8, 9, 11, 14, 19):
        assert notebook["cells"][index]["source"] == baseline["cells"][index]["source"]
    assert notebook["cells"][23]["source"] == baseline["cells"][21]["source"]
    code = code_cells()
    combined = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = builder.baseline.prefix.source_of(cell)
            ast.parse(source)
            combined.append(source)
            assert cell["execution_count"] is None and cell["outputs"] == []
    ast.parse("\n".join(combined))
    assert "load_florence" not in "\n".join(combined)
    assert 'DTD_PREFIX_NORM_FUSION_EPOCHS", "50"' in code[4]
    assert builder.PREFIX + "_vitl16" in code[4]
    assert "IMAGE_SIZE = 384" in code[4] and "CONTEXT_TOP_K = 1" in code[4]
    assert "RUN_VALIDATION_DIAGNOSTICS = True" in code[2]
    assert '"schema": "dtd_prefix_norm_fusion_v1"' in code[17]
    assert "make_norm_calibrated_model_class(MTACSModel" in code[17]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert "num_workers=0" in code[15]
    utilities = {}
    for node in ast.parse(code[6]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.endswith("SHA256"):
            utilities[node.targets[0].id] = ast.literal_eval(node.value)
    expected = "".join(code[i] for i in (8, 9, 11, 12, 14, 15))
    assert utilities["HELPER_SOURCE_SHA256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert notebook["metadata"]["dtd_ablation"]["norm_calibrated_fusion"]
    assert notebook["metadata"]["dtd_ablation"]["architecture_study"] == builder.STAGE


def test_configuration_does_not_inherit_baseline_seed_or_output(monkeypatch):
    monkeypatch.setenv("DTD_PREFIX_CANDIDATE_BASELINE_SEED", "123")
    monkeypatch.setenv("DTD_CANDIDATE_ONLY_SEED", "456")
    monkeypatch.delenv("DTD_PREFIX_NORM_FUSION_SEED", raising=False)
    monkeypatch.delenv("DTD_SPLIT", raising=False)
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["TRAIN_SEEDS"] == [42] and not ns["EARLY_STOPPING"]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    assert ns["FUSION_NORM_EPS"] == 1e-6
    mode, = ns["SELECTED_ABLATIONS"]
    assert mode["candidate_fusion"] and not mode["query_fusion"] and not mode["semantic_selector"]
    assert ns["ARCHITECTURE_STUDY"]["norm_calibrated_fusion"]
    assert not any(ns["ARCHITECTURE_STUDY"][key] for key in ("alignment", "partial_ot", "token_fusion", "aswp"))


def test_output_guard_rejects_baseline_before_writing_artifacts(tmp_path):
    ns = {"os": os}
    exec(code_cells()[2], ns)
    path = tmp_path / "protocol.json"
    old = '{"schema": "dtd_prefix_candidate_baseline_v1", "architecture_study": {"stage": "01_prefix_candidate_only_baseline"}}'
    path.write_text(old)
    with pytest.raises(ValueError, match="another experiment"):
        ns["validate_norm_output_directory"](tmp_path)
    assert path.read_text() == old
    path.write_text(json.dumps({"schema": "dtd_prefix_norm_fusion_v1", "architecture_study": ns["ARCHITECTURE_STUDY"]}))
    ns["validate_norm_output_directory"](tmp_path)
    path.unlink()
    checkpoint = tmp_path / "runs/candidate_only/seed_42/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="checkpoints"):
        ns["validate_norm_output_directory"](tmp_path)


def test_same_parameters_initialization_selector_and_off_branch_but_unit_text_fusion(toy_runtime):
    ns = runtime(toy_runtime)
    raw_class = ns["make_ablation_model_class"](ToyBase, ToyVisual)
    ns["seed_ablation"](42)
    raw = raw_class(clip_dim=4).eval()
    network = ns["create_ablation_model"](42).eval()
    assert sum(p.numel() for p in network.parameters()) == sum(p.numel() for p in raw.parameters())
    for name, parameter in raw.state_dict().items():
        torch.testing.assert_close(parameter, network.state_dict()[name])
    mode = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](mode, 42)
    images = torch.randn(2, 3, 2, 2)
    q = torch.randn(2, 4)
    c = banks["c"]["embeddings"][:2]
    valid = torch.ones(2, dtype=torch.bool)
    feature_bank = ns["refresh_ablation_bank"](network)
    excluded = torch.tensor([0, 1])
    first = raw.select_context(images, feature_bank, q, valid, banks, excluded)
    second = network.select_context(images, feature_bank, q + 9, ~valid, banks, excluded)
    torch.testing.assert_close(first["indices"], second["indices"])
    assert (second["indices"] != excluded).all()
    off_raw = raw.forward_ablated(images, images, q, valid, c, ~valid, mode)
    off_new = network.forward_ablated(images, images, q + 9, valid, c * 20, ~valid, mode)
    torch.testing.assert_close(off_raw["logits"], off_new["logits"])
    network.capture_fusion_diagnostics = True
    out = network.forward_ablated(images, images, q, valid, c, valid, mode)
    diag = out["fusion_diagnostics"]
    torch.testing.assert_close(diag["candidate_fusion_norm"], torch.ones(2))
    torch.testing.assert_close(diag["visual_norm"], torch.ones(2))
    torch.testing.assert_close(diag["weighted_text_visual_ratio"], out["candidate_gate"].detach())
    assert out["query_gate"].eq(0).all() and out["candidate_gate"].gt(0).all()
    assert not torch.allclose(raw.forward_ablated(images, images, q, valid, c, valid, mode)["logits"], out["logits"])
    out["logits"].square().sum().backward()
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert network.text_projection[0].weight.grad.abs().sum() == 0


def test_normalization_removes_positive_projection_scale_not_caption_content(toy_runtime):
    ns = runtime(toy_runtime)
    model = ns["create_ablation_model"](42).eval()
    model.capture_fusion_diagnostics = True
    images = torch.randn(2, 3, 2, 2)
    text = torch.randn(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    mode = ns["SELECTED_ABLATIONS"][0]
    original = model.forward_ablated(images, images, text, valid, text, valid, mode)
    hook = model.context_text_projection.register_forward_hook(lambda module, inputs, output: output * 100)
    try:
        scaled = model.forward_ablated(images, images, text + 9, valid, text, valid, mode)
    finally:
        hook.remove()
    torch.testing.assert_close(original["logits"], scaled["logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(scaled["fusion_diagnostics"]["candidate_raw_projection_norm"],
                               original["fusion_diagnostics"]["candidate_raw_projection_norm"] * 100)
    changed = model.forward_ablated(images, images, text, valid, -text, valid, mode)
    assert not torch.allclose(original["logits"], changed["logits"])


def test_shuffle_preserves_masks_pool_and_original_banks_including_singleton(toy_runtime):
    ns = runtime(toy_runtime)
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    banks["c"]["valid"][1] = False
    banks["c"]["embeddings"][1] = 0
    original = banks["c"]["embeddings"].clone()
    shuffled, assignment = ns["make_validation_caption_shuffle"](banks, 33)
    torch.testing.assert_close(original, banks["c"]["embeddings"])
    torch.testing.assert_close(banks["c"]["valid"], shuffled["c"]["valid"])
    assert not assignment.loc[~assignment.caption_available, "source_changed"].any()
    assert assignment.loc[assignment.caption_available, "source_changed"].all()
    assert set(shuffled["c"]["source_ids"]) == set(ns["CANDIDATE_IDS"])
    assert shuffled["q"] is banks["q"]
    again, _ = ns["make_validation_caption_shuffle"](banks, 33)
    torch.testing.assert_close(shuffled["c"]["embeddings"], again["c"]["embeddings"])
    banks["c"]["valid"][2] = False
    single, manifest = ns["make_validation_caption_shuffle"](banks, 33)
    assert not manifest.source_changed.any()
    torch.testing.assert_close(single["c"]["embeddings"], banks["c"]["embeddings"])


def test_validation_interventions_use_one_selection_per_batch_and_restore_model(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](mode, 42)
    network = ns["create_ablation_model"](42)
    before = {key: value.clone() for key, value in network.state_dict().items()}
    bank = ns["refresh_ablation_bank"](network)
    calls = []
    select = network.select_context

    def track(*args, **kwargs):
        calls.append(1)
        return select(*args, **kwargs)

    monkeypatch.setattr(network, "select_context", track)
    loader = ns["make_ablation_loaders"](42)["val"]
    summary, predictions, norms, assignment = ns["collect_norm_validation_interventions"](network, loader, banks, bank, 33)
    assert len(calls) == len(loader)
    assert network.training and not network.capture_fusion_diagnostics
    for key, value in before.items():
        torch.testing.assert_close(value, network.state_dict()[key])
    assert set(predictions.image_id) == set(ns["val_df"].image_id)
    assert predictions.groupby("image_id").selected_context_id.nunique().eq(1).all()
    assert len(predictions) == 18 and len(summary) == 3
    off = predictions.loc[predictions.intervention.eq("text_off")]
    assert not off.candidate_caption_valid.any() and off.candidate_gate.eq(0).all()
    assert off.weighted_text_visual_ratio.eq(0).all()
    assert predictions.query_gate.eq(0).all()
    assert np.isfinite(norms.dropna()[["mean", "p05", "median", "p95", "max"]]).all().all()
    assert summary.loc[summary.intervention.eq("aligned"), "mean_abs_logit_change"].eq(0).all()
    assert summary.loc[summary.intervention.eq("text_off"), "mean_abs_logit_change"].gt(0).all()
    assert summary.candidate_caption_available_rate.nunique() == 1
    assert len(assignment) == len(ns["CANDIDATE_IDS"])
    with pytest.raises(ValueError, match="non-validation"):
        ns["collect_norm_validation_interventions"](network, ns["make_ablation_loaders"](42)["test"], banks, bank, 33)
    assert network.training and not network.capture_fusion_diagnostics


def test_training_resume_validation_diagnostic_cache_locked_test_and_report(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    with pytest.raises(RuntimeError, match="Finish"):
        ns["run_norm_validation_interventions"](mode, 42)
    exec(code_cells()[19], ns)
    run_dir = ns["RUNS_DIR"] / "candidate_only" / "seed_42"
    stamp = (run_dir / "best.pt").stat().st_mtime_ns
    exec(code_cells()[19], ns)
    assert (run_dir / "best.pt").stat().st_mtime_ns == stamp
    exec(code_cells()[21], ns)
    summary = ns["interventions"].copy()
    for column in ("ci95_low", "ci95_high"):
        summary[column] = pd.to_numeric(summary[column])
    cached = ns["run_norm_validation_interventions"](mode, 42)
    pd.testing.assert_frame_equal(summary, cached, check_dtype=False, atol=1e-12, rtol=1e-12)
    diag = run_dir / "validation_norm_diagnostics"
    assert {path.name for path in diag.iterdir()} == {"metadata.json", "validation_interventions.csv",
        "validation_intervention_predictions.csv", "validation_norm_summary.csv", "validation_caption_shuffle_assignments.csv"}
    assert not (run_dir / "test_metrics.json").exists()
    assert (run_dir / "best.pt").stat().st_mtime_ns == stamp
    saved_val = ns["read_json"](run_dir / "val_metrics.json")
    aligned = cached.loc[cached.intervention.eq("aligned")].iloc[0]
    for metric in ("accuracy", "macro_f1", "loss"):
        assert aligned[metric] == pytest.approx(saved_val[metric], abs=1e-6)
    exec(code_cells()[23], ns)
    assert (ns["OUT_PUT_DIR"] / "locked_test_plan.json").is_file()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update(plt=plt, audit_summary=pd.DataFrame({"valid_rate": [1.]}, index=["train"]))
    exec(code_cells()[25], ns)
    results = pd.read_csv(ns["OUT_PUT_DIR"] / (builder.PREFIX + "_results.csv"))
    assert results.norm_calibrated_fusion.all() and results.mean_query_gate.eq(0).all()
    assert set(results.split) == {"val", "test"}
    report = (ns["OUT_PUT_DIR"] / (builder.PREFIX + "_report.md")).read_text(encoding="utf-8")
    assert "Fixed-context validation text interventions" in report
    assert "not semantic alignment" in report
    ns["DIAGNOSTIC_SHUFFLE_SEED"] += 1
    with pytest.raises(ValueError, match="Cached validation diagnostics"):
        ns["run_norm_validation_interventions"](mode, 42)
    plt.close("all")
