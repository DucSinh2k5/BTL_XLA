import ast
import hashlib
import importlib.util
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch

import build_dtd_prefix_semantic_selector_notebook as builder
from test_dtd_prefix_mltm_alignment_notebook import runtime as alignment_runtime
from test_dtd_text_ablation import ToyBase, ToyVisual, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {i: builder.alignment.norm.baseline.prefix.source_of(c)
            for i, c in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = alignment_runtime(toy_runtime)
    exec(code_cells()[2], ns)
    nodes = [n for n in ast.parse(code_cells()[12]).body if isinstance(n, ast.FunctionDef)
             and n.name in {"validate_alignment_pool", "make_semantic_selector_model_class"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "study04_helpers", "exec"), ns)
    bank_nodes = [n for n in ast.parse(code_cells()[9]).body
                  if isinstance(n, ast.FunctionDef) and n.name == "make_ablation_banks"]
    exec(compile(ast.Module(body=bank_nodes, type_ignores=[]), "study04_banks", "exec"), ns)
    ns["AblationModel"] = ns["make_semantic_selector_model_class"](ToyBase, ToyVisual)
    exec(code_cells()[14], ns)
    exec(code_cells()[15], ns)
    ns.update(CLIP_EMBED_DIM=4, PROTOCOL_FINGERPRINT="semantic-selector-test", BOOTSTRAP_REPEATS=0)
    ns["make_norm_validation_loader"] = lambda seed: ns["make_ablation_loaders"](seed)["val"]
    return ns


def test_standalone_notebook_preserves_old_artifacts_and_frozen_caption_scope():
    root = builder.PROJECT_ROOT
    paths = [builder.alignment.TARGET, builder.alignment.norm.TARGET,
             root / "stydu2_da_train.ipynb", root / "study3_datrain.ipynb",
             root / "data/captions/dtd/grounded_controls/captions_prefix_control.csv"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original = builder.alignment.build_notebook()
    for i in [8, 11, 19, 23]:
        assert notebook["cells"][i]["source"] == original["cells"][i]["source"]
    assert len(notebook["cells"]) == len({c["id"] for c in notebook["cells"]}) == 26
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
            assert not c["outputs"] and c["execution_count"] is None
    code = code_cells()
    assert "load_florence" not in "\n".join(code.values())
    assert 'DTD_PREFIX_SEMANTIC_SELECTOR_EPOCHS", "50"' in code[4]
    assert '"semantic_selector": semantic_selector_config()' in code[17]
    assert builder.PREFIX + "_vitl16" in code[4]
    utility = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(code[6]).body
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id.endswith("SHA256")}
    assert utility["HELPER_SOURCE_SHA256"] == hashlib.sha256(
        "".join(code[i] for i in (8, 9, 11, 12, 14, 15)).encode()).hexdigest()
    assert notebook["metadata"]["dtd_ablation"]["semantic_selector"]
    assert not notebook["metadata"]["dtd_ablation"]["query_fusion"]


def test_configuration_has_distinct_output_guard_and_no_query_caption(tmp_path, monkeypatch):
    monkeypatch.setenv("DTD_PREFIX_MLTM_ALIGNMENT_SEED", "99")
    monkeypatch.delenv("DTD_PREFIX_SEMANTIC_SELECTOR_SEED", raising=False)
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["TRAIN_SEEDS"] == [42]
    ns["validate_semantic_selector_settings"]()
    assert ns["SELECTED_ABLATIONS"][0]["semantic_selector"]
    assert not ns["SELECTED_ABLATIONS"][0]["query_fusion"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"schema": "dtd_prefix_mltm_alignment_v1"}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="another experiment"):
        ns["validate_semantic_selector_output_directory"](tmp_path)
    assert path.read_bytes() == before
    ns["ALIGNMENT_WEIGHT"] = 0
    with pytest.raises(ValueError, match="alignment supervision"):
        ns["validate_semantic_selector_settings"]()


def test_saved_output_audit_recomputes_metrics_without_modifying_sources(tmp_path, monkeypatch):
    path = builder.PROJECT_ROOT / "tools/project_management/audit_dtd_architecture_study02_03.py"
    spec = importlib.util.spec_from_file_location("audit_study02_03", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    paths = [builder.PROJECT_ROOT / name for _, name, _, _ in audit.SOURCES]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    monkeypatch.setattr(audit, "REPORT", tmp_path)
    audit.audit()
    evidence = json.loads((tmp_path / "study02_03_output_evidence.json").read_text())
    assert not evidence["executes_notebook_cells"] and not evidence["loads_checkpoint_pickle"]
    assert all(evidence["protocol_matches"].values()) and not evidence["runtime_versions_match"]
    assert evidence["studies"]["02"]["metrics"]["test"]["macro_f1"] == pytest.approx(.849329342465812)
    assert evidence["studies"]["03"]["metrics"]["test"]["macro_f1"] == pytest.approx(.8487223321568519)
    assert evidence["paired_study03_vs_study02"]["test"]["improved"] == 6
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_same_parameters_and_rng_as_study03_plus_persisted_ramp_epoch(toy_runtime):
    ns = runtime(toy_runtime)
    network = ns["create_ablation_model"](42)
    after = torch.get_rng_state().clone()
    control_ns = alignment_runtime(toy_runtime)
    control = control_ns["create_ablation_model"](42)
    assert torch.equal(after, torch.get_rng_state())
    assert sum(p.numel() for p in network.parameters()) == sum(p.numel() for p in control.parameters())
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, network.state_dict()[key])
    for epoch, expected in [(0, 0), (5, 0), (6, .025), (10, .125), (50, .125)]:
        network.semantic_selector_epoch.fill_(epoch)
        assert float(network.semantic_selector_weight().detach()) == pytest.approx(expected)
    reloaded = ns["create_ablation_model"](42)
    reloaded.load_state_dict(network.state_dict())
    assert int(reloaded.semantic_selector_epoch) == 50


def inputs(ns):
    network = ns["create_ablation_model"](42).eval()
    network.semantic_selector_epoch.fill_(50)
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    batch = next(iter(ns["make_ablation_loaders"](42)["train"]))
    query = batch["query_image"]
    features = ns["refresh_ablation_bank"](network)
    excluded = torch.full((len(query),), -1, dtype=torch.long)
    return network, banks, query, features, excluded


def test_semantic_scores_do_not_use_query_text_and_exclude_self_from_statistics(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, query, features, excluded = inputs(ns)
    zeros, false = torch.zeros(len(query), 4), torch.zeros(len(query), dtype=torch.bool)
    a = network.select_context(query, features, zeros, false, banks, excluded, semantic=True)
    b = network.select_context(query, features, torch.full_like(zeros, float("nan")), ~false,
                               banks, excluded, semantic=True)
    torch.testing.assert_close(a["selector_scores"], b["selector_scores"])
    banks["c"]["valid"][1] = False
    excluded[0] = 0
    result = network.select_context(query, features, zeros, false, banks, excluded, semantic=True)
    assert torch.isneginf(result["selector_scores"][0, 0])
    assert result["indices"][0] != 0
    assert result["semantic_scores"][:, 1].eq(0).all()
    # Row0 has only one valid nonself caption, so its semantic branch is neutral.
    assert result["semantic_scores"][0].eq(0).all()
    assert result["semantic_scores"].abs().max() <= ns["SELECTOR_SEMANTIC_Z_CLIP"]


def test_missing_or_constant_captions_fall_back_without_penalizing_visual_candidates(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, query, features, excluded = inputs(ns)
    q, valid = torch.zeros(len(query), 4), torch.zeros(len(query), dtype=torch.bool)
    for kind in ["missing", "constant"]:
        if kind == "missing":
            banks["c"]["valid"].zero_()
            banks["c"]["embeddings"].fill_(float("nan"))
        else:
            banks["c"]["valid"].fill_(True)
            banks["c"]["embeddings"].fill_(1.)
        result = network.select_context(query, features, q, valid, banks, excluded, semantic=True)
        control = network.select_context(query, features, q, valid, banks, excluded, semantic=False)
        assert result["semantic_scores"].eq(0).all()
        torch.testing.assert_close(result["indices"], control["indices"])
        torch.testing.assert_close(result["log_prob"], control["log_prob"])


def test_no_cross_query_batch_coupling_and_reject_test_queries_before_inference(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    network, banks, query, features, excluded = inputs(ns)
    q, valid = torch.zeros(len(query), 4), torch.zeros(len(query), dtype=torch.bool)
    together = network.select_context(query, features, q, valid, banks, excluded, semantic=True)
    alone = network.select_context(query[:1], features, q[:1], valid[:1], banks, excluded[:1], semantic=True)
    torch.testing.assert_close(together["selector_scores"][:1], alone["selector_scores"], atol=1e-6, rtol=1e-6)

    def forbidden(*args, **kwargs):
        raise AssertionError("Test queries must be rejected before forward")

    monkeypatch.setattr(network, "select_context", forbidden)
    test_loader = ns["make_ablation_loaders"](42)["test"]
    with pytest.raises(ValueError, match="non-validation"):
        ns["collect_selector_validation_interventions"](network, test_loader, banks, features, 2026)


def test_selector_gradient_reaches_shared_projection_metric_mix_and_image_not_clip_or_query_text(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, query, features, excluded = inputs(ns)
    query = query.detach().requires_grad_(True)
    q_text = torch.randn(len(query), 4, requires_grad=True)
    banks["c"]["embeddings"].requires_grad_(True)
    result = network.select_context(query, features, q_text, torch.zeros(len(query), dtype=torch.bool),
                                    banks, excluded, semantic=True)
    loss = result["selector_scores"][:, 0].sum()
    loss.backward()
    for parameter in [network.context_text_projection[0].weight, network.alignment_log_diagonal,
                      network.selector.semantic_mix_logit]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert q_text.grad is None and banks["c"]["embeddings"].grad is None


def test_zero_weight_training_exactly_matches_study03(toy_runtime, tmp_path):
    control = alignment_runtime(toy_runtime)
    control.update(OUT_PUT_DIR=tmp_path / "control", RUNS_DIR=tmp_path / "control/runs")
    expected = control["train_one_configuration"](control["SELECTED_ABLATIONS"][0], 42)
    ns = runtime(toy_runtime)
    ns.update(SELECTOR_SEMANTIC_MAX_WEIGHT=0., OUT_PUT_DIR=tmp_path / "zero", RUNS_DIR=tmp_path / "zero/runs")
    actual = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    a, b = pd.read_csv(expected / "history.csv"), pd.read_csv(actual / "history.csv")
    pd.testing.assert_frame_equal(a, b[a.columns], atol=1e-7, rtol=1e-7)
    assert b.train_selector_semantic_weight.eq(0).all()


def test_epoch_boundary_resume_preserves_semantic_schedule_and_optimizer(toy_runtime, tmp_path):
    reference = runtime(toy_runtime)
    reference.update(EPOCHS=3, LR_SCHEDULE_EPOCHS=3, KEEP_LAST_CHECKPOINT=True,
        SELECTOR_VISUAL_WARMUP_EPOCHS=0, SELECTOR_SEMANTIC_RAMP_EPOCHS=1,
        OUT_PUT_DIR=tmp_path / "reference", RUNS_DIR=tmp_path / "reference/runs")
    full = reference["train_one_configuration"](reference["SELECTED_ABLATIONS"][0], 42)
    ns = runtime(toy_runtime)
    ns.update(EPOCHS=3, LR_SCHEDULE_EPOCHS=3, KEEP_LAST_CHECKPOINT=True,
        SELECTOR_VISUAL_WARMUP_EPOCHS=0, SELECTOR_SEMANTIC_RAMP_EPOCHS=1,
        OUT_PUT_DIR=tmp_path / "resumed", RUNS_DIR=tmp_path / "resumed/runs")
    original = ns["train_ablation_epoch"]

    def pause(network, loader, banks, bank, optimizer, scaler, epoch):
        if epoch == 2:
            raise RuntimeError("simulated interruption")
        return original(network, loader, banks, bank, optimizer, scaler, epoch)

    ns["train_ablation_epoch"] = pause
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    ns["train_ablation_epoch"] = original
    resumed = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    pd.testing.assert_frame_equal(pd.read_csv(full / "history.csv"), pd.read_csv(resumed / "history.csv"))
    a = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key])
    assert int(b["model"]["semantic_selector_epoch"]) == 3


def test_training_both_diagnostic_paths_test_report_and_selector_only_shuffle(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    ns.update(SELECTOR_VISUAL_WARMUP_EPOCHS=0, SELECTOR_SEMANTIC_RAMP_EPOCHS=1)
    code = code_cells()
    exec(code[19], ns)
    run = ns["RUNS_DIR"] / "candidate_only/seed_42"
    history = pd.read_csv(run / "history.csv")
    assert len(history) == 2 and history.train_selector_semantic_weight.gt(0).all()
    assert history.train_alignment_loss.gt(0).all()
    exec(code[21], ns)
    fusion = pd.read_csv(run / "validation_norm_diagnostics/validation_intervention_predictions.csv")
    selector = pd.read_csv(run / "validation_selector_diagnostics/validation_selector_predictions.csv")
    assert fusion.groupby("image_id").selected_context_id.nunique().eq(1).all()
    assert set(selector.intervention) == {"aligned", "visual_selector_only", "selector_text_shuffled"}
    assert selector.candidate_caption_source_id.equals(selector.selected_context_id)
    shuffled = selector.loc[selector.intervention.eq("selector_text_shuffled")]
    assert (shuffled.selector_caption_source_id != shuffled.selected_context_id).all()
    assert not (run / "test_metrics.json").exists()
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert int(state["model"]["semantic_selector_epoch"]) == state["epoch"]
    cached = ns["run_selector_validation_interventions"](ns["SELECTED_ABLATIONS"][0], 42)
    assert len(cached) == 3
    exec(code[23], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update(plt=plt, audit_summary=pd.DataFrame({"valid_rate": [1.]}, index=["train"]))
    exec(code[25], ns)
    results = pd.read_csv(ns["OUT_PUT_DIR"] / (builder.PREFIX + "_results.csv"))
    assert results.semantic_selector_enabled.all() and results.alignment_enabled.all()
    assert results.mean_query_gate.eq(0).all() and set(results.split) == {"val", "test"}
    report = (ns["OUT_PUT_DIR"] / (builder.PREFIX + "_report.md")).read_text(encoding="utf-8")
    assert "selector configuration" in report and "context reselected" in report
    plt.close("all")
