import ast
import hashlib
import json
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch

import build_dtd_prefix_mltm_alignment_notebook as builder
from test_dtd_prefix_norm_fusion_notebook import runtime as norm_runtime
from test_dtd_text_ablation import ToyBase, ToyVisual, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {i: builder.norm.baseline.prefix.source_of(cell) for i, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = norm_runtime(toy_runtime)
    exec(code_cells()[2], ns)
    source = builder.HELPERS.read_text(encoding="utf-8")
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "alignment_helpers", "exec"), ns)
    ns["AblationModel"] = ns["make_mltm_alignment_model_class"](ToyBase, ToyVisual)
    exec(code_cells()[14], ns)
    exec(code_cells()[15], ns)
    ns.update(CLIP_EMBED_DIM=4, PROTOCOL_FINGERPRINT="alignment-test", BOOTSTRAP_REPEATS=0)
    ns["make_norm_validation_loader"] = lambda seed: ns["make_ablation_loaders"](seed)["val"]
    return ns


def test_notebook_standalone_single_change_scope_and_previous_artifacts_unchanged():
    paths = [builder.norm.TARGET, builder.norm.baseline.TARGET,
             builder.norm.PROJECT_ROOT / "dtd-architecture-study-01.ipynb",
             builder.norm.baseline.MANIFEST,
             builder.norm.baseline.MANIFEST.parent / builder.norm.baseline.expected_prefix()["filename"]]
    before = {path: path.read_bytes() for path in paths}
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(notebook["cells"]) == 26 and len({c["id"] for c in notebook["cells"]}) == 26
    original = builder.norm.build_notebook()
    for i in (8, 9, 11, 19, 21, 23):
        assert notebook["cells"][i]["source"] == original["cells"][i]["source"]
    code = code_cells()
    combined = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = builder.norm.baseline.prefix.source_of(cell)
            ast.parse(source)
            combined.append(source)
            assert not cell["outputs"] and cell["execution_count"] is None
    ast.parse("\n".join(combined))
    assert "load_florence" not in "\n".join(combined) and "import ot" not in "\n".join(combined)
    assert 'DTD_PREFIX_MLTM_ALIGNMENT_EPOCHS", "50"' in code[4]
    assert builder.PREFIX + "_vitl16" in code[4]
    assert "validate_alignment_settings()" in code[17]
    assert '"alignment": alignment_protocol_config()' in code[17]
    assert "make_mltm_alignment_model_class(MTACSModel" in code[17]
    assert "write_alignment_manifest(banks, run_dir)" in code[14]
    utility_nodes = [node for node in ast.parse(code[6]).body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "HELPER_SOURCE_SHA256"]
    actual = hashlib.sha256("".join(code[i] for i in (8, 9, 11, 12, 14, 15)).encode()).hexdigest()
    assert ast.literal_eval(utility_nodes[0].value) == actual
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["alignment"] and metadata["norm_calibrated_fusion"] and not metadata["partial_ot"]
    assert not metadata["query_fusion"] and not metadata["semantic_selector"]


def test_config_and_output_guard_separate_from_study02(tmp_path, monkeypatch):
    monkeypatch.delenv("DTD_PREFIX_MLTM_ALIGNMENT_SEED", raising=False)
    monkeypatch.setenv("DTD_PREFIX_NORM_FUSION_SEED", "99")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    ns["validate_alignment_settings"]()
    assert ns["TRAIN_SEEDS"] == [42] and ns["ALIGNMENT_WEIGHT"] == .05
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and not ns["GENERATE_CAPTIONS_IF_MISSING"]
    assert ns["ARCHITECTURE_STUDY"]["alignment"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"schema": "dtd_prefix_norm_fusion_v1"}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="another experiment"):
        ns["validate_alignment_output_directory"](tmp_path)
    assert path.read_bytes() == before
    ns["ALIGNMENT_WEIGHT"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ns["validate_alignment_settings"]()


def test_sinkhorn_matches_closed_form_and_has_uniform_marginals_and_gradients(toy_runtime):
    ns = runtime(toy_runtime)
    cost = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64, requires_grad=True)
    plan = ns["uniform_log_sinkhorn"](cost, .5, 50).exp()
    q = math.exp(-2)
    expected = torch.tensor([[1., q], [q, 1.]], dtype=torch.float64) / (2 * (1 + q))
    torch.testing.assert_close(plan, expected)
    torch.testing.assert_close(plan.sum(0), torch.full((2,), .5, dtype=torch.float64))
    target, _, _ = ns["caption_block_target"](["first caption", "second caption"], cost.device, cost.dtype)
    loss = (target * (target.clamp_min(1e-30).log() - plan.log())).sum()
    assert float(loss.detach()) == pytest.approx(math.log1p(q))
    loss.backward()
    assert torch.isfinite(cost.grad).all() and cost.grad.abs().sum() > 0
    small = torch.tensor([[.1, .2], [.3, .15]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: ns["uniform_log_sinkhorn"](x, .2, 30), (small,))
    with pytest.raises(ValueError, match="square"):
        ns["uniform_log_sinkhorn"](torch.ones(2, 3), .1, 10)


def test_duplicate_caption_target_is_balanced_without_false_one_to_one_negatives(toy_runtime):
    ns = runtime(toy_runtime)
    target, positives, groups = ns["caption_block_target"](
        ["Pale Rough Material", "pale rough material", "dark smooth material"], torch.device("cpu"), torch.float64)
    assert groups == 2 and positives[0, 1] and not positives[0, 2]
    assert target[0, 0] == target[0, 1] == pytest.approx(1 / 6)
    assert target[2, 2] == pytest.approx(1 / 3)
    torch.testing.assert_close(target.sum(0), torch.full((3,), 1 / 3, dtype=torch.float64))
    torch.testing.assert_close(target.sum(1), torch.full((3,), 1 / 3, dtype=torch.float64))


def test_alignment_gradients_reach_shared_text_and_metric_not_image_teacher_or_query(toy_runtime):
    ns = runtime(toy_runtime)
    network = ns["create_ablation_model"](42).train()
    images = torch.randn(3, 8, requires_grad=True)
    texts = torch.randn(3, 4, requires_grad=True)
    before_rng = torch.get_rng_state().clone()
    result = ns["alignment_transport_objective"](network, images, texts,
        ["rough pale material", "smooth dark material", "tiny bright marks"])
    assert torch.equal(before_rng, torch.get_rng_state())
    assert result["loss"].isfinite() and result["marginal_error"] < ns["ALIGNMENT_MAX_MARGINAL_ERROR"]
    result["loss"].backward()
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert network.alignment_log_diagonal.grad.abs().sum() > 0
    assert images.grad is None and texts.grad is None
    assert network.backbone.weight.grad is None and network.text_projection[0].weight.grad is None
    assert torch.isfinite(network.alignment_log_diagonal.grad).all()
    with torch.no_grad():
        network.alignment_log_diagonal.copy_(torch.linspace(-100, 100, 8))
    diagonal = network.alignment_diagonal()
    assert diagonal.gt(0).all() and torch.isfinite(diagonal).all()
    assert float(diagonal.mean().detach()) == pytest.approx(1.0)


def test_same_common_initialization_fusion_and_selector_with_only_extra_metric_parameters(toy_runtime):
    ns = runtime(toy_runtime)
    network = ns["create_ablation_model"](42).eval()
    ns["seed_ablation"](42)
    control = ns["make_norm_calibrated_model_class"](ToyBase, ToyVisual)(clip_dim=4).eval()
    assert sum(p.numel() for p in network.parameters()) - sum(p.numel() for p in control.parameters()) == 8
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, network.state_dict()[key])
    images, text = torch.randn(2, 3, 2, 2), torch.randn(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    spec = ns["SELECTED_ABLATIONS"][0]
    original = control.forward_ablated(images, images, text, valid, text, valid, spec)
    updated = network.forward_ablated(images, images, text, valid, text, valid, spec)
    torch.testing.assert_close(original["logits"], updated["logits"])
    with torch.no_grad():
        network.alignment_log_diagonal.add_(50)
    torch.testing.assert_close(updated["logits"], network.forward_ablated(images, images, text, valid, text, valid, spec)["logits"])


def test_sampler_is_own_training_pool_only_masked_and_resume_deterministic(toy_runtime):
    ns = runtime(toy_runtime)
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    banks["c"]["valid"][1] = False
    original_rng = np.random.get_state()
    first = ns["alignment_batch_positions"](banks, 42, 3, 2)
    assert set(first.tolist()) == {0, 2}
    assert len(set(first.tolist())) == len(first)
    after_rng = np.random.get_state()
    np.testing.assert_array_equal(original_rng[1], after_rng[1])
    torch.rand(99)
    torch.testing.assert_close(first, ns["alignment_batch_positions"](banks, 42, 3, 2))
    banks["c"]["source_ids"][0] = "val/0"
    with pytest.raises(ValueError, match="own captions"):
        ns["validate_alignment_pool"](banks)


def test_zero_alignment_weight_matches_norm_control_training(toy_runtime, tmp_path):
    control_ns = norm_runtime(toy_runtime)
    control_ns["OUT_PUT_DIR"] = tmp_path / "control"
    control_ns["RUNS_DIR"] = tmp_path / "control/runs"
    expected_dir = control_ns["train_one_configuration"](control_ns["SELECTED_ABLATIONS"][0], 42)
    ns = runtime(toy_runtime)
    ns["ALIGNMENT_WEIGHT"] = 0.
    ns["OUT_PUT_DIR"] = tmp_path / "zero_alignment"
    ns["RUNS_DIR"] = tmp_path / "zero_alignment/runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    expected = pd.read_csv(expected_dir / "history.csv")
    actual = pd.read_csv(actual_dir / "history.csv")
    pd.testing.assert_frame_equal(expected, actual[expected.columns], atol=1e-7, rtol=1e-7)
    assert actual.train_alignment_loss.eq(0).all() and actual.train_alignment_weight.eq(0).all()


def test_epoch_boundary_resume_restores_metric_optimizer_and_sampler(toy_runtime, tmp_path):
    reference = runtime(toy_runtime)
    reference.update(EPOCHS=3, LR_SCHEDULE_EPOCHS=3, KEEP_LAST_CHECKPOINT=True,
                     OUT_PUT_DIR=tmp_path / "reference", RUNS_DIR=tmp_path / "reference/runs")
    full = reference["train_one_configuration"](reference["SELECTED_ABLATIONS"][0], 42)
    ns = runtime(toy_runtime)
    ns.update(EPOCHS=3, LR_SCHEDULE_EPOCHS=3, KEEP_LAST_CHECKPOINT=True,
              OUT_PUT_DIR=tmp_path / "resumed", RUNS_DIR=tmp_path / "resumed/runs")
    original_train = ns["train_ablation_epoch"]

    def stop_at_second_epoch(network, loader, banks, features, optimizer, scaler, epoch):
        if epoch == 2:
            raise RuntimeError("simulated pause after completed epoch")
        return original_train(network, loader, banks, features, optimizer, scaler, epoch)

    ns["train_ablation_epoch"] = stop_at_second_epoch
    with pytest.raises(RuntimeError, match="simulated pause"):
        ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    ns["train_ablation_epoch"] = original_train
    resumed = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    pd.testing.assert_frame_equal(pd.read_csv(full / "history.csv"), pd.read_csv(resumed / "history.csv"),
                                  atol=1e-7, rtol=1e-7)
    a = torch.load(full / "last.pt", weights_only=False, map_location="cpu")
    b = torch.load(resumed / "last.pt", weights_only=False, map_location="cpu")
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key])


def test_actual_training_aux_logging_diagnostics_cache_test_and_report(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    code = code_cells()
    exec(code[19], ns)
    run_dir = ns["RUNS_DIR"] / "candidate_only/seed_42"
    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == 2 and history.train_alignment_loss.gt(0).all()
    assert history.train_alignment_weight.tolist() == pytest.approx([.01, .02])
    assert history.train_alignment_marginal_error_max.lt(ns["ALIGNMENT_MAX_MARGINAL_ERROR"]).all()
    manifest = pd.read_csv(run_dir / "alignment_pair_manifest.csv")
    assert manifest.image_id.equals(manifest.caption_source_id)
    assert set(manifest.image_id) <= set(ns["train_df"].image_id)
    stamp = (run_dir / "best.pt").stat().st_mtime_ns
    exec(code[19], ns)
    assert (run_dir / "best.pt").stat().st_mtime_ns == stamp
    exec(code[21], ns)
    fit_path = run_dir / "validation_norm_diagnostics/training_pool_alignment_summary.csv"
    fit = pd.read_csv(fit_path)
    assert fit.scope.eq("training_pool_fit_not_held_out_retrieval").all()
    assert fit.eligible_pairs.eq(3).all() and fit.metric_diagonal_min.gt(0).all()
    cached = ns["run_norm_validation_interventions"](ns["SELECTED_ABLATIONS"][0], 42)
    assert len(cached) == 3 and not (run_dir / "test_metrics.json").exists()
    exec(code[23], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update(plt=plt, audit_summary=pd.DataFrame({"valid_rate": [1.]}, index=["train"]))
    exec(code[25], ns)
    results = pd.read_csv(ns["OUT_PUT_DIR"] / (builder.PREFIX + "_results.csv"))
    assert results.alignment_enabled.all() and results.norm_calibrated_fusion.all()
    assert results.mean_query_gate.eq(0).all() and set(results.split) == {"val", "test"}
    assert (ns["OUT_PUT_DIR"] / (builder.PREFIX + "_alignment_training.png")).is_file()
    assert (ns["OUT_PUT_DIR"] / "training_pool_alignment_summary.csv").is_file()
    report = (ns["OUT_PUT_DIR"] / (builder.PREFIX + "_report.md")).read_text(encoding="utf-8")
    assert "Auxiliary alignment configuration" in report and "not held-out retrieval" in report
    plt.close("all")
