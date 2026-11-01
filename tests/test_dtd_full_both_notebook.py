
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import os
from functools import lru_cache

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import build_mtacs_dtd_both_fusion_notebook as both
import build_mtacs_dtd_candidate_only_notebook as candidate
import build_mtacs_dtd_full_both_notebook as builder
import build_mtacs_dtd_query_only_notebook as query
import build_mtacs_dtd_visual_only_notebook as visual
import mtacs_dtd_ablation_cells as original
from test_dtd_both_fusion_notebook import runtime as both_runtime
from test_dtd_query_only_notebook import audited_rows
from test_dtd_text_ablation import spec, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: visual.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = both_runtime(toy_runtime)
    code = code_cells()
    exec(code[2], ns)
    nodes = [node for node in ast.parse(code[9]).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "full_both_banks", "exec"), ns)
    exec(code[14], ns)
    exec(code[15], ns)
    ns.update({"CLIP_EMBED_DIM": 4, "PROTOCOL_FINGERPRINT": "full-both-test"})
    return ns


def controlled_selector(ns):
    network = ns["create_ablation_model"](42).eval()
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    banks["c"]["embeddings"] = torch.tensor([[-1., 0, 0, 0], [1., 0, 0, 0], [0., 1, 0, 0]])
    banks["c"]["valid"] = torch.ones(3, dtype=torch.bool)
    visual_scores = torch.tensor([[1., .99, -.5]], requires_grad=True)
    network.score_full_candidate_pool = lambda images, bank: (
        visual_scores.expand(len(images), -1), torch.zeros(len(images), 8))
    images = torch.zeros(1, 3, 2, 2)
    text = torch.tensor([[1., 0, 0, 0]])
    return network, banks, images, text, visual_scores


def select(network, banks, images, text, semantic=True, valid=None, excluded=None, sample=False):
    return network.select_context(images, torch.zeros(3, 8), text,
        valid if valid is not None else torch.ones(len(images), dtype=torch.bool), banks,
        excluded if excluded is not None else torch.full((len(images),), -1, dtype=torch.long),
        semantic=semantic, sample=sample)


def test_only_full_both_is_selected_with_requested_defaults_and_prior_files_unchanged(monkeypatch):
    before = {module.TARGET: module.TARGET.read_bytes() for module in (visual, query, candidate, both)}
    notebook = builder.build_notebook()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(notebook["cells"]) == 24
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-full-both-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "full_both" and metadata["caption_generation"]
    assert metadata["default_epochs"] == 50 and metadata["default_caption_input_policy"] == "dtd_lexical_v2"
    code = code_cells()
    assert 'DTD_FULL_BOTH_EPOCHS", "50"' in code[4]
    assert "ablation_05_full_both_vitl16" in code[4]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert code[8] == visual.source_of(both.build_notebook()["cells"][8])
    assert "ensure_dtd_caption_csv" in code[9] and "ablation_04_both_fusion_vitl16" in code[9]
    assert '"selector_sampling": "categorical_policy_on_visual_semantic_z"' in code[17]
    assert '"caption_input_policy": CAPTION_INPUT_POLICY' in code[17]
    assert '"invalid_semantic_pairs": "zero_contribution"' in code[17]
    monkeypatch.setenv("DTD_ABLATIONS", "visual_only,both_fusion")
    ns = {"os": os}
    exec(code[2], ns)
    assert len(ns["SELECTED_ABLATIONS"]) == 1
    mode = ns["SELECTED_ABLATIONS"][0]
    assert mode["name"] == "full_both" and mode["query_fusion"] and mode["candidate_fusion"] and mode["semantic_selector"]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and ns["TRAIN_SEEDS"] == [42]
    assert not ns["EARLY_STOPPING"]


def test_architecture_fusion_losses_transforms_and_caption_pipeline_match_notebook_04():
    previous = {index: visual.source_of(cell) for index, cell in enumerate(both.build_notebook()["cells"])}
    code = code_cells()
    assert code[11] == previous[11] and code[12] == previous[12]
    names = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu"]
    assert visual.definitions(code[14], names) == visual.definitions(previous[14], names)
    for name in ("train_transform", "eval_transform"):
        node = next(node for node in ast.parse(previous[4]).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert ast.get_source_segment(previous[4], node) in code[4]
    assert "CANDIDATE_POOL_FRACTION = 0.20" in code[4]
    assert "CONTEXT_TOP_K != 1" in code[17]


def test_text_scores_change_selection_and_match_the_documented_formula(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, images, text, visual = controlled_selector(ns)
    assert torch.sigmoid(network.selector.semantic_mix_logit).item() == .5
    image_only = select(network, banks, images, text, semantic=False)
    mixed = select(network, banks, images, text)
    reversed_query = select(network, banks, images, -text)
    assert int(image_only["indices"][0]) == 0 and int(mixed["indices"][0]) == 1
    assert int(reversed_query["indices"][0]) == 0
    visual_z = (visual - visual.mean(1, keepdim=True)) / visual.std(1, keepdim=True, unbiased=False)
    semantic = F.normalize(text, dim=1) @ F.normalize(banks["c"]["embeddings"], dim=1).T
    semantic_z = (semantic - semantic.mean(1, keepdim=True)) / semantic.std(1, keepdim=True, unbiased=False)
    distribution = torch.distributions.Categorical(logits=visual_z + .5 * semantic_z)
    torch.testing.assert_close(mixed["log_prob"], distribution.log_prob(mixed["indices"]))
    torch.testing.assert_close(mixed["entropy"], distribution.entropy().mean())
    changed = {**banks, "c": {**banks["c"], "embeddings": -banks["c"]["embeddings"]}}
    assert int(select(network, changed, images, text)["indices"][0]) == 0
    with torch.no_grad():
        network.selector.semantic_mix_logit.fill_(-20)
    assert int(select(network, banks, images, text)["indices"][0]) == 0


def test_policy_gradient_updates_semantic_mix_and_visual_scores(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, images, text, visual = controlled_selector(ns)
    parameter = network.selector.semantic_mix_logit
    optimizer = torch.optim.SGD([parameter], lr=.1)
    details = select(network, banks, images, text)
    loss = -details["log_prob"].mean() - .01 * details["entropy"]
    loss.backward()
    assert torch.isfinite(parameter.grad) and parameter.grad.abs() > 0
    assert torch.isfinite(visual.grad).all() and visual.grad.abs().sum() > 0
    optimizer.step()
    assert parameter.item() != 0.0


def test_invalid_text_pairs_fall_back_to_visual_and_never_affect_scores(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, images, text, _ = controlled_selector(ns)
    visual_only = select(network, banks, images, text, semantic=False)
    invalid_query = select(network, banks, images, text, valid=torch.zeros(1, dtype=torch.bool))
    no_candidate_text = {**banks, "c": {**banks["c"], "valid": torch.zeros(3, dtype=torch.bool)}}
    invalid_pool = select(network, no_candidate_text, images, text)
    for details in (invalid_query, invalid_pool):
        for key in ("indices", "log_prob", "entropy"):
            torch.testing.assert_close(details[key], visual_only[key])
    banks["c"]["valid"][2] = False
    masked = select(network, banks, images, text)
    other = {**banks, "c": {**banks["c"], "embeddings": banks["c"]["embeddings"].clone()}}
    other["c"]["embeddings"][2] = torch.tensor([999., -12., 53., 1.])
    changed_invalid = select(network, other, images, text)
    torch.testing.assert_close(masked["log_prob"], changed_invalid["log_prob"])
    assert torch.isfinite(masked["entropy"])
    banks["c"]["valid"] = torch.tensor([False, True, False])
    singleton = select(network, banks, images, text)
    torch.testing.assert_close(singleton["log_prob"], visual_only["log_prob"])


def test_self_is_excluded_even_when_its_caption_has_best_similarity(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, images, text, _ = controlled_selector(ns)
    images, text = images.repeat(32, 1, 1, 1), text.repeat(32, 1)
    excluded = torch.ones(32, dtype=torch.long)
    for sample in (False, True):
        details = select(network, banks, images, text, excluded=excluded, sample=sample)
        assert details["indices"].ne(excluded).all()
        assert torch.isfinite(details["log_prob"]).all()


def test_banks_are_id_aligned_and_run_forward_enables_semantic_selection(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    original_view = ns["TEXT_CACHE"]["views"]["screened"]
    order = torch.arange(17, -1, -1)
    ns["TEXT_CACHE"] = {"views": {"screened": {
        "image_ids": [original_view["image_ids"][int(i)] for i in order],
        "texts": [original_view["texts"][int(i)] for i in order],
        "valid": original_view["valid"][order], "embeddings": original_view["embeddings"][order]}}}
    banks = ns["make_ablation_banks"](mode, 42)
    torch.testing.assert_close(banks["c"]["embeddings"], original_view["embeddings"][:3])
    assert banks["c"]["source_ids"] == ns["CANDIDATE_IDS"]
    with pytest.raises(ValueError, match="full_both only"):
        ns["make_ablation_banks"](mode, 42, {"semantic_selector": False})
    network = ns["create_ablation_model"](42).eval()
    original_selector = network.select_context
    calls = []

    def selector(*args, **kwargs):
        calls.append(kwargs["semantic"])
        return original_selector(*args, **kwargs)

    network.select_context = selector
    batch = next(iter(ns["make_ablation_loaders"](42)["train"]))
    output, details, indices, inputs = ns["run_forward"](network, batch, banks, ns["refresh_ablation_bank"](network), training=True)
    assert calls == [True]
    torch.testing.assert_close(inputs[3], banks["c"]["embeddings"][indices])
    torch.testing.assert_close(inputs[5], ns["candidate_image_bank"][indices])
    loss = output["logits"].sum() - details["log_prob"].mean()
    loss.backward()
    assert network.text_projection[0].weight.grad.abs().sum() > 0
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(network.selector.semantic_mix_logit.grad)
    ns["CANDIDATE_IDS"] = ["test/0"]
    with pytest.raises(ValueError, match="training images only"):
        ns["make_ablation_banks"](mode, 42)


@pytest.mark.parametrize("source_kind", ["fresh", "partial_cached", "complete_attached"])
def test_caption_creation_resume_and_reuse_keep_both_roles_and_policy(toy_runtime, tmp_path, source_kind):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows["raw_caption"] = "The image is a close-up of a pale rough material with tiny dark marks."
    rows["final_caption"] = "The image is a close up of a"
    rows["caption_valid"] = False
    rows["leakage_detected"] = True
    rows["semantic_leakage_detected"] = True
    rows["semantic_leakage_score"] = .913577
    cached = ns["OUT_PUT_DIR"] / "shared" / "captions.csv"
    cached.parent.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "attached_captions.csv"
    if source_kind == "complete_attached":
        rows.to_csv(source, index=False)
    elif source_kind == "partial_cached":
        rows.iloc[:2].to_csv(cached, index=False)
    calls = []

    def generate(frames, output, clip_asset):
        assert source_kind != "complete_attached"
        existing = pd.read_csv(output) if output.is_file() else rows.iloc[:0]
        calls.append(len(existing))
        remaining = rows.loc[~rows.image_id.isin(existing.image_id)]
        pd.concat([existing, remaining], ignore_index=True).to_csv(output, index=False)

    def encode(views, *args):
        return {"embedding_dim": 512, "fingerprint": "cleaned-both-roles", "views": {
            name: {**view, "embeddings": torch.ones(len(view["texts"]), 512),
                   "valid": torch.as_tensor(view["valid"], dtype=torch.bool)} for name, view in views.items()}}

    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1,
               "CAPTIONS_INPUT_PATH": str(source) if source.is_file() else "",
               "resolve_clip_path": lambda *args: tmp_path / "clip",
               "run_local_dtd_caption_generation": generate, "encode_view_cache": encode})
    exec(code_cells()[9], ns)
    assert calls == ([] if source_kind == "complete_attached" else [2 if source_kind == "partial_cached" else 0])
    assert ns["caption_audit"].screened_valid.all()
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2"
    assert all("close up" not in text.lower() for text in ns["TEXT_CACHE"]["views"]["screened"]["texts"])
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    assert banks["q"]["embeddings"].shape == (18, 512) and banks["c"]["embeddings"].shape == (3, 512)
    before = ns["CAPTION_CSV"].read_bytes()
    exec(code_cells()[9], ns)
    assert ns["CAPTION_CSV"].read_bytes() == before
    assert len(calls) == int(source_kind != "complete_attached")
    if source.is_file():
        assert ns["CAPTION_CSV"].read_bytes() == source.read_bytes()


@pytest.mark.parametrize("missing_role", ["query", "pool"])
def test_all_invalid_branch_stops_before_training(toy_runtime, tmp_path, missing_role):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    missing = rows.split.eq("train") if missing_role == "query" else rows.image_id.isin(ns["CANDIDATE_IDS"])
    rows.loc[missing, "caption_valid"] = False
    source = tmp_path / "invalid.csv"
    rows.to_csv(source, index=False)
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CAPTIONS_INPUT_PATH": str(source)})
    message = "No eligible training captions" if missing_role == "query" else "No eligible candidate-pool captions"
    with pytest.raises(ValueError, match=message):
        exec(code_cells()[9], ns)


def test_training_resume_final_test_report_and_semantic_mix_metrics(toy_runtime, monkeypatch, capsys):
    ns = runtime(toy_runtime)
    code = code_cells()
    mode = ns["SELECTED_ABLATIONS"][0]
    exec(code[19], ns)
    assert "sem_mix=" in capsys.readouterr().out
    run_dir = ns["RUNS_DIR"] / "full_both" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"full_both"}
    assert (run_dir / "full_text_manifest.json").is_file() and (run_dir / "full_text_assignments.csv").is_file()
    history = pd.read_csv(run_dir / "history.csv")
    assert history.val_selector_semantic_mix.between(0, 1).all()
    stamp = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](mode, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == stamp
    exec(code[21], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update({"plt": plt, "audit_summary": pd.DataFrame({"valid_rate": [1.]}, index=["train"]),
               "role_coverage": pd.DataFrame({"valid_rate": [1.]},
                   index=pd.MultiIndex.from_tuples([("candidate_pool", "train")], names=["role", "split"]))})
    exec(code[23], ns)
    results = pd.read_csv(ns["OUT_PUT_DIR"] / "full_both_results.csv")
    assert set(results.split) == {"val", "test"}
    assert results.mean_query_gate.gt(0).all() and results.mean_candidate_gate.gt(0).all()
    assert results.selector_semantic_mix.between(0, 1).all()
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    best_mix = torch.sigmoid(best["model"]["selector.semantic_mix_logit"]).item()
    assert results.loc[results.split.eq("test"), "selector_semantic_mix"].iloc[0] == pytest.approx(best_mix)
    report = (ns["OUT_PUT_DIR"] / "full_both_report.md").read_text()
    assert "Multimodal selector" in report and "dtd_lexical_v2" in report and "selector_semantic_mix" in report
    plt.close("all")


def test_training_math_matches_original_full_both_suite_configuration(toy_runtime):
    original_dir = original.train_one_configuration(spec("full_both"), 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "full_both_runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
