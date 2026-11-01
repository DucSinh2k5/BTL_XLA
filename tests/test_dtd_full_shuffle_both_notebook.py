
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch

import build_mtacs_dtd_both_fusion_notebook as both
import build_mtacs_dtd_candidate_only_notebook as candidate
import build_mtacs_dtd_full_both_notebook as full
import build_mtacs_dtd_full_shuffle_both_notebook as builder
import build_mtacs_dtd_query_only_notebook as query
import build_mtacs_dtd_visual_only_notebook as visual
import mtacs_dtd_ablation_cells as original
from test_dtd_full_both_notebook import runtime as full_runtime
from test_dtd_query_only_notebook import audited_rows
from test_dtd_text_ablation import spec, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: visual.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = full_runtime(toy_runtime)
    code = code_cells()
    exec(code[2], ns)
    nodes = [node for node in ast.parse(code[9]).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "shuffle_banks", "exec"), ns)
    exec(code[14], ns)
    ns.update({"CLIP_EMBED_DIM": 4, "PROTOCOL_FINGERPRINT": "full-shuffle-both-test"})
    return ns


def test_standalone_single_configuration_defaults_and_prior_artifacts_unchanged(monkeypatch):
    before = {module.TARGET: module.TARGET.read_bytes() for module in (visual, query, candidate, both, full)}
    notebook = builder.build_notebook()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(notebook["cells"]) == 24
    all_code = []
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-full-shuffle-both-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
            assert cell["outputs"] == [] and cell["execution_count"] is None
            all_code.append(visual.source_of(cell))
    ast.parse("\n".join(all_code))
    assert "import mtacs_dtd_ablation_cells" not in "\n".join(all_code)
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "full_shuffle_both" and metadata["caption_generation"]
    assert metadata["mask_preserving_shuffle"] and metadata["default_epochs"] == 50
    code = code_cells()
    assert 'DTD_FULL_SHUFFLE_BOTH_EPOCHS", "50"' in code[4]
    assert "ablation_06_full_shuffle_both_vitl16" in code[4]
    assert "ablation_05_full_both_vitl16" in code[9] and "ensure_dtd_caption_csv" in code[9]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert '"recipient_masks": "preserved_exactly"' in code[17]
    assert '"shuffle_seed_offsets": {"q": 1009, "c": 2017}' in code[17]
    monkeypatch.setenv("DTD_ABLATIONS", "full_both,visual_only")
    ns = {"os": os}
    exec(code[2], ns)
    assert len(ns["SELECTED_ABLATIONS"]) == 1
    mode = ns["SELECTED_ABLATIONS"][0]
    assert mode["name"] == "full_shuffle_both" and mode["q_transform"] == mode["c_transform"] == "shuffle"
    assert mode["query_fusion"] and mode["candidate_fusion"] and mode["semantic_selector"]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and ns["TRAIN_SEEDS"] == [42]
    assert not ns["EARLY_STOPPING"]


def test_fusion_selector_losses_transforms_caption_pipeline_unchanged_from_05():
    previous = {index: visual.source_of(cell) for index, cell in enumerate(full.build_notebook()["cells"])}
    code = code_cells()
    for index in (8, 11, 12, 15):
        assert code[index] == previous[index]
    names = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu", "evaluate_ablation"]
    assert visual.definitions(code[14], names) == visual.definitions(previous[14], names)
    assert visual.definitions(code[9], ["deranged_sources"]) == visual.definitions(
        project_path("mtacs_dtd_ablation_cells.py").read_text(encoding="utf-8"), ["deranged_sources"])
    for name in ("train_transform", "eval_transform"):
        node = next(node for node in ast.parse(previous[4]).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert ast.get_source_segment(previous[4], node) in code[4]
    assert "CANDIDATE_POOL_FRACTION = 0.20" in code[4]
    assert '"selector_sampling": "categorical_policy_on_visual_semantic_z"' in code[17]
    assert "CONTEXT_TOP_K != 1" in code[17]


def test_each_role_deranges_only_valid_captions_within_split_and_pool_and_preserves_masks(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    view = ns["TEXT_CACHE"]["views"]["screened"]
    before_vectors, before_valid = view["embeddings"].clone(), view["valid"].clone()
    banks = ns["make_ablation_banks"](mode, 42)
    split_by_id = {str(image_id): split for split, frame in ns["SPLIT_FRAMES"].items() for image_id in frame.image_id}
    lookup = {image_id: index for index, image_id in enumerate(view["image_ids"])}
    pool = set(ns["CANDIDATE_IDS"])
    for role, offset, recipients in (("q", 1009, view["image_ids"]), ("c", 2017, ns["CANDIDATE_IDS"])):
        expected_indices, _ = original.deranged_sources(view["image_ids"], before_valid, split_by_id,
                                                        ns["CANDIDATE_IDS"], 42 + offset)
        for index, receiver in enumerate(recipients):
            source = banks[role]["source_ids"][index]
            original_position = lookup[receiver]
            assert source == view["image_ids"][expected_indices[original_position]]
            assert split_by_id[source] == split_by_id[receiver]
            assert (source in pool) == (receiver in pool)
            assert bool(banks[role]["valid"][index]) == bool(before_valid[original_position])
            assert banks[role]["texts"][index] == view["texts"][lookup[source]]
            torch.testing.assert_close(banks[role]["embeddings"][index], before_vectors[lookup[source]])
            if before_valid[original_position]:
                assert source != receiver
            else:
                assert source == receiver and banks[role]["embeddings"][index].count_nonzero() == 0
    assert set(banks["c"]["source_ids"]) <= pool
    torch.testing.assert_close(view["embeddings"], before_vectors)
    torch.testing.assert_close(view["valid"], before_valid)
    table = ns["shuffle_role_inputs"](banks)
    assert table.caption_available.equals(table.original_caption_available)
    assert table.caption_available.equals(table.donor_caption_available)
    assert table.loc[table.caption_available, "source_id_changed"].all()
    assert table.split.equals(table.caption_source_split)


def test_shuffling_is_seeded_fixed_label_free_and_does_not_consume_training_rng(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()
    first = ns["make_ablation_banks"](mode, 42)
    second = ns["make_ablation_banks"](mode, 42)
    assert torch.equal(torch.get_rng_state(), torch_state)
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and np.array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    for role in ("q", "c"):
        assert first[role]["source_ids"] == second[role]["source_ids"]
        torch.testing.assert_close(first[role]["embeddings"], second[role]["embeddings"])
    changed = ns["make_ablation_banks"](mode, 43)
    assert first["q"]["source_ids"] != changed["q"]["source_ids"]
    ns["SPLIT_FRAMES"] = {split: frame.assign(class_id=999) for split, frame in ns["SPLIT_FRAMES"].items()}
    without_labels = ns["make_ablation_banks"](mode, 42)
    assert first["q"]["source_ids"] == without_labels["q"]["source_ids"]
    assert first["c"]["source_ids"] == without_labels["c"]["source_ids"]


def test_singletons_stay_aligned_without_changing_availability_and_are_reported(toy_runtime):
    ns = runtime(toy_runtime)
    view = ns["TEXT_CACHE"]["views"]["screened"]
    singletons = {"train/0", "train/3", "val/0", "test/0"}
    view["valid"] = torch.tensor([image_id in singletons for image_id in view["image_ids"]])
    view["embeddings"][~view["valid"]] = 0
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    assert set(banks["singletons"]["q"]) == singletons
    assert banks["singletons"]["c"] == ["train/0"]
    assert banks["q"]["source_ids"] == view["image_ids"]
    assert banks["c"]["source_ids"] == ns["CANDIDATE_IDS"]
    table = ns["shuffle_role_inputs"](banks)
    assert table.loc[table.caption_available, "singleton_not_deranged"].all()
    assert not table.source_id_changed.any()
    assert table.caption_available.sum() == len(singletons) + 1


def test_summary_distinguishes_changed_donor_ids_from_identical_text(toy_runtime):
    ns = runtime(toy_runtime)
    view = ns["TEXT_CACHE"]["views"]["screened"]
    view["texts"] = ["Identical valid caption for every source"] * len(view["image_ids"])
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    table = ns["shuffle_role_inputs"](banks)
    run_dir = ns["RUNS_DIR"] / "full_shuffle_both" / "seed_42"
    run_dir.mkdir(parents=True)
    table.to_csv(run_dir / "shuffle_model_inputs.csv", index=False)
    ns["markdown_table"] = lambda frame: frame.to_csv(index=False)
    exec(builder.SHUFFLE_REPORT, ns)
    summary = ns["shuffle_summary"]
    assert summary.mask_preserved_rate.eq(1).all()
    assert summary.changed_source_id_rate_valid.eq(1).all()
    assert summary.changed_text_rate_valid.eq(0).all()


def test_shuffled_vectors_reach_both_fusion_and_selector_but_images_stay_aligned(toy_runtime):
    ns = runtime(toy_runtime)
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    network = ns["create_ablation_model"](42).eval()
    selector = network.select_context
    calls = []

    def spy(query_images, feature_bank, q_text, q_valid, supplied_banks, *args, **kwargs):
        calls.append((q_text.clone(), supplied_banks["c"]["embeddings"].clone(), kwargs["semantic"]))
        return selector(query_images, feature_bank, q_text, q_valid, supplied_banks, *args, **kwargs)

    network.select_context = spy
    batch = next(iter(ns["make_ablation_loaders"](42)["train"]))
    output, details, indices, inputs = ns["run_forward"](network, batch, banks, ns["refresh_ablation_bank"](network), training=True)
    expected_query, _ = ns["batch_caption_inputs"](banks, batch["image_id"])
    assert calls[0][2] is True
    torch.testing.assert_close(calls[0][0], expected_query)
    torch.testing.assert_close(calls[0][1], banks["c"]["embeddings"])
    torch.testing.assert_close(inputs[1], expected_query)
    torch.testing.assert_close(inputs[3], banks["c"]["embeddings"][indices])
    torch.testing.assert_close(inputs[5], ns["candidate_image_bank"][indices])
    excluded = torch.tensor([ns["CANDIDATE_POSITION"].get(image_id, -1) for image_id in batch["image_id"]])
    assert indices.ne(excluded).all()
    loss = output["logits"].sum() - details["log_prob"].mean()
    loss.backward()
    assert network.text_projection[0].weight.grad.abs().sum() > 0
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(network.selector.semantic_mix_logit.grad)


def test_cache_id_order_alignment_and_invalid_settings_are_checked(toy_runtime):
    ns = runtime(toy_runtime)
    view = ns["TEXT_CACHE"]["views"]["screened"]
    order = torch.arange(len(view["image_ids"]) - 1, -1, -1)
    ns["TEXT_CACHE"] = {"views": {"screened": {
        "image_ids": [view["image_ids"][int(i)] for i in order], "texts": [view["texts"][int(i)] for i in order],
        "valid": view["valid"][order], "embeddings": view["embeddings"][order]}}}
    mode = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](mode, 42)
    table = ns["shuffle_role_inputs"](banks)
    assert table.caption_available.equals(table.original_caption_available)
    assert set(table.loc[table.role.eq("c"), "caption_source_id"]) <= set(ns["CANDIDATE_IDS"])
    with pytest.raises(ValueError, match="full_shuffle_both only"):
        ns["make_ablation_banks"](mode, 42, {"q_transform": "aligned"})
    ns["CANDIDATE_IDS"] = ["test/0"]
    with pytest.raises(ValueError, match="training images only"):
        ns["make_ablation_banks"](mode, 42)


@pytest.mark.parametrize("source_kind", ["fresh", "partial_cached", "complete_attached"])
def test_caption_generation_resume_and_reuse_leave_source_csv_unshuffled(toy_runtime, tmp_path, source_kind):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows["raw_caption"] = [f"The image is a close-up of a pale rough fabric with tiny marks{i}." for i in range(len(rows))]
    rows["final_caption"] = "The image is a close up of a"
    rows["caption_valid"], rows["leakage_detected"] = False, True
    rows["semantic_leakage_detected"], rows["semantic_leakage_score"] = True, .913577
    rows.loc[1, ["semantic_leakage_detected", "leakage_detected", "semantic_leakage_score"]] = [False, False, .1]
    cached = ns["OUT_PUT_DIR"] / "shared" / "captions.csv"
    cached.parent.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "attached_captions.csv"
    if source_kind == "complete_attached":
        rows.to_csv(source, index=False)
    elif source_kind == "partial_cached":
        rows.iloc[:2].to_csv(cached, index=False)
    generated = []

    def generate(frames, output, clip_asset):
        assert source_kind != "complete_attached"
        existing = pd.read_csv(output) if output.is_file() else rows.iloc[:0]
        generated.append(len(existing))
        pd.concat([existing, rows.loc[~rows.image_id.isin(existing.image_id)]], ignore_index=True).to_csv(output, index=False)

    def encode(views, *args):
        valid = torch.as_tensor(views["screened"]["valid"], dtype=torch.bool)
        matrix = torch.arange(len(valid) * 512, dtype=torch.float32).reshape(len(valid), 512)
        matrix[~valid] = 0
        return {"embedding_dim": 512, "fingerprint": "pre-shuffle-text", "views": {
            "screened": {**views["screened"], "embeddings": matrix, "valid": valid}}}

    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CAPTIONS_INPUT_PATH": str(source) if source.is_file() else "",
               "resolve_clip_path": lambda *args: tmp_path / "clip", "run_local_dtd_caption_generation": generate,
               "encode_view_cache": encode})
    exec(code_cells()[9], ns)
    before = ns["CAPTION_CSV"].read_bytes()
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    table = ns["shuffle_role_inputs"](banks)
    assert table.caption_available.equals(table.original_caption_available)
    assert table.loc[table.caption_available, "source_id_changed"].all()
    assert not banks["q"]["valid"][1] and not banks["c"]["valid"][1]
    assert banks["q"]["embeddings"][1].eq(0).all() and banks["c"]["embeddings"][1].eq(0).all()
    assert ns["CAPTION_CSV"].read_bytes() == before
    assert (ns["SHARED_DIR"] / "caption_source_inputs.csv").is_file()
    exec(code_cells()[9], ns)
    assert ns["CAPTION_CSV"].read_bytes() == before
    assert generated == ([] if source_kind == "complete_attached" else [2 if source_kind == "partial_cached" else 0])
    if source.is_file():
        assert source.read_bytes() == before


def test_training_resume_final_test_and_actual_caption_report(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    code = code_cells()
    mode = ns["SELECTED_ABLATIONS"][0]
    exec(code[19], ns)
    run_dir = ns["RUNS_DIR"] / "full_shuffle_both" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"full_shuffle_both"}
    assert (run_dir / "shuffle_text_manifest.json").is_file()
    assignments = pd.read_csv(run_dir / "shuffle_model_inputs.csv")
    assert assignments.caption_available.equals(assignments.original_caption_available)
    assert assignments.loc[assignments.caption_available, "source_id_changed"].all()
    stamp = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](mode, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == stamp
    exec(code[21], ns)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    assert predictions.query_caption_source_id.ne(predictions.image_id).all()
    assert predictions.candidate_caption_source_id.ne(predictions.selected_context_id).all()
    assert set(predictions.candidate_caption_source_id) <= set(ns["CANDIDATE_IDS"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update({"plt": plt, "audit_summary": pd.DataFrame({"valid_rate": [1.]}, index=["train"]),
               "role_coverage": pd.DataFrame({"valid_rate": [1.]},
                   index=pd.MultiIndex.from_tuples([("candidate_pool", "train")], names=["role", "split"]))})
    exec(code[23], ns)
    metrics = pd.read_csv(ns["OUT_PUT_DIR"] / "full_shuffle_both_results.csv")
    assert set(metrics.split) == {"val", "test"} and metrics.selector_semantic_mix.between(0, 1).all()
    summary = pd.read_csv(ns["OUT_PUT_DIR"] / "full_shuffle_both_shuffle_summary.csv")
    assert summary.mask_preserved_rate.eq(1).all()
    assert summary.changed_source_id_rate_valid.eq(1).all()
    report = (ns["OUT_PUT_DIR"] / "full_shuffle_both_report.md").read_text()
    assert "Mask-preserving caption shuffle" in report and "dtd_lexical_v2" in report
    plt.close("all")


def test_interrupted_run_resumes_same_caption_maps_rng_and_metrics(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "uninterrupted"
    first = ns["train_one_configuration"](mode, 42)
    expected_history = pd.read_csv(first / "history.csv")
    expected_inputs = (first / "shuffle_model_inputs.csv").read_bytes()
    original_train = ns["train_ablation_epoch"]
    calls = []

    def interrupt_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("test interruption")
        return original_train(*args, **kwargs)

    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "interrupted"
    ns["train_ablation_epoch"] = interrupt_second
    with pytest.raises(RuntimeError, match="test interruption"):
        ns["train_one_configuration"](mode, 42)
    ns["train_ablation_epoch"] = original_train
    resumed = ns["train_one_configuration"](mode, 42)
    actual_history = pd.read_csv(resumed / "history.csv")
    pd.testing.assert_frame_equal(expected_history, actual_history, atol=1e-7, rtol=1e-7)
    assert (resumed / "shuffle_model_inputs.csv").read_bytes() == expected_inputs


def test_training_math_matches_original_shuffle_suite_configuration(toy_runtime):
    mode = spec("full_shuffle_both", q_transform="shuffle", c_transform="shuffle")
    original_dir = original.train_one_configuration(mode, 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "new_shuffle_runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
