
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import json
import os
import sys
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import build_mtacs_dtd_both_fusion_notebook as both
import build_mtacs_dtd_candidate_only_notebook as candidate
import build_mtacs_dtd_full_both_notebook as full
import build_mtacs_dtd_full_shuffle_both_notebook as shuffle
import build_mtacs_dtd_full_unscreened_notebook as builder
import build_mtacs_dtd_query_only_notebook as query
import build_mtacs_dtd_visual_only_notebook as visual
import mtacs_dtd_ablation_cells as original
from test_dtd_full_both_notebook import controlled_selector, runtime as full_runtime, select
from test_dtd_query_only_notebook import audited_rows
from test_dtd_text_ablation import spec, toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {index: visual.source_of(cell) for index, cell in enumerate(builder.build_notebook()["cells"])}


def runtime(toy_runtime):
    ns = full_runtime(toy_runtime)
    code = code_cells()
    exec(code[2], ns)
    exec(code[8], ns)
    nodes = [node for node in ast.parse(code[9]).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "unscreened_banks", "exec"), ns)
    exec(code[14], ns)
    ns.update(CLIP_EMBED_DIM=4, PROTOCOL_FINGERPRINT="full-unscreened-test")
    return ns


def test_saved_artifact_defaults_and_prior_notebooks_are_unchanged(monkeypatch):
    before = {module.TARGET: module.TARGET.read_bytes()
              for module in (visual, query, candidate, both, full, shuffle)}
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(notebook["cells"]) == 24
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-full-unscreened-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "full_unscreened" and metadata["caption_generation"]
    assert metadata["default_epochs"] == 50 and metadata["default_caption_view"] == "unscreened"
    assert metadata["default_caption_input_policy"] == "dtd_lexical_v2" and not metadata["leakage_masks_applied"]
    code = code_cells()
    assert 'DTD_FULL_UNSCREENED_EPOCHS", "50"' in code[4]
    assert "ablation_07_full_unscreened_vitl16" in code[4]
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code[1]
    assert "ablation_05_full_both_vitl16" in code[9] and "ablation_06_full_shuffle_both_vitl16" in code[9]
    assert "--no-index" in code[8] and "ensure_dtd_caption_csv" in code[9]
    assert "build_mtacs" not in "\n".join(code[i] for i in code if notebook["cells"][i]["cell_type"] == "code")
    monkeypatch.setenv("DTD_ABLATIONS", "visual_only,full_both")
    ns = {"os": os}
    exec(code[2], ns)
    assert len(ns["SELECTED_ABLATIONS"]) == 1
    mode = ns["SELECTED_ABLATIONS"][0]
    assert mode == spec("full_unscreened", view="unscreened")
    assert ns["TRAIN_SEEDS"] == [42] and not ns["EARLY_STOPPING"]
    assert '"caption_assignment": "aligned_no_shuffle"' in code[17]
    assert '"leakage_masks_applied": False' in code[17]


def test_architecture_training_math_and_generation_match_notebook_05():
    previous = {index: visual.source_of(cell) for index, cell in enumerate(full.build_notebook()["cells"])}
    code = code_cells()
    assert code[11] == previous[11] and code[12] == previous[12]
    names = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu"]
    assert visual.definitions(code[14], names) == visual.definitions(previous[14], names)
    assert code[8].startswith(previous[8])
    for name in ("train_transform", "eval_transform"):
        node = next(node for node in ast.parse(previous[4]).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert ast.get_source_segment(previous[4], node) in code[4]
    assert "CANDIDATE_POOL_FRACTION = 0.20" in code[4] and "CONTEXT_TOP_K != 1" in code[17]


def test_same_cleaned_text_as_05_without_flags_class_or_redaction_masks(toy_runtime):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows["raw_caption"] = "The image is a close-up of a pale rough material with tiny dark marks."
    rows.loc[0, "raw_caption"] = "banded banded banded banded banded"
    rows["caption_valid"] = False
    rows["leakage_detected"] = True
    rows["semantic_leakage_detected"] = False
    rows["semantic_leakage_score"] = np.nan
    before = rows.copy(deep=True)
    reference, old_views = ns["prepare_standalone_caption_views"](rows, ["banded"], .3, "dtd_lexical_v2")
    audit, views = ns["prepare_unscreened_caption_views"](rows, ["banded"], .3, "dtd_lexical_v2")
    pd.testing.assert_frame_equal(before, rows)
    assert audit.final_caption.tolist() == reference.final_caption.tolist()
    assert not reference.screened_valid.any() and audit.unscreened_valid.all()
    assert not old_views["unscreened"]["valid"][0] and views["unscreened"]["valid"][0]
    assert views["unscreened"]["texts"][0] == "banded banded banded banded banded"
    assert audit.restored_by_unscreened.all() and not audit.leakage_masks_applied.any()
    assert not audit.semantic_score_matches_input.any()


def test_unscreened_availability_does_not_depend_on_flags_scores_or_labels(toy_runtime):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    expected, view = ns["prepare_unscreened_caption_views"](rows, ["banded"], .3, "dtd_lexical_v2")
    rows["caption_valid"] = False
    rows["leakage_detected"] = True
    rows["semantic_leakage_detected"] = True
    rows["semantic_leakage_score"] = [np.nan, .99, -.8] * 6
    rows["class_id"] = list(range(len(rows)))
    changed, actual = ns["prepare_unscreened_caption_views"](rows, ["banded"], -.99, "dtd_lexical_v2")
    assert expected.final_caption.tolist() == changed.final_caption.tolist()
    np.testing.assert_array_equal(view["unscreened"]["valid"], actual["unscreened"]["valid"])
    assert view["unscreened"]["texts"] == actual["unscreened"]["texts"]


def test_empty_short_fallback_and_normalized_word_budget(toy_runtime):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"]).iloc[:5].copy()
    rows["raw_caption"] = ["", "few words", "The image shows a pale rough material with tiny dark marks.",
                           "The image shows it's rough with tiny marks and many more words", ""]
    rows.loc[0, "final_caption"] = ""
    rows.loc[4, "final_caption"] = "pale surface with many tiny dark marks"
    audit, views = ns["prepare_unscreened_caption_views"](rows, ["banded"], .3, "dtd_lexical_v2")
    assert views["unscreened"]["valid"].tolist() == [False, False, True, True, True]
    assert audit.word_count.le(8).all()
    assert audit.loc[4, "model_text_source"] == "final_caption_fallback"
    assert audit.loc[2, "model_text_source"] == "raw_caption_cleaned_and_truncated"
    assert "image shows" not in audit.loc[2, "final_caption"].lower()


def test_comparison_reports_actual_and_reference_masks_by_role(toy_runtime):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows.loc[0, "final_caption"] = "banded surface with many tiny dark marks"
    audit, _ = ns["prepare_unscreened_caption_views"](rows, ["banded"], .3, "dtd_lexical_v2")
    table = ns["unscreened_filter_comparison"](audit, ns["CANDIDATE_IDS"])
    train = table.loc[table.role.eq("query") & table.split.eq("train")].iloc[0]
    pool = table.loc[table.role.eq("candidate_pool")].iloc[0]
    assert train.images == 6 and train.screened_reference_count == 5 and train.unscreened_count == 6
    assert train.additional_usable_count == 1 and train.kept_direct_class_terms == 1
    assert pool.images == 3 and pool.unscreened_count == 3 and pool.additional_usable_count == 1
    assert set(table.split) == {"train", "val", "test"}


def test_banks_preserve_original_image_alignment_and_enable_both_gates_and_selector(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    view = ns["TEXT_CACHE"]["views"]["unscreened"]
    order = torch.arange(17, -1, -1)
    ns["TEXT_CACHE"] = {"views": {"unscreened": {
        "image_ids": [view["image_ids"][int(i)] for i in order],
        "texts": [view["texts"][int(i)] for i in order],
        "valid": view["valid"][order], "embeddings": view["embeddings"][order]}}}
    banks = ns["make_ablation_banks"](mode, 42)
    torch.testing.assert_close(banks["c"]["embeddings"], view["embeddings"][:3])
    assert banks["q"]["source_ids"] == ns["TEXT_CACHE"]["views"]["unscreened"]["image_ids"]
    assert banks["c"]["source_ids"] == ns["CANDIDATE_IDS"] and not banks["singletons"]
    with pytest.raises(ValueError, match="full_unscreened only"):
        ns["make_ablation_banks"](mode, 42, {"q_transform": "shuffle"})
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
    assert all(ns["CANDIDATE_IDS"][int(i)] != image_id for i, image_id in zip(indices, batch["image_id"]))
    torch.testing.assert_close(inputs[3], banks["c"]["embeddings"][indices])
    torch.testing.assert_close(inputs[5], ns["candidate_image_bank"][indices])
    loss = output["logits"].sum() - details["log_prob"].mean()
    loss.backward()
    q_text, q_valid = ns["batch_caption_inputs"](banks, batch["image_id"])
    torch.testing.assert_close(output["query_gate"].gt(0), q_valid)
    assert output["candidate_gate"].gt(0).all()
    assert network.text_projection[0].weight.grad.abs().sum() > 0
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(network.selector.semantic_mix_logit.grad)
    ns["CANDIDATE_IDS"] = ["test/0"]
    with pytest.raises(ValueError, match="training images only"):
        ns["make_ablation_banks"](mode, 42)


def test_multimodal_selector_retains_05_behavior_and_learnable_mix(toy_runtime):
    ns = runtime(toy_runtime)
    network, banks, images, text, visual_scores = controlled_selector(ns)
    image_only = select(network, banks, images, text, semantic=False)
    mixed = select(network, banks, images, text)
    assert int(image_only["indices"][0]) == 0 and int(mixed["indices"][0]) == 1
    invalid = select(network, banks, images, text, valid=torch.zeros(1, dtype=torch.bool))
    torch.testing.assert_close(invalid["log_prob"], image_only["log_prob"])
    (-mixed["log_prob"].mean() - .01 * mixed["entropy"]).backward()
    assert network.selector.semantic_mix_logit.grad.abs() > 0 and visual_scores.grad.abs().sum() > 0


@pytest.mark.parametrize("source_kind", ["fresh", "partial_cached", "complete_attached"])
def test_caption_generation_resume_reuse_does_not_apply_rejection_flags(toy_runtime, tmp_path, source_kind):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows["raw_caption"] = "The image is a close-up of a banded surface with tiny dark marks."
    rows["caption_valid"] = False
    rows["leakage_detected"] = True
    rows["semantic_leakage_detected"] = True
    rows["semantic_leakage_score"] = .99
    ns["class_names"] = ["banded"]
    cached = ns["OUT_PUT_DIR"] / "shared" / "captions.csv"
    cached.parent.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "attached.csv"
    if source_kind == "complete_attached":
        rows.to_csv(source, index=False)
    elif source_kind == "partial_cached":
        rows.iloc[:2].to_csv(cached, index=False)
    calls = []

    def generate(frames, output, clip_path):
        assert source_kind != "complete_attached"
        existing = pd.read_csv(output) if output.is_file() else rows.iloc[:0]
        calls.append(len(existing))
        remaining = rows.loc[~rows.image_id.isin(existing.image_id)]
        pd.concat([existing, remaining], ignore_index=True).to_csv(output, index=False)

    def encode(views, *args):
        assert set(views) == {"unscreened"}
        return {"embedding_dim": 512, "fingerprint": "unscreened", "views": {
            name: {**view, "embeddings": torch.ones(len(view["texts"]), 512),
                   "valid": torch.as_tensor(view["valid"], dtype=torch.bool)} for name, view in views.items()}}

    ns.update(PROJECT_DIR=tmp_path, DTD_SPLIT=1,
              CAPTIONS_INPUT_PATH=str(source) if source.is_file() else "",
              resolve_clip_path=lambda *args: tmp_path / "clip",
              run_local_dtd_caption_generation=generate, encode_view_cache=encode)
    exec(code_cells()[9], ns)
    assert calls == ([] if source_kind == "complete_attached" else [2 if source_kind == "partial_cached" else 0])
    assert not ns["caption_audit"].screened_valid.any() and ns["caption_audit"].unscreened_valid.all()
    assert ns["TEXT_CACHE"]["views"]["unscreened"]["valid"].all()
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    assert banks["q"]["valid"].sum() == 18 and banks["c"]["valid"].sum() == 3
    model_inputs = pd.read_csv(ns["SHARED_DIR"] / "caption_model_inputs.csv")
    assert model_inputs.model_caption_valid.all() and not model_inputs.screened_reference_valid.any()
    assert model_inputs.model_caption.str.startswith("banded").all()
    before = ns["CAPTION_CSV"].read_bytes()
    exec(code_cells()[9], ns)
    assert ns["CAPTION_CSV"].read_bytes() == before
    assert len(calls) == int(source_kind != "complete_attached")
    if source.is_file():
        assert source.read_bytes() == before


@pytest.mark.parametrize("missing_role", ["query", "pool"])
def test_missing_caption_availability_stops_before_clip_or_training(toy_runtime, tmp_path, missing_role):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    missing = rows.split.eq("train") if missing_role == "query" else rows.image_id.isin(ns["CANDIDATE_IDS"])
    rows.loc[missing, "final_caption"] = "too short"
    source = tmp_path / "missing.csv"
    rows.to_csv(source, index=False)
    ns.update(PROJECT_DIR=tmp_path, DTD_SPLIT=1, CAPTIONS_INPUT_PATH=str(source))
    message = "No available pre-filter training captions" if missing_role == "query" else "No available pre-filter candidate-pool captions"
    with pytest.raises(ValueError, match=message):
        exec(code_cells()[9], ns)


def test_real_cache_path_encodes_flagged_terms_and_zeros_only_unavailable_text(toy_runtime, tmp_path, monkeypatch):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows["final_caption"] = "banded surface with many tiny dark marks"
    rows["caption_valid"] = False
    rows["leakage_detected"] = True
    rows.loc[1, "final_caption"] = ""
    audit, views = ns["prepare_unscreened_caption_views"](rows, ["banded"], .3, "dtd_lexical_v2")
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "config.json").write_text(json.dumps({"model_type": "clip"}))
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
            calls.extend(texts)
            return Tokens(input_ids=torch.ones(len(texts), 1, dtype=torch.long))

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=77, projection_dim=4)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids):
            return SimpleNamespace(text_embeds=torch.ones(len(input_ids), 4))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPTokenizer=Tokenizer, CLIPTextModelWithProjection=Encoder))
    cache_path = tmp_path / "unscreened.pt"
    result = ns["encode_view_cache"]({"unscreened": views["unscreened"]}, clip, cache_path)
    cache = result["views"]["unscreened"]
    assert calls == ["banded surface with many tiny dark marks"]
    assert cache["valid"].sum() == 17 and cache["embeddings"][1].count_nonzero() == 0
    assert cache["embeddings"][0].count_nonzero() == 4 and not audit.screened_valid.any()
    ns["encode_view_cache"]({"unscreened": views["unscreened"]}, clip, cache_path)
    assert len(calls) == 1


def test_training_resume_final_test_report_and_assignment_manifest(toy_runtime, monkeypatch, capsys):
    ns = runtime(toy_runtime)
    code = code_cells()
    mode = ns["SELECTED_ABLATIONS"][0]
    exec(code[19], ns)
    assert "sem_mix=" in capsys.readouterr().out
    run_dir = ns["RUNS_DIR"] / "full_unscreened" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"full_unscreened"}
    manifest = json.loads((run_dir / "unscreened_text_manifest.json").read_text())
    assert not manifest["leakage_masks_applied"] and manifest["view"] == "unscreened"
    assignments = pd.read_csv(run_dir / "unscreened_text_assignments.csv")
    assert assignments.caption_source_id.eq(assignments.image_id).all()
    assert assignments.loc[assignments.role.eq("c"), "caption_source_id"].isin(ns["CANDIDATE_IDS"]).all()
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
    ns.update(plt=plt, audit_summary=pd.DataFrame({"valid_rate": [1.]}, index=["train"]),
              role_coverage=pd.DataFrame({"valid_rate": [1.]},
                  index=pd.MultiIndex.from_tuples([("candidate_pool", "train")], names=["role", "split"])),
              caption_comparison=pd.DataFrame({"unscreened_count": [18], "screened_reference_count": [0]}))
    exec(code[23], ns)
    results = pd.read_csv(ns["OUT_PUT_DIR"] / "full_unscreened_results.csv")
    assert set(results.split) == {"val", "test"}
    assert results.mean_query_gate.gt(0).all() and results.mean_candidate_gate.gt(0).all()
    assert results.selector_semantic_mix.between(0, 1).all()
    report = (ns["OUT_PUT_DIR"] / "full_unscreened_report.md").read_text()
    assert "Leakage filters bypassed" in report and "higher scores do not alone prove leakage" in report
    assert "dtd_lexical_v2" in report
    plt.close("all")


def test_training_math_matches_original_unscreened_with_identical_fixed_inputs(toy_runtime):
    original_dir = original.train_one_configuration(spec("full_unscreened", view="unscreened"), 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "unscreened_runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
