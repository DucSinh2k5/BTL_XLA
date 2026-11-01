
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

import build_mtacs_dtd_both_fusion_notebook as builder
import build_mtacs_dtd_candidate_only_notebook as candidate
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
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "both_banks", "exec"), ns)
    exec(code[14], ns)
    ns.update({"CLIP_EMBED_DIM": 4, "PROTOCOL_FINGERPRINT": "both-fusion-test"})
    return ns


def test_single_configuration_defaults_and_no_changes_to_prior_notebooks(monkeypatch):
    before = {module.TARGET: module.TARGET.read_bytes() for module in (visual, query, candidate)}
    notebook = builder.build_notebook()
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(notebook["cells"]) == 24
    for cell in notebook["cells"]:
        assert cell["id"].startswith("dtd-both-fusion-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "both_fusion" and metadata["caption_generation"]
    assert metadata["default_epochs"] == 50 and metadata["default_caption_input_policy"] == "dtd_lexical_v2"
    all_code = "\n".join(visual.source_of(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
    ast.parse(all_code)
    assert 'DTD_BOTH_FUSION_EPOCHS", "50"' in all_code
    assert "EARLY_STOPPING = False" in all_code
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[1]
    assert "ensure_dtd_caption_csv" in all_code and "--no-index" in all_code
    assert "validation_interventions" not in all_code
    assert "_ablation_04_both_fusion_vitl16" in code_cells()[4]
    monkeypatch.setenv("DTD_ABLATIONS", "visual_only,full_both")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert len(ns["SELECTED_ABLATIONS"]) == 1
    mode = ns["SELECTED_ABLATIONS"][0]
    assert mode["name"] == "both_fusion" and mode["query_fusion"] and mode["candidate_fusion"]
    assert not mode["semantic_selector"]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2" and ns["TRAIN_SEEDS"] == [42]
    assert '"caption_input_policy": CAPTION_INPUT_POLICY' in code_cells()[17]


def test_visual_architecture_losses_and_transforms_match_other_ablations():
    code = code_cells()
    second = query.build_notebook()
    for index in (11, 12):
        assert code[index].strip() == visual.source_of(second["cells"][index]).strip()
    names = ["ablation_supcon", "train_ablation_epoch", "optimizer_for", "checkpoint_cpu"]
    assert visual.definitions(code[14], names) == visual.definitions(visual.source_of(second["cells"][14]), names)
    settings = visual.source_of(second["cells"][4])
    for name in ("train_transform", "eval_transform"):
        node = next(node for node in ast.parse(settings).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert ast.get_source_segment(settings, node) in code[4]
    assert "CANDIDATE_POOL_FRACTION = 0.20" in code[4]


def test_both_banks_align_by_id_and_candidates_cannot_include_heldout_images(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    view = ns["TEXT_CACHE"]["views"]["screened"]
    order = torch.arange(len(view["image_ids"]) - 1, -1, -1)
    ns["TEXT_CACHE"] = {"views": {"screened": {
        "image_ids": [view["image_ids"][int(index)] for index in order],
        "texts": [view["texts"][int(index)] for index in order],
        "valid": view["valid"][order], "embeddings": view["embeddings"][order]}}}
    banks = ns["make_ablation_banks"](mode, 42)
    torch.testing.assert_close(banks["c"]["embeddings"], view["embeddings"][:3])
    assert banks["c"]["source_ids"] == ns["CANDIDATE_IDS"]
    assert set(banks["c"]["source_ids"]) <= set(ns["SPLIT_FRAMES"]["train"].image_id)
    q_text, q_valid = ns["batch_caption_inputs"](banks, ["val/3", "train/1"])
    torch.testing.assert_close(q_text, view["embeddings"][[9, 1]])
    assert q_valid.all()
    with pytest.raises(ValueError, match="both_fusion only"):
        ns["make_ablation_banks"](mode, 42, {"semantic_selector": True})
    ns["CANDIDATE_IDS"] = ["val/0"]
    with pytest.raises(ValueError, match="training images only"):
        ns["make_ablation_banks"](mode, 42)
    ns["CANDIDATE_IDS"] = ["train/0"]
    ns["TEXT_CACHE"]["views"]["screened"]["image_ids"][0] = "duplicate_id"
    with pytest.raises(ValueError, match="missing query"):
        ns["make_ablation_banks"](mode, 42)


def test_each_text_branch_affects_logits_receives_gradients_and_masks_independently(toy_runtime):
    ns = runtime(toy_runtime)
    mode = ns["SELECTED_ABLATIONS"][0]
    banks = ns["make_ablation_banks"](mode, 42)
    network = ns["create_ablation_model"](42).eval()
    images = torch.randn(2, 3, 2, 2)
    q_text, q_valid = ns["batch_caption_inputs"](banks, ["train/3", "train/4"])
    c_text, c_valid = banks["c"]["embeddings"][:2], banks["c"]["valid"][:2]
    output = network.forward_ablated(images, images, q_text, q_valid, c_text, c_valid, mode)
    changed_q = network.forward_ablated(images, images, -q_text, q_valid, c_text, c_valid, mode)
    changed_c = network.forward_ablated(images, images, q_text, q_valid, -c_text, c_valid, mode)
    assert not torch.allclose(output["logits"], changed_q["logits"])
    assert not torch.allclose(output["logits"], changed_c["logits"])
    assert output["query_gate"].gt(0).all() and output["candidate_gate"].gt(0).all()
    masked = network.forward_ablated(images, images, q_text, torch.tensor([False, True]),
                                     c_text, torch.tensor([True, False]), mode)
    assert masked["query_gate"][0] == 0 and masked["candidate_gate"][0] > 0
    assert masked["candidate_gate"][1] == 0 and masked["query_gate"][1] > 0
    output["logits"].sum().backward()
    assert network.text_projection[0].weight.grad.abs().sum() > 0
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert network.selector.semantic_mix_logit.grad is None


def test_visual_selection_is_invariant_to_both_text_roles_in_train_and_eval(toy_runtime):
    ns = runtime(toy_runtime)
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    network = ns["create_ablation_model"](42).eval()
    bank = ns["refresh_ablation_bank"](network)
    batch = next(iter(ns["make_ablation_loaders"](42)["train"]))
    changed_banks = {**banks,
        "q": {**banks["q"], "embeddings": -banks["q"]["embeddings"] * 99, "valid": ~banks["q"]["valid"]},
        "c": {**banks["c"], "embeddings": -banks["c"]["embeddings"] * 77, "valid": ~banks["c"]["valid"]}}
    for training in (False, True):
        torch.manual_seed(91)
        _, first, selected, inputs = ns["run_forward"](network, batch, banks, bank, training=training)
        torch.manual_seed(91)
        _, second, other, _ = ns["run_forward"](network, batch, changed_banks, bank, training=training)
        torch.testing.assert_close(selected, other)
        torch.testing.assert_close(first["log_prob"], second["log_prob"])
        excluded = torch.tensor([ns["CANDIDATE_POSITION"].get(image_id, -1) for image_id in batch["image_id"]])
        assert selected.ne(excluded).all()
        torch.testing.assert_close(inputs[3], banks["c"]["embeddings"][selected])
        torch.testing.assert_close(inputs[4], banks["c"]["valid"][selected])
        torch.testing.assert_close(inputs[5], ns["candidate_image_bank"][selected])


@pytest.mark.parametrize("csv_source", ["fresh", "legacy_attached"])
def test_caption_creation_or_reuse_caches_both_roles_with_new_policy(toy_runtime, tmp_path, monkeypatch, csv_source):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    rows["raw_caption"] = "The image is a close-up of a pale rough material with tiny dark marks."
    rows["final_caption"] = "The image is a close up of a"
    rows["caption_valid"] = False
    rows["leakage_detected"] = True
    rows["semantic_leakage_detected"] = True
    rows["semantic_leakage_score"] = .913577
    rows.loc[1, ["semantic_leakage_detected", "leakage_detected"]] = False
    rows.loc[1, "semantic_leakage_score"] = .1
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "config.json").write_text(json.dumps({"model_type": "clip", "projection_dim": 512}))
    (clip / "model.safetensors").write_bytes(b"fake weights")
    calls = {"generation": 0, "models": 0, "texts": []}

    def generate(frames, output, clip_path):
        assert csv_source == "fresh"
        calls["generation"] += 1
        rows.to_csv(output, index=False)

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
    source = tmp_path / "attached_captions.csv"
    if csv_source == "legacy_attached":
        rows.to_csv(source, index=False)
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CLIP_MODEL_LOCAL_PATH": str(clip),
               "CAPTIONS_INPUT_PATH": str(source) if source.is_file() else "",
               "run_local_dtd_caption_generation": generate})
    exec(code_cells()[9], ns)
    cache = ns["TEXT_CACHE"]["views"]["screened"]
    assert cache["embeddings"].shape == (18, 512)
    assert not cache["valid"][1] and cache["embeddings"][1].eq(0).all()
    assert all("close up" not in text.lower() for text in cache["texts"])
    assert calls["texts"] == ["pale rough material with tiny dark marks"]
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    assert banks["q"]["valid"].sum() == 17 and banks["c"]["valid"].sum() == 2
    assert len(banks["c"]["source_ids"]) == 3
    assert ns["role_coverage"].loc[("candidate_pool", "train"), "valid_captions"] == 2
    original_bytes = ns["CAPTION_CSV"].read_bytes()
    exec(code_cells()[9], ns)
    assert calls["generation"] == int(csv_source == "fresh") and calls["models"] == 1
    assert ns["CAPTION_CSV"].read_bytes() == original_bytes
    assert (ns["SHARED_DIR"] / "caption_model_inputs.csv").is_file()
    assert (ns["SHARED_DIR"] / "caption_role_coverage.csv").is_file()


@pytest.mark.parametrize("missing_role", ["query", "pool"])
def test_each_all_invalid_training_branch_stops_before_loading_clip(toy_runtime, tmp_path, missing_role):
    ns = runtime(toy_runtime)
    rows = audited_rows(ns["SPLIT_FRAMES"])
    mask = rows.split.eq("train") if missing_role == "query" else rows.image_id.isin(ns["CANDIDATE_IDS"])
    rows.loc[mask, "caption_valid"] = False
    path = tmp_path / "bad.csv"
    rows.to_csv(path, index=False)
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CAPTIONS_INPUT_PATH": str(path)})
    message = "No eligible training captions" if missing_role == "query" else "No eligible candidate-pool captions"
    with pytest.raises(ValueError, match=message):
        exec(code_cells()[9], ns)


def test_training_resume_final_test_and_report_are_isolated(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    code = code_cells()
    mode = ns["SELECTED_ABLATIONS"][0]
    exec(code[19], ns)
    run_dir = ns["RUNS_DIR"] / "both_fusion" / "seed_42"
    assert {path.name for path in ns["RUNS_DIR"].iterdir()} == {"both_fusion"}
    assert (run_dir / "both_text_manifest.json").is_file()
    assigned = pd.read_csv(run_dir / "both_text_assignments.csv")
    assert assigned.role.eq("q").sum() == 18 and assigned.role.eq("c").sum() == 3
    timestamp = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](mode, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == timestamp
    exec(code[21], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update({"plt": plt, "audit_summary": pd.DataFrame({"valid_rate": [1.0]}, index=["train"]),
               "role_coverage": pd.DataFrame({"valid_rate": [1.0]},
                   index=pd.MultiIndex.from_tuples([("candidate_pool", "train")], names=["role", "split"]))})
    exec(code[23], ns)
    result = pd.read_csv(ns["OUT_PUT_DIR"] / "both_fusion_results.csv")
    assert set(result.split) == {"val", "test"}
    assert result.mean_query_gate.gt(0).all() and result.mean_candidate_gate.gt(0).all()
    report = (ns["OUT_PUT_DIR"] / "both_fusion_report.md").read_text()
    assert "Query and candidate caption coverage" in report and "dtd_lexical_v2" in report
    plt.close("all")


def test_training_math_matches_original_both_fusion_configuration(toy_runtime):
    mode = spec("both_fusion", semantic_selector=False)
    original_dir = original.train_one_configuration(mode, 42)
    expected = pd.read_csv(original_dir / "history.csv")
    ns = runtime(toy_runtime)
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "both_runs"
    actual_dir = ns["train_one_configuration"](ns["SELECTED_ABLATIONS"][0], 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    columns = [name for name in expected if not name.endswith("caption_rate")]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], atol=1e-7, rtol=1e-7)
