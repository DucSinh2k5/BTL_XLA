
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import build_mtacs_dtd_ablation_notebook as builder
import mtacs_dtd_ablation_cells as ab


def captions_frame():
    return pd.DataFrame({
        "image_id": [f"item_{i}" for i in range(5)], "split": ["train"] * 5,
        "final_caption": ["Wooden surface with thin parallel stripes", "rough material with small pale marks",
                          "rough material with small dark marks", "", "a striped sheet with pale edges"],
        "caption_valid": [False, True, True, False, False],
        "leakage_detected": [True, False, False, False, True],
        "semantic_leakage_detected": [True, False, True, False, True],
        "semantic_leakage_score": [.6, .2, .6, 0, .6],
    })


def test_caption_redaction_and_screening_do_not_use_example_labels():
    frame = captions_frame()
    audit, views = ab.prepare_caption_views(frame, ["striped", "bumpy"], .3)
    assert views["screened"]["valid"].tolist() == [False, True, False, False, False]
    assert "stripes" not in views["redacted"]["texts"][0]
    assert "striped" not in views["redacted"]["texts"][4]
    np.testing.assert_array_equal(views["unscreened"]["valid"], views["redacted"]["valid"])
    assert views["term_only"]["texts"][0] == "stripes"
    assert views["screened"]["texts"][0] == frame.final_caption.iloc[0]
    other = frame.assign(class_id=[2, 1, 0, 2, 1])
    _, other_views = ab.prepare_caption_views(other, ["striped", "bumpy"], .3)
    for name in views:
        assert views[name]["texts"] == other_views[name]["texts"]
        np.testing.assert_array_equal(views[name]["valid"], other_views[name]["valid"])
    assert "normalized_caption" not in frame
    assert audit.alias_or_class_term_hit.tolist() == [True, False, False, False, True]


def test_shuffle_preserves_split_pool_membership_and_exact_availability():
    ids = ["train/a", "train/b", "train/c", "train/d", "val/a", "val/b", "val/c", "test/a"]
    valid = torch.tensor([True, True, True, True, True, True, False, True])
    split = {key: key.split("/")[0] for key in ids}
    pool = {"train/a", "train/b"}
    positions, singletons = ab.deranged_sources(ids, valid, split, pool, 13)
    for i, position in enumerate(positions):
        assert valid[i] == valid[position]
        assert split[ids[i]] == split[ids[position]]
        assert (ids[i] in pool) == (ids[position] in pool)
        if valid[i] and ids[i] not in singletons:
            assert i != position
    assert singletons == ["test/a"]
    np.testing.assert_array_equal(positions, ab.deranged_sources(ids, valid, split, pool, 13)[0])


class ToySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def score(self, query, bank):
        return F.normalize(query, dim=1) @ F.normalize(bank, dim=1).T * self.logit_scale


class ToyVisual(nn.Module):
    def forward_selected_context(self, query, context, return_features=False):
        features = F.normalize(self.query_norm(self._pool_backbone_features(query)
                                               + self._pool_backbone_features(context)), dim=1)
        return self.mtacs_classifier(features), features, None


class ToyBase(ToyVisual):
    def __init__(self, *args, clip_dim=4, **kwargs):
        super().__init__()
        self.backbone = nn.Linear(12, 8)
        self.query_norm = nn.LayerNorm(8)
        self.selector = ToySelector()
        self.text_projection = nn.Sequential(nn.Linear(clip_dim, 8), nn.LayerNorm(8), nn.Dropout(.1))
        self.text_gate = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1))
        self.mtacs_norm = nn.LayerNorm(8)
        self.mtacs_classifier = nn.Sequential(nn.Dropout(.1), nn.Linear(8, 3))

    def _pool_backbone_features(self, images):
        return self.backbone(images.reshape(len(images), -1))

    def score_full_candidate_pool(self, images, bank):
        features = self._pool_backbone_features(images)
        return self.selector.score(features, bank), features


def spec(name="full_both", **overrides):
    return {"name": name, "view": "screened", "q_transform": "aligned", "c_transform": "aligned",
            "query_fusion": True, "candidate_fusion": True, "semantic_selector": True, **overrides}


@pytest.fixture
def toy_runtime(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    ab.seed_ablation(10)
    ids = [f"{split}/{i}" for split in ("train", "val", "test") for i in range(6)]
    embeddings = torch.randn(len(ids), 4)
    valid = torch.ones(len(ids), dtype=torch.bool)
    valid[5] = False
    embeddings[~valid] = 0
    cache = {"embedding_dim": 4, "views": {key: {
        "image_ids": ids, "texts": [f"{key} caption {i}" for i in range(len(ids))],
        "valid": valid.clone(), "embeddings": embeddings.clone() if key != "redacted" else embeddings.clone() * .5,
    } for key in ("screened", "unscreened", "redacted", "term_only")}}
    frames = {split: pd.DataFrame({"image_id": [i for i in ids if i.startswith(split + "/")],
                                  "class_id": [0, 1, 2, 0, 1, 2]}) for split in ("train", "val", "test")}
    values = {
        "DEVICE": torch.device("cpu"), "TEXT_CACHE": cache, "SPLIT_BY_ID": {i: i.split("/")[0] for i in ids},
        "CANDIDATE_IDS": ids[:3], "CANDIDATE_POSITION": {key: i for i, key in enumerate(ids[:3])},
        "candidate_image_bank": torch.randn(3, 3, 2, 2), "NUM_CLASSES": 3,
        "class_names": ["a", "b", "c"], "FEATURE_DIM": 8, "CLIP_EMBED_DIM": 4, "SEED": 42,
        "DINOV3_REPO_DIR": "unused", "DINOV3_MODEL_NAME": "toy", "EPOCHS": 2, "WARMUP_EPOCHS": 1,
        "PATIENCE": 3, "FREEZE_BACKBONE_EPOCHS": 0, "LABEL_SMOOTHING": .1,
        "SUPCON_WEIGHT": .05, "SUPCON_TEMPERATURE": .1, "TACS_POLICY_WEIGHT": 1.0,
        "TACS_ENTROPY_WEIGHT": .01, "GRAD_CLIP_NORM": 1.0, "BACKBONE_LR": .001,
        "BACKBONE_WEIGHT_DECAY": .01, "TACS_SELECTOR_LR": .01, "TACS_SELECTOR_WEIGHT_DECAY": .01,
        "HEAD_LR": .01, "HEAD_WEIGHT_DECAY": .01, "RESUME_ABLATIONS": True,
        "KEEP_LAST_CHECKPOINT": False, "RUNS_DIR": tmp_path / "runs", "OUT_PUT_DIR": tmp_path,
        "PROTOCOL_FINGERPRINT": "toy-protocol", "TRAIN_SEEDS": [42], "BOOTSTRAP_REPEATS": 5,
        "train_df": frames["train"], "val_df": frames["val"], "test_df": frames["test"],
        "SPLIT_FRAMES": frames, "caption_audit": pd.DataFrame({
            "image_id": ids, "normalized_caption": [f"plain material example {i}" for i in range(len(ids))],
            "screened_valid": valid.numpy()}),
        "resolve_dinov3_weights": lambda: "unused", "AblationModel": ab.make_ablation_model_class(ToyBase, ToyVisual),
    }
    for name, value in values.items():
        monkeypatch.setattr(ab, name, value, raising=False)
    images = {split: torch.randn(6, 3, 2, 2) for split in frames}

    def loaders(seed):
        return {split: DataLoader([
            {"query_image": images[split][i], "label": i % 3, "image_id": f"{split}/{i}"} for i in range(6)
        ], batch_size=3, shuffle=split == "train", generator=torch.Generator().manual_seed(seed)) for split in frames}

    def refresh(network, bank):
        with torch.no_grad():
            return network._pool_backbone_features(bank).detach().clone()

    monkeypatch.setattr(ab, "make_ablation_loaders", loaders)
    monkeypatch.setattr(ab, "refresh_candidate_feature_bank", refresh, raising=False)
    return values


def test_banks_shuffle_independent_roles_preserve_masks_and_training_pool(toy_runtime):
    aligned = ab.make_ablation_banks(spec(), 42)
    shuffled = ab.make_ablation_banks(spec(), 42, {"q_transform": "shuffle", "c_transform": "shuffle"})
    for role in ("q", "c"):
        assert torch.equal(aligned[role]["valid"], shuffled[role]["valid"])
        assert not torch.equal(aligned[role]["embeddings"], shuffled[role]["embeddings"])
    assert set(shuffled["c"]["source_ids"]) == set(ab.CANDIDATE_IDS)
    assert all(ab.SPLIT_BY_ID[source] == ab.SPLIT_BY_ID[target]
               for target, source in zip(ab.TEXT_CACHE["views"]["screened"]["image_ids"], shuffled["q"]["source_ids"]))
    zero = ab.make_ablation_banks(spec(), 42, {"q_transform": "zero"})
    assert not zero["q"]["valid"].any()
    assert torch.count_nonzero(zero["q"]["embeddings"]) == 0
    assert torch.equal(aligned["c"]["embeddings"], zero["c"]["embeddings"])
    ab.TEXT_CACHE["views"]["redacted"]["valid"][0] = False
    ab.TEXT_CACHE["views"]["redacted"]["embeddings"][0] = 0
    redacted = ab.make_ablation_banks(spec(), 42, {"view": "redacted", "mask_view": "screened"})
    assert torch.equal(aligned["q"]["valid"], redacted["q"]["valid"])
    assert torch.equal(aligned["c"]["valid"], redacted["c"]["valid"])
    assert redacted["q"]["embeddings"][0].count_nonzero() == 0


def test_disabled_text_is_inert_and_candidate_text_is_actually_fused(toy_runtime):
    network = ab.create_ablation_model(42).eval()
    images = torch.randn(2, 3, 2, 2)
    a, b = torch.randn(2, 4), torch.randn(2, 4)
    valid = torch.ones(2, dtype=torch.bool)
    visual = spec(query_fusion=False, candidate_fusion=False)
    first = network.forward_ablated(images, images, a, valid, b, valid, visual)
    second = network.forward_ablated(images, images, a * 100, valid, b * -100, valid, visual)
    torch.testing.assert_close(first["logits"], second["logits"])
    assert first["query_gate"].sum() == first["candidate_gate"].sum() == 0
    candidate = spec(query_fusion=False, candidate_fusion=True)
    first = network.forward_ablated(images, images, a, valid, b, valid, candidate)
    second = network.forward_ablated(images, images, a, valid, b.flip(0), valid, candidate)
    assert not torch.allclose(first["logits"], second["logits"])
    first["logits"].sum().backward()
    assert network.context_text_projection[0].weight.grad.abs().sum() > 0
    assert network.text_projection[0].weight.grad.abs().sum() == 0


def test_selector_excludes_self_and_semantic_policy_has_finite_gradients(toy_runtime):
    network = ab.create_ablation_model(42)
    images = torch.randn(3, 3, 2, 2)
    bank = ab.refresh_ablation_bank(network)
    banks = ab.make_ablation_banks(spec(), 42)
    text, valid = ab.batch_caption_inputs(banks, ["train/0", "train/1", "train/2"])
    excluded = torch.arange(3)
    result = network.select_context(images, bank, text, valid, banks, excluded, semantic=True)
    assert (result["indices"] != excluded).all()
    assert torch.isfinite(result["log_prob"]).all()
    (-result["log_prob"].mean()).backward()
    gradient = network.selector.semantic_mix_logit.grad
    assert torch.isfinite(gradient) and gradient.abs() > 0
    assert network.backbone.weight.grad.abs().sum() > 0
    visual = network.select_context(images, bank, text, valid, banks, excluded, semantic=False)
    missing = network.select_context(images, bank, text, torch.zeros_like(valid), banks, excluded, semantic=True)
    torch.testing.assert_close(visual["log_prob"], missing["log_prob"])


def test_paired_statistics_align_ids_and_reject_different_ground_truth(monkeypatch):
    monkeypatch.setattr(ab, "NUM_CLASSES", 2, raising=False)
    frame = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "ground_truth": [0, 0, 1, 1],
                          "prediction": [0, 1, 1, 0]})
    metrics = ab.paired_f1_interval(frame, frame.iloc[::-1], repeats=10)
    assert metrics["delta_macro_f1"] == metrics["ci95_low"] == metrics["ci95_high"] == 0
    with pytest.raises(ValueError):
        ab.paired_f1_interval(frame, frame.assign(ground_truth=[1, 0, 1, 1]))


def test_training_interventions_final_test_and_resume_cache(toy_runtime, monkeypatch):
    choices = [spec("visual_only", query_fusion=False, candidate_fusion=False, semantic_selector=False), spec()]
    monkeypatch.setattr(ab, "SELECTED_ABLATIONS", choices, raising=False)
    ab.train_one_configuration(choices[0], 42)
    with pytest.raises(RuntimeError, match="Complete every"):
        ab.evaluate_locked_test()
    assert not (ab.OUT_PUT_DIR / "locked_test_plan.json").exists()
    ab.train_one_configuration(choices[1], 42)
    before = (ab.RUNS_DIR / "full_both/seed_42/best.pt").stat().st_mtime_ns
    ab.train_one_configuration(choices[1], 42)
    assert before == (ab.RUNS_DIR / "full_both/seed_42/best.pt").stat().st_mtime_ns
    assert not (ab.RUNS_DIR / "full_both/seed_42/last.pt").exists()
    intervention = ab.validation_interventions(choices[1], 42)
    fixed = intervention[intervention.intervention.str.contains("fixed_context")]
    assert fixed.changed_context_rate.eq(0).all()
    test = ab.evaluate_locked_test()
    assert len(test) == 2
    prediction_path = ab.RUNS_DIR / "full_both/seed_42/test_predictions.csv"
    timestamp = prediction_path.stat().st_mtime_ns
    ab.evaluate_locked_test()
    assert prediction_path.stat().st_mtime_ns == timestamp
    assert len(ab.summarize_ablation("val")) == 2
    assert len(ab.caption_probes()) == 5
    checkpoint = torch.load(ab.RUNS_DIR / "full_both/seed_42/best.pt", weights_only=False)
    assert checkpoint["identity"]["spec"]["candidate_fusion"] is True
    assert (ab.RUNS_DIR / "full_both/seed_42/caption_source_assignments.csv").is_file()


def test_interrupted_training_resumes_same_rng_and_metrics(toy_runtime, monkeypatch, tmp_path):
    configuration = spec("resume_check")
    monkeypatch.setattr(ab, "RUNS_DIR", tmp_path / "uninterrupted")
    uninterrupted_dir = ab.train_one_configuration(configuration, 42)
    expected = pd.read_csv(uninterrupted_dir / "history.csv")
    original_train = ab.train_ablation_epoch
    calls = []

    def interrupt_second_epoch(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        return original_train(*args, **kwargs)

    monkeypatch.setattr(ab, "RUNS_DIR", tmp_path / "interrupted")
    monkeypatch.setattr(ab, "train_ablation_epoch", interrupt_second_epoch)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        ab.train_one_configuration(configuration, 42)
    monkeypatch.setattr(ab, "train_ablation_epoch", original_train)
    actual_dir = ab.train_one_configuration(configuration, 42)
    actual = pd.read_csv(actual_dir / "history.csv")
    pd.testing.assert_frame_equal(expected, actual, atol=1e-7, rtol=1e-7)


def test_generated_notebook_is_standalone_and_has_no_stale_outputs():
    notebook = builder.build_notebook()
    assert notebook["nbformat"] == 4
    code = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = builder.source_of(cell)
            ast.parse(source, filename=f"cell_{index}")
            assert cell["execution_count"] is None and cell["outputs"] == []
            code.append(source)
    combined = "\n".join(code)
    compile(combined, "standalone_notebook", "exec")
    assert "import mtacs_dtd_ablation_cells" not in combined
    assert "def make_ablation_model_class" in combined
    assert "candidate_self_positions" not in combined
    assert "candidate_labels_tensor[" not in combined
    namespace = {"os": __import__("os")}
    exec(builder.CONFIG, namespace)
    choices = namespace["SELECTED_ABLATIONS"]
    assert len(choices) == 8
    assert any(item["candidate_fusion"] and not item["query_fusion"] for item in choices)
    assert namespace["GENERATE_CAPTIONS_IF_MISSING"] is True


def test_explicit_caption_input_cannot_silently_reuse_a_different_cache(tmp_path):
    import shutil
    source = tmp_path / "input.csv"
    source.write_text("image_id,final_caption\na,first caption\n", encoding="utf-8")
    namespace = {"OUT_PUT_DIR": tmp_path / "ablation", "PROJECT_DIR": tmp_path, "DTD_SPLIT": 1,
                 "CAPTIONS_INPUT_PATH": str(source), "GENERATE_CAPTIONS_IF_MISSING": False,
                 "Path": Path, "shutil": shutil}
    exec(builder.CAPTION_SETUP, namespace)
    assert namespace["CAPTION_CSV"].read_bytes() == source.read_bytes()
    source.write_text("image_id,final_caption\na,changed caption\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        exec(builder.CAPTION_SETUP, namespace)


def test_dual_text_fusion_integrates_with_actual_source_model_inheritance(monkeypatch):
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(8, 8)

        def forward(self, tokens, rope=None):
            return self.projection(tokens + tokens.mean(1, keepdim=True))

    class TinyDino(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size, self.num_prefix_tokens = 1, 1
            self.embedding = nn.Linear(3, 8)
            self.blocks = nn.ModuleList([TinyBlock()])
            self.norm = nn.LayerNorm(8)

        def prepare_tokens_with_masks(self, images, masks=None):
            patches = self.embedding(images.flatten(2).transpose(1, 2))
            return torch.cat([patches.mean(1, keepdim=True), patches], 1)

        def forward_features(self, images):
            tokens = self.prepare_tokens_with_masks(images)
            for block in self.blocks:
                tokens = block(tokens)
            return {"x_norm_clstoken": self.norm(tokens[:, 0])}

    source = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    model_source = builder.source_of(source["cells"][14])
    namespace = {"nn": nn, "torch": torch, "F": F, "FEATURE_DIM": 8, "NUM_CLASSES": 3,
                 "TOKEN_HEAD_DROPOUT": .1, "QUERY_CONTEXT_GATE_DROPOUT": .1, "QUERY_CONTEXT_GATE_BIAS": -.7,
                 "TACS_SELECTOR_TEMPERATURE": 1., "RETRIEVAL_SCORE_WEIGHT": 0.,
                 "load_dinov3_model": lambda *args: TinyDino()}
    names = {"DinoV3FrozenSimilarityContextClassifier", "DinoV3GatedTopKContextClassifier",
             "TACSSelector", "DinoV3TACSClassifier", "MTACSModel"}
    for node in ast.parse(model_source).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            exec(ast.get_source_segment(model_source, node), namespace)
    model_class = ab.make_ablation_model_class(namespace["MTACSModel"], namespace["DinoV3TACSClassifier"])
    model = model_class("", "", "", 3, 8, clip_dim=4).eval()
    q_image = torch.randn(2, 3, 2, 2, requires_grad=True)
    c_image = torch.randn(2, 3, 2, 2, requires_grad=True)
    text, valid = torch.randn(2, 4), torch.ones(2, dtype=torch.bool)
    output = model.forward_ablated(q_image, c_image, text, valid, text.flip(0), valid, spec())
    F.cross_entropy(output["logits"], torch.tensor([0, 1])).backward()
    assert q_image.grad.abs().sum() > 0 and c_image.grad.abs().sum() > 0
    assert model.text_projection[0].weight.grad.abs().sum() > 0
    assert model.context_text_projection[0].weight.grad.abs().sum() > 0
