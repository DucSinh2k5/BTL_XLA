"""Real tiny DINOv3 CPU tests; caption service injection is explicit test-only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
for directory in ("src/models", "src/training", "src/captioning", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / directory))
import tacs_visual_web as visual
import train_sun397_tacs_web as common
import mtacs_multimodal_web as runtime
import sun397_caption_web as captions
import train_sun397_mtacs_web as workflow
import build_sun397_mtacs_web_notebook as builder


@pytest.fixture(scope="module")
def dino():
    repo = Path(os.getenv("DINOV3_TEST_REPO", str(ROOT / "tmp/dinov3_web_export_test")))
    if not (repo / "dinov3/models/vision_transformer.py").is_file():
        pytest.skip("Set DINOV3_TEST_REPO to official DINOv3 for CPU integration tests")
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    return repo, DinoVisionTransformer


def tiny_config():
    return {**workflow.default_config(), "feature_dim": 32, "num_classes": 3, "image_size": 32,
            "epochs": 1, "batch_size": 2, "effective_batch_size": 4, "eval_batch_size": 2,
            "feature_batch_size": 2, "workers": 0}


def tiny_model(dino, config=None):
    cls = dino[1]
    backbone = cls(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4,
                   n_storage_tokens=2, ffn_ratio=2, drop_path_rate=0.0)
    backbone.init_weights()
    return runtime.MultimodalTACS(backbone, config or tiny_config())


def fixture_caption(_request):
    text = "dense neutral shapes covering textured flat surface"
    vector = torch.arange(1, 513, dtype=torch.float32).reshape(1, 512)
    return {"rows": [{"raw_caption": text, "final_caption": text, "caption_valid": True}],
            "embeddings": torch.nn.functional.normalize(vector, dim=1), "valid": torch.tensor([True])}


def image_rows(root, count, split, offset=0):
    names = ["a/abbey", "b/bedroom", "c/courtyard"]
    rows = []
    for i in range(count):
        image_id = f"{names[i % 3]}/sun_{split}_{i + offset}.png"
        path = root / image_id
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.random.default_rng(i + offset).integers(0, 256, (35, 39, 3), dtype=np.uint8)).save(path)
        rows.append({"image_id": image_id, "image_path": str(path), "class_id": i % 3,
                     "class_name": names[i % 3], "split": split, "image_sha256": visual.file_sha256(path)})
    return rows


def fixture_assets(root):
    assets = {}
    for name in ("florence", "clip"):
        folder = root / name
        folder.mkdir(parents=True)
        # These intentionally are not loadable real model assets. Unit tests
        # inject the service; production always validates/loads actual weights.
        visual.write_json(folder / "config.json", {"test_fixture_not_model": True})
        (folder / "fixture.txt").write_text("unit test only", encoding="utf-8")
        assets[name], assets[name + "_sha256"] = str(folder), captions.tree_identity(folder)
    return assets


def fixture_texts(shared, rows, model_assets):
    shared.mkdir(parents=True)
    records = []
    torch.manual_seed(2)
    embeddings = torch.nn.functional.normalize(torch.randn(len(rows), 512), dim=1)
    masks = torch.ones(len(rows), dtype=torch.bool)
    masks[1] = False
    embeddings[~masks] = 0
    for i, row in enumerate(rows):
        records.append({"image_id": row["image_id"], "split": row["split"], "image_sha256": row["image_sha256"],
            "raw_caption": "dense neutral shapes covering textured flat surface", "final_caption": "dense neutral shapes covering textured flat surface",
            "caption_valid": bool(masks[i]), "leakage_detected": False, "lexical_leakage_detected": False,
            "semantic_similarity_score": 0.45, "semantic_nearest_class": "abbey", "semantic_similarity_flag": True,
            "caption_policy": captions.POLICY, "captioner_assets_sha256": model_assets["florence_sha256"],
            "clip_assets_sha256": model_assets["clip_sha256"], "vlm_model": "test_fixture", "prompt_version": "test_fixture"})
    common.csv_write(shared / "captions.csv", records, captions.COLUMNS)
    cache = {"embeddings": embeddings, "valid": masks, "image_ids": [r["image_id"] for r in rows],
             "csv_sha256": visual.file_sha256(shared / "captions.csv"), "policy": captions.POLICY,
             "clip_sha256": model_assets["clip_sha256"], "captioner_sha256": model_assets["florence_sha256"]}
    common.atomic_torch_save(shared / "text_embeddings.pt", cache)
    visual.write_json(shared / "caption_protocol.json", {"test_fixture": True})
    visual.write_json(shared / "caption_settings.json", {"classes": ["a/abbey", "b/bedroom", "c/courtyard"],
        "settings": {"policy": captions.POLICY, "task": captions.PROMPT, "precision": "float32", "max_new_tokens": 48,
                     "semantic_audit_threshold": 0.30, "batch_size": 2, "aliases": []}, "asset_identity": model_assets})
    return cache


def test_notebook_is_standalone_and_matches_generated_file():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert notebook["metadata"]["tacs"]["backbone"] == "dinov3_vitl16"
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for filename in builder.SOURCES:
        assert filename in code
    assert "DINOV3_WEIGHTS" in code and "OFFLINE_ASSETS_ROOT" in code
    assert "workflow.prepare_captions" in code and "workflow.export_bundle" in code
    assert "--no-index" in code and "cold_subprocess_reload" in code
    assert "dinov3_vits16" not in code and "import transformers" not in "".join(notebook["cells"][2]["source"])
    assert all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in notebook["cells"])


def test_production_config_and_policy_are_not_dtd_or_top3():
    config = workflow.default_config()
    workflow.validate_config(config)
    assert config["caption_used"] and config["context_top_k"] == 1 and config["epochs"] == 100
    for change in ({"backbone": "dinov3_vits16"}, {"caption_policy": "dtd_lexical_v2"}, {"caption_used": False}, {"context_top_k": 3}):
        with pytest.raises(ValueError):
            workflow.validate_config({**config, **change})


def test_full_production_l16_architecture_on_meta(dino):
    with torch.device("meta"):
        model = runtime.build_model(dino[0], workflow.default_config())
    assert model.backbone.embed_dim == 1024 and len(model.backbone.blocks) == 24
    assert model.text_projection[0].in_features == 512 and model.text_projection[0].out_features == 1024
    assert model.classifier[-1].out_features == 397
    assert next(model.parameters()).device.type == "meta"


def test_prefix_cleanup_length_and_sun_class_terms():
    terms = captions.forbidden_terms(["a/abbey", "a/airplane_cabin", "b/bedroom", "c/church/indoor"])
    final, valid, _ = captions.clean_caption("The image is a close up of a dark wooden surface with narrow parallel ridges", terms)
    assert final == "dark wooden surface with narrow parallel ridges" and valid
    assert not captions.clean_caption("The image shows a bedroom with wooden walls and lights", terms)[1]
    assert not captions.clean_caption("church with tall pale walls and windows", terms)[1]
    assert not captions.clean_caption("small pale shapes", terms)[1]
    assert len(captions.clean_caption("one two three four five six seven eight nine ten", terms)[0].split()) == 8
    assert "indoor" not in terms


def test_masked_z_zero_variance_and_exclusion_gradients():
    values = torch.ones(2, 4, requires_grad=True)
    valid = torch.tensor([[True, False, False, False], [False, False, False, False]])
    result = runtime.masked_z(values, valid)
    assert torch.count_nonzero(result) == 0
    result.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_full_both_text_fusion_and_selector_gradients(dino):
    torch.manual_seed(4)
    model = tiny_model(dino).eval()
    images = torch.randn(2, 3, 32, 32)
    qtext, ctext = torch.randn(2, 512), torch.randn(2, 512)
    valid = torch.ones(2, dtype=torch.bool)
    logits, details = model(images, images.flip(0), qtext, valid, ctext, valid, return_attention=True)
    plain = model(images, images.flip(0), qtext, valid, ctext, valid)
    torch.testing.assert_close(logits, plain, rtol=0, atol=0)
    assert (details["q_gate"] > 0).all() and (details["c_gate"] > 0).all()
    assert details["attention"]["query_classifier"].shape == (2, 2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    for parameter in (model.query_gate[-1].weight, model.candidate_gate[-1].weight, model.text_projection[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    bank = torch.randn(4, 32)
    scores, _ = model.score(images, bank, [0, 1], qtext, valid, torch.randn(4, 512), torch.ones(4, dtype=torch.bool))
    assert torch.isneginf(scores[0, 0]) and torch.isneginf(scores[1, 1])
    torch.nn.functional.cross_entropy(scores, torch.tensor([2, 3])).backward()
    assert model.semantic_mix_logit.grad is not None and torch.isfinite(model.semantic_mix_logit.grad)


def test_invalid_text_is_zero_effect_and_visual_selector_fallback(dino):
    model = tiny_model(dino).eval()
    images, bank = torch.randn(2, 3, 32, 32), torch.randn(4, 32)
    zeros, invalid = torch.zeros(2, 512), torch.zeros(2, dtype=torch.bool)
    bad = torch.full_like(zeros, float("nan"))
    with torch.no_grad():
        expected = model(images, images.flip(0), zeros, invalid, zeros, invalid)
        actual, details = model(images, images.flip(0), bad, invalid, bad, invalid, return_details=True)
        torch.testing.assert_close(actual, expected)
        assert details["q_gate"].sum() == 0 and details["c_gate"].sum() == 0
        scores, query = model.score(images, bank, None, bad, invalid, torch.randn(4, 512), torch.ones(4, dtype=torch.bool))
        assert scores.argmax(1).tolist() == (query @ bank.T).argmax(1).tolist()
    with pytest.raises(ValueError):
        model.score(images, bank)


def test_csv_duplicate_schema_and_text_cache_binding(tmp_path):
    rows = image_rows(tmp_path / "images", 3, "train")
    assets = fixture_assets(tmp_path / "assets")
    cache = fixture_texts(tmp_path / "shared", rows, assets)
    table = workflow.TextTable(cache, tmp_path / "shared/captions.csv", rows, tiny_config())
    values, masks = table.gather([rows[2]["image_id"], rows[1]["image_id"]], "cpu")
    torch.testing.assert_close(values[0], cache["embeddings"][2])
    assert masks.tolist() == [True, False]
    bad = {**cache, "csv_sha256": "wrong"}
    with pytest.raises(ValueError):
        workflow.TextTable(bad, tmp_path / "shared/captions.csv", rows, tiny_config())
    (tmp_path / "old.csv").write_text("image_id,caption\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        captions.load_csv(tmp_path / "old.csv")


def test_offline_wheel_zip_is_flat_and_cannot_escape(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    with zipfile.ZipFile(root / "offline_wheels.zip", "w") as handle:
        handle.writestr("../../tokenizers-0.21.4-test.whl", b"unit-test-wheel-not-installable")
        handle.writestr("ignored.txt", b"ignored")
    destination = tmp_path / "unpacked"
    assert captions.wheel_directories(root, destination) == [str(destination.resolve())]
    assert (destination / "tokenizers-0.21.4-test.whl").is_file()
    assert not (tmp_path / "tokenizers-0.21.4-test.whl").exists()


def test_legacy_asset_validation_refuses_native_or_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError):
        captions.validate_assets(tmp_path / "missing", tmp_path / "missing_clip")


def test_caption_worker_csv_resume_and_similarity_is_audit_only(tmp_path, monkeypatch):
    class FixtureEngine:
        generated = 0
        fail_first_batch = True

        def __init__(self, payload):
            self.terms = captions.forbidden_terms(payload["classes"])

        def generate(self, images):
            if self.fail_first_batch and len(images) > 1:
                FixtureEngine.fail_first_batch = False
                raise torch.cuda.OutOfMemoryError("simulated unit test OOM")
            FixtureEngine.generated += len(images)
            return ["The image shows a dense neutral surface with small pale shapes" for _ in images]

        def encode(self, texts):
            return torch.nn.functional.normalize(torch.arange(1, 513, dtype=torch.float32).repeat(len(texts), 1), dim=1)

        def audit(self, texts):
            rows = []
            for raw in texts:
                final, valid, matched = captions.clean_caption(raw, self.terms)
                rows.append({"raw_caption": raw, "final_caption": final, "caption_valid": valid,
                    "leakage_detected": bool(matched), "lexical_leakage_detected": bool(matched),
                    "semantic_similarity_score": 0.9, "semantic_nearest_class": "abbey", "semantic_similarity_flag": True,
                    "caption_policy": captions.POLICY})
            vectors = self.encode(texts)
            masks = torch.tensor([r["caption_valid"] for r in rows])
            vectors[~masks] = 0
            return rows, vectors, masks

    monkeypatch.setattr(captions, "CaptionEngine", FixtureEngine)
    records = image_rows(tmp_path / "images", 4, "train")
    assets = fixture_assets(tmp_path / "assets")
    settings = {"policy": captions.POLICY, "task": captions.PROMPT, "max_new_tokens": 48,
                "batch_size": 2, "precision": "float32", "semantic_audit_threshold": 0.30, "aliases": []}
    payload = {"mode": "prepare", "records": records, "classes": ["a/abbey", "b/bedroom", "c/courtyard"],
               "settings": settings, "asset_identity": assets, "output": str(tmp_path / "shared/captions.csv"),
               "result": str(tmp_path / "result.pt")}
    captions.worker(payload)
    assert FixtureEngine.generated == 4
    result = torch.load(tmp_path / "result.pt", weights_only=True)
    assert result["embeddings"].shape == (4, 512) and result["valid"].all()
    rows = captions.load_csv(tmp_path / "shared/captions.csv")
    assert len(rows) == 4
    assert all(captions.truth(r["semantic_similarity_flag"]) and captions.truth(r["caption_valid"]) for r in rows.values())
    old_sha = visual.file_sha256(tmp_path / "shared/captions.csv")
    captions.worker(payload)
    assert FixtureEngine.generated == 4 and visual.file_sha256(tmp_path / "shared/captions.csv") == old_sha
    with pytest.raises(ValueError, match="protocol changed"):
        captions.worker({**payload, "settings": {**settings, "max_new_tokens": 64}})


def test_train_resume_export_and_fresh_caption_adapter(tmp_path, dino, monkeypatch):
    monkeypatch.setattr(workflow, "validate_config", lambda config: None)
    config, classes = tiny_config(), ["a/abbey", "b/bedroom", "c/courtyard"]
    train = image_rows(tmp_path / "images", 6, "train")
    val = image_rows(tmp_path / "images", 3, "val", 100)
    pool = [dict(row, position=i) for i, row in enumerate(train[:4])]
    output = tmp_path / "run"
    assets = fixture_assets(tmp_path / "assets")
    shared = output / "shared"
    cache = fixture_texts(shared, train + val, assets)
    table = workflow.TextTable(cache, shared / "captions.csv", train + val, config)
    identity_assets = {"caption_assets": assets, "caption_csv_sha256": visual.file_sha256(shared / "captions.csv"),
                       "text_cache_sha256": visual.file_sha256(shared / "text_embeddings.pt"),
                       "caption_settings_sha256": visual.file_sha256(shared / "caption_settings.json")}
    common.seed_all(42)
    model = tiny_model(dino, config)
    best, history = workflow.train_run(model, classes, train, val, pool, table, config, identity_assets, output, "cpu")
    assert len(history) == 1 and np.isfinite(history[0]["train_loss"])
    assert history[0]["val_q_gate"] > 0 and history[0]["val_c_gate"] >= 0
    _, again = workflow.train_run(model, classes, train, val, pool, table, config, identity_assets, output, "cpu")
    assert len(again) == 1
    with pytest.raises(ValueError, match="another run"):
        workflow.train_run(model, classes, train, val, pool, table, {**config, "epochs": 2}, identity_assets, output, "cpu")
    bank = common.refresh_bank(model, pool, config, "cpu")
    metrics, predictions = workflow.evaluate(model, common.make_loader(val, config), bank, pool, table, config, "cpu")
    factory = lambda repo, cfg: tiny_model(dino, cfg)
    with pytest.raises(ValueError, match="pool/order"):
        workflow.export_bundle(model, best, pool[::-1], classes, table, config, dino[0], output, val[0], metrics,
                               predictions, assets, None, model_factory=factory, caption_runner=fixture_caption)
    archive = workflow.export_bundle(model, best, pool, classes, table, config, dino[0], output, val[0], metrics,
                                     predictions, assets, None, model_factory=factory, caption_runner=fixture_caption)
    assert archive.is_file()
    root = archive.with_suffix("")
    manifest = runtime.verify_bundle(root)
    smoke = visual.read_json(root / "smoke_test.json")
    assert smoke["passed"] and smoke["test_injected_services"]
    assert not smoke["fresh_florence_clip_integration"]  # never claim mock captioning is a real integration check
    predictor = runtime.BundlePredictor(root, "cpu", factory, fixture_caption)
    cached = predictor.predict(val[0]["image_path"])
    assert cached["evidence"]["query_caption"]["source"] == "verified_cache"
    fresh = predictor.predict(val[0]["image_path"], use_caption_cache=False)
    assert fresh["evidence"]["query_caption"]["source"] == "fresh_florence"
    assert set(fresh["evidence"]["attention"]) == {"query_classifier", "candidate_selector"}
    uploaded = predictor.predict(Path(train[0]["image_path"]).read_bytes())
    assert uploaded["evidence"]["candidate"]["image_id"] != train[0]["image_id"]
    assert len(uploaded["top_k"]) == 3
    assert "caption" in fresh["evidence"]["candidate"]
    assert (root / "models/florence/config.json").is_file() and (root / "models/clip/config.json").is_file()
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        handle.extractall(tmp_path / "cold")
    cold = tmp_path / "cold" / root.name
    driver = '''import runpy, sys
from pathlib import Path
import torch
torch.set_num_threads(1)
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "runtime"))
import mtacs_multimodal_web as runtime
def factory(repo, config):
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    backbone = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4,
        n_storage_tokens=2, ffn_ratio=2, drop_path_rate=0.0)
    backbone.init_weights()
    return runtime.MultimodalTACS(backbone, config)
def caption_fixture(request):
    text = "dense neutral shapes covering textured flat surface"
    values = torch.nn.functional.normalize(torch.arange(1,513,dtype=torch.float32).reshape(1,512),dim=1)
    return {"rows":[{"raw_caption":text,"final_caption":text,"caption_valid":True}],"embeddings":values,"valid":torch.tensor([True])}
runtime.BundlePredictor.__init__.__defaults__ = ("cpu", factory, caption_fixture, None)
sys.argv = [str(root / "runtime/verify_export.py"),"--bundle",str(root),"--image",str(next((root / "examples").glob("query.*"))),
            "--image-id","held_out","--device","cpu"]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    process = subprocess.run([sys.executable, "-c", driver, str(cold)], env=env, cwd=cold, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    assert (cold / "examples/reload.pt").is_file()
    # The existing complete bundle can be reused without altering its bytes.
    original_sha = visual.file_sha256(archive)
    assert workflow.export_bundle(model, best, pool, classes, table, config, dino[0], output, val[0], metrics,
        predictions, assets, None, model_factory=factory, caption_runner=fixture_caption) == archive
    assert visual.file_sha256(archive) == original_sha
    # Aligned banks are checked beyond simple file-integrity verification.
    text_bank = torch.load(root / "candidate_text.pt", weights_only=True)
    text_bank["embeddings"] = text_bank["embeddings"].flip(0)
    text_bank["valid"] = text_bank["valid"].flip(0)
    torch.save(text_bank, root / "candidate_text.pt")
    manifest["files"]["candidate_text.pt"] = {"bytes": (root / "candidate_text.pt").stat().st_size,
                                               "sha256": visual.file_sha256(root / "candidate_text.pt")}
    visual.write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="not aligned"):
        runtime.BundlePredictor(root, "cpu", factory, fixture_caption)


def test_export_blocks_incomplete_training(tmp_path, dino):
    with pytest.raises(FileNotFoundError):
        workflow.export_bundle(tiny_model(dino), {}, [], [], None, tiny_config(), dino[0], tmp_path, {}, {}, [], {}, None)


def test_cold_checker_and_cli_have_real_fresh_path():
    compile(workflow.COLD_VERIFY_SOURCE, "verify_export.py", "exec")
    assert "use_caption_cache=False" in workflow.COLD_VERIFY_SOURCE
    assert "last_query_embedding" in workflow.COLD_VERIFY_SOURCE
    assert "/kaggle/input" not in workflow.COLD_VERIFY_SOURCE
