"""CPU tests use a tiny official DINOv3, not fake deployment predictions."""
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
for relative in ("src/models", "src/training", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / relative))
import tacs_visual_web as runtime
import train_sun397_tacs_web as workflow
import build_sun397_tacs_web_notebook as builder


@pytest.fixture(scope="module")
def dino_class():
    path = Path(os.getenv("DINOV3_TEST_REPO", str(ROOT / "tmp/dinov3_web_export_test")))
    if not (path / "dinov3/models/vision_transformer.py").is_file():
        pytest.skip("Set DINOV3_TEST_REPO to official DINOv3 source for actual-backbone CPU checks")
    sys.path.insert(0, str(path))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer, path


def tiny_config():
    config = workflow.default_config()
    config.update(feature_dim=32, num_classes=3, image_size=32, epochs=1, batch_size=2,
                  effective_batch_size=4, feature_batch_size=2, eval_batch_size=2, workers=0)
    return config


def tiny_model(dino_class, config=None):
    cls, _ = dino_class
    config = config or tiny_config()
    backbone = cls(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4,
                   n_storage_tokens=2, ffn_ratio=2, drop_path_rate=0.0)
    backbone.init_weights()
    return runtime.VisualTACS(backbone, config)


def make_rows(folder, count, split, offset=0, classes=None):
    classes = classes or ["a/abbey", "b/bedroom", "c/courtyard"]
    result = []
    for i in range(count):
        class_id = i % len(classes)
        image_id = f"{classes[class_id]}/sun_{split}_{i + offset:04}.png"
        path = folder / image_id
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(i + offset + 123).integers(0, 256, (40, 38, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        result.append({"image_id": image_id, "image_path": str(path), "class_id": class_id,
                       "class_name": classes[class_id], "split": split})
    return result


def test_notebook_standalone_and_no_s16_or_text():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 18
    assert all(c.get("outputs", []) == [] for c in notebook["cells"])
    assert "EMBEDDED_SOURCES" in code and "RUN_FINAL_TEST" in code
    assert "DINOV3_WEIGHTS" in code and "RUN_EXPORT" in code
    assert "dinov3_vits16" not in code and "dinov3_vitb16" not in code
    assert "import transformers" not in code
    assert "TOKENIZERS_PARALLELISM" in code
    assert "train_sun397_tacs_web.py" in code and "cold_subprocess_reload" in code


def test_production_config_refuses_small_backbone_or_singleton_policy():
    config = workflow.default_config()
    workflow.validate_config(config)
    for change in ({"backbone": "dinov3_vits16"}, {"feature_dim": 384}, {"batch_size": 1}, {"caption_used": True}):
        with pytest.raises(ValueError):
            workflow.validate_config({**config, **change})


def test_rope_order_is_concatenation_not_interleaving():
    sin = torch.arange(12).reshape(3, 4)
    cos = sin + 100
    doubled = runtime.concatenate_rope((sin, cos))
    assert torch.equal(doubled[0], torch.cat([sin, sin], dim=-2))
    assert not torch.equal(doubled[0], sin.repeat_interleave(2, dim=-2))
    with pytest.raises(TypeError):
        runtime.concatenate_rope(sin)


def test_self_exclusion_and_no_eligible_candidate():
    scores = torch.tensor([[9.0, 2.0, 3.0], [1.0, 9.0, 4.0]])
    masked = runtime.mask_self(scores, [0, 1])
    assert masked.argmax(1).tolist() == [2, 2]
    assert scores[0, 0] == 9
    with pytest.raises(ValueError):
        runtime.mask_self(torch.full((1, 2), -torch.inf))
    with pytest.raises(ValueError):
        runtime.mask_self(scores, [3, 0])


def test_exact_st_pixel_gradient_matches_full_matmul():
    torch.manual_seed(10)
    bank = torch.randn(5, 3, 4, 4).half()
    weights = torch.nn.functional.one_hot(torch.tensor([2, 4]), 5).float().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    actual = runtime.straight_through_images(weights, bank, chunk_size=2)
    expected = (reference_weights @ bank.float().flatten(1)).reshape_as(actual)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(weights.grad, reference_weights.grad)
    scores = torch.randn(2, 5, requires_grad=True)
    hard = torch.nn.functional.gumbel_softmax(scores, tau=0.7, hard=True)
    runtime.straight_through_images(hard, bank, 2).square().sum().backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        runtime.straight_through_images(torch.ones(2, 5) / 5, bank)


def test_official_selector_features_match_native_and_attention_preserves_logits(dino_class):
    model = tiny_model(dino_class).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model.backbone.forward_features(images)["x_norm_clstoken"]
        actual, selector_map = model.encode_selector(images, return_attention=True)
        torch.testing.assert_close(actual, expected)
        logits = model(images[:1], images[1:])
        explained, maps = model(images[:1], images[1:], return_attention=True)
        torch.testing.assert_close(logits, explained, rtol=0, atol=0)
    assert tuple(selector_map.shape) == (2, 2, 2)
    assert tuple(maps["query_classifier"].shape) == (1, 2, 2)
    assert tuple(maps["context_classifier"].shape) == (1, 2, 2)
    assert torch.isfinite(selector_map).all() and (selector_map >= 0).all()


def test_attention_weights_match_official_attention_computation(dino_class):
    model = tiny_model(dino_class).eval()
    tokens, rope, _ = model._tokens(torch.randn(1, 3, 32, 32))
    block = model.backbone.blocks[0]
    with torch.no_grad():
        qkv = block.attn.qkv(block.norm1(tokens))
        batch, count, triple = qkv.shape
        heads, dim = block.attn.num_heads, triple // 3
        q, k, v = qkv.reshape(batch, count, 3, heads, dim // heads).unbind(2)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        q, k = block.attn.apply_rope(q, k, rope)
        weights = (q @ k.transpose(-2, -1) / (dim // heads) ** 0.5).softmax(-1)
        explicit_output = (weights @ v).transpose(1, 2).reshape(batch, count, dim)
        official_output = block.attn.compute_attention(qkv, rope=rope)
        torch.testing.assert_close(explicit_output, official_output, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model._cls_attention(block, tokens, rope), weights.mean(1)[:, 0], atol=1e-6, rtol=1e-5)


def test_local_factory_does_not_execute_hubconf(dino_class, monkeypatch):
    from dinov3.hub import backbones
    calls = []

    class ShapeOnlyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 1024
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])

    def constructor(**kwargs):
        calls.append(kwargs)
        return ShapeOnlyBackbone()

    monkeypatch.setattr(backbones, "dinov3_vitl16", constructor)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: pytest.fail("Unrelated hubconf must not execute"))
    model = runtime.build_model(dino_class[1], workflow.default_config())
    assert model.head[-1].in_features == 1024
    assert calls == [{"pretrained": False}]


def test_production_l16_construction_on_meta_device(dino_class):
    # Verify the real L/16 factory without allocating 300M parameters on CPU/GPU.
    with torch.device("meta"):
        model = runtime.build_model(dino_class[1], workflow.default_config())
    assert model.backbone.embed_dim == 1024
    assert len(model.backbone.blocks) == 24
    assert model.head[-1].out_features == 397
    assert next(model.parameters()).device.type == "meta"
    assert sum(parameter.numel() for parameter in model.parameters()) > 300_000_000


def test_training_selector_matches_native_per_block_positional_augmentation(dino_class):
    model = tiny_model(dino_class).train()
    model.backbone.rope_embed.rescale_coords = 2.0
    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(123)
    expected = model.backbone.forward_features(images)["x_norm_clstoken"]
    torch.manual_seed(123)
    actual = model.encode_selector(images)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    assert model.backbone.patch_embed.proj.weight.grad is not None


def test_dataset_splits_pool_and_missing_file_fail(tmp_path):
    classes = ["a/abbey", "b/bedroom"]
    (tmp_path / "ClassName.txt").write_text("\n".join("/" + c for c in classes), encoding="utf-8")
    train = make_rows(tmp_path, 100, "train", classes=classes)
    test = make_rows(tmp_path, 100, "test", offset=100, classes=classes)
    train_file, test_file = tmp_path / "Training_01.txt", tmp_path / "Testing_01.txt"
    train_file.write_text("\n".join("/" + r["image_id"] for r in train), encoding="utf-8")
    test_file.write_text("\n".join("/" + r["image_id"] for r in test), encoding="utf-8")
    config = {**workflow.default_config(), "num_classes": 2}
    names, actual_train, val, actual_test, pool = workflow.load_sun397(tmp_path, train_file, test_file, config)
    assert names == classes
    assert [len(x) for x in (actual_train, val, actual_test, pool)] == [90, 10, 100, 18]
    assert {r["image_id"] for r in pool} <= {r["image_id"] for r in actual_train}
    assert not ({r["image_id"] for r in pool} & {r["image_id"] for r in val + actual_test})
    Path(actual_test[0]["image_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        workflow.load_sun397(tmp_path, train_file, test_file, config)


def test_path_guard_and_flat_attention():
    for path in ("../outside", "/absolute", "C:/test", "a\\b"):
        with pytest.raises(ValueError):
            runtime.bundle_path(ROOT, path)
    image = Image.new("RGB", (40, 20), "gray")
    overlay, informative = runtime.attention_overlay(image, np.ones((2, 2)))
    assert overlay.size == (40, 20) and not informative
    with pytest.raises(ValueError):
        runtime.normalize_map([[float("nan")]])


def test_f1_matches_sklearn_when_available():
    metrics = pytest.importorskip("sklearn.metrics")
    truth, prediction = [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2]
    actual = workflow.classification_metrics(truth, prediction, 3)
    assert actual["accuracy"] == metrics.accuracy_score(truth, prediction)
    assert actual["macro_f1"] == pytest.approx(metrics.f1_score(truth, prediction, average="macro"))


def test_metric_values_and_absent_class():
    actual = workflow.classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2], 3)
    assert actual == {"accuracy": 0.5, "macro_f1": 0.5}
    assert workflow.classification_metrics([0], [0], 2) == {"accuracy": 1.0, "macro_f1": 0.5}


def test_train_resume_export_reload_and_checksum(tmp_path, dino_class, monkeypatch):
    # Only test dimensions/class count are overridden; the actual training path runs.
    monkeypatch.setattr(workflow, "validate_config", lambda config: None)
    config = tiny_config()
    classes = ["a/abbey", "b/bedroom", "c/courtyard"]
    train = make_rows(tmp_path / "images", 6, "train")
    val = make_rows(tmp_path / "images", 3, "val", offset=100)
    pool = [dict(r, position=i) for i, r in enumerate(train[:4])]
    output = tmp_path / "run"
    workflow.seed_all(42)
    model = tiny_model(dino_class, config)
    best, history = workflow.train_run(model, classes, train, val, pool, config, {"fixture": "official_tiny_dino"}, output, "cpu")
    assert len(history) == 1 and best["epoch"] == 1
    assert np.isfinite(history[0]["train_loss"])
    assert (output / "last.pt").is_file() and (output / "training_complete.json").is_file()
    best_again, resumed = workflow.train_run(model, classes, train, val, pool, config, {"fixture": "official_tiny_dino"}, output, "cpu")
    assert len(resumed) == 1
    for key, value in best["model_state_dict"].items():
        torch.testing.assert_close(value, best_again["model_state_dict"][key])
    with pytest.raises(ValueError, match="another run"):
        workflow.train_run(model, classes, train, val, pool, {**config, "epochs": 2}, {"fixture": "official_tiny_dino"}, output, "cpu")
    bank = workflow.refresh_bank(model, pool, config, "cpu")
    metrics, rows = workflow.evaluate(model, workflow.make_loader(val, config), bank, None, pool, config, "cpu")
    factory = lambda repo, cfg: tiny_model(dino_class, cfg)
    with pytest.raises(ValueError, match="pool/order"):
        workflow.export_bundle(model, best, pool[::-1], classes, config, dino_class[1], output, val[0], metrics, rows, model_factory=factory)
    archive = workflow.export_bundle(model, best, pool, classes, config, dino_class[1], output, val[0], metrics, rows, model_factory=factory)
    assert archive.is_file()
    archive_hash = runtime.file_sha256(archive)
    assert workflow.export_bundle(model, best, pool, classes, config, dino_class[1], output, val[0], metrics, rows, model_factory=factory) == archive
    assert runtime.file_sha256(archive) == archive_hash
    root = archive.with_suffix("")
    manifest = runtime.verify_bundle(root)
    assert all(not Path(name).is_absolute() for name in manifest["files"])
    assert len(runtime.read_json(root / "candidate_pool.json")) == len(pool)
    assert runtime.read_json(root / "smoke_test.json")["passed"]
    assert not runtime.read_json(root / "smoke_test.json")["cold_subprocess_reload"]
    assert "runtime/verify_export.py" in manifest["files"]
    predictor = runtime.BundlePredictor(root, model_factory=factory)
    result = predictor.predict(Path(train[0]["image_path"]).read_bytes(), explain=True)
    assert result["evidence"]["candidate"]["image_id"] != train[0]["image_id"]
    assert result["evidence"]["query_caption"]["status"] == "not_used"
    assert set(result["evidence"]["attention"]) == {"query_classifier", "candidate_selector"}
    assert all(a["method"] == runtime.ATTENTION_METHOD and a["grid_shape"] == [2, 2] for a in result["evidence"]["attention"].values())
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert root.name + "/model.pt" in handle.namelist()
        assert all("__pycache__" not in name for name in handle.namelist())
        handle.extractall(tmp_path / "cold")
    cold_root = tmp_path / "cold" / root.name
    # Only the backbone dimensions are injected; run the real bundled cold checker
    # in a fresh process without access to project modules or training Inputs.
    driver = '''import runpy, sys
from pathlib import Path
import torch
torch.set_num_threads(1)
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "runtime"))
import tacs_visual_web as runtime
def tiny_factory(repo, config):
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    backbone = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=32, depth=2,
        num_heads=4, n_storage_tokens=2, ffn_ratio=2, drop_path_rate=0.0)
    backbone.init_weights()
    return runtime.VisualTACS(backbone, config)
runtime.BundlePredictor.__init__.__defaults__ = ("cpu", tiny_factory)
sys.argv = [str(root / "runtime/verify_export.py"), "--bundle", str(root),
    "--image", str(next((root / "examples").glob("query.*"))), "--image-id", "held_out", "--device", "cpu"]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env.pop("PYTHONPATH", None)
    process = subprocess.run([sys.executable, "-c", driver, str(cold_root)], cwd=cold_root,
                             env=env, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Cold bundle inference" in process.stdout
    assert (cold_root / "examples/reload_logits.pt").is_file()
    assert len(runtime.read_json(cold_root / "examples/prediction.json")["evidence"]["attention"]) == 2
    image = root / runtime.read_json(root / "candidate_pool.json")[0]["file"]
    image.write_bytes(b"corrupted image")
    with pytest.raises(ValueError, match="checksum"):
        runtime.verify_bundle(root)


def test_budget_guard_and_source_identity(tmp_path, dino_class):
    config = tiny_config()
    model = tiny_model(dino_class)
    with pytest.raises(FileNotFoundError):
        workflow.export_bundle(model, {}, [], [], config, dino_class[1], tmp_path, {}, {}, [])
    first = workflow.run_identity(config, ["x"], {"train": []}, {"sha": "a"})
    second = workflow.run_identity({**config, "epochs": 50}, ["x"], {"train": []}, {"sha": "a"})
    assert first != second


def test_cold_checker_compiles_and_does_not_require_training_mounts():
    compile(workflow.COLD_VERIFY_SOURCE, "verify_export.py", "exec")
    assert 'vendor/dinov3' in workflow.COLD_VERIFY_SOURCE
    assert 'plain_logits' in workflow.COLD_VERIFY_SOURCE
    assert '/kaggle/input' not in workflow.COLD_VERIFY_SOURCE
