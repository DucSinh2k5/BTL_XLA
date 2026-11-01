"""Small CPU integration checks with the actual official DINOv3 transformer.

No trained score is invented and no production L/16 weights are downloaded.
Export fixtures remain explicitly unverified and cannot be served as production.
"""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/models", "src/training", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / relative))
import isic_segmentation_web as net
import train_isic_segmentation_web as work
import build_isic_segmentation_web_notebooks as builder


@pytest.fixture(scope="module")
def dino():
    repo = Path(os.getenv("DINOV3_TEST_REPO", str(ROOT / "artifacts/deployment/sun397/tacs_visual/vendor/dinov3")))
    if not (repo / "dinov3/models/vision_transformer.py").is_file():
        pytest.skip("Supply DINOV3_TEST_REPO for actual-backbone checks")
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer


def config(mode="tacs"):
    value = work.default_config(mode)
    value.update(feature_dim=32, cross_heads=4, decoder_channels=8, image_size=32, epochs=1,
        batch_size=1, effective_batch_size=4, eval_batch_size=2, feature_batch_size=2,
        warmup_epochs=0, strict_official_counts=False)
    return value


def make_model(dino, cfg):
    backbone = dino(img_size=32, patch_size=16, embed_dim=32, depth=4, num_heads=4,
        n_storage_tokens=2, ffn_ratio=2, drop_path_rate=0.)
    backbone.init_weights()
    return net.ISICSegmentor(backbone, cfg)


def dataset(root):
    for role, count, offset in (("train", 10, 0), ("val", 2, 100), ("test", 2, 200)):
        for index in range(count):
            identity = f"ISIC_{index + offset:07d}"
            image = np.random.default_rng(index + offset).integers(0, 256, (42, 38, 3), dtype=np.uint8)
            mask = np.zeros((42, 38), dtype=np.uint8)
            mask[8:32, 9 + index % 3:30] = 255
            for directory, pixels, suffix in (("images", image, ".png"), ("masks", mask, "_segmentation.png")):
                folder = root / role / directory
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(folder / (identity + suffix))


def text_table(path, frames):
    rows = sum(frames.values(), [])
    values = torch.nn.functional.normalize(torch.randn(len(rows), 512), dim=1)
    valid = torch.ones(len(rows), dtype=torch.bool)
    valid[1] = False
    values[~valid] = 0
    work.atomic_torch(path, {"image_ids": [row["image_id"] for row in rows], "embeddings": values,
        "valid": valid, "identity": "fixture", "caption_csv_sha256": "fixture"})
    return work.TextTable(path)


@pytest.mark.parametrize("mode", ["tacs", "mtacs"])
def test_notebooks_are_self_contained_and_current(mode):
    notebook = builder.build_notebook(mode)
    assert json.loads(builder.TARGETS[mode].read_text(encoding="utf-8")) == notebook
    sections = {cell["metadata"]["section"]: "".join(cell["source"]) for cell in notebook["cells"]}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    payload = builder.embedded_sources(mode)
    for source in payload.values():
        ast.parse(source)
    assert '"100"' in sections["configuration"] and "IMAGE_SIZE = 448" in sections["configuration"]
    assert 'TOKENIZERS_PARALLELISM' in sections["preflight"]
    assert "run_training" in sections["training"] and "export_bundle" in sections["export"]
    if mode == "mtacs":
        assert "prepare_captions" in sections["captions"]
        core = types.ModuleType("caption_core_fixture")
        exec(compile(payload["isic_caption_core.py"], "isic_caption_core.py", "exec"), core.__dict__)
        assert core.POLICY == "isic_visible_attributes_v1"
        assert core.caption_policy("The image shows uneven brown skin with darker central pigment")[1]
        assert not core.caption_policy("brown melanoma with darker central pigment")[1]
        assert 'settings.get("device", "cuda:0")' in payload["isic_caption_core.py"]
        assert "early_stopping=False" in payload["isic_caption_core.py"]
    else:
        assert "isic_caption_core.py" not in payload and "captions" not in sections


def test_official_counts_pairing_and_group_exclusion(tmp_path):
    dataset(tmp_path)
    cfg = config()
    with pytest.raises(ValueError, match="2000"):
        work.load_isic(tmp_path, {**cfg, "strict_official_counts": True})
    frames, pool = work.load_isic(tmp_path, cfg)
    assert [len(frames[role]) for role in ("train", "val", "test")] == [10, 2, 2]
    assert len(pool) == 2
    _, again = work.load_isic(tmp_path, cfg)
    assert [r["image_id"] for r in pool] == [r["image_id"] for r in again]
    allowed = work.allowed_candidates([pool[0]], pool, "cpu")
    assert allowed.tolist() == [[False, True]]
    grouped = [{**pool[0], "group_id": "patient"}, {**pool[1], "group_id": "patient"}]
    with pytest.raises(ValueError, match="without context"):
        work.allowed_candidates([grouped[0]], grouped, "cpu")
    group_file = tmp_path / "groups.csv"
    work.csv_write(group_file, [{"image_id": row["image_id"], "group_id": row["image_id"]} for row in sum(frames.values(), [])])
    grouped_frames, _ = work.load_isic(tmp_path, cfg, group_file)
    assert all(row["group_id"] for row in grouped_frames["train"])
    # A changed raster in another split is rejected by content, not filename.
    import shutil
    shutil.copy2(frames["train"][0]["image_path"], frames["test"][0]["image_path"])
    with pytest.raises(ValueError, match="different splits"):
        work.load_isic(tmp_path, cfg)


def test_original_folder_layout_accepts_test_v2(tmp_path):
    import shutil
    dataset(tmp_path / "prepared")
    original = tmp_path / "original"
    for role, name in (("train", "Training"), ("val", "Validation"), ("test", "Test_v2")):
        shutil.copytree(tmp_path / "prepared" / role / "images", original / f"ISIC-2017_{name}_Data")
        shutil.copytree(tmp_path / "prepared" / role / "masks", original / f"ISIC-2017_{name}_Part1_GroundTruth")
    frames, pool = work.load_isic(original, config())
    assert len(frames["test"]) == 2 and len(pool) == 2


def test_st_token_bank_has_exact_dense_gradient():
    torch.manual_seed(4)
    bank = torch.randn(5, 7, 8).half()
    logits = torch.randn(2, 5, requires_grad=True)
    weights = torch.nn.functional.gumbel_softmax(logits, tau=.2, hard=True, dim=1)
    output = net.select_bank(weights, bank, chunk=2)
    upstream = torch.randn_like(output)
    grad = torch.autograd.grad(output, logits, upstream, retain_graph=True)[0]
    dense = (weights @ bank.float().flatten(1)).reshape_as(output)
    expected = torch.autograd.grad(dense, logits, upstream)[0]
    assert torch.equal(output, dense)
    assert torch.allclose(grad, expected, atol=1e-6)


@pytest.mark.parametrize("mode", ["tacs", "mtacs"])
def test_full_production_l16_architecture_on_meta(dino, mode):
    repo = Path(sys.modules["dinov3"].__file__).resolve().parents[1]
    cfg = work.default_config(mode)
    with torch.device("meta"):
        model = net.build_model(repo, cfg)
    assert len(model.backbone.blocks) == len(model.context_blocks) == 24
    assert model.grid == (28, 28) and model.stage_indices == [5, 11, 17, 23]
    assert all(block.attention.embed_dim == 1024 for block in model.context_blocks)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    if mode == "mtacs":
        assert model.query_text.projection[1].weight.shape == (1024, 512)
        assert model.candidate_text.projection[1].weight.shape == (1024, 512)


@pytest.mark.parametrize("mode", ["tacs", "mtacs"])
def test_native_rope_per_block_gradients_and_attention(dino, mode):
    torch.manual_seed(42)
    cfg = config(mode)
    model = make_model(dino, cfg).eval()
    images = torch.randn(3, 3, 32, 32)
    initial = model.prepare(images)
    actual, attention = model.encode_selector_tokens(initial, True)
    native = model.backbone.forward_features(images)["x_norm_clstoken"]
    assert torch.allclose(actual, native, atol=1e-6)
    assert attention.shape == (3, 2, 2)
    qt = ct = torch.randn(1, 512) if mode == "mtacs" else None
    qv = cv = torch.ones(1, dtype=torch.bool) if mode == "mtacs" else None
    plain, _ = model.decode(initial[:1], initial[1:2], qt, qv, ct, cv)
    explained, details = model.decode(initial[:1], initial[1:2], qt, qv, ct, cv, True)
    assert torch.allclose(plain, explained, atol=1e-6)
    assert details["query_segmentor"].shape == (1, 2, 2)
    assert len(model.context_blocks) == 4
    model.train()
    assert not model.backbone.training
    scores = model.selector(actual[:1], actual[1:])
    weights = torch.nn.functional.gumbel_softmax(scores, hard=True, tau=.1, dim=1)
    selected = net.select_bank(weights, initial[1:].half())
    logits, _ = model.decode(initial[:1], selected, qt, qv, ct, cv)
    target = torch.zeros_like(logits)
    target[:, :, 8:24, 8:24] = 1
    net.segmentation_loss(logits, target).mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.selector.parameters())
    assert all(block.attention.in_proj_weight.grad is not None for block in model.context_blocks)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_invalid_caption_masks_remove_both_fusion_and_semantic_effect(dino):
    cfg = config("mtacs")
    model = make_model(dino, cfg).eval()
    initial = model.prepare(torch.randn(3, 3, 32, 32))
    features, _ = model.encode_selector_tokens(initial)
    q, c = torch.randn(1, 512), torch.randn(2, 512)
    zero = torch.zeros(1, dtype=torch.bool)
    invalid = torch.zeros(2, dtype=torch.bool)
    scores = model.score(features[:1], features[1:], torch.ones(1, 2, dtype=torch.bool), q, zero, c, invalid)
    changed = model.score(features[:1], features[1:], torch.ones(1, 2, dtype=torch.bool), q * 100, zero, -c, invalid)
    assert torch.equal(scores, changed)
    first, gates = model.decode(initial[:1], initial[1:2], q, zero, c[:1], zero)
    second, _ = model.decode(initial[:1], initial[1:2], -q * 100, zero, -c[:1], zero)
    assert torch.equal(first, second)
    assert gates["q_gate"].item() == gates["c_gate"].item() == 0


@pytest.mark.parametrize("mode", ["tacs", "mtacs"])
def test_singleton_microbatch_training_resume_and_best(dino, tmp_path, mode):
    dataset(tmp_path / "dataset")
    cfg = config(mode)
    work.seed_all(42)
    frames, pool = work.load_isic(tmp_path / "dataset", cfg)
    model = make_model(dino, cfg)
    bank = work.build_bank(model, pool, cfg, tmp_path / "bank.pt", "tiny-backbone")
    table = text_table(tmp_path / "text.pt", frames) if mode == "mtacs" else None
    before = {key: value.clone() for key, value in model.selector.state_dict().items()}
    history, best = work.run_training(model, frames, pool, bank, cfg, tmp_path / "run", mode, table)
    assert len(history) == 1 and best["epoch"] == 1
    assert all(np.isfinite(value) for key, value in history[0].items())
    assert history[0]["train_policy_loss"] != 0, "Policy must not disappear with a physical batch of 1"
    assert any(not torch.equal(value, before[key]) for key, value in model.selector.state_dict().items())
    if mode == "mtacs":
        assert history[0]["val_q_gate"] > 0 and history[0]["val_c_gate"] >= 0
    second = make_model(dino, cfg)
    second.backbone.load_state_dict(model.backbone.state_dict())
    resumed, checkpoint = work.run_training(second, frames, pool, bank, cfg, tmp_path / "run", mode, table)
    assert resumed == history
    for key, value in checkpoint["state"].items():
        assert torch.equal(second.state_dict()[key].cpu(), value)
    with pytest.raises(ValueError, match="identity"):
        work.run_training(second, frames, pool, bank, cfg, tmp_path / "run", "different", table)


def test_visual_export_reconstructs_all_model_parts_and_masks(dino, tmp_path, monkeypatch):
    dataset(tmp_path / "dataset")
    cfg = config()
    frames, pool = work.load_isic(tmp_path / "dataset", cfg)
    model = make_model(dino, cfg)
    bank = work.build_bank(model, pool, cfg, tmp_path / "bank.pt", "tiny")
    output = tmp_path / "run"
    work.run_training(model, frames, pool, bank, cfg, output, "fixture")
    weights = tmp_path / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    work.atomic_torch(weights, model.backbone.state_dict())
    def factory(configuration):
        value = make_model(dino, configuration)
        value.backbone.load_state_dict(torch.load(weights, weights_only=True))
        return value
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "LICENSE").write_text("Test fixture only", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    sources = builder.embedded_sources("tacs")
    for name, source in sources.items():
        (runtime / name).write_text(source, encoding="utf-8")
    root, archive = work.export_bundle(model, frames, pool, bank, cfg, output, vendor, weights, runtime,
        {"fixture": True}, cold_verify=False, factory=factory)
    assert archive is None
    manifest = net.io_utils.read_json(root / "manifest.json")
    assert manifest["status"] == "fixture_unverified"
    assert (root / "models/selector.pt").is_file() and (root / "models/segmentor.pt").is_file()
    assert len(list((root / "candidate_images").glob("*.png"))) == 2
    predictor = net.Predictor(root, factory=factory)
    result = predictor.predict(root / "example/query.png")
    assert result["logits"].shape == (1, 1, 32, 32)
    assert result["probabilities"].shape == (42, 38)
    assert result["ground_truth"]["verified"] is True
    assert result["evidence"]["attention"]["query_segmentor"]["grid_shape"] == [2, 2]
    assert result["evidence"]["attention"]["candidate_selector"]["grid_shape"] == [2, 2]
    with pytest.raises(ValueError, match="production"):
        net.Predictor(root)
    # Run the actual CLI entry point and its output serialization with the
    # explicitly injected tiny backbone; production remains strict L/16.
    original_predictor = net.Predictor
    with monkeypatch.context() as patch:
        patch.setattr(net, "Predictor", lambda *args, **kwargs: original_predictor(*args, factory=factory, **kwargs))
        patch.setattr(sys, "argv", ["isic_segmentation_web.py", "--bundle", str(root),
            "--image", str(root / "example/query.png"), "--output", str(tmp_path / "cli")])
        net.main()
    for name in ("mask.png", "overlay.png", "probabilities.npy", "logits.pt", "prediction.json"):
        assert (tmp_path / "cli" / name).is_file()
    manifest["status"] = "staged"
    net.io_utils.write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        net.Predictor(root, factory=factory)
    manifest["status"] = "fixture_unverified"
    net.io_utils.write_json(root / "manifest.json", manifest)
    # Annotation is display/evaluation only: changing it cannot change logits.
    annotation = next(row for row in frames["val"] if row["image_id"] == result["ground_truth"]["image_id"])
    mask_path = root / predictor.ground_truth_index[annotation["pixel_sha256"]]["mask"]
    Image.new("L", (38, 42), 0).save(mask_path)
    changed = predictor.predict(root / "example/query.png")
    assert torch.equal(result["logits"], changed["logits"])
    assert result["comparison"]["dice"] != changed["comparison"]["dice"]
    # A new load verifies artifacts and refuses the changed annotation file.
    with pytest.raises(ValueError, match="integrity"):
        net.Predictor(root, factory=factory)


def test_interrupted_epoch_resume_matches_uninterrupted_training(dino, tmp_path, monkeypatch):
    dataset(tmp_path / "dataset")
    cfg = {**config(), "epochs": 2}
    frames, pool = work.load_isic(tmp_path / "dataset", cfg)
    def fresh():
        work.seed_all(19)
        return make_model(dino, cfg)
    model = fresh()
    bank = work.build_bank(model, pool, cfg, tmp_path / "bank.pt", "tiny")
    epoch_function = work.train_epoch
    def interrupt(*args, **kwargs):
        epoch = args[-1]
        if epoch == 2:
            raise KeyboardInterrupt("fixture interruption")
        return epoch_function(*args, **kwargs)
    monkeypatch.setattr(work, "train_epoch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        work.run_training(model, frames, pool, bank, cfg, tmp_path / "resumed", "same")
    monkeypatch.setattr(work, "train_epoch", epoch_function)
    resumed_model = fresh()
    resumed_history, _ = work.run_training(resumed_model, frames, pool, bank, cfg, tmp_path / "resumed", "same")
    uninterrupted = fresh()
    direct_history, _ = work.run_training(uninterrupted, frames, pool, bank, cfg, tmp_path / "direct", "same")
    assert resumed_history == direct_history
    for key, value in resumed_model.trainable_state().items():
        assert torch.equal(value, uninterrupted.trainable_state()[key])


def test_full_both_export_reloads_text_gates_and_actual_candidate_caption(dino, tmp_path, monkeypatch):
    dataset(tmp_path / "dataset")
    cfg = config("mtacs")
    frames, pool = work.load_isic(tmp_path / "dataset", cfg)
    model = make_model(dino, cfg)
    bank = work.build_bank(model, pool, cfg, tmp_path / "bank.pt", "tiny")
    caption_output = tmp_path / "captions"
    table = text_table(caption_output / "text_cache/clip_text.pt", frames)
    output = tmp_path / "run"
    work.run_training(model, frames, pool, bank, cfg, output, "full_both_fixture", table)
    work.csv_write(caption_output / "captions.csv", [{"image_id": row["image_id"], "split": row["split"],
        "final_caption": "uneven brown surface with dark central pigment"} for row in sum(frames.values(), [])])
    net.io_utils.write_json(caption_output / "web_caption_settings.json", {"fixture": True})
    assets = {key: str(tmp_path / key) for key in ("florence", "clip", "offline_assets_root")}
    for key in ("florence", "clip"):
        folder = Path(assets[key])
        folder.mkdir()
        (folder / "config.json").write_text('{"test_fixture":true}', encoding="utf-8")
    core_module = types.ModuleType("isic_caption_core")
    exec(compile(builder.caption_core(), "isic_caption_core.py", "exec"), core_module.__dict__)
    monkeypatch.setitem(sys.modules, "isic_caption_core", core_module)
    sys.path.insert(0, str(ROOT / "src/captioning"))
    import isic_caption_web
    probes = []
    monkeypatch.setattr(isic_caption_web, "probe_export", lambda *args: probes.append(args) or {"fixture_only": True})
    weights = tmp_path / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    work.atomic_torch(weights, model.backbone.state_dict())
    def factory(configuration):
        value = make_model(dino, configuration)
        value.backbone.load_state_dict(torch.load(weights, weights_only=True))
        return value
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name, source in builder.embedded_sources("mtacs").items():
        (runtime / name).write_text(source, encoding="utf-8")
    root, archive = work.export_bundle(model, frames, pool, bank, cfg, output, vendor, weights, runtime,
        {"fixture": True}, caption_assets=assets, caption_output=caption_output, cold_verify=False, factory=factory)
    assert archive is None and len(probes) == 1
    selector = torch.load(root / "models/selector.pt", weights_only=True)
    segmentor = torch.load(root / "models/segmentor.pt", weights_only=True)
    assert "semantic_mix_logit" in selector
    assert any(key.startswith("query_text.") for key in segmentor)
    assert any(key.startswith("candidate_text.") for key in segmentor)
    predictor = net.Predictor(root, factory=factory)
    result = predictor.predict(root / "example/query.png")
    assert result["evidence"]["query_caption"]["source"] == "verified_pixels"
    selected = result["evidence"]["candidate"]
    assert selected["caption"] == predictor.captions[selected["image_id"]]["final_caption"]
    assert result["evidence"]["fusion"]["q_gate"] > 0
    for key in ("florence", "clip"):
        assert (root / "caption_models" / key / "config.json").is_file()
    # New pixels must not borrow a cached dataset caption.
    with pytest.raises(RuntimeError, match="new M-TACS image"):
        predictor.predict(Image.new("RGB", (38, 42), "red"))


def test_clip_preflight_rejects_incomplete_tokenizer_and_shards(tmp_path, monkeypatch):
    core = types.ModuleType("isic_caption_core")
    exec(compile(builder.caption_core(), "isic_caption_core.py", "exec"), core.__dict__)
    monkeypatch.setitem(sys.modules, "isic_caption_core", core)
    sys.path.insert(0, str(ROOT / "src/captioning"))
    import isic_caption_web
    net.io_utils.write_json(tmp_path / "config.json", {"model_type": "clip", "projection_dim": 512,
        "vision_config": {"patch_size": 32}})
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        isic_caption_web.validate_clip_assets(tmp_path)
    for name in ("tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weights"):
        isic_caption_web.validate_clip_assets(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"explicit fixture")
    assert isic_caption_web.validate_clip_assets(tmp_path) == tmp_path
    net.io_utils.write_json(tmp_path / "pytorch_model.bin.index.json", {"weight_map": {"tensor": "missing.bin"}})
    with pytest.raises(FileNotFoundError, match="shards"):
        isic_caption_web.validate_clip_assets(tmp_path)
