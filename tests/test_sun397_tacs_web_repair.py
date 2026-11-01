"""Regression checks for the interrupted visual notebook's memory repair."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/models", "src/training", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / relative))
import tacs_visual_web as runtime
import train_sun397_tacs_web as legacy
import repair_sun397_tacs_web_notebook as builder
from test_sun397_tacs_web_export import dino_class, make_rows, tiny_config, tiny_model


@pytest.fixture
def repaired(tmp_path):
    path = tmp_path / "repaired_training.py"
    path.write_text(builder.repaired_training_source(), encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("repaired_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repair_config():
    return {**tiny_config(), "training_engine": "memory_bounded_two_pass_v2"}


def test_repaired_notebook_embeds_checked_inference_and_compilable_training():
    notebook = json.loads(builder.TARGET.read_text(encoding="utf-8"))
    original = json.loads(builder.BACKUP.read_text(encoding="utf-8"))
    assert notebook == builder.build_notebook(original)
    assert len(notebook["cells"]) == len(original["cells"])
    assert any(c.get("outputs") for c in original["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), "cell", "exec")
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    embedded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "EMBEDDED_SOURCES" for t in n.targets))
    assert embedded["tacs_visual_web.py"] == builder.MODEL.read_text(encoding="utf-8")
    assert embedded["train_sun397_tacs_web.py"] == builder.repaired_training_source()
    code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "COMPUTE_BATCH_SIZE" in code and "LOGICAL_BATCH_SIZE = 64" in code
    assert "gradient_checkpointing=True" in code
    assert "execution=EXECUTION, resume_from=RESUME_FROM_DIR" in code
    assert "expandable_segments:True" in code
    assert "/datasets/sinhnguync/dinov3-code/" in code
    assert "shutil.rmtree(OUTPUT_DIR" not in code.replace("# shutil.rmtree(OUTPUT_DIR", "# disabled")


def test_execution_chunk_can_be_12_without_changing_logical_batch(repaired):
    config = repaired.default_config()
    repaired.validate_config(config)
    assert config["batch_size"] == config["effective_batch_size"] == 64
    options = repaired.execution_options(config, {"compute_batch_size": 12})
    assert options["compute_batch_size"] == 12
    for supplied in ({"compute_batch_size": 0}, {"compute_batch_size": 65}, {"pixel_bank_device": "other"}):
        with pytest.raises(ValueError):
            repaired.execution_options(config, supplied)
    with pytest.raises(ValueError, match="checkpointing"):
        repaired.validate_config({**config, "gradient_checkpointing": False})


def test_advantage_uses_full_logical_batch_not_each_compute_chunk(repaired):
    rewards = torch.tensor([1.0, 3.0, 10.0, 18.0])
    full = repaired.policy_advantage(rewards)
    chunks = torch.cat([repaired.policy_advantage(part) for part in rewards.chunk(2)])
    torch.testing.assert_close(full.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(full.std(unbiased=False), torch.tensor(1.0))
    assert not torch.allclose(full, chunks)
    assert torch.isfinite(repaired.policy_advantage(torch.ones(4))).all()


def test_selector_rng_replay_preserves_global_stream_and_native_scores(repaired, dino_class):
    config = repair_config()
    model = tiny_model(dino_class, config).train()
    model.backbone.rope_embed.rescale_coords = 2.0
    images = torch.randn(2, 3, 32, 32)
    features = torch.randn(4, 32)
    state = repaired._selector_rng("cpu")
    with torch.no_grad():
        expected, _ = model.score(images, features)
    stream_after_prepass = torch.get_rng_state().clone()
    with repaired._replay_selector_rng(state, "cpu"):
        actual, _ = model.score(images, features)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.get_rng_state(), stream_after_prepass)
    actual.square().mean().backward()
    assert model.backbone.patch_embed.proj.weight.grad.abs().sum() > 0


def test_two_pass_training_uses_chunked_graphs_and_full_reward_batch(repaired, dino_class, tmp_path, monkeypatch):
    config = {**repair_config(), "batch_size": 4, "effective_batch_size": 4, "gumbel_tau": 0.7}
    model = tiny_model(dino_class, config)
    train = make_rows(tmp_path, 4, "train")
    pool = [dict(r, position=i) for i, r in enumerate(train)]
    bank = repaired.refresh_bank(model, pool, config, "cpu")
    pixels = repaired.prepare_training_pixels(pool, config, "cpu", repaired.execution_options(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    scores_with_grad, rewards_received, task_context_grad = [], [], []
    original_score, original_adv = model.score, repaired.policy_advantage

    def score(*args, **kwargs):
        scores, features = original_score(*args, **kwargs)
        assert len(scores) <= 2
        if torch.is_grad_enabled():
            scores.retain_grad()
            scores_with_grad.append(scores)
        return scores, features

    def advantage(rewards):
        rewards_received.append(rewards.clone())
        return original_adv(rewards)

    original_st = runtime.straight_through_images

    def st(weights, *args):
        context = original_st(weights, *args)
        context.retain_grad()
        task_context_grad.append(context)
        return context

    monkeypatch.setattr(model, "score", score)
    monkeypatch.setattr(repaired, "policy_advantage", advantage)
    monkeypatch.setattr(runtime, "straight_through_images", st)
    metrics = repaired.train_epoch(model, repaired.make_loader(train, config, True), bank, pixels, pool,
                                   optimizer, scaler, config, "cpu", 1, {"compute_batch_size": 2, "profile_batches": 0})
    assert len(rewards_received) == 1 and len(rewards_received[0]) == 4
    assert metrics["optimizer_steps"] == 1 and metrics["images_per_second"] > 0
    assert all(s.grad is not None and torch.isfinite(s.grad).all() for s in scores_with_grad)
    assert sum(float(s.grad.abs().sum()) for s in scores_with_grad) > 0
    assert all(c.grad is not None and c.grad.abs().sum() > 0 for c in task_context_grad)
    assert model.gradient_checkpointing


def test_cpu_fallback_pixel_values_and_explicit_cuda_guard(repaired, tmp_path):
    config = repair_config()
    pool = make_rows(tmp_path, 3, "train")
    options = repaired.execution_options(config, {"pixel_bank_device": "cpu"})
    actual = repaired.prepare_training_pixels(pool, config, "cpu", options)
    expected = legacy.build_image_bank(pool, config)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="without a CUDA"):
        repaired.prepare_training_pixels(pool, config, "cpu", {**options, "pixel_bank_device": "cuda"})


def legacy_checkpoint(tmp_path, dino_class, monkeypatch):
    config = {**tiny_config(), "epochs": 2}
    classes = ["a/abbey", "b/bedroom", "c/courtyard"]
    train = make_rows(tmp_path / "images", 4, "train")
    val = make_rows(tmp_path / "images", 3, "val", offset=100)
    pool = [dict(r, position=i) for i, r in enumerate(train)]
    assets = {"fixture": "native_tiny_dino"}
    model = tiny_model(dino_class, config)
    old = tmp_path / "legacy"
    original_epoch = legacy.train_epoch

    def interrupt(*args, **kwargs):
        if args[-1] == 2:
            raise KeyboardInterrupt("test interrupted after first saved epoch")
        return original_epoch(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(legacy, "validate_config", lambda config: None)
        patch.setattr(legacy, "train_epoch", interrupt)
        with pytest.raises(KeyboardInterrupt):
            legacy.train_run(model, classes, train, val, pool, config, assets, old, "cpu")
    return model, classes, train, val, pool, config, assets, old


def test_checked_migration_preserves_weights_optimizer_and_originals(repaired, dino_class, tmp_path, monkeypatch):
    model, classes, train, val, pool, old_config, assets, old = legacy_checkpoint(tmp_path, dino_class, monkeypatch)
    before = {name: runtime.file_sha256(old / name) for name in ("last.pt", "best.pt", "protocol.json")}
    config = {**old_config, "training_engine": "memory_bounded_two_pass_v2"}
    output = tmp_path / "repaired"
    frames = {"train": train, "val": val, "pool": pool}
    audit = repaired.migrate_legacy_run(old, output, config, classes, frames, assets)
    assert audit["completed_epochs"] == 1
    original = torch.load(old / "last.pt", map_location="cpu", weights_only=True)
    migrated = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert migrated["identity"] != original["identity"]
    assert migrated["epoch"] == original["epoch"] == 1
    assert migrated["scheduler_state_dict"] == original["scheduler_state_dict"]
    assert migrated["optimizer_state_dict"]["param_groups"] == original["optimizer_state_dict"]["param_groups"]
    for key, value in original["model_state_dict"].items():
        torch.testing.assert_close(value, migrated["model_state_dict"][key], atol=0, rtol=0)
    for key, state in original["optimizer_state_dict"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, migrated["optimizer_state_dict"]["state"][key][name], atol=0, rtol=0)
    assert before == {name: runtime.file_sha256(old / name) for name in before}
    with pytest.raises(ValueError, match="identity"):
        repaired.migrate_legacy_run(old, tmp_path / "bad_order", config, classes,
                                    {**frames, "pool": pool[::-1]}, assets)
    with pytest.raises(ValueError, match="logical batch"):
        repaired.migrate_legacy_run(old, tmp_path / "bad_horizon", {**config, "epochs": 3}, classes, frames, assets)


def test_legacy_resume_finishes_and_exports_same_inference_contract(repaired, dino_class, tmp_path, monkeypatch):
    model, classes, train, val, pool, old_config, assets, old = legacy_checkpoint(tmp_path, dino_class, monkeypatch)
    config = {**old_config, "training_engine": "memory_bounded_two_pass_v2"}
    monkeypatch.setattr(repaired, "validate_config", lambda config: None)
    output = tmp_path / "repaired"
    best, history = repaired.train_run(model, classes, train, val, pool, config, assets, output, "cpu",
                                       execution={"compute_batch_size": 1, "profile_batches": 0}, resume_from=old)
    assert [row["epoch"] for row in history] == [1, 2]
    assert history[1]["train_compute_batch_size"] == 1
    with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["train_compute_batch_size"] == ""
    assert rows[1]["train_compute_batch_size"] == "1"
    assert float(rows[1]["refresh_seconds"]) >= 0
    assert rows[1]["train_pixel_bank_device"] == "cpu"
    assert runtime.read_json(output / "protocol.json")["migration"]["completed_epochs"] == 1
    bank = repaired.refresh_bank(model, pool, config, "cpu")
    metrics, predictions = repaired.evaluate(model, repaired.make_loader(val, config), bank, None, pool, config, "cpu")
    factory = lambda repo, cfg: tiny_model(dino_class, cfg)
    archive = repaired.export_bundle(model, best, pool, classes, config, dino_class[1], output,
                                     val[0], metrics, predictions, model_factory=factory)
    root = archive.with_suffix("")
    assert runtime.verify_bundle(root)["status"] == "ready"
    assert runtime.read_json(root / "protocol.json")["migration"]["completed_epochs"] == 1
    predictor = runtime.BundlePredictor(root, model_factory=factory)
    result = predictor.predict(val[0]["image_path"], explain=True)
    assert set(result["evidence"]["attention"]) == {"query_classifier", "candidate_selector"}
    # Execution chunk changes are not experiment identity changes.
    same_best, same_history = repaired.train_run(model, classes, train, val, pool, config, assets, output, "cpu",
                                                execution={"compute_batch_size": 2, "profile_batches": 0}, resume_from=old)
    assert len(same_history) == 2 and same_best["epoch"] == best["epoch"]


def test_registered_config_keeps_old_logical_batch_and_rejects_lr_change(repaired, tmp_path):
    old = legacy.default_config()
    source = tmp_path / "old"
    source.mkdir()
    runtime.write_json(source / "protocol.json", {"config": old})
    requested = repaired.default_config()
    saved = repaired.config_for_resume(requested, tmp_path / "new", source)
    assert saved["batch_size"] == old["batch_size"] == 2
    assert saved["gradient_checkpointing"]
    with pytest.raises(ValueError, match="backbone_lr"):
        repaired.config_for_resume({**requested, "backbone_lr": 0.01}, tmp_path / "new", source)


def test_failed_first_epoch_is_not_claimed_as_resumed_weights(repaired, tmp_path):
    config = legacy.default_config()
    source = tmp_path / "old"
    source.mkdir()
    frames = {"train": [], "val": [], "pool": []}
    identity = repaired._legacy_identity(config, ["x"], frames, {"fixture": 1})
    runtime.write_json(source / "protocol.json", {"config": config, "identity": identity,
                                               "source_sha256": repaired.LEGACY_SOURCE_SHA256, "assets": {"fixture": 1}})
    new_config = {**config, "training_engine": "memory_bounded_two_pass_v2"}
    audit = repaired.migrate_legacy_run(source, tmp_path / "new", new_config, ["x"], frames, {"fixture": 1})
    assert audit["completed_epochs"] == 0
    assert not (tmp_path / "new/last.pt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA is required for GPU bank and replay checks")
def test_cuda_bank_and_autocast_two_pass_gradients(repaired, dino_class, tmp_path):
    config = {**repair_config(), "batch_size": 4, "effective_batch_size": 4, "gumbel_tau": 0.7}
    model = tiny_model(dino_class, config).cuda()
    train = make_rows(tmp_path, 4, "train")
    pool = [dict(r, position=i) for i, r in enumerate(train)]
    options = repaired.execution_options(config, {"compute_batch_size": 2, "pixel_bank_device": "cuda",
                                                "gpu_reserve_gib": 0, "profile_batches": 1})
    pixels = repaired.prepare_training_pixels(pool, config, "cuda", options)
    assert pixels.is_cuda and pixels.dtype == torch.float32
    torch.testing.assert_close(pixels.cpu(), legacy.build_image_bank(pool, config).float(), atol=0, rtol=0)
    bank = repaired.refresh_bank(model, pool, config, "cuda", amp=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=not torch.cuda.is_bf16_supported())
    metrics = repaired.train_epoch(model, repaired.make_loader(train, config, True), bank, pixels, pool,
                                   optimizer, scaler, config, "cuda", 1, options)
    assert metrics["optimizer_steps"] == 1 and metrics["peak_allocated_gib"] > 0
    assert metrics["pixel_bank_device"].startswith("cuda")
