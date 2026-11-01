"""Full_both-specific memory, text-alignment, cache and checkpoint regressions."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/models", "src/training", "src/captioning", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / relative))
import tacs_visual_web as visual
import train_sun397_tacs_web as legacy_common
import train_sun397_mtacs_web as legacy
import sun397_caption_web as captions
import mtacs_multimodal_web as runtime
import repair_sun397_mtacs_web_notebook as builder
from test_sun397_mtacs_web_export import (
    dino, fixture_assets, fixture_caption, fixture_texts, image_rows, tiny_config, tiny_model,
)


@pytest.fixture
def repaired(tmp_path, monkeypatch):
    def load(name, filename, source):
        path = tmp_path / filename
        path.write_text(source, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("train_sun397_tacs_web", "train_sun397_tacs_web.py", builder.visual_training_source())
    return load("repaired_mtacs_training", "train_sun397_mtacs_web.py", builder.repaired_training_source())


def repair_config():
    return {**tiny_config(), "training_engine": "memory_bounded_two_pass_v2"}


def registered_texts(shared, rows, assets):
    cache = fixture_texts(shared, rows, assets)
    settings = visual.read_json(shared / "caption_settings.json")["settings"]
    classes = ["a/abbey", "b/bedroom", "c/courtyard"]
    payload = {"records": rows, "classes": classes, "asset_identity": assets, "settings": settings}
    visual.write_json(shared / "caption_protocol.json", captions.caption_protocol(payload))
    return cache, settings, classes


def bound_assets(shared, models):
    return {"caption_assets": models, "caption_csv_sha256": visual.file_sha256(shared / "captions.csv"),
            "text_cache_sha256": visual.file_sha256(shared / "text_embeddings.pt"),
            "caption_settings_sha256": visual.file_sha256(shared / "caption_settings.json")}


def test_repaired_notebook_compiles_and_preserves_all_inference_sources():
    notebook = json.loads(builder.TARGET.read_text(encoding="utf-8"))
    assert notebook == builder.build_notebook()
    assert notebook["metadata"]["tacs"]["repair"] == "memory_bounded_two_pass_v2"
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    embedded = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "EMBEDDED_SOURCES" for t in node.targets))
    for name in ("tacs_visual_web.py", "mtacs_multimodal_web.py", "sun397_caption_web.py"):
        assert embedded[name] == builder.SOURCES[name].read_text(encoding="utf-8")
    assert embedded["train_sun397_tacs_web.py"] == builder.visual_training_source()
    assert embedded["train_sun397_mtacs_web.py"] == builder.repaired_training_source()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), "cell", "exec")
    all_code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "workflow.stage_previous_caption_cache" in all_code
    assert "execution=EXECUTION, resume_from=RESUME_FROM_DIR" in all_code
    assert "gradient_checkpointing=True" in all_code


def test_config_retains_full_both_and_accepts_independent_compute_chunk(repaired):
    config = repaired.default_config()
    repaired.validate_config(config)
    assert config["batch_size"] == config["effective_batch_size"] == 64
    assert config["caption_used"] and config["method"] == "M_TACS_full_both"
    assert config["text_dim"] == 512 and config["context_top_k"] == 1
    assert config["caption_policy"] == captions.POLICY
    assert repaired.common.execution_options(config, {"compute_batch_size": 12})["compute_batch_size"] == 12
    for changed in ({"gradient_checkpointing": False}, {"caption_used": False}, {"context_top_k": 3}):
        with pytest.raises(ValueError):
            repaired.validate_config({**config, **changed})


def test_complete_caption_cache_is_copied_and_reused_without_generation(repaired, tmp_path, monkeypatch):
    records = image_rows(tmp_path / "images", 6, "train")
    models = fixture_assets(tmp_path / "assets")
    source, target = tmp_path / "old", tmp_path / "new/shared"
    cache, settings, classes = registered_texts(source / "shared", records, models)
    before = {name: visual.file_sha256(source / "shared" / name) for name in
              ("captions.csv", "text_embeddings.pt", "caption_protocol.json", "caption_settings.json")}
    assert repaired.stage_previous_caption_cache(source, target, records, classes, models, settings) == 6
    assert before == {name: visual.file_sha256(target / name) for name in before}
    monkeypatch.setattr(captions, "invoke_worker", lambda *a, **kw: pytest.fail("Complete cache must not regenerate captions"))
    # Production supplies the already-probed caption child environment. This
    # fixture must not invoke a real dependency installer on the CPU test host.
    actual, _ = repaired.prepare_captions(records, classes, models, settings, target, "cpu", None, {"TEST_ONLY": "1"})
    torch.testing.assert_close(actual["embeddings"], cache["embeddings"], atol=0, rtol=0)
    assert repaired.stage_previous_caption_cache(source, target, records, classes, models, settings) == 6
    assert before == {name: visual.file_sha256(source / "shared" / name) for name in before}


def test_partial_cache_can_progress_without_being_replaced_by_old_partial(repaired, tmp_path):
    records = image_rows(tmp_path / "images", 6, "train")
    models = fixture_assets(tmp_path / "assets")
    source, target = tmp_path / "old", tmp_path / "new/shared"
    _, settings, classes = registered_texts(source / "shared", records, models)
    all_rows = list(captions.load_csv(source / "shared/captions.csv").values())
    legacy_common.csv_write(source / "shared/captions.csv", all_rows[:3], captions.COLUMNS)
    (source / "shared/text_embeddings.pt").unlink()
    (source / "shared/caption_settings.json").unlink()
    assert repaired.stage_previous_caption_cache(source, target, records, classes, models, settings) == 3
    legacy_common.csv_write(target / "captions.csv", all_rows, captions.COLUMNS)
    assert repaired.stage_previous_caption_cache(source, target, records, classes, models, settings) == 6
    assert len(captions.load_csv(source / "shared/captions.csv")) == 3


def test_checkpoint_cannot_resume_with_missing_or_replaced_caption_cache(repaired, tmp_path):
    records = image_rows(tmp_path / "images", 3, "train")
    models = fixture_assets(tmp_path / "assets")
    source = tmp_path / "old"
    _, settings, classes = registered_texts(source / "shared", records, models)
    legacy_common.atomic_torch_save(source / "last.pt", {"test_marker": True})
    (source / "shared/text_embeddings.pt").unlink()
    with pytest.raises(FileNotFoundError, match="complete shared"):
        repaired.stage_previous_caption_cache(source, tmp_path / "new/shared", records, classes, models, settings)
    (source / "last.pt").unlink()
    with pytest.raises(ValueError, match="settings/assets"):
        repaired.stage_previous_caption_cache(source, tmp_path / "other/shared", records, classes, models,
                                              {**settings, "max_new_tokens": 99})


def test_caption_settings_retain_registered_generation_batch(repaired, tmp_path):
    rows = image_rows(tmp_path / "images", 3, "train")
    models = fixture_assets(tmp_path / "assets")
    source = tmp_path / "old"
    _, settings, _ = registered_texts(source / "shared", rows, models)
    requested = {**settings, "batch_size": 8}
    restored = repaired.caption_settings_for_resume(requested, tmp_path / "new/shared", source)
    assert restored == settings and restored["batch_size"] == 2
    with pytest.raises(ValueError, match="token budget"):
        repaired.caption_settings_for_resume({**requested, "max_new_tokens": 99}, tmp_path / "new/shared", source)


def test_reward_pairs_policy_image_with_exact_corresponding_caption(repaired, dino, tmp_path, monkeypatch):
    config = repair_config()
    model = tiny_model(dino, config).train()
    records = image_rows(tmp_path / "images", 6, "train")
    models = fixture_assets(tmp_path / "assets")
    cache, _, _ = registered_texts(tmp_path / "shared", records, models)
    table = repaired.TextTable(cache, tmp_path / "shared/captions.csv", records, config)
    candidates = records[:4]
    pixels = legacy_common.build_image_bank(candidates, config)
    candidate_text, candidate_valid = table.gather([r["image_id"] for r in candidates], "cpu")
    query = torch.stack([visual.preprocess(r["image_path"], config) for r in records[4:]])
    text, valid = table.gather([r["image_id"] for r in records[4:]], "cpu")
    actions, labels = torch.tensor([3, 0]), torch.tensor([1, 2])
    original = model.forward
    calls = []

    def forward(*args, **kwargs):
        assert not torch.is_grad_enabled() and not model.training
        if len(args) == 1:
            torch.testing.assert_close(kwargs["query_text"], text)
            assert torch.equal(kwargs["query_valid"], valid)
            calls.append("query_caption_baseline")
        else:
            torch.testing.assert_close(args[1], pixels[actions].float(), atol=0, rtol=0)
            torch.testing.assert_close(args[2], text, atol=0, rtol=0)
            torch.testing.assert_close(args[4], candidate_text[actions], atol=0, rtol=0)
            assert torch.equal(args[5], candidate_valid[actions])
            calls.append("matching_candidate_image_and_caption")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    rewards = repaired._reward_pass(model, query, labels, text, valid, actions, pixels, candidate_text, candidate_valid, "cpu")
    assert rewards.shape == (2,) and not rewards.requires_grad and model.training
    assert calls == ["query_caption_baseline", "matching_candidate_image_and_caption"]


def test_chunked_full_both_preserves_pixel_text_gate_and_selector_gradients(repaired, dino, tmp_path, monkeypatch):
    torch.manual_seed(4)
    config = {**repair_config(), "batch_size": 4, "effective_batch_size": 4, "gumbel_tau": 0.7}
    model = tiny_model(dino, config)
    model.backbone.rope_embed.rescale_coords = 2.0
    records = image_rows(tmp_path / "images", 4, "train")
    models = fixture_assets(tmp_path / "assets")
    cache, _, _ = registered_texts(tmp_path / "shared", records, models)
    table = repaired.TextTable(cache, tmp_path / "shared/captions.csv", records, config)
    candidates = [dict(row, position=i) for i, row in enumerate(records)]
    bank = repaired.common.refresh_bank(model, candidates, config, "cpu")
    pixels = legacy_common.build_image_bank(candidates, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    original_score, original_forward, original_st = model.score, model.forward, visual.straight_through_images
    original_advantage, original_step = repaired.common.policy_advantage, optimizer.step
    weights_seen, selected_text_seen, reward_groups, gradients = [], [], [], {}

    def score(*args, **kwargs):
        result = original_score(*args, **kwargs)
        assert result[0].shape[0] <= 2
        return result

    def st(weights, *args):
        weights.retain_grad()
        weights_seen.append(weights)
        return original_st(weights, *args)

    def forward(*args, **kwargs):
        if model.training and torch.is_grad_enabled() and len(args) >= 6:
            args[4].retain_grad()
            selected_text_seen.append(args[4])
        return original_forward(*args, **kwargs)

    def advantage(rewards):
        reward_groups.append(rewards.clone())
        return original_advantage(rewards)

    def step(*args, **kwargs):
        for name, param in (("query_gate", model.query_gate[-1].weight), ("candidate_gate", model.candidate_gate[-1].weight),
                            ("text_projection", model.text_projection[0].weight), ("semantic_mix", model.semantic_mix_logit)):
            gradients[name] = param.grad.detach().clone()
        return original_step(*args, **kwargs)

    monkeypatch.setattr(model, "score", score)
    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(visual, "straight_through_images", st)
    monkeypatch.setattr(repaired.common, "policy_advantage", advantage)
    monkeypatch.setattr(optimizer, "step", step)
    metrics = repaired.train_epoch(model, legacy_common.make_loader(records, config, True), bank, pixels,
                                   candidates, table, config, optimizer, scaler, "cpu", 1,
                                   {"compute_batch_size": 2, "profile_batches": 0})
    assert len(reward_groups) == 1 and len(reward_groups[0]) == 4
    assert metrics["optimizer_steps"] == 1 and metrics["q_gate"] > 0 and metrics["c_gate"] > 0
    assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in weights_seen)
    assert sum(float(w.grad.abs().sum()) for w in weights_seen) > 0
    assert sum(float(t.grad.abs().sum()) for t in selected_text_seen) > 0
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients.values())
    assert model.gradient_checkpointing and metrics["compute_batch_size"] == 2


def interrupted_legacy(tmp_path, dino, monkeypatch):
    config = {**tiny_config(), "epochs": 2}
    train = image_rows(tmp_path / "images", 4, "train")
    val = image_rows(tmp_path / "images", 3, "val", offset=100)
    pool = [dict(row, position=i) for i, row in enumerate(train)]
    source = tmp_path / "legacy"
    models = fixture_assets(tmp_path / "assets")
    cache, settings, classes = registered_texts(source / "shared", train + val, models)
    table = legacy.TextTable(cache, source / "shared/captions.csv", train + val, config)
    assets = bound_assets(source / "shared", models)
    model = tiny_model(dino, config)
    original = legacy.train_epoch

    def interrupt(*args, **kwargs):
        if args[-1] == 2:
            raise KeyboardInterrupt("Test interrupt after first saved epoch")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(legacy, "validate_config", lambda config: None)
        patch.setattr(legacy, "train_epoch", interrupt)
        with pytest.raises(KeyboardInterrupt):
            legacy.train_run(model, classes, train, val, pool, table, config, assets, source, "cpu")
    return model, config, train, val, pool, models, settings, classes, assets, source


def test_legacy_resume_keeps_caption_optimizer_bindings_and_exports(repaired, dino, tmp_path, monkeypatch):
    model, old_config, train, val, pool, models, settings, classes, assets, source = interrupted_legacy(tmp_path, dino, monkeypatch)
    originals = {name: visual.file_sha256(source / name) for name in ("last.pt", "best.pt", "protocol.json", "shared/captions.csv")}
    config = {**old_config, "training_engine": "memory_bounded_two_pass_v2"}
    output = tmp_path / "repaired"
    repaired.stage_previous_caption_cache(source, output / "shared", train + val, classes, models, settings)
    cache = torch.load(output / "shared/text_embeddings.pt", map_location="cpu", weights_only=True)
    table = repaired.TextTable(cache, output / "shared/captions.csv", train + val, config)
    monkeypatch.setattr(repaired, "validate_config", lambda config: None)
    best, history = repaired.train_run(model, classes, train, val, pool, table, config, assets, output, "cpu",
                                       execution={"compute_batch_size": 1, "profile_batches": 0}, resume_from=source)
    assert [row["epoch"] for row in history] == [1, 2]
    protocol = visual.read_json(output / "protocol.json")
    assert protocol["migration"]["completed_epochs"] == 1
    assert protocol["migration"]["caption_csv_sha256"] == assets["caption_csv_sha256"]
    assert protocol["migration"]["text_cache_sha256"] == assets["text_cache_sha256"]
    assert originals == {name: visual.file_sha256(source / name) for name in originals}
    with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["train_compute_batch_size"] == "" and saved[1]["train_compute_batch_size"] == "1"
    bank = repaired.common.refresh_bank(model, pool, config, "cpu")
    metrics, predictions = repaired.evaluate(model, legacy_common.make_loader(val, config), bank, pool, table, config, "cpu")
    factory = lambda repo, cfg: tiny_model(dino, cfg)
    archive = repaired.export_bundle(model, best, pool, classes, table, config, dino[0], output, val[0], metrics,
                                     predictions, models, None, model_factory=factory, caption_runner=fixture_caption)
    root = archive.with_suffix("")
    assert runtime.verify_bundle(root)["status"] == "ready"
    predictor = runtime.BundlePredictor(root, "cpu", factory, fixture_caption)
    result = predictor.predict(val[0]["image_path"], use_caption_cache=False)
    assert set(result["evidence"]["attention"]) == {"query_classifier", "candidate_selector"}
    assert "caption" in result["evidence"]["candidate"]
    assert result["evidence"]["query_caption"]["source"] == "fresh_florence"
    assert visual.read_json(root / "protocol.json")["migration"]["completed_epochs"] == 1
    assert visual.read_json(root / "smoke_test.json")["test_injected_services"]
    # Changing compute chunk is permitted without starting a different run.
    _, again = repaired.train_run(model, classes, train, val, pool, table, config, assets, output, "cpu",
                                  execution={"compute_batch_size": 2, "profile_batches": 0}, resume_from=source)
    assert len(again) == 2
    with pytest.raises(ValueError, match="Legacy classes/pool"):
        repaired.migrate_legacy_run(source, tmp_path / "wrong_pool", config, classes,
                                    {"train": train, "val": val, "pool": pool[::-1]}, assets)
    with pytest.raises(ValueError, match="logical batch"):
        repaired.migrate_legacy_run(source, tmp_path / "wrong_horizon", {**config, "epochs": 3}, classes,
                                    {"train": train, "val": val, "pool": pool}, assets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required for full_both GPU bank test")
def test_cuda_full_both_replay_and_gpu_pixels(repaired, dino, tmp_path):
    config = {**repair_config(), "batch_size": 4, "effective_batch_size": 4, "gumbel_tau": 0.7}
    model = tiny_model(dino, config).cuda()
    records = image_rows(tmp_path / "images", 4, "train")
    models = fixture_assets(tmp_path / "assets")
    cache, _, _ = registered_texts(tmp_path / "shared", records, models)
    table = repaired.TextTable(cache, tmp_path / "shared/captions.csv", records, config)
    pool = [dict(row, position=i) for i, row in enumerate(records)]
    options = repaired.common.execution_options(config, {"compute_batch_size": 2, "pixel_bank_device": "cuda",
                                                       "gpu_reserve_gib": 0, "profile_batches": 1})
    pixels = repaired.common.prepare_training_pixels(pool, config, "cuda", options)
    assert pixels.is_cuda and pixels.dtype == torch.float32
    bank = repaired.common.refresh_bank(model, pool, config, "cuda", amp=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=not torch.cuda.is_bf16_supported())
    metrics = repaired.train_epoch(model, legacy_common.make_loader(records, config, True), bank, pixels, pool,
                                   table, config, optimizer, scaler, "cuda", 1, options)
    assert metrics["optimizer_steps"] == 1 and metrics["q_gate"] > 0 and metrics["c_gate"] > 0
