"""Exercise both snapshot exports with real tiny DINOv3 and no retraining."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/models", "src/training", "src/captioning", "src/deployment", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / relative))
import tacs_visual_web as original_visual
import train_sun397_tacs_web as original_common
import train_sun397_mtacs_web as original_mtacs_training
import sun397_export_only as exporter
import build_sun397_export_only_notebooks as builder
from test_sun397_tacs_web_export import dino_class, make_rows, tiny_config as visual_config, tiny_model as visual_model
from test_sun397_mtacs_web_export import (
    dino, image_rows, fixture_assets, fixture_caption, fixture_texts,
    tiny_config as text_config, tiny_model as text_model,
)


def fallback_sources(variant):
    names = list(builder.SOURCE_PATHS) if variant == "mtacs" else ["tacs_visual_web.py", "train_sun397_tacs_web.py"]
    return {name: builder.SOURCE_PATHS[name].read_text(encoding="utf-8") for name in names}


def allow_runtime_imports(monkeypatch):
    for name in builder.SOURCE_PATHS:
        monkeypatch.delitem(sys.modules, Path(name).stem, raising=False)


def source_fingerprints(run):
    return {path.relative_to(run).as_posix(): exporter.sha(path) for path in run.rglob("*") if path.is_file()}


@pytest.fixture
def visual_run(tmp_path, dino_class, monkeypatch):
    config = visual_config()
    classes = ["a/abbey", "b/bedroom", "c/courtyard"]
    images = tmp_path / "images"
    train, val, test = make_rows(images, 6, "train"), make_rows(images, 3, "val", 100), make_rows(images, 3, "test", 200)
    pool = [dict(row, position=i) for i, row in enumerate(train[:4])]
    run = tmp_path / "saved_visual_run"
    assets = {"dino_source_sha256": original_common.source_tree_identity(dino_class[1]), "pretrained_weights_sha256": "not-needed-for-export"}
    model = visual_model(dino_class, config)
    with monkeypatch.context() as patch:
        patch.setattr(original_common, "validate_config", lambda config: None)
        original_common.train_run(model, classes, train, val, pool, config, assets, run, "cpu")
    original_common.csv_write(run / "test_split.csv", test)
    (run / "last.pt").unlink()
    (run / "training_complete.json").unlink()
    return {"run": run, "images": images, "repo": dino_class[1], "config": config,
            "train": train, "val": val, "pool": pool, "native_factory": lambda cfg: visual_model(dino_class, cfg)}


@pytest.fixture
def multimodal_run(tmp_path, dino, monkeypatch):
    config = text_config()
    classes = ["a/abbey", "b/bedroom", "c/courtyard"]
    images = tmp_path / "images"
    train, val, test = image_rows(images, 6, "train"), image_rows(images, 3, "val", 100), image_rows(images, 3, "test", 200)
    pool = [dict(row, position=i) for i, row in enumerate(train[:4])]
    run = tmp_path / "saved_text_run"
    models = fixture_assets(tmp_path / "frozen_assets")
    cache = fixture_texts(run / "shared", train + val + test, models)
    table = original_mtacs_training.TextTable(cache, run / "shared/captions.csv", train + val + test, config)
    assets = {"dino_source_sha256": original_common.source_tree_identity(dino[0]), "caption_assets": models,
              "caption_csv_sha256": exporter.sha(run / "shared/captions.csv"),
              "text_cache_sha256": exporter.sha(run / "shared/text_embeddings.pt"),
              "caption_settings_sha256": exporter.sha(run / "shared/caption_settings.json")}
    model = text_model(dino, config)
    with monkeypatch.context() as patch:
        patch.setattr(original_mtacs_training, "validate_config", lambda config: None)
        original_mtacs_training.train_run(model, classes, train, val, pool, table, config, assets, run, "cpu")
    original_common.csv_write(run / "test_split.csv", test)
    (run / "last.pt").unlink()
    (run / "training_complete.json").unlink()
    # Preserve the original code as in the user's saved Kaggle Output.
    folder = run / "embedded_runtime"
    folder.mkdir()
    for name, source in fallback_sources("mtacs").items():
        (folder / name).write_text(source, encoding="utf-8", newline="\n")
    return {"run": run, "images": images, "repo": dino[0], "models": models,
            "pool": pool, "native_factory": lambda cfg: text_model(dino, cfg)}


def prepare_case(case, variant, output, monkeypatch, full_test=False):
    allow_runtime_imports(monkeypatch)
    extra = {}
    if variant == "mtacs" and "models" in case:
        extra = {"florence": case["models"]["florence"], "clip": case["models"]["clip"], "caption_runner": fixture_caption}
    context = exporter.prepare(case["run"], case["images"], case["repo"], output, variant,
                               fallback_sources(variant), device="cpu", feature_batch_size=2, workers=0,
                               run_full_test=full_test, model_factory=lambda repo, cfg: case["native_factory"](cfg), **extra)
    for key in ("train_sun397_tacs_web", "train_sun397_mtacs_web"):
        if key in context["modules"]:
            monkeypatch.setattr(context["modules"][key], "train_run", lambda *a, **kw: pytest.fail("Export must never train"))
    if variant == "mtacs":
        monkeypatch.setattr(context["modules"]["train_sun397_mtacs_web"], "prepare_captions",
                            lambda *a, **kw: pytest.fail("Export must not regenerate the full caption CSV"))
    return context


def test_notebooks_are_standalone_no_training_and_match_generated_files():
    for variant, path in builder.TARGETS.items():
        notebook = builder.build_notebook(variant)
        assert json.loads(path.read_text(encoding="utf-8")) == notebook
        assert notebook["metadata"]["tacs"]["purpose"] == "export_only"
        assert notebook["metadata"]["tacs"]["training_called"] is False
        assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, "cell", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
            # Do not search inside embedded source strings: inspect actual calls.
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"train_run", "train_epoch", "prepare_captions", "build_image_bank"}
        config = "".join(notebook["cells"][4]["source"])
        assert "RUN_FULL_TEST = False" in config
        assert "DINOV3_WEIGHTS" not in config


def test_visual_best_only_export_no_last_or_completion_and_weights_unchanged(visual_run, tmp_path, monkeypatch):
    before = source_fingerprints(visual_run["run"])
    context = prepare_case(visual_run, "tacs", tmp_path / "export", monkeypatch)
    assert context["completion"].startswith("not_recorded")
    exporter.build(context)
    # Expensive pool encoding is reusable if verification or ZIP publication is retried.
    original_model_sha = exporter.sha(context["stage"] / "model.pt")
    monkeypatch.setattr(context["modules"]["train_sun397_tacs_web"], "refresh_bank",
                        lambda *a, **kw: pytest.fail("Completed bank must not be encoded twice"))
    exporter.build(context)
    assert exporter.sha(context["stage"] / "model.pt") == original_model_sha
    archive = exporter.verify_and_publish(context)
    root = archive.with_suffix("")
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    original = torch.load(visual_run["run"] / "best.pt", map_location="cpu", weights_only=True)
    for key, value in original["model_state_dict"].items():
        torch.testing.assert_close(state["model_state_dict"][key], value, atol=0, rtol=0)
    assert not (root / "training_complete.json").exists()
    assert not (visual_run["run"] / "last.pt").exists()
    metrics = exporter.json_read(root / "metrics.json")
    assert metrics["test"] is None and metrics["test_status"] == "not_run_export_only"
    assert not (root / "test_predictions.csv").exists()
    assert exporter.json_read(root / "export_provenance.json")["training_called"] is False
    context["runtime"].verify_bundle(root)
    prediction = exporter.json_read(root / "examples/prediction.json")
    assert set(prediction["evidence"]["attention"]) == {"query_classifier", "candidate_selector"}
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
    assert source_fingerprints(visual_run["run"]) == before
    # Idempotent verified export should reuse exactly the same archive bytes.
    previous = exporter.sha(archive)
    exporter.build(context)
    assert exporter.verify_and_publish(context) == archive
    assert exporter.sha(archive) == previous


def test_full_both_export_preserves_original_cache_models_and_candidate_text(multimodal_run, tmp_path, monkeypatch):
    before = source_fingerprints(multimodal_run["run"])
    context = prepare_case(multimodal_run, "mtacs", tmp_path / "export", monkeypatch)
    worker = context["modules"]["sun397_caption_web"]
    fixed_source = Path(worker.__file__).read_text(encoding="utf-8")
    assert "vectors[~mask] = 0" not in fixed_source and "vectors[~masks] = 0" not in fixed_source
    assert context["source_provenance"]["sun397_caption_web.py"]["caption_inference_tensor_fix"]
    exporter.build(context)
    archive = exporter.verify_and_publish(context)
    root = archive.with_suffix("")
    bank = torch.load(root / "candidate_text.pt", map_location="cpu", weights_only=True)
    original = torch.load(multimodal_run["run"] / "shared/text_embeddings.pt", map_location="cpu", weights_only=True)
    positions = [original["image_ids"].index(row["image_id"]) for row in multimodal_run["pool"]]
    torch.testing.assert_close(bank["embeddings"], original["embeddings"][positions], atol=0, rtol=0)
    assert torch.equal(bank["valid"], original["valid"][positions])
    for name in ("captions.csv", "text_embeddings.pt", "caption_protocol.json", "caption_settings.json"):
        assert exporter.sha(root / name) == exporter.sha(multimodal_run["run"] / "shared" / name)
    assert (root / "models/florence/config.json").is_file() and (root / "models/clip/config.json").is_file()
    assert exporter.json_read(root / "metrics.json")["test"] is None
    result = exporter.json_read(root / "examples/prediction.json")
    assert result["evidence"]["query_caption"]["source"] == "fresh_florence"
    assert "caption" in result["evidence"]["candidate"]
    smoke = exporter.json_read(root / "smoke_test.json")
    assert smoke["test_injected_services"] and not smoke["fresh_florence_clip_integration"]
    assert source_fingerprints(multimodal_run["run"]) == before
    assert not (root / "last.pt").exists() and not (root / "training_complete.json").exists()


def test_optional_test_evaluation_is_explicit_and_never_training(visual_run, tmp_path, monkeypatch):
    context = prepare_case(visual_run, "tacs", tmp_path / "evaluated", monkeypatch, full_test=True)
    exporter.build(context)
    metrics = exporter.json_read(context["stage"] / "metrics.json")
    assert metrics["test_status"] == "evaluated_by_export" and metrics["test"]["images"] == 3
    assert (context["stage"] / "test_predictions.csv").is_file()


def test_saved_pool_order_tampering_is_rejected(visual_run, tmp_path, monkeypatch):
    path = visual_run["run"] / "candidate_pool.csv"
    rows = exporter.csv_read(path)
    original_common.csv_write(path, rows[::-1])
    with pytest.raises(ValueError, match="pool/order"):
        prepare_case(visual_run, "tacs", tmp_path / "export", monkeypatch)


def test_wrong_architecture_or_missing_sidecars_are_not_guessed(visual_run, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="requested SUN397"):
        prepare_case(visual_run, "mtacs", tmp_path / "wrong_variant", monkeypatch)
    (visual_run["run"] / "candidate_pool.csv").unlink()
    with pytest.raises(FileNotFoundError, match="original run folder"):
        prepare_case(visual_run, "tacs", tmp_path / "missing", monkeypatch)


def test_mismatched_marker_is_not_overwritten_or_bypassed(visual_run, tmp_path, monkeypatch):
    path = visual_run["run"] / "training_complete.json"
    exporter.json_write(path, {"identity": "wrong", "best_sha256": "wrong", "best_epoch": 1})
    with pytest.raises(ValueError, match="completion marker"):
        prepare_case(visual_run, "tacs", tmp_path / "export", monkeypatch)
    assert exporter.json_read(path)["identity"] == "wrong"


def test_caption_cache_change_or_changed_text_weights_is_rejected(multimodal_run, tmp_path, monkeypatch):
    file = multimodal_run["run"] / "shared/captions.csv"
    original = file.read_bytes()
    file.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="caption cache binding"):
        prepare_case(multimodal_run, "mtacs", tmp_path / "bad_cache", monkeypatch)
    file.write_bytes(original)
    (Path(multimodal_run["models"]["clip"]) / "fixture.txt").write_text("different weights fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="text model weights"):
        prepare_case(multimodal_run, "mtacs", tmp_path / "bad_models", monkeypatch)


def test_staged_artifact_corruption_is_not_silently_reused(visual_run, tmp_path, monkeypatch):
    context = prepare_case(visual_run, "tacs", tmp_path / "export", monkeypatch)
    exporter.build(context)
    (context["stage"] / "candidate_features.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Staged export artifact changed"):
        exporter.build(context)
