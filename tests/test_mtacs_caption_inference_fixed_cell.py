"""Test the complete Kaggle cell without Florence downloads or a GPU."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/models", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / relative))
import export_mtacs_caption_inference_fixed_cell as builder


@pytest.fixture
def cell():
    return builder.build_cell()


@pytest.fixture
def payload(cell):
    tree = ast.parse(cell)
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "EMBEDDED_SOURCES" for t in node.targets))


def test_full_cell_is_complete_and_matches_exported_file(cell, payload):
    assert builder.TARGET.read_text(encoding="utf-8") == cell
    assert len(payload) == 5
    for source in payload.values():
        compile(source, "embedded_module", "exec")
    assert "vectors[~mask] = 0" not in payload["sun397_caption_web.py"]
    assert "vectors[~masks] = 0" not in payload["sun397_caption_web.py"]
    original = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    tree = ast.parse("".join(original["cells"][6]["source"]))
    original_payload = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "EMBEDDED_SOURCES" for t in node.targets))
    assert all(source == original_payload[name] for name, source in payload.items() if name != "sun397_caption_web.py")


@pytest.fixture
def service(payload):
    module = types.ModuleType("test_fixed_caption_service")
    module.__file__ = str(builder.TARGET)
    exec(compile(payload["sun397_caption_web.py"], "fixed_caption_worker", "exec"), module.__dict__)
    return module


def inference_encoder(texts):
    with torch.inference_mode():
        return torch.nn.functional.normalize(torch.arange(1, 513, dtype=torch.float32).repeat(len(texts), 1), dim=1)


def engine_fixture(service):
    # Exercise the real audit implementation with an actual inference tensor.
    engine = service.CaptionEngine.__new__(service.CaptionEngine)
    engine.settings = {"semantic_audit_threshold": 0.30}
    engine.terms = ["bedroom"]
    engine.labels = ["abbey", "bedroom"]
    engine.class_vectors = torch.nn.functional.normalize(torch.ones(2, 512), dim=1)
    engine.encode = inference_encoder
    return engine


def test_real_audit_masks_an_inference_tensor_without_inplace_update(service):
    engine = engine_fixture(service)
    assert torch.is_inference(engine.encode(["test"]))
    rows, vectors, mask = engine.audit(["dense neutral shapes covering textured flat surface",
                                      "bedroom with pale walls and wooden furniture"])
    assert mask.tolist() == [True, False]
    assert rows[1]["leakage_detected"]
    assert not torch.is_inference(vectors)
    assert vectors[1].count_nonzero() == 0 and vectors[0].count_nonzero() == 512
    vectors[0, 0] = 0  # Normal downstream tensor operations remain valid.


def test_real_worker_final_encode_is_also_safe(service, tmp_path, monkeypatch):
    from PIL import Image
    image = tmp_path / "query.png"
    Image.new("RGB", (16, 16), "gray").save(image)
    engine = engine_fixture(service)
    engine.generate = lambda images: ["dense neutral shapes covering textured flat surface"] * len(images)
    monkeypatch.setattr(service, "CaptionEngine", lambda payload: engine)
    payload = {"mode": "prepare", "classes": engine.labels,
               "settings": {"batch_size": 2, "max_new_tokens": 48},
               "asset_identity": {"florence_sha256": "fixture", "clip_sha256": "fixture"},
               "records": [{"image_id": "a/abbey/query.png", "split": "train", "image_path": str(image), "image_sha256": "fixture"}],
               "output": str(tmp_path / "shared/captions.csv"), "result": str(tmp_path / "result.pt")}
    service.worker(payload)
    result = torch.load(payload["result"], map_location="cpu", weights_only=True)
    assert result["valid"].tolist() == [True] and result["embeddings"].shape == (1, 512)


def stage_namespace(cell, tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "sun397_caption_web", raising=False)
    directory = tmp_path / "output/embedded_runtime_v2"
    directory.mkdir(parents=True)
    namespace = {"Path": Path, "sys": sys, "importlib": importlib,
                 "OUTPUT_DIR": directory.parent, "RUNTIME_DIR": directory}
    functions = [node for node in ast.parse(cell).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "staging_helpers", "exec"), namespace)
    return namespace


def test_staging_upgrades_only_the_known_bug_and_is_idempotent(cell, payload, tmp_path, monkeypatch):
    ns = stage_namespace(cell, tmp_path, monkeypatch)
    source = payload["sun397_caption_web.py"]
    previous = source
    for old, new in builder.PAIRS:
        previous = previous.replace(new, old)
    path = ns["RUNTIME_DIR"] / "sun397_caption_web.py"
    path.write_text(previous, encoding="utf-8")
    ns["_stage_embedded_runtime"](path.name, source)
    assert path.read_text(encoding="utf-8") == source
    assert (path.parent / "sun397_caption_web.before_inference_tensor_fix.py").read_text(encoding="utf-8") == previous
    ns["_stage_embedded_runtime"](path.name, source)


def test_staging_does_not_rewrite_a_registered_cache(cell, payload, tmp_path, monkeypatch):
    ns = stage_namespace(cell, tmp_path, monkeypatch)
    source = payload["sun397_caption_web.py"]
    previous = source
    for old, new in builder.PAIRS:
        previous = previous.replace(new, old)
    path = ns["RUNTIME_DIR"] / "sun397_caption_web.py"
    path.write_text(previous, encoding="utf-8")
    shared = ns["OUTPUT_DIR"] / "shared"
    shared.mkdir()
    (shared / "caption_protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="NEW OUTPUT_DIR"):
        ns["_stage_embedded_runtime"](path.name, source)
    assert path.read_text(encoding="utf-8") == previous
