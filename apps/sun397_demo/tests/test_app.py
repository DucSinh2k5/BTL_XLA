import hashlib
from io import BytesIO
import pytest

from fastapi.testclient import TestClient
from PIL import Image

from sun397_demo.app import app
from sun397_demo.inference import InferenceUnavailable
from sun397_demo.labels import LabelCatalog


client = TestClient(app)


class StubManager:
    def __init__(self):
        self.ready = False
        self.started = self.closed = False
        self.received = None

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    def status(self, model):
        return {"available": self.ready, "code": "ready" if self.ready else "checkpoint_not_connected",
                "message": "Test model"}

    async def predict(self, model, content):
        if not self.ready:
            raise InferenceUnavailable("Test model unavailable")
        self.received = content
        return {"model": model, "prediction": {"class_id": 0, "label": "a/abbey", "confidence": 0.6},
                "evidence": {"query_caption": {"text": "caption from test service", "valid": False, "status": "masked"},
                             "candidate": {"image_id": "a/abbey/example.jpg", "position": 0, "caption_valid": True},
                             "attention": {"query_classifier": {"status": "ready"},
                                           "candidate_selector": {"status": "ready"}}}}


@pytest.fixture(autouse=True)
def manager(monkeypatch):
    service = StubManager()
    monkeypatch.setattr(app.state, "inference", service)
    catalog = LabelCatalog({})
    catalog.classes = ["a/abbey", "b/bedroom"]
    catalog.pools = {name: {"a/abbey/example.jpg": {"class_id": 0, "label": "a/abbey",
                     "image_id": "a/abbey/example.jpg", "split": "train", "position": 0}}
                     for name in ("tacs", "mtacs")}
    catalog.loaded = True
    monkeypatch.setattr(app.state, "labels", catalog)
    return service


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (36, 24), "#276c65").save(buffer, format="PNG")
    return buffer.getvalue()


def test_original_tacs_status_is_visual_only():
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tacs"
    assert body["architecture"]["dataset"] == "SUN397"
    assert body["architecture"]["context_top_k"] == 1
    assert body["architecture"]["variant"] == "tacs_paper_visual_only"
    assert "DINOv3 ViT-L/16" in body["architecture"]["chips"]
    assert "384 × 384" in body["architecture"]["chips"]
    assert all(node["kind"] != "text" for lane in body["architecture"]["lanes"] for node in lane["nodes"])
    assert body["inference"]["available"] is False


def test_mtacs_status_has_query_and_candidate_text():
    response = client.get("/api/status?model=mtacs")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mtacs"
    assert body["architecture"]["variant"] == "mtacs_dual_context_text"
    assert {node["id"] for lane in body["architecture"]["lanes"] for node in lane["nodes"]} >= {
        "query-caption", "pool-captions", "semantic-score", "text-fusion"
    }
    assert body["inference"]["available"] is False


def test_home_and_assets_are_served():
    page = client.get("/")
    assert page.status_code == 200
    assert 'href="/mtacs"' in page.text
    assert 'href="/"' in page.text
    assert client.get("/mtacs").status_code == 200
    assert 'id="query-caption-text"' in page.text
    assert 'id="query-evidence-image"' in page.text
    assert 'id="query-attention-image"' in page.text
    assert 'id="candidate-image"' in page.text
    assert 'id="candidate-attention-image"' in page.text
    assert 'id="candidate-caption-text"' in page.text
    assert 'id="candidate-class-label"' in page.text
    assert 'id="result-ground-truth"' in page.text
    assert 'id="prediction-confidence"' in page.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/page.js").status_code == 200
    assert 'id="demo-page"' in page.text
    assert page.text.index('/assets/page.js') < page.text.index('/assets/app.js')
    assert client.get("/assets/sample-library.png").status_code == 200


def test_inspect_valid_png_does_not_store_or_predict():
    content = png_bytes()
    response = client.post("/api/images/inspect", files={"file": ("texture.png", content, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert (body["width"], body["height"]) == (36, 24)
    assert body["sha256"] == hashlib.sha256(content).hexdigest()
    assert body["stored"] is False
    assert "prediction" not in body


def test_inspect_rejects_non_image():
    response = client.post("/api/images/inspect", files={"file": ("fake.jpg", b"not a photo", "image/jpeg")})
    assert response.status_code == 415


def test_inspect_rejects_large_file(monkeypatch):
    monkeypatch.setattr("sun397_demo.app.MAX_UPLOAD_BYTES", 10)
    response = client.post("/api/images/inspect", files={"file": ("texture.png", png_bytes(), "image/png")})
    assert response.status_code == 413


def test_mtacs_classification_is_explicitly_unavailable():
    response = client.post("/api/classify?model=mtacs", files={"file": ("texture.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    body = response.json()
    assert body["model"] == "mtacs"
    assert body["prediction"] is None
    assert body["evidence"]["query_caption"] == {
        "text": None, "status": "caption_model_unavailable"
    }
    assert body["evidence"]["attention"] == {
        "query_classifier": {"image_url": None, "status": "checkpoint_not_connected"},
        "candidate_selector": {"image_url": None, "status": "checkpoint_not_connected"},
    }
    assert body["evidence"]["candidate"] == {
        "image_url": None,
        "image_id": None,
        "caption": None,
        "status": "candidate_pool_unavailable",
    }


def test_original_tacs_does_not_request_captions():
    response = client.post("/api/classify?model=tacs", files={"file": ("texture.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    body = response.json()
    assert body["model"] == "tacs"
    assert body["evidence"]["query_caption"]["status"] == "not_used"
    assert body["evidence"]["candidate"]["image_url"] is None
    assert body["evidence"]["attention"]["query_classifier"]["image_url"] is None
    assert body["evidence"]["attention"]["candidate_selector"]["image_url"] is None


def test_unknown_model_is_rejected():
    assert client.get("/api/status?model=unknown").status_code == 422


def test_classification_rejects_invalid_image_before_reporting_model_status():
    response = client.post("/api/classify", files={"file": ("fake.jpg", b"not a photo", "image/jpeg")})
    assert response.status_code == 415


def test_startup_preloads_and_shutdown_closes_workers(manager):
    with TestClient(app) as running:
        assert manager.started
        assert running.get("/healthz").json() == {"ok": True}
    assert manager.closed


@pytest.mark.parametrize("model", ["tacs", "mtacs"])
def test_loaded_model_receives_original_bytes_and_returns_evidence(manager, model):
    manager.ready = True
    content = png_bytes()
    response = client.post(f"/api/classify?model={model}", files={"file": ("query.png", content, "image/png")})
    assert response.status_code == 200
    assert manager.received == content
    assert response.json()["model"] == model
    assert response.json()["prediction"]["label"] == "a/abbey"
    assert response.json()["evidence"]["query_caption"]["status"] == "masked"
    assert response.json()["evidence"]["attention"]["candidate_selector"]["status"] == "ready"
    assert response.json()["evidence"]["candidate"]["label"] == "a/abbey"
    assert response.json()["ground_truth"]["class_id"] is None
    assert response.json()["comparison"]["status"] == "unavailable"


def test_readiness_checks_both_models(manager):
    assert client.get("/readyz").status_code == 503
    manager.ready = True
    assert client.get("/readyz").json()["ready"] is True
    assert client.get("/api/status?model=mtacs").json()["inference"]["available"] is True


def test_inference_failure_never_returns_a_prediction(manager, monkeypatch):
    async def fail(model, content):
        raise RuntimeError("private implementation failure")
    monkeypatch.setattr(manager, "predict", fail)
    response = client.post("/api/classify", files={"file": ("query.png", png_bytes(), "image/png")})
    assert response.status_code == 500
    assert response.json()["prediction"] is None
    assert "private implementation" not in response.text


@pytest.mark.parametrize("model", ["tacs", "mtacs"])
def test_verified_ground_truth_and_correctness_do_not_enter_model(manager, model):
    manager.ready = True
    content = png_bytes()
    app.state.labels.by_hash[hashlib.sha256(content).hexdigest()] = [
        {"image_id": "b/bedroom/query.jpg", "class_id": 1, "label": "b/bedroom", "split": "test"}]
    inspected = client.post("/api/images/inspect", files={"file": ("renamed.png", content, "image/png")}).json()
    assert inspected["ground_truth"]["verified"] is True
    assert inspected["ground_truth"]["class_id"] == 1
    response = client.post(f"/api/classify?model={model}", files={"file": ("renamed.png", content, "image/png")})
    assert response.status_code == 200
    assert response.json()["comparison"]["status"] == "incorrect"
    assert manager.received == content


def test_filename_is_never_used_to_invent_ground_truth(manager):
    manager.ready = True
    result = client.post("/api/classify", files={"file": ("a/abbey/sun_known.png", png_bytes(), "image/png")}).json()
    assert result["ground_truth"]["label"] is None
    assert result["comparison"]["is_match"] is None


def test_manual_reference_is_explicitly_unverified(manager):
    manager.ready = True
    result = client.post("/api/classify", data={"reference_class_id": "0"},
                         files={"file": ("external.png", png_bytes(), "image/png")}).json()
    assert result["ground_truth"]["source"] == "user_provided"
    assert result["ground_truth"]["verified"] is False
    assert result["comparison"]["status"] == "matches_reference"


def test_conflicting_manual_reference_cannot_override_dataset_ground_truth(manager):
    manager.ready = True
    content = png_bytes()
    app.state.labels.by_hash[hashlib.sha256(content).hexdigest()] = [
        {"image_id": "b/bedroom/query.jpg", "class_id": 1, "label": "b/bedroom", "split": "test"}]
    response = client.post("/api/classify", data={"reference_class_id": "0"},
                           files={"file": ("query.png", content, "image/png")})
    assert response.status_code == 422
    assert manager.received is None


def test_class_list_endpoint_and_invalid_reference(manager):
    assert client.get("/api/classes").json()["classes"][0] == {"class_id": 0, "label": "a/abbey"}
    response = client.post("/api/classify", data={"reference_class_id": "396"},
                           files={"file": ("query.png", png_bytes(), "image/png")})
    assert response.status_code == 422
