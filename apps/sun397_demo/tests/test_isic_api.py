"""ISIC HTTP contracts. Fixture outputs are never served by production routes."""
from io import BytesIO
import base64
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sun397_demo.app import app
from sun397_demo.inference import InferenceUnavailable, Settings
from sun397_demo.isic_backend import SegmentationManager


def png(color=0, size=(30, 20)):
    stream = BytesIO()
    Image.new("L", size, color).save(stream, format="PNG")
    return stream.getvalue()


def url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class StubSegmentation:
    def __init__(self):
        self.ready = False
        self.received = None
        self.verified = False

    def status(self, model):
        return {"available": self.ready, "can_predict": self.ready,
                "code": "ready" if self.ready else "bundle_missing", "message": "ISIC test fixture"}

    async def predict(self, model, content):
        self.received = content
        if not self.ready:
            raise InferenceUnavailable("ISIC test fixture unavailable")
        return {"model": model, "task": "binary_segmentation", "prediction": {"mask_url": url(png(255)),
            "overlay_url": url(png()), "foreground_fraction": 1.0}, "evidence": {"candidate": {"image_id": "ISIC_fixture"}},
            "ground_truth": {"verified": self.verified, "status": "verified_pixels" if self.verified else "unavailable"},
            "comparison": {"status": "available", "dice": .75, "iou": .6} if self.verified else {"status": "unavailable"}}


@pytest.fixture
def service(monkeypatch):
    stub = StubSegmentation()
    monkeypatch.setattr(app.state, "segmentation", stub)
    return stub


client = TestClient(app)


def test_both_isic_pages_and_static_assets():
    for path in ("/isic", "/isic/mtacs"):
        response = client.get(path)
        assert response.status_code == 200
        assert 'id="isic-page"' in response.text
        assert 'id="result-canvas"' in response.text
        assert 'id="reference-input"' in response.text
        assert 'src="/isic/assets/isic.js?v=' in response.text
        assert 'src="/assets/studio.js?v=' in response.text
        assert 'id="context-image"' in response.text
    assert client.get("/isic/assets/isic.js").status_code == 200
    assert client.get("/isic/assets/isic.css").status_code == 200
    assert 'href="/isic"' in client.get("/").text


@pytest.mark.parametrize("model", ["tacs", "mtacs"])
def test_architecture_distinguishes_full_both(service, model):
    data = client.get("/api/isic/status", params={"model": model}).json()
    assert data["model"] == model and data["dataset"] == "ISIC2017"
    assert data["architecture"]["text_enabled"] is (model == "mtacs")
    assert data["architecture"]["blocks"] == 24
    assert data["architecture"]["context_top_k"] == 1
    assert data["inference"]["can_predict"] is False


def test_missing_checkpoint_never_returns_mock_prediction(service):
    response = client.post("/api/isic/segment", files={"file": ("ISIC.png", png(), "image/png")})
    assert response.status_code == 503
    assert response.json()["prediction"] is None
    assert response.json()["code"] == "bundle_missing"


def test_inspect_and_reject_invalid_image_without_inference(service):
    response = client.post("/api/isic/inspect", files={"file": ("ISIC.png", png(), "image/png")})
    assert response.json()["width"] == 30 and response.json()["stored"] is False
    assert service.received is None
    response = client.post("/api/isic/inspect", files={"file": ("bad.png", b"text", "image/png")})
    assert response.status_code == 415
    assert client.get("/api/isic/status?model=other").status_code == 422


@pytest.mark.parametrize("model", ["tacs", "mtacs"])
def test_reference_mask_is_post_inference_only_and_labelled(service, model):
    service.ready = True
    image = png(80)
    response = client.post("/api/isic/segment", params={"model": model}, files={
        "file": ("ISIC.png", image, "image/png"), "reference_mask": ("ref.png", png(255), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert service.received == image
    assert result["model"] == model
    assert result["ground_truth"]["verified"] is False
    assert result["ground_truth"]["status"] == "user_reference"
    assert result["comparison"]["dice"] == 1
    assert result["comparison"]["iou"] == 1


def test_verified_dataset_mask_not_overwritten_by_user_reference(service):
    service.ready = service.verified = True
    result = client.post("/api/isic/segment", files={"file": ("query.png", png(), "image/png"),
        "reference_mask": ("ref.png", png(0), "image/png")}).json()
    assert result["ground_truth"]["verified"] is True
    assert result["comparison"]["dice"] == .75
    assert result["reference_mask"]["verified"] is False
    assert result["reference_comparison"]["dice"] < .01


@pytest.mark.parametrize("mask", [png(127), png(255, (2, 2)), b"not PNG"])
def test_invalid_reference_stops_before_model(service, mask):
    service.ready = True
    response = client.post("/api/isic/segment", files={"file": ("query.png", png(), "image/png"),
        "reference_mask": ("ref.png", mask, "image/png")})
    assert response.status_code == 422
    assert service.received is None


def test_manager_missing_and_unverified_bundles(tmp_path):
    settings = Settings(bundles={"tacs": tmp_path / "tacs", "mtacs": tmp_path / "mtacs"},
                        python="unused", dependencies=None, runtime=tmp_path / "runtime", autoload=False)
    manager = SegmentationManager(settings)
    assert manager.status("tacs")["code"] == "bundle_missing"
    (tmp_path / "tacs").mkdir()
    (tmp_path / "tacs/manifest.json").write_text(json.dumps({"schema": "isic2017_segmentation_web_v1",
        "status": "fixture_unverified", "mode": "tacs"}))
    (tmp_path / "tacs/config.json").write_text("{}")
    assert manager.status("tacs")["code"] == "bundle_invalid"
    assert all(worker.process is None for worker in manager.workers.values())


@pytest.mark.parametrize("model", ["tacs", "mtacs"])
def test_real_worker_ipc_serializes_masks_and_rejects_wrong_model(tmp_path, model):
    import os
    from pathlib import Path
    import subprocess
    import sys
    fixture_packages = tmp_path / "packages"
    fixture_packages.mkdir()
    # No real model is loaded. This subprocess fixture is not a web bundle.
    (fixture_packages / "torch.py").write_text(
        "def set_num_threads(value): pass\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n", encoding="utf-8")
    bundle = tmp_path / "fixture_bundle"
    runtime = bundle / "runtime"
    runtime.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps({"mode": model}))
    (runtime / "isic_caption_web.py").write_text(
        "class BundledCaptionEngine:\n"
        "    def __init__(self, *args, **kwargs): pass\n", encoding="utf-8")
    (runtime / "isic_segmentation_web.py").write_text(
        "import json,base64,io\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "import numpy as np\nfrom PIL import Image\n"
        "def png_data_url(image):\n"
        "    stream=io.BytesIO(); image.save(stream,format='PNG')\n"
        "    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()\n"
        "io_utils=SimpleNamespace(png_data_url=png_data_url)\n"
        "class Predictor:\n"
        "    def __init__(self,root,device,caption_engine=None):\n"
        "        self.config=json.loads((Path(root)/'config.json').read_text())\n"
        "        self.config.update(backbone='fixture_only',image_size=30)\n"
        "        self.pool=[1,2]; self.model=SimpleNamespace(requires_grad_=lambda value:None)\n"
        "    def predict(self,path):\n"
        "        return dict(model=self.config['mode'],task='binary_segmentation',logits=None,\n"
        "            probabilities=np.zeros((20,30),dtype=np.float32),\n"
        "            prediction=dict(mask_url=png_data_url(Image.new('L',(30,20),0))))\n",
        encoding="utf-8")
    request_root = tmp_path / "service" / "requests"
    request_root.mkdir(parents=True)
    image, output = request_root / "upload.image", request_root / "result.json"
    image.write_bytes(png())
    request = {"command": "predict", "model": model, "image": str(image), "output": str(output)}
    commands = [json.dumps({**request, "model": "wrong"}), json.dumps(request), '{"command":"stop"}']
    script = Path(__file__).resolve().parents[1] / "isic_worker.py"
    env = dict(os.environ, PYTHONPATH=str(fixture_packages), PYTHONUTF8="1")
    completed = subprocess.run([sys.executable, str(script), "--model", model, "--bundle", str(bundle),
        "--runtime", str(tmp_path / "service")], input="\n".join(commands) + "\n",
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line.removeprefix("SUN397_EVENT ")) for line in completed.stdout.splitlines()
              if line.startswith("SUN397_EVENT ")]
    assert [event["event"] for event in events] == (
        ["progress", "progress", "ready", "prediction_error", "result"] if model == "mtacs"
        else ["progress", "ready", "prediction_error", "result"])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["model"] == model
    assert "logits" not in result and "probabilities" not in result
    assert result["prediction"]["probability_url"].startswith("data:image/png;base64,")
    assert result["timing"]["inference_seconds"] >= 0


