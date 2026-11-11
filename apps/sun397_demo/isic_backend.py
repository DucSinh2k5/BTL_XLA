"""ISIC endpoints and lazy isolated workers. The API process never imports torch."""
from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageChops

from .inference import InferenceUnavailable, ModelWorker, Settings, PROJECT

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent / "isic"
SCHEMA = "isic2017_segmentation_web_v1"
ModelId = Literal["tacs", "mtacs"]


def configuration():
    base = Settings.from_environment()
    path = PROJECT / "configs/web.local.json"
    local = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    root = PROJECT / "artifacts/deployment/isic2017"
    caption_override = os.getenv("ISIC_MTACS_CAPTION_OVERRIDE", local.get("isic_mtacs_caption_override", ""))
    return Settings(bundles={
        "tacs": Path(os.getenv("ISIC_TACS_BUNDLE", local.get("isic_tacs_bundle", root / "tacs_visual"))).resolve(),
        "mtacs": Path(os.getenv("ISIC_MTACS_BUNDLE", local.get("isic_mtacs_bundle", root / "mtacs_full_both"))).resolve()},
        python=base.python, dependencies=base.dependencies,
        runtime=Path(os.getenv("ISIC_WEB_RUNTIME", PROJECT / ".runtime/isic_demo")).resolve(),
        device=os.getenv("ISIC_DEVICE", base.device), autoload=False, threads=base.threads,
        startup_timeout=base.startup_timeout, prediction_timeout=base.prediction_timeout,
        caption_weight_cache=base.caption_weight_cache,
        caption_override=Path(caption_override).resolve() if caption_override else None)


class SegmentationWorker(ModelWorker):
    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.state, self.message = "connected", "Model đã kết nối; sẽ nạp khi phân đoạn ảnh."

    def start(self):
        with self._lock:
            if self.state == "ready" or self._closed.is_set():
                return
            self.state, self.message = "loading", "Đang nạp model phân đoạn ISIC…"
            try:
                self.settings.runtime.mkdir(parents=True, exist_ok=True)
                logs = self.settings.runtime / "logs"
                logs.mkdir(exist_ok=True)
                self._log = (logs / f"{self.model}.log").open("a", encoding="utf-8", buffering=1)
                env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                    TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                env.pop("PYTHONPATH", None)
                if self.settings.dependencies:
                    env["PYTHONPATH"] = str(self.settings.dependencies)
                if self.settings.caption_weight_cache:
                    env["SUN397_CAPTION_WEIGHT_CACHE"] = str(self.settings.caption_weight_cache)
                command = [self.settings.python, "-u", str(Path(__file__).with_name("isic_worker.py")),
                    "--model", self.model, "--bundle", str(self.settings.bundles[self.model]),
                    "--runtime", str(self.settings.runtime), "--device", self.settings.device,
                    "--threads", str(self.settings.threads), "--parent-pid", str(os.getpid())]
                if self.model == "mtacs" and self.settings.caption_override:
                    command.extend(["--caption-override", str(self.settings.caption_override)])
                kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=self._log, text=True, encoding="utf-8", bufsize=1, env=env, **kwargs)
                threading.Thread(target=self._read_events, daemon=True).start()
                event = self._event(time.monotonic() + self.settings.startup_timeout)
                if event["event"] != "ready":
                    raise InferenceUnavailable("Worker chưa xác nhận model ISIC sẵn sàng.")
                self.details = event.get("details", {})
                self.state, self.message = "ready", "Model ISIC sẵn sàng phân đoạn."
            except Exception:
                LOG.exception("Cannot load ISIC %s", self.model)
                self.state, self.message = "error", "Không nạp được model ISIC. Kiểm tra bundle và log máy chủ."
                self.stop()


class SegmentationManager:
    def __init__(self, settings=None, gate=None):
        self.settings = settings or configuration()
        self.workers = {name: SegmentationWorker(name, self.settings) for name in ("tacs", "mtacs")}
        self.gate = gate or asyncio.Lock()

    def status(self, model):
        root = self.settings.bundles[model]
        empty = {"available": False, "can_predict": False, "code": "bundle_missing",
                 "message": "Chưa kết nối bộ model ISIC. Bạn vẫn có thể chọn và xem ảnh."}
        if not (root / "manifest.json").is_file():
            return empty
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            if manifest.get("schema") != SCHEMA or manifest.get("status") != "ready" or manifest.get("mode") != model:
                raise ValueError("Wrong or unverified ISIC bundle")
            if config.get("mode") != model or config.get("backbone") != "dinov3_vitl16" or config.get("image_size") != 448:
                raise ValueError("Wrong model architecture")
            required = ["runtime/isic_segmentation_web.py", "runtime/tacs_visual_web.py", "models/selector.pt",
                "models/segmentor.pt", "candidate_bank.pt", "candidate_pool.json", "ground_truth_index.json",
                manifest["backbone_file"]]
            if model == "mtacs":
                required += ["runtime/isic_caption_web.py", "runtime/isic_caption_core.py", "text_embeddings.pt",
                    "caption_lookup.json", "caption_settings.json", "caption_models/florence/config.json",
                    "caption_models/clip/config.json"]
            for name in required:
                relative = Path(name)
                target = (root / relative).resolve()
                if relative.is_absolute() or not target.is_relative_to(root) or not target.is_file():
                    raise ValueError("Incomplete or unsafe bundle")
        except (OSError, ValueError, KeyError, TypeError):
            return {**empty, "code": "bundle_invalid", "message": "Bộ model ISIC chưa đầy đủ hoặc chưa qua kiểm tra export."}
        worker = self.workers[model]
        state = worker.status()
        intervention = {}
        if model == "mtacs" and self.settings.caption_override:
            from .isic_caption_override import verify_override
            try:
                override = verify_override(self.settings.caption_override, root)
                intervention = {"caption_experiment": {
                    "method": override["method"], "policy": override["policy"],
                    "experiment": override["experiment"], "segmentation_checkpoint_retrained": False,
                    "masked_captions": override["report"]["new_masked"]}}
            except (OSError, ValueError, KeyError, TypeError):
                return {**empty, "code": "caption_override_invalid", "message": "Bản caption thử nghiệm chưa hợp lệ; kiểm tra CSV và CLIP cache."}
        return {**state, "can_predict": state["code"] != "loading", "backbone": config["backbone"],
                "image_size": config["image_size"], "context_top_k": 1, **intervention}

    async def predict(self, model, content):
        if not self.status(model)["can_predict"]:
            raise InferenceUnavailable(self.status(model)["message"])
        async with self.gate:
            # Only one ISIC worker stays resident. A model switch must not load
            # two more L/16 backbones alongside the existing SUN397 demo.
            for name, other in list(self.workers.items()):
                if name != model and other.process is not None:
                    await asyncio.to_thread(other.stop)
                    self.workers[name] = SegmentationWorker(name, self.settings)
            worker = self.workers[model]
            if worker._closed.is_set():
                worker = self.workers[model] = SegmentationWorker(model, self.settings)
            await asyncio.to_thread(worker.start)
            if not worker.status()["available"]:
                raise InferenceUnavailable(worker.message)
            return await asyncio.to_thread(worker.predict, content)

    async def close(self):
        for worker in self.workers.values():
            await asyncio.to_thread(worker.stop)


def architecture(model):
    text = model == "mtacs"
    return {"backbone": "DINOv3 ViT-L/16", "image_size": 448, "context_top_k": 1, "blocks": 24,
        "pool_fraction": .2, "text_enabled": text, "stages": [6, 12, 18, 24],
        "steps": [
            {"title": "Ảnh query", "detail": "Đặc trưng từ backbone frozen", "kind": "visual"},
            {"title": "Candidate pool", "detail": "20% train · một ảnh context", "kind": "visual"},
            {"title": "Selector học", "detail": "Ảnh + CLIP text" if text else "Utility từ đặc trưng ảnh", "kind": "text" if text else "visual"},
            {"title": "Context fusion", "detail": "Gated cross-attention · 24 block", "kind": "fusion"},
            {"title": "DPT nhẹ", "detail": "Mask phân đoạn nhị phân", "kind": "output"}]}


async def read_reference(file, dimensions):
    if file is None or not file.filename:
        return None
    try:
        data = await file.read(12 * 1024 * 1024 + 1)
        if len(data) > 12 * 1024 * 1024:
            raise ValueError("Mask vượt quá 12 MB.")
        with Image.open(BytesIO(data)) as source:
            if source.format != "PNG" or source.size != dimensions:
                raise ValueError("Mask cần là PNG và có cùng kích thước với ảnh đầu vào.")
            image = source.convert("L")
            histogram = image.histogram()
            if any(count for index, count in enumerate(histogram) if index not in {0, 1, 255}):
                raise ValueError("Mask cần là ảnh nhị phân đen trắng (0/1 hoặc 0/255).")
            return image.point(lambda value: 255 if value else 0)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    finally:
        await file.close()


def add_reference(result, reference):
    if reference is None:
        return result
    data_url = result["prediction"]["mask_url"]
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("Invalid prediction mask URL")
    with Image.open(BytesIO(base64.b64decode(data_url[len(prefix):], validate=True))) as source:
        prediction = source.convert("L").point(lambda value: 255 if value >= 128 else 0)
    if prediction.size != reference.size:
        raise ValueError("Prediction and reference mask dimensions differ")
    overlap = ImageChops.multiply(prediction, reference).histogram()[255]
    p, t = prediction.histogram()[255], reference.histogram()[255]
    handle = BytesIO()
    reference.save(handle, format="PNG")
    mask = {"status": "user_reference", "verified": False,
        "mask_url": prefix + base64.b64encode(handle.getvalue()).decode("ascii")}
    comparison = {"status": "available", "source": "user_reference", "dice": (2 * overlap + 1e-6) / (p + t + 1e-6),
                  "iou": (overlap + 1e-6) / (p + t - overlap + 1e-6)}
    result["reference_mask"], result["reference_comparison"] = mask, comparison
    if not result.get("ground_truth", {}).get("verified"):
        result["ground_truth"], result["comparison"] = mask, comparison
    return result


def create_router(image_metadata):
    router = APIRouter()

    @router.get("/isic", include_in_schema=False)
    @router.get("/isic/mtacs", include_in_schema=False)
    def page():
        return FileResponse(ROOT / "index.html")

    @router.get("/api/isic/status")
    def status(request: Request, model: ModelId = "tacs"):
        return {"model": model, "dataset": "ISIC2017", "task": "binary_segmentation",
            "architecture": architecture(model), "inference": request.app.state.segmentation.status(model)}

    @router.post("/api/isic/inspect")
    async def inspect(file: UploadFile = File(...)):
        try:
            metadata, _ = await image_metadata(file)
            return metadata
        finally:
            await file.close()

    @router.post("/api/isic/segment")
    async def segment(request: Request, file: UploadFile = File(...), model: ModelId = "tacs",
                      reference_mask: UploadFile | None = File(None)):
        try:
            metadata, content = await image_metadata(file)
            reference = await read_reference(reference_mask, (metadata["width"], metadata["height"]))
        finally:
            await file.close()
            if reference_mask is not None:
                await reference_mask.close()
        manager = request.app.state.segmentation
        try:
            # Reference masks never cross the worker boundary.
            result = await manager.predict(model, content)
            if result.get("model") != model or result.get("task") != "binary_segmentation" or not result.get("prediction"):
                raise ValueError("Worker returned an invalid segmentation result")
            return JSONResponse(content=add_reference(result, reference))
        except InferenceUnavailable as exc:
            return JSONResponse(status_code=503, content={"model": model, "code": manager.status(model)["code"],
                "message": str(exc), "prediction": None, "evidence": None})
        except Exception:
            LOG.exception("ISIC inference failed for %s", model)
            return JSONResponse(status_code=500, content={"model": model, "code": "inference_error",
                "message": "Phân đoạn chưa thành công. Kiểm tra log model trên máy chủ rồi thử lại.", "prediction": None})
    return router
