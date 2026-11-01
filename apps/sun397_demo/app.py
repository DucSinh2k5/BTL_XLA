"""SUN397 demo backed by verified, preloaded TACS / M-TACS bundles."""

from __future__ import annotations

import hashlib
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .inference import InferenceManager, InferenceUnavailable
from .labels import LabelCatalog


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 36_000_000
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}

ARCHITECTURE = {
    "dataset": "SUN397",
    "task": "Phân loại cảnh",
    "variant": "mtacs_dual_context_text",
    "classes": 397,
    "context_top_k": 1,
    "chips": ["SUN397 · 397 lớp", "DINOv3 ViT-L/16", "384 × 384", "CLIP ViT-B/32", "Context K = 1"],
    "lanes": [
        {
            "id": "query",
            "title": "Query",
            "nodes": [
                {"id": "query-image", "label": "Ảnh query", "detail": "Ảnh được chọn", "kind": "image"},
                {"id": "query-caption", "label": "Caption query", "detail": "Florence-2", "kind": "text"},
                {"id": "query-text", "label": "Vector văn bản", "detail": "CLIP ViT-B/32", "kind": "text"},
            ],
        },
        {
            "id": "pool",
            "title": "Candidate pool",
            "nodes": [
                {"id": "pool-images", "label": "Ảnh candidate", "detail": "20% ảnh train SUN397", "kind": "image"},
                {"id": "pool-captions", "label": "Caption candidate", "detail": "Đã kiểm tra leak", "kind": "text"},
                {"id": "pool-text", "label": "Vector candidate", "detail": "CLIP ViT-B/32", "kind": "text"},
            ],
        },
        {
            "id": "selector",
            "title": "Chọn context",
            "nodes": [
                {"id": "visual-score", "label": "Điểm thị giác", "detail": "DINOv3 ViT-L/16", "kind": "image"},
                {"id": "semantic-score", "label": "Điểm ngữ nghĩa", "detail": "Query / candidate text", "kind": "text"},
                {"id": "selected-context", "label": "1 context được chọn", "detail": "Ảnh + caption tương ứng", "kind": "fusion"},
            ],
        },
        {
            "id": "classifier",
            "title": "Phân loại",
            "nodes": [
                {"id": "visual-fusion", "label": "Fusion ảnh", "detail": "Query + context", "kind": "image"},
                {"id": "text-fusion", "label": "Gated text fusion", "detail": "Query + context captions", "kind": "text"},
                {"id": "prediction", "label": "Nhãn SUN397", "detail": "397 lớp cảnh", "kind": "fusion"},
            ],
        },
    ],
}

TACS_ARCHITECTURE = {
    "dataset": "SUN397",
    "task": "Phân loại cảnh",
    "variant": "tacs_paper_visual_only",
    "classes": 397,
    "context_top_k": 1,
    "chips": ["SUN397 · 397 lớp", "DINOv3 ViT-L/16", "384 × 384", "Chỉ ảnh", "Context K = 1"],
    "lanes": [
        {
            "id": "query",
            "title": "Query",
            "nodes": [
                {"id": "query-image", "label": "Ảnh query", "detail": "Ảnh được chọn", "kind": "image"},
                {"id": "query-features", "label": "Đặc trưng query", "detail": "Selector ViT-L/16", "kind": "image"},
            ],
        },
        {
            "id": "pool",
            "title": "Candidate pool",
            "nodes": [
                {"id": "pool-images", "label": "Ảnh candidate", "detail": "20% ảnh train SUN397", "kind": "image"},
                {"id": "pool-features", "label": "Đặc trưng candidate", "detail": "Selector ViT-L/16", "kind": "image"},
            ],
        },
        {
            "id": "selector",
            "title": "Chọn context",
            "nodes": [
                {"id": "utility-score", "label": "Điểm hữu ích", "detail": "Tích vô hướng đặc trưng ảnh", "kind": "image"},
                {"id": "selected-context", "label": "1 ảnh được chọn", "detail": "Argmax trên candidate pool", "kind": "fusion"},
            ],
        },
        {
            "id": "classifier",
            "title": "Phân loại",
            "nodes": [
                {"id": "paired-images", "label": "Cặp ảnh đầu vào", "detail": "Query + context được chọn", "kind": "image"},
                {"id": "patch-concat", "label": "Ghép patch embeddings", "detail": "Downstream ViT-L/16", "kind": "fusion"},
                {"id": "prediction", "label": "Nhãn SUN397", "detail": "397 lớp cảnh", "kind": "fusion"},
            ],
        },
    ],
}

ARCHITECTURES = {"tacs": TACS_ARCHITECTURE, "mtacs": ARCHITECTURE}
ModelId = Literal["tacs", "mtacs"]

@asynccontextmanager
async def lifespan(application):
    await asyncio.to_thread(application.state.labels.load)
    await application.state.inference.start()
    try:
        yield
    finally:
        await application.state.inference.close()


app = FastAPI(title="SUN397 TACS / M-TACS Demo", version="1.0.0", lifespan=lifespan)
app.state.inference = InferenceManager()
app.state.labels = LabelCatalog(app.state.inference.settings.bundles)
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/mtacs")
def mtacs_home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/api/status")
def status(request: Request, model: ModelId = "tacs") -> dict:
    return {
        "model": model,
        "architecture": ARCHITECTURES[model],
        "inference": request.app.state.inference.status(model),
        "upload": {
            "max_bytes": MAX_UPLOAD_BYTES,
            "formats": list(ALLOWED_FORMATS.values()),
        },
    }


@app.get("/healthz")
def health() -> dict:
    return {"ok": True}


@app.get("/api/classes")
def class_labels(request: Request) -> dict:
    catalog = request.app.state.labels
    return {"dataset": "SUN397", "classes": [{"class_id": index, "label": name}
            for index, name in enumerate(catalog.classes)]}


@app.get("/readyz")
def readiness(request: Request) -> JSONResponse:
    models = {name: request.app.state.inference.status(name) for name in ("tacs", "mtacs")}
    ready = all(value["available"] for value in models.values())
    return JSONResponse(status_code=200 if ready else 503, content={"ready": ready, "models": models})


async def _image_metadata(file: UploadFile) -> tuple[dict, bytes]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Ảnh cần có tên file.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 12 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")

    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415, detail="Chỉ hỗ trợ ảnh JPEG, PNG hoặc WebP."
                )
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413, detail="Ảnh vượt quá giới hạn 36 triệu pixel."
                )
            image.load()
            width, height = ImageOps.exif_transpose(image).size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=415, detail="File không phải ảnh hợp lệ.") from None

    metadata = {
        "accepted": True,
        "filename": PureWindowsPath(file.filename).name[:180],
        "format": image_format,
        "mime_type": ALLOWED_FORMATS[image_format],
        "width": width,
        "height": height,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "stored": False,
    }
    return metadata, data


@app.post("/api/images/inspect")
async def inspect_image(request: Request, file: UploadFile = File(...)) -> dict:
    try:
        metadata, _ = await _image_metadata(file)
        metadata["ground_truth"] = request.app.state.labels.lookup(metadata["sha256"])
        return metadata
    finally:
        await file.close()


@app.post("/api/classify")
async def classify(request: Request, file: UploadFile = File(...), model: ModelId = "tacs",
                   reference_class_id: int | None = Form(None, ge=0, le=396)) -> JSONResponse:
    try:
        metadata, data = await _image_metadata(file)
    finally:
        await file.close()
    manager = request.app.state.inference
    catalog = request.app.state.labels
    try:
        ground_truth = catalog.reference(metadata["sha256"], reference_class_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        result = await manager.predict(model, data)
        return JSONResponse(content=catalog.enrich(model, result, ground_truth))
    except InferenceUnavailable as exc:
        state = manager.status(model)
        message, code, http_status = str(exc), state["code"], 503
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Inference failed for %s", model)
        message, code, http_status = "Suy luận thất bại; xem log model trên máy chủ.", "inference_error", 500
    return JSONResponse(
        status_code=http_status,
        content={
            "model": model,
            "code": code,
            "message": message,
            "prediction": None,
            "evidence": {
                "query_caption": {
                    "text": None,
                    "status": "not_used" if model == "tacs" else "caption_model_unavailable",
                },
                "attention": {
                    "query_classifier": {
                        "image_url": None,
                        "status": "checkpoint_not_connected",
                    },
                    "candidate_selector": {
                        "image_url": None,
                        "status": "checkpoint_not_connected",
                    },
                },
                "candidate": {
                    "image_url": None,
                    "image_id": None,
                    "caption": None,
                    "status": "candidate_pool_unavailable",
                },
            },
        },
    )
