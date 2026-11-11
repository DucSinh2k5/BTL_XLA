"""Build an offline, caption-only ISIC FirstValid experiment; no segmentation training."""
from __future__ import annotations

import ast
import hashlib
import json
import textwrap

from _project_bootstrap import PROJECT_ROOT
import build_isic_attribute_v2_notebook as v2

METHOD = "florence_attributes_firstvalid_v3"
TARGET = PROJECT_ROOT / "notebooks/02_improvements/segmentation/caption_generation/florence_attributes_firstvalid_v3/isic2017/isic2017-captions-only-florence-attributes-firstvalid-v3.ipynb"


def runtime_source():
    source = v2.caption_core()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test))
    source = "".join(source.splitlines(keepends=True)[:main.lineno - 1])
    old_method = f"METHOD = {v2.METHOD!r}"
    if source.count(old_method) != 1:
        raise ValueError("Recheck the attributes-v2 core method definition")
    source = source.replace(old_method, f"METHOD = {METHOD!r}", 1)
    helpers = (PROJECT_ROOT / "src/captioning/isic_caption_firstvalid_v3.py").read_text("utf-8")
    additions = []
    for node in ast.parse(helpers).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            additions.append(ast.get_source_segment(helpers, node))
    source += "\n\n" + "\n\n\n".join(additions)
    source += '\n\nif __name__ == "__main__":\n    firstvalid_worker_main(sys.argv[1], sys.argv[2])\n'
    ast.parse(source)
    return source


def cell(kind, section, source):
    source = textwrap.dedent(source).lstrip("\n")
    result = {"cell_type": kind, "id": hashlib.sha256(section.encode()).hexdigest()[:12],
              "metadata": {"section": section}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


CONFIGURATION = '''
import os
# Set BEFORE importing transformers/tokenizers. No DataLoader/fork is used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from pathlib import Path
import json
import sys

CAPTIONS_ONLY = True  # This notebook has no segmentation training cells.
if CAPTIONS_ONLY is not True:
    raise ValueError("Keep CAPTIONS_ONLY=True here; train M-TACS separately with the reviewed new CSV")

# Leave empty for discovery when exactly one different old CSV is attached.
# Example: /kaggle/input/datasets/YOUR_NAME/YOUR_BUNDLE/isic2017_mtacs_vitl16_448_web_bundle
# Accepts an old bundle, a saved run, shared/caption_generation, or its captions.csv.
PRIOR_CAPTIONS_SOURCE = os.getenv("ISIC_FIRSTVALID_PRIOR_SOURCE", "")
ISIC_ROOT = os.getenv("ISIC_ROOT", "")  # Folder containing the original ISIC RGB images.
FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "")  # Complete Microsoft Florence-2-base folder.
OFFLINE_ASSETS_ROOT = os.getenv("OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")

ENCODE_CLIP_TEXT = False  # CSV-only by default; CLIP weights are optional.
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "")

OUTPUT_DIR = Path(os.getenv("ISIC_FIRSTVALID_OUTPUT", "/kaggle/working/outputs/isic2017_captions_attributes_firstvalid_v3")).expanduser().resolve()
SEARCH_ROOTS = [Path("/kaggle/input")]
SETTINGS = {
    "seed": 42, "device": "cuda:0", "precision": "float32", "batch_size": 4,
    "max_new_tokens": 96,  # Original prefix setting retained as metadata, not a new initial pass.
    "retry_greedy_tokens": 128,
    "retry_beam_tokens": 192, "retry_beams": 4, "retry_returns": 3,
}
print("Caption-only:", CAPTIONS_ONLY, "| CLIP cache:", ENCODE_CLIP_TEXT)
print("New output:", OUTPUT_DIR)
'''

SOURCE_AND_IMAGES = '''
# Read only registered pixel/split metadata. No segmentation masks are opened.
SOURCE_DIR, SOURCE_PROTOCOL, SOURCE_STATUS = core.resolve_prior_source(PRIOR_CAPTIONS_SOURCE, SEARCH_ROOTS)
if OUTPUT_DIR == SOURCE_DIR or OUTPUT_DIR in SOURCE_DIR.parents or SOURCE_DIR in OUTPUT_DIR.parents:
    raise ValueError("Use a separate output folder outside the old source; do not overwrite a run/bundle")
CAPTION_RECORDS = core.recover_image_records(SOURCE_PROTOCOL, ISIC_ROOT, SEARCH_ROOTS)
core.atomic_json(OUTPUT_DIR / "resolved_image_records.json", CAPTION_RECORDS)
from collections import Counter
print("Old caption source:", SOURCE_DIR)
print("Exact saved split:", dict(Counter(r["split"] for r in CAPTION_RECORDS)))
print("Original CSV SHA256:", SOURCE_STATUS["csv_sha256"])
'''

ASSETS = '''
# Use the Florence weights/wheels already exported with the old bundle when present.
model_ref = FLORENCE_MODEL_REF
if not model_ref and (SOURCE_DIR / "caption_models/florence").is_dir():
    model_ref = str(SOURCE_DIR / "caption_models/florence")
if not model_ref:
    attached = Path("/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
    model_ref = str(attached) if attached.is_dir() else core.MODEL_ID
MODEL_PATH, BACKEND = core.florence_checkpoint(model_ref, [SOURCE_DIR, *SEARCH_ROOTS])
if BACKEND != "legacy":
    raise ValueError("Use the original Microsoft Florence-2-base checkpoint with offline Transformers 4.49")
core.validate_florence_prefix_folder(MODEL_PATH)
PAYLOAD = {
    "output": str(OUTPUT_DIR), "model_path": str(MODEL_PATH), "settings": SETTINGS,
    "records": CAPTION_RECORDS, "prior_captions_source": str(SOURCE_DIR),
    "code_sha256": core.file_sha256(RUNTIME_FILE),
}
registration, old_captions = core.prior_caption_registration(PAYLOAD)
if len(old_captions) != len(CAPTION_RECORDS):
    raise ValueError("The source must contain registered long captions for every image")
asset_root = Path(OFFLINE_ASSETS_ROOT)
if not asset_root.is_dir():
    asset_root = SOURCE_DIR
CAPTION_ENV = core.caption_runtime_environment("legacy", asset_root, OUTPUT_DIR / "caption_runtime")
if ENCODE_CLIP_TEXT:
    clip_ref = CLIP_MODEL_LOCAL_PATH
    if not clip_ref and (SOURCE_DIR / "caption_models/clip").is_dir():
        clip_ref = str(SOURCE_DIR / "caption_models/clip")
    PAYLOAD["clip_path"] = str(core.resolve_clip_model(clip_ref, [SOURCE_DIR, *SEARCH_ROOTS]))
REQUEST_FILE = OUTPUT_DIR / "request.json"
core.atomic_json(REQUEST_FILE, PAYLOAD)
print("Frozen Florence:", MODEL_PATH)
print("Reusable long captions:", len(old_captions))
'''

GENERATION = '''
# The child process uses isolated offline Transformers 4.49. The kernel is unchanged.
# Resume uses original image hashes, decoding settings and per-image retry audits.
core.run_worker(RUNTIME_FILE, REQUEST_FILE, "generate", CAPTION_ENV, OUTPUT_DIR / "generate.log")
NEW_CAPTIONS, CAPTION_STATUS = core.validated_export(PAYLOAD)
print("Complete CSV:", CAPTION_STATUS["csv_path"])
print("Baseline v2 masked:", CAPTION_STATUS["baseline_v2_masked"])
print("Recovered via retries:", CAPTION_STATUS["recovered_from_v2"])
print("Final masked:", CAPTION_STATUS["final_masked"])
'''

REPORTS = '''
import pandas as pd
from IPython.display import display
QUALITY_REPORT = core.write_firstvalid_reports(OUTPUT_DIR)
display(pd.read_csv(OUTPUT_DIR / "mask_comparison.csv"))
review = pd.read_csv(OUTPUT_DIR / "caption_retries_review.csv").fillna("")
recovered = review[review["final_valid"].astype(str).str.lower().eq("true")
                   & review["baseline_valid"].astype(str).str.lower().eq("false")]
print("Sample of recovered captions (check these against their images):")
display(recovered.head(12))
print("The clinical blacklist, 5–8 word rule and attribute checks were NOT relaxed.")
print("Fewer masks do not establish factual caption accuracy or higher segmentation Dice.")
'''

REVIEW_IMAGES = '''
# Visual review of a fixed small sample, never used to pick hyperparameters from test metrics.
from PIL import Image, ImageOps
from IPython.display import Markdown
record_by_id = {r["image_id"]:r for r in CAPTION_RECORDS}
review_sample = recovered.head(8) if len(recovered) else review.head(4)
for row in review_sample.to_dict("records"):
    display(Markdown("**" + row["image_id"] + "** (" + row["split"] + ")"))
    with Image.open(record_by_id[row["image_id"]]["image_path"]) as raw:
        thumbnail = ImageOps.exif_transpose(raw).convert("RGB")
        thumbnail.thumbnail((360, 260))
        display(thumbnail)
    print("V2:", row["baseline_caption"], "| valid:", row["baseline_valid"])
    print("New:", row["final_caption"], "| valid:", row["final_valid"], "|", row["selection_origin"])
'''

OPTIONAL_CLIP = '''
if ENCODE_CLIP_TEXT:
    core.run_worker(RUNTIME_FILE, REQUEST_FILE, "encode", CAPTION_ENV, OUTPUT_DIR / "encode.log")
    print("Optional NEW CLIP cache:", OUTPUT_DIR / "text_cache/clip_text.pt")
else:
    print("Skipped CLIP encoding: captions.csv is ready; no CLIP weights required in CSV-only mode.")
'''

DOWNLOADS = '''
import zipfile
from IPython.display import FileLink, display
core.atomic_json(OUTPUT_DIR / "experiment_settings.json", {
    "method": core.METHOD, "policy": core.POLICY, "settings": SETTINGS,
    "caption_csv_sha256": CAPTION_STATUS["csv_sha256"],
    "caption_only": True, "segmentation_training": False, "filters_relaxed": False,
    "prior_csv_sha256": SOURCE_STATUS["csv_sha256"],
    "clip_text_encoded": ENCODE_CLIP_TEXT,
    "runtime_note": "Future training/inference must use this retry/extraction policy and new embeddings, not an old checkpoint as a retrained result.",
})
# Small archive: captions, audit, decoding/runtime provenance. No model weights or RGB dataset.
archive = OUTPUT_DIR / "isic2017_captions_firstvalid_v3.zip"
top_files = ["captions.csv", "status.json", "mask_comparison.csv", "caption_retries_review.csv",
             "caption_quality_report.json", "experiment_settings.json", "caption_firstvalid_runtime.py"]
selected = [OUTPUT_DIR / name for name in top_files]
selected += sorted((OUTPUT_DIR / "generation_protocol").rglob("*.json"))
selected += sorted((OUTPUT_DIR / "retry_audit").rglob("*.json"))
if ENCODE_CLIP_TEXT:
    selected += sorted(p for p in (OUTPUT_DIR / "text_cache").rglob("*") if p.is_file())
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for path in selected:
        handle.write(path, path.relative_to(OUTPUT_DIR).as_posix())
print("Download captions.csv for the next M-TACS experiment; keep the ZIP for provenance.")
for name in ("captions.csv", "mask_comparison.csv", "caption_retries_review.csv", archive.name):
    display(FileLink(str(OUTPUT_DIR / name)))
print("Done. No segmentation model was trained or modified.")
'''


def build_notebook():
    runtime = runtime_source()
    embed = '''import importlib.util
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_FILE = OUTPUT_DIR / "caption_firstvalid_runtime.py"
RUNTIME_FILE.write_text(''' + repr(runtime) + ''', encoding="utf-8")
spec = importlib.util.spec_from_file_location("isic_caption_firstvalid_runtime", RUNTIME_FILE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
core.validate_retry_settings(SETTINGS)
print("Method:", core.METHOD, "| unchanged policy:", core.POLICY)
'''
    entries = [
        ("markdown", "overview", """# ISIC — captions only: Florence Attributes + FirstValid v3

Notebook riêng để xuất **captions.csv mới**, không train Florence hoặc mô hình segmentation.
Mặc định và bắt buộc `CAPTIONS_ONLY = True`.

1. Tái áp dụng policy **attributes-v2** lên câu dài đã được đăng ký từ run/bundle cũ.
2. Giữ nguyên mọi caption v2 đã đạt.
3. Với ảnh v2 bị mask: thử greedy dài hơn (128 token); nếu chưa đạt, thử 3 ứng viên từ beam search (beam=4, 192 token), lấy câu đạt đầu tiên.
4. Vẫn kiểm tra toàn bộ câu gốc theo blacklist chẩn đoán; vẫn yêu cầu 5–8 từ và nội dung thuộc tính. Không có ứng viên đạt thì giữ mask.

Mục tiêu là giảm mask so với v2 nhờ thêm phương án sinh, **không nới bộ lọc và không cam kết mask về 0**.
Caption tự sinh không phải ground truth. Ít mask hơn không chứng minh caption đúng hơn hoặc Dice cao hơn.
Không dùng GT mask/nhãn bệnh làm đầu vào và không chọn caption dựa trên metrics test.
"""),
        ("markdown", "inputs", """## Inputs và cách chạy offline

Attach: **output/bundle ISIC M-TACS Florence cũ** (có captions.csv, status và generation protocol), **ảnh RGB ISIC gốc**, **Microsoft Florence-2-base** và **offline wheels Transformers 4.49** tương thích Python trên Kaggle.
Bundle đầy đủ có thể đã chứa Florence và wheels. Có thể dùng input `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi` như trước.
**Không cần DINO, best.pt segmentation hoặc GT masks.** CLIP chỉ cần nếu bật `ENCODE_CLIP_TEXT`.

Bật GPU, sửa đường dẫn ở cell dưới nếu auto-discovery tìm thấy nhiều nguồn, rồi **Run All**.
Nếu runtime dùng Transformers 5.x, captioner chạy subprocess 4.49 từ offline wheels; không sửa Transformers của kernel và không cần Internet.
Chạy lại cùng output để resume. Khi thay cấu hình hoặc nguồn CSV, dùng output mới.
"""),
        ("code", "configuration", CONFIGURATION),
        ("code", "embedded_caption_runtime", embed),
        ("code", "source_and_images", SOURCE_AND_IMAGES),
        ("code", "offline_model_assets", ASSETS),
        ("code", "generate_and_resume", GENERATION),
        ("code", "mask_comparison", REPORTS),
        ("code", "visual_review", REVIEW_IMAGES),
        ("code", "optional_clip_cache", OPTIONAL_CLIP),
        ("code", "download_outputs", DOWNLOADS),
    ]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "isic_caption_firstvalid_v3": {"method": METHOD, "policy": v2.POLICY,
                             "caption_only": True, "filters_relaxed": False, "no_segmentation_training": True}},
            "cells": [cell(*entry) for entry in entries]}


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)
