"""Build the standalone ISIC full_both recovery/export notebook (no training)."""
from __future__ import annotations

import ast
import hashlib
import json

from _project_bootstrap import PROJECT_ROOT
from build_isic_segmentation_web_notebooks import embedded_sources

TARGET = PROJECT_ROOT / "notebooks/04_deployment/isic2017/03_export_only/isic2017-mtacs-full-both-florence-vitl16-448-export-only.ipynb"


def cell(kind, source, section):
    source = source.strip() + "\n"
    value = {"cell_type": kind, "id": hashlib.sha256((section + source).encode()).hexdigest()[:12],
             "metadata": {"section": section}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value


def build_notebook():
    exporters = {name: (PROJECT_ROOT / "src/deployment" / name).read_text("utf-8")
                 for name in ("isic_export_only.py", "isic_mtacs_export_only.py")}
    fallback = embedded_sources("mtacs")
    for source in [*exporters.values(), *fallback.values()]:
        ast.parse(source)
    cells = []

    def add(kind, source, section):
        cells.append(cell(kind, source, section))

    add("markdown", """# ISIC-2017 · M-TACS full_both · export từ best.pt đã train

Attach **output cũ thành Kaggle Dataset**, rồi chạy các cell theo thứ tự. Notebook **không train lại**,
không tạo lại CSV, không thay text embedding và không sửa output cũ. Khôi phục đúng ViT-L/16, 448×448, K=1;
query text + candidate text vào fusion và selector như checkpoint đã lưu. Epoch, metric, split và pool đọc từ run thật,
không lấy số liệu của TACS. Run bị dừng vẫn ghi `training_complete=false`; `ready` chỉ xác nhận export đã kiểm nạp lại.

**Inputs cần attach:**
- Output M-TACS trong ảnh: `best.pt`, `protocol.json`, `run_binding.json`, `candidate_pool.csv`, ba split CSV,
  `embedded_runtime/`, `history.csv` và `shared/caption_generation/` (giữ CSV, text_cache, settings, status, generation_protocol).
  `last.pt` không bắt buộc; có thì chỉ đọc metadata. Có `shared/candidate_bank.pt` sẽ dùng lại, thiếu thì encode đúng pool cũ.
- ISIC-2017 ảnh + mask gốc; DINOv3 source và frozen `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` đúng run.
- **Đúng Florence-2-base Microsoft + CLIP ViT-B/32 đã dùng lúc tạo caption**, cùng offline wheels từ notebook cũ
  (`transformers 4.49.x`, `tokenizers 0.21.x`, `huggingface-hub<1`, và các dependency cần thiết).
  Output train thường không chứa lại hai model pretrained này: attach thêm Input gốc.

Output đủ cho web: backbone, selector, segmentor (mọi context block/DPT/text projector/gate), candidate bank và ảnh,
mask GT để đối chiếu, captions.csv, text cache, Florence/CLIP + processor/tokenizer, runtime và hai map attention thật.
Attention không cần model train riêng. Caption là mô tả tự động từ pixels, không phải nhãn GT segmentation.
ZIP được ghi thành từng phần 256 MiB để tải riêng, không phải tải toàn bộ folder lớn.
""", "overview")
    add("code", '''import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PIP_NO_INDEX"] = "1"
import sys, importlib.util, json
from pathlib import Path
missing = [name for name in ("torch", "numpy", "PIL", "tqdm") if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Missing offline packages: " + ", ".join(missing))
import torch
if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    raise RuntimeError("PyTorch with SDPA is required")
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.is_available())
''', "preflight")
    add("markdown", """## 1. Đường dẫn

`SOURCE_RUN_DIR=""` tự tìm đúng một run **mtacs** trong `/kaggle/input`; nếu có nhiều run, điền folder Input bạn vừa upload
hoặc folder cụ thể chứa `best.pt`. Ví dụ `/kaggle/input/datasets/TEN_CUA_BAN/OUTPUT_MTACS/outputs/isic2017_mtacs_...`.
Không điền đường dẫn tới source DINOv3 vào biến này. Giữ OUTPUT_DIR riêng với Input cũ.

Dataset/model có thể tự dò tìm; nếu có nhiều bản, điền đường dẫn chính xác. Hash phải khớp run đã train.
Không cần hạ Transformers trong kernel: chỉ caption subprocess dùng 4.49 từ wheels offline.
""", "paths_intro")
    add("code", '''SOURCE_RUN_DIR = os.getenv("ISIC_MTACS_SAVED_RUN", "")
CAPTION_OUTPUT_DIR = os.getenv("ISIC_MTACS_SAVED_CAPTIONS", "")  # Rỗng: dùng shared/caption_generation của run cũ.
ISIC_ROOT = os.getenv("ISIC_ROOT", "")
DINOV3_REPO_DIR = Path(os.getenv("DINOV3_REPO_DIR", "/kaggle/input/datasets/nguyenducxu/dinov3-code/dinov3-main"))
DINOV3_WEIGHTS = Path(os.getenv("DINOV3_WEIGHTS", "/kaggle/input/datasets/sinhnguync/dinov3-vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
CLIP_MODEL_REF = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
OFFLINE_ASSETS_ROOT = os.getenv("FLORENCE_OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("ISIC_MTACS_EXPORT_OUTPUT", str(PROJECT_DIR / "isic2017_mtacs_snapshot_web_export_v1")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_BATCH_SIZE = 4
RUN_FULL_TEST = False  # Bật nếu cần chạy đủ test của checkpoint; không cần cho demo.
COLD_RELOAD_DEVICE = DEVICE
CAPTION_VERIFY_DEVICE = DEVICE  # Caption và segmentation được kiểm lần lượt, không cùng giữ VRAM.
DOWNLOAD_PART_MIB = 256
print("Export output:", OUTPUT_DIR, "| device:", DEVICE)
''', "configuration")
    add("code", '''EXPORT_CODE_DIR = OUTPUT_DIR / "notebook_exporter"
EXPORT_CODE_DIR.mkdir(parents=True, exist_ok=True)
EXPORTERS = ''' + repr(exporters) + '''
FALLBACK_SOURCES = ''' + repr(fallback) + '''
for name, source in EXPORTERS.items():
    target = EXPORT_CODE_DIR / name
    if target.is_file() and target.read_text(encoding="utf-8") != source:
        raise ValueError("Exporter changed under this output. Choose a new OUTPUT_DIR.")
    target.write_text(source, encoding="utf-8")
    loaded = sys.modules.get(Path(name).stem)
    if loaded and Path(loaded.__file__).resolve() != target.resolve():
        raise RuntimeError("Another exporter is imported. Restart the kernel and Run All.")
sys.path.insert(0, str(EXPORT_CODE_DIR.resolve()))
import isic_export_only as shared_export
import isic_mtacs_export_only as exporter
''', "embedded_exporter")
    add("markdown", """## 2. Khớp checkpoint, split, pool và captions đã train

Runtime của chính output cũ được ưu tiên và phải khớp `run_binding.json`. Chỉ dùng fallback nhúng khi đúng hash;
không đoán kiến trúc hoặc chia lại dữ liệu. Kiểm SHA ảnh/mask/model/CSV/cache, policy và mask text.
Nếu thiếu file caption cũ, giữ output làm Dataset đầy đủ; không sinh CSV mới để ghép vào checkpoint này.
""", "prepare_intro")
    add("code", '''CONTEXT = exporter.prepare(
    SOURCE_RUN_DIR, ISIC_ROOT, DINOV3_REPO_DIR, DINOV3_WEIGHTS, OUTPUT_DIR, FALLBACK_SOURCES,
    caption_output=CAPTION_OUTPUT_DIR, florence_ref=FLORENCE_MODEL_REF, clip_ref=CLIP_MODEL_REF,
    offline_assets=OFFLINE_ASSETS_ROOT, device=DEVICE, feature_batch_size=FEATURE_BATCH_SIZE,
    run_full_test=RUN_FULL_TEST,
)
display(CONTEXT["snapshot"])
display(CONTEXT["caption_coverage"])
print("Source run:", CONTEXT["run"])
''', "prepare")
    add("markdown", """## 3. Xuất model và dữ liệu portable

Giữ nguyên best.pt gốc; tách phần học được thành `selector.pt` và `segmentor.pt`, kèm frozen backbone.
Copy đúng Florence/CLIP đã đăng ký, dùng nguyên CSV/embedding. Giải phóng model trước khi kiểm caption và nạp lại ở process khác.
Không copy last.pt/optimizer, per-image generated caches, legacy_packages hay log lớn vào ZIP.
""", "build_intro")
    add("code", '''CONTEXT = shared_export.build(CONTEXT)
print("Staged bundle:", CONTEXT["stage"])
display(shared_export.read_json(CONTEXT["stage"] / "metrics.json"))
for name in ("models/selector.pt", "models/segmentor.pt", "candidate_bank.pt", "captions.csv", "text_embeddings.pt"):
    path = CONTEXT["stage"] / name
    print(name, round(path.stat().st_size / 1024**2, 2), "MiB")
''', "build")
    add("markdown", """## 4. Kiểm nạp lại và ảnh mới

Process caption độc lập nạp Florence/CLIP từ bundle, bắt buộc sinh một caption mới từ pixels và kiểm CLIP tái tạo
vector của một caption đã train. Process segmentation độc lập kiểm logits với checkpoint, context ID, hai map attention,
và forward full_both khi dùng caption vừa sinh. Không thay CSV đã train. Chỉ thành công mới ghi `ready`.
Một ảnh minh họa không phải metric toàn test; mặc định `metrics.json` để test=null nếu chưa chạy đánh giá test.
""", "verify_intro")
    add("code", '''BUNDLE_DIR = exporter.verify_and_publish(CONTEXT, device=COLD_RELOAD_DEVICE, caption_device=CAPTION_VERIFY_DEVICE)
display(shared_export.read_json(BUNDLE_DIR / "smoke_test.json"))
display(shared_export.read_json(BUNDLE_DIR / "example/fresh_caption_check.json"))
print("Verified bundle:", BUNDLE_DIR)
''', "verify")
    add("code", '''from IPython.display import display, Image as DisplayImage
prediction = shared_export.read_json(BUNDLE_DIR / "example/prediction.json")
display({"query_caption": prediction["evidence"]["query_caption"],
         "candidate_id": prediction["evidence"]["candidate"]["image_id"],
         "candidate_caption": prediction["evidence"]["candidate"]["caption"],
         "fusion": prediction["evidence"]["fusion"], "not_test_set_metrics": True})
for name in ("mask", "overlay", "query_segmentor", "candidate_selector"):
    print(name)
    display(DisplayImage(filename=str(BUNDLE_DIR / "example" / (name + ".png")), width=420))
''', "preview")
    add("markdown", """## 5. Tải từng phần ZIP

Tải **tất cả** `.zip.partNNN` và hai file nhỏ `download_manifest.json`, `join_downloads.py` vào cùng folder máy tính.
Chạy `python join_downloads.py`, rồi giải nén ZIP đã ghép. Script kiểm SHA256 và không cần PyTorch.
Các phần không phải ZIP độc lập. Nếu web dùng Python/platform khác Kaggle, wheels native phải phù hợp platform;
model weights/CSV/runtime vẫn giữ nguyên. Máy web cần môi trường Python/PyTorch tương thích.
""", "downloads_intro")
    add("code", '''DOWNLOAD_DIR = exporter.package_downloads(CONTEXT, part_mib=DOWNLOAD_PART_MIB)
from IPython.display import FileLink
download = shared_export.read_json(DOWNLOAD_DIR / "download_manifest.json")
print("Download", len(download["parts"]), "parts + download_manifest.json + join_downloads.py")
for item in download["parts"]:
    print(item["name"], round(item["bytes"] / 1024**2, 2), "MiB")
    display(FileLink(str(DOWNLOAD_DIR / item["name"])))
for name in ("download_manifest.json", "join_downloads.py"):
    display(FileLink(str(DOWNLOAD_DIR / name)))
''', "downloads")
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "tacs": {"purpose": "checkpoint_snapshot_export_only", "dataset": "ISIC2017", "task": "binary_segmentation",
                 "mode": "mtacs", "method": "full_both", "training_not_resumed": True, "uses_saved_caption_cache": True}}}


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)
