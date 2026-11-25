"""Build one self-contained ISIC visual checkpoint-snapshot web export notebook."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from _project_bootstrap import PROJECT_ROOT

SAVED_RUN = PROJECT_ROOT / "results (3)/outputs/isic2017_tacs_vitl16_448_web_v2"
TARGET = PROJECT_ROOT / "notebooks/04_deployment/isic2017/03_export_only/isic2017-tacs-visual-vitl16-448-export-only.ipynb"
EXPORTER = PROJECT_ROOT / "src/deployment/isic_export_only.py"


def make_cell(kind, source, section):
    source = source.strip() + "\n"
    value = {"cell_type": kind, "id": hashlib.sha256((section + source).encode()).hexdigest()[:12],
             "metadata": {"section": section}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value


def build_notebook():
    protocol = json.loads((SAVED_RUN / "protocol.json").read_text(encoding="utf-8"))
    runtime = {p.name: p.read_text(encoding="utf-8") for p in sorted((SAVED_RUN / "embedded_runtime").glob("*.py"))}
    assert set(runtime) == {"tacs_visual_web.py", "isic_segmentation_web.py", "train_isic_segmentation_web.py"}
    exporter = EXPORTER.read_text(encoding="utf-8")
    for source in [exporter, *runtime.values()]:
        ast.parse(source)
    cells = []
    def add(kind, source, section):
        cells.append(make_cell(kind, source, section))
    add("markdown", """# ISIC-2017 · TACS ViT-L/16, 448 · xuất web từ best.pt

**Không train, không optimizer, không chạy lại 100 epoch.** Khôi phục đúng checkpoint, pool và runtime của run đã tải về.
Run gốc dừng ở epoch **94/100**, best tại **49**, validation Dice **0.886179** và IoU **0.813151**.
Đây là snapshot demo của run bị gián đoạn; không ghi giả rằng đã train đủ 100 epoch. `ready` chỉ xác nhận bộ export đã qua kiểm tra nạp lại.
Giữ nguyên split sau loại ảnh trùng: **2.000 train / 149 val / 597 test**, pool **400 ảnh train**. Không chia lại hoặc lấy pool mới.

**Inputs cần attach:** folder có `best.pt`, `protocol.json`, `run_binding.json`, `candidate_pool.csv`, ba split CSV;
ISIC-2017 ảnh + mask gốc; source DINOv3 và đúng frozen pretrained `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
Giữ `history.csv`, `embedded_runtime/` và `shared/candidate_bank.pt` nếu đã tải. `last.pt` chỉ được đọc metadata khi có, không cần optimizer.
Runtime đúng run đã được nhúng làm fallback, chỉ chấp nhận nếu hash khớp. **Không cần Florence/CLIP/caption cho bản visual này.**

Output chứa backbone, `selector.pt`, `segmentor.pt` (toàn bộ 24 context blocks + DPT), bank, ảnh pool, mask GT,
runtime, source, provenance và ví dụ mask/attention thật. Attention không cần model train riêng.
Mặc định bỏ đánh giá toàn test để tiết kiệm thời gian; smoke test một ảnh không phải metric benchmark.
ZIP được ghi trực tiếp thành các phần 256 MiB để tải từng file; không cần tạo thêm một ZIP lớn trước khi chia.
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
    raise RuntimeError("A compatible PyTorch with SDPA is required")
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(2)
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
''', "preflight")
    add("markdown", """## Đường dẫn
`SOURCE_RUN_DIR=""` tự tìm một run TACS phù hợp trong `/kaggle/input`. Có thể đặt đúng folder chứa best.pt hoặc folder Input bao ngoài nó.
Không đặt OUTPUT_DIR bên trong run cũ. Dataset/source/weights khác hash sẽ bị từ chối; không tải qua mạng.
GPU giúp tạo bank/nạp lại nhanh hơn. Giữ `RUN_FULL_TEST=False` cho demo nhanh; đổi thành True nếu cần đánh giá checkpoint trên 597 ảnh test đã lưu.
""", "paths_intro")
    add("code", f'''SOURCE_RUN_DIR = os.getenv("ISIC_TACS_SAVED_RUN", "")
ISIC_ROOT = os.getenv("ISIC_ROOT", "")
DINOV3_REPO_DIR = Path(os.getenv("DINOV3_REPO_DIR", "/kaggle/input/datasets/nguyenducxu/dinov3-code/dinov3-main"))
DINOV3_WEIGHTS = Path(os.getenv("DINOV3_WEIGHTS", "/kaggle/input/datasets/sinhnguync/dinov3-vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("ISIC_TACS_EXPORT_OUTPUT", str(PROJECT_DIR / "isic2017_tacs_snapshot_web_export_v1")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_BATCH_SIZE = 4
RUN_FULL_TEST = False
COLD_RELOAD_DEVICE = DEVICE  # Có thể chọn "cpu" để kiểm riêng trên CPU; sẽ chậm hơn.
DOWNLOAD_PART_MIB = 256
EXPECTED_RUN_IDENTITY = {protocol['identity']!r}
print("New export:", OUTPUT_DIR, "| device:", DEVICE)
''', "configuration")
    add("code", '''EXPORT_CODE_DIR = OUTPUT_DIR / "notebook_exporter"
EXPORT_CODE_DIR.mkdir(parents=True, exist_ok=True)
EXPORTER_SOURCE = ''' + repr(exporter) + '''
FALLBACK_SOURCES = ''' + repr(runtime) + '''
exporter_file = EXPORT_CODE_DIR / "isic_export_only.py"
if exporter_file.is_file() and exporter_file.read_text(encoding="utf-8") != EXPORTER_SOURCE:
    raise ValueError("Exporter changed under this output. Preserve it and select a new OUTPUT_DIR.")
exporter_file.write_text(EXPORTER_SOURCE, encoding="utf-8")
loaded = sys.modules.get("isic_export_only")
if loaded and Path(loaded.__file__).resolve() != exporter_file.resolve():
    raise RuntimeError("Another exporter is imported. Restart the kernel.")
sys.path.insert(0, str(EXPORT_CODE_DIR.resolve()))
import isic_export_only as exporter
''', "embedded_exporter")
    add("markdown", """## Xác minh run và tài nguyên
Đọc metadata checkpoint bằng meta storage; không nạp optimizer 1,2 GB vào RAM/GPU.
Giữ nguyên thứ tự CSV. Đường dẫn ảnh/mask được tìm lại theo ID và **kiểm SHA256 + pixel hash**, không đoán theo tên file.
Nếu attach đúng Input ISIC nhưng auto-discovery thấy nhiều bản, đặt ISIC_ROOT tới folder chứa train/val/test của bản gốc.
""", "prepare_intro")
    add("code", '''SAVED_RUN = exporter.discover_run(SOURCE_RUN_DIR)
if exporter.read_json(SAVED_RUN / "protocol.json")["identity"] != EXPECTED_RUN_IDENTITY:
    raise ValueError("This notebook is bound to the TACS ISIC run from results (3). Attach that run.")
CONTEXT = exporter.prepare(
    SAVED_RUN, ISIC_ROOT, DINOV3_REPO_DIR, DINOV3_WEIGHTS, OUTPUT_DIR, FALLBACK_SOURCES,
    device=DEVICE, feature_batch_size=FEATURE_BATCH_SIZE, run_full_test=RUN_FULL_TEST,
)
display(CONTEXT["snapshot"])
''', "prepare")
    add("markdown", """## Xuất các phần model và dữ liệu cho web
Khôi phục best.pt vào đúng frozen backbone, tách utility selector / 24 context blocks + DPT.
Ưu tiên dùng lại `shared/candidate_bank.pt` đã lưu; thiếu bank thì chỉ encode lại đúng 400 ảnh, không cập nhật weights.
Source/weights/checkpoint/split gốc được đọc, không sửa. Mask GT được đóng gói để đối chiếu sau dự đoán.
""", "build_intro")
    add("code", '''CONTEXT = exporter.build(CONTEXT)
print("Staged bundle:", CONTEXT["stage"])
display(exporter.read_json(CONTEXT["stage"] / "metrics.json"))
for item in ["models/selector.pt", "models/segmentor.pt", "candidate_bank.pt"]:
    path = CONTEXT["stage"] / item
    print(item, round(path.stat().st_size / 1024**2, 2), "MiB")
''', "build")
    add("markdown", """## Nạp lại độc lập, kiểm logits / context / attention
Giải phóng model trong kernel rồi dùng process mới, chỉ từ bundle đã copy. So với logits của best.pt đã khôi phục;
bật/tắt attention phải giữ logits, context ID phải khớp và có đủ hai map 28×28.
Chỉ sau khi thành công mới ghi `manifest.json: ready`. Trạng thái train dở vẫn ghi riêng, không đổi thành hoàn thành.
""", "verify_intro")
    add("code", '''BUNDLE_DIR = exporter.verify_and_publish(CONTEXT, device=COLD_RELOAD_DEVICE)
display(exporter.read_json(BUNDLE_DIR / "smoke_test.json"))
display(exporter.read_json(BUNDLE_DIR / "training_snapshot.json"))
print("Verified web bundle:", BUNDLE_DIR)
''', "verify")
    add("code", '''from IPython.display import display, Image as DisplayImage
result = exporter.read_json(BUNDLE_DIR / "example/prediction.json")
display({"selected_context_id": result["evidence"]["candidate"]["image_id"],
         "single_example_comparison": result["comparison"], "not_test_set_metrics": True})
for name in ("mask", "overlay", "query_segmentor", "candidate_selector"):
    print(name)
    display(DisplayImage(filename=str(BUNDLE_DIR / "example" / (name + ".png")), width=420))
''', "preview")
    add("markdown", """## Tải từng phần ZIP
Chạy cell dưới để tạo `downloads/`. Tải **tất cả** `.zip.partNNN`, `download_manifest.json` và `join_downloads.py` vào cùng folder.
Trên Windows chạy `python join_downloads.py`, rồi giải nén ZIP đã ghép. Script kiểm SHA256 từng phần và toàn ZIP, không cần PyTorch.
Các file part không phải ZIP độc lập để giải nén riêng. Chỉ dùng bundle có status ready; giữ best.pt gốc riêng.
""", "downloads_intro")
    add("code", '''DOWNLOAD_DIR = exporter.package_downloads(CONTEXT, part_mib=DOWNLOAD_PART_MIB)
from IPython.display import FileLink
download = exporter.read_json(DOWNLOAD_DIR / "download_manifest.json")
print("Download all", len(download["parts"]), "parts + the two small helper files below:")
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
                 "mode": "tacs", "source_run_identity": protocol["identity"], "training_not_resumed": True}}}


if __name__ == "__main__":
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)
