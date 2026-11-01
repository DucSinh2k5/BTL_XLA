"""Build two self-contained ISIC segmentation train/evaluate/export notebooks."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from _project_bootstrap import PROJECT_ROOT
import build_isic_florence_prefix_full_both_notebook as florence_builder


TARGET_ROOT = PROJECT_ROOT / "notebooks/04_deployment/isic2017"
TARGETS = {
    "tacs": TARGET_ROOT / "01_tacs_visual/isic2017-tacs-visual-vitl16-448-train-export-web.ipynb",
    "mtacs": TARGET_ROOT / "02_mtacs_full_both/isic2017-mtacs-full-both-florence-vitl16-448-train-export-web.ipynb",
}


def caption_core():
    source = florence_builder.embed_caption_core()
    replacements = {
        'self.device = torch.device("cuda:0")': 'self.device = torch.device(settings.get("device", "cuda:0"))',
        'self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16':
            'self.dtype = torch.float32',
        '"dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",':
            '"dtype": "float32",',
        'num_beams=1, do_sample=False, use_cache=True)':
            'num_beams=1, do_sample=False, use_cache=True, early_stopping=False)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Florence source changed; inspect before rebuilding: {old}")
        source = source.replace(old, new, 1)
    source += '\n\nif __name__ == "__main__":\n    import torch\n    torch.set_float32_matmul_precision("highest")\n    torch.backends.cuda.matmul.allow_tf32 = False\n    torch.backends.cudnn.allow_tf32 = False\n    florence_worker_main(sys.argv[1], sys.argv[2], generate_records=generate_isic_records, encode_texts=encode_isic_texts)\n'
    ast.parse(source)
    return source


def embedded_sources(mode):
    sources = {name: (PROJECT_ROOT / folder / name).read_text(encoding="utf-8") for name, folder in (
        ("tacs_visual_web.py", "src/models"), ("isic_segmentation_web.py", "src/models"),
        ("train_isic_segmentation_web.py", "src/training"))}
    if mode == "mtacs":
        sources["isic_caption_core.py"] = caption_core()
        sources["isic_caption_web.py"] = (PROJECT_ROOT / "src/captioning/isic_caption_web.py").read_text(encoding="utf-8")
    for source in sources.values():
        ast.parse(source)
    return sources


def cell(kind, source, section):
    source = source.strip() + "\n"
    result = {"cell_type": kind, "id": hashlib.sha256((section + source).encode()).hexdigest()[:12],
              "metadata": {"section": section}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(outputs=[], execution_count=None)
    return result


def build_notebook(mode):
    multimodal = mode == "mtacs"
    name = "M-TACS full_both + Florence-Prefix" if multimodal else "TACS visual"
    cells = []

    def add(kind, source, section):
        cells.append(cell(kind, source, section))

    add("markdown", f'''# ISIC-2017 — {name}: train, evaluate and complete web export

Notebook độc lập cho segmentation nhị phân vùng tổn thương da. Chạy các cell lần lượt từ trên xuống.
Mặc định **DINOv3 ViT-L/16, ảnh 448×448, K=1, seed 42, 100 epoch**. Có thể đặt 50 epoch ở cell cấu hình
TRƯỚC lần chạy đầu; không thay ngân sách/split/pool/caption dưới cùng output khi resume.

Kiến trúc theo cơ chế segmentation của [TACS, §3–4.2](https://arxiv.org/html/2512.00489v1):
backbone pretrained đóng băng, gated cross-attention query→context sau MỖI transformer block (24 block với L/16),
DPT nhẹ lấy bốn độ sâu, selector học chọn trên toàn pool train cố định, hard Gumbel ST cho task loss và
categorical REINFORCE cho reward giảm task loss. Backbone dùng chung cho selector và segmentor.
Đây là tái dựng theo mô tả paper, có utility projection head học được trên CLS frozen và một DPT nhẹ được triển khai rõ trong source.
Không dùng bản cũ chỉ thêm hai cross-attention block sau backbone. Không tuyên bố giống từng dòng code tác giả chưa cung cấp.
Paper dùng **S/16, 256×256 và trung bình 3 seed**; bản L/16-448 một seed này phải báo cáo riêng.

{'M-TACS dùng caption query và caption của đúng candidate được chọn trong fusion; selector kết hợp utility ảnh và cosine CLIP text có hệ số học được. Caption do Florence sinh tự động, không phải ground truth. Notebook tự tạo/resume captions.csv; không cần CSV có sẵn.' if multimodal else 'TACS này chỉ dùng ảnh; không có Florence, CLIP, caption, nhánh text hay nhãn/mask candidate trong đầu vào mô hình.'}

Output cuối là ZIP tự chứa weights backbone, selector, segmentor (toàn bộ attention/DPT{'/text gates' if multimodal else ''}),
ảnh pool, bank khớp checkpoint, mask ground truth để ĐỐI CHIẾU, config, split, runtime, source DINOv3,
metrics, ví dụ mask/probability/overlay và attention thật. {'M-TACS còn đóng gói đầy đủ Florence-2 + CLIP, tokenizer/processor, CSV và text cache.' if multimodal else ''}
Mask GT không đi vào selector, captioner hay forward dự đoán; chỉ dùng làm target loss/metrics hoặc hiển thị sau dự đoán.

Attention không cần một model `.pt` riêng. Runtime xuất self-attention CLS→patch ở block cuối của segmentor/query
và encoder selector/candidate, trung bình các head. Đây không phải Grad-CAM hoặc bằng chứng nhân quả/chẩn đoán;
map selector không giải thích trực tiếp các projection/điểm text. Không huấn luyện baseline no-context độc lập ở notebook này.

Các cell không tải dữ liệu/model qua mạng. {'Chỉ caption subprocess được cài dependency 4.49 vào thư mục riêng từ wheels Input bằng --no-index; kernel train giữ nguyên Transformers.' if multimodal else 'Không cần Transformers cho notebook visual.'}
Web FastAPI hiện tại chưa được thêm route segmentation: bộ ZIP chuẩn bị adapter/artifacts để nối vào web ở bước sau.
Không cam kết số giờ hoặc hết lỗi trên mọi GPU/runtime; các preflight và cold reload sẽ dừng với thông báo cụ thể nếu Input/môi trường không khớp.
''', "overview")
    add("markdown", '''## 1. Offline preflight

Đặt tokenizer parallelism trước mọi import/worker. Mặc định DataLoader workers=0; caption models ở child process và được giải phóng trước khi train segmentor.
''', "preflight_intro")
    add("code", '''import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PIP_NO_INDEX"] = "1"

import importlib.util
import sys
from pathlib import Path
if sys.version_info < (3, 10):
    raise RuntimeError("Use Python >=3.10")
missing = [name for name in ("torch", "numpy", "PIL", "tqdm", "matplotlib") if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Missing offline packages: " + ", ".join(missing))
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU accelerator for full ViT-L/16 training")
if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    raise RuntimeError("A recent compatible PyTorch with SDPA is required")
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("Python:", sys.version.split()[0], "| PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
''', "preflight")
    add("markdown", '''## 2. Input paths và ngân sách

Input bắt buộc: ISIC-2017 **Part1 segmentation** với official train/val/test và mask; local DINOv3 source;
checkpoint **dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth**. Dataset được kiểm đủ 2.000/150/600 ảnh và ghép mask theo ISIC ID.
Hỗ trợ `train/images,masks`, `val/images,masks`, `test/images,masks` hoặc các thư mục gốc `ISIC-2017_Training_Data` / `...Part1_GroundTruth` tương ứng.
Để ISIC_ROOT rỗng sẽ tự tìm một root duy nhất. Nếu Input có nhiều bản, đặt đường dẫn chính xác.

Nếu có patient/lesion metadata, đặt GROUPS_CSV (cột image_id,group_id cho toàn bộ ảnh) để loại cùng bệnh nhân khỏi mọi pairing.
Không có metadata: bảo đảm loại self và pixel trùng, protocol ghi rõ chưa kiểm chứng việc loại mọi ảnh cùng bệnh nhân.
Không đoán patient ID từ số filename. Pool lấy 20% train (400 ảnh), vẫn là query train hợp lệ nhưng không tự chọn mình.

Effective batch=8, physical batch tùy VRAM. Reward chuẩn hóa trên cả effective batch trước optimizer step;
physical batch=1 không làm policy loss biến mất vì chuẩn hóa từng ảnh. AMP chỉ dùng train; eval/export FP32.
Giảm physical batch TRƯỚC lần train đầu nếu thiếu VRAM, không tự hạ backbone hay image size.
''', "configuration_intro")
    code = f'''PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("ISIC_{mode.upper()}_WEB_OUTPUT", str(PROJECT_DIR / "outputs/isic2017_{mode}_vitl16_448_web_v1")))
ISIC_ROOT = os.getenv("ISIC_ROOT", "")
GROUPS_CSV = os.getenv("ISIC_GROUPS_CSV", "")
DINOV3_REPO_DIR = Path(os.getenv("DINOV3_REPO_DIR", "/kaggle/input/datasets/nguyenducxu/dinov3-code/dinov3-main"))
DINOV3_WEIGHTS = Path(os.getenv("DINOV3_WEIGHTS", "/kaggle/input/datasets/sinhnguync/dinov3-vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
DEVICE = torch.device("cuda")
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
EPOCHS = int(os.getenv("ISIC_WEB_EPOCHS", "100"))
TRAIN_BATCH_SIZE = int(os.getenv("ISIC_WEB_BATCH", "4" if GPU_GIB >= 80 else "2" if GPU_GIB >= 40 else "1"))
EFFECTIVE_BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42
IMAGE_SIZE = 448
RUN_TRAINING = True
RUN_FINAL_TEST = True
RUN_EXPORT = True
print("Output:", OUTPUT_DIR)
print("Physical/effective batch:", TRAIN_BATCH_SIZE, EFFECTIVE_BATCH_SIZE, "| epochs:", EPOCHS)
'''
    if multimodal:
        code += '''
# Additional Inputs: complete original Microsoft Florence-2-base, full CLIP
# ViT-B/32, offline wheels (4.49, tokenizers 0.21.x, hub<1, einops/timm/accelerate).
FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
OFFLINE_ASSETS_ROOT = os.getenv("FLORENCE_OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
CAPTION_SETTINGS = {"batch_size": 8, "max_new_tokens": 64, "seed": SEED, "precision": "float32"}
CAPTION_OUTPUT = OUTPUT_DIR / "shared/caption_generation"
'''
    add("code", code, "configuration")
    add("markdown", "## 3. Ghi runtime được nhúng trong notebook\n\nKhông cần upload file `.py` từ repo. Source này cũng được đóng gói vào ZIP để inference dùng đúng kiến trúc đã train.", "runtime_intro")
    add("code", '''import json
EMBEDDED_SOURCES = ''' + repr(embedded_sources(mode)) + '''
RUNTIME_DIR = OUTPUT_DIR / "embedded_runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
for name, source in EMBEDDED_SOURCES.items():
    destination = RUNTIME_DIR / name
    if destination.is_file() and destination.read_text(encoding="utf-8") != source:
        raise ValueError("Runtime changed under this output. Keep the old run and choose another OUTPUT_DIR")
    destination.write_text(source, encoding="utf-8")
sys.path.insert(0, str(RUNTIME_DIR.resolve()))
import isic_segmentation_web as model_code
import train_isic_segmentation_web as workflow
for module in (model_code, workflow):
    if not Path(module.__file__).resolve().is_relative_to(RUNTIME_DIR.resolve()):
        raise RuntimeError("A different ISIC runtime is imported. Restart the kernel and Run All")
CONFIG = workflow.default_config("''' + mode + '''")
CONFIG.update(epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH_SIZE,
              workers=NUM_WORKERS, seed=SEED, image_size=IMAGE_SIZE,
              eval_batch_size=2 if GPU_GIB >= 40 else 1,
              feature_batch_size=4 if GPU_GIB >= 40 else 2)
workflow.validate_config(CONFIG)
workflow.seed_all(SEED)
display(CONFIG)
''', "runtime")
    add("markdown", "## 4. Xác minh official ISIC splits và tạo pool\n\nKhông random chia lại val/test. Mask resize dùng nearest-neighbor. ID hoặc pixel trùng giữa các split sẽ bị từ chối.", "dataset_intro")
    add("code", '''DINOV3_REPO_DIR, DINOV3_WEIGHTS = workflow.resolve_dino_assets(DINOV3_REPO_DIR, DINOV3_WEIGHTS, CONFIG)
FRAMES, CANDIDATE_POOL = workflow.load_isic(ISIC_ROOT, CONFIG, GROUPS_CSV)
print("Train/val/test/pool:", len(FRAMES["train"]), len(FRAMES["val"]), len(FRAMES["test"]), len(CANDIDATE_POOL))
print("Backbone:", DINOV3_WEIGHTS)
print("Patient groups:", "provided and checked" if GROUPS_CSV else "not supplied; self/identical-pixel exclusion only")
BACKBONE_SHA = model_code.io_utils.file_sha256(DINOV3_WEIGHTS)
ASSETS = {"backbone_sha256": BACKBONE_SHA, "dino_source": workflow.source_identity(DINOV3_REPO_DIR)}
''', "dataset")
    if multimodal:
        add("markdown", '''## 5. Tạo/resume captions.csv và CLIP text embeddings

Florence-2-base chạy task `<MORE_DETAILED_CAPTION>` từ pixels; bỏ boilerplate rồi lấy 5–8 từ.
Policy **isic_visible_attributes_v1 / Florence-Prefix** giữ cách tạo caption đã dùng cho ISIC trước đây:
lọc từ chẩn đoán/đánh giá ác tính trong toàn câu gốc, kiểm độ dài và token-limit. Không dùng ngưỡng
CLIP 0.30 để loại hết caption. CLIP ViT-B/32 chỉ encode văn bản hợp lệ thành vector 512 chiều.
Florence/CLIP dùng FP32 cả lúc tạo CSV lẫn adapter demo CPU/GPU; precision được khóa trong provenance.
Caption không hợp lệ vẫn lưu để kiểm tra nhưng vector/gate bằng 0. Đây là nhãn mô tả tự động, không phải GT của segmentation.

Mỗi ảnh hoàn thành có cache riêng được ghi nguyên tử; CSV đầy đủ ở `shared/caption_generation/captions.csv`.
Rerun cùng output sẽ tiếp tục phần còn thiếu; provenance khóa hashes ảnh/model/prompt/policy. Không cần tạo CSV bên ngoài.
Kernel Transformers 5.x được giữ nguyên; child 4.49 dùng wheels offline phù hợp Python của Kaggle.
''', "captions_intro")
        add("code", '''import isic_caption_web as caption_service
TEXT_CACHE_PATH, CAPTION_ASSETS = caption_service.prepare_captions(
    FRAMES, FLORENCE_MODEL_REF, CLIP_MODEL_LOCAL_PATH, OFFLINE_ASSETS_ROOT,
    CAPTION_OUTPUT, CAPTION_SETTINGS, RUNTIME_DIR)
TEXT_TABLE = workflow.TextTable(TEXT_CACHE_PATH)
coverage = []
for role, rows in list(FRAMES.items()) + [("candidate_pool", CANDIDATE_POOL)]:
    _, valid = TEXT_TABLE.gather(rows, "cpu")
    coverage.append({"split": role, "images": len(rows), "valid": int(valid.sum()),
                     "valid_fraction": float(valid.float().mean())})
display(coverage)
if coverage[0]["valid"] == 0 or coverage[-1]["valid"] == 0:
    raise ValueError("No usable query/candidate captions. Inspect CSV; do not silently call this full_both")
ASSETS.update(caption_csv_sha256=model_code.io_utils.file_sha256(CAPTION_OUTPUT / "captions.csv"),
              text_cache_sha256=model_code.io_utils.file_sha256(TEXT_CACHE_PATH),
              caption_settings_sha256=model_code.io_utils.file_sha256(CAPTION_OUTPUT / "web_caption_settings.json"))
torch.cuda.empty_cache()
''', "captions")
    else:
        add("code", "TEXT_TABLE = None\nCAPTION_ASSETS = None\nCAPTION_OUTPUT = None", "visual_only_inputs")
    add("markdown", "## 6. Khóa run identity và kiểm tra mô hình thật\n\nKiểm tra đủ 24 gated blocks; backbone frozen/eval; token/RoPE đúng native DINOv3; bật attention không thay logits.", "model_intro")
    add("code", '''RUN_BINDING = {"config": CONFIG, "assets": ASSETS, "runtime": workflow.source_identity(RUNTIME_DIR),
    "splits": {role: [{k: r[k] for k in ("image_id", "image_sha256", "mask_sha256", "group_id")} for r in rows]
               for role, rows in FRAMES.items()}, "pool_ids": [r["image_id"] for r in CANDIDATE_POOL]}
RUN_IDENTITY = workflow.object_hash(RUN_BINDING)
if (OUTPUT_DIR / "protocol.json").is_file() and model_code.io_utils.read_json(OUTPUT_DIR / "protocol.json")["identity"] != RUN_IDENTITY:
    raise ValueError("This output has another run/config. Use a fresh OUTPUT_DIR, or restore the exact original configuration")
model_code.io_utils.write_json(OUTPUT_DIR / "run_binding.json", RUN_BINDING)
for role, rows in FRAMES.items():
    workflow.csv_write(OUTPUT_DIR / f"{role}_split.csv", rows)
workflow.csv_write(OUTPUT_DIR / "candidate_pool.csv", CANDIDATE_POOL)
model = model_code.build_model(DINOV3_REPO_DIR, CONFIG, DINOV3_WEIGHTS).to(DEVICE).eval()
assert len(model.context_blocks) == len(model.backbone.blocks) == 24
assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
probe = torch.stack([model_code.io_utils.preprocess(row["image_path"], CONFIG) for row in CANDIDATE_POOL[:2]]).to(DEVICE)
with torch.no_grad():
    initial = model.prepare(probe)
    native = model.backbone.forward_features(probe)["x_norm_clstoken"]
    actual, _ = model.encode_selector_tokens(initial)
    assert torch.allclose(native, actual, atol=1e-4, rtol=1e-4)
    qt = qv = ct = cv = None
    if TEXT_TABLE is not None:
        text, valid = TEXT_TABLE.gather(CANDIDATE_POOL[:2], DEVICE)
        qt, qv, ct, cv = text[:1], valid[:1], text[1:], valid[1:]
    plain, _ = model.decode(initial[:1], initial[1:], qt, qv, ct, cv)
    explained, details = model.decode(initial[:1], initial[1:], qt, qv, ct, cv, return_attention=True)
    assert plain.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(plain).all()
    assert torch.allclose(plain, explained, atol=1e-5, rtol=1e-5)
    assert details["query_segmentor"].shape == (1, IMAGE_SIZE // 16, IMAGE_SIZE // 16)
print("Real DINOv3 + per-block segmentation/attention preflight passed")
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
del probe, initial, native, actual, plain, explained, details
torch.cuda.empty_cache()
workflow.seed_all(SEED)
''', "model")
    add("markdown", '''## 7. Frozen candidate bank

Bank lưu initial patch/prefix tokens và CLS features của ảnh pool deterministic. Không cache một context cuối rồi
chỉ nối sau backbone: cặp query/context vẫn đi qua từng block và từng gated cross-attention khi segment.
Bank tokens float16 được ghi rõ trong config, được dùng thống nhất ở train/eval/web; feature scoring float32.
ST backward tính đúng đạo hàm weight×bank theo từng chunk CPU→GPU, không tạo toàn pool image/token graph trên GPU.
''', "bank_intro")
    add("code", '''BANK = workflow.build_bank(model, CANDIDATE_POOL, CONFIG, OUTPUT_DIR / "shared/candidate_bank.pt", BACKBONE_SHA)
print("Pool IDs:", len(BANK["image_ids"]), "| tokens:", tuple(BANK["tokens"].shape), "| CLS:", tuple(BANK["features"].shape))
print("CPU token bank GiB:", BANK["tokens"].numel() * BANK["tokens"].element_size() / 1024**3)
''', "bank")
    add("markdown", '''## 8. Train/resume hybrid TACS

Task loss = BCE-with-logits + soft Dice. `L = L_task + 0.5 L_policy` (entropy regularizer mặc định 0).
Gumbel ST chọn một context cho đường task có gradient; categorical policy lấy action riêng và reward đúng ảnh/caption của action đó.
Reward = loss(query không context) − loss(query + context). M-TACS giữ nguyên caption query trong cả hai vế.
Advantage chuẩn hóa trên effective batch 8; log_prob tính cùng policy trước khi update.

Lưu best theo validation Dice @0.5; last chứa optimizer/scheduler/scaler/RNG để resume.
Không dừng sớm mặc định, không chọn threshold theo test. Nếu interruption, giữ cả thư mục OUTPUT_DIR và chạy lại
cùng config. best/last chỉ lưu phần học được; backbone pretrained bị freeze và được khóa bằng SHA256 trong run binding.
''', "training_intro")
    add("code", '''if RUN_TRAINING:
    HISTORY, BEST = workflow.run_training(model, FRAMES, CANDIDATE_POOL, BANK, CONFIG, OUTPUT_DIR, RUN_IDENTITY, TEXT_TABLE)
else:
    BEST = torch.load(OUTPUT_DIR / "best.pt", map_location="cpu", weights_only=True)
    if BEST["identity"] != RUN_IDENTITY:
        raise ValueError("Wrong checkpoint identity")
    model.restore_trainable(BEST["state"])
    HISTORY = torch.load(OUTPUT_DIR / "last.pt", map_location="cpu", weights_only=True)["history"]
print("Best validation epoch:", BEST["epoch"], "|", BEST["validation"])
''', "training")
    add("markdown", "## 9. Test best checkpoint và vẽ lịch sử\n\nMetric chính là mean per-image Dice/IoU nhị phân @0.5 tại 448×448. Không so trực tiếp một seed L/16-448 với trung bình S/16-256 mà thiếu chú thích.", "evaluation_intro")
    add("code", '''best_digest = model_code.io_utils.file_sha256(OUTPUT_DIR / "best.pt")
if RUN_FINAL_TEST:
    test_metrics, test_rows = workflow.evaluate(model, workflow.make_loader(FRAMES["test"], CONFIG),
        CANDIDATE_POOL, BANK, CONFIG, TEXT_TABLE, desc="Final test (best validation)")
    METRICS = {"dataset": "ISIC2017", "task": "binary_segmentation", "mode": CONFIG["mode"],
        "seed": SEED, "image_size": IMAGE_SIZE, "backbone": CONFIG["backbone"], "threshold": .5,
        "aggregation": "mean_per_image_at_network_resolution", "best_epoch": BEST["epoch"],
        "validation": BEST["validation"], "best_sha256": best_digest, "run_identity": RUN_IDENTITY,
        **{f"test_{k}": v for k, v in test_metrics.items()}}
    model_code.io_utils.write_json(OUTPUT_DIR / "metrics.json", METRICS)
    workflow.csv_write(OUTPUT_DIR / "test_predictions.csv", test_rows)
else:
    METRICS = model_code.io_utils.read_json(OUTPUT_DIR / "metrics.json")
    if METRICS["best_sha256"] != best_digest or METRICS["run_identity"] != RUN_IDENTITY:
        raise ValueError("Saved test metrics belong to another best checkpoint")
display(METRICS)
import matplotlib.pyplot as plt
figure, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, key in zip(axes, ("dice", "iou", "loss")):
    for role in ("train", "val"):
        axis.plot([r["epoch"] for r in HISTORY], [r[f"{role}_{key}"] for r in HISTORY], label=role)
    axis.set_title(key)
    axis.set_xlabel("Epoch")
    axis.legend()
figure.tight_layout()
figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
plt.show()
''', "evaluation")
    add("markdown", '''## 10. Export đầy đủ, cold reload và ZIP tải về

Cell này bắt buộc training đã kết thúc đủ ngân sách (hoặc patience nếu bạn đã cấu hình từ đầu).
Tách weights, dựng lại model mới, so logits/context với best trong RAM, kiểm attention không đổi mask,
chạy thêm process CPU offline từ chính bundle rồi mới ghi `status=ready` và ZIP.
M-TACS còn buộc sinh caption mới từ Florence/CLIP đã copy để tránh text cache che mất model thiếu.

Giữ vài GB trống cho bundle + ZIP; thao tác copy weights và kiểm SHA có thể lâu. CPU reload chỉ để kiểm export, không train lại.
ZIP là bộ model/runtime/data phục vụ demo; máy chạy web vẫn cần Python/PyTorch và dependency tương thích.
''', "export_intro")
    add("code", '''if RUN_EXPORT:
    BUNDLE_DIR, ZIP_FILE = workflow.export_bundle(model, FRAMES, CANDIDATE_POOL, BANK, CONFIG,
        OUTPUT_DIR, DINOV3_REPO_DIR, DINOV3_WEIGHTS, RUNTIME_DIR, METRICS,
        caption_assets=CAPTION_ASSETS, caption_output=CAPTION_OUTPUT)
    print("Verified bundle:", BUNDLE_DIR)
    print("Download this complete ZIP:", ZIP_FILE)
    from IPython.display import FileLink
    display(FileLink(str(ZIP_FILE)))
''', "export")
    add("markdown", '''## 11. Những file cần giữ

**Đưa lên web:** tải nguyên ZIP `exports/..._web_bundle.zip`, không chỉ best.pt.
Trong đó có `models/<official DINO checkpoint>`, `models/selector.pt`, `models/segmentor.pt`,
`candidate_bank.pt`, `candidate_pool.json`, `candidate_images/`, `ground_truth_masks/`, `ground_truth_index.json`,
`runtime/`, `vendor/dinov3/`, `config.json`, `manifest.json`, metrics và ví dụ.
M-TACS thêm `caption_models/florence/`, `caption_models/clip/`, `captions.csv`, `text_embeddings.pt`,
`caption_lookup.json`, `caption_settings.json`, provenance và offline wheels tìm được.

**Resume train:** giữ toàn OUTPUT_DIR, nhất là best.pt, last.pt, protocol.json, run_binding.json, embedded_runtime và shared/.
Deployment ZIP không chứa optimizer. Tải lại notebook đang chạy kèm output/curves để làm bằng chứng báo cáo.

**Attention:** đã nằm trong kiến trúc/weights hiện có, runtime tính theo từng ảnh; không có model attention giả hoặc cần train thêm.
Mask/caption/heatmap chỉ được xuất khi tính thật thành công. Dữ liệu y tế và model/source giữ nguyên điều kiện sử dụng của chủ sở hữu.
''', "download_guide")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for mode, target in TARGETS.items():
        notebook = build_notebook(mode)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Created {target} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
