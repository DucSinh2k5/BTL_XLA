"""Build standalone SUN397 visual TACS ViT-L/16 training plus web deployment export."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from _project_bootstrap import PROJECT_ROOT


TARGET = PROJECT_ROOT / "notebooks/04_deployment/sun397/01_tacs_visual/sun397-tacs-visual-vitl16-train-export-web.ipynb"
MODEL_SOURCE = PROJECT_ROOT / "src/models/tacs_visual_web.py"
TRAINING_SOURCE = PROJECT_ROOT / "src/training/train_sun397_tacs_web.py"


def cell(kind, source):
    source = source.strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
              "id": hashlib.sha256(source.encode()).hexdigest()[:12]}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    payload = {"tacs_visual_web.py": MODEL_SOURCE.read_text(encoding="utf-8"),
               "train_sun397_tacs_web.py": TRAINING_SOURCE.read_text(encoding="utf-8")}
    for source in payload.values():
        ast.parse(source)
    cells = [cell("markdown", '''# SUN397 TACS visual ViT L16 training and complete web export

Notebook 1 of the SUN397 web deployment series. Train the **image-only TACS** model, evaluate the best validation checkpoint, and export a verified local web bundle. **DINOv3 ViT-L/16 only, 1024 dimensions, 384 pixels, K=1.** No S/16 checkpoint is loaded or converted. No captions, CLIP, Florence, text gates or context-label prior are used.

The visual core follows TACS: one shared backbone for selector/task, CLS dot-product utility over a fixed 20% training pool, patch concatenation before the task transformer blocks, hard Gumbel straight-through task gradients plus categorical policy reward. This is a declared L/16 adaptation, not exact reproduction of the paper's S/16, 256px, 3-seed results. Here: SUN partition 01; stratified 10% validation held out from official train; seed 42; 100 epochs by default; minimum validation CE checkpoint. Batch sizes adapt to VRAM, with gradient accumulation to 64. A detached candidate-feature bank is refreshed at epoch start and again before validation.

Attach **SUN397 images + ClassName.txt + Training_01.txt + Testing_01.txt**, **local DINOv3 repository with hubconf.py and license files**, and **DINOv3 ViT-L/16 pretrained .pth**. No project Python files need to be uploaded: the runtime/training source is embedded below. No pip installation or Internet request is made by notebook cells. Configure the paths in section 2 before Run All.

The final ZIP contains full best weights, exact class order, all pool images, pool manifest, checkpoint-bound feature bank, vendored DINOv3 code, preprocessing, inference CLI/Python adapter, environment, metrics, predictions, SHA256 checks and example real attention outputs. It is usable without the original Kaggle Inputs after extraction, in a compatible Python/PyTorch environment. The current FastAPI web still needs to import this adapter; this notebook does not silently modify the web.

**Attention is computed by the trained backbone, not a separate model.** Query map: last-block classifier CLS attention to query patches while receiving context. Candidate map: last-block selector encoder CLS attention to the selected image. Mean over heads; raw grids and display overlays are included. These maps are not Grad-CAM, not class-conditioned causal explanations, and not a claim to reproduce every Figure 4 visualization detail. The no-context comparison model is not separately trained here.

The export cell refuses incomplete training, a changed best checkpoint, mismatched source/config/pool, failed cold offline reload, changed predictions when attention is enabled, or failed logits parity. It never fills fake predictions or heatmaps. Keep `last.pt` plus this run's `best.pt` to resume; do not alter the epoch horizon or pool under the same output directory. Test is run after training; no test-based selection. The final export is a private deployment artifact subject to original model/dataset license terms.

References: [TACS](https://arxiv.org/abs/2512.00489), [DINOv3 official source](https://github.com/facebookresearch/dinov3).'''),
        cell("markdown", "## 1 Offline environment check\nSet tokenizer parallelism before any tokenizer or DataLoader worker. This visual notebook does not import Transformers."),
        cell("code", '''import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import importlib.util
import sys
from pathlib import Path
if sys.version_info < (3, 10):
    raise RuntimeError("Python >=3.10 is required")
missing = [name for name in ("torch", "numpy", "PIL", "tqdm") if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Missing packages: " + ", ".join(missing) + ". Attach compatible offline wheels or prepare the environment first.")
import torch
if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    raise RuntimeError("Use a compatible recent PyTorch with scaled_dot_product_attention")
if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
    raise RuntimeError("Use a recent GPU PyTorch with torch.amp.GradScaler support")
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before this full ViT-L/16 training run")
print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))'''),
        cell("markdown", '''## 2 Paths and training budget

Change paths here when your Kaggle Input owner/slug differs. `DINOV3_WEIGHTS` may be one exact .pth file or a folder containing exactly one ViT-L/16 .pth. Use the official LVD1689M `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` with its original filename, not S/16, B/16, SAT or L/16-plus. Set EPOCHS before the first run; changing 100 to 50 during resume is rejected because it changes the scheduler and run identity.

Defaults require about 3.2 GB CPU RAM for pool pixels plus RAM/VRAM for the L/16 model, optimizer and workers. Exact memory and runtime depend on the device. Start with the VRAM batch policy below and reduce TRAIN_BATCH_SIZE before starting a new run if necessary; keep it at least 2 for within-batch policy advantage. Export also needs free disk for best weights, pool images and ZIP. No automatic change of backbone or image size occurs.'''),
        cell("code", '''DATASET_ROOT = Path(os.getenv("SUN397_ROOT", "/kaggle/input/datasets/hiuphmnhtrung/sun397/SUN397/SUN397"))
TRAIN_PARTITION = Path(os.getenv("SUN397_TRAIN_PARTITION", "/kaggle/input/datasets/hiuphmnhtrung/sun397/sun397_other/Training_01.txt"))
TEST_PARTITION = Path(os.getenv("SUN397_TEST_PARTITION", "/kaggle/input/datasets/hiuphmnhtrung/sun397/sun397_other/Testing_01.txt"))
DINOV3_REPO_DIR = Path(os.getenv("DINOV3_REPO_DIR", "/kaggle/input/datasets/nguyenducxu/dinov3-code/dinov3-main"))
DINOV3_WEIGHTS = Path(os.getenv("DINOV3_WEIGHTS", "/kaggle/input/datasets/sinhnguync/dinov3-vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("SUN397_TACS_WEB_OUTPUT", str(PROJECT_DIR / "outputs/sun397_tacs_visual_vitl16_web_v1")))
DEVICE = torch.device("cuda")
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
TRAIN_BATCH_SIZE = int(os.getenv("SUN397_TACS_WEB_BATCH", "8" if GPU_GIB >= 80 else "4" if GPU_GIB >= 40 else "2"))
EPOCHS = int(os.getenv("SUN397_TACS_WEB_EPOCHS", "100"))
SEED = 42
IMAGE_SIZE = 384
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = int(os.getenv("SUN397_TACS_WEB_WORKERS", "0"))
RUN_TRAINING = True
RUN_FINAL_TEST = True
RUN_EXPORT = True
print("Output:", OUTPUT_DIR, "| train batch:", TRAIN_BATCH_SIZE, "| epochs:", EPOCHS)
print("Original paper: S/16, 256. This declared deployment adaptation: L/16, 384.")'''),
        cell("markdown", "## 3 Materialize the embedded runtime\nOnly this notebook is needed on Kaggle. The same model/inference source is copied into the final bundle."),
        cell("code", '''import hashlib
import json
RUNTIME_DIR = OUTPUT_DIR / "embedded_runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_SOURCES = ''' + repr(payload) + '''
for filename, source in EMBEDDED_SOURCES.items():
    path = RUNTIME_DIR / filename
    if path.is_file() and path.read_text(encoding="utf-8") != source:
        raise ValueError("Runtime source changed in this output directory; preserve it and start a new run")
    path.write_text(source, encoding="utf-8")
sys.path.insert(0, str(RUNTIME_DIR.resolve()))
import tacs_visual_web as web_runtime
import train_sun397_tacs_web as workflow
if Path(web_runtime.__file__).resolve() != (RUNTIME_DIR / "tacs_visual_web.py").resolve():
    raise RuntimeError("Another runtime is already imported. Restart the kernel and Run All.")
CONFIG = workflow.default_config()
CONFIG.update(epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH_SIZE,
              workers=NUM_WORKERS, seed=SEED, image_size=IMAGE_SIZE,
              feature_batch_size=16 if GPU_GIB >= 40 else 8, eval_batch_size=8 if GPU_GIB >= 40 else 4)
workflow.validate_config(CONFIG)
workflow.seed_all(SEED)
display(CONFIG)'''),
        cell("markdown", '''## 4 Validate local assets and official SUN397 splits

Expected partition 01: 19,850 official train and 19,850 test. Hold out 1,985 stratified validation images from official train, retain 17,865 training queries; a fixed 3,573-image pool is sampled from actual train only. Pool images can also be training queries, with exact self-ID masking. ClassName.txt order is authoritative. Missing images stop the run instead of changing the dataset silently.'''),
        cell("code", '''for path in (DATASET_ROOT, TRAIN_PARTITION, TEST_PARTITION, DINOV3_REPO_DIR):
    if not path.exists():
        raise FileNotFoundError(f"Required Input not found: {path}. Correct section 2 before proceeding.")
WEIGHTS_FILE = workflow.resolve_weights(DINOV3_WEIGHTS)
ASSETS = {
    "pretrained_weights_sha256": web_runtime.file_sha256(WEIGHTS_FILE),
    "dino_source_sha256": workflow.source_tree_identity(DINOV3_REPO_DIR),
    "train_partition_sha256": web_runtime.file_sha256(TRAIN_PARTITION),
    "test_partition_sha256": web_runtime.file_sha256(TEST_PARTITION),
}
class_names, train_rows, val_rows, test_rows, candidate_rows = workflow.load_sun397(
    DATASET_ROOT, TRAIN_PARTITION, TEST_PARTITION, CONFIG)
print("Train / val / test / pool:", len(train_rows), len(val_rows), len(test_rows), len(candidate_rows))
print("Backbone weights:", WEIGHTS_FILE)
assert len(class_names) == 397
workflow.csv_write(OUTPUT_DIR / "test_split.csv", test_rows)
web_runtime.write_json(OUTPUT_DIR / "environment.json", workflow.environment())'''),
        cell("markdown", '''## 5 Construct L16 and check token positions and real attention

Check solo selector features against the official DINOv3 forward_features implementation. Check pair logits with and without attention capture. For 384/16, both raw maps must have a 24 by 24 grid. These operations do not update weights. No S/16 fallback, missing-RoPE fallback or feature-averaging fallback is permitted.'''),
        cell("code", '''model = web_runtime.build_model(DINOV3_REPO_DIR, CONFIG, WEIGHTS_FILE).to(DEVICE)
model.eval()
probe = torch.stack([web_runtime.preprocess(row["image_path"], CONFIG) for row in candidate_rows[:2]]).to(DEVICE)
with torch.no_grad():
    official_features = model.backbone.forward_features(probe)["x_norm_clstoken"]
    selector_features = model.encode_selector(probe)
    if not torch.allclose(official_features, selector_features, atol=1e-4, rtol=1e-4):
        raise RuntimeError("Selector token/RoPE path does not match official DINOv3")
    plain = model(probe[:1], probe[1:])
    explained, pair_maps = model(probe[:1], probe[1:], return_attention=True)
    _, selector_map = model.encode_selector(probe[1:], return_attention=True)
    if not torch.allclose(plain, explained, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Capturing attention changed classifier logits")
    expected_grid = (IMAGE_SIZE // 16, IMAGE_SIZE // 16)
    assert tuple(pair_maps["query_classifier"].shape[-2:]) == expected_grid
    assert tuple(selector_map.shape[-2:]) == expected_grid
    assert torch.isfinite(plain).all() and torch.isfinite(selector_map).all()
    scores, _ = model.score(probe, selector_features, excluded_positions=[0, 1])
    assert scores.argmax(1).tolist() == [1, 0]
print("L/16 dimensions:", model.backbone.embed_dim, "| blocks:", len(model.backbone.blocks))
print("Real attention method:", web_runtime.ATTENTION_METHOD, "| grid:", expected_grid)
del probe, official_features, selector_features, plain, explained, pair_maps, selector_map, scores
torch.cuda.empty_cache()
workflow.seed_all(SEED)'''),
        cell("markdown", '''## 6 Train and resume only this run

Both task straight-through gradients and categorical policy gradients update the shared selector backbone. CPU pixel banking uses a chunked exact derivative of hard selection weights multiplied by the fixed pixel bank; it is not an index-only estimator. Policy rewards compare detached query-only and paired CE under the same eval-mode task network. No separate no-context classifier is trained.

Best uses minimum validation CE, not test accuracy. Last checkpoint includes optimizer, scheduler, scaler, RNG and loader generator states. Interrupts preserve the last completed epoch; an interrupted partial epoch is replayed. The final incomplete accumulation group is scaled correctly. With multiple workers, exact augmentation replay is environment-dependent; default workers=0 is simplest for resume.

A new Kaggle session may have empty /kaggle/working. Attach the previous saved Output and restore the full run folder to OUTPUT_DIR, including last.pt, best.pt, protocol.json and embedded_runtime, before resuming. The deployment ZIP alone is not a training-resume checkpoint.'''),
        cell("code", '''if RUN_TRAINING:
    BEST, HISTORY = workflow.train_run(model, class_names, train_rows, val_rows, candidate_rows,
                                       CONFIG, ASSETS, OUTPUT_DIR, DEVICE)
else:
    completion = web_runtime.read_json(OUTPUT_DIR / "training_complete.json")
    BEST = torch.load(OUTPUT_DIR / "best.pt", map_location="cpu", weights_only=True)
    expected = workflow.run_identity(CONFIG, class_names, {"train": train_rows, "val": val_rows, "pool": candidate_rows}, ASSETS)
    if completion["identity"] != expected or BEST["identity"] != expected:
        raise ValueError("Completed run does not match this notebook/configuration")
    model.load_state_dict(BEST["model_state_dict"], strict=True)
    model.eval()
print("Best validation epoch:", BEST["epoch"], "| validation:", BEST["val_metrics"])'''),
        cell("markdown", '''## 7 Locked final test from best weights

Rebuild candidate features after reloading best, then use float32 raw-image evaluation exactly as the portable runtime does. This final test can differ slightly from AMP validation. Export records precision explicitly. Candidate labels are diagnostics only, never selector/classifier inputs.'''),
        cell("code", '''if not RUN_FINAL_TEST:
    raise RuntimeError("Set RUN_FINAL_TEST=True to produce the evaluated deployment bundle")
completion = web_runtime.read_json(OUTPUT_DIR / "training_complete.json")
if completion["identity"] != BEST["identity"] or completion["epochs_completed"] != CONFIG["epochs"]:
    raise RuntimeError("Complete this run's training before final test/export")
model.load_state_dict(BEST["model_state_dict"], strict=True)
model.eval()
FINAL_BANK = workflow.refresh_bank(model, candidate_rows, CONFIG, DEVICE, amp=False)
TEST_METRICS, TEST_PREDICTIONS = workflow.evaluate(
    model, workflow.make_loader(test_rows, CONFIG), FINAL_BANK, None, candidate_rows, CONFIG, DEVICE,
    label="Official SUN397 test / best checkpoint / fp32", amp=False)
web_runtime.write_json(OUTPUT_DIR / "test_metrics.json", {**TEST_METRICS, "best_epoch": BEST["epoch"], "precision": "float32"})
workflow.csv_write(OUTPUT_DIR / "test_predictions.csv", TEST_PREDICTIONS)
display(TEST_METRICS)
del FINAL_BANK
torch.cuda.empty_cache()'''),
        cell("markdown", '''## 8 Export complete bundle and verify a cold offline reload

The export copies every candidate image, the exact source and license notices, full best weights and aligned bank. A new Python process loads only the staged bundle (pretrained=False), checks that DINOv3 came from vendor/, predicts a held-out validation example, computes both real attention maps and confirms attention does not alter logits. The notebook compares its own best-model selection/logits to the reloaded bundle, verifies file hashes and ZIP CRC before publishing the ZIP.

Keep `last.pt` for training resume. It is intentionally not inside the deployment ZIP; neither optimizer state nor a second pretrained DINO file is needed for inference. Rerunning this cell reuses a complete ZIP only after verifying its hashes and binding to this run's best checkpoint. A partial or mismatched export is rejected and preserved for inspection.'''),
        cell("code", '''if RUN_EXPORT:
    EXPORT_ZIP = workflow.export_bundle(model, BEST, candidate_rows, class_names, CONFIG, DINOV3_REPO_DIR,
                                        OUTPUT_DIR, val_rows[0], TEST_METRICS, TEST_PREDICTIONS)
    print("Download this complete ZIP:", EXPORT_ZIP)
else:
    raise RuntimeError("Training finished, but no deploy bundle was created. Set RUN_EXPORT=True and run this section.")'''),
        cell("markdown", '''## 9 Inspect actual query and candidate evidence and download

These are generated by the reloaded trained model, not placeholder maps. A flat attention map is flagged instead of replaced. Attention highlights locations, not textual names of features. This is one shared trained network with two attention readouts, not an additional model that needs separate training.'''),
        cell("code", '''import base64
from IPython.display import FileLink, Image as DisplayImage, display
bundle_dir = EXPORT_ZIP.with_suffix("")
smoke = web_runtime.read_json(bundle_dir / "smoke_test.json")
result = web_runtime.read_json(bundle_dir / "examples/prediction.json")
display(smoke)
display(result["prediction"])
for role in ("query_classifier", "candidate_selector"):
    item = result["evidence"]["attention"][role]
    print(role, "|", item["status"], "|", item["method"])
    display(DisplayImage(data=base64.b64decode(item["image_url"].split(",", 1)[1])))
display(FileLink(str(EXPORT_ZIP)))
print("Also available in Kaggle Output under:", EXPORT_ZIP.relative_to(PROJECT_DIR) if EXPORT_ZIP.is_relative_to(PROJECT_DIR) else EXPORT_ZIP)
print("Next step: connect BundlePredictor to the existing FastAPI TACS page; M-TACS is notebook 2, not part of this run.")'''),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python", "version": "3.10"},
                                          "tacs": {"dataset": "SUN397", "purpose": "train_export_web", "variant": "visual", "backbone": "dinov3_vitl16"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
