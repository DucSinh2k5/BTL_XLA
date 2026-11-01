"""Build two standalone SUN397 best.pt-to-web notebooks without training."""
from __future__ import annotations

import ast
import json

from _project_bootstrap import PROJECT_ROOT
from build_sun397_tacs_web_notebook import cell


FOLDER = PROJECT_ROOT / "notebooks/04_deployment/sun397/03_export_only"
TARGETS = {
    "tacs": FOLDER / "sun397-tacs-visual-vitl16-export-only.ipynb",
    "mtacs": FOLDER / "sun397-mtacs-full-both-vitl16-export-only.ipynb",
}
EXPORT_SOURCE = PROJECT_ROOT / "src/deployment/sun397_export_only.py"
SOURCE_PATHS = {
    "tacs_visual_web.py": PROJECT_ROOT / "src/models/tacs_visual_web.py",
    "train_sun397_tacs_web.py": PROJECT_ROOT / "src/training/train_sun397_tacs_web.py",
    "mtacs_multimodal_web.py": PROJECT_ROOT / "src/models/mtacs_multimodal_web.py",
    "sun397_caption_web.py": PROJECT_ROOT / "src/captioning/sun397_caption_web.py",
    "train_sun397_mtacs_web.py": PROJECT_ROOT / "src/training/train_sun397_mtacs_web.py",
}


def build_notebook(variant):
    if variant not in TARGETS:
        raise ValueError(variant)
    multimodal = variant == "mtacs"
    label = "M-TACS full_both" if multimodal else "TACS visual"
    slug = "mtacs_full_both" if multimodal else "tacs_visual"
    names = list(SOURCE_PATHS) if multimodal else ["tacs_visual_web.py", "train_sun397_tacs_web.py"]
    payload = {name: SOURCE_PATHS[name].read_text(encoding="utf-8") for name in names}
    exporter_source = EXPORT_SOURCE.read_text(encoding="utf-8")
    for source in list(payload.values()) + [exporter_source]:
        ast.parse(source)
    text_note = (
        "Attach the complete original shared/ folder: captions.csv, text_embeddings.pt, caption_protocol.json and caption_settings.json. "
        "Frozen Florence-2-base and CLIP ViT-B/32 weights must also be attached, plus compatible offline wheels when required. "
        "The full CSV is NOT regenerated. Candidate text vectors are sliced from the saved cache in exact candidate order. "
        "One fresh caption on the smoke-test image verifies the bundled novel-image pipeline. The two known inference-tensor masking bugs "
        "are fixed only in the exported worker source and logged in provenance; caption policy, existing CSV and vectors are not rewritten."
        if multimodal else
        "No captions, Florence, CLIP or text vectors are needed. There is no separately trained selector or attention model: "
        "the complete shared visual backbone and classifier are loaded from best.pt."
    )
    cells = [cell("markdown", f"""# SUN397 {label} L/16: export existing best.pt, no retraining

This notebook recovers a complete web demo bundle from the saved **{label} checkpoint**, not from a new training run. """
                  + """It never calls training, constructs an optimizer, loads last.pt, downloads models or invents training_complete.json. """
                  + """It builds the backbone with pretrained=False and strictly loads all weights from best.pt, so the original DINO pretrained .pth is unnecessary.

Required: the original saved run folder with best.pt, protocol.json, class_names.json, candidate_pool.csv, train_split.csv and validation_split.csv; SUN397 images; and the original DINOv3 code folder with license. Keep embedded_runtime/ or embedded_runtime_v2/ when available. Known embedded source is only a fallback when its inference hash matches the checkpoint protocol. A checkpoint from a different architecture is rejected, not silently converted. Production requires ViT-L/16, 384px, K=1 and 397 classes.

""" + text_note + """

**Time-saving defaults:** encode only the saved candidate pool once, do not resample it, and skip full test-set evaluation. RUN_FULL_TEST=False creates a clearly labeled checkpoint-snapshot demo; a one-image smoke test is not a test-set metric. Original test metrics, if present, are preserved as unverified reference files, not represented as a new evaluation. Enable RUN_FULL_TEST only when a new fp32 full-test run is desired.

The output directory is separate. Original weights, source, CSV, cache, pool and protocol are read-only. Interrupted exports retain verified model/bank stages for retry. A cold offline process reloads the bundle, checks predictions against best.pt, and computes both actual visual attention maps before ZIP publication. No extra attention model is required. The exported adapter is ready to be wired into the existing FastAPI web; this notebook does not modify that app.

Restart the kernel before running this notebook after a training notebook to avoid importing another run's modules. Only load trusted checkpoint/code files. No Internet or global Transformers downgrade is needed. Offline caption dependencies may be prepared in an isolated child environment for M-TACS. Kaggle Linux wheels are not Windows installers; the web host still needs compatible Python/PyTorch/dependencies. Keep original model/dataset license terms.
"""),
        cell("markdown", "## 1 Offline preflight\nNo project Python upload is required: the exporter and known fallback source are embedded below."),
        cell("code", '''import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import sys
import importlib.util
from pathlib import Path
missing = [name for name in ("torch", "numpy", "PIL", "tqdm") if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Missing offline packages: " + ", ".join(missing))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
'''),
        cell("markdown", """## 2 Point to the existing run, not a new training output

SOURCE_RUN_DIR must be the exact folder containing best.pt and its sidecars, not the Kaggle Input root and not a deploy ZIP. Edit YOUR_SAVED_RUN_FOLDER below after attaching the saved Output as an Input. You may also set it to the original /kaggle/working/outputs/... folder in the same session after restarting the kernel. SOURCE_RUN_DIR can be best.pt itself when its sidecars are in the same directory.

Attach the dataset and DINO source, but no DINO pretrained weights are needed. FEATURE_BATCH_SIZE=8 is an inference batch for candidate encoding, not a training batch. EXAMPLE_IMAGE=None selects a nonduplicate saved validation image; an explicit external image can be used for a demo smoke test. Leave RUN_FULL_TEST=False for the fastest export. Output must be outside the original run.
"""),
        cell("code", f'''VARIANT = {variant!r}
SOURCE_RUN_DIR = Path(os.getenv("SUN397_{variant.upper()}_SAVED_RUN", "/kaggle/input/YOUR_SAVED_RUN_FOLDER"))
DATASET_ROOT = Path(os.getenv("SUN397_ROOT", "/kaggle/input/datasets/hiuphmnhtrung/sun397/SUN397/SUN397"))
DINOV3_REPO_DIR = Path(os.getenv("DINOV3_REPO_DIR", "/kaggle/input/datasets/sinhnguync/dinov3-code/dinov3-main"))
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("SUN397_{variant.upper()}_EXPORT_ONLY_OUTPUT", str(PROJECT_DIR / "exports_sun397_{slug}_vitl16")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_BATCH_SIZE = 8
NUM_WORKERS = min(4, os.cpu_count() or 1)
EXAMPLE_IMAGE = None
RUN_FULL_TEST = False
''' + ('''FLORENCE_MODEL_DIR = Path(os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base"))
CLIP_MODEL_DIR = Path(os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit"))
OFFLINE_ASSETS_ROOT = Path(os.getenv("OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi"))
''' if multimodal else "") + '''print("Read-only source:", SOURCE_RUN_DIR)
print("New export output:", OUTPUT_DIR)
print("Device:", DEVICE, "| full test:", RUN_FULL_TEST)
'''),
        cell("markdown", "## 3 Embedded exporter\nImporting the original training helpers only defines functions; no training or full-caption generation function is called."),
        cell("code", '''EXPORT_CODE_DIR = OUTPUT_DIR / "notebook_exporter"
EXPORT_CODE_DIR.mkdir(parents=True, exist_ok=True)
EXPORTER_SOURCE = ''' + repr(exporter_source) + '''
FALLBACK_SOURCES = ''' + repr(payload) + '''
exporter_path = EXPORT_CODE_DIR / "sun397_export_only.py"
if exporter_path.is_file() and exporter_path.read_text(encoding="utf-8") != EXPORTER_SOURCE:
    raise ValueError("Exporter source changed under this output; preserve it and choose a new OUTPUT_DIR")
if not exporter_path.is_file():
    exporter_path.write_text(EXPORTER_SOURCE, encoding="utf-8")
existing = sys.modules.get("sun397_export_only")
if existing is not None and Path(existing.__file__).resolve() != exporter_path.resolve():
    raise RuntimeError("Another exporter is imported; restart the kernel")
sys.path.insert(0, str(EXPORT_CODE_DIR.resolve()))
import sun397_export_only as exporter
'''),
        cell("markdown", """## 4 Validate snapshot, original pool and source identity

This checks checkpoint/config/class order, the saved split/pool identity, the original DINO source, and the exact inference runtime. It does not need last.pt or a completion marker. A present but mismatched completion marker is an error, not something to overwrite. M-TACS additionally validates original caption hashes, masks, image bindings and frozen text assets. No random replacement pool, substitute captions or partial state_dict load is allowed.
"""),
        cell("code", '''CONTEXT = exporter.prepare(
    SOURCE_RUN_DIR, DATASET_ROOT, DINOV3_REPO_DIR, OUTPUT_DIR, VARIANT, FALLBACK_SOURCES,
    device=DEVICE, feature_batch_size=FEATURE_BATCH_SIZE, workers=NUM_WORKERS,
    example_image=EXAMPLE_IMAGE, run_full_test=RUN_FULL_TEST,
''' + ('''    florence=FLORENCE_MODEL_DIR, clip=CLIP_MODEL_DIR, offline_assets_root=OFFLINE_ASSETS_ROOT,
''' if multimodal else "") + ''')
display({"variant": CONTEXT["variant"], "best_epoch": CONTEXT["checkpoint"]["epoch"],
         "candidate_count": len(CONTEXT["candidates"]), "completion": CONTEXT["completion"],
         "run_full_test": RUN_FULL_TEST})
'''),
        cell("markdown", """## 5 Build the missing model and bank artifacts

Strictly load the full best weights without a pretrained DINO .pth, copy only the saved pool, encode its images once in fp32, and bind every feature to model.pt and exact image order. Verified completed stages are reused on retry. No optimizer or training loop is involved. M-TACS slices the original CLIP cache for candidate_text.pt, copies the existing CSV/vectors and complete frozen caption models. It never calls full caption regeneration.

If RUN_FULL_TEST=False, metrics.json explicitly says not_run_export_only; it does not invent scores. Optional full test uses the saved test split and current snapshot, not test-time training. Training completion is not fabricated for export.
"""),
        cell("code", '''CONTEXT = exporter.build(CONTEXT)
print("Bundle directory (staging until verified):", CONTEXT["stage"])
display(exporter.json_read(CONTEXT["stage"] / "metrics.json"))
'''),
        cell("markdown", """## 6 Cold reload, actual attention and ZIP

A fresh process loads only the staged bundle. It verifies logits with/without attention and matches the original loaded best checkpoint. M-TACS also forces one fresh Florence/CLIP caption using the bundled weights, so novel-image inference is exercised. Failures preserve the staged files for inspection/retry; an incomplete bundle is not published as ready.
"""),
        cell("code", '''EXPORT_ZIP = exporter.verify_and_publish(CONTEXT)
print("Download:", EXPORT_ZIP)
'''),
        cell("markdown", "## 7 Inspect the exported files and evidence\nDownload the complete ZIP, not only model.pt. Keep the original best/run folder separately for provenance or future training resume."),
        cell("code", '''import base64
from IPython.display import FileLink, Image as DisplayImage, display
BUNDLE = EXPORT_ZIP.with_suffix("")
for name in ("model.pt", "candidate_features.pt", "candidate_text.pt"):
    path = BUNDLE / name
    if path.is_file():
        print(name, "|", round(path.stat().st_size / 1024**2, 2), "MiB")
display(exporter.json_read(BUNDLE / "smoke_test.json"))
display(exporter.json_read(BUNDLE / "metrics.json"))
result = exporter.json_read(BUNDLE / "examples/prediction.json")
display(result["prediction"])
display(result["evidence"]["query_caption"])
display(result["evidence"]["candidate"] | {"image_url": "shown below"})
display(DisplayImage(data=base64.b64decode(result["evidence"]["candidate"]["image_url"].split(",", 1)[1])))
for role, value in result["evidence"]["attention"].items():
    print(role, "|", value["method"], "|", value["status"])
    display(DisplayImage(data=base64.b64decode(value["image_url"].split(",", 1)[1])))
display(FileLink(str(EXPORT_ZIP)))
''')]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"},
                         "tacs": {"purpose": "export_only", "dataset": "SUN397", "variant": variant,
                                  "backbone": "dinov3_vitl16", "training_called": False, "default_full_test": False}}}


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for variant, path in TARGETS.items():
        path.write_text(json.dumps(build_notebook(variant), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("Created:", path)


if __name__ == "__main__":
    main()
