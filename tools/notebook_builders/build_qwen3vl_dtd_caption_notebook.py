"""Build one standalone offline DTD notebook for Qwen3-VL 8B and 32B captions."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import json


TARGET = PROJECT_ROOT / "notebooks/02_improvements/caption_generation/qwen3_vl/dtd/qwen3-vl-dtd-captions-8b-32b.ipynb"


def definition(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.get_source_segment(source, node)
    raise ValueError(f"Cannot embed {name}")


def embedded_core():
    folder = PROJECT_ROOT / "src/captioning"
    core = (folder / "dtd_qwen_caption.py").read_text(encoding="utf-8")
    sources = {name: (folder / name).read_text(encoding="utf-8") for name in (
        "dtd_caption_finetune.py", "dtd_caption_screening.py", "dtd_grounded_caption.py")}
    tree = ast.parse(core)
    replacements = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module not in {
                "dtd_caption_finetune", "dtd_caption_screening", "dtd_grounded_caption"}:
            continue
        replacement = "\n\n".join(definition(sources[node.module + ".py"], alias.name) for alias in node.names)
        replacements.append((node.lineno - 1, node.end_lineno, replacement))
    lines = core.splitlines(keepends=True)
    for first, last, replacement in reversed(replacements):
        lines[first:last] = [replacement + "\n"]
    result = "".join(lines).replace("import csv\n", "import csv\nfrom collections import defaultdict\nfrom pathlib import PurePosixPath\n", 1)
    ast.parse(result)
    if "from dtd_" in result:
        raise ValueError("A project-local dependency escaped notebook embedding")
    return result


def cell(kind, source):
    source = source.strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(outputs=[], execution_count=None)
    return result


def build_notebook():
    core = embedded_core()
    worker = core + '\n\nif __name__ == "__main__":\n    worker_main(sys.argv[1], sys.argv[2])\n'
    cells = [
        cell("markdown", """
# DTD Captions: Qwen3-VL-8B-Instruct vs Qwen3-VL-32B-Instruct

Generate two new full-dataset caption CSVs, then train the existing **M-TACS notebook 05 full_both**
separately with each CSV. This notebook performs **frozen inference only**, not Qwen fine-tuning or classifier training.
Both methods use the same fixed prompt, image pixel budget, greedy decoding, dtype, seed and `dtd_lexical_v2` policy.
No CLIP image-text reranking, caption retries or class-conditioned repair is added. This changes the captioner,
not the number of selected context images: downstream **K remains 1**.

DTD split 1 uses the official `labels/train1.txt`, `val1.txt`, `test1.txt`: 1,880 images each, 5,640 total.
An image-derived caption is generated for every query/candidate image. The VLM receives only its image pixels
and the same English prompt, never an image ID, path, class label, class list, reference caption or test score.
The global 47-class vocabulary is used only AFTER generation for lexical checks and diagnostic CLIP scores.

Run all cells in order. 8B generation and audit finish before 32B starts; each stage runs in its own process,
which releases GPU memory on exit. Per-image atomic caches resume missing work after interruption.
Only complete, audited exports are named `captions.csv`; smoke tests/partial outputs cannot masquerade as complete files.
"""),
        cell("code", """
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import json
import sys
from pathlib import Path

print("Offline mode: no model downloads, API calls, pip installs or DataLoader/fork workers.")
"""),
        cell("code", """
# Set exact local folders when auto-discovery cannot find the attached checkpoints.
DTD_ROOT = Path(os.getenv("DTD_ROOT", "/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd"))
DTD_SPLIT = int(os.getenv("DTD_SPLIT", "1"))
QWEN_8B_MODEL_PATH = os.getenv("QWEN_8B_MODEL_PATH", "")
QWEN_32B_MODEL_PATH = os.getenv("QWEN_32B_MODEL_PATH", "")
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
SEARCH_ROOTS = [Path("/kaggle/input")]

# Optional attached site-packages folder containing modern Transformers AND compatible dependencies.
# Leave blank when the Kaggle kernel already supports Qwen3-VL. Florence's 4.49 runtime is NOT sufficient.
QWEN_RUNTIME_SITE_PACKAGES = os.getenv("QWEN_RUNTIME_SITE_PACKAGES", "")
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_ROOT = Path(os.getenv("DTD_QWEN_CAPTION_OUTPUT", str(
    PROJECT_DIR / "outputs" / f"qwen3vl_dtd_split{DTD_SPLIT}_captions_v1")))
RUN_8B = True
RUN_32B = True
MAX_NEW_IMAGES = 0  # 0 = full dataset; set 8 for a smoke test, then reset to 0 and rerun generation cells.

SETTINGS = {
    "batch_size": 1,  # Same conservative batch for BOTH models; increase before starting a new experiment.
    "dtype": "bfloat16",  # Use float16 for BOTH models if the GPU does not support BF16.
    "min_pixels": 256 * 256,
    "max_pixels": 512 * 512,
    "max_new_tokens": 48,
    "seed": 42,
    "audit_batch_size": 128,
    "semantic_threshold": 0.30,  # Diagnostic only, never a hard mask in dtd_lexical_v2.
}
METHOD_OUTPUTS = {
    "qwen3_vl_8b_instruct": OUTPUT_ROOT / "qwen3_vl_8b_instruct",
    "qwen3_vl_32b_instruct": OUTPUT_ROOT / "qwen3_vl_32b_instruct",
}
print("Output root:", OUTPUT_ROOT)
"""),
        cell("markdown", """
## Required Offline Inputs and Hardware

Attach the **complete original floating-point Instruct** folders for:
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-32B-Instruct`
- The same full CLIP ViT-B/32 checkpoint used by your DTD ablations, and the DTD dataset.

Set `QWEN_8B_MODEL_PATH` / `QWEN_32B_MODEL_PATH` to the folder containing `config.json`, all safetensors shards
and their index, tokenizer files, chat template, processor and image-preprocessor configuration. Automatic discovery
accepts only an unambiguous folder named for the requested Instruct model. Size is verified from safetensors headers;
the config alone cannot prove Instruct vs Thinking, so attach the correct official repository. All asset contents,
including templates and weights, are hashed into the saved protocol. Large-model hashing can take several minutes.

The worker needs PyTorch/torchvision, Pillow, numpy, tqdm, accelerate and **Transformers with Qwen3-VL support
(>=4.57)**. A compatible current Kaggle 5.x installation can be used. The original offline Florence bundle's
Transformers 4.49 wheels do NOT support Qwen3-VL. No package changes are made automatically. With Internet disabled,
attach compatible offline dependencies beforehand or set `QWEN_RUNTIME_SITE_PACKAGES` to a prepared runtime folder.
Do not point it at the legacy Florence runtime or unzip individual packages without their compatible dependencies.

Default inference is BF16 + SDPA, batch 1, no quantization, CPU offload or FlashAttention installation.
32B weights alone require roughly 60+ GiB at 16-bit precision; activations, image tokens and generation cache need
additional memory. An 80/96 GiB-class GPU is the intended starting point, not a guarantee of fitting every setting.
8B is lighter but a 16 GiB GPU can still be insufficient. Free other GPU models before Run All.
Changing dtype/resolution/prompt/settings/assets/runtime requires a NEW output root; do not mix protocols on resume.
"""),
        cell("code", core),
        cell("code", """
validate_settings(SETTINGS)
if type(MAX_NEW_IMAGES) is not int or MAX_NEW_IMAGES < 0:
    raise ValueError("MAX_NEW_IMAGES must be a nonnegative integer")
if not DTD_ROOT.is_dir():
    fallback = Path("/kaggle/input/describable-textures-dataset-dtd/dtd")
    if fallback.is_dir():
        DTD_ROOT = fallback
caption_records, class_names = load_caption_manifest(DTD_ROOT, DTD_SPLIT)
print("Hashing DTD images to lock the same input pixels for both captioners...")
image_fingerprints = fingerprint_records(caption_records)
for record, fingerprint in zip(caption_records, image_fingerprints):
    record["image_sha256"] = fingerprint["sha256"]
for role in ("train", "val", "test"):
    print(role, sum(row["split"] == role for row in caption_records))
print("Total:", len(caption_records), "| classes:", len(class_names))
print("Fixed VLM prompt:", CAPTION_PROMPT)
"""),
        cell("markdown", """
## Runtime Preflight and Isolated Workers

This preflight checks imports in the SAME Python environment used by the workers, before loading any weights.
If it fails on an old/missing library, attach the required offline runtime. Do not downgrade the kernel to 4.49.
The parent notebook does not import Transformers or keep either Qwen on GPU. No `qwen-vl-utils` package is needed:
PIL pixels are passed directly through the native processor; generated tokens are decoded after trimming the prompt.
"""),
        cell("code", """
worker_env = worker_environment(QWEN_RUNTIME_SITE_PACKAGES)
check_worker_runtime(worker_env)
clip_dir = resolve_clip_model(CLIP_MODEL_LOCAL_PATH, SEARCH_ROOTS)
print("CLIP diagnostic auditor:", clip_dir)
for method, reference in (("qwen3_vl_8b_instruct", QWEN_8B_MODEL_PATH),
                           ("qwen3_vl_32b_instruct", QWEN_32B_MODEL_PATH)):
    try:
        model_path = resolve_qwen_model(reference, method, SEARCH_ROOTS)
        _, parameters, stored_bytes = validate_qwen_folder(model_path, method)
        print(method, "|", model_path, "| stored parameters:", f"{parameters:,}",
              "| checkpoint GiB:", round(stored_bytes / 1024**3, 1))
    except (OSError, ValueError) as error:
        print(method, "not ready:", error)
print("A missing 32B input does not erase completed 8B outputs; fix its path before running the 32B cell.")
"""),
        cell("code", "QWEN_WORKER_SOURCE = " + repr(worker) + "\n" + '''

def run_caption_method(method, reference):
    model_path = resolve_qwen_model(reference, method, SEARCH_ROOTS)
    runtime_dir = OUTPUT_ROOT / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    worker_path = runtime_dir / "qwen_dtd_worker.py"
    worker_path.write_text(QWEN_WORKER_SOURCE, encoding="utf-8")
    request_path = runtime_dir / f"{method}.request.json"
    payload = {"method": method, "model_path": str(model_path), "clip_path": str(clip_dir),
               "output": str(METHOD_OUTPUTS[method].resolve()), "records": caption_records,
               "class_names": class_names, "dtd_split": DTD_SPLIT, "settings": SETTINGS,
               "max_new_images": MAX_NEW_IMAGES, "code_sha256": object_hash(QWEN_WORKER_SOURCE)}
    atomic_json(request_path, payload)
    log_path = runtime_dir / f"{method}.log"
    # Audit starts only after the Qwen process exits, so its GPU weights are released.
    run_worker(worker_path, request_path, "generate", worker_env, log_path)
    run_worker(worker_path, request_path, "audit", worker_env, log_path)
    return json.loads((METHOD_OUTPUTS[method] / "status.json").read_text(encoding="utf-8"))
'''),
        cell("markdown", """
## 1. Qwen3-VL-8B-Instruct

Run/resume 8B first. Cached images are not generated again, including flagged captions: no hidden retries
or generic replacements turn a failure into fabricated evidence. If an inference error is not CUDA OOM,
the worker stops visibly. OOM halves batches; OOM at batch 1 stops and preserves committed caches.
"""),
        cell("code", """
if RUN_8B:
    status_8b = run_caption_method("qwen3_vl_8b_instruct", QWEN_8B_MODEL_PATH)
    print(status_8b)
else:
    print("8B generation disabled by RUN_8B=False")
"""),
        cell("markdown", """
## 2. Qwen3-VL-32B-Instruct

Run/resume 32B with the SAME prompt/settings. The previous model is not resident on GPU.
This cell requires a real 32B checkpoint; it never silently substitutes 8B or Florence.
If only a smoke test was run, reset `MAX_NEW_IMAGES=0` and rerun these two generation cells to finish.
"""),
        cell("code", """
if RUN_32B:
    status_32b = run_caption_method("qwen3_vl_32b_instruct", QWEN_32B_MODEL_PATH)
    print(status_32b)
else:
    print("32B generation disabled by RUN_32B=False")
"""),
        cell("markdown", """
## Export Contract and Policy

Exports retain the required ablation columns and `caption_input_policy=dtd_lexical_v2`.
`raw_caption` and `final_caption` hold the SAME cleaned short model input, because the existing classifier
cleans raw_caption before encoding it. The full untouched Qwen response is retained as `source_long_caption`.
Generic openings are removed and at most 8 normalized words are kept using the EXISTING cleaner.
Captions with fewer than 5 words or a canonical DTD class term are masked, not invented or repaired.
Original response length, cleanup, alias hits and token-limit stops remain visible in the audit columns.

CLIP compares the final short text with the same two class templates as earlier experiments. Its similarity/flag
is diagnostic ONLY and does not alter caption_valid or leakage_detected. Alias hits are also diagnostic-only,
so this is the same outer policy, not a newly stricter hidden mask. No synonym/semantic no-leak guarantee is made.
These are **generated captions**, not human-verified ground truth. Coverage and CLIP scores are not classification metrics.
"""),
        cell("code", """
import pandas as pd
from IPython.display import display

summary_rows, complete_exports = summarize_exports(METHOD_OUTPUTS, caption_records)
export_dir = OUTPUT_ROOT / "exports"
published = {}
for method, rows in complete_exports.items():
    filename = "captions_qwen3vl_8b_instruct.csv" if "8b" in method else "captions_qwen3vl_32b_instruct.csv"
    destination = export_dir / filename
    atomic_csv(destination, rows, COLUMNS)
    published[method] = {"csv_path": str(destination.resolve()), "csv_sha256": file_sha256(destination),
                         "rows": len(rows), "captioner_fingerprint": rows[0]["captioner_fingerprint"],
                         "audit_fingerprint": rows[0]["audit_fingerprint"]}
    print("COMPLETE EXPORT:", destination, "| rows:", len(rows))
if summary_rows:
    summary = pd.DataFrame(summary_rows)
    display(summary.drop(columns=["csv_sha256", "csv_path"]))
    atomic_csv(OUTPUT_ROOT / "caption_generation_summary.csv", summary_rows, list(summary_rows[0]))
atomic_json(OUTPUT_ROOT / "caption_methods_manifest.json", {
    "schema": 1, "dataset": "DTD", "dtd_split": DTD_SPLIT, "prompt": CAPTION_PROMPT,
    "caption_input_policy": "dtd_lexical_v2", "methods": published,
    "both_complete": len(published) == 2, "captioner_finetuning": False, "classifier_training": False})
if len(published) != 2:
    print("Both methods are not complete yet. Resume the missing model; never train from partial CSVs.")
"""),
        cell("code", """
def review_train_image(index=0):
    training = [row for row in caption_records if row["split"] == "train"]
    if not 0 <= index < len(training):
        raise IndexError("Choose an official training-image index")
    record = training[index]
    image = load_rgb(record["image_path"])
    image.thumbnail((700, 500))
    display(image)
    for method, rows in complete_exports.items():
        row = next(row for row in rows if row["image_id"] == record["image_id"])
        print(method)
        print("  Response:", row["source_long_caption"])
        print("  Model input:", row["final_caption"], "| valid:", row["caption_valid"],
              "| alias diagnostic:", row["alias_hit_diagnostic"])
    image.close()

review_train_image(0)  # Review training images, not test examples for prompt tuning.
"""),
        cell("markdown", """
## Paired Caption Diagnostics

This comparison checks the same image IDs/order/splits and reports natural availability masks.
It does NOT hold the selected context fixed: full_both uses caption similarity in selection, so changing CSVs
can change text fusion, selected context images and the training trajectory. Any downstream result is a
whole-pipeline captioner ablation. These diagnostics do not measure visual correctness or prove label leakage.
"""),
        cell("code", """
if len(complete_exports) == 2:
    a = complete_exports["qwen3_vl_8b_instruct"]
    b = complete_exports["qwen3_vl_32b_instruct"]
    paired = [{"image_id": left["image_id"], "split": left["split"],
               "caption_8b": left["final_caption"], "caption_32b": right["final_caption"],
               "valid_8b": bool_value(left["caption_valid"]), "valid_32b": bool_value(right["caption_valid"]),
               "same_text": normalize(left["final_caption"]) == normalize(right["final_caption"])}
              for left, right in zip(a, b)]
    atomic_csv(OUTPUT_ROOT / "paired_caption_audit.csv", paired, list(paired[0]))
    paired_frame = pd.DataFrame(paired)
    display(paired_frame.groupby("split")[["valid_8b", "valid_32b", "same_text"]].mean())
    print("No test accuracy or F1 has been measured by this notebook.")
else:
    print("Finish both methods before paired caption diagnostics.")
"""),
        cell("markdown", """
## Train Two Fresh M-TACS Ablations

Use the ORIGINAL **notebook 05 full_both**, not the four Florence-method notebooks with fixed filename/SHA guards.
Make two copies of notebook 05. Attach the two complete exports to Kaggle, one chosen CSV per classifier run.
At the END of each copy's configuration cell, after its default caption variables, set:

```python
# Copy A: Qwen3-VL-8B captions
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_ATTACHED_DATASET/captions_qwen3vl_8b_instruct.csv"
CAPTION_INPUT_POLICY = "dtd_lexical_v2"
GENERATE_CAPTIONS_IF_MISSING = False
os.environ["DTD_FULL_BOTH_OUTPUT"] = "/kaggle/working/outputs/dtd_full_both_qwen3vl_8b_seed42"
```

```python
# Copy B: Qwen3-VL-32B captions
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_ATTACHED_DATASET/captions_qwen3vl_32b_instruct.csv"
CAPTION_INPUT_POLICY = "dtd_lexical_v2"
GENERATE_CAPTIONS_IF_MISSING = False
os.environ["DTD_FULL_BOTH_OUTPUT"] = "/kaggle/working/outputs/dtd_full_both_qwen3vl_32b_seed42"
```

Set the output variable BEFORE running notebook 05's dataset/output-directory cell. Keep DTD split 1,
pool seed/training seed 42, K=1, 50 epochs, matched batch/optimizer/loss/augmentation and validation-loss selection.
Use a new output directory and fresh classifier/text caches for each CSV; never resume Florence/8B weights as a 32B run.
Neither Qwen checkpoint is needed during classifier training when captions are already precomputed.
The shared raw CSV is read-only; `shared/caption_model_inputs.csv` records actual cleaned text/masks.

First check training-only coverage and visible correctness, then run the locked comparisons. Do not adjust
prompts, thresholds or decoding to chase repeated test F1. Consider multiple seeds/splits before claiming a stable gain.
A bigger captioner, longer response, higher lexical validity or higher CLIP class score does not guarantee better M-TACS.

Official references: [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL),
[8B model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[32B model](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
[native Transformers processor/generation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl).
"""),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"dtd-qwen-caption-{index:02d}"
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "dtd_caption_experiment": {"schema": 1, "methods": ["qwen3_vl_8b_instruct", "qwen3_vl_32b_instruct"],
            "default_dtd_split": 1, "default_policy": "dtd_lexical_v2", "classifier_training": False,
            "captioner_finetuning": False, "offline": True}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
