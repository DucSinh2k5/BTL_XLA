"""Swap only the caption backend of the ISIC full-both experiment for Florence-Prefix."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_isic_qwen8b_full_both_notebook as control
from build_qwen3vl_dtd_caption_notebook import definition
from dtd_caption_generation_cells import HELPERS


TARGET = PROJECT_ROOT / "notebooks/02_improvements/segmentation/full_both/florence_prefix/isic2017/mtacs-isic2017-448-florence-prefix-full-both.ipynb"
BACKEND = PROJECT_ROOT / "src/captioning/isic_florence_prefix.py"


def embed_caption_core():
    shared = control.embed_caption_core()
    backend = BACKEND.read_text(encoding="utf-8")
    imports = """from __future__ import annotations
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
"""
    # Reuse the tested pixel-only ISIC pipeline; no Qwen model/API code is embedded.
    names = (
        "atomic_csv", "atomic_json", "checkpoint_asset_identity", "file_sha256", "object_hash", "read_csv_rows",
        "clean_dtd_caption", "available_rows", "cache_path", "cached_row", "local_model_configs", "lock_protocol",
        "qwen_asset_identity", "resolve_clip_model", "run_worker", "runtime_versions", "safetensors_parameter_count",
        "weight_files", "POLICY", "CLINICAL_PATTERN", "COLUMNS", "caption_policy", "validate_caption_records",
        "validate_text_cache",
    )
    parts = [imports] + [definition(shared, name) for name in names]
    parts += [definition(backend, name) for name in (
        "METHOD", "MODEL_ID", "TASK_PROMPT", "validate_florence_prefix_folder", "IsicFlorencePrefixCaptioner")]
    parts.append("MODEL_IDS = {METHOD: MODEL_ID}\nCAPTION_PROMPT = TASK_PROMPT")
    for name in ("make_generation_row", "generate_isic_records", "validated_export", "encode_isic_texts"):
        source = definition(shared, name)
        if name == "generate_isic_records":
            source = control.replace_once(source, "factory=IsicQwenCaptioner", "factory=IsicFlorencePrefixCaptioner")
        source = source.replace("ISIC Qwen3-VL-8B", "ISIC Florence-Prefix").replace("Qwen OOM", "Florence OOM")
        source = source.replace("original Qwen response", "original Florence response")
        parts.append(source)
    parts += [definition(HELPERS, name) for name in (
        "florence_checkpoint", "offline_caption_wheels", "caption_runtime_environment")]
    parts.append(definition(backend, "florence_worker_main"))
    result = "\n\n".join(parts) + "\n"
    ast.parse(result)
    return result


def build_notebook():
    notebook = control.build_notebook()
    by_section = {cell["metadata"]["section"]: cell for cell in notebook["cells"]}

    def put(section, source):
        by_section[section].update(control.cell(by_section[section]["cell_type"], source, "florence_prefix_" + section))
        by_section[section]["metadata"]["section"] = section

    def source(section):
        return "".join(by_section[section]["source"])

    config = control.replace_once(source("configuration"), 'os.getenv("ISIC_QWEN8B_OUTPUT", str(PROJECT_DIR / "outputs" / "mtacs_isic2017_448_qwen8b_full_both"))',
                                  'os.getenv("ISIC_FLORENCE_PREFIX_OUTPUT", str(PROJECT_DIR / "outputs" / "mtacs_isic2017_448_florence_prefix_full_both"))')
    config = config[:config.index("# Offline input paths.")] + '''# Frozen Florence-Prefix, with a separate legacy caption runtime.
ISIC_ROOT = os.getenv("ISIC_ROOT", "")
FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
OFFLINE_ASSETS_ROOT = os.getenv("FLORENCE_OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
SEARCH_ROOTS = [Path(OFFLINE_ASSETS_ROOT), Path("/kaggle/input")]
CAPTION_OUTPUT = OUT_PUT_DIR / "caption_generation"
RESUME_TRAINING = True
MAX_NEW_CAPTION_IMAGES = 0  # 8=smoke test; reset to 0 and resume before training.
CAPTION_SETTINGS = {"batch_size": 8, "max_new_tokens": 48, "seed": 42}
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a GPU accelerator before Run All")
'''
    put("configuration", config)
    core = embed_caption_core()
    worker = core + '''\nif __name__ == "__main__":
    florence_worker_main(sys.argv[1], sys.argv[2], generate_records=generate_isic_records,
                         encode_texts=encode_isic_texts)
'''
    put("embedded_caption_worker", core)
    put("generate_and_encode", "ISIC_WORKER_SOURCE = " + repr(worker) + "\n" + '''
caption_records = []
for role, frame in (('train', train_df), ('val', val_df), ('test', test_df)):
    caption_records.extend({'image_id': str(row.image_id), 'image_path': str(row.image_path),
                            'split': role, 'image_sha256': file_sha256(row.image_path)}
                           for row in frame.itertuples(index=False))
validate_caption_records(caption_records)
local_florence, backend = florence_checkpoint(FLORENCE_MODEL_REF, SEARCH_ROOTS)
if backend != 'legacy':
    raise ValueError('Use the original Microsoft Florence-2-base legacy checkpoint for this Prefix control')
florence_path, parameter_count = validate_florence_prefix_folder(local_florence)
clip_path = resolve_clip_model(CLIP_MODEL_LOCAL_PATH, SEARCH_ROOTS)
runtime_dir = OUT_PUT_DIR / 'worker_runtime'
runtime_dir.mkdir(parents=True, exist_ok=True)
# Only this child uses 4.49; offline --target setup does not replace kernel packages.
florence_env = caption_runtime_environment('legacy', OFFLINE_ASSETS_ROOT, runtime_dir)
clip_env = dict(os.environ)
clip_env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                HF_HUB_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1')
clip_probe = subprocess.run([sys.executable, '-c',
    'from transformers import AutoTokenizer, CLIPTextModelWithProjection; import accelerate, torch; print(torch.__version__)'],
    env=clip_env, capture_output=True, text=True)
if clip_probe.returncode:
    raise RuntimeError('The kernel runtime cannot load local CLIP text weights. ' + clip_probe.stderr[-3000:])
worker_path = runtime_dir / 'isic_florence_prefix_worker.py'
worker_path.write_text(ISIC_WORKER_SOURCE, encoding='utf-8')
caption_payload = {'records': caption_records, 'settings': CAPTION_SETTINGS,
    'model_path': str(florence_path), 'clip_path': str(clip_path), 'output': str(CAPTION_OUTPUT),
    'max_new_images': MAX_NEW_CAPTION_IMAGES, 'code_sha256': object_hash(ISIC_WORKER_SOURCE)}
request_path = runtime_dir / 'request.json'
atomic_json(request_path, caption_payload)
run_worker(worker_path, request_path, 'generate', florence_env, runtime_dir / 'generate.log')
run_worker(worker_path, request_path, 'encode', clip_env, runtime_dir / 'encode.log')
print('Florence and CLIP workers exited; GPU memory is released before segmentation initialization.')
''')
    protocol = control.replace_once(source("locked_protocol_and_evaluation"),
                                    "isic2017_qwen8b_full_both_448", "isic2017_florence_prefix_full_both_448")
    # Hash the actual training cells, not a label-only copy of the other experiment.
    training_hash = hashlib.sha256("\n".join(source(name) for name in (
        "visual_transforms", "multimodal_model", "frozen_visual_banks", "unchanged_loss_and_loaders",
        "aligned_text_banks", "train_and_resume")).encode() +
        protocol[protocol.index("BASELINE_DICE ="):].encode()).hexdigest()
    protocol = control.replace_once(protocol, protocol.splitlines()[0], f"TRAINING_CODE_SHA256 = {training_hash!r}")
    put("locked_protocol_and_evaluation", protocol)
    test = source("test_and_comparison").replace("Qwen3-VL-8B full_both 448", "Florence-Prefix full_both 448")
    evidence_path = PROJECT_ROOT / "reports/paper_tables/isic_visual_vs_qwen8b_comparison_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    old = evidence["caption_run"]["metrics"]
    qwen_dice, qwen_iou = old["test_dice"]["value_displayed"], old["test_iou"]["value_displayed"]
    test = control.replace_once(test, "comparison = pd.DataFrame([\n", "comparison = pd.DataFrame([\n" +
        f"    {{'method': 'Qwen3-VL-8B full_both 448, historical seed42', 'test_dice': {qwen_dice!r},\n"
        f"     'test_iou': {qwen_iou!r}, 'new_run': False}},\n")
    test = control.replace_once(test, "comparison['dice_delta_percentage_points'] =",
        "comparison['dice_delta_vs_qwen8b_pp'] = (comparison.test_dice - " + repr(qwen_dice) + ") * 100\n"
        "comparison['iou_delta_vs_qwen8b_pp'] = (comparison.test_iou - " + repr(qwen_iou) + ") * 100\n"
        "comparison['dice_delta_percentage_points'] =")
    put("test_and_comparison", test)
    put("overview", '''# ISIC 2017: Florence-Prefix Captions + Full-Both M-TACS Segmentation

Generate/resume new ISIC captions.csv, encode with frozen CLIP, train the same query/candidate text fusion and
visual+text selector as the Qwen8B run, then compare held-out test Dice/IoU against both historical runs.
This is frozen caption inference, not Florence fine-tuning. No DTD CSV or diagnosis/mask enters the captioner.

**Why Prefix:** the saved DTD split1/seed42 classifier results show macro-F1 0.810651 (Prefix), 0.809382 (FirstValid),
0.790596 (MultiGrounded), 0.779611 (SingleGrounded). Prefix was best among those Florence variants on DTD,
not a proven ISIC winner. This transfer is exploratory after previous test results were observed.

## Offline Inputs and Run Order

1. ISIC 2017 standard image/mask splits: 2,000 train / 150 val / 600 test.
2. The same local DINOv3 code and **ViT-S/16** weights used in the visual and Qwen segmentation experiments.
3. Original **microsoft/Florence-2-base** legacy model folder, including weights, tokenizer, processor and its three
   Python code files. Default: `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`.
4. Attached `offline_wheels.zip`, `wheels.zip` or wheel folder with Transformers 4.49.0, tokenizers 0.21.x,
   huggingface-hub <1 and required einops/timm/accelerate, compatible with this Kaggle Python/platform.
   These are only needed if a compatible legacy child environment is not already available.
5. Full local CLIP ViT-B/32 checkpoint and tokenizer. **No Qwen checkpoint is needed.**

Run All from the top. Legacy Florence runs in a child process with offline dependencies installed into its own
`worker_runtime/legacy_packages` when needed (`pip --no-index --target`, never an Internet install).
The kernel's Transformers is not downgraded. CLIP runs in a separate child using the kernel environment.
Both workers exit before segmentation initializes; tokenizer parallelism is disabled before imports/workers.

The segmentation recipe is unchanged: 448, frozen DINOv3 ViT-S/16, two final-token gated cross-attention blocks,
DPT-lite, BCE+Dice, training-only 400-image pool, K=1, batch8, seed42, max100 epochs, patience15.
Best validation Dice selects the checkpoint; test does not select prompts, policies or epochs.
''')
    put("caption_policy", '''## Florence-Prefix with the Same ISIC Screening Policy

Every image receives the same Florence task token `<MORE_DETAILED_CAPTION>`, greedy decoding (`num_beams=1`,
`do_sample=False`, max48 new tokens), without references, filenames, diagnoses or masks.
Remove image-introduction boilerplate and retain the first 5-8 visual words (at most eight; do not fabricate missing words).
No teacher, LoRA, multi-candidate reranking or CLIP image-text ranking is used.

This transfers the DTD **Prefix procedure**, not the DTD 47-class rejection policy. Keep `isic_visible_attributes_v1`
from the Qwen control: reject too-short, diagnosis-containing or token-limit responses; preserve source_long_caption
and natural validity masks. Clinical terms are checked across the full response, not just the retained prefix.
Invalid text has zero CLIP vectors and zero effective gates; its image remains in the dataset and candidate pool.
These are heuristic checks, not proof of factual accuracy or absence of shortcuts. Inspect caption_coverage.csv.

Per-image atomic caches resume interrupted generation. `MAX_NEW_CAPTION_IMAGES=8` is an optional smoke test;
training is blocked until it is reset to 0 and all 2,750 images are exported. A short/truncated output is not silently repaired.
''')
    put("test_policy", '''## Locked Test and Historical Comparisons

Finish training/early stopping before test. Use the best validation-Dice checkpoint once and reuse locked test exports.
The table includes earlier displayed outputs: visual Dice 85.60%, IoU 77.38%; Qwen8B Dice 84.8689%, IoU 76.5407%.
Both are historical, rounded stdout references, **not rerun paired controls or multi-seed means**.
Florence results are measured by this run only; no Florence test score is pre-filled.

Keep captions, protocols, caches and checkpoints under this experiment's independent output root. Do not reuse a
Qwen/visual checkpoint here. Resume restores optimizer/scheduler/scaler/RNG state, but augmentation workers can prevent
bitwise equivalence to uninterrupted training. Do not tune the method from these test outcomes.
''')
    put("artifacts_and_limits", '''## Artifacts and Interpretation

Default output: `/kaggle/working/outputs/mtacs_isic2017_448_florence_prefix_full_both/`.
Keep caption_generation/captions.csv, generated/ resume caches, source responses, CLIP text cache, both locked
protocols, caption_coverage.csv, split/pool CSVs, best/last checkpoints, history, metrics.json, test_predictions.csv,
segmentation_comparison.csv, learning curves and query/context/caption/mask examples.

Report as **ISIC2017, 448, Florence-Prefix full_both, seed42**. Captioner/decoder/runtime and resulting masks differ
from Qwen; the segmentation architecture, fusion, selector and training rules remain the same. Compare coverage and
gates as well as Dice/IoU. One run does not establish statistical superiority or general medical caption quality.
The DTD rank motivated this transfer but does not guarantee transfer gains. No new GPU metrics were measured by the builder.

API: [Microsoft Florence-2-base model card](https://huggingface.co/microsoft/Florence-2-base).
The notebook uses the existing offline legacy checkpoint with its matching 4.49 runtime, not an automatic checkpoint conversion.
''')
    metadata = notebook["metadata"]["isic_full_both"]
    metadata.update(caption_model="microsoft/Florence-2-base", caption_method="florence_prefix_control",
                    training_code_sha256=training_hash, caption_backend="legacy_isolated_4.49",
                    qwen_historical_reference={"dice": qwen_dice, "iou": qwen_iou,
                        "source_sha256": evidence["notebook_sha256"]},
                    method_selection_dataset="DTD", method_selection_is_exploratory=True)
    return notebook


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
