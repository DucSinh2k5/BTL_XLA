"""Build one offline ISIC caption-generation + full_both segmentation notebook."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json
from pathlib import Path

from build_qwen3vl_dtd_caption_notebook import definition


SOURCE = PROJECT_ROOT / "tacs_isic_segement_448.ipynb"
TARGET = PROJECT_ROOT / "notebooks/02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/mtacs-isic2017-448-qwen3-vl-8b-full-both.ipynb"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old[:100]!r}")
    return source.replace(old, new, 1)


def embed_caption_core():
    folder = PROJECT_ROOT / "src/captioning"
    source = (folder / "isic_qwen_caption.py").read_text(encoding="utf-8")
    dependencies = {name: (folder / (name + ".py")).read_text(encoding="utf-8") for name in (
        "dtd_caption_finetune", "dtd_caption_screening", "dtd_qwen_caption")}
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module in dependencies:
            definitions = [definition(dependencies[node.module], name.name) for name in node.names]
            if node.module == "dtd_qwen_caption":
                definitions.insert(0, definition(dependencies[node.module], "CAPTION_PROMPT"))
            edits.append((node.lineno - 1, node.end_lineno, "\n\n".join(definitions) + "\n"))
    for first, last, replacement in reversed(edits):
        lines[first:last] = [replacement]
    result = "".join(lines)
    ast.parse(result)
    if any(isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("dtd_", "isic_")) for node in ast.parse(result).body):
        raise ValueError("Project-local dependency escaped worker embedding")
    return result


def cell(kind, source, label):
    source = source.strip() + "\n"
    result = {"cell_type": kind, "id": hashlib.sha256(label.encode()).hexdigest()[:12],
              "metadata": {"section": label}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    raw = SOURCE.read_bytes()
    base = json.loads(raw.decode("utf-8"))
    sources = ["".join(c.get("source", [])) for c in base["cells"]]
    core = embed_caption_core()
    worker = core + '\n\nif __name__ == "__main__":\n    isic_worker_main(sys.argv[1], sys.argv[2])\n'
    pieces = (PROJECT_ROOT / "src/training/mtacs_isic_notebook_cells.py").read_text(encoding="utf-8")
    sections = {}
    for name in ("TEXT_BANK_CELL", "PROTOCOL_CELL", "TRAIN_CELL", "TEST_CELL"):
        node = next(n for n in ast.parse(pieces).body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        sections[name] = ast.literal_eval(node.value)
    model_core = (PROJECT_ROOT / "src/training/mtacs_isic_full_both.py").read_text(encoding="utf-8")
    model_source = sources[12].split("\n\nmodel = TACSSegmentationModel(")[0]
    if model_source == sources[12]:
        raise ValueError("Cannot separate baseline model definitions from instantiation")
    config = replace_once(sources[4], 'OUT_PUT_DIR = PROJECT_DIR / "outputs" / "tacs_isic2017_segmentation"',
                          'OUT_PUT_DIR = Path(os.getenv("ISIC_QWEN8B_OUTPUT", str(PROJECT_DIR / "outputs" / "mtacs_isic2017_448_qwen8b_full_both")))')
    config += '''

# Offline input paths. Leave QWEN_8B_MODEL_PATH empty only if discovery finds exactly one Instruct folder.
ISIC_ROOT = os.getenv("ISIC_ROOT", "")
QWEN_8B_MODEL_PATH = os.getenv("QWEN_8B_MODEL_PATH", "")
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
QWEN_RUNTIME_SITE_PACKAGES = os.getenv("QWEN_RUNTIME_SITE_PACKAGES", "")
SEARCH_ROOTS = [Path("/kaggle/input")]
CAPTION_OUTPUT = OUT_PUT_DIR / "caption_generation"
RESUME_TRAINING = True
MAX_NEW_CAPTION_IMAGES = 0  # 0=all 2,750 images; 8=smoke test, then reset to 0 and resume.
CAPTION_SETTINGS = {"batch_size": 1, "dtype": "bfloat16", "min_pixels": 256 * 256,
                    "max_pixels": 512 * 512, "max_new_tokens": 48, "seed": 42}
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a GPU accelerator before Run All")
'''
    dataset = replace_once(sources[6], "DATASET_ROOT = find_isic_dataset_root()",
                           'DATASET_ROOT = Path(ISIC_ROOT) if ISIC_ROOT else find_isic_dataset_root()')
    for name, frame in (("train_split", "train_df"), ("val_split", "val_df"), ("test_split", "test_df"), ("candidate_pool", "candidate_df")):
        dataset = replace_once(dataset, f'{frame}.to_csv(OUT_PUT_DIR / "{name}.csv", index=False)\n', "")
    dataset += '''

all_split_ids = pd.concat([train_df.image_id, val_df.image_id, test_df.image_id]).astype(str)
if all_split_ids.duplicated().any():
    raise ValueError("Duplicate/overlapping image IDs across ISIC train, val and test")
if [len(train_df), len(val_df), len(test_df)] != [2000, 150, 600]:
    raise ValueError("Expected ISIC2017 standard 2000/150/600 image-mask counts; verify ISIC_ROOT")
'''
    instantiate = '''
FullBothISICModel = make_isic_full_both_class(TACSSegmentationModel)
model = FullBothISICModel(DINOV3_REPO_DIR, DINOV3_MODEL_NAME, resolve_dinov3_weights(),
                          feature_dim=FEATURE_DIM, text_dim=TEXT_DIM).to(DEVICE)
print('Full-both segmentation trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
'''
    training_hash = hashlib.sha256("\n".join([sources[8], sources[10], model_source, model_core, sources[14],
                                               sources[16], *sections.values()]).encode()).hexdigest()
    test_stdout = "".join("".join(o.get("text", [])) if isinstance(o.get("text", []), list) else o["text"]
                          for o in base["cells"][20].get("outputs", []) if o.get("output_type") == "stream")
    import re
    baseline = {name: float(re.search(rf"^test_{name}:\s*([\d.]+)", test_stdout, re.M)[1]) for name in ("dice", "iou")}
    launch = 'ISIC_WORKER_SOURCE = ' + repr(worker) + '\n' + '''
caption_records = []
for role, frame in (('train', train_df), ('val', val_df), ('test', test_df)):
    caption_records.extend({'image_id': str(row.image_id), 'image_path': str(row.image_path),
                            'split': role, 'image_sha256': file_sha256(row.image_path)}
                           for row in frame.itertuples(index=False))
validate_caption_records(caption_records)
runtime_env = worker_environment(QWEN_RUNTIME_SITE_PACKAGES)
check_worker_runtime(runtime_env)
qwen_path = resolve_qwen_model(QWEN_8B_MODEL_PATH, METHOD, SEARCH_ROOTS)
clip_path = resolve_clip_model(CLIP_MODEL_LOCAL_PATH, SEARCH_ROOTS)
runtime_dir = OUT_PUT_DIR / 'worker_runtime'
runtime_dir.mkdir(parents=True, exist_ok=True)
worker_path = runtime_dir / 'isic_caption_worker.py'
worker_path.write_text(ISIC_WORKER_SOURCE, encoding='utf-8')
caption_payload = {'records': caption_records, 'settings': CAPTION_SETTINGS,
    'model_path': str(qwen_path), 'clip_path': str(clip_path), 'output': str(CAPTION_OUTPUT),
    'max_new_images': MAX_NEW_CAPTION_IMAGES, 'code_sha256': object_hash(ISIC_WORKER_SOURCE)}
request_path = runtime_dir / 'request.json'
atomic_json(request_path, caption_payload)
run_worker(worker_path, request_path, 'generate', runtime_env, runtime_dir / 'generate.log')
run_worker(worker_path, request_path, 'encode', runtime_env, runtime_dir / 'encode.log')
print('Caption and CLIP workers exited; their model weights are no longer resident on the GPU.')
'''
    visual = '''
import textwrap
state = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
if state['protocol_id'] != PROTOCOL_ID:
    raise ValueError('Wrong visualization checkpoint')
model.load_state_dict(state['model_state_dict'])
del state
model.eval()
sample_batch = next(iter(test_loader))
moved = move_batch_to_device(sample_batch)
with torch.inference_mode(), torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
    q_tokens, q_global = model.encode_images(moved['images'])
    q_text, q_valid = query_text_inputs(moved['image_id'])
    details = full_both_selection(q_global, q_text, q_valid)
    indices = details['selected_indices']
    logits, _, _ = model.decode_full_both(q_tokens, gather_candidate_tokens(indices), q_text, q_valid,
        candidate_text_bank[indices], candidate_text_valid[indices])
predicted = (torch.sigmoid(logits.float()) > THRESHOLD).cpu()
count = min(4, len(indices))
fig, axes = plt.subplots(count, 4, figsize=(16, 4.5 * count), squeeze=False)
for j in range(count):
    image_id = str(moved['image_id'][j])
    query_row = test_df.loc[test_df.image_id.eq(image_id)].iloc[0]
    candidate_row = candidate_df.iloc[int(indices[j])]
    with Image.open(query_row.image_path) as im:
        axes[j, 0].imshow(im.convert('RGB'))
    axes[j, 0].set_title(textwrap.fill('Query: ' + caption_by_id[image_id] + (' [masked]' if not q_valid[j] else ''), 40), fontsize=9)
    axes[j, 1].imshow(moved['masks'][j, 0].cpu(), cmap='gray')
    axes[j, 1].set_title('Ground truth mask')
    axes[j, 2].imshow(predicted[j, 0], cmap='gray')
    axes[j, 2].set_title('Predicted mask')
    with Image.open(candidate_row.image_path) as im:
        axes[j, 3].imshow(im.convert('RGB'))
    caption = caption_by_id[str(candidate_row.image_id)]
    axes[j, 3].set_title(textwrap.fill('Context: ' + caption + (' [masked]' if not candidate_text_valid[indices[j]] else ''), 40), fontsize=9)
    for ax in axes[j]:
        ax.axis('off')
fig.tight_layout()
fig.savefig(OUT_PUT_DIR / 'segmentation_examples_with_captions.png', dpi=150)
plt.show()
'''
    cells = [
        cell("markdown", '''
# ISIC 2017: Qwen3-VL-8B Captions + Full-Both M-TACS Segmentation

One offline notebook: generate/resume **captions.csv** for all 2,750 images, encode them with frozen CLIP,
train segmentation with query/candidate text in fusion **and** selector, and evaluate Dice/IoU on 600 test images.
This is a research extension of the executed 448-resolution TACS variant, not Qwen fine-tuning or clinical diagnosis.

The visual backbone, two final-token cross-attention blocks, DPT-lite head, BCE+Dice loss, 20% training candidate
pool, one selected context, batch 8, 100-epoch limit and patience 15 are retained from the visual notebook.
Best checkpoint uses validation hard Dice. Text heads and the text-dependent selector are new trainable components.
No test score is used to tune captions or select checkpoints.

## Offline Inputs

1. ISIC 2017 image/mask dataset with `train`, `val`, `test` folders (2,000/150/600).
2. Local DINOv3 code and the **ViT-S/16** weights from the visual baseline.
3. Complete **Qwen/Qwen3-VL-8B-Instruct** floating-point checkpoint, tokenizer, processor and chat template.
4. Full local CLIP ViT-B/32 checkpoint, including tokenizer and text-projection weights.
5. Compatible Qwen3-VL Transformers runtime (the native class must import; legacy Florence 4.49 is insufficient).
   If using an attached runtime, set `QWEN_RUNTIME_SITE_PACKAGES` to its compatible site-packages directory.

Set exact paths in configuration if discovery is ambiguous. No pip install, network request or external API is made.
Caption batch 1, BF16 and bounded image pixels are defaults; 8B can exceed 16 GiB VRAM. Free other GPU models.
Qwen generation and CLIP encoding run sequentially in separate processes **before** segmentation is initialized.
Keep the entire output folder to resume; changing assets, prompt, data or training protocol requires a new output root.
''', "overview"),
        cell("code", '''
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import importlib.util
required = ['numpy', 'pandas', 'torch', 'torchvision', 'PIL', 'matplotlib', 'tqdm']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError('Missing offline dependencies: ' + ', '.join(missing))
print('Offline mode; tokenizer parallelism disabled before imports/workers.')
''', "offline_preflight"),
        cell("code", config, "configuration"),
        cell("code", dataset, "standard_splits"),
        cell("code", sources[8], "visual_transforms"),
        cell("markdown", '''
## Pixel-only Captions: ISIC Policy, Not DTD Policy

The VLM sees one RGB image and the same fixed prompt, never masks, diagnoses, filenames, labels or test metrics.
Descriptions contain 5-8 English visual words. The original response is retained; short/clinical-diagnosis/token-limit
outputs are masked rather than repaired or silently retried. **isic_visible_attributes_v1** deliberately does not
forbid DTD texture words or reject captions by CLIP class-similarity 0.30. This is binary segmentation, not 47-class DTD.
The lexical diagnosis check is a heuristic, not proof of factual accuracy or absence of shortcuts.
Invalid text gets zero vectors while its image remains in every original split and candidate pool.

Set `MAX_NEW_CAPTION_IMAGES=8` for a generation smoke test; encoding/training stops on an incomplete CSV.
Set it back to 0 and rerun to finish. Per-image atomic caches resume committed work; only complete exports are `captions.csv`.
''', "caption_policy"),
        cell("code", core, "embedded_caption_worker"),
        cell("code", launch, "generate_and_encode"),
        cell("code", sections["TEXT_BANK_CELL"], "aligned_text_banks"),
        cell("markdown", '''
## Full-Both Segmentation Model

Query CLIP text is projected and gated into each query patch token. The selected candidate caption is independently
projected/gated into context tokens, before the same visual cross-attention + DPT-lite decoder.
Selector score = original visual score + sigmoid(learned mix) * valid query/candidate CLIP text cosine.
An invalid query/candidate text pair contributes zero semantic score. Self candidates are masked after score fusion.
The segmentation-loss path uses straight-through weights for **both** selected image tokens and candidate text vectors;
the policy path samples hard actions from the same masked scores. Its reward compares query-image+query-text against
that same input plus selected candidate-image+candidate-text. Captions are never treated as ground truth masks.
''', "architecture"),
        cell("code", sources[10] + "\n\n" + model_source + "\n\n" + model_core + instantiate, "multimodal_model"),
        cell("code", sources[14], "frozen_visual_banks"),
        cell("code", sources[16], "unchanged_loss_and_loaders"),
        cell("code", f'TRAINING_CODE_SHA256 = {training_hash!r}\nBASELINE_DICE = {baseline["dice"]!r}\nBASELINE_IOU = {baseline["iou"]!r}\n' + sections["PROTOCOL_CELL"], "locked_protocol_and_evaluation"),
        cell("code", sections["TRAIN_CELL"], "train_and_resume"),
        cell("markdown", '''
## Locked Test and Comparison

Finish the declared run or its early stopping before this cell. Test uses the best validation checkpoint; subsequent
executions reuse saved test metrics. Exports contain full precision metrics and every test image's selected context,
captions, availability masks and gates. The visual reference below is the earlier rounded stdout (Dice 85.60%, IoU 77.38%),
not a freshly rerun paired control. The new run can be higher or lower; no result is pre-filled.

Resume restores last model/optimizer/scheduler/scaler/RNG state. With multiprocessing augmentation workers, bitwise
equivalence to an uninterrupted run is not guaranteed. Do not change epochs/hyperparameters inside an existing protocol.
''', "test_policy"),
        cell("code", sections["TEST_CELL"], "test_and_comparison"),
        cell("code", visual, "visualize_query_context_text_masks"),
        cell("markdown", '''
## Outputs and Interpretation

Keep `caption_generation/` (CSV, per-image resume caches and text-cache protocol), `training_protocol/`, split/pool CSVs,
`best_val_checkpoint.pt`, `checkpoint_last.pt`, `training_history.csv`, `completed.json`, `metrics.json`,
`test_predictions.csv`, `caption_coverage.csv`, `segmentation_comparison.csv` and both figures.
Qwen is a frozen caption generator, CLIP is a frozen text encoder; only the segmentation/text/selector heads train.
This has not been fine-tuned as a medical captioner and does not provide a clinical diagnosis.

Report this as **ISIC 2017, 448, Qwen3-VL-8B full_both, seed 42**. It is not the paper's 256/three-seed result.
Use matched-protocol visual controls and additional seeds to attribute gains; monitor text availability and context
selection distributions, and do not tune policy/prompt/thresholds from test outcomes. No test result has been measured
by the notebook builder.

Qwen API references: [official model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).
''', "artifacts_and_limits"),
    ]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "isic_full_both": {"schema": 1, "dataset": "ISIC2017", "caption_model": "Qwen/Qwen3-VL-8B-Instruct",
            "experiment": "full_both", "caption_generation": True, "image_size": 448, "seed": 42,
            "epoch_limit": 100, "patience": 15, "context_k": 1,
            "baseline_notebook": SOURCE.relative_to(PROJECT_ROOT).as_posix(),
            "baseline_notebook_sha256": hashlib.sha256(raw).hexdigest(),
            "baseline_metrics_are_rounded_stdout": baseline, "training_code_sha256": training_hash,
            "caption_policy": "isic_visible_attributes_v1"}}, "cells": cells}


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
