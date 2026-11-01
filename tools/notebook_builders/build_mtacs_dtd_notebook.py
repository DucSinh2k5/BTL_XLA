"""Build a self-contained offline DTD caption-generation and M-TACS notebook."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import json
from pathlib import Path


ROOT = PROJECT_ROOT
SOURCE = project_path('mtacs-train-captions-vitl16.ipynb')
CAPTION_SOURCE = project_path('generate_sun397_captions.py')
TARGET = project_path('mtacs-train-captions-vitl16-dtd.ipynb')


def cell_source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label}, found {count}")
    return source.replace(old, new, 1)


def replace_block(source, start, end, replacement, label):
    first = source.find(start)
    last = source.find(end, first + len(start)) if first >= 0 else -1
    if first < 0 or last < 0:
        raise RuntimeError(f"Could not locate {label}")
    return source[:first] + replacement + source[last:]


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


notebook = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
cells = notebook["cells"]

set_source(cells[0], """# M-TACS ViT-L/16 on DTD: captions, training, evaluation

Run the cells in order. Each official DTD split supplies 1,880 train, 1,880 validation and 1,880 test images. Twenty percent of the official training split forms the candidate pool; only training images are candidates. This notebook creates `captions.csv` for all three splits from the images with a local Florence-2 model, audits the captions with a local CLIP model, caches CLIP text embeddings, trains M-TACS and evaluates the best validation checkpoint on test.

The captioner receives image pixels only. No ground-truth class name or label is passed to it. Invalid or flagged captions get a zero text embedding, but their images remain usable. Lexical and CLIP checks reduce direct label leakage; they cannot prove that every synonym or implied texture class has been removed. Test labels are used only for final evaluation.

The notebook is offline and does not download or install packages. Attach DTD, DINOv3 code and ViT-L/16 weights, a complete Florence-2-base model folder, and a local CLIP ViT-B/32 folder before running all cells. With Transformers 5.x, attach the native `florence-community/Florence-2-base` checkpoint; with Transformers 4.49.x, attach the original `microsoft/Florence-2-base` checkpoint. These checkpoint formats are not interchangeable. Set `FLORENCE_MODEL_REF` or `CLIP_MODEL_LOCAL_PATH` in the configuration cell when automatic local discovery does not find your attached folder.

`captions.csv` is written after every inference batch. A later run with the same output directory resumes missing images. Keep the output and its CLIP cache when continuing a Kaggle session.
""")

set_source(cells[1], """## Run order

1. Attach the DTD dataset at `/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd`, the local Florence-2 model, local CLIP model, DINOv3 code and ViT-L/16 weights.
2. Run all cells from top to bottom. The caption cell creates or resumes `captions.csv`; the next cells validate it and cache CLIP embeddings.
3. The training cell selects a checkpoint using DTD validation split 1. The final cell evaluates DTD test split 1 once.

For a 10-split benchmark, repeat this experiment with `DTD_SPLIT` set to 1 through 10 and separate output directories. Caption text can be reused across splits, but each split requires a separate training run and candidate pool.
""")

set_source(cells[2], """# Offline preflight: no pip or network requests are made by this notebook.
import importlib.util
from importlib.metadata import version

_required_modules = {
    "numpy": "numpy", "pandas": "pandas", "torch": "torch", "torchvision": "torchvision",
    "PIL": "Pillow", "matplotlib": "matplotlib", "tqdm": "tqdm", "sklearn": "scikit-learn",
    "transformers": "transformers", "accelerate": "accelerate", "einops": "einops", "timm": "timm",
}
_missing_modules = [package for module, package in _required_modules.items() if importlib.util.find_spec(module) is None]
if _missing_modules:
    raise RuntimeError("Missing offline packages: " + ", ".join(_missing_modules))
_transformers_version = version("transformers")
FLORENCE_BACKEND = "legacy" if _transformers_version.startswith("4.49.") else "native" if _transformers_version.split(".")[0] == "5" else None
if FLORENCE_BACKEND is None:
    print(f"Warning: Transformers {_transformers_version} has not been tested here; use 4.49.x or 5.x")
print("Offline package check passed; transformers", _transformers_version, "| Florence backend:", FLORENCE_BACKEND)
""")

config = cell_source(cells[3])
config = replace_once(
    config,
    'CAPTIONS_INPUT_PATH = os.getenv("CAPTIONS_INPUT_PATH", "/kaggle/input/datasets/nguyenducxu/captions-csv/captions (3).csv")  # Optional: set to the exact /kaggle/input/.../captions.csv path if auto-discovery misses it.',
    'FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "florence-community/Florence-2-base" if FLORENCE_BACKEND == "native" else "microsoft/Florence-2-base")  # Complete local checkpoint matching the Transformers backend.',
    "Florence model reference",
)
config = replace_once(config, 'PROMPT_VERSION = "florence2_detailed_v1"',
                      'PROMPT_VERSION = "florence2_detailed_v2"', "caption prompt version")
config = replace_once(config, 'CAPTION_MAX_IMAGES_PER_SPLIT = None  # Keep None for the complete benchmark dataset.',
                      'DTD_SPLIT = int(os.getenv("DTD_SPLIT", "1"))  # Official DTD split, 1..10.', "DTD split")
config = replace_once(config,
    'PROMPT_TEXT = "Florence-2 task <MORE_DETAILED_CAPTION>; retain 5-8 readable visual-attribute words and reject SUN397 class-name leakage."',
    'CAPTION_MAX_NEW_TOKENS = 48\nCAPTION_SEMANTIC_THRESHOLD = 0.30\nCAPTION_WORKERS = min(8, os.cpu_count() or 1)\nPROMPT_TEXT = "Florence-2 detailed captions; retain 5-8 visual words and screen DTD class terms."',
    "caption controls",
)
config = replace_once(config,
    'VLM_MODEL = "microsoft/Florence-2-base"',
    'VLM_MODEL = FLORENCE_MODEL_REF',
    "caption model identity",
)
set_source(cells[3], config)

set_source(cells[4], """## DTD split 1 and candidate pool

The official `labels/train1.txt`, `val1.txt`, and `test1.txt` files define the split. Validation and test images never enter the candidate pool.
""")

settings = cell_source(cells[5])
settings = replace_once(settings,
    "OUT_PUT_DIR = PROJECT_DIR / 'outputs' / 'mtacs_sun397_visual_text_context_vitl16'",
    "OUT_PUT_DIR = PROJECT_DIR / 'outputs' / f'mtacs_dtd_split{DTD_SPLIT}_captions_vitl16'",
    "output directory",
)
settings = replace_block(settings, "# --- ĐƯỜNG DẪN DATASET (SUN397) ---\n",
    "# --- ĐƯỜNG DẪN DINOv3 REPO & WEIGHTS ---\n",
    """# DTD images and official annotation files.
DATASET_ROOT = Path(os.getenv('DTD_ROOT', '/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd'))
if not DATASET_ROOT.is_dir():
    fallback = Path('/kaggle/input/describable-textures-dataset-dtd/dtd')
    if fallback.is_dir():
        DATASET_ROOT = fallback
IMAGE_ROOT = DATASET_ROOT / 'images'
LABEL_ROOT = DATASET_ROOT / 'labels'
if not IMAGE_ROOT.is_dir() or not LABEL_ROOT.is_dir():
    raise FileNotFoundError(
        f'DTD images/ or labels/ not found below {DATASET_ROOT}. Set DTD_ROOT to the attached dtd directory.'
    )
if not 1 <= DTD_SPLIT <= 10:
    raise ValueError('DTD_SPLIT must be in 1..10')

""", "DTD input paths")
settings = replace_once(settings, "NUM_CLASSES = 397  # SUN397 có 397 scene categories",
                        "NUM_CLASSES = 47  # DTD texture attributes", "class count")
settings = replace_once(settings,
    "USE_BBOX_CROP = False  # SUN397 là bài toán scene classification, không dùng crop bbox",
    "USE_BBOX_CROP = False  # DTD texture classification uses full images.", "crop setting")
settings = replace_once(settings, "VAL_FRACTION = 0.10",
                        "VAL_FRACTION = None  # Official DTD val split is used.", "validation fraction")
settings = replace_once(settings,
    "print(f'GPU memory: {GPU_MEMORY_GB:.1f} GiB; ViT-L batch sizes: train={BATCH_SIZE}, retrieval={RETRIEVAL_BATCH_SIZE}, feature-refresh={CANDIDATE_FEATURE_BANK_BATCH_SIZE}')",
    "print(f'GPU memory: {GPU_MEMORY_GB:.1f} GiB; ViT-L batch sizes: train={BATCH_SIZE}, retrieval={RETRIEVAL_BATCH_SIZE}, feature-refresh={CANDIDATE_FEATURE_BANK_BATCH_SIZE}')\nCAPTION_BATCH_SIZE = int(os.getenv('CAPTION_BATCH_SIZE', '64' if GPU_MEMORY_GB >= 40 else '16'))",
    "caption batch size",
)
settings = replace_block(settings, "def load_class_names(dataset_root):\n", "candidate_rng = np.random.default_rng(SEED)\n",
    """from pathlib import PurePosixPath

class_names = sorted(path.name for path in IMAGE_ROOT.iterdir() if path.is_dir())
if len(class_names) != NUM_CLASSES:
    raise RuntimeError(f'Expected {NUM_CLASSES} DTD texture classes in {IMAGE_ROOT}; found {len(class_names)}')
class_to_id = {name: index for index, name in enumerate(class_names)}

def load_dtd_split(name):
    split_file = LABEL_ROOT / f'{name}{DTD_SPLIT}.txt'
    if not split_file.is_file():
        raise FileNotFoundError(f'Official DTD split file missing: {split_file}')
    records = []
    for raw in split_file.read_text(encoding='utf-8-sig').splitlines():
        rel = raw.strip().replace('\\\\', '/').lstrip('/')
        if not rel:
            continue
        parts = PurePosixPath(rel).parts
        if parts and parts[0] == 'images':
            parts = parts[1:]
        if len(parts) != 2 or parts[0] not in class_to_id or '..' in parts:
            raise ValueError(f'Unexpected path in {split_file}: {raw!r}')
        rel = PurePosixPath(*parts).as_posix()
        image_path = IMAGE_ROOT.joinpath(*parts)
        if not image_path.is_file():
            raise FileNotFoundError(f'DTD split refers to missing image: {image_path}')
        records.append({
            'image_id': rel, 'rel_path': rel, 'fname': parts[-1],
            'image_path': str(image_path.resolve()), 'class_id': class_to_id[parts[0]],
            'split': name, 'is_train': int(name == 'train'),
            'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 99999, 'bbox_y2': 99999,
        })
    frame = pd.DataFrame.from_records(records)
    if len(frame) != NUM_CLASSES * 40 or frame['image_id'].duplicated().any():
        raise RuntimeError(f'{split_file} should contain 40 distinct images per class; found {len(frame)} rows')
    counts = frame.groupby('class_id').size()
    if len(counts) != NUM_CLASSES or not counts.eq(40).all():
        raise RuntimeError(f'{split_file} has an unexpected per-class distribution')
    frame['row_id'] = np.arange(len(frame), dtype=int)
    return frame

train_df = load_dtd_split('train')
val_df = load_dtd_split('val')
test_df = load_dtd_split('test')
all_ids = pd.concat([train_df['image_id'], val_df['image_id'], test_df['image_id']])
if all_ids.duplicated().any():
    raise RuntimeError('DTD train/val/test split files overlap')
if TRAIN_ON_FULL_TRAIN:
    raise ValueError('TRAIN_ON_FULL_TRAIN is incompatible with the held-out DTD validation split')
full_train_df = train_df.copy()

""", "official DTD split loader")
settings = settings.replace("# Stable image identity shared by caption and embedding caches.\n"
    "candidate_df['image_id'] = candidate_df['rel_path'].astype(str).str.replace('\\\\', '/', regex=False)\n"
    "for _split_df in (train_df, val_df, test_df):\n"
    "    _split_df['image_id'] = _split_df['rel_path'].astype(str).str.replace('\\\\', '/', regex=False)",
    "# image_id is the stable DTD relative path created by load_dtd_split().")
set_source(cells[5], settings)

set_source(cells[6], """## Generate and audit DTD captions

This cell uses the local Florence-2 model and local CLIP model. It writes each completed batch to `captions.csv` so rerunning it resumes missing images. Captions are inferred from images alone. The lexical and CLIP similarity checks flag text that may reveal DTD class names; both are heuristic. Check the valid-caption rate before training.
""")

caption_script = CAPTION_SOURCE.read_text(encoding="utf-8")
caption_tree = ast.parse(caption_script)
caption_nodes = {
    node.name: ast.get_source_segment(caption_script, node)
    for node in caption_tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
}
function_names = [
    "normalize_phrase", "build_forbidden_terms", "audit_caption", "load_existing", "append_rows",
    "CaptionDataset", "collate_images", "_model_type", "resolve_local_model", "load_florence",
    "generate_batch", "load_semantic_auditor", "encode_clip_texts", "semantic_audit_captions",
]
missing = set(function_names) - set(caption_nodes)
if missing:
    raise RuntimeError(f"Caption source functions not found: {sorted(missing)}")
resolver = caption_nodes["resolve_local_model"]
resolver = replace_once(
    resolver,
    'Path.home() / ".cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots",',
    'Path.home() / ".cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots",\n'
    '        Path.home() / ".cache/huggingface/hub/models--florence-community--Florence-2-base/snapshots",',
    "community Florence cache",
)
resolver = replace_once(
    resolver,
    'matching = [path for path in candidates if _model_type(path) == expected]',
    '''matching = []
    for path in candidates:
        model_type = _model_type(path)
        if expected in {"florence2_native", "florence2_legacy"}:
            if model_type != "florence2":
                continue
            try:
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            legacy = "AutoModelForCausalLM" in config.get("auto_map", {})
            if legacy != (expected == "florence2_legacy"):
                continue
        elif model_type != expected:
            continue
        matching.append(path)''',
    "Florence checkpoint format filter",
)
resolver = replace_once(
    resolver,
    'requested_name = "florence" if expected == "florence2" else "clip"',
    'requested_name = "florence" if expected.startswith("florence2") else "clip"',
    "Florence checkpoint sorting",
)
caption_nodes["resolve_local_model"] = resolver
caption_nodes["load_florence"] = '''def load_florence(model_id: str, device: torch.device, local_files_only: bool):
    if FLORENCE_BACKEND not in {"legacy", "native"}:
        raise RuntimeError(
            f"Transformers {_transformers_version} is not supported here; use 4.49.x with the Microsoft checkpoint "
            "or 5.x with the florence-community checkpoint."
        )
    expected_format = "florence2_legacy" if FLORENCE_BACKEND == "legacy" else "florence2_native"
    local_model = resolve_local_model(model_id, expected_format)
    if local_model is None:
        checkpoint = "microsoft/Florence-2-base" if FLORENCE_BACKEND == "legacy" else "florence-community/Florence-2-base"
        raise FileNotFoundError(
            f"No compatible local Florence-2 checkpoint for Transformers {_transformers_version}. "
            f"Attach the complete {checkpoint} folder as a Kaggle Input and set FLORENCE_MODEL_REF to its path. "
            "DTD images and CLIP weights do not contain Florence-2 weights."
        )
    from transformers import AutoProcessor
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(local_model), trust_remote_code=FLORENCE_BACKEND == "legacy", local_files_only=True)
    if FLORENCE_BACKEND == "legacy":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model), torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True, local_files_only=True,
        )
    else:
        from transformers import Florence2ForConditionalGeneration
        model = Florence2ForConditionalGeneration.from_pretrained(
            str(local_model), dtype=dtype, local_files_only=True,
        )
    print(f"Using local Florence-2 ({FLORENCE_BACKEND}): {local_model}")
    return model.to(device).eval(), processor, dtype'''
caption_nodes["load_semantic_auditor"] = replace_once(
    caption_nodes["load_semantic_auditor"],
    'model = CLIPModel.from_pretrained(model_path, torch_dtype=dtype, local_files_only=local_files_only).to(device).eval()',
    'dtype_arg = {"dtype" if FLORENCE_BACKEND == "native" else "torch_dtype": dtype}\n'
    '    model = CLIPModel.from_pretrained(model_path, local_files_only=local_files_only, **dtype_arg).to(device).eval()',
    "CLIP dtype compatibility",
)
caption_functions = "\n\n\n".join(caption_nodes[name] for name in function_names)
caption_functions = caption_functions.replace(
    caption_nodes["encode_clip_texts"],
    "@torch.inference_mode()\n" + caption_nodes["encode_clip_texts"],
)
caption_functions = caption_functions.replace("SUN class names", "DTD class names")
caption_functions = caption_functions.replace(
    "The shown Kaggle inputs include SUN397 and CLIP, ",
    "DTD images do not include Florence-2 weights; ",
)
caption_header = """# === DTD: RESUMABLE FLORENCE-2 CAPTIONS WITH CLIP LEAKAGE AUDIT ===
import csv
import gc
import json
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

TASK_PROMPT = CAPTION_TASK_PROMPT
CSV_COLUMNS = [
    "image_id", "split", "raw_caption", "final_caption", "caption_valid",
    "leakage_detected", "lexical_leakage_detected", "semantic_leakage_detected",
    "semantic_leakage_score", "semantic_leakage_class", "retry_count", "vlm_model", "prompt_version",
]

"""
caption_run = """

CAPTION_CSV = OUT_PUT_DIR / 'captions.csv'
caption_records = []
for split_name, frame in (('train', train_df), ('val', val_df), ('test', test_df)):
    caption_records.extend({
        'image_id': str(row.image_id), 'image_path': str(row.image_path), 'split': split_name,
    } for row in frame[['image_id', 'image_path']].itertuples(index=False))
if len({row['image_id'] for row in caption_records}) != len(caption_records):
    raise RuntimeError('Duplicate image IDs in DTD caption records')

existing_captions = load_existing(CAPTION_CSV)
expected_splits = {row['image_id']: row['split'] for row in caption_records}
for image_id, old in existing_captions.items():
    if image_id not in expected_splits or old.get('split') != expected_splits[image_id]:
        raise ValueError(f'Existing captions.csv does not match DTD split {DTD_SPLIT}: {image_id}')
    if old.get('prompt_version') != PROMPT_VERSION:
        raise ValueError(f'Existing captions.csv has another prompt version for {image_id}; use a new output directory')
    if old.get('vlm_model') != VLM_MODEL:
        raise ValueError(f'Existing captions.csv uses another Florence model for {image_id}; use a new output directory')
todo_captions = [row for row in caption_records if row['image_id'] not in existing_captions]
print(f'DTD captions: cached={len(existing_captions):,}, remaining={len(todo_captions):,}, output={CAPTION_CSV}')

if todo_captions:
    if DEVICE.type != 'cuda':
        raise RuntimeError('Enable a GPU accelerator before generating all DTD captions and training ViT-L/16')
    clip_asset = resolve_local_model(CLIP_MODEL_LOCAL_PATH, 'clip')
    if clip_asset is None:
        raise FileNotFoundError('A complete local CLIP ViT-B/32 model is required; attach it as Kaggle Input or set CLIP_MODEL_LOCAL_PATH')
    CLIP_MODEL_LOCAL_PATH = str(clip_asset)
    florence_model = florence_processor = clip_auditor = clip_audit_tokenizer = None
    try:
        florence_model, florence_processor, florence_dtype = load_florence(
            FLORENCE_MODEL_REF, DEVICE, local_files_only=True
        )
        clip_auditor, clip_audit_tokenizer, class_vectors, class_labels = load_semantic_auditor(
            CLIP_MODEL_LOCAL_PATH, class_names, DEVICE, florence_dtype, local_files_only=True
        )
        forbidden_terms = build_forbidden_terms(class_names) if ENABLE_LEAKAGE_CHECK else []
        caption_loader = DataLoader(
            CaptionDataset(todo_captions), batch_size=CAPTION_BATCH_SIZE, shuffle=False,
            num_workers=CAPTION_WORKERS, persistent_workers=CAPTION_WORKERS > 0,
            collate_fn=collate_images,
        )
        effective_batch = CAPTION_BATCH_SIZE
        for batch in tqdm(caption_loader, total=len(caption_loader), desc='DTD Florence-2 captions'):
            for row, image, error in batch:
                if error:
                    raise RuntimeError(f"Cannot read {row['image_id']}: {error}")
            generated = []
            offset = 0
            while offset < len(batch):
                group = batch[offset:offset + effective_batch]
                try:
                    generated.extend(generate_batch(
                        florence_model, florence_processor, [image for _, image, _ in group],
                        DEVICE, florence_dtype, CAPTION_MAX_NEW_TOKENS,
                    ))
                    offset += len(group)
                except torch.cuda.OutOfMemoryError:
                    if effective_batch == 1:
                        raise
                    effective_batch = max(1, effective_batch // 2)
                    torch.cuda.empty_cache()
                    tqdm.write(f'Caption batch reduced to {effective_batch} after CUDA OOM')
            batch_rows = []
            final_texts = []
            for (row, _image, _error), raw in zip(batch, generated):
                final, lexical_valid, lexical_leak, _matched = audit_caption(raw, forbidden_terms)
                batch_rows.append({
                    'image_id': row['image_id'], 'split': row['split'], 'raw_caption': raw,
                    'final_caption': final, 'caption_valid': lexical_valid,
                    'leakage_detected': lexical_leak, 'lexical_leakage_detected': lexical_leak,
                    'semantic_leakage_detected': False, 'semantic_leakage_score': 0.0,
                    'semantic_leakage_class': '', 'retry_count': 0,
                    'vlm_model': VLM_MODEL, 'prompt_version': PROMPT_VERSION,
                })
                final_texts.append(final)
            semantic_results = semantic_audit_captions(
                final_texts, clip_auditor, clip_audit_tokenizer, class_vectors, class_labels,
                DEVICE, florence_dtype, CAPTION_SEMANTIC_THRESHOLD, batch_size=256,
            )
            for row, (score, matched_class, semantic_leak) in zip(batch_rows, semantic_results):
                row['semantic_leakage_score'] = round(score, 6)
                row['semantic_leakage_class'] = matched_class
                row['semantic_leakage_detected'] = semantic_leak
                row['leakage_detected'] = bool(row['leakage_detected'] or semantic_leak)
                row['caption_valid'] = bool(row['caption_valid'] and not semantic_leak)
            append_rows(CAPTION_CSV, batch_rows)
    finally:
        del florence_model, florence_processor, clip_auditor, clip_audit_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

completed_captions = load_existing(CAPTION_CSV)
if len(completed_captions) != len(caption_records):
    raise RuntimeError(f'Caption CSV is incomplete: {len(completed_captions)}/{len(caption_records)}; rerun this cell to resume')
print('Complete DTD captions:', len(completed_captions))
print('Valid captions:', sum(str(row['caption_valid']).lower() in {'true', '1'} for row in completed_captions.values()))
"""
cells.insert(7, code_cell(caption_header + caption_functions + caption_run))

# Indices from the source notebook now shift by one after the new caption cell.
caption_validation = cell_source(cells[8])
caption_validation = replace_block(caption_validation,
    'if not CAPTION_CSV.is_file():\n    if CAPTIONS_INPUT_PATH is not None:\n',
    'def _normalize_phrase(text):\n',
    "if not CAPTION_CSV.is_file():\n    raise FileNotFoundError(f'DTD caption CSV missing: {CAPTION_CSV}; run the generation cell above')\n\n",
    "old external CSV import",
)
caption_validation = replace_block(caption_validation,
    'if not USE_CACHED_CAPTIONS:\n', 'caption_df = pd.read_csv(CAPTION_CSV).fillna("")\n',
    "", "old standalone caption requirement")
caption_validation = replace_once(caption_validation,
    'Resume the standalone caption script, then rerun this cell.',
    'Rerun the caption generation cell above to resume.', "caption resume hint")
caption_validation = replace_once(caption_validation,
    '    if CAPTION_MAX_IMAGES_PER_SPLIT is not None:\n'
    '        samples = samples.groupby("split", group_keys=False).head(int(CAPTION_MAX_IMAGES_PER_SPLIT)).reset_index(drop=True)\n',
    '', "old SUN397 caption sample limit")
caption_validation = replace_once(caption_validation,
    '    "clip_model": CLIP_MODEL_NAME,\n',
    '    "clip_model": CLIP_MODEL_NAME,\n'
    '    "semantic_threshold": CAPTION_SEMANTIC_THRESHOLD,\n'
    '    "dtd_split": DTD_SPLIT,\n',
    "caption audit metadata",
)
set_source(cells[8], caption_validation)

embedding_cache = cell_source(cells[10])
embedding_cache = replace_once(embedding_cache,
    '                and len(candidate_cache["embeddings"]) == len(caption_ids)\n',
    '                and len(candidate_cache["embeddings"]) == len(caption_ids)\n'
    '                and torch.count_nonzero(candidate_cache["embeddings"][~caption_valid_mask]).item() == 0\n',
    "invalid text embedding cache mask",
)
embedding_cache = replace_once(embedding_cache,
    '    clip_source = str(CLIP_MODEL_LOCAL_PATH) if CLIP_MODEL_LOCAL_PATH else CLIP_MODEL_NAME\n',
    '    clip_asset = resolve_local_model(CLIP_MODEL_LOCAL_PATH, "clip")\n'
    '    if clip_asset is None:\n'
    '        raise FileNotFoundError("A complete local CLIP ViT-B/32 model is required to encode DTD captions")\n'
    '    CLIP_MODEL_LOCAL_PATH = str(clip_asset)\n'
    '    clip_source = CLIP_MODEL_LOCAL_PATH\n',
    "offline CLIP text encoder path",
)
set_source(cells[10], embedding_cache)

dataset = cell_source(cells[12])
dataset = dataset.replace('SUN397TACSQueryDataset', 'DTDTACSQueryDataset')
set_source(cells[12], dataset)

# Exclude a candidate image when the same training image is used as query.
model = cell_source(cells[14])
model = replace_once(model,
    '    def select_full_pool(self, query_images, candidate_feature_bank, use_gumbel=True):\n'
    '        selector_scores, query_features = self.score_full_candidate_pool(query_images, candidate_feature_bank)\n',
    '    def select_full_pool(self, query_images, candidate_feature_bank, excluded_positions=None, use_gumbel=True):\n'
    '        selector_scores, query_features = self.score_full_candidate_pool(query_images, candidate_feature_bank)\n'
    '        if excluded_positions is not None:\n'
    '            positions = excluded_positions.to(selector_scores.device, dtype=torch.long)\n'
    '            in_pool = positions >= 0\n'
    '            if in_pool.any():\n'
    '                selector_scores = selector_scores.clone()\n'
    '                selector_scores[torch.arange(len(positions), device=selector_scores.device)[in_pool], positions[in_pool]] = -torch.inf\n',
    "self-candidate exclusion",
)
set_source(cells[14], model)

helpers = cell_source(cells[15])
helpers += """

candidate_position_by_id = {str(image_id): index for index, image_id in enumerate(candidate_df['image_id'])}

def candidate_self_positions(image_ids):
    return torch.tensor(
        [candidate_position_by_id.get(str(image_id), -1) for image_id in image_ids],
        dtype=torch.long, device=DEVICE,
    )
"""
set_source(cells[15], helpers)

training = cell_source(cells[17])
training = replace_once(training,
    'details = network.select_full_pool(query, bank_device, use_gumbel=True)',
    'details = network.select_full_pool(query, bank_device, excluded_positions=candidate_self_positions(batch["image_id"]), use_gumbel=True)',
    "train self-candidate exclusion",
)
training = replace_once(training,
    'details = network.select_full_pool(query, bank_device, use_gumbel=False) if mode_uses_visual_selector(mode) else None',
    'details = network.select_full_pool(query, bank_device, excluded_positions=candidate_self_positions(batch["image_id"]), use_gumbel=False) if mode_uses_visual_selector(mode) else None',
    "eval self-candidate exclusion",
)
training = training.replace('"dataset": "SUN397"', '"dataset": "DTD"')
training = replace_once(training,
    '"val_fraction": VAL_FRACTION, "dino_model": DINOV3_MODEL_NAME,',
    '"dtd_split": DTD_SPLIT, "dino_model": DINOV3_MODEL_NAME,',
    "DTD config metadata",
)
training = replace_once(training,
    '"candidate_pool_size": len(candidate_df), "tacs_gumbel_tau": TACS_GUMBEL_TAU,',
    '"candidate_pool_size": len(candidate_df), "dtd_split": DTD_SPLIT, "tacs_gumbel_tau": TACS_GUMBEL_TAU,',
    "DTD checkpoint metadata",
)
set_source(cells[17], training)

for cell in cells:
    if cell["cell_type"] == "markdown":
        source = cell_source(cell)
        source = source.replace("SUN397", "DTD")
        source = source.replace("generate_sun397_captions.py", "the caption generation cell")
        set_source(cell, source)
    elif cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {TARGET}")
