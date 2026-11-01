"""Build the standalone visual-only run without changing the existing DTD suite."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = PROJECT_ROOT
SOURCE = project_path('notebookb7a9dba884.ipynb')
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-01-visual-only.ipynb')


def source_of(cell):
    return "".join(cell.get("source", []))


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected exactly one source marker: {before!r}")
    return source.replace(before, after, 1)


def definitions(source, names):
    nodes = {node.name: ast.get_source_segment(source, node)
             for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    missing = set(names) - set(nodes)
    if missing:
        raise ValueError(f"Source definitions missing: {sorted(missing)}")
    return "\n\n\n".join(nodes[name] for name in names) + "\n"


def cell(kind, source, index):
    result = {"id": f"dtd-visual-only-{index:02d}", "cell_type": kind, "metadata": {},
              "source": (source.strip() + "\n").splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


INTRO = """# DTD ablation 01: visual_only, ViT-L/16

This notebook trains only `visual_only`: query image + one selected context image.
Query text fusion, candidate text fusion and semantic text selection are all disabled.
There is no caption generation, caption CSV, CLIP loading, Florence-2 loading, or Transformers pinning.

The visual selector, query/context image fusion, classifier head, losses, augmentations and
optimizer settings are copied from the existing DTD ablation notebook. The unused text layers
remain in the model to keep the same architecture and initialization as the other ablations;
their inputs are zeros and their gates are masked to zero. This is the suite's visual baseline,
not a claim of an exact reproduction of every choice in the original TACS paper.

Official DTD split 1 is used by default: 1,880 train / 1,880 validation / 1,880 test images.
Only 20% of training images form the candidate pool (376 images); a query cannot select itself.
Default training: seed 42, 50 epochs, no early stopping, best checkpoint selected by validation loss.
Fresh runs use a 50-epoch LR schedule; later standalone ablations must use the same horizon.
Do not mix their scores with an older checkpoint trained on a 100-epoch LR schedule.

Attach DTD, the local DINOv3 code and ViT-L/16 weights, then run all cells in order with a GPU.
Output is isolated in `outputs/mtacs_dtd_split1_ablation_01_visual_only_vitl16`.
Resume uses this notebook's own `last.pt`; reruns skip completed training and reuse cached test results.
The older suite's caption-dependent checkpoint protocol is not silently imported or overwritten.
Keep the older notebook/output separately if you need to resume its existing checkpoint.
"""

PREFLIGHT = """import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import importlib.util
_required = {"numpy": "numpy", "pandas": "pandas", "torch": "torch",
             "torchvision": "torchvision", "PIL": "Pillow", "matplotlib": "matplotlib",
             "tqdm": "tqdm", "sklearn": "scikit-learn", "timm": "timm", "einops": "einops"}
_missing = [package for module, package in _required.items() if importlib.util.find_spec(module) is None]
if _missing:
    raise RuntimeError("Missing local packages: " + ", ".join(_missing))
print("Offline visual-only preflight passed; no caption models are required.")
"""

CONFIG = """import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm

EXPERIMENT_MODE = "visual_only"
DTD_SPLIT = int(os.getenv("DTD_SPLIT", "1"))
TRAIN_SEEDS = [int(os.getenv("DTD_VISUAL_ONLY_SEED", "42"))]
ABLATION_NAMES = ["visual_only"]
SELECTED_ABLATIONS = [{"name": "visual_only", "view": "disabled",
                      "q_transform": "aligned", "c_transform": "aligned",
                      "query_fusion": False, "candidate_fusion": False,
                      "semantic_selector": False}]
RESUME_ABLATIONS = True
EARLY_STOPPING = False
KEEP_LAST_CHECKPOINT = False  # Retain best.pt; keep last.pt while a run is unfinished.
RUN_FINAL_TEST = True  # Set False for validation-only development.
BOOTSTRAP_REPEATS = 200
CLIP_EMBED_DIM = 512  # Only the width of the masked text layers, not a loaded model.
"""

MASKED_INPUTS = """RUNS_DIR = OUT_PUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FRAMES = {"train": train_df, "val": val_df, "test": test_df}
CANDIDATE_IDS = candidate_df.image_id.astype(str).tolist()
CANDIDATE_POSITION = {image_id: index for index, image_id in enumerate(CANDIDATE_IDS)}
_image_ids = [str(image_id) for frame in SPLIT_FRAMES.values() for image_id in frame.image_id]
# Shape-compatible placeholders keep the original forward graph without tokenization.
TEXT_CACHE = {"embedding_dim": CLIP_EMBED_DIM, "views": {"disabled": {
    "image_ids": _image_ids, "texts": [""] * len(_image_ids),
    "embeddings": torch.zeros(len(_image_ids), CLIP_EMBED_DIM),
    "valid": torch.zeros(len(_image_ids), dtype=torch.bool),
}}}


def make_ablation_banks(spec, seed, override=None):
    settings = {**spec, **(override or {})}
    if settings["name"] != "visual_only" or any(
        settings[key] for key in ("query_fusion", "candidate_fusion", "semantic_selector")
    ):
        raise ValueError("This notebook supports visual_only only; use a separate notebook for text ablations.")
    view = TEXT_CACHE["views"]["disabled"]
    ids = view["image_ids"]
    lookup = {image_id: index for index, image_id in enumerate(ids)}
    candidate_positions = torch.tensor([lookup[image_id] for image_id in CANDIDATE_IDS], dtype=torch.long)
    return {"settings": settings, "q_index": lookup, "singletons": {},
            "q": {"embeddings": view["embeddings"], "valid": view["valid"],
                  "source_ids": ids, "texts": view["texts"]},
            "c": {"embeddings": view["embeddings"][candidate_positions].to(DEVICE),
                  "valid": view["valid"][candidate_positions].to(DEVICE),
                  "source_ids": list(CANDIDATE_IDS), "texts": [""] * len(CANDIDATE_IDS)}}
"""

PROTOCOL = """if EPOCHS < 1 or PATIENCE < 1 or BOOTSTRAP_REPEATS < 0:
    raise ValueError("Invalid training settings")
if CONTEXT_TOP_K != 1:
    raise ValueError("The visual-only comparison uses one context image per query")
ensure_dinov3_assets()
weights_path = resolve_dinov3_weights()
protocol = {
    "schema": "dtd_visual_only_v1", "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "helper_source_sha256": HELPER_SOURCE_SHA256, "dtd_split": DTD_SPLIT, "pool_seed": SEED,
    "split_image_ids": {split: frame.image_id.astype(str).tolist() for split, frame in SPLIT_FRAMES.items()},
    "candidate_ids": CANDIDATE_IDS, "class_names": class_names,
    "weights_path": str(weights_path), "weights_bytes": Path(weights_path).stat().st_size,
    "weights_mtime_ns": Path(weights_path).stat().st_mtime_ns,
    "preprocessing": {"train": repr(train_transform), "eval": repr(eval_transform)},
    "text_inputs": "disabled_zero_masked", "masked_text_dimension": CLIP_EMBED_DIM,
    "training": {name: globals()[name] for name in (
        "DINOV3_MODEL_NAME", "FEATURE_DIM", "IMAGE_SIZE", "CONTEXT_TOP_K", "CANDIDATE_POOL_FRACTION",
        "EPOCHS", "BATCH_SIZE", "NUM_WORKERS", "BACKBONE_LR", "HEAD_LR", "TACS_SELECTOR_LR", "BACKBONE_WEIGHT_DECAY",
        "HEAD_WEIGHT_DECAY", "TACS_SELECTOR_WEIGHT_DECAY", "LABEL_SMOOTHING", "TOKEN_HEAD_DROPOUT",
        "FREEZE_BACKBONE_EPOCHS", "WARMUP_EPOCHS", "PATIENCE", "GRAD_CLIP_NORM", "SUPCON_WEIGHT", "SUPCON_TEMPERATURE",
        "TACS_POLICY_WEIGHT", "TACS_ENTROPY_WEIGHT", "TACS_SELECTOR_TEMPERATURE",
        "QUERY_CONTEXT_GATE_DROPOUT", "QUERY_CONTEXT_GATE_BIAS")},
    "selector_sampling": "categorical_policy_on_visual_z", "checkpoint_selection": "minimum_validation_cross_entropy",
    "test_hyperparameter_tuning": False,
}
protocol = resolve_training_protocol(protocol)
LR_SCHEDULE_EPOCHS = int(protocol["training"]["EPOCHS"])
PROTOCOL_FINGERPRINT = stable_hash(protocol)
ensure_training_budget()
write_json(OUT_PUT_DIR / "protocol.json", protocol)
plan = {"protocol": PROTOCOL_FINGERPRINT, "experiments": SELECTED_ABLATIONS, "seeds": TRAIN_SEEDS}
plan_path = OUT_PUT_DIR / "experiment_plan.json"
if (OUT_PUT_DIR / "locked_test_plan.json").is_file() and plan_path.is_file() and read_json(plan_path) != plan:
    raise ValueError("The final-test plan is already locked")
write_json(plan_path, plan)
write_json(OUT_PUT_DIR / "environment.json", {
    "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None,
})

AblationModel = make_ablation_model_class(MTACSModel, DinoV3TACSClassifier)
candidate_image_bank, _candidate_labels_unused, candidate_fnames = build_candidate_image_bank(
    candidate_df, eval_transform, device=torch.device(CANDIDATE_IMAGE_BANK_DEVICE), dtype=CANDIDATE_IMAGE_BANK_DTYPE)
del _candidate_labels_unused
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("Only visual_only will run:", EPOCHS, "epochs; seed(s):", TRAIN_SEEDS)
print("Output:", OUT_PUT_DIR)
display(pd.DataFrame(SELECTED_ABLATIONS))
"""

REPORT = """validation_summary = summarize_ablation("val")
test_summary = summarize_ablation("test") if RUN_FINAL_TEST else pd.DataFrame()
status = ablation_run_status()
display(status)
result_tables = [frame.assign(split=name) for name, frame in (("val", validation_summary), ("test", test_summary))
                 if not frame.empty]
results = pd.concat(result_tables, ignore_index=True) if result_tables else pd.DataFrame(
    columns=["experiment", "seed", "split", "epochs_trained", "best_epoch", "accuracy", "macro_f1", "loss"])
results.to_csv(OUT_PUT_DIR / "visual_only_results.csv", index=False)
display(results.round(4))


def markdown_table(frame):
    if frame.empty:
        return "No completed metrics yet."
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
              for row in frame.round(5).fillna("").itertuples(index=False, name=None)]
    return "\\n".join(lines)


report = [f"# DTD split {DTD_SPLIT}: visual_only", "",
          f"Epoch budget: {EPOCHS}; LR schedule horizon: {LR_SCHEDULE_EPOCHS}; early stopping: {EARLY_STOPPING}.",
          "Query image + one training-pool context image. No query/candidate text or semantic selector.",
          "The masked text layers retain the suite classifier architecture; no caption models are loaded.", "",
          "## Run status", "", markdown_table(status), "", "## Metrics", "", markdown_table(results), "",
          "Candidate pool contains training images only; self-selection is excluded.",
          "Query/candidate gates are zero by design. Caption coverage is not measured in this notebook.",
          "This is one ablation result, not the comparison of all eight configurations.",
          "Keep split, pool seed, training seed, preprocessing, budget and LR horizon fixed in later notebooks.",
          "Use validation for development; do not select later variants by repeated test-score tuning."]
(OUT_PUT_DIR / "visual_only_report.md").write_text("\\n".join(report) + "\\n", encoding="utf-8")

histories = []
for seed in TRAIN_SEEDS:
    path = RUNS_DIR / "visual_only" / f"seed_{seed}" / "history.csv"
    if path.is_file():
        try:
            history = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not history.empty:
            histories.append((seed, history))
if histories:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for seed, history in histories:
        axes[0].plot(history.epoch, history.train_loss, label=f"train seed {seed}")
        axes[0].plot(history.epoch, history.val_loss, label=f"val seed {seed}")
        axes[1].plot(history.epoch, history.val_macro_f1, label=f"seed {seed}")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Validation macro-F1")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.savefig(OUT_PUT_DIR / "visual_only_training.png", dpi=160, bbox_inches="tight")
    plt.show()
print("Saved results:", OUT_PUT_DIR / "visual_only_results.csv")
print("Saved report:", OUT_PUT_DIR / "visual_only_report.md")
"""


def build_notebook():
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    cells = {index: source_of(value) for index, value in enumerate(source["cells"])}
    settings = replace_once(cells[5], "os.getenv('DTD_ABLATION_OUTPUT', str(PROJECT_DIR / 'outputs' / f'mtacs_dtd_split{DTD_SPLIT}_text_ablation_vitl16'))",
                            "os.getenv('DTD_VISUAL_ONLY_OUTPUT', str(PROJECT_DIR / 'outputs' / f'mtacs_dtd_split{DTD_SPLIT}_ablation_01_visual_only_vitl16'))")
    settings = replace_once(settings, 'EPOCHS = int(os.getenv("DTD_ABLATION_EPOCHS", "50"))',
                            'EPOCHS = int(os.getenv("DTD_VISUAL_ONLY_EPOCHS", "50"))')
    settings = replace_once(settings, "CAPTION_BATCH_SIZE = int(os.getenv('CAPTION_BATCH_SIZE', '64' if GPU_MEMORY_GB >= 40 else '16'))\n", "")
    utilities = cells[9].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    for unused in ("from sklearn.linear_model import LogisticRegression\n", "from sklearn.pipeline import make_pipeline\n",
                   "from sklearn.preprocessing import StandardScaler\n"):
        utilities = replace_once(utilities, unused, "")
    masked_inputs = MASKED_INPUTS + "\n\n" + definitions(cells[10], ["batch_caption_inputs"])
    training = replace_once(cells[15], 'run_dir / "caption_assignment_manifest.json"', 'run_dir / "disabled_text_manifest.json"')
    training = replace_once(training, 'run_dir / "caption_source_assignments.csv"', 'run_dir / "disabled_text_assignments.csv"')
    evaluation = definitions(cells[16], ["evaluate_locked_test", "summarize_ablation"])
    evaluation = replace_once(evaluation, '    caption_label_diagnostics("test")\n', "")
    helper_hash = hashlib.sha256((masked_inputs + cells[13] + cells[14] + training + evaluation).encode()).hexdigest()
    utilities += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(raw).hexdigest()!r}\nHELPER_SOURCE_SHA256 = {helper_hash!r}\n"
    sections = [
        ("markdown", INTRO), ("code", PREFLIGHT), ("code", CONFIG),
        ("markdown", "## Official DTD split, fixed candidate pool and training settings"), ("code", settings),
        ("markdown", "## Shared utilities and disabled text placeholders"), ("code", utilities), ("code", masked_inputs),
        ("markdown", "## Same visual model, selector and masked fusion architecture as the suite"),
        ("code", cells[13]), ("code", cells[14]),
        ("markdown", "## Training and evaluation functions; no caption probes or text interventions"),
        ("code", training), ("code", evaluation),
        ("markdown", "## Protocol and candidate image bank"), ("code", PROTOCOL),
        ("markdown", "## Train visual_only only: 50 epochs by default"), ("code", cells[21]),
        ("markdown", "## Final test after this visual_only run completes"), ("code", cells[25]),
        ("markdown", "## Result table, learning curves and report"), ("code", REPORT),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": copy.deepcopy(source["metadata"]["kernelspec"]),
                             "language_info": {"name": "python"}, "accelerator": "GPU",
                             "dtd_ablation": {"experiment": "visual_only", "schema": 1, "source": SOURCE.name,
                                              "default_epochs": 50, "caption_generation": False}},
                "cells": [cell(kind, text, index) for index, (kind, text) in enumerate(sections)]}
    for value in notebook["cells"]:
        if value["cell_type"] == "code":
            ast.parse(source_of(value))
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
