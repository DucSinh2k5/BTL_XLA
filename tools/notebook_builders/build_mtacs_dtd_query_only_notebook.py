"""Build standalone DTD ablation 02 with screened query-caption fusion only."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from pathlib import Path

import build_mtacs_dtd_visual_only_notebook as visual
import dtd_caption_generation_cells as captions


ROOT = PROJECT_ROOT
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-02-query-only.ipynb')

INTRO = """# DTD ablation 02: query_only, ViT-L/16

Only `query_only` is trained: query image + one selected context image + screened query caption.
The query-text fusion gate is enabled. Candidate-text fusion and semantic text selection are disabled.
The selector uses image features only; candidate labels/captions are not selector inputs.

This notebook retains the architecture, visual selector, losses, augmentations, split, candidate pool
and optimizer settings from notebook 01. Default: official DTD split 1, seed 42, 50 epochs with a
50-epoch LR schedule and no early stopping; best checkpoint is selected by validation loss.
DTD partitions contain 1,880 images each; the pool contains 376 training images and excludes the query itself.

Attach DTD, local DINOv3 code/ViT-L weights and a local CLIP ViT-B/32 folder. The caption cell
reuses a complete audited DTD `captions.csv`, or creates/resumes it with local Florence-2 when missing.
Set `CAPTIONS_INPUT_PATH` to reuse another run's CSV. SUN397 captions are rejected.
For generation, attach `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi` with `Florence-2-base`
and its offline wheels. Set `FLORENCE_MODEL_REF` / `OFFLINE_ASSETS_ROOT` if your paths differ.
The legacy checkpoint runs in a separate Python process with offline Transformers 4.49 dependencies
when necessary; the notebook kernel is not downgraded and no restart or Internet is requested.
Compatible wheels for the current Python/platform are required. A complete CSV skips Florence entirely.
Generation receives image pixels only, never per-image labels or filenames. Each finished batch is flushed.

CAPTION_INPUT_POLICY defaults to dtd_lexical_v2; legacy_strict remains available for comparison.
Legacy strict reproduces
the existing length/lexical/CLIP-0.30 hard gate. DTD lexical v2 removes only the generic opening
before retaining 5-8 words from raw_caption, rejects direct class terms, and treats CLIP as a
diagnostic rather than a calibrated leak detector. It never restores an unexplained invalid caption.
Both policies preserve the original CSV. caption_model_inputs.csv records actual input text/masks;
CLIP is re-encoded for those texts. Old CLIP audit scores are marked when their text no longer matches.
Neither policy proves that synonyms or implicit class information have been removed.
Invalid captions have zero vectors and a zero query gate; their images remain in training.
The valid-caption rate is printed before training. The CLIP encoder is released after caching embeddings.

Output is isolated in `outputs/mtacs_dtd_split1_ablation_02_query_only_vitl16`.
Resume uses this notebook's own last.pt; old suite/visual-only checkpoints are not imported or overwritten.
Run all cells in order. Use validation for development, not repeated test-score tuning.
"""

CONFIG_EXTRA = """
CAPTIONS_INPUT_PATH = os.getenv("DTD_CAPTIONS_INPUT_PATH", "")  # Exact attached DTD captions.csv if needed.
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
CAPTION_SEMANTIC_THRESHOLD = 0.30  # Fixed screening policy; never auto-tuned using test scores.
CAPTION_INPUT_POLICY = os.getenv("DTD_CAPTION_INPUT_POLICY", "dtd_lexical_v2")  # Or legacy_strict; use the same policy across text ablations.
""" + captions.CONFIG

ASSET_FUNCTIONS = """def find_dtd_caption_csv(split_frames, explicit_path, cached_path, roots, allow_partial=False, allow_missing=False):
    def checked(path):
        return caption_progress(path, split_frames) if allow_partial else load_ablation_captions(path, split_frames)

    cached_path = Path(cached_path)
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DTD_CAPTIONS_INPUT_PATH not found: {path}")
        checked(path)
        if cached_path.is_file() and path.resolve() != cached_path.resolve() and path.read_bytes() != cached_path.read_bytes():
            raise ValueError("Requested captions differ from the cached CSV; use a new output directory.")
        return path
    if cached_path.is_file():
        checked(cached_path)
        return cached_path
    visited = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = []
            for current, dirs, files in os.walk(root):
                dirs[:] = sorted(name for name in dirs if name.lower() not in {"images", "__pycache__", ".cache"})
                if len(Path(current).relative_to(root).parts) >= 8:
                    dirs[:] = []
                candidates.extend(Path(current) / name for name in sorted(files)
                                  if "caption" in name.lower() and name.lower().endswith(".csv"))
        else:
            continue
        for path in candidates:
            key = str(path.resolve())
            if key in visited:
                continue
            visited.add(key)
            try:
                load_ablation_captions(path, split_frames)
            except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            return path
    if allow_missing:
        return None
    raise FileNotFoundError(
        "No complete audited DTD captions.csv found. Attach the previous DTD notebook output and set "
        "CAPTIONS_INPUT_PATH to its exact CSV, or enable local Florence generation."
    )


def resolve_clip_path(model_ref, roots):
    requested = Path(model_ref).expanduser()
    candidates = [requested] if requested.is_dir() else []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name.lower() not in {"images", "__pycache__", ".cache"})
            if len(Path(current).relative_to(root).parts) >= 7:
                dirs[:] = []
            if "config.json" in files:
                candidates.append(Path(current))
    for path in candidates:
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if config.get("model_type") == "clip" and int(config.get("projection_dim", 512)) == 512:
            return path
    raise FileNotFoundError(
        "Attach the complete local CLIP ViT-B/32 folder and set CLIP_MODEL_LOCAL_PATH. No models are downloaded."
    )
"""

TEXT_STAGE = """SHARED_DIR = OUT_PUT_DIR / "shared"
RUNS_DIR = OUT_PUT_DIR / "runs"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FRAMES = {"train": train_df, "val": val_df, "test": test_df}
CANDIDATE_IDS = candidate_df.image_id.astype(str).tolist()
CANDIDATE_POSITION = {image_id: index for index, image_id in enumerate(CANDIDATE_IDS)}
CAPTION_CSV = SHARED_DIR / "captions.csv"
caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [
    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_03_candidate_only_vitl16" / "shared" / "captions.csv",
    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_text_ablation_vitl16" / "shared" / "captions.csv",
    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_captions_vitl16" / "captions.csv",
    Path("/kaggle/input"),
])
caption_df = load_ablation_captions(CAPTION_CSV, SPLIT_FRAMES)
caption_audit, caption_views = prepare_standalone_caption_views(
    caption_df, class_names, CAPTION_SEMANTIC_THRESHOLD, CAPTION_INPUT_POLICY)
caption_audit.to_csv(SHARED_DIR / "caption_audit.csv", index=False)
caption_audit[["image_id", "split", "final_caption", "screened_valid", "input_caption_policy"]].rename(
    columns={"final_caption": "model_caption", "screened_valid": "model_caption_valid"}
).to_csv(SHARED_DIR / "caption_model_inputs.csv", index=False)
caption_filter_diagnostics(caption_audit, CAPTION_SEMANTIC_THRESHOLD, SHARED_DIR)
audit_summary = caption_audit.groupby("split").agg(
    count=("image_id", "size"), valid_captions=("screened_valid", "sum"),
    valid_rate=("screened_valid", "mean"), class_term_rate=("exact_class_term_hit", "mean"))
audit_summary.to_csv(SHARED_DIR / "caption_audit_summary.csv")
print("Audited DTD captions ready:", CAPTION_CSV)
print("Caption input policy:", CAPTION_INPUT_POLICY)
display(audit_summary)
if not caption_audit.loc[caption_audit.split.eq("train"), "screened_valid"].any():
    raise ValueError("No eligible training captions: query_only cannot learn a text branch. "
                     "See shared/caption_filter_diagnostics_train.csv and caption_samples_train.csv. "
                     "Do not regenerate a complete CSV; inspect raw_caption and the explicit caption-input policy.")
if audit_summary.valid_rate.min() < 0.10:
    print("WARNING: low screened-caption coverage. Report coverage with results; do not tune the threshold on test.")
clip_asset = resolve_clip_path(CLIP_MODEL_LOCAL_PATH, [Path("/kaggle/input"), PROJECT_DIR])
TEXT_CACHE = encode_view_cache({"screened": caption_views["screened"]}, clip_asset, SHARED_DIR / "query_text_embeddings.pt")
CLIP_EMBED_DIM = int(TEXT_CACHE["embedding_dim"])
if CLIP_EMBED_DIM != 512:
    raise ValueError("Expected 512-dimensional CLIP vectors to match notebook 01's classifier architecture.")
print("Query-caption vectors:", tuple(TEXT_CACHE["views"]["screened"]["embeddings"].shape))


def make_ablation_banks(spec, seed, override=None):
    settings = {**spec, **(override or {})}
    if (settings["name"] != "query_only" or settings["view"] != "screened" or not settings["query_fusion"]
            or settings["candidate_fusion"] or settings["semantic_selector"]
            or settings["q_transform"] != "aligned" or settings["c_transform"] != "aligned"):
        raise ValueError("This notebook supports query_only only: query text on, candidate text/semantic selector off.")
    view = TEXT_CACHE["views"]["screened"]
    ids = view["image_ids"]
    return {"settings": settings, "q_index": {image_id: index for index, image_id in enumerate(ids)}, "singletons": {},
            "q": {"embeddings": view["embeddings"], "valid": view["valid"],
                  "source_ids": ids, "texts": view["texts"]},
            "c": {"embeddings": torch.zeros(len(CANDIDATE_IDS), CLIP_EMBED_DIM, device=DEVICE),
                  "valid": torch.zeros(len(CANDIDATE_IDS), dtype=torch.bool, device=DEVICE),
                  "source_ids": list(CANDIDATE_IDS), "texts": [""] * len(CANDIDATE_IDS)}}
"""


def build_notebook():
    base = visual.build_notebook()
    code = {index: visual.source_of(cell) for index, cell in enumerate(base["cells"])}
    suite = json.loads(visual.SOURCE.read_text(encoding="utf-8"))
    suite_code = {index: visual.source_of(cell) for index, cell in enumerate(suite["cells"])}
    preflight = visual.replace_once(visual.PREFLIGHT, '"timm": "timm", "einops": "einops"',
                                    '"timm": "timm", "einops": "einops", "transformers": "transformers"')
    preflight = visual.replace_once(preflight, "Offline visual-only preflight passed; no caption models are required.",
                                    "Offline query-only preflight passed; caption cell can create/resume CSV with local Florence.")
    config = visual.CONFIG.replace("visual_only", "query_only").replace("DTD_VISUAL_ONLY_SEED", "DTD_QUERY_ONLY_SEED")
    config = visual.replace_once(config, '"view": "disabled"', '"view": "screened"')
    config = visual.replace_once(config, '"query_fusion": False', '"query_fusion": True')
    config = visual.replace_once(config, "Only the width of the masked text layers, not a loaded model.",
                                "Must match notebook 01; checked against the local CLIP text encoder.")
    config += CONFIG_EXTRA
    settings = code[4].replace("DTD_VISUAL_ONLY_OUTPUT", "DTD_QUERY_ONLY_OUTPUT").replace(
        "ablation_01_visual_only_vitl16", "ablation_02_query_only_vitl16").replace("DTD_VISUAL_ONLY_EPOCHS", "DTD_QUERY_ONLY_EPOCHS")
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    utilities = "import shutil\n" + utilities
    text_functions = suite_code[10]
    for name in ("deranged_sources", "make_ablation_banks", "batch_caption_inputs"):
        text_functions = visual.replace_once(text_functions, visual.definitions(suite_code[10], [name]).strip(), "")
    text_functions += "\n\n" + ASSET_FUNCTIONS + "\n\n" + captions.embedded_helpers()
    text_stage = TEXT_STAGE + "\n\n" + visual.definitions(suite_code[10], ["batch_caption_inputs"])
    training = code[12].replace("disabled_text_manifest.json", "query_text_manifest.json").replace(
        "disabled_text_assignments.csv", "query_text_assignments.csv")
    protocol = visual.PROTOCOL.replace("dtd_visual_only_v1", "dtd_query_only_v1").replace(
        "Only visual_only will run:", "Only query_only will run:").replace("visual-only comparison", "query-only comparison")
    protocol = visual.replace_once(protocol, '"text_inputs": "disabled_zero_masked",',
        '"text_inputs": "screened_query_only", "semantic_threshold": CAPTION_SEMANTIC_THRESHOLD,\n'
        '    "caption_input_policy": CAPTION_INPUT_POLICY,\n'
        '    "caption_csv_sha256": hashlib.sha256(CAPTION_CSV.read_bytes()).hexdigest(),\n'
        '    "caption_embedding_cache": TEXT_CACHE["fingerprint"],')
    report = visual.REPORT.replace("visual_only", "query_only")
    report = visual.replace_once(report,
        "Query image + one training-pool context image. No query/candidate text or semantic selector.",
        "Query image + one context image + screened query text. Candidate text and semantic selector are disabled.")
    report = visual.replace_once(report,
        "The masked text layers retain the suite classifier architecture; no caption models are loaded.",
        "Same classifier architecture as notebook 01; Florence creates/resumes CSV if needed, then frozen CLIP caches vectors.")
    report = visual.replace_once(report,
        "Query/candidate gates are zero by design. Caption coverage is not measured in this notebook.",
        "Query gate is learned for valid captions; candidate gate is always zero. The caption policy is recorded in the protocol.")
    report += '\nif "audit_summary" in globals():\n    display(audit_summary)\n    with (OUT_PUT_DIR / "query_only_report.md").open("a", encoding="utf-8") as handle:\n        handle.write("\\n## Screened caption coverage\\n\\n" + markdown_table(audit_summary.reset_index()) + "\\n")\n'
    report += '\nwith (OUT_PUT_DIR / "query_only_report.md").open("a", encoding="utf-8") as handle:\n    handle.write(f"\\nCaption input policy: {CAPTION_INPUT_POLICY}. Semantic scores are diagnostic-only for dtd_lexical_v2. Policy changes define different experiments; do not mix their protocols.\\n")\n'
    helper_hash = hashlib.sha256((text_functions + text_stage + code[9] + code[10] + training + code[13]).encode()).hexdigest()
    utilities += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\nHELPER_SOURCE_SHA256 = {helper_hash!r}\n"
    sections = [
        ("markdown", INTRO), ("code", preflight), ("code", config),
        ("markdown", "## Same official split, pool and hyperparameters as notebook 01"), ("code", settings),
        ("markdown", "## Shared utilities"), ("code", utilities),
        ("markdown", "## Create/resume Florence captions.csv if needed, screen captions and cache frozen CLIP vectors"),
        ("code", text_functions), ("code", text_stage),
        ("markdown", "## Same visual backbone and classifier, query-text gate enabled"), ("code", code[9]), ("code", code[10]),
        ("markdown", "## Training and evaluation functions"), ("code", training), ("code", code[13]),
        ("markdown", "## Protocol and candidate image bank"), ("code", protocol),
        ("markdown", "## Train query_only only: 50 epochs by default"), ("code", code[17]),
        ("markdown", "## Final test after query_only finishes"), ("code", code[19]),
        ("markdown", "## Result table, learning curves and caption coverage"), ("code", report),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": copy.deepcopy(base["metadata"]["kernelspec"]),
                             "language_info": {"name": "python"}, "accelerator": "GPU",
                             "dtd_ablation": {"experiment": "query_only", "schema": 1, "source": visual.SOURCE.name,
                                              "default_epochs": 50, "caption_generation": True}},
                "cells": [visual.cell(kind, source, index) for index, (kind, source) in enumerate(sections)]}
    for cell in notebook["cells"]:
        cell["id"] = cell["id"].replace("dtd-visual-only-", "dtd-query-only-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
