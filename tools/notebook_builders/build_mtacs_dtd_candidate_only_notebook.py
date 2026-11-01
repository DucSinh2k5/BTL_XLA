"""Build standalone DTD ablation 03 with selected-candidate caption fusion only."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from pathlib import Path

import build_mtacs_dtd_query_only_notebook as query
import build_mtacs_dtd_visual_only_notebook as visual


ROOT = PROJECT_ROOT
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-03-candidate-only.ipynb')

INTRO = """# DTD ablation 03: candidate_only, ViT-L/16

Only `candidate_only` is trained: query image + one selected context image + that context image's caption.
Query-text fusion and semantic text selection are disabled. The visual selector chooses the context
using image features only; its selected index also gathers the aligned candidate-caption vector.
Candidate labels are never model/selector inputs. Invalid context captions fall back to image-only fusion.

This notebook retains the backbone, classifier architecture, augmentations, losses and optimizer
settings of notebooks 01/02. Defaults: official DTD split 1, seed 42, 50 epochs with a 50-epoch LR
schedule, no early stopping, and best checkpoint selected by validation loss. Train/val/test each
contain 1,880 images; the context pool contains 376 training images. A query cannot select itself.

Attach DTD, local DINOv3 code/ViT-L weights and CLIP ViT-B/32. This notebook reuses a complete
audited DTD captions.csv, or generates/resumes it using local Florence-2 if missing. Attach
`/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi` with Florence-2-base and compatible offline
wheels when generating captions. The legacy model runs in an isolated subprocess, leaving the
kernel's Transformers unchanged. All loading/installing uses local files, never the Internet.
CAPTIONS_INPUT_PATH can reuse notebook 02's CSV. Each completed generation batch is flushed.
The CSV covers the entire DTD split for reuse across ablations; the captioner sees pixels only.
Only training-pool caption vectors enter this run's classifier. Query/validation/test captions
never enter its forward pass. A complete existing CSV skips Florence entirely.

Use the same explicit CAPTION_INPUT_POLICY as notebook 02: legacy_strict reproduces the CLIP-0.30
hard gate; dtd_lexical_v2 cleans generic openings, screens direct class terms and uses CLIP only
as a diagnostic. The original CSV is preserved and actual input text/masks are exported separately.
No threshold is fitted from test scores. Candidate-pool caption coverage is shown before training; all-invalid pools stop early
with an explanation instead of silently running another image-only model.

Output is isolated in outputs/mtacs_dtd_split1_ablation_03_candidate_only_vitl16.
Resume uses this notebook's own last.pt, not a checkpoint from another ablation. Existing notebooks
and outputs are not overwritten. Run all cells in order with the same GPU/batch settings as 01/02.
"""

BANK_FUNCTION = """def make_ablation_banks(spec, seed, override=None):
    settings = {**spec, **(override or {})}
    if (settings["name"] != "candidate_only" or settings["view"] != "screened"
            or settings["query_fusion"] or not settings["candidate_fusion"] or settings["semantic_selector"]
            or settings["q_transform"] != "aligned" or settings["c_transform"] != "aligned"):
        raise ValueError("This notebook supports candidate_only only: candidate text on, query text/semantic selector off.")
    view = TEXT_CACHE["views"]["screened"]
    ids = view["image_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in the candidate caption cache")
    lookup = {image_id: index for index, image_id in enumerate(ids)}
    missing = set(CANDIDATE_IDS) - set(lookup)
    if missing:
        raise ValueError(f"Candidate caption cache is missing {len(missing)} pool images")
    positions = torch.tensor([lookup[image_id] for image_id in CANDIDATE_IDS], dtype=torch.long)
    query_ids = [str(image_id) for frame in SPLIT_FRAMES.values() for image_id in frame.image_id]
    return {"settings": settings, "q_index": {image_id: index for index, image_id in enumerate(query_ids)}, "singletons": {},
            "q": {"embeddings": torch.zeros(len(query_ids), CLIP_EMBED_DIM),
                  "valid": torch.zeros(len(query_ids), dtype=torch.bool),
                  "source_ids": query_ids, "texts": [""] * len(query_ids)},
            "c": {"embeddings": view["embeddings"].index_select(0, positions).to(DEVICE),
                  "valid": torch.as_tensor(view["valid"], dtype=torch.bool).index_select(0, positions).to(DEVICE),
                  "source_ids": [ids[int(index)] for index in positions],
                  "texts": [view["texts"][int(index)] for index in positions]}}
"""


def build_notebook():
    notebook = copy.deepcopy(query.build_notebook())
    code = {index: visual.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: INTRO}
    changes[1] = visual.replace_once(code[1], "Offline query-only preflight", "Offline candidate-only preflight")
    changes[2] = code[2].replace("query_only", "candidate_only").replace("DTD_QUERY_ONLY_SEED", "DTD_CANDIDATE_ONLY_SEED")
    changes[2] = visual.replace_once(changes[2], '"query_fusion": True', '"query_fusion": False')
    changes[2] = visual.replace_once(changes[2], '"candidate_fusion": False', '"candidate_fusion": True')
    changes[4] = code[4].replace("DTD_QUERY_ONLY_OUTPUT", "DTD_CANDIDATE_ONLY_OUTPUT").replace(
        "ablation_02_query_only_vitl16", "ablation_03_candidate_only_vitl16").replace("DTD_QUERY_ONLY_EPOCHS", "DTD_CANDIDATE_ONLY_EPOCHS")
    text_stage = query.TEXT_STAGE.split("def make_ablation_banks", 1)[0]
    text_stage = visual.replace_once(text_stage,
        'caption_df = load_ablation_captions(CAPTION_CSV, SPLIT_FRAMES)\ncaption_audit, caption_views = prepare_standalone_caption_views(\n'
        '    caption_df, class_names, CAPTION_SEMANTIC_THRESHOLD, CAPTION_INPUT_POLICY)',
        'caption_df = load_ablation_captions(CAPTION_CSV, SPLIT_FRAMES)\n'
        'pool_caption_df = caption_df.set_index("image_id").loc[CANDIDATE_IDS].reset_index()\n'
        'caption_audit, caption_views = prepare_standalone_caption_views(\n'
        '    pool_caption_df, class_names, CAPTION_SEMANTIC_THRESHOLD, CAPTION_INPUT_POLICY)')
    text_stage = visual.replace_once(text_stage, '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_text_ablation_vitl16"',
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_02_query_only_vitl16" / "shared" / "captions.csv",\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_text_ablation_vitl16"')
    text_stage = text_stage.replace(
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_03_candidate_only_vitl16" / "shared" / "captions.csv",\n', "")
    text_stage = text_stage.replace("caption_audit.csv", "candidate_caption_audit.csv").replace(
        "caption_audit_summary.csv", "candidate_caption_audit_summary.csv")
    text_stage = visual.replace_once(text_stage,
        'if not caption_audit.loc[caption_audit.split.eq("train"), "screened_valid"].any():\n'
        '    raise ValueError("No eligible training captions: query_only cannot learn a text branch. "\n'
        '                     "See shared/caption_filter_diagnostics_train.csv and caption_samples_train.csv. "\n'
        '                     "Do not regenerate a complete CSV; inspect raw_caption and the explicit caption-input policy.")',
        'if not caption_audit["screened_valid"].any():\n'
        '    raise ValueError("No eligible candidate-pool captions: candidate_only cannot learn its text branch. "\n'
        '                     "Inspect the training-pool caption diagnostics and CAPTION_INPUT_POLICY; do not force invalid masks on.")')
    text_stage = text_stage.replace("query_text_embeddings.pt", "candidate_text_embeddings.pt").replace(
        "Query-caption vectors:", "Training-pool candidate-caption vectors:")
    batch_inputs = visual.definitions(code[9], ["batch_caption_inputs"])
    changes[9] = text_stage + "\n\n" + BANK_FUNCTION + "\n\n" + batch_inputs
    changes[14] = code[14].replace("query_text_manifest.json", "candidate_text_manifest.json").replace(
        "query_text_assignments.csv", "candidate_text_assignments.csv")
    changes[14] = visual.replace_once(changes[14],
        'for role, ids in (("q", TEXT_CACHE["views"][spec["view"]]["image_ids"]), ("c", CANDIDATE_IDS)):',
        'for role, ids in (("q", banks["q"]["source_ids"]), ("c", CANDIDATE_IDS)):')
    changes[17] = code[17].replace("query_only", "candidate_only").replace("query-only comparison", "candidate-only comparison")
    changes[17] = visual.replace_once(changes[17], '"text_inputs": "screened_candidate_only",',
        '"text_inputs": "screened_candidate_only", "text_scope": "training_candidate_pool_only",')
    changes[23] = code[23].replace("query_only", "candidate_only")
    changes[23] = visual.replace_once(changes[23],
        "Query image + one context image + screened query text. Candidate text and semantic selector are disabled.",
        "Query image + one context image + its screened caption. Query text and semantic selector are disabled.")
    changes[23] = visual.replace_once(changes[23],
        "Same classifier architecture as notebook 01; Florence creates/resumes CSV if needed, then frozen CLIP caches vectors.",
        "Same classifier architecture as notebooks 01/02; Florence creates/resumes CSV if needed, then CLIP encodes pool captions only.")
    changes[23] = visual.replace_once(changes[23],
        "Query gate is learned for valid captions; candidate gate is always zero. The caption policy is recorded in the protocol.",
        "Query gate is always zero; candidate gate is learned for valid selected captions. The caption policy is recorded in the protocol.")
    changes[23] = changes[23].replace("## Screened caption coverage", "## Candidate-pool caption coverage")
    changes[7] = "## Create/resume Florence captions.csv if needed; encode training-pool captions only\n"
    changes[10] = "## Same visual backbone and classifier, selected-candidate text gate enabled\n"
    changes[18] = "## Train candidate_only only: 50 epochs by default\n"
    changes[20] = "## Final test after candidate_only finishes\n"
    changes[22] = "## Result table, learning curves and candidate-pool caption coverage\n"
    helpers = code[8] + changes[9] + code[11] + code[12] + changes[14] + code[15]
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = (utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\n"
                  f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n")
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = cell["id"].replace("dtd-query-only-", "dtd-candidate-only-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    notebook["metadata"]["dtd_ablation"]["experiment"] = "candidate_only"
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
