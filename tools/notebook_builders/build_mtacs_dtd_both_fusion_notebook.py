"""Build DTD ablation 04: both caption-fusion roles, image-only selection."""
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
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-04-both-fusion.ipynb')

INTRO = """# DTD ablation 04: both_fusion, ViT-L/16

Only `both_fusion` is trained: query image + query caption + one selected context image + that
context image's caption. Both text-fusion gates are enabled; the selector uses image features only.
There is no text similarity in candidate selection. The selected image index also gathers its
aligned caption vector. Candidate labels are never classifier/selector inputs.

The architecture, classifier head, visual selector, losses, augmentations and optimizer settings
are unchanged from notebooks 01/02/03. Defaults: official DTD split 1, seed 42, 50 epochs with a
50-epoch LR schedule, no early stopping, best checkpoint selected by validation loss. Each split
has 1,880 images. The fixed pool contains 376 training images; a query cannot select itself.
Validation/test images and their captions never become candidates. Their own image-derived query
captions are available for classification, using the same image-only caption pipeline as training.

CAPTION_INPUT_POLICY defaults to `dtd_lexical_v2`, matching the updated notebooks 02/03. Generic
openings are removed before retaining 5-8 words from raw_caption. Direct DTD class names are
screened; CLIP audit scores are diagnostic, not a hard gate or a calibrated leak probability.
Legacy scores are marked when they refer to text before cleanup. This does not prove absence of
synonyms/implicit class information. `legacy_strict` is an explicit alternative experiment, not
interchangeable with the default policy. Do not mix policies when comparing ablation scores.

Attach DTD, local DINOv3 code/ViT-L weights and CLIP ViT-B/32. This notebook reuses a complete
audited DTD captions.csv from notebooks 02/03 when available. Otherwise it creates/resumes the CSV
with local Florence-2, flushing each finished batch. Caption generation sees image pixels only,
never per-image labels or filenames. Attach `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi`
with Florence-2-base and compatible offline wheels if generation is needed. Legacy Florence runs
in an isolated subprocess; all dependencies/models stay offline and the kernel is not downgraded.
Use CAPTIONS_INPUT_PATH, FLORENCE_MODEL_REF and OFFLINE_ASSETS_ROOT for different attached paths.

The original CSV is preserved. shared/caption_model_inputs.csv records actual text and masks;
CLIP caches vectors for those texts. Candidate vectors are gathered only from the training pool.
Invalid captions mask their own text gate to zero independently; valid text in the other role
remains usable. All-invalid query/pool branches stop before training rather than silently changing
the intended ablation. Coverage for each role is exported in shared/caption_role_coverage.csv.

Output is isolated in outputs/mtacs_dtd_split1_ablation_04_both_fusion_vitl16. Resume uses this
notebook's own last.pt; checkpoints from another ablation are not imported or overwritten.
Run all cells in order with the same GPU/batch settings as notebooks 01/02/03. Final test opens
only after this configuration completes its declared epoch budget. Select settings on validation.
"""

BANK_FUNCTION = """def make_ablation_banks(spec, seed, override=None):
    settings = {**spec, **(override or {})}
    if (settings["name"] != "both_fusion" or settings["view"] != "screened"
            or not settings["query_fusion"] or not settings["candidate_fusion"] or settings["semantic_selector"]
            or settings["q_transform"] != "aligned" or settings["c_transform"] != "aligned"):
        raise ValueError("This notebook supports both_fusion only: both text gates on, semantic selector off.")
    view = TEXT_CACHE["views"]["screened"]
    ids = view["image_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in the caption cache")
    lookup = {image_id: index for index, image_id in enumerate(ids)}
    query_ids = {str(image_id) for frame in SPLIT_FRAMES.values() for image_id in frame.image_id}
    if query_ids - set(lookup):
        raise ValueError("Caption cache is missing query images")
    training_ids = set(SPLIT_FRAMES["train"].image_id.astype(str))
    if set(CANDIDATE_IDS) - training_ids:
        raise ValueError("Candidate text bank must contain training images only")
    if set(CANDIDATE_IDS) - set(lookup):
        raise ValueError("Caption cache is missing training-pool images")
    positions = torch.tensor([lookup[image_id] for image_id in CANDIDATE_IDS], dtype=torch.long)
    valid = torch.as_tensor(view["valid"], dtype=torch.bool)
    return {"settings": settings, "q_index": lookup, "singletons": {},
            "q": {"embeddings": view["embeddings"], "valid": valid,
                  "source_ids": list(ids), "texts": view["texts"]},
            "c": {"embeddings": view["embeddings"].index_select(0, positions).to(DEVICE),
                  "valid": valid.index_select(0, positions).to(DEVICE),
                  "source_ids": [ids[int(index)] for index in positions],
                  "texts": [view["texts"][int(index)] for index in positions]}}
"""

ROLE_COVERAGE = """pool_caption_audit = caption_audit.loc[caption_audit.image_id.isin(CANDIDATE_IDS)]
role_coverage = pd.concat([
    caption_audit[["split", "screened_valid"]].assign(role="query"),
    pool_caption_audit[["split", "screened_valid"]].assign(role="candidate_pool"),
], ignore_index=True).groupby(["role", "split"]).agg(
    count=("screened_valid", "size"), valid_captions=("screened_valid", "sum"),
    valid_rate=("screened_valid", "mean"))
role_coverage.to_csv(SHARED_DIR / "caption_role_coverage.csv")
display(role_coverage)
if not pool_caption_audit.screened_valid.any():
    raise ValueError("No eligible candidate-pool captions: both_fusion would silently become query_only. "
                     "Inspect the pool audit and CAPTION_INPUT_POLICY before training.")
"""


def build_notebook():
    notebook = copy.deepcopy(query.build_notebook())
    code = {index: visual.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: INTRO}
    changes[1] = code[1].replace("query-only", "both-fusion")
    changes[2] = code[2].replace("query_only", "both_fusion").replace("DTD_QUERY_ONLY_SEED", "DTD_BOTH_FUSION_SEED")
    changes[2] = visual.replace_once(changes[2], '"candidate_fusion": False', '"candidate_fusion": True')
    changes[4] = code[4].replace("DTD_QUERY_ONLY_OUTPUT", "DTD_BOTH_FUSION_OUTPUT").replace(
        "ablation_02_query_only_vitl16", "ablation_04_both_fusion_vitl16").replace("DTD_QUERY_ONLY_EPOCHS", "DTD_BOTH_FUSION_EPOCHS")
    text_stage = query.TEXT_STAGE.split("def make_ablation_banks", 1)[0]
    text_stage = visual.replace_once(text_stage,
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n',
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_02_query_only_vitl16" / "shared" / "captions.csv",\n')
    text_stage = text_stage.replace("query_only cannot learn a text branch", "both_fusion cannot learn its query-text branch")
    text_stage = visual.replace_once(text_stage, "clip_asset = resolve_clip_path(CLIP_MODEL_LOCAL_PATH,",
        ROLE_COVERAGE + "\nclip_asset = resolve_clip_path(CLIP_MODEL_LOCAL_PATH,")
    text_stage = text_stage.replace("query_text_embeddings.pt", "both_text_embeddings.pt").replace(
        "Query-caption vectors:", "Shared query/pool-caption vectors:")
    changes[9] = text_stage + "\n\n" + BANK_FUNCTION + "\n\n" + visual.definitions(code[9], ["batch_caption_inputs"])
    changes[14] = code[14].replace("query_text_manifest.json", "both_text_manifest.json").replace(
        "query_text_assignments.csv", "both_text_assignments.csv")
    changes[17] = code[17].replace("query_only", "both_fusion").replace("query-only comparison", "both-fusion comparison")
    changes[17] = visual.replace_once(changes[17], '"text_inputs": "screened_both_fusion",',
        '"text_inputs": "screened_both_fusion", "text_scope": "query_and_training_pool",')
    report = code[23].replace("query_only", "both_fusion")
    report = visual.replace_once(report,
        "Query image + one context image + screened query text. Candidate text and semantic selector are disabled.",
        "Query image + query caption + one context image + its caption. Both text gates are enabled; selection is image-only.")
    report = visual.replace_once(report,
        "Query gate is learned for valid captions; candidate gate is always zero. The caption policy is recorded in the protocol.",
        "Both gates are learned and independently masked for invalid captions. The caption policy is recorded in the protocol.")
    report += '\nif "role_coverage" in globals():\n    display(role_coverage)\n    with (OUT_PUT_DIR / "both_fusion_report.md").open("a", encoding="utf-8") as handle:\n        handle.write("\\n## Query and candidate caption coverage\\n\\n" + markdown_table(role_coverage.reset_index()) + "\\n")\n'
    changes[23] = report
    changes[7] = "## Create/resume Florence captions.csv; cache text for both fusion roles\n"
    changes[10] = "## Same classifier: both text gates enabled, image-only selector\n"
    changes[18] = "## Train both_fusion only: 50 epochs by default\n"
    changes[20] = "## Final test after both_fusion finishes\n"
    changes[22] = "## Result table, learning curves and coverage of both text roles\n"
    helpers = code[8] + changes[9] + code[11] + code[12] + changes[14] + code[15]
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = (utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\n"
                  f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n")
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = cell["id"].replace("dtd-query-only-", "dtd-both-fusion-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    notebook["metadata"]["dtd_ablation"].update(experiment="both_fusion", default_caption_input_policy="dtd_lexical_v2")
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
