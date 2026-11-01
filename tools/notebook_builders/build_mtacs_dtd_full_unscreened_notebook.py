"""Build DTD ablation 07 with aligned captions before leakage filtering."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from pathlib import Path

import build_mtacs_dtd_full_both_notebook as full
import build_mtacs_dtd_visual_only_notebook as visual


ROOT = PROJECT_ROOT
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-07-full-unscreened.ipynb')

INTRO = """# DTD ablation 07: full_unscreened, ViT-L/16

Train only `full_unscreened`: query image + its aligned pre-filter caption + one selected context
image + that context image's aligned pre-filter caption. There is no caption shuffle or redaction.
The classifier, both text-fusion gates and visual+semantic selector match notebook 05 exactly.
Score = visual_z + sigmoid(semantic_mix_logit) * semantic_z; the mixing coefficient starts at 0.5
and is learned with the same policy/entropy objective. The frozen CLIP encoder embeds the actual
unscreened texts. The chosen image index gathers its own caption, never a different donor image.

"Before filtering" means BEFORE LEAKAGE MASKS, not the full long Florence paragraph: construct
exactly the same cleaned 5-8-word text as notebook 05, preferring raw_caption and falling back to
final_caption when necessary. With dtd_lexical_v2, remove generic openings before truncation.
Only nonempty 5-8-word caption availability is enforced. Do not reject or alter text because of
caption_valid, lexical/semantic leakage flags, class terms, aliases, CLIP scores or whether a
redacted counterpart would be empty. Do not invent captions for missing/short text. Unavailable
captions still have zero vectors/gates; their images remain valid visual candidates.

CAPTION_INPUT_POLICY defaults to dtd_lexical_v2 for SAME TEXT CONSTRUCTION and reference auditing
as notebook 05; its screened rejection mask is NOT applied to this experiment. view=unscreened is
recorded separately in the protocol. Both roles and the selector use the unscreened availability
mask. The original CSV flags/scores remain unchanged for diagnostics, including markers when a
legacy score refers to text before cleanup. Source-ID alignment is preserved for all images.

This is an intentionally unscreened shortcut/leakage diagnostic, not a label-safe model claim.
Caption generation still receives image pixels only, never per-image labels or filenames. No
ground-truth class text is injected into captions. DTD texture words can legitimately describe
pixels and also name classes. A higher score than 05 does not by itself prove label leakage:
screened and unscreened runs also differ in caption coverage. shared/caption_filter_comparison.csv
reports both masks, restored caption counts and retained class terms for query splits/pool.
It is not a fixed-mask class-word intervention; report these coverage differences with metrics.

Default: official DTD split 1, seed 42, 50 epochs with a 50-epoch LR schedule, no early stopping,
best checkpoint selected by validation loss. Keep notebook 05's data, pool, initialization,
augmentation, losses and optimizer settings. Train/val/test each have 1,880 images; the fixed pool
contains 376 training images. A query cannot select itself. Validation/test images and captions
never enter the candidate pool. Their own image-derived query captions are available at inference.
Final test opens only after this configuration completes its declared epoch budget.

Attach DTD, local DINOv3 code/ViT-L weights and CLIP ViT-B/32. Reuse a complete DTD captions.csv
from notebook 05/06/04/02/03, or create/resume it with local Florence-2, flushing completed batches.
The CSV retains raw text even when generation-time audit flags reject it; those flags do not mask
this run. For offline generation, attach `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi`
with Florence-2-base and compatible offline wheels. Legacy Florence uses an isolated subprocess;
there are no network downloads or kernel downgrades. Configure CAPTIONS_INPUT_PATH,
FLORENCE_MODEL_REF and OFFLINE_ASSETS_ROOT for other local paths. A complete CSV skips generation.

The source CSV is never rewritten. shared/caption_model_inputs.csv records the actual input text,
unscreened masks and notebook-05 reference masks. CLIP caches the unscreened view separately from
screened caches. Each run exports unscreened_text_assignments.csv and unscreened_text_manifest.json.
Logs/history/metrics retain selector_semantic_mix. Results/checkpoints are isolated in
outputs/mtacs_dtd_split1_ablation_07_full_unscreened_vitl16; only this run's last.pt resumes it.
Previous notebooks/outputs are unchanged. Run all cells with the same GPU/batch setup as 05.
"""

UNSCREENED_HELPERS = """def prepare_unscreened_caption_views(frame, names, threshold, policy):
    audit, views = prepare_standalone_caption_views(frame, names, threshold, policy)
    available = audit.word_count.between(5, 8).to_numpy() & audit.normalized_caption.ne("").to_numpy()
    audit["unscreened_valid"] = available
    audit["restored_by_unscreened"] = available & ~audit.screened_valid.to_numpy()
    audit["caption_view"] = "unscreened"
    audit["leakage_masks_applied"] = False
    audit["model_text_source"] = "final_caption_fallback"
    if policy == "dtd_lexical_v2" and "raw_caption" in frame:
        has_raw = frame.raw_caption.fillna("").astype(str).str.strip().ne("").to_numpy()
        audit.loc[has_raw, "model_text_source"] = "raw_caption_cleaned_and_truncated"
    views["unscreened"] = {"image_ids": audit.image_id.astype(str).tolist(),
        "texts": audit.final_caption.astype(str).tolist(), "valid": available.copy()}
    return audit, views


def unscreened_filter_comparison(audit, candidate_ids):
    groups = [("query", split, frame) for split, frame in audit.groupby("split", sort=False)]
    groups.append(("candidate_pool", "train", audit.loc[audit.image_id.isin(candidate_ids)]))
    rows = []
    for role, split, frame in groups:
        available = frame.unscreened_valid
        rows.append({"role": role, "split": split, "images": len(frame),
            "screened_reference_count": int(frame.screened_valid.sum()),
            "unscreened_count": int(available.sum()),
            "additional_usable_count": int(frame.restored_by_unscreened.sum()),
            "kept_direct_class_terms": int((available & frame.exact_class_term_hit).sum()),
            "kept_alias_or_class_terms": int((available & frame.alias_or_class_term_hit).sum())})
    return pd.DataFrame(rows)
"""

TEXT_STAGE = """caption_df = load_ablation_captions(CAPTION_CSV, SPLIT_FRAMES)
caption_audit, caption_views = prepare_unscreened_caption_views(
    caption_df, class_names, CAPTION_SEMANTIC_THRESHOLD, CAPTION_INPUT_POLICY)
caption_audit.to_csv(SHARED_DIR / "caption_audit.csv", index=False)
caption_audit[["image_id", "split", "final_caption", "unscreened_valid", "screened_valid",
               "input_caption_policy", "caption_view", "leakage_masks_applied", "model_text_source"]].rename(
    columns={"final_caption": "model_caption", "unscreened_valid": "model_caption_valid",
             "screened_valid": "screened_reference_valid"}
).to_csv(SHARED_DIR / "caption_model_inputs.csv", index=False)
caption_comparison = unscreened_filter_comparison(caption_audit, CANDIDATE_IDS)
caption_comparison.to_csv(SHARED_DIR / "caption_filter_comparison.csv", index=False)
audit_summary = caption_audit.groupby("split").agg(
    count=("image_id", "size"), valid_captions=("unscreened_valid", "sum"),
    valid_rate=("unscreened_valid", "mean"), class_term_rate=("exact_class_term_hit", "mean"),
    screened_reference_count=("screened_valid", "sum"), restored_captions=("restored_by_unscreened", "sum"))
audit_summary.to_csv(SHARED_DIR / "caption_audit_summary.csv")
train_audit = caption_audit.loc[caption_audit.split.eq("train")]
train_audit[["image_id", "raw_caption", "final_caption", "word_count", "screened_valid", "unscreened_valid",
             "exact_class_term_hit", "semantic_leakage_score", "semantic_score_matches_input"]
            if "raw_caption" in train_audit else
            ["image_id", "final_caption", "word_count", "screened_valid", "unscreened_valid",
             "exact_class_term_hit", "semantic_leakage_score", "semantic_score_matches_input"]].head(12).to_csv(
                 SHARED_DIR / "unscreened_samples_train.csv", index=False)
print("Aligned pre-leakage-filter captions:", CAPTION_CSV)
print("Text preprocessing:", CAPTION_INPUT_POLICY, "| view=unscreened | leakage masks NOT applied")
display(audit_summary)
display(caption_comparison)
print("Coverage changes are part of this experiment; higher accuracy alone is not evidence of label leakage.")
if not train_audit.unscreened_valid.any():
    raise ValueError("No available pre-filter training captions: inspect raw_caption/final_caption for empty or short text.")
pool_caption_audit = caption_audit.loc[caption_audit.image_id.isin(CANDIDATE_IDS)]
role_coverage = pd.concat([
    caption_audit[["split", "unscreened_valid"]].assign(role="query"),
    pool_caption_audit[["split", "unscreened_valid"]].assign(role="candidate_pool"),
], ignore_index=True).groupby(["role", "split"]).agg(
    count=("unscreened_valid", "size"), valid_captions=("unscreened_valid", "sum"),
    valid_rate=("unscreened_valid", "mean"))
role_coverage.to_csv(SHARED_DIR / "caption_role_coverage.csv")
display(role_coverage)
if not pool_caption_audit.unscreened_valid.any():
    raise ValueError("No available pre-filter candidate-pool captions: inspect missing/short pool captions before training.")
clip_asset = resolve_clip_path(CLIP_MODEL_LOCAL_PATH, [Path("/kaggle/input"), PROJECT_DIR])
TEXT_CACHE = encode_view_cache({"unscreened": caption_views["unscreened"]}, clip_asset, SHARED_DIR / "unscreened_text_embeddings.pt")
CLIP_EMBED_DIM = int(TEXT_CACHE["embedding_dim"])
if CLIP_EMBED_DIM != 512:
    raise ValueError("Expected 512-dimensional CLIP vectors to match the other ablations.")
print("Unscreened query/pool vectors:", tuple(TEXT_CACHE["views"]["unscreened"]["embeddings"].shape))
"""


def build_notebook():
    notebook = copy.deepcopy(full.build_notebook())
    code = {index: visual.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: INTRO}
    changes[1] = code[1].replace("full-both", "full-unscreened")
    config = code[2].replace("full_both", "full_unscreened").replace("DTD_FULL_BOTH_SEED", "DTD_FULL_UNSCREENED_SEED")
    changes[2] = visual.replace_once(config, '"view": "screened"', '"view": "unscreened"')
    changes[4] = code[4].replace("DTD_FULL_BOTH_OUTPUT", "DTD_FULL_UNSCREENED_OUTPUT").replace(
        "ablation_05_full_both_vitl16", "ablation_07_full_unscreened_vitl16").replace("DTD_FULL_BOTH_EPOCHS", "DTD_FULL_UNSCREENED_EPOCHS")
    changes[8] = code[8] + "\n\n" + UNSCREENED_HELPERS
    prefix = code[9].split("caption_df = load_ablation_captions", 1)[0]
    prefix = visual.replace_once(prefix,
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n',
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_05_full_both_vitl16" / "shared" / "captions.csv",\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_06_full_shuffle_both_vitl16" / "shared" / "captions.csv",\n')
    bank_functions = visual.definitions(code[9], ["make_ablation_banks", "batch_caption_inputs"])
    bank_functions = bank_functions.replace("full_both", "full_unscreened").replace('["screened"]', '["unscreened"]').replace(
        'settings["view"] != "screened"', 'settings["view"] != "unscreened"')
    changes[9] = prefix + TEXT_STAGE + "\n\n" + bank_functions
    changes[14] = code[14].replace("full_text_manifest.json", "unscreened_text_manifest.json").replace(
        "full_text_assignments.csv", "unscreened_text_assignments.csv")
    changes[14] = visual.replace_once(changes[14], '"view": spec["view"], "singletons_not_deranged": banks["singletons"],',
        '"view": spec["view"], "singletons_not_deranged": banks["singletons"],\n'
        '        "leakage_masks_applied": False, "availability_rule": "cleaned_nonempty_5_to8_words",')
    protocol = code[17].replace("full_both", "full_unscreened").replace("full-both comparison", "full-unscreened comparison")
    protocol = visual.replace_once(protocol, '"text_inputs": "screened_full_unscreened",',
        '"text_inputs": "aligned_pre_leakage_filter_captions",')
    changes[17] = visual.replace_once(protocol, '"caption_input_policy": CAPTION_INPUT_POLICY,',
        '"caption_input_policy": CAPTION_INPUT_POLICY, "caption_view": "unscreened",\n'
        '    "leakage_masks_applied": False, "availability_rule": "cleaned_nonempty_5_to8_words",\n'
        '    "screened_reference": "notebook_05_same_text_construction", "caption_assignment": "aligned_no_shuffle",')
    report = code[23].replace("full_both", "full_unscreened")
    report = visual.replace_once(report,
        "Query image + query caption + one context image + its caption. Both text gates are enabled; selection combines image and caption similarities.",
        "Aligned pre-leakage-filter query/context captions are used for both fusion and multimodal selection. No shuffle/redaction.")
    report = visual.replace_once(report,
        "Both gates are learned and independently masked for invalid captions. The caption policy is recorded in the protocol.",
        "Both gates use caption availability only, not CSV validity/leakage flags or class-term/CLIP masks.")
    report = report.replace("## Screened caption coverage", "## Unscreened caption availability")
    report += '\nif "caption_comparison" in globals():\n    with (OUT_PUT_DIR / "full_unscreened_report.md").open("a", encoding="utf-8") as handle:\n        handle.write("\\n## Leakage filters bypassed\\n\\n" + markdown_table(caption_comparison) + "\\n\\nText is constructed as in 05, but its leakage mask is not applied. Missing/short text is still unavailable. This comparison also changes coverage; higher scores do not alone prove leakage.\\n")\n'
    changes[23] = report
    changes[7] = "## Create/resume captions.csv; retain aligned pre-leakage-filter text for both roles\n"
    changes[10] = "## Same fusion and multimodal selector as 05; unscreened availability masks\n"
    changes[18] = "## Train full_unscreened only: 50 epochs by default\n"
    changes[20] = "## Final unscreened-control test after the declared epoch budget\n"
    changes[22] = "## Metrics, filter/coverage comparison, learned semantic mix and learning curves\n"
    helpers = changes[8] + changes[9] + code[11] + code[12] + changes[14] + code[15]
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = (utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\n"
                  f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n")
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = cell["id"].replace("dtd-full-both-", "dtd-full-unscreened-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    notebook["metadata"]["dtd_ablation"].update(experiment="full_unscreened",
        default_caption_input_policy="dtd_lexical_v2", default_caption_view="unscreened", leakage_masks_applied=False)
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
