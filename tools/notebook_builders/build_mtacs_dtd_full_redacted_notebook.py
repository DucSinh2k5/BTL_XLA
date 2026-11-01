"""Build DTD ablation 08: redact class names/aliases in both caption roles."""
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
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-08-full-redacted.ipynb')

INTRO = """# DTD ablation 08: full_redacted, ViT-L/16

Train only `full_redacted`: query image + its redacted caption + one selected context image +
that context image's redacted caption. Both text-fusion gates and the visual+semantic selector
match notebook 05. CLIP embeds the REDACTED text for BOTH fusion and semantic selection. There
is no caption shuffle. Selecting an image gathers its own caption by the same candidate index.
Score = visual_z + sigmoid(semantic_mix_logit) * semantic_z; the trainable mix starts at 0.5.

CAPTION_INPUT_POLICY defaults to dtd_lexical_v2: use the same cleaned 5-8-word text as notebooks
05/07, preferring raw_caption and falling back to final_caption. Redact BEFORE rejecting text
for class terms: otherwise class-bearing captions would already be absent. Remove all official
class names and the fixed DTD_EXTRA_TERMS aliases, case-insensitively at word boundaries. Apply
the SAME complete vocabulary to every image, never only that image's ground-truth class. Repeat
removal if deleting a term exposes another compound alias. Do not inject labels, mask tokens,
new caption words or regenerated text. The source CSV and its audit flags remain unchanged.

This is a redacted view, not notebook 05's screened mask. Availability requires a 5-8-word
pre-redaction caption and nonempty normalized text AFTER redaction. A remaining 1-4-word caption
stays available: do not impose another minimum length or refill it with later words. If redaction
removes every word, use a zero vector/gate. Missing/short source captions stay unavailable. CSV
caption_valid/leakage flags and CLIP leakage-audit thresholds do not discard surviving text.
The CLIP audit scores refer to original/legacy text, NOT the redacted input, and are diagnostic.

The primary content control is notebook 07 full_unscreened with the SAME CSV, split, seed and
training settings. Source availability is identical except when redaction empties the text.
shared/caption_redaction_comparison.csv reports this coverage difference for each query split
and the training candidate pool. This is not a fixed-mask causal estimate if coverage differs;
no extra matched-mask run is trained automatically. Compare to 05 with its different screened
coverage as well. DTD class words are also genuine visual texture attributes, so a drop in scores
does not by itself prove illicit label leakage. Finite aliases cannot remove all implied class
information. Freeze the vocabulary before evaluation; never tune aliases using test results.

Default: official DTD split 1, seed 42, 50 epochs, a 50-epoch LR schedule, no early stopping.
Best checkpoint is chosen by validation loss. Data, initialization, architecture, losses,
augmentation, optimizer and one-context selection match 05. Train/val/test each contain 1,880
images; the fixed candidate pool contains 376 training images only. A query cannot select
itself. Held-out images/captions never enter the pool. Their own image-derived, identically
redacted query captions are available at inference. Candidate labels are not model inputs.
Final test opens only after this run completes its declared epoch budget.

Attach DTD, local DINOv3 code/ViT-L weights and CLIP ViT-B/32. Reuse a complete original DTD
captions.csv from 07/06/05/04/02/03, or create/resume it offline with local Florence-2, flushing
completed batches. Do not pass caption_model_inputs.csv as the source: it is not the original
audited CSV. Generation receives pixels only, never per-image labels or filenames. Attach
`/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi` with Florence-2-base and compatible wheels
for offline legacy generation. It uses an isolated subprocess, no downloads/kernel downgrade.
Configure CAPTIONS_INPUT_PATH, FLORENCE_MODEL_REF and OFFLINE_ASSETS_ROOT if paths differ.
A complete CSV skips Florence. TOKENIZERS_PARALLELISM=false is set before tokenizer imports.

Exports: shared/caption_model_inputs.csv (before/after text and actual masks), caption_audit.csv,
caption_redaction_vocabulary.json, caption_redaction_comparison.csv and redacted CLIP vectors.
Each run exports redacted_text_assignments.csv and redacted_text_manifest.json. Coverage,
query/candidate gates and selector_semantic_mix are recorded with metrics. Outputs/checkpoints
are isolated in outputs/mtacs_dtd_split1_ablation_08_full_redacted_vitl16. Resume uses this run's
own last.pt only; previous notebooks/checkpoints are unchanged. Run all cells from top to bottom.
"""

REDACTION_HELPERS = r'''def redact_caption_terms(text, pattern):
    import re
    current = str(text).strip()
    removed = []
    while pattern.search(current):
        removed.extend(match.group(0) for match in pattern.finditer(current))
        current = re.sub(r"\s+", " ", pattern.sub(" ", current)).strip()
    return current, removed


def prepare_redacted_caption_views(frame, names, threshold, policy):
    audit, views = prepare_standalone_caption_views(frame, names, threshold, policy)
    pattern = caption_term_pattern(names)
    pairs = [redact_caption_terms(text, pattern) for text in audit.final_caption]
    redacted = [text for text, removed in pairs]
    source_available = audit.word_count.between(5, 8).to_numpy() & audit.normalized_caption.ne("").to_numpy()
    surviving = np.array([bool(normalized_text(text)) for text in redacted])
    available = source_available & surviving
    audit["pre_redaction_available"] = source_available
    audit["redacted_caption"] = redacted
    audit["redacted_word_count"] = [len(normalized_text(text).split()) for text in redacted]
    audit["redacted_valid"] = available
    audit["emptied_by_redaction"] = source_available & ~surviving
    audit["redaction_changed"] = audit.final_caption.ne(audit.redacted_caption)
    audit["redacted_removed_terms"] = [" | ".join(removed) for text, removed in pairs]
    audit["remaining_class_or_alias_hit"] = [bool(pattern.search(text)) for text in redacted]
    if audit.remaining_class_or_alias_hit.any():
        raise ValueError("Class/alias redaction left a known term in model input")
    audit["caption_view"] = "redacted"
    audit["model_text_source"] = "final_caption_fallback"
    if policy == "dtd_lexical_v2" and "raw_caption" in frame:
        has_raw = frame.raw_caption.fillna("").astype(str).str.strip().ne("").to_numpy()
        audit.loc[has_raw, "model_text_source"] = "raw_caption_cleaned_and_truncated"
    audit["semantic_score_matches_redacted_input"] = audit.semantic_score_matches_input & ~audit.redaction_changed
    views["redacted"] = {"image_ids": audit.image_id.astype(str).tolist(),
        "texts": redacted, "valid": available.copy()}
    return audit, views


def redaction_comparison(audit, candidate_ids):
    groups = [("query", split, frame) for split, frame in audit.groupby("split", sort=False)]
    groups.append(("candidate_pool", "train", audit.loc[audit.image_id.isin(candidate_ids)]))
    rows = []
    for role, split, frame in groups:
        rows.append({"role": role, "split": split, "images": len(frame),
            "screened_reference_count": int(frame.screened_valid.sum()),
            "pre_redaction_available": int(frame.pre_redaction_available.sum()),
            "redacted_available": int(frame.redacted_valid.sum()),
            "emptied_by_redaction": int(frame.emptied_by_redaction.sum()),
            "changed_available_captions": int((frame.pre_redaction_available & frame.redaction_changed).sum()),
            "remaining_class_or_alias_hits": int(frame.remaining_class_or_alias_hit.sum())})
    return pd.DataFrame(rows)
'''

TEXT_STAGE = '''caption_df = load_ablation_captions(CAPTION_CSV, SPLIT_FRAMES)
caption_audit, caption_views = prepare_redacted_caption_views(
    caption_df, class_names, CAPTION_SEMANTIC_THRESHOLD, CAPTION_INPUT_POLICY)
caption_audit.to_csv(SHARED_DIR / "caption_audit.csv", index=False)
caption_audit[["image_id", "split", "final_caption", "redacted_caption", "redacted_valid",
               "pre_redaction_available", "screened_valid", "emptied_by_redaction", "redacted_removed_terms",
               "input_caption_policy", "caption_view", "model_text_source",
               "semantic_score_matches_redacted_input"]].rename(columns={
    "final_caption": "pre_redaction_caption", "redacted_caption": "model_caption",
    "redacted_valid": "model_caption_valid", "screened_valid": "screened_reference_valid"
}).to_csv(SHARED_DIR / "caption_model_inputs.csv", index=False)
redaction_vocabulary = {"class_names": list(class_names),
    "aliases": {name: list(DTD_EXTRA_TERMS.get(name, [])) for name in class_names},
    "pattern": caption_term_pattern(class_names).pattern,
    "rule": "case_insensitive_word_boundary_deletion_until_no_match", "uses_per_image_labels": False}
write_json(SHARED_DIR / "caption_redaction_vocabulary.json", redaction_vocabulary)
caption_comparison = redaction_comparison(caption_audit, CANDIDATE_IDS)
caption_comparison.to_csv(SHARED_DIR / "caption_redaction_comparison.csv", index=False)
audit_summary = caption_audit.groupby("split").agg(
    count=("image_id", "size"), valid_captions=("redacted_valid", "sum"),
    valid_rate=("redacted_valid", "mean"), pre_redaction_count=("pre_redaction_available", "sum"),
    changed_captions=("redaction_changed", "sum"), emptied_captions=("emptied_by_redaction", "sum"))
audit_summary.to_csv(SHARED_DIR / "caption_audit_summary.csv")
train_audit = caption_audit.loc[caption_audit.split.eq("train")]
sample_columns = [name for name in ("image_id", "raw_caption", "final_caption", "redacted_caption",
    "word_count", "redacted_word_count", "redacted_valid", "redacted_removed_terms",
    "semantic_leakage_score", "semantic_score_matches_redacted_input") if name in train_audit]
train_audit[sample_columns].head(12).to_csv(SHARED_DIR / "redacted_samples_train.csv", index=False)
print("Aligned redacted captions:", CAPTION_CSV)
print("Preprocessing:", CAPTION_INPUT_POLICY, "| view=redacted | vocabulary applies to every image")
display(audit_summary)
display(caption_comparison)
print("After redaction, 1-4 remaining words are allowed; empty text is masked, never refilled.")
print("Stored CLIP leakage scores are legacy diagnostics, not scores of the redacted inputs.")
if not train_audit.redacted_valid.any():
    raise ValueError("No usable redacted training captions: inspect missing/short source text or text emptied by redaction.")
pool_caption_audit = caption_audit.loc[caption_audit.image_id.isin(CANDIDATE_IDS)]
role_coverage = pd.concat([
    caption_audit[["split", "redacted_valid"]].assign(role="query"),
    pool_caption_audit[["split", "redacted_valid"]].assign(role="candidate_pool"),
], ignore_index=True).groupby(["role", "split"]).agg(
    count=("redacted_valid", "size"), valid_captions=("redacted_valid", "sum"),
    valid_rate=("redacted_valid", "mean"))
role_coverage.to_csv(SHARED_DIR / "caption_role_coverage.csv")
display(role_coverage)
if not pool_caption_audit.redacted_valid.any():
    raise ValueError("No usable redacted candidate-pool captions: inspect missing/short or emptied captions before training.")
clip_asset = resolve_clip_path(CLIP_MODEL_LOCAL_PATH, [Path("/kaggle/input"), PROJECT_DIR])
TEXT_CACHE = encode_view_cache({"redacted": caption_views["redacted"]}, clip_asset, SHARED_DIR / "redacted_text_embeddings.pt")
CLIP_EMBED_DIM = int(TEXT_CACHE["embedding_dim"])
if CLIP_EMBED_DIM != 512:
    raise ValueError("Expected 512-dimensional CLIP vectors to match the other ablations.")
print("Redacted query/pool vectors:", tuple(TEXT_CACHE["views"]["redacted"]["embeddings"].shape))
'''


def build_notebook():
    notebook = copy.deepcopy(full.build_notebook())
    code = {index: visual.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: INTRO, 1: code[1].replace("full-both", "full-redacted")}
    config = code[2].replace("full_both", "full_redacted").replace("DTD_FULL_BOTH_SEED", "DTD_FULL_REDACTED_SEED")
    config = config.replace("# Fixed screening policy; never auto-tuned using test scores.",
                            "# Legacy audit/reference only; not a gate for redacted inputs.")
    config = config.replace("# Or legacy_strict; use the same policy across text ablations.",
                            "# Same preprocessing as 05/07; redacted view sets availability.")
    changes[2] = visual.replace_once(config, '"view": "screened"', '"view": "redacted"')
    changes[4] = code[4].replace("DTD_FULL_BOTH_OUTPUT", "DTD_FULL_REDACTED_OUTPUT").replace(
        "ablation_05_full_both_vitl16", "ablation_08_full_redacted_vitl16").replace("DTD_FULL_BOTH_EPOCHS", "DTD_FULL_REDACTED_EPOCHS")
    changes[8] = code[8] + "\n\n" + REDACTION_HELPERS
    prefix = code[9].split("caption_df = load_ablation_captions", 1)[0]
    prefix = visual.replace_once(prefix,
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n',
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_07_full_unscreened_vitl16" / "shared" / "captions.csv",\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_06_full_shuffle_both_vitl16" / "shared" / "captions.csv",\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_05_full_both_vitl16" / "shared" / "captions.csv",\n')
    bank_functions = visual.definitions(code[9], ["make_ablation_banks", "batch_caption_inputs"])
    bank_functions = bank_functions.replace("full_both", "full_redacted").replace('["screened"]', '["redacted"]').replace(
        'settings["view"] != "screened"', 'settings["view"] != "redacted"')
    changes[9] = prefix + TEXT_STAGE + "\n\n" + bank_functions
    training = code[14].replace("full_text_manifest.json", "redacted_text_manifest.json").replace(
        "full_text_assignments.csv", "redacted_text_assignments.csv")
    changes[14] = visual.replace_once(training, '"view": spec["view"], "singletons_not_deranged": banks["singletons"],',
        '"view": spec["view"], "singletons_not_deranged": banks["singletons"],\n'
        '        "redaction_rule": "all_classes_and_fixed_aliases_word_boundary_deletion",\n'
        '        "availability_rule": "source_5_to8_words_and_nonempty_after_redaction",')
    protocol = code[17].replace("full_both", "full_redacted").replace("full-both comparison", "full-redacted comparison")
    protocol = visual.replace_once(protocol, '"text_inputs": "screened_full_redacted",',
        '"text_inputs": "aligned_class_alias_redacted_captions",')
    changes[17] = visual.replace_once(protocol, '"caption_input_policy": CAPTION_INPUT_POLICY,',
        '"caption_input_policy": CAPTION_INPUT_POLICY, "caption_view": "redacted",\n'
        '    "redaction_vocabulary": redaction_vocabulary, "uses_per_image_labels_for_redaction": False,\n'
        '    "availability_rule": "source_5_to8_words_and_nonempty_after_redaction",\n'
        '    "caption_assignment": "aligned_no_shuffle", "primary_content_control": "notebook_07_full_unscreened",')
    report = code[23].replace("full_both", "full_redacted")
    report = visual.replace_once(report,
        "Query image + query caption + one context image + its caption. Both text gates are enabled; selection combines image and caption similarities.",
        "Query/context captions have every known class/alias removed before CLIP encoding for both fusion and multimodal selection. No shuffle.")
    report = visual.replace_once(report,
        "Both gates are learned and independently masked for invalid captions. The caption policy is recorded in the protocol.",
        "Both gates use redacted availability: a 5-8-word source with nonempty surviving text, even if fewer than 5 words remain.")
    report = report.replace("## Screened caption coverage", "## Redacted caption availability")
    report += '\nif "caption_comparison" in globals():\n    with (OUT_PUT_DIR / "full_redacted_report.md").open("a", encoding="utf-8") as handle:\n        handle.write("\\n## Redaction and coverage\\n\\n" + markdown_table(caption_comparison) + "\\n\\nThe vocabulary is frozen across all images and does not use per-image labels. Compare to 07 with the same CSV/settings; emptied captions change availability. A score drop does not by itself prove label leakage, and finite aliases do not remove all implicit class information.\\n")\n'
    changes[23] = report
    changes[7] = "## Create/resume original captions.csv; redact class names and fixed aliases for both roles\n"
    changes[10] = "## Same fusion and multimodal selector as 05; redacted CLIP vectors only\n"
    changes[18] = "## Train full_redacted only: 50 epochs by default\n"
    changes[20] = "## Final redacted-control test after the declared epoch budget\n"
    changes[22] = "## Metrics, redaction/coverage comparison, semantic mix and learning curves\n"
    helpers = changes[8] + changes[9] + code[11] + code[12] + changes[14] + code[15]
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = (utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\n"
                  f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n")
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = cell["id"].replace("dtd-full-both-", "dtd-full-redacted-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    notebook["metadata"]["dtd_ablation"].update(experiment="full_redacted",
        default_caption_input_policy="dtd_lexical_v2", default_caption_view="redacted",
        redaction_uses_per_image_labels=False, redaction_vocabulary="DTD_class_names_and_fixed_DTD_EXTRA_TERMS")
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
