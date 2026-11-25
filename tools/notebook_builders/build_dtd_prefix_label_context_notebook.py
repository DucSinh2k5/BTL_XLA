"""Build study 06: study01 fusion, candidate text = training-pool label prompt + its screened caption."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_dtd_prefix_candidate_baseline_notebook as baseline
import build_dtd_prefix_norm_fusion_notebook as norm
import build_dtd_prefix_text_token_notebook as token


TARGET = PROJECT_ROOT / "notebooks/02_improvements/multimodal_architecture_ablation/06_candidate_only_label_context/dtd/mtacs-dtd-vitl16-prefix-candidate-only-label-context.ipynb"
HELPERS = PROJECT_ROOT / "src/training/mtacs_dtd_label_context_cells.py"
STAGE = "06_prefix_candidate_only_label_context"
PREFIX = "architecture_06_prefix_candidate_only_label_context"


def helper_sources():
    source = HELPERS.read_text(encoding="utf-8")
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def build_notebook():
    notebook = baseline.build_notebook()
    source_of, replace_once = baseline.prefix.source_of, baseline.prefix.replace_once
    code = {i: source_of(cell) for i, cell in enumerate(notebook["cells"])}
    helpers, norm_helpers = helper_sources(), norm.helper_sources()
    code[0] = '''# DTD Architecture Study 06: Candidate-Only / Label-Aware Candidate Text

Controlled change relative to study 01 (best so far, test F1 85.30): keep the exact study01 model,
selector and gated candidate-text fusion, but change WHAT the candidate text says.
Study01 candidate text = Florence caption of the selected candidate image (derived from its pixels, so mostly
redundant with the image). Study06 candidate text = "a photo of a <candidate class> texture, <its screened caption>".

Why this is allowed: the candidate pool contains TRAINING images only, so each candidate's class label is known
at inference time, exactly like a retrieval / kNN memory. The QUERY's label is never used as an input; query text
stays off; exact-self exclusion still forbids a train query from choosing itself. This is a label-aware
retrieval-augmented variant of M-TACS, not the original TACS protocol: report it as such.

Expected behaviour: the reward favours candidates whose (image + label text) lowers the loss, so the selector may
learn to retrieve same-class neighbours and the gate may learn to trust their label. Watch the gate, the
validation interventions and the same-class rate in the per-image predictions.

Protocol (unchanged): DTD official split1, 1880/1880/1880, 376 training candidates, K=1, ViT-L/16, 384x384,
seed42, 50 epochs, minimum validation-loss checkpoint, locked final test. Frozen Prefix CSV (embedded below).
Attach DTD, DINOv3 code + ViT-L/16 weights and CLIP ViT-B/32. Run All in order.
References only, never this model's metrics: study01 test F1 0.853032, visual_only (A01) 0.8482.
'''
    code[1] = code[1].replace("Prefix candidate-only baseline", "Prefix label-aware candidate text")
    code[2] = code[2].replace("DTD_PREFIX_CANDIDATE_BASELINE_SEED", "DTD_PREFIX_LABEL_CONTEXT_SEED").replace(
        baseline.STAGE, STAGE)
    code[2] = replace_once(code[2], "'token_fusion': False,", "'token_fusion': False, 'label_context': True,")
    code[2] += '''
FUSION_NORM_EPS = 1e-6
LABEL_CONTEXT_TEMPLATE = "a photo of a {label} texture"  # Candidate (training-pool) label only; never the query's.
RUN_VALIDATION_DIAGNOSTICS = True
DIAGNOSTIC_SHUFFLE_SEED = 2026  # Shuffles label texts among candidates; images stay fixed.
'''
    code[2] += "\n\n" + helpers["validate_label_context_output_directory"]
    code[4] = code[4].replace("DTD_PREFIX_CANDIDATE_BASELINE_OUTPUT", "DTD_PREFIX_LABEL_CONTEXT_OUTPUT").replace(
        "architecture_01_prefix_candidate_only_vitl16", PREFIX + "_vitl16").replace(
        "DTD_PREFIX_CANDIDATE_BASELINE_EPOCHS", "DTD_PREFIX_LABEL_CONTEXT_EPOCHS")
    code[4] = replace_once(code[4], "NUM_WORKERS = 0 if os.name == 'nt' else min(8, os.cpu_count() or 0)",
        "NUM_WORKERS = int(os.getenv('DTD_PREFIX_LABEL_CONTEXT_WORKERS', str(0 if os.name == 'nt' else min(8, os.cpu_count() or 0))))\n"
        "if NUM_WORKERS < 0:\n    raise ValueError('Training workers must be nonnegative')")
    code[4] = replace_once(code[4], "OUTPUT_DIR = OUT_PUT_DIR",
                          "validate_label_context_output_directory(OUT_PUT_DIR)\nOUTPUT_DIR = OUT_PUT_DIR")
    code[9] += "\n\n" + helpers["candidate_context_texts"] + "\n\n\n" + helpers["make_label_context_banks"] + '''


CANDIDATE_CLASS_IDS = candidate_df.class_id.astype(int).tolist()
label_context_texts = candidate_context_texts(CANDIDATE_IDS, CANDIDATE_CLASS_IDS, class_names,
                                              TEXT_CACHE["views"]["screened"], LABEL_CONTEXT_TEMPLATE)
LABEL_TEXT_CACHE = encode_view_cache(
    {"label_context": {"image_ids": list(CANDIDATE_IDS), "texts": label_context_texts,
                       "valid": np.ones(len(CANDIDATE_IDS), dtype=bool)}},
    clip_asset, SHARED_DIR / "candidate_label_context_embeddings.pt")
make_ablation_banks = make_label_context_banks(make_ablation_banks, LABEL_TEXT_CACHE["views"]["label_context"],
                                               CANDIDATE_IDS, DEVICE)
pd.DataFrame({"candidate_image_id": CANDIDATE_IDS, "candidate_class_id": CANDIDATE_CLASS_IDS,
              "label_context_text": label_context_texts}).to_csv(SHARED_DIR / "candidate_label_context_texts.csv", index=False)
print("Label-aware candidate texts:", len(label_context_texts), "| example:", label_context_texts[0])
'''
    code[12] += "\n\n" + helpers["make_label_context_model_class"]
    names = ("make_validation_caption_shuffle", "collect_norm_validation_interventions",
             "run_norm_validation_interventions", "make_norm_validation_loader")
    code[15] += "\n\n" + "\n\n\n".join(norm_helpers[name] for name in names)
    code[17] = replace_once(code[17], '"schema": "dtd_prefix_candidate_baseline_v1",',
        '"schema": "dtd_prefix_label_context_v1",\n'
        '    "candidate_text": {"source": "training_pool_label_prompt_plus_screened_caption",\n'
        '                       "template": LABEL_CONTEXT_TEMPLATE, "query_label_used": False,\n'
        '                       "fusion": "study01_gated_residual_unchanged"},\n'
        '    "validation_diagnostics": {"shuffle_seed": DIAGNOSTIC_SHUFFLE_SEED, "workers": 0},')
    code[17] = replace_once(code[17], "AblationModel = make_ablation_model_class(MTACSModel, DinoV3TACSClassifier)",
        "AblationModel = make_label_context_model_class(MTACSModel, DinoV3TACSClassifier)")
    code[17] = code[17].replace("Architecture baseline 01: Prefix / candidate_only:",
                                "Architecture study 06: Prefix / candidate_only / label-aware candidate text:")
    code[18] = "## Train candidate-only with label-aware candidate text: 50 epochs by default"
    code[20] = "## Locked final test after training and validation diagnostics"
    code[22] = "## Study 06 metrics, provenance, learning curves and validation diagnostics"
    report = code[23].replace("architecture_01_prefix_candidate_only", PREFIX)
    report = report.replace("architecture baseline 01 / Florence-Prefix / candidate_only",
                            "architecture study 06 / Florence-Prefix / candidate_only + label-aware candidate text")
    report = replace_once(report, "token_fusion_enabled=False,", "token_fusion_enabled=False, label_context_enabled=True,")
    report = report.replace("Unchanged candidate_only architecture; the exact Prefix CSV is frozen and CLIP encodes training-pool captions only.",
                            "Study01 architecture; candidate text = training-pool label prompt + its screened caption. Query labels are never inputs.")
    report = report.replace("Baseline for architecture experiments; no alignment, Partial OT, semantic selector, token fusion or ASWP is enabled.",
                            "Label-aware retrieval-augmented variant (not the original TACS protocol); no alignment, semantic selector, token fusion or ASWP.")
    report = report.replace("Historical F1 values are not metrics of this baseline.",
                            "Historical F1 values, including study01 F1 0.853032, are not metrics of this model.")
    report += f'''
diagnostic_path = OUT_PUT_DIR / "validation_interventions.csv"
if diagnostic_path.is_file():
    validation_diagnostics = pd.read_csv(diagnostic_path)
    display(validation_diagnostics.round(6))
    with (OUT_PUT_DIR / "{PREFIX}_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\\n## Fixed-context validation text interventions\\n\\n" + markdown_table(validation_diagnostics) + "\\n")
        handle.write("\\ntext_off masks the label-aware candidate text; text_shuffled permutes it among candidates "
                     "(a wrong label for the same image). A large drop for both means the model relies on the candidate label.\\n")
'''
    code[23] = report
    code[6] = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    code[6] += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(baseline.TARGET.read_bytes()).hexdigest()!r}\n"
    helper_text = "".join(code[i].strip() + "\n" for i in (8, 9, 11, 12, 14, 15))
    code[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helper_text.encode()).hexdigest()!r}\n"
    for i, cell in enumerate(notebook["cells"]):
        cell["source"] = (code[i].strip() + "\n").splitlines(keepends=True)
    notebook["cells"][20:20] = [norm.code_cell('''## Validation-only candidate-text interventions

No training or checkpoint selection occurs here. Uses the completed study06 best checkpoint only,
keeps image contexts fixed across aligned/off/shuffled within each query, and never reads test queries.
''', "markdown"), norm.code_cell('''intervention_tables = []
if RUN_VALIDATION_DIAGNOSTICS:
    for spec in SELECTED_ABLATIONS:
        for training_seed in TRAIN_SEEDS:
            intervention_tables.append(run_norm_validation_interventions(spec, training_seed))
if intervention_tables:
    interventions = pd.concat(intervention_tables, ignore_index=True)
    interventions.to_csv(OUT_PUT_DIR / "validation_interventions.csv", index=False)
    display(interventions.round(6))
else:
    print("Validation interventions disabled; no diagnostic metrics are fabricated.")
''')]
    notebook["cells"].insert(9, token.embedded_caption_cell())  # right before the CSV binding cell
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"dtd-prefix-label-context-{i:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(
        architecture_study=STAGE, label_context=True,
        candidate_text="training_pool_label_prompt_plus_screened_caption", validation_text_interventions=True)
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
