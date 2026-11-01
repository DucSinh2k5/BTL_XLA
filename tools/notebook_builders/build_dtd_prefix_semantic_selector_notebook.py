"""Build study04: aligned image-to-candidate-caption selection, no query caption."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_dtd_prefix_mltm_alignment_notebook as alignment


TARGET = PROJECT_ROOT / "notebooks/02_improvements/multimodal_architecture_ablation/04_candidate_only_semantic_selector/dtd/mtacs-dtd-vitl16-prefix-candidate-only-semantic-selector.ipynb"
HELPERS = PROJECT_ROOT / "src/training/mtacs_dtd_semantic_selector_cells.py"
STAGE = "04_prefix_candidate_only_semantic_selector"
PREFIX = "architecture_04_prefix_candidate_only_semantic_selector"


def build_notebook():
    notebook = alignment.build_notebook()
    source_of = alignment.norm.baseline.prefix.source_of
    replace_once = alignment.norm.baseline.prefix.replace_once
    code = {i: source_of(cell) for i, cell in enumerate(notebook["cells"])}
    source = HELPERS.read_text(encoding="utf-8")
    helpers = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef)}
    code[0] = '''# DTD Architecture Study 04: Aligned Query-Image / Candidate-Caption Selector

Question: can candidate captions improve CONTEXT SELECTION, rather than only fusion after image-only retrieval?
This is a controlled extension of study03, NOT a claim that study03 was the best classifier.
Saved study02/study03 split1/seed42 test F1: 0.849329 / 0.848722; original study01: 0.853032.
On validation, off/shuffled captions did not reduce F1 in either completed run. Study03 learned training-pool
image-caption correspondence, but that alone did not improve classification. These are historical observations,
not this notebook's prefilled metrics and not a basis for tuning test hyperparameters.

Keep study03's normalized candidate-text fusion, own-image/caption auxiliary alignment and positive diagonal metric.
The only new architectural branch scores QUERY IMAGE versus each VALID TRAINING CANDIDATE CAPTION,
using the SAME aligned caption projection and metric. Query captions remain zero/masked everywhere.
This is NOT the old query-caption/candidate-caption selector and does not generate query captions at inference.

scores = original_visual_z + alpha(epoch) * clipped_masked_semantic_z
semantic_raw = negative Mahalanobis distance(query_DINO_CLS, projected_candidate_CLIP_caption)
alpha(epoch) = 0.25 * sigmoid(existing_semantic_mix_logit) * ramp(epoch)

First 5 epochs: image-only retrieval while own-pair alignment starts learning.
Epochs 6..10: ramp the semantic coefficient to its learned bounded value; thereafter retain it.
The ramp epoch is stored in the checkpoint, including early best checkpoints and resume.
Invalid captions have ZERO semantic residual but remain eligible visual candidates. Self candidates are excluded;
semantic statistics exclude invalid/self columns. Fewer than two eligible captions or constant scores use visual-only fallback.
Semantic z is clipped to [-3, 3]; these conservative settings are declared starting values, not validated optima.
No new trainable parameters beyond study03: reuse its existing semantic_mix_logit. No batch-coupled inference OT.

Protocol unchanged: exact Florence-Prefix CSV, dtd_lexical_v2, DTD official split1, 1880/1880/1880,
376 training-only candidates, K=1, self-exclusion, DINOv3 ViT-L16 at 384, seed42, same batch/worker policy,
50 epochs and 50-epoch LR schedule, no early stopping, best minimum-validation-classification-loss checkpoint.
Train independently from pretrained weights, NOT from study02/03 best.pt. No captioner training/new CSV.
Alignment weight=0.05 and all study03 auxiliary settings remain unchanged. Captions remain synthetic weak inputs.

Inputs: DTD, complete local DINOv3 code plus ViT-L16 weights, complete CLIP ViT-B/32 and
captions_prefix_control.csv (hash verified). No project code, Florence/Qwen weights, old checkpoints,
external transport package, pip install or Internet are required. Adjust asset paths in configuration/settings.

Run All. Own output directory: outputs/mtacs_dtd_split1_architecture_04_prefix_candidate_only_semantic_selector_vitl16/.
After training, two SEPARATE validation diagnostics use this notebook's own best checkpoint:
1. Fusion interventions: aligned / text off / shuffled text, FIXING the aligned selected context image.
2. Selector interventions: aligned / visual selector only / shuffled SELECTOR captions; RESELECT context,
   then gather the selected image's ORIGINAL caption for fusion. This isolates selection from fusion corruption.
Each is a same-checkpoint inference intervention, not an independently retrained ablation.
Use validation for development; RUN_FINAL_TEST=False keeps the final test closed. Default True runs it once after completion.

Set SELECTOR_SEMANTIC_MAX_WEIGHT=0 in a NEW output directory for a retrained study03-equivalent control.
No guarantee of improved metrics. If selector interventions do not hurt when disabled/shuffled, report that result.
Tokenizer parallelism is disabled early; diagnostic loaders use zero workers. Training workers match prior defaults.
'''
    code[1] = code[1].replace("Prefix m-LTM-inspired alignment", "Prefix aligned semantic selector")
    code[2] = code[2].replace("DTD_PREFIX_MLTM_ALIGNMENT_SEED", "DTD_PREFIX_SEMANTIC_SELECTOR_SEED").replace(alignment.STAGE, STAGE)
    code[2] = code[2].replace('"semantic_selector": False}', '"semantic_selector": True}')
    code[2] = code[2].replace("'semantic_selector': False", "'semantic_selector': True")
    code[2] = code[2].replace("validate_alignment_output_directory", "validate_semantic_selector_output_directory").replace(
        "dtd_prefix_mltm_alignment_v1", "dtd_prefix_semantic_selector_v1").replace(
        "study03 must not overwrite study01/02 or other runs", "study04 must not overwrite study01/02/03 or other runs").replace(
        "no study03 protocol", "no study04 protocol")
    code[2] += '''

SELECTOR_SEMANTIC_MAX_WEIGHT = 0.25  # New-directory zero control: set 0.0 before Run All.
SELECTOR_VISUAL_WARMUP_EPOCHS = 5
SELECTOR_SEMANTIC_RAMP_EPOCHS = 5
SELECTOR_SEMANTIC_Z_CLIP = 3.0
RUN_SELECTOR_DIAGNOSTICS = True
'''
    code[2] += "\n\n" + helpers["semantic_selector_config"] + "\n\n" + helpers["validate_semantic_selector_settings"]
    code[4] = code[4].replace("DTD_PREFIX_MLTM_ALIGNMENT_", "DTD_PREFIX_SEMANTIC_SELECTOR_").replace(
        alignment.PREFIX, PREFIX).replace("validate_alignment_output_directory", "validate_semantic_selector_output_directory")
    # Accept the attachment paths used by the executed study03 without altering older templates.
    code[4] = code[4].replace("'/kaggle/input/datasets/nguyenducxu/dinov3-code/dinov3-main'",
                            "'/kaggle/input/datasets/ducvutrung123/dinov3-code-hubconf/dinov3-main'")
    code[4] = code[4].replace("'/kaggle/input/datasets/nguyenducxu/dinov3-vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'",
                            "'/kaggle/input/datasets/sinhnguync/dinov3-vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'")
    code[9] = replace_once(code[9],
        'or settings["query_fusion"] or not settings["candidate_fusion"] or settings["semantic_selector"]',
        'or settings["query_fusion"] or not settings["candidate_fusion"]')
    code[9] = code[9].replace("candidate text on, query text/semantic selector off.",
                            "candidate text on, query text off; aligned candidate captions only.")
    code[12] = replace_once(code[12],
        'if settings["name"] != "candidate_only" or settings["query_fusion"] or settings["semantic_selector"]:',
        'if settings["name"] != "candidate_only" or settings["query_fusion"]:')
    code[12] = code[12].replace("Alignment must not enable query text or the semantic selector",
                              "Own-pair alignment must not enable query text")
    code[12] += "\n\n" + helpers["make_semantic_selector_model_class"]
    code[14] = replace_once(code[14], "    network.train()\n    for parameter in network.backbone.parameters():",
                           "    network.semantic_selector_epoch.fill_(int(epoch))\n    network.train()\n    for parameter in network.backbone.parameters():")
    code[14] = replace_once(code[14], '"alignment_unique_captions", "alignment_informative")',
                           '"alignment_unique_captions", "alignment_informative", "selector_semantic_weight")')
    code[14] = replace_once(code[14], '("alignment_informative", alignment["informative"])):',
                           '("alignment_informative", alignment["informative"]),\n'
                           '                           ("selector_semantic_weight", details["semantic_weight"])):')
    code[14] = replace_once(code[14], '''f'align_weight={training["alignment_weight"]:.3f} ot_error={training["alignment_marginal_error_max"]:.2e}')''',
        '''f'align_weight={training["alignment_weight"]:.3f} ot_error={training["alignment_marginal_error_max"]:.2e} '
                  f'selector_alpha={training["selector_semantic_weight"]:.4f}')''')
    code[15] = replace_once(code[15],
        'if banks["settings"]["name"] != "candidate_only" or banks["settings"]["semantic_selector"]:',
        'if banks["settings"]["name"] != "candidate_only":')
    code[15] = code[15].replace("Norm diagnostics require candidate_only with an image-only selector",
                              "Fusion diagnostics require the candidate_only configuration")
    code[15] = code[15].replace("dtd_mltm_alignment_validation_v1", "dtd_semantic_selector_fusion_validation_v1")
    code[15] += "\n\n" + helpers["collect_selector_validation_interventions"] + "\n\n" + helpers["run_selector_validation_interventions"]
    code[17] = replace_once(code[17], "validate_alignment_settings()", "validate_semantic_selector_settings()")
    code[17] = replace_once(code[17], '"schema": "dtd_prefix_mltm_alignment_v1", "alignment": alignment_protocol_config(),',
        '"schema": "dtd_prefix_semantic_selector_v1", "alignment": alignment_protocol_config(),\n'
        '    "semantic_selector": semantic_selector_config(),')
    code[17] = replace_once(code[17], '"selector_sampling": "categorical_policy_on_visual_z"',
                           '"selector_sampling": "categorical_policy_on_visual_z_plus_aligned_image_caption_residual"')
    code[17] = code[17].replace("make_mltm_alignment_model_class(MTACSModel", "make_semantic_selector_model_class(MTACSModel")
    code[17] = code[17].replace("Architecture study 03: Prefix / candidate_only / norm fusion + m-LTM-inspired alignment:",
                              "Architecture study 04: Prefix / candidate_only / aligned image-caption selector:")
    code[17] = replace_once(code[17], 'display(pd.DataFrame(SELECTED_ABLATIONS))',
                           'display(pd.DataFrame(SELECTED_ABLATIONS))\nprint("Semantic selector:", semantic_selector_config())')
    code[18] = "## Train study04 independently for 50 epochs: same fusion/alignment, new selector branch"
    code[20] = '''## Validation-only fusion and selector interventions

The first table keeps selected context IMAGES fixed and changes only their fusion text.
The second table changes only selector text/use, RESELECTS context images, then uses their ORIGINAL captions for fusion.
Neither table trains weights or reads test queries. Do not interpret them as retrained visual-only/zero-weight runs.
'''
    code[21] += '''

selector_tables = []
if RUN_SELECTOR_DIAGNOSTICS:
    for spec in SELECTED_ABLATIONS:
        for training_seed in TRAIN_SEEDS:
            selector_tables.append(run_selector_validation_interventions(spec, training_seed))
if selector_tables:
    selector_interventions = pd.concat(selector_tables, ignore_index=True)
    selector_interventions.to_csv(OUT_PUT_DIR / "validation_selector_interventions.csv", index=False)
    display(selector_interventions.round(6))
else:
    print("Selector interventions disabled; no diagnostic metrics are fabricated.")
'''
    code[24] = "## Study04 classification, alignment fit and separate selector/fusion diagnostics"
    code[25] = code[25].replace(alignment.PREFIX, PREFIX).replace("architecture study 03", "architecture study 04")
    code[25] = code[25].replace("semantic_selector_enabled=False", "semantic_selector_enabled=True")
    code[25] = code[25].replace("candidate_only + norm fusion + auxiliary alignment", "candidate_only + alignment + image-caption selector")
    code[25] = code[25].replace("Query text and semantic selector are disabled.", "Query text is disabled; the aligned image-caption selector is enabled.")
    code[25] = code[25].replace("no Partial OT, semantic selector, token fusion or ASWP.", "bounded aligned semantic selector; no Partial OT, token fusion or ASWP.")
    code[25] = code[25].replace("Compare study03 against study02 under the same protocol.",
                              "Compare study04 to study03; the selector is the new branch, not a proven improvement.")
    code[25] += f'''
selector_path = OUT_PUT_DIR / "validation_selector_interventions.csv"
with (OUT_PUT_DIR / "{PREFIX}_report.md").open("a", encoding="utf-8") as handle:
    handle.write("\\n## Image-to-candidate-caption selector configuration\\n\\n" + json.dumps(semantic_selector_config(), indent=2) + "\\n")
    if selector_path.is_file():
        selector_summary = pd.read_csv(selector_path)
        display(selector_summary.round(6))
        handle.write("\\n## Validation selector interventions, context reselected\\n\\n" + markdown_table(selector_summary) + "\\n")
    handle.write("\\nThe fusion table fixes context images. The selector table reselects contexts but keeps each selected image's original fusion caption. "
                 "Zero selector weight needs its own fresh directory for a retrained control. No query caption is used.\\n")
'''
    code[6] = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    code[6] += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(alignment.TARGET.read_bytes()).hexdigest()!r}\n"
    helper_text = "".join(code[i].strip() + "\n" for i in (8, 9, 11, 12, 14, 15))
    code[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helper_text.encode()).hexdigest()!r}\n"
    for i, cell in enumerate(notebook["cells"]):
        cell["source"] = (code[i].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-prefix-semantic-selector-{i:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(architecture_study=STAGE,
        semantic_selector=True, query_fusion=False, selector_query_modality="image_only_no_query_caption",
        selector_method="aligned_query_image_candidate_caption_residual_v1", selector_default_max_weight=.25)
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
