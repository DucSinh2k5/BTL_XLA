"""Build study 02: fixed Prefix CSV, candidate-only, normalized text fusion."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_dtd_prefix_candidate_baseline_notebook as baseline


TARGET = PROJECT_ROOT / "notebooks/02_improvements/multimodal_architecture_ablation/02_candidate_only_norm_fusion/dtd/mtacs-dtd-vitl16-prefix-candidate-only-norm-fusion.ipynb"
HELPERS = PROJECT_ROOT / "src/training/mtacs_dtd_norm_fusion_cells.py"
STAGE = "02_prefix_candidate_only_norm_fusion"
PREFIX = "architecture_02_prefix_candidate_only_norm_fusion"


def helper_sources():
    source = HELPERS.read_text(encoding="utf-8")
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def code_cell(source, kind="code"):
    cell = {"cell_type": kind, "metadata": {}, "source": (source.strip() + "\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def build_notebook():
    notebook = baseline.build_notebook()
    source_of, replace_once = baseline.prefix.source_of, baseline.prefix.replace_once
    code = {i: source_of(cell) for i, cell in enumerate(notebook["cells"])}
    helpers = helper_sources()
    code[0] = '''# DTD Architecture Study 02: Candidate-Only / Norm-Calibrated Fusion

Controlled change relative to study 01: L2-normalize the projected candidate caption vector
before both the learned gate and the additive fusion. The contextual visual representation
already uses L2 normalization. No extra trainable parameters, loss, query text or semantic selector are added.
The no-context reward branch is retained unchanged; it has no active candidate caption.

Keep the exact Florence-Prefix CSV and dtd_lexical_v2: no caption generation or policy changes.
Query text remains zero/masked. The selector uses image features only; gather caption/image at the same index.
Invalid captions stay masked, but their images remain candidates. Caption validity is heuristic, not ground truth.
Normalization controls scale; it does not prove semantic alignment or guarantee a higher score.

Protocol: DTD official split1, 1880 train/1880 validation/1880 test, 376 training candidates,
K=1 and exact-self exclusion, DINOv3 ViT-L/16, 384x384, pool/training seed42,
50 epochs with the original 50-epoch LR schedule, no early stopping, minimum validation-loss checkpoint.
Optimizer, losses, gate initialization, augmentations, batch policy and backbone/selector are unchanged.
Do not initialize training with study 01's best.pt; this experiment trains from scratch or resumes its own run.

Attach DTD, DINOv3 local code and ViT-L/16 weights, full CLIP ViT-B/32, and captions_prefix_control.csv.
The notebook is standalone and offline; no Florence/Qwen checkpoint, project Python files or pip install is needed.
Set CAPTIONS_INPUT_PATH if hash-checked discovery does not find your attached Prefix CSV.
The source CSV is copied byte-for-byte; only training-pool captions are encoded.

Output: outputs/mtacs_dtd_split1_architecture_02_prefix_candidate_only_norm_fusion_vitl16/.
Do not point this path at study 01. A distinct protocol prevents accidental checkpoint/cache reuse.
The historical study01 test F1 0.853032 is a reference only, never this model's prefilled metric.

Run All in order. After training, validation diagnostics compare aligned / text off / shuffled text
using this model's own best checkpoint and the SAME selected context images. Shuffle only eligible
training-pool captions, preserve masks, and report duplicate-text/source changes separately.
Export per-image predictions, gate/norm statistics and paired validation deltas before the normal locked final test.
These diagnostics measure test-time dependence, not the effect of training a separate visual-only model.
They do not diagnose study01 weights; study01 checkpoint interventions require its own compatible workflow.

TOKENIZERS_PARALLELISM is disabled before imports/workers. Diagnostic loading always uses zero workers.
Training workers keep study01 defaults for comparability; optionally set DTD_PREFIX_NORM_FUSION_WORKERS=0
before running settings if notebook multiprocessing errors recur. Record this change; worker-count changes can
alter augmentation RNG and are not a bitwise-matched comparison to the original 8-worker run.
Use validation for development; set RUN_FINAL_TEST=False while selecting changes. No test-time tuning.
'''
    code[1] = code[1].replace("Prefix candidate-only baseline", "Prefix norm-calibrated fusion")
    code[2] = code[2].replace("DTD_PREFIX_CANDIDATE_BASELINE_SEED", "DTD_PREFIX_NORM_FUSION_SEED").replace(
        baseline.STAGE, STAGE)
    code[2] = replace_once(code[2], "'alignment': False, 'partial_ot': False, 'semantic_selector': False,",
        "'norm_calibrated_fusion': True, 'alignment': False, 'partial_ot': False, 'semantic_selector': False,")
    code[2] += '''
FUSION_NORM_EPS = 1e-6
RUN_VALIDATION_DIAGNOSTICS = True
DIAGNOSTIC_SHUFFLE_SEED = 2026  # Fixed label-free training-pool shuffle; no test selection.
'''
    code[2] += "\n\n" + helpers["validate_norm_output_directory"]
    code[4] = code[4].replace("DTD_PREFIX_CANDIDATE_BASELINE_OUTPUT", "DTD_PREFIX_NORM_FUSION_OUTPUT").replace(
        "architecture_01_prefix_candidate_only_vitl16", PREFIX + "_vitl16").replace(
        "DTD_PREFIX_CANDIDATE_BASELINE_EPOCHS", "DTD_PREFIX_NORM_FUSION_EPOCHS")
    code[4] = replace_once(code[4], "NUM_WORKERS = 0 if os.name == 'nt' else min(8, os.cpu_count() or 0)",
        "NUM_WORKERS = int(os.getenv('DTD_PREFIX_NORM_FUSION_WORKERS', str(0 if os.name == 'nt' else min(8, os.cpu_count() or 0))))\n"
        "if NUM_WORKERS < 0:\n    raise ValueError('Training workers must be nonnegative')")
    code[4] = replace_once(code[4], "OUTPUT_DIR = OUT_PUT_DIR",
                          "validate_norm_output_directory(OUT_PUT_DIR)\nOUTPUT_DIR = OUT_PUT_DIR")
    code[12] += "\n\n" + helpers["make_norm_calibrated_model_class"]
    names = ("make_validation_caption_shuffle", "collect_norm_validation_interventions",
             "run_norm_validation_interventions", "make_norm_validation_loader")
    code[15] += "\n\n" + "\n\n\n".join(helpers[name] for name in names)
    code[17] = replace_once(code[17], '"schema": "dtd_prefix_candidate_baseline_v1",',
        '"schema": "dtd_prefix_norm_fusion_v1",\n'
        '    "fusion": {"candidate_text": "l2_after_projection_before_gate_and_fusion", "epsilon": FUSION_NORM_EPS,\n'
        '               "visual": "existing_context_visual_l2", "no_context_reward_branch": "unchanged"},\n'
        '    "validation_diagnostics": {"shuffle_seed": DIAGNOSTIC_SHUFFLE_SEED, "workers": 0},')
    code[17] = replace_once(code[17], "AblationModel = make_ablation_model_class(MTACSModel, DinoV3TACSClassifier)",
        "AblationModel = make_norm_calibrated_model_class(MTACSModel, DinoV3TACSClassifier)")
    code[17] = code[17].replace("Architecture baseline 01: Prefix / candidate_only:",
                                "Architecture study 02: Prefix / candidate_only / norm fusion:")
    code[18] = "## Train candidate-only with norm-calibrated fusion: 50 epochs by default"
    code[20] = "## Locked final test after training and validation diagnostics"
    code[22] = "## Study 02 metrics, provenance, learning curves and validation diagnostics"
    report = code[23].replace("architecture_01_prefix_candidate_only", PREFIX)
    report = report.replace("architecture baseline 01 / Florence-Prefix / candidate_only",
                            "architecture study 02 / Florence-Prefix / candidate_only + norm fusion")
    report = replace_once(report, 'candidate_text_enabled=True, semantic_selector_enabled=False, alignment_enabled=False,',
        'candidate_text_enabled=True, norm_calibrated_fusion=True, semantic_selector_enabled=False, alignment_enabled=False,')
    report = report.replace("Unchanged candidate_only architecture; the exact Prefix CSV is frozen and CLIP encodes training-pool captions only.",
                            "Candidate-only with L2-normalized projected candidate text; the exact Prefix CSV is frozen.")
    report = report.replace("Baseline for architecture experiments; no alignment, Partial OT, semantic selector, token fusion or ASWP is enabled.",
                            "Scale-control experiment, not semantic alignment; no Partial OT, semantic selector, token fusion or ASWP is enabled.")
    report = report.replace("Historical F1 values are not metrics of this baseline.",
                            "Historical F1 values, including study01 F1 0.853032, are not metrics of this model.")
    report += f'''
diagnostic_path = OUT_PUT_DIR / "validation_interventions.csv"
if diagnostic_path.is_file():
    validation_diagnostics = pd.read_csv(diagnostic_path)
    display(validation_diagnostics.round(6))
    with (OUT_PUT_DIR / "{PREFIX}_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\\n## Fixed-context validation text interventions\\n\\n" + markdown_table(validation_diagnostics) + "\\n")
        handle.write("\\nNorm/gate distributions and per-image interventions are in each run's validation_norm_diagnostics/ directory. "
                     "A small gate is not a percentage contribution; inspect weighted_text_visual_ratio. "
                     "Shuffled duplicate captions can remain text-identical; consult the shuffle assignment export.\\n")
'''
    code[23] = report
    code[6] = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    code[6] += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(baseline.TARGET.read_bytes()).hexdigest()!r}\n"
    # Hash final cell text, including newly embedded diagnostics/model definitions.
    helper_text = "".join(code[i].strip() + "\n" for i in (8, 9, 11, 12, 14, 15))
    code[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helper_text.encode()).hexdigest()!r}\n"
    for i, cell in enumerate(notebook["cells"]):
        cell["source"] = (code[i].strip() + "\n").splitlines(keepends=True)
    notebook["cells"][20:20] = [code_cell('''## Validation-only caption interventions and fusion norms

No training or checkpoint selection occurs here. This uses the completed study02 best checkpoint only,
keeps image contexts fixed across aligned/off/shuffled within each query, and never reads test queries.
The paired confidence interval describes query resampling, not variation across training seeds.
''', "markdown"), code_cell('''intervention_tables = []
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
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"dtd-prefix-norm-fusion-{i:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(
        architecture_study=STAGE, norm_calibrated_fusion=True,
        candidate_text_normalization="l2_after_projection_before_gate_and_fusion",
        validation_text_interventions=True)
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
