"""Build study03: normalized candidate-caption fusion plus auxiliary m-LTM-inspired alignment."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_dtd_prefix_norm_fusion_notebook as norm


TARGET = PROJECT_ROOT / "notebooks/02_improvements/multimodal_architecture_ablation/03_candidate_only_mltm_alignment/dtd/mtacs-dtd-vitl16-prefix-candidate-only-mltm-alignment.ipynb"
HELPERS = PROJECT_ROOT / "src/training/mtacs_dtd_alignment_cells.py"
STAGE = "03_prefix_candidate_only_mltm_alignment"
PREFIX = "architecture_03_prefix_candidate_only_mltm_alignment"


def replace_definition(source, name, replacement):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    old = ast.get_source_segment(source, node)
    return norm.baseline.prefix.replace_once(source, old, replacement)


def build_notebook():
    notebook = norm.build_notebook()
    source_of, replace_once = norm.baseline.prefix.source_of, norm.baseline.prefix.replace_once
    code = {i: source_of(cell) for i, cell in enumerate(notebook["cells"])}
    helper_source = HELPERS.read_text(encoding="utf-8")
    helpers = {node.name: ast.get_source_segment(helper_source, node) for node in ast.parse(helper_source).body
               if isinstance(node, ast.FunctionDef)}
    code[0] = '''# DTD Architecture Study 03: Candidate-Only / Norm Fusion + m-LTM-Inspired Alignment

Controlled comparison: study02 versus study03 adds one auxiliary alignment module/objective.
Fusion, visual selector, query-text mask, gate initialization and all original classification/policy losses remain unchanged.
Keep the fixed Florence-Prefix CSV, dtd_lexical_v2, official DTD split1, 376 training candidates,
K=1/self-exclusion, DINOv3 ViT-L/16 at 384, batch policy and seed42, 50 epochs, no early stopping,
and the minimum validation-classification-loss checkpoint criterion. Do not compare only to study01:
study03 retains study02's L2 normalization before the candidate gate and fusion.

The auxiliary objective samples eligible TRAINING-POOL images and EACH IMAGE'S OWN caption uniformly,
not the retrieved context caption as a pseudo-positive for its query. Image anchors use the detached
epoch-start backbone feature bank. Alignment updates the SAME candidate text projection that feeds fusion,
plus a positive diagonal Mahalanobis metric. No extra image forward passes, caption labels or ground-truth class names.
CLIP is frozen; query text is never used, including for this loss. The image backbone still learns from the original task loss.

The method is an adaptation inspired by [m-LTM, Luong et al.](https://arxiv.org/abs/2405.10084),
not a faithful reproduction of its audio-text system. A differentiable log-domain balanced Sinkhorn plan
is matched to paired targets by KL. Project differences: detached image teacher, trace-normalized positive DIAGONAL metric
(1024 extra parameters), and a block target for repeated canonical captions instead of false one-to-one negatives.
No Partial OT, text-aware selector, query caption, token fusion or ASWP is enabled.

Default alignment weight=0.05, linear warmup=5 epochs, batch=32 own-image/caption pairs,
entropy epsilon=0.1, 100 Sinkhorn iterations, max RELATIVE marginal error=0.005.
These are declared starting settings, not tuned on test. Use validation if changing them and a fresh output directory.
Weight=0 is an explicit zero-loss control; it does not consume extra RNG or add inference effects.
Metric initialization is identity and does not perturb common model parameter initialization.
Text projection dropout is skipped only for deterministic alignment; its learned Linear/LayerNorm weights are shared with fusion.

Inputs are exactly study02: DTD, local DINOv3 code/ViT-L16 weights, full CLIP ViT-B/32,
and the hash-checked captions_prefix_control.csv. No study01/02 best.pt, Florence/Qwen weights,
caption references, POT package, project Python files, downloads or pip install are needed.
The small differentiable transport solver is embedded and numerically tested; captions are not regenerated.

Train from scratch, or resume study03's OWN unfinished run. Output is independent:
outputs/mtacs_dtd_split1_architecture_03_prefix_candidate_only_mltm_alignment_vitl16/.
Protocol/output guards reject study01/02 directories and incompatible alignment settings.
Sampler indices derive from training seed, epoch and step, so epoch-boundary resume needs no hidden sampler state.

Run All in order. After 50 epochs, validation diagnostics retain aligned/off/shuffled text with fixed images,
and an additional training-pool fit table measures alignment mass, pair cosine and metric condition.
That table is NOT held-out retrieval quality or proof that captions are ground truth.
The final test remains locked to the classification-selected best checkpoint; do not select it by alignment loss or test F1.
Set RUN_FINAL_TEST=False during validation-only development. No study02 or study03 GPU scores are prefilled.

Training worker defaults match prior runs; diagnostic loading uses zero workers. If notebook multiprocessing
errors recur, set DTD_PREFIX_MLTM_ALIGNMENT_WORKERS=0 before settings, record the deviation and match it in controls.
Normalization plus matching encourages a useful representation; it does not guarantee higher classification metrics.
'''
    code[1] = code[1].replace("Prefix norm-calibrated fusion", "Prefix m-LTM-inspired alignment")
    code[2] = code[2].replace("DTD_PREFIX_NORM_FUSION_SEED", "DTD_PREFIX_MLTM_ALIGNMENT_SEED").replace(norm.STAGE, STAGE)
    code[2] = code[2].replace("validate_norm_output_directory", "validate_alignment_output_directory").replace(
        "dtd_prefix_norm_fusion_v1", "dtd_prefix_mltm_alignment_v1").replace(
        "study02 must not overwrite study01 or other runs", "study03 must not overwrite study01/02 or other runs").replace(
        "no study02 protocol", "no study03 protocol")
    code[2] += '''

ALIGNMENT_WEIGHT = 0.05
ALIGNMENT_WARMUP_EPOCHS = 5
ALIGNMENT_BATCH_SIZE = 32
ALIGNMENT_OT_EPSILON = 0.1
ALIGNMENT_SINKHORN_ITERS = 100
ALIGNMENT_MAX_MARGINAL_ERROR = 0.005  # Relative to the uniform 1/b marginal.
ALIGNMENT_SAMPLE_SEED = 2026
ALIGNMENT_METRIC_FLOOR = 1e-4
ALIGNMENT_DISTANCE_EPS = 1e-8
ARCHITECTURE_STUDY["alignment"] = ALIGNMENT_WEIGHT > 0
ARCHITECTURE_STUDY["alignment_method"] = "mltm_inspired_image_caption_v1"
'''
    code[2] += "\n\n" + helpers["alignment_protocol_config"] + "\n\n" + helpers["validate_alignment_settings"]
    code[4] = code[4].replace("DTD_PREFIX_NORM_FUSION_", "DTD_PREFIX_MLTM_ALIGNMENT_").replace(norm.PREFIX, PREFIX).replace(
        "validate_norm_output_directory", "validate_alignment_output_directory")
    model_helpers = [name for name in helpers if name not in {"alignment_protocol_config", "validate_alignment_settings"}]
    code[12] += "\n\n" + "\n\n\n".join(helpers[name] for name in model_helpers)
    code[14] = replace_definition(code[14], "create_ablation_model", '''def create_ablation_model(seed):
    seed_ablation(seed)
    network = AblationModel(DINOV3_REPO_DIR, DINOV3_MODEL_NAME, resolve_dinov3_weights(), NUM_CLASSES,
                            FEATURE_DIM, allow_feature_fallback=False, clip_dim=CLIP_EMBED_DIM).to(DEVICE)
    network.alignment_training_seed = int(seed)
    return network''')
    code[14] = replace_once(code[14], '("loss", "reward", "policy_loss", "query_gate", "candidate_gate")',
        '("loss", "reward", "policy_loss", "query_gate", "candidate_gate", "alignment_loss", "alignment_weight",\n'
        '                              "alignment_positive_mass", "alignment_pair_cosine", "alignment_pairs",\n'
        '                              "alignment_unique_captions", "alignment_informative")')
    code[14] = replace_once(code[14], '    for batch in tqdm(loader, desc=f"train epoch {epoch}", leave=False):',
        '    alignment_max_error = 0.0\n    for batch_index, batch in enumerate(tqdm(loader, desc=f"train epoch {epoch}", leave=False)):')
    code[14] = replace_once(code[14], '        if not torch.isfinite(loss):', '''        alignment_weight = ALIGNMENT_WEIGHT * min(1.0, epoch / max(ALIGNMENT_WARMUP_EPOCHS, 1))
        if alignment_weight > 0:
            alignment = candidate_alignment_loss(network, banks, feature_bank, epoch, batch_index)
            loss = loss + alignment_weight * alignment["loss"]
        else:
            alignment = {key: loss.new_zeros(()) for key in
                         ("loss", "positive_mass", "pair_cosine", "pairs", "unique_captions", "informative", "marginal_error")}
        alignment_max_error = max(alignment_max_error, float(alignment["marginal_error"]))
        if not torch.isfinite(loss):''')
    code[14] = replace_once(code[14], '("candidate_gate", output["candidate_gate"].mean())):',
        '("candidate_gate", output["candidate_gate"].mean()),\n'
        '                           ("alignment_loss", alignment["loss"]), ("alignment_weight", loss.new_tensor(alignment_weight)),\n'
        '                           ("alignment_positive_mass", alignment["positive_mass"]),\n'
        '                           ("alignment_pair_cosine", alignment["pair_cosine"]),\n'
        '                           ("alignment_pairs", alignment["pairs"]),\n'
        '                           ("alignment_unique_captions", alignment["unique_captions"]),\n'
        '                           ("alignment_informative", alignment["informative"])):')
    code[14] = replace_once(code[14], 'return {**metric_scores(labels_all, predictions), **{key: value / count for key, value in totals.items()}}',
        'return {**metric_scores(labels_all, predictions), **{key: value / count for key, value in totals.items()},\n'
        '            "alignment_marginal_error_max": alignment_max_error}')
    code[14] = replace_once(code[14], '    optimizer = optimizer_for(network)',
        '    write_alignment_manifest(banks, run_dir)\n    optimizer = optimizer_for(network)')
    code[14] = replace_once(code[14], '''f'c_gate={validation["mean_candidate_gate"]:.3f}')''',
        '''f'c_gate={validation["mean_candidate_gate"]:.3f} align_loss={training["alignment_loss"]:.5f} '
                  f'align_weight={training["alignment_weight"]:.3f} ot_error={training["alignment_marginal_error_max"]:.2e}')''')
    code[15] = code[15].replace('"schema": "dtd_norm_fusion_validation_v1"', '"schema": "dtd_mltm_alignment_validation_v1"')
    code[15] = replace_once(code[15], '"shuffle": target / "validation_caption_shuffle_assignments.csv"}',
        '"shuffle": target / "validation_caption_shuffle_assignments.csv",\n'
        '             "alignment": target / "training_pool_alignment_summary.csv"}')
    code[15] = replace_once(code[15], '        target.mkdir(parents=True, exist_ok=True)',
        '        alignment_fit = training_pool_alignment_summary(network, banks, bank).assign(\n'
        '            experiment=spec["name"], seed=seed, best_epoch=completed["best_epoch"])\n'
        '        target.mkdir(parents=True, exist_ok=True)')
    code[15] = replace_once(code[15], '("shuffle", assignments)):', '("shuffle", assignments), ("alignment", alignment_fit)):')
    code[17] = 'validate_alignment_settings()\n' + code[17]
    code[17] = replace_once(code[17], '"schema": "dtd_prefix_norm_fusion_v1",',
                          '"schema": "dtd_prefix_mltm_alignment_v1", "alignment": alignment_protocol_config(),')
    code[17] = replace_once(code[17], 'make_norm_calibrated_model_class(MTACSModel, DinoV3TACSClassifier)',
                          'make_mltm_alignment_model_class(MTACSModel, DinoV3TACSClassifier)')
    code[17] = code[17].replace('Architecture study 02: Prefix / candidate_only / norm fusion:',
                               'Architecture study 03: Prefix / candidate_only / norm fusion + m-LTM-inspired alignment:')
    code[18] = "## Train candidate-only with norm fusion and auxiliary alignment: 50 epochs"
    code[20] = code[20].replace("study02", "study03")
    code[24] = "## Study03 classification results, validation interventions and training-pool alignment fit"
    report = code[25].replace(norm.PREFIX, PREFIX).replace("architecture study 02", "architecture study 03")
    report = report.replace('alignment_enabled=False,', 'alignment_enabled=ALIGNMENT_WEIGHT > 0, alignment_method="mltm_inspired_image_caption_v1",')
    report = report.replace('candidate_only + norm fusion",', 'candidate_only + norm fusion + auxiliary alignment",')
    report = report.replace('Scale-control experiment, not semantic alignment; no Partial OT, semantic selector, token fusion or ASWP is enabled.',
        'Auxiliary image-caption alignment with a positive diagonal Mahalanobis metric; no Partial OT, semantic selector, token fusion or ASWP.')
    report += f'''
alignment_fit_tables = []
for seed in TRAIN_SEEDS:
    path = RUNS_DIR / "candidate_only" / f"seed_{{seed}}" / "validation_norm_diagnostics/training_pool_alignment_summary.csv"
    if path.is_file():
        alignment_fit_tables.append(pd.read_csv(path))
if alignment_fit_tables:
    alignment_fit = pd.concat(alignment_fit_tables, ignore_index=True)
    alignment_fit.to_csv(OUT_PUT_DIR / "training_pool_alignment_summary.csv", index=False)
    display(alignment_fit.round(6))
    with (OUT_PUT_DIR / "{PREFIX}_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\\n## Training-pool alignment fit, not held-out retrieval\\n\\n" + markdown_table(alignment_fit) + "\\n")
with (OUT_PUT_DIR / "{PREFIX}_report.md").open("a", encoding="utf-8") as handle:
    handle.write("\\n## Auxiliary alignment configuration\\n\\n" + json.dumps(alignment_protocol_config(), indent=2) + "\\n")
    handle.write("\\nCompare study03 against study02 under the same protocol. Captions remain synthetic weak pairs, not verified ground truth. "
                 "m-LTM-inspired adaptation: https://arxiv.org/abs/2405.10084 . No faithful audio-text reproduction is claimed.\\n")
if histories:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for seed, history in histories:
        axes[0].plot(history.epoch, history.train_alignment_loss, label=f"KL seed {{seed}}")
        axes[1].plot(history.epoch, history.train_alignment_positive_mass, label=f"positive mass seed {{seed}}")
    axes[0].set_ylabel("Auxiliary transport KL")
    axes[1].set_ylabel("Transport mass on caption-positive blocks")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.savefig(OUT_PUT_DIR / "{PREFIX}_alignment_training.png", dpi=160, bbox_inches="tight")
    plt.show()
'''
    code[25] = report
    code[6] = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    code[6] += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(norm.TARGET.read_bytes()).hexdigest()!r}\n"
    helper_text = "".join(code[i].strip() + "\n" for i in (8, 9, 11, 12, 14, 15))
    code[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helper_text.encode()).hexdigest()!r}\n"
    for i, cell in enumerate(notebook["cells"]):
        cell["source"] = (code[i].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-prefix-mltm-alignment-{i:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(architecture_study=STAGE, alignment=True,
        alignment_method="mltm_inspired_image_caption_v1", alignment_default_weight=.05,
        metric_family="positive_trace_normalized_diagonal_mahalanobis", partial_ot=False)
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
