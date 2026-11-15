"""Build study 05: fixed Prefix CSV, candidate caption as one in-ViT text token."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import base64
import gzip
import hashlib
import json
import textwrap

import build_dtd_prefix_candidate_baseline_notebook as baseline
import build_dtd_prefix_norm_fusion_notebook as norm


TARGET = PROJECT_ROOT / "notebooks/02_improvements/multimodal_architecture_ablation/05_candidate_only_text_token/dtd/mtacs-dtd-vitl16-prefix-candidate-only-text-token.ipynb"
HELPERS = PROJECT_ROOT / "src/training/mtacs_dtd_text_token_cells.py"
STAGE = "05_prefix_candidate_only_text_token"
PREFIX = "architecture_05_prefix_candidate_only_text_token"


def helper_sources():
    source = HELPERS.read_text(encoding="utf-8")
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def embedded_caption_cell():
    """Cell that writes the exact frozen Prefix CSV when CAPTIONS_INPUT_PATH is not an attached file."""
    expected = baseline.expected_prefix()
    data = (baseline.MANIFEST.parent / expected["filename"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("Local Prefix CSV does not match the manifest SHA-256")
    blob = "\n".join(textwrap.wrap(base64.b64encode(gzip.compress(data, 9, mtime=0)).decode(), 100))
    return norm.code_cell(f'''# Create captions_prefix_control.csv: the exact Florence-Prefix CSV of study01 is embedded below
# (gzip+base64), so no Florence model, GPU or extra Kaggle input is needed and the SHA-256 still matches.
import base64, gzip, hashlib
from pathlib import Path

_PREFIX_CSV_GZ_B64 = """
{blob}
"""

_given = Path(CAPTIONS_INPUT_PATH).expanduser() if CAPTIONS_INPUT_PATH else None
if _given is not None and _given.is_file():
    print("Using the attached captions CSV:", _given)
else:
    if CAPTIONS_INPUT_PATH:
        print(f"CAPTIONS_INPUT_PATH is not a file ({{CAPTIONS_INPUT_PATH}}); writing the embedded Prefix CSV instead.")
    _target = Path.cwd() / CAPTION_METHOD_EXPECTED["filename"]
    if not _target.is_file() or hashlib.sha256(_target.read_bytes()).hexdigest() != CAPTION_METHOD_EXPECTED["sha256"]:
        _data = gzip.decompress(base64.b64decode("".join(_PREFIX_CSV_GZ_B64.split())))
        if hashlib.sha256(_data).hexdigest() != CAPTION_METHOD_EXPECTED["sha256"]:
            raise ValueError("Embedded captions CSV does not match the expected SHA-256")
        _target.write_bytes(_data)
    CAPTIONS_INPUT_PATH = str(_target)
    print("Captions CSV ready:", _target, f"({{_target.stat().st_size:,}} bytes, sha256 verified)")
''')


def build_notebook():
    notebook = baseline.build_notebook()
    source_of, replace_once = baseline.prefix.source_of, baseline.prefix.replace_once
    code = {i: source_of(cell) for i, cell in enumerate(notebook["cells"])}
    helpers, norm_helpers = helper_sources(), norm.helper_sources()
    code[0] = '''# DTD Architecture Study 05: Candidate-Only / In-ViT Text Token

Controlled change relative to study 01: the selected candidate's caption no longer enters as a late
residual on the pooled feature. Instead its CLIP vector is projected to one 1024-d token and inserted
into the paired query+context token sequence, right after CLS/registers and before the patch tokens,
so every ViT block can attend to it. DINOv3 rotates only the trailing patch tokens, so this token gets no RoPE.

Initialization: the token is scale * LayerNorm(Linear(clip)), with a learned scalar scale starting at 0;
invalid/missing captions use a learned null token starting at 0. At step 0 every pair therefore sees the same
constant extra token, close to (not bitwise equal to) the visual-only path. The late candidate gate is unused.
Query text stays disabled and the selector uses image features only, exactly as in study 01.
New parameters: one Linear(512->1024)+LayerNorm, one scalar and one 1024-d null token, trained with the head LR.

Keep the exact Florence-Prefix CSV and dtd_lexical_v2: no caption generation or policy changes.
Invalid captions stay masked, but their images remain candidates. Caption validity is heuristic, not ground truth.

Protocol: DTD official split1, 1880 train/1880 validation/1880 test, 376 training candidates,
K=1 and exact-self exclusion, DINOv3 ViT-L/16, 384x384, pool/training seed42,
50 epochs with the original 50-epoch LR schedule, no early stopping, minimum validation-loss checkpoint.
Optimizer, losses, augmentations, batch policy and backbone/selector are unchanged.
Do not initialize from another study's best.pt; this experiment trains from scratch or resumes its own run.

Attach DTD, DINOv3 local code and ViT-L/16 weights, and full CLIP ViT-B/32. The notebook is standalone and offline.
captions_prefix_control.csv is embedded: if CAPTIONS_INPUT_PATH is blank or not a file, a cell writes the exact
CSV to the working directory and verifies its SHA-256. An attached CSV file at CAPTIONS_INPUT_PATH is used instead.

Output: outputs/mtacs_dtd_split1_architecture_05_prefix_candidate_only_text_token_vitl16/.
References only, never this model's metrics: study01 test F1 0.853032, visual_only (A01) test F1 0.8482.

Run All in order. After training, validation diagnostics compare aligned / text off (null token) / shuffled
captions with this model's best checkpoint and the SAME selected context images. "candidate_gate" here reports
|token scale| on active captions and weighted_text_visual_ratio = token norm / mean query patch-token norm.
A gain counts only if text_off or shuffling lowers validation F1; confirm with more seeds before claiming it.
Use validation for development; set RUN_FINAL_TEST=False while selecting changes. No test-time tuning.
'''
    code[1] = code[1].replace("Prefix candidate-only baseline", "Prefix in-ViT text token")
    code[2] = code[2].replace("DTD_PREFIX_CANDIDATE_BASELINE_SEED", "DTD_PREFIX_TEXT_TOKEN_SEED").replace(
        baseline.STAGE, STAGE)
    code[2] = replace_once(code[2], "'token_fusion': False,", "'token_fusion': True,")
    code[2] += '''
FUSION_NORM_EPS = 1e-6
TEXT_TOKEN_INIT_SCALE = 0.0  # Zero-init gate on the text token; learned with the head LR.
RUN_VALIDATION_DIAGNOSTICS = True
DIAGNOSTIC_SHUFFLE_SEED = 2026  # Fixed label-free training-pool shuffle; no test selection.
'''
    code[2] += "\n\n" + helpers["validate_text_token_output_directory"]
    code[4] = code[4].replace("DTD_PREFIX_CANDIDATE_BASELINE_OUTPUT", "DTD_PREFIX_TEXT_TOKEN_OUTPUT").replace(
        "architecture_01_prefix_candidate_only_vitl16", PREFIX + "_vitl16").replace(
        "DTD_PREFIX_CANDIDATE_BASELINE_EPOCHS", "DTD_PREFIX_TEXT_TOKEN_EPOCHS")
    code[4] = replace_once(code[4], "NUM_WORKERS = 0 if os.name == 'nt' else min(8, os.cpu_count() or 0)",
        "NUM_WORKERS = int(os.getenv('DTD_PREFIX_TEXT_TOKEN_WORKERS', str(0 if os.name == 'nt' else min(8, os.cpu_count() or 0))))\n"
        "if NUM_WORKERS < 0:\n    raise ValueError('Training workers must be nonnegative')")
    code[4] = replace_once(code[4], "OUTPUT_DIR = OUT_PUT_DIR",
                          "validate_text_token_output_directory(OUT_PUT_DIR)\nOUTPUT_DIR = OUT_PUT_DIR")
    code[12] += "\n\n" + helpers["make_text_token_model_class"]
    names = ("make_validation_caption_shuffle", "collect_norm_validation_interventions",
             "run_norm_validation_interventions", "make_norm_validation_loader")
    code[15] += "\n\n" + "\n\n\n".join(norm_helpers[name] for name in names)
    code[17] = replace_once(code[17], '"schema": "dtd_prefix_candidate_baseline_v1",',
        '"schema": "dtd_prefix_text_token_v1",\n'
        '    "fusion": {"candidate_text": "one_prefix_token_in_paired_vit_no_rope", "late_candidate_gate": "unused",\n'
        '               "init_scale": TEXT_TOKEN_INIT_SCALE, "invalid_caption": "learned_null_token",\n'
        '               "no_context_reward_branch": "unchanged"},\n'
        '    "validation_diagnostics": {"shuffle_seed": DIAGNOSTIC_SHUFFLE_SEED, "workers": 0},')
    code[17] = replace_once(code[17], "AblationModel = make_ablation_model_class(MTACSModel, DinoV3TACSClassifier)",
        "AblationModel = make_text_token_model_class(MTACSModel, DinoV3TACSClassifier)")
    code[17] = code[17].replace("Architecture baseline 01: Prefix / candidate_only:",
                                "Architecture study 05: Prefix / candidate_only / in-ViT text token:")
    code[18] = "## Train candidate-only with the in-ViT candidate text token: 50 epochs by default"
    code[20] = "## Locked final test after training and validation diagnostics"
    code[22] = "## Study 05 metrics, provenance, learning curves and validation diagnostics"
    report = code[23].replace("architecture_01_prefix_candidate_only", PREFIX)
    report = report.replace("architecture baseline 01 / Florence-Prefix / candidate_only",
                            "architecture study 05 / Florence-Prefix / candidate_only + in-ViT text token")
    report = replace_once(report, "token_fusion_enabled=False,", "token_fusion_enabled=True,")
    report = report.replace("Unchanged candidate_only architecture; the exact Prefix CSV is frozen and CLIP encodes training-pool captions only.",
                            "Candidate caption as one in-ViT prefix token (zero-init scale); the exact Prefix CSV is frozen.")
    report = report.replace("Baseline for architecture experiments; no alignment, Partial OT, semantic selector, token fusion or ASWP is enabled.",
                            "Token-level fusion experiment; no alignment, Partial OT, semantic selector, query text or ASWP is enabled.")
    report = report.replace("Historical F1 values are not metrics of this baseline.",
                            "Historical F1 values, including study01 F1 0.853032, are not metrics of this model.")
    report += f'''
diagnostic_path = OUT_PUT_DIR / "validation_interventions.csv"
if diagnostic_path.is_file():
    validation_diagnostics = pd.read_csv(diagnostic_path)
    display(validation_diagnostics.round(6))
    with (OUT_PUT_DIR / "{PREFIX}_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\\n## Fixed-context validation text interventions\\n\\n" + markdown_table(validation_diagnostics) + "\\n")
        handle.write("\\ntext_off replaces every candidate caption with the learned null token. "
                     "candidate_gate is |token scale|, not a late-fusion gate; "
                     "weighted_text_visual_ratio is token norm over mean query patch-token norm.\\n")
'''
    code[23] = report
    code[6] = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    code[6] += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(baseline.TARGET.read_bytes()).hexdigest()!r}\n"
    helper_text = "".join(code[i].strip() + "\n" for i in (8, 9, 11, 12, 14, 15))
    code[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helper_text.encode()).hexdigest()!r}\n"
    for i, cell in enumerate(notebook["cells"]):
        cell["source"] = (code[i].strip() + "\n").splitlines(keepends=True)
    notebook["cells"][20:20] = [norm.code_cell('''## Validation-only caption interventions and text-token norms

No training or checkpoint selection occurs here. This uses the completed study05 best checkpoint only,
keeps image contexts fixed across aligned/off/shuffled within each query, and never reads test queries.
The paired confidence interval describes query resampling, not variation across training seeds.
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
    notebook["cells"].insert(9, embedded_caption_cell())  # right before the CSV binding cell
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"dtd-prefix-text-token-{i:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(
        architecture_study=STAGE, token_fusion=True,
        candidate_text_fusion="one_prefix_token_in_paired_vit_no_rope_zero_init_scale",
        validation_text_interventions=True)
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
