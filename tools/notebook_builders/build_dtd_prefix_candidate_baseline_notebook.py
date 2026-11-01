"""Build architecture-study baseline 01: candidate_only with the exact Prefix CSV."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_dtd_caption_method01_prefix_notebook as prefix
import build_mtacs_dtd_candidate_only_notebook as candidate


TARGET = PROJECT_ROOT / "notebooks/02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/mtacs-dtd-vitl16-prefix-candidate-only-baseline.ipynb"
MANIFEST = PROJECT_ROOT / "data/captions/dtd/grounded_controls/caption_methods_manifest.json"
SOURCE = candidate.TARGET
STAGE = "01_prefix_candidate_only_baseline"


def expected_prefix():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = next(row for row in manifest["methods"] if row["generation_method"] == "prefix_control")
    source = MANIFEST.parent / expected["filename"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("The canonical Florence-Prefix CSV no longer matches its recorded manifest")
    return expected


def caption_helpers(source):
    needed = {"normalized_text", "bool_value", "DTD_EXTRA_TERMS", "caption_term_pattern",
              "load_ablation_captions", "prepare_caption_views", "encode_view_cache", "resolve_clip_path",
              "clean_dtd_caption", "prepare_standalone_caption_views", "caption_filter_diagnostics"}
    selected = []
    found = set()
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in needed:
            selected.append(ast.get_source_segment(source, node))
            found.add(name)
    if found != needed:
        raise ValueError(f"Missing caption helpers: {sorted(needed - found)}")
    return "\n\n\n".join(selected) + "\n\n" + prefix.PREFIX_INPUT_HELPERS


def build_notebook():
    expected = expected_prefix()
    notebook = candidate.build_notebook()
    code = {i: prefix.source_of(cell) for i, cell in enumerate(notebook["cells"])}
    changes = {0: f'''# DTD Architecture Study 01: Florence-Prefix / Candidate-Only Baseline

The first notebook for the alignment, selector, token-fusion and pooling experiments.
This run is the **unchanged candidate_only architecture**, bound to the exact Florence-Prefix CSV.
It does not implement m-LTM, Partial OT, a text-aware selector, a resampler or ASWP yet.

Input: query image + one selected training-context image + that context image's caption.
The selector uses image features only. It gathers the caption at the selected image index.
Query text is zero/masked for every train/val/test query. Invalid candidate captions are also zero/masked,
but their images remain available. Candidate class labels are not model or selector inputs.

Retained recipe: DINOv3 ViT-L/16, 1024 visual dimensions, 384x384 images, official DTD split1,
1,880 train / 1,880 validation / 1,880 test, a fixed 376-image training pool, exact self-exclusion,
K=1, seed42 and 50 epochs with a 50-epoch LR schedule, no early stopping.
Best checkpoint uses minimum validation cross-entropy. Batch sizes retain the original GPU-memory policy
(32 train on >=80 GiB, 16 on >=40 GiB, 8 on >=20 GiB, otherwise 4); match the GPU/batch policy in later runs.

Attach DTD, local DINOv3 code, ViT-L/16 weights, full CLIP ViT-B/32 and **captions_prefix_control.csv**.
Only this notebook is required; no project Python files, Florence/Qwen checkpoint, caption references,
offline legacy wheels or Internet access are needed. No captions are generated, repaired or substituted.
Set CAPTIONS_INPUT_PATH to the exact uploaded CSV, or leave it blank for filename + SHA256 discovery.
The CSV must match `{expected['sha256']}`; renamed copies work with an explicit path if the bytes are unchanged.

Policy is fixed to dtd_lexical_v2. The original CSV is copied byte-for-byte; the shared audit/input export records
the exact cleaned text/masks. CLIP class scores are diagnostics, not a 0.30 hard rejection or proof of no leakage.
The CSV contains {expected['rows']:,} rows with {expected['stored_valid_total']:,} stored-valid captions;
only the 376 training-pool rows are text-encoded and used by this baseline.
These are model-generated captions, not human-verified ground truth.

Run All from the top. Default output is independent of all prior runs:
`outputs/mtacs_dtd_split1_architecture_01_prefix_candidate_only_vitl16/`.
Resume only this run's own checkpoint. Tokenizer parallelism is disabled before tokenizer use or DataLoader workers.
The final test runs only after the declared training budget finishes; set RUN_FINAL_TEST=False during development.
Keep the entire output directory for later matched comparisons. Do not copy old full_both/candidate_only checkpoints.

No score is prefilled. The earlier candidate_only F1 85.0321% came from another caption export;
Florence-Prefix F1 81.0651% came from full_both. Neither number is this new baseline's result.
Later variants must keep this CSV and the data/pool/training protocol fixed and select settings using validation.
'''}
    changes[1] = prefix.replace_once(code[1],
        "Offline candidate-only preflight passed; caption cell can create/resume CSV with local Florence.",
        "Offline Prefix candidate-only baseline preflight passed; precomputed CSV only, no caption generation.")
    changes[1] = prefix.replace_once(changes[1], 'os.environ["TRANSFORMERS_OFFLINE"] = "1"',
        'os.environ["TRANSFORMERS_OFFLINE"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"')
    config = code[2].split("GENERATE_CAPTIONS_IF_MISSING =", 1)[0]
    config = prefix.replace_once(config, 'os.getenv("DTD_CANDIDATE_ONLY_SEED", "42")',
                                  'os.getenv("DTD_PREFIX_CANDIDATE_BASELINE_SEED", "42")')
    config = prefix.replace_once(config,
        'CAPTIONS_INPUT_PATH = os.getenv("DTD_CAPTIONS_INPUT_PATH", "")  # Exact attached DTD captions.csv if needed.',
        'CAPTIONS_INPUT_PATH = os.getenv("DTD_PREFIX_CAPTIONS_INPUT_PATH", "")  # Exact uploaded Prefix CSV, or blank for hash-checked discovery.')
    config = prefix.replace_once(config,
        'CAPTION_INPUT_POLICY = os.getenv("DTD_CAPTION_INPUT_POLICY", "dtd_lexical_v2")  # Or legacy_strict; use the same policy across text ablations.',
        'CAPTION_INPUT_POLICY = "dtd_lexical_v2"  # Fixed across this architecture study.')
    config = prefix.replace_once(config,
        'CAPTION_SEMANTIC_THRESHOLD = 0.30  # Fixed screening policy; never auto-tuned using test scores.',
        'CAPTION_SEMANTIC_THRESHOLD = 0.30  # Legacy diagnostic only under dtd_lexical_v2; not a hard rejection.')
    config += '\nGENERATE_CAPTIONS_IF_MISSING = False  # Frozen CSV input; never regenerate in this experiment.\n'
    config += f"CAPTION_METHOD_EXPECTED = {expected!r}\n"
    config += f"ARCHITECTURE_STUDY = {{'group': 'dtd_prefix_architecture_ablation_v1', 'stage': {STAGE!r},\n"
    config += "    'alignment': False, 'partial_ot': False, 'semantic_selector': False,\n"
    config += "    'token_fusion': False, 'aswp': False, 'caption_supervision': 'model_generated_not_human_ground_truth'}\n"
    config += '\nif DTD_SPLIT != 1:\n    raise ValueError("The frozen Prefix CSV uses DTD split1; another split needs a separately declared caption experiment")\n'
    changes[2] = config
    changes[4] = code[4].replace("DTD_CANDIDATE_ONLY_OUTPUT", "DTD_PREFIX_CANDIDATE_BASELINE_OUTPUT").replace(
        "ablation_03_candidate_only_vitl16", "architecture_01_prefix_candidate_only_vitl16").replace(
        "DTD_CANDIDATE_ONLY_EPOCHS", "DTD_PREFIX_CANDIDATE_BASELINE_EPOCHS")
    changes[4] = changes[4].replace("/kaggle/input/datasets/quctriunguyn/dinov3-code/", "/kaggle/input/datasets/nguyenducxu/dinov3-code/").replace(
        "/kaggle/input/datasets/quctriunguyn/dinov3-vitl16/", "/kaggle/input/datasets/nguyenducxu/dinov3-vitl16/")
    changes[7] = "## Bind the exact Prefix CSV; audit and encode training-pool captions only\n"
    changes[8] = caption_helpers(code[8])
    first = code[9].index("caption_source = ensure_dtd_caption_csv(")
    last = code[9].index("caption_df = load_ablation_captions", first)
    binding = '''caption_source, CAPTION_BINDING = stage_method_caption_csv(
    CAPTIONS_INPUT_PATH, CAPTION_CSV, SPLIT_FRAMES, CAPTION_METHOD_EXPECTED,
    local_candidates=[CAPTION_CSV, Path.cwd() / CAPTION_METHOD_EXPECTED["filename"],
        Path.cwd() / "data/captions/dtd/grounded_controls" / CAPTION_METHOD_EXPECTED["filename"],
        PROJECT_DIR / CAPTION_METHOD_EXPECTED["filename"]],
    roots=[Path("/kaggle/input")],
)
write_json(OUT_PUT_DIR / "caption_method.json", CAPTION_BINDING)
write_json(OUT_PUT_DIR / "architecture_study.json", ARCHITECTURE_STUDY)
'''
    changes[9] = code[9][:first] + binding + code[9][last:]
    changes[17] = prefix.replace_once(code[17], '"schema": "dtd_candidate_only_v1",',
        '"schema": "dtd_prefix_candidate_baseline_v1", "architecture_study": ARCHITECTURE_STUDY,\n'
        '    "caption_method": CAPTION_BINDING,')
    changes[17] = prefix.replace_once(changes[17], 'print("Only candidate_only will run:",',
        'print("Architecture baseline 01: Prefix / candidate_only:",')
    changes[18] = "## Train the Prefix candidate-only baseline: 50 epochs by default\n"
    changes[20] = "## Locked final test after the baseline finishes\n"
    changes[22] = "## Baseline results, provenance, learning curves and candidate-caption coverage\n"
    report = prefix.replace_once(code[23], 'results.to_csv(OUT_PUT_DIR / "candidate_only_results.csv", index=False)',
        'results = results.assign(study_stage=ARCHITECTURE_STUDY["stage"], caption_method=CAPTION_BINDING["method"],\n'
        '    caption_csv_sha256=CAPTION_BINDING["sha256"], query_text_enabled=False,\n'
        '    candidate_text_enabled=True, semantic_selector_enabled=False, alignment_enabled=False,\n'
        '    token_fusion_enabled=False, aswp_enabled=False)\n'
        'results.to_csv(OUT_PUT_DIR / "candidate_only_results.csv", index=False)')
    report = report.replace("candidate_only_results.csv", "architecture_01_prefix_candidate_only_results.csv").replace(
        "candidate_only_report.md", "architecture_01_prefix_candidate_only_report.md").replace(
        "candidate_only_training.png", "architecture_01_prefix_candidate_only_training.png")
    report = prefix.replace_once(report, 'f"# DTD split {DTD_SPLIT}: candidate_only"',
        'f"# DTD split {DTD_SPLIT}: architecture baseline 01 / Florence-Prefix / candidate_only"')
    report = prefix.replace_once(report,
        "Same classifier architecture as notebooks 01/02; Florence creates/resumes CSV if needed, then CLIP encodes pool captions only.",
        "Unchanged candidate_only architecture; the exact Prefix CSV is frozen and CLIP encodes training-pool captions only.")
    report = prefix.replace_once(report,
        "This is one ablation result, not the comparison of all eight configurations.",
        "Baseline for architecture experiments; no alignment, Partial OT, semantic selector, token fusion or ASWP is enabled.")
    report += '''
with (OUT_PUT_DIR / "architecture_01_prefix_candidate_only_report.md").open("a", encoding="utf-8") as handle:
    handle.write("\\n## Caption provenance\\n\\n" + json.dumps(CAPTION_BINDING, indent=2) + "\\n")
    handle.write("\\nThese captions are synthetic model inputs, not human-verified ground truth. "
                 "Historical F1 values are not metrics of this baseline.\\n")
'''
    changes[23] = report
    helpers = "".join(changes.get(i, code[i]) for i in (8, 9, 11, 12, 14, 15))
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(SOURCE.read_bytes()).hexdigest()!r}\n"
    changes[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n"
    for i, cell in enumerate(notebook["cells"]):
        if i in changes:
            cell["source"] = (changes[i].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-prefix-candidate-baseline-{i:02d}"
        if cell["cell_type"] == "code":
            ast.parse(prefix.source_of(cell))
            cell.update(outputs=[], execution_count=None)
            cell["metadata"] = {k: v for k, v in cell.get("metadata", {}).items() if k not in {"execution", "trusted"}}
    notebook["metadata"]["dtd_ablation"].update(caption_generation=False, caption_method="prefix_control",
        caption_display_name="Florence-Prefix", caption_csv_sha256=expected["sha256"],
        captioner_fingerprint=expected["captioner_fingerprint"], default_caption_input_policy="dtd_lexical_v2",
        architecture_study=STAGE, default_epochs=50, image_size=384, context_k=1,
        backbone="dinov3_vitl16", query_fusion=False, candidate_fusion=True, semantic_selector=False)
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created", TARGET)


if __name__ == "__main__":
    main()
