"""Create a separate notebook-05 classifier run for an attached Qwen3-VL-8B CSV."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import copy
import hashlib
import json


SOURCE = PROJECT_ROOT / "notebooks/02_improvements/text_context_ablation/05_full_both/dtd/mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb"
TARGET = PROJECT_ROOT / "notebooks/02_improvements/caption_ablation/05_qwen3_vl_8b/dtd/mtacs-train-vitl16-dtd-caption-ablation-05-qwen3-vl-8b.ipynb"


def source_of(cell):
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected exactly one fragment: {before[:100]!r}")
    return source.replace(before, after, 1)


def caption_helpers(source):
    wanted = {"normalized_text", "bool_value", "DTD_EXTRA_TERMS", "caption_term_pattern", "load_ablation_captions",
              "prepare_caption_views", "encode_view_cache", "resolve_clip_path", "clean_dtd_caption",
              "prepare_standalone_caption_views", "caption_filter_diagnostics"}
    definitions, found = [], set()
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in wanted:
            definitions.append(ast.get_source_segment(source, node))
            found.add(name)
    if found != wanted:
        raise ValueError(f"Missing original caption helpers: {wanted - found}")
    binding = (PROJECT_ROOT / "src/captioning/dtd_qwen_ablation_input.py").read_text(encoding="utf-8")
    binding = binding.replace("from dtd_caption_screening import clean_dtd_caption\n", "")
    return "\n\n".join(definitions) + "\n\n" + binding


def build_notebook():
    notebook = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    code = {index: source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: """# DTD Caption Ablation: Qwen3-VL-8B-Instruct + M-TACS Full-Both

This is a separate copy of **notebook 05 full_both**, trained with the complete Qwen3-VL-8B-Instruct caption CSV.
The classifier is unchanged: query image + its caption, one selected context image + that context's own caption,
both text fusion gates, and visual + semantic context selection. CLIP stays frozen. No candidate label is an input.
This is a captioner comparison, not a new classifier architecture or a rerun of all eight text-role ablations.

Defaults remain official DTD split 1 (1,880 train / 1,880 validation / 1,880 test), pool seed 42, training seed 42,
K=1, 376 training-pool images, 50 epochs and no early stopping. Queries cannot select themselves. Validation/test
images and their captions never enter the candidate pool. Best checkpoint selection remains minimum validation loss;
final test runs only after the full declared training budget. Keep GPU/batch/optimizer/augmentation fixed for comparison.

## Kaggle Inputs

Attach your **complete audited Qwen8B CSV**, DTD, DINOv3 code + ViT-L/16 pretrained weights, and the same local
CLIP ViT-B/32 folder as before. **No Qwen weights, Florence weights, reference annotations or caption generation
runtime are needed here.** The notebook only encodes the precomputed text and trains M-TACS, entirely offline.
It neither downloads packages/models nor fine-tunes Qwen. Project helpers are embedded; upload this notebook directly.

The uploaded file may be named `captions.csv`, `captions_qwen3vl_8b_instruct.csv`, or another filename.
Set `CAPTIONS_INPUT_PATH` to its exact `/kaggle/input/...` path in the configuration cell, or leave it blank to
discover a unique complete Qwen8B CSV by metadata, not filename alone. Florence/32B/partial/mixed files are rejected.
If multiple DIFFERENT Qwen8B CSVs are attached, choose an explicit path; there is no automatic best-file selection.
The actual CSV SHA256 is computed and locked on first use. The builder has NOT inspected your new CSV or invented a hash.
You may optionally set `EXPECTED_CAPTIONS_SHA256` from the generation manifest for an additional check.

Policy remains `dtd_lexical_v2`: 5-8 cleaned English words, canonical DTD class terms screened; class-CLIP scores
and alias hits are diagnostic-only, not extra rejection gates. The exported short raw/final text and natural masks
must match the generator's contract. Invalid captions get zero embeddings while their images remain available.
Stored-valid text is not proof of correctness, human ground truth or absence of implicit class information.

Changing CSV can change both fusion inputs AND selected context images because full_both uses text in its selector.
Report caption coverage with accuracy/macro-F1; do not tune the caption policy using repeated test results.

Outputs, CLIP vectors and checkpoints are isolated in `outputs/mtacs_dtd_split1_caption_05_qwen3vl_8b_vitl16/`.
Start fresh initially. Resume only this method's OWN unfinished run with the same protocol and CSV hash.
Previous Florence notebooks, checkpoints and CSVs are never overwritten or silently adopted.
"""}
    changes[1] = replace_once(code[1],
        'print("Offline full-both preflight passed; caption cell can create/resume CSV with local Florence.")',
        'print("Offline Qwen8B-caption full_both training: precomputed CSV only; no Qwen/Florence generation.")')
    config = code[2].split("GENERATE_CAPTIONS_IF_MISSING =", 1)[0]
    config = replace_once(config, 'os.getenv("DTD_FULL_BOTH_SEED", "42")',
                          'os.getenv("DTD_QWEN8B_FULL_BOTH_SEED", "42")')
    config = replace_once(config,
        'CAPTIONS_INPUT_PATH = os.getenv("DTD_CAPTIONS_INPUT_PATH", "")  # Exact attached DTD captions.csv if needed.',
        'CAPTIONS_INPUT_PATH = os.getenv("DTD_QWEN8B_CAPTIONS_INPUT_PATH", "")  # Set exact Kaggle CSV path, or discover by Qwen8B metadata.')
    config = replace_once(config,
        'CAPTION_INPUT_POLICY = os.getenv("DTD_CAPTION_INPUT_POLICY", "dtd_lexical_v2")  # Or legacy_strict; use the same policy across text ablations.',
        'CAPTION_INPUT_POLICY = "dtd_lexical_v2"  # Same outer policy as previous caption-method comparisons.')
    config += '''GENERATE_CAPTIONS_IF_MISSING = False  # Precomputed Qwen8B captions only; no generation fallback.
EXPECTED_CAPTIONS_SHA256 = os.getenv("DTD_QWEN8B_CAPTIONS_SHA256", "")  # Optional hash from the generation manifest.
if DTD_SPLIT != 1:
    raise ValueError("This matched caption-method comparison uses official DTD split 1")
'''
    changes[2] = config
    changes[4] = code[4].replace("DTD_FULL_BOTH_OUTPUT", "DTD_QWEN8B_FULL_BOTH_OUTPUT").replace(
        "ablation_05_full_both_vitl16", "caption_05_qwen3vl_8b_vitl16").replace("DTD_FULL_BOTH_EPOCHS", "DTD_QWEN8B_FULL_BOTH_EPOCHS")
    changes[7] = "## Bind the uploaded Qwen8B CSV, audit masks, and cache CLIP text embeddings\n"
    changes[8] = caption_helpers(code[8])
    start = code[9].index("caption_source = ensure_dtd_caption_csv(")
    end = code[9].index("caption_df = load_ablation_captions", start)
    binding = '''caption_source, CAPTION_BINDING = stage_qwen8b_caption_csv(
    CAPTIONS_INPUT_PATH, CAPTION_CSV, SPLIT_FRAMES, class_names,
    CAPTION_INPUT_POLICY, GENERATE_CAPTIONS_IF_MISSING, EXPECTED_CAPTIONS_SHA256,
    local_candidates=[CAPTION_CSV, Path.cwd() / "captions.csv", Path.cwd() / "captions_qwen3vl_8b_instruct.csv"],
    roots=[Path("/kaggle/input")],
)
write_json(OUT_PUT_DIR / "caption_method.json", CAPTION_BINDING)
'''
    changes[9] = code[9][:start] + binding + code[9][end:]
    changes[17] = replace_once(code[17], '"schema": "dtd_full_both_v1",',
        '"schema": "dtd_qwen3vl8b_full_both_v1", "caption_method": CAPTION_BINDING,')
    changes[17] = replace_once(changes[17], 'print("Only full_both will run:",',
        'print("Qwen3-VL-8B captions with full_both:",')
    changes[18] = "## Train Qwen8B-caption full_both for 50 epochs\n"
    changes[20] = "## Final test after the declared training budget\n"
    changes[22] = "## Qwen8B caption-method metrics, learning curves, gates and coverage\n"
    report = replace_once(code[23], 'results.to_csv(OUT_PUT_DIR / "full_both_results.csv", index=False)',
        'results = results.assign(caption_method=CAPTION_BINDING["method"], caption_method_name=CAPTION_BINDING["display_name"],\n'
        '    caption_csv_sha256=CAPTION_BINDING["sha256"])\n'
        'results.to_csv(OUT_PUT_DIR / "full_both_results.csv", index=False)')
    report = report.replace("full_both_results.csv", "caption_05_qwen3vl_8b_results.csv").replace(
        "full_both_report.md", "caption_05_qwen3vl_8b_report.md").replace("full_both_training.png", "caption_05_qwen3vl_8b_training.png")
    report = replace_once(report, 'f"# DTD split {DTD_SPLIT}: full_both"',
        'f"# DTD split {DTD_SPLIT}: Qwen3-VL-8B-Instruct captions / full_both"')
    report = replace_once(report,
        '"Same classifier architecture as notebook 01; Florence creates/resumes CSV if needed, then frozen CLIP caches vectors.",',
        '"Same classifier/training as notebook 05; precomputed Qwen8B captions are hash-locked, no captions are regenerated.",')
    report = replace_once(report, '"This is one ablation result, not the comparison of all eight configurations.",',
        '"This run measures the Qwen8B caption method with full_both; compare to separately trained caption methods.",')
    report += '''
with (OUT_PUT_DIR / "caption_05_qwen3vl_8b_report.md").open("a", encoding="utf-8") as handle:
    handle.write("\\n## Caption provenance\\n\\n" + json.dumps(CAPTION_BINDING, indent=2) + "\\n")
'''
    changes[23] = report
    helpers = "".join(changes.get(index, code[index]) for index in (8, 9, 11, 12, 14, 15))
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(SOURCE.read_bytes()).hexdigest()!r}\n"
    changes[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n"
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-caption-qwen8b-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
            cell["metadata"] = {key: value for key, value in cell.get("metadata", {}).items() if key not in {"execution", "trusted"}}
    notebook["metadata"]["dtd_ablation"].update(caption_generation=False, captioner_finetuning=False,
        caption_method="qwen3_vl_8b_instruct", caption_display_name="Qwen3-VL-8B-Instruct",
        caption_csv_sha256="computed_and_locked_at_runtime", default_epochs=50,
        default_caption_input_policy="dtd_lexical_v2", source=str(SOURCE.relative_to(PROJECT_ROOT)))
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
