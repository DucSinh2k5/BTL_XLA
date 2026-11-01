"""Bind notebook 05 full_both to the uploaded Florence-Prefix caption control."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from dtd_caption_finetune import read_csv_rows


ROOT = PROJECT_ROOT
FOLDER = project_path('all_4_captions')
SOURCE = project_path('mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb')
TARGET = project_path('all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-01-florence-prefix.ipynb')
SPECS = [
    (1, "Florence-Prefix", "prefix_control", "captions_prefix_control.csv"),
    (2, "Florence-FirstValid", "first_valid_control", "captions_first_valid_control.csv"),
    (3, "Florence-SingleGrounded", "single_grounded_control", "captions_single_grounded_control.csv"),
    (4, "Florence-MultiGrounded", "multibeam_grounded", "captions_multibeam_grounded.csv"),
]


def source_of(cell):
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected exactly one source fragment: {before[:100]!r}")
    return source.replace(before, after, 1)


def inspect_caption_files(folder=FOLDER):
    catalog, expected_ids, fingerprints = [], None, set()
    required = ["image_id", "split", "raw_caption", "final_caption", "caption_valid", "generation_method",
                "caption_input_policy", "captioner_fingerprint", "prompt_version", "vlm_model"]
    for number, display_name, method, filename in SPECS:
        path = Path(folder) / filename
        rows = read_csv_rows(path, required)
        ids = [(row["image_id"], row["split"]) for row in rows]
        if len(rows) != 5640 or len({image_id for image_id, _ in ids}) != 5640:
            raise ValueError(f"{filename}: expected 5,640 distinct DTD image IDs")
        splits = dict(Counter(row["split"] for row in rows))
        if splits != {"train": 1880, "val": 1880, "test": 1880}:
            raise ValueError(f"{filename}: unexpected DTD split counts")
        per_class = Counter((row["split"], row["image_id"].split("/")[0]) for row in rows)
        if len(per_class) != 47 * 3 or set(per_class.values()) != {40}:
            raise ValueError(f"{filename}: expected 40 images per texture per split")
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("The four controls must use identical image IDs, order and split assignments")
        expected_ids = ids
        if {row["generation_method"] for row in rows} != {method}:
            raise ValueError(f"{filename}: contents do not match generation_method={method}")
        if {row["caption_input_policy"] for row in rows} != {"dtd_lexical_v2"}:
            raise ValueError(f"{filename}: unexpected caption policy")
        fingerprint = {row["captioner_fingerprint"] for row in rows}
        prompt = {row["prompt_version"] for row in rows}
        model = {row["vlm_model"] for row in rows}
        if len(fingerprint) != 1 or len(prompt) != 1 or len(model) != 1:
            raise ValueError(f"{filename}: mixed caption generation metadata")
        fingerprints.update(fingerprint)
        valid_by_split = {split: sum(row["split"] == split and row["caption_valid"].lower() in {"true", "1", "yes"}
                                    for row in rows) for split in ("train", "val", "test")}
        catalog.append({"number": number, "display_name": display_name, "generation_method": method,
            "filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows), "split_counts": splits, "stored_valid_by_split": valid_by_split,
            "stored_valid_total": sum(valid_by_split.values()), "captioner_fingerprint": next(iter(fingerprint)),
            "prompt_version": next(iter(prompt)), "vlm_model": next(iter(model)), "policy": "dtd_lexical_v2"})
    if len(fingerprints) != 1:
        raise ValueError("The four uploaded CSVs do not belong to the same caption-generation protocol")
    return catalog


PREFIX_INPUT_HELPERS = '''
import csv


def verify_method_caption_csv(path, split_frames, expected):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("Wrong or modified caption CSV. This notebook requires the uploaded Florence-Prefix file; "
                         "do not substitute another method or a matched-mask CSV.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "split", "raw_caption", "final_caption", "caption_valid", "generation_method",
                    "captioner_fingerprint", "caption_input_policy", "prompt_version", "vlm_model"}
        if not required.issubset(reader.fieldnames or []) or len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError("Caption CSV has missing or duplicate columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Caption CSV contains a malformed/partial row")
    ids = [row["image_id"] for row in rows]
    expected_ids = {str(image_id): split for split, frame in split_frames.items() for image_id in frame.image_id}
    if (len(expected_ids) != sum(len(frame) for frame in split_frames.values()) or len(set(ids)) != len(ids)
            or set(ids) != set(expected_ids) or any(row["split"] != expected_ids[row["image_id"]] for row in rows)):
        raise ValueError("Caption CSV does not match the official DTD train/val/test image IDs and splits")
    for column, key in (("generation_method", "generation_method"), ("captioner_fingerprint", "captioner_fingerprint"),
                        ("prompt_version", "prompt_version"), ("vlm_model", "vlm_model"), ("caption_input_policy", "policy")):
        if {row[column] for row in rows} != {expected[key]}:
            raise ValueError(f"Unexpected {column}; another caption method cannot be used in this notebook")
    return {"method": expected["generation_method"], "display_name": expected["display_name"],
            "sha256": digest, "captioner_fingerprint": expected["captioner_fingerprint"],
            "prompt_version": expected["prompt_version"], "policy": expected["policy"], "rows": len(rows)}


def resolve_method_caption_csv(explicit_path, expected, local_candidates=(), roots=()):
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CAPTIONS_INPUT_PATH not found: {path}. Attach the Prefix CSV and set its exact path.")
        return path.resolve()
    candidates = [Path(path) for path in local_candidates if Path(path).is_file()]
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            depth = len(Path(current).relative_to(root).parts)
            dirs[:] = sorted(name for name in dirs if name not in {"images", ".git", "__pycache__", "node_modules"}) if depth < 7 else []
            if expected["filename"] in files:
                candidates.append(Path(current) / expected["filename"])
    matching = {str(path.resolve()): path.resolve() for path in candidates
                if hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]}
    if not matching:
        raise FileNotFoundError(
            f"No matching {expected['filename']} found. Attach this CSV as Kaggle Input, then set CAPTIONS_INPUT_PATH "
            "to its exact path. Caption regeneration and fallback to another captions.csv are disabled.")
    return min(matching.values(), key=lambda path: (len(str(path)), str(path)))


def stage_method_caption_csv(explicit_path, cached_path, split_frames, expected, local_candidates=(), roots=()):
    if GENERATE_CAPTIONS_IF_MISSING or CAPTION_INPUT_POLICY != "dtd_lexical_v2":
        raise ValueError("This comparison requires precomputed captions and dtd_lexical_v2; do not enable regeneration")
    source = resolve_method_caption_csv(explicit_path, expected, local_candidates, roots)
    binding = verify_method_caption_csv(source, split_frames, expected)
    cached = Path(cached_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError("Output directory already contains another caption CSV. Use a new output directory; nothing was overwritten.")
    if cached.resolve() != source.resolve() and not cached.is_file():
        shutil.copy2(source, cached)
    verify_method_caption_csv(cached, split_frames, expected)
    write_json(cached.parent / "caption_source.json", {**binding, "source": str(source), "generation_needed": False})
    print("Bound caption method:", binding["display_name"], "| CSV:", source)
    print("Caption CSV SHA256:", binding["sha256"])
    return cached, binding
'''


def build_notebook(catalog=None):
    catalog = inspect_caption_files() if catalog is None else catalog
    prefix = catalog[0]
    notebook = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    code = {index: source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: f'''# DTD Caption Ablation 01: Florence-Prefix + M-TACS Full-Both

Method 1 of the four caption-generation controls: **Florence-Prefix** (`prefix_control`).
This is the freshly generated greedy-prefix CSV, not a replay of the historical Florence CSV results.
The CSV is already complete: 5,640 DTD images, 1,880 in each official split, with
{prefix["stored_valid_total"]:,} stored-valid captions. Stored validity is not proof of visual truth or absence of aliases.

This notebook is copied from notebook 05 full_both. It trains the same query-image/query-text plus
one selected context-image/context-text classifier, with the same visual+semantic selector,
losses, transforms and optimizer. Only the caption input binding, provenance and artifact names change.
Both text fusion gates and the semantic selector stay enabled; this is NOT visual_only or query_only.

Use official DTD split 1, pool seed 42, training seed 42, K=1 and 50 epochs by default.
Batch size retains notebook 05's VRAM-dependent policy (32 on >=80 GiB); use the same GPU/batch policy
for the other three caption methods. Checkpoint selection remains minimum validation loss.
Candidate pool: 376 training images, self-selection excluded; validation/test images never enter the pool.
Each query may use its own image-derived caption without supplying its classification label.

Attach `captions_prefix_control.csv`, DTD, DINOv3 code/ViT-L weights and local CLIP, then Run All.
**No Florence model, reference captions, caption generation or fine-tuning is required here.**
Leave `CAPTIONS_INPUT_PATH` blank to discover the exact filename and hash in Kaggle Inputs, or set
its exact attached path. A different method/modified CSV is rejected, not silently substituted.
There are no project-local imports; only this notebook and the attached assets are needed on Kaggle.

Policy stays `dtd_lexical_v2`: caption content/masks are audited as in notebook 05, class-CLIP scores
are diagnostic only. Alias/synonym leakage is not certified absent. Invalid text is masked, not replaced.
This is a natural-mask comparison; matched-mask CSVs require a separately declared experiment.

Outputs/checkpoints/CLIP cache are isolated in `outputs/mtacs_dtd_split1_caption_01_florence_prefix_vitl16/`.
Train from scratch on the first run; resume only this notebook's own unfinished run in this directory.
Do not copy checkpoints or caches from notebook 05 or another caption method. Report test once after 50 epochs.
No metrics are prefilled; historical full_both scores are not this method's results.
'''}
    changes[1] = code[1].replace("Offline full-both preflight passed; caption cell can create/resume CSV with local Florence.",
        "Offline Florence-Prefix training preflight passed; precomputed captions only, no generation.")
    config = replace_once(code[2], 'os.getenv("DTD_FULL_BOTH_SEED", "42")', 'os.getenv("DTD_CAPTION_01_PREFIX_SEED", "42")')
    config = replace_once(config,
        'CAPTIONS_INPUT_PATH = os.getenv("DTD_CAPTIONS_INPUT_PATH", "")  # Exact attached DTD captions.csv if needed.',
        'CAPTIONS_INPUT_PATH = os.getenv("DTD_PREFIX_CAPTIONS_INPUT_PATH", "")  # Blank: discover exact Prefix filename + hash in Inputs.')
    config = replace_once(config,
        'CAPTION_INPUT_POLICY = os.getenv("DTD_CAPTION_INPUT_POLICY", "dtd_lexical_v2")  # Or legacy_strict; use the same policy across text ablations.',
        'CAPTION_INPUT_POLICY = "dtd_lexical_v2"  # Fixed across all four caption methods.')
    config = replace_once(config,
        'GENERATE_CAPTIONS_IF_MISSING = True  # Reuse complete CSVs; otherwise create/resume with local models.',
        'GENERATE_CAPTIONS_IF_MISSING = False  # Never regenerate or silently substitute another method.')
    config += "\nCAPTION_METHOD_EXPECTED = " + repr(prefix) + "\n"
    config += '\nif DTD_SPLIT != 1:\n    raise ValueError("These uploaded captions use official DTD split 1; another split needs its own declared CSV experiment")\n'
    changes[2] = config
    changes[4] = code[4].replace("DTD_FULL_BOTH_OUTPUT", "DTD_CAPTION_01_PREFIX_OUTPUT").replace(
        "ablation_05_full_both_vitl16", "caption_01_florence_prefix_vitl16").replace("DTD_FULL_BOTH_EPOCHS", "DTD_CAPTION_01_PREFIX_EPOCHS")
    changes[7] = "## Load the exact Florence-Prefix CSV; audit both text roles and cache CLIP vectors\n"
    changes[8] = code[8] + "\n\n" + PREFIX_INPUT_HELPERS
    first = code[9].index("caption_source = ensure_dtd_caption_csv(")
    last = code[9].index("caption_df = load_ablation_captions", first)
    binding = '''caption_source, CAPTION_BINDING = stage_method_caption_csv(
    CAPTIONS_INPUT_PATH, CAPTION_CSV, SPLIT_FRAMES, CAPTION_METHOD_EXPECTED,
    local_candidates=[Path.cwd() / CAPTION_METHOD_EXPECTED["filename"],
        Path.cwd() / "all_4_captions" / CAPTION_METHOD_EXPECTED["filename"],
        PROJECT_DIR / CAPTION_METHOD_EXPECTED["filename"]],
    roots=[Path("/kaggle/input")],
)
write_json(OUT_PUT_DIR / "caption_method.json", CAPTION_BINDING)
'''
    changes[9] = code[9][:first] + binding + code[9][last:]
    changes[17] = replace_once(code[17], '"schema": "dtd_full_both_v1",',
        '"schema": "dtd_caption_method_full_both_v1", "caption_method": CAPTION_BINDING,')
    changes[17] = replace_once(changes[17], 'print("Only full_both will run:",',
        'print(CAPTION_METHOD_EXPECTED["display_name"], "with full_both:",')
    changes[18] = "## Train Florence-Prefix full_both only: 50 epochs by default\n"
    changes[22] = "## Florence-Prefix metrics, learning curves, gates and caption coverage\n"
    report = replace_once(code[23], 'results.to_csv(OUT_PUT_DIR / "full_both_results.csv", index=False)',
        'results = results.assign(caption_method=CAPTION_BINDING["method"], caption_method_name=CAPTION_BINDING["display_name"],\n'
        '    caption_csv_sha256=CAPTION_BINDING["sha256"])\n'
        'results.to_csv(OUT_PUT_DIR / "full_both_results.csv", index=False)')
    report = report.replace("full_both_results.csv", "caption_01_florence_prefix_results.csv").replace(
        "full_both_report.md", "caption_01_florence_prefix_report.md").replace("full_both_training.png", "caption_01_florence_prefix_training.png")
    report = replace_once(report, 'f"# DTD split {DTD_SPLIT}: full_both"',
        'f"# DTD split {DTD_SPLIT}: Florence-Prefix / full_both"')
    report = replace_once(report,
        "Same classifier architecture as notebook 01; Florence creates/resumes CSV if needed, then frozen CLIP caches vectors.",
        "Same classifier/training as notebook 05; the precomputed Prefix CSV is hash-locked, no captions are regenerated.")
    report = replace_once(report, "This is one ablation result, not the comparison of all eight configurations.",
        "This is caption method 01 only; train the other three caption methods independently for the comparison table.")
    report += '''
with (OUT_PUT_DIR / "caption_01_florence_prefix_report.md").open("a", encoding="utf-8") as handle:
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
        cell["id"] = f"dtd-caption-prefix-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell))
            cell.update(outputs=[], execution_count=None)
            cell["metadata"] = {key: value for key, value in cell.get("metadata", {}).items()
                                if key not in {"execution", "trusted"}}
    notebook["metadata"]["dtd_ablation"].update(caption_generation=False, default_epochs=50,
        caption_method="prefix_control", caption_display_name="Florence-Prefix", caption_csv_sha256=prefix["sha256"],
        captioner_fingerprint=prefix["captioner_fingerprint"], default_caption_input_policy="dtd_lexical_v2")
    return notebook


def main():
    catalog = inspect_caption_files()
    notebook = build_notebook(catalog)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (FOLDER / "caption_methods_manifest.json").write_text(json.dumps({"methods": catalog,
        "created_training_notebooks": {"prefix_control": TARGET.relative_to(ROOT).as_posix()}, "no_training_executed": True},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
    for method in catalog:
        print(method["number"], method["display_name"], method["filename"], method["stored_valid_total"])


if __name__ == "__main__":
    main()
