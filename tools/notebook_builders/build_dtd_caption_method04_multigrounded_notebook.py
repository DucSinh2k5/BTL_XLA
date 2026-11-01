"""Bind the unchanged full_both classifier to Florence-MultiGrounded, method 04."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import hashlib
import json

import build_dtd_caption_method01_prefix_notebook as prefix
import build_dtd_caption_method02_firstvalid_notebook as firstvalid
import build_dtd_caption_method03_singlegrounded_notebook as single


FOLDER = prefix.FOLDER
TARGET = project_path('all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-04-florence-multigrounded.ipynb')


def build_notebook(catalog=None):
    catalog = prefix.inspect_caption_files(FOLDER) if catalog is None else catalog
    selected = catalog[3]
    if selected["generation_method"] != "multibeam_grounded" or selected["number"] != 4:
        raise ValueError("Method 04 requires the uploaded multibeam_grounded CSV")
    notebook = single.build_notebook(catalog)
    code = {index: prefix.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: f'''# DTD Caption Ablation 04: Florence-MultiGrounded + M-TACS Full-Both

Method 4 of the four caption-generation controls: **Florence-MultiGrounded** (`multibeam_grounded`).
For each image, the caption generator proposed one greedy Florence description and three beam
hypotheses. It extracted eligible 5-8-word spans and selected a span by CLIP image-text cosine
similarity to that same image. The supplied CSV already contains the chosen short text.
Three beam hypotheses are caption alternatives, NOT three context images: M-TACS still uses K=1.

This is the paired comparison to method 3 Florence-SingleGrounded. Both use the same extraction,
lexical policy and image-CLIP ranking; MultiGrounded expands the proposal pool beyond the single
greedy description. These two uploaded CSVs share image IDs, split assignments and validity masks.
The main CSV has 5,640 images, 1,880 per official split, and {selected["stored_valid_total"]:,} stored-valid captions.
Differences can still change the context selected downstream; contexts are not artificially held fixed.

The global class vocabulary was used for lexical screening and diagnostic class scores, not a
per-image target for ranking. Higher optimized CLIP similarity is NOT independent evidence of
better captions or improved classification. Generated descriptions can be wrong, generic or contain
aliases/implicit class information; full coverage is not a no-leak certificate or caption ground truth.

The M-TACS classifier and training/test functions remain identical to notebook 05 and methods 1-3.
Inputs remain query image + query caption + one selected context image + its own caption; both text
fusion gates and the visual+semantic selector stay enabled. CLIP is frozen. No caption ranking is
rerun during classifier training and its offline scores are not added as new classifier features.

Defaults: official DTD split 1, pool seed 42, training seed 42, K=1, 50 epochs, no early stopping.
The pool contains 376 training images, excludes the query itself, and never includes val/test images.
Select checkpoints by minimum validation loss. Keep the same optimizer, losses, augmentations and
GPU/batch policy as methods 1-3 (batch 32 on >=80 GiB under the original VRAM-dependent settings).

Attach `captions_multibeam_grounded.csv`, DTD, DINOv3 code/ViT-L weights and local CLIP, then Run All.
Leave `CAPTIONS_INPUT_PATH` blank to discover the exact filename/hash in Inputs or set its path.
Captions are precomputed: no Florence weights, reference annotations, captioner training, generation
or Internet are required here. There are no project-local imports on Kaggle. Wrong/modified files
are rejected instead of falling back to the other caption methods.

Policy remains `dtd_lexical_v2`; class-CLIP audit scores are diagnostic-only. Keep image-caption
ranking scores, text-class audit scores and learned task-selector scores conceptually separate.
Do not enable regeneration or replace this natural-mask CSV with a matched-mask export to bypass a guard.

Outputs, text cache and checkpoints are isolated in
`outputs/mtacs_dtd_split1_caption_04_florence_multigrounded_vitl16/`.
Train from scratch initially and resume only this method's own unfinished run. Do not load classifiers
or caches from methods 1-3 or historical notebook 05. Final test opens after the declared 50-epoch
budget, with no test-score tuning or prefilled metrics.
'''}
    changes[1] = code[1].replace("Florence-SingleGrounded", "Florence-MultiGrounded")
    config = prefix.replace_once(code[2], "CAPTION_METHOD_EXPECTED = " + repr(catalog[2]),
        "CAPTION_METHOD_EXPECTED = " + repr(selected))
    changes[2] = config.replace("DTD_CAPTION_03_SINGLE_GROUNDED_SEED", "DTD_CAPTION_04_MULTI_GROUNDED_SEED").replace(
        "DTD_SINGLE_GROUNDED_CAPTIONS_INPUT_PATH", "DTD_MULTI_GROUNDED_CAPTIONS_INPUT_PATH").replace(
        "discover exact SingleGrounded filename", "discover exact MultiGrounded filename")
    changes[4] = code[4].replace("DTD_CAPTION_03_SINGLE_GROUNDED_OUTPUT", "DTD_CAPTION_04_MULTI_GROUNDED_OUTPUT").replace(
        "caption_03_florence_singlegrounded_vitl16", "caption_04_florence_multigrounded_vitl16").replace(
        "DTD_CAPTION_03_SINGLE_GROUNDED_EPOCHS", "DTD_CAPTION_04_MULTI_GROUNDED_EPOCHS")
    changes[7] = "## Load the exact Florence-MultiGrounded CSV; audit both roles and cache CLIP vectors\n"
    changes[8] = prefix.replace_once(code[8], "This notebook requires the uploaded Florence-SingleGrounded file;",
        "This notebook requires the uploaded Florence-MultiGrounded file;")
    changes[8] = prefix.replace_once(changes[8], "Attach the SingleGrounded CSV and set its exact path.",
        "Attach the MultiGrounded CSV and set its exact path.")
    changes[18] = "## Train Florence-MultiGrounded full_both only: 50 epochs by default\n"
    changes[22] = "## Florence-MultiGrounded metrics, learning curves, gates and caption coverage\n"
    changes[23] = code[23].replace("caption_03_florence_singlegrounded", "caption_04_florence_multigrounded").replace(
        "Florence-SingleGrounded / full_both", "Florence-MultiGrounded / full_both").replace(
        "precomputed SingleGrounded CSV", "precomputed MultiGrounded CSV").replace(
        "This is caption method 03 only;", "This is caption method 04 only;")
    helpers = "".join(changes.get(index, code[index]) for index in (8, 9, 11, 12, 14, 15))
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(prefix.SOURCE.read_bytes()).hexdigest()!r}\n"
    changes[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n"
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-caption-multigrounded-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(prefix.source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(caption_method="multibeam_grounded",
        caption_display_name="Florence-MultiGrounded", caption_csv_sha256=selected["sha256"],
        captioner_fingerprint=selected["captioner_fingerprint"], caption_generation=False, default_epochs=50)
    return notebook


def main():
    catalog = prefix.inspect_caption_files(FOLDER)
    notebook = build_notebook(catalog)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    path = FOLDER / "caption_methods_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    created = manifest.get("created_training_notebooks", {})
    for method, target in (("prefix_control", prefix.TARGET), ("first_valid_control", firstvalid.TARGET),
                           ("single_grounded_control", single.TARGET), ("multibeam_grounded", TARGET)):
        if target.is_file():
            created[method] = target.relative_to(prefix.ROOT).as_posix()
    manifest.update(methods=catalog, created_training_notebooks=created, no_training_executed=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
    print("Bound CSV:", catalog[3]["filename"], "| SHA256:", catalog[3]["sha256"])


if __name__ == "__main__":
    main()
