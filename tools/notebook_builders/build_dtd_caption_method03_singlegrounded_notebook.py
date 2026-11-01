"""Bind the unchanged full_both classifier to Florence-SingleGrounded, method 03."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import hashlib
import json

import build_dtd_caption_method01_prefix_notebook as prefix
import build_dtd_caption_method02_firstvalid_notebook as firstvalid


FOLDER = prefix.FOLDER
TARGET = project_path('all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-03-florence-singlegrounded.ipynb')


def build_notebook(catalog=None):
    catalog = prefix.inspect_caption_files(FOLDER) if catalog is None else catalog
    selected = catalog[2]
    if selected["generation_method"] != "single_grounded_control" or selected["number"] != 3:
        raise ValueError("Method 03 requires the uploaded single_grounded_control CSV")
    notebook = firstvalid.build_notebook(catalog)
    code = {index: prefix.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: f'''# DTD Caption Ablation 03: Florence-SingleGrounded + M-TACS Full-Both

Method 3 of the four caption-generation controls: **Florence-SingleGrounded** (`single_grounded_control`).
The saved caption generator extracted eligible 5-8-word spans from one greedy Florence description
per image, then selected a span using CLIP image-text cosine similarity to that same image.
No additional beam hypotheses are used by this control. The global class vocabulary is used only
for lexical screening and diagnostic class scores, never as a per-image label or ranking target.

This is the paired comparison to method 2 Florence-FirstValid: both methods use the same greedy
description/span pool and caption availability, but FirstValid takes the first eligible span while
SingleGrounded ranks eligible spans by image-text cosine. The uploaded CSV has 5,640 images,
1,880 per official split, with {selected["stored_valid_total"]:,} stored-valid captions. Text can change the
context chosen by the downstream semantic selector; this is not a fixed-context intervention.

Higher CLIP similarity is optimized by caption selection, not independent evidence of better
descriptions or higher classification F1. Generated captions may be generic, wrong, or contain
aliases/implicit class cues. Stored validity and full coverage are not ground truth or no-leak proofs.

The M-TACS architecture, fusion, visual+semantic selector and training/test functions remain
identical to notebook 05 and methods 1-2. Inputs: query image + query caption + one selected context
image + its own caption. Both text gates and the semantic selector remain enabled. CLIP stays frozen;
no captioner training or new classifier architecture is introduced.

Defaults: official DTD split 1, pool seed 42, training seed 42, K=1, 50 epochs, no early stopping.
The pool is 376 training images only; self-selection is excluded and val/test images never enter it.
Select checkpoints by minimum validation loss and keep the same GPU/batch policy, optimizer,
losses and augmentation as the preceding methods. Batch is 32 on >=80 GiB under the original VRAM policy.

Attach `captions_single_grounded_control.csv`, DTD, DINOv3 code/ViT-L weights and local CLIP, then Run All.
Leave `CAPTIONS_INPUT_PATH` blank to discover the exact filename/hash in Inputs, or set its path.
Captions are already generated: no Florence weights, annotation references, generation, fine-tuning
or Internet are required for this training notebook. There are no project-local imports on Kaggle.
Wrong/modified CSVs are rejected rather than silently falling back to FirstValid or Prefix.

Policy remains `dtd_lexical_v2` with class-CLIP scores diagnostic-only. The caption-ranking cosine
and diagnostic text-to-class score are different quantities; neither is the task selector's utility.
Do not enable regeneration or substitute a matched-mask CSV to bypass a guard.

Outputs, text cache and checkpoints are isolated in
`outputs/mtacs_dtd_split1_caption_03_florence_singlegrounded_vitl16/`.
Train from scratch initially and resume only this method's own unfinished run. Never reuse methods
1-2 or historical notebook 05 checkpoints. Final test opens after the 50-epoch budget; no scores are prefilled.
'''}
    changes[1] = code[1].replace("Florence-FirstValid", "Florence-SingleGrounded")
    config = prefix.replace_once(code[2], "CAPTION_METHOD_EXPECTED = " + repr(catalog[1]),
        "CAPTION_METHOD_EXPECTED = " + repr(selected))
    changes[2] = config.replace("DTD_CAPTION_02_FIRST_VALID_SEED", "DTD_CAPTION_03_SINGLE_GROUNDED_SEED").replace(
        "DTD_FIRST_VALID_CAPTIONS_INPUT_PATH", "DTD_SINGLE_GROUNDED_CAPTIONS_INPUT_PATH").replace(
        "discover exact FirstValid filename", "discover exact SingleGrounded filename")
    changes[4] = code[4].replace("DTD_CAPTION_02_FIRST_VALID_OUTPUT", "DTD_CAPTION_03_SINGLE_GROUNDED_OUTPUT").replace(
        "caption_02_florence_firstvalid_vitl16", "caption_03_florence_singlegrounded_vitl16").replace(
        "DTD_CAPTION_02_FIRST_VALID_EPOCHS", "DTD_CAPTION_03_SINGLE_GROUNDED_EPOCHS")
    changes[7] = "## Load the exact Florence-SingleGrounded CSV; audit both roles and cache CLIP vectors\n"
    changes[8] = prefix.replace_once(code[8], "This notebook requires the uploaded Florence-FirstValid file;",
        "This notebook requires the uploaded Florence-SingleGrounded file;")
    changes[8] = prefix.replace_once(changes[8], "Attach the FirstValid CSV and set its exact path.",
        "Attach the SingleGrounded CSV and set its exact path.")
    changes[18] = "## Train Florence-SingleGrounded full_both only: 50 epochs by default\n"
    changes[22] = "## Florence-SingleGrounded metrics, learning curves, gates and caption coverage\n"
    changes[23] = code[23].replace("caption_02_florence_firstvalid", "caption_03_florence_singlegrounded").replace(
        "Florence-FirstValid / full_both", "Florence-SingleGrounded / full_both").replace(
        "precomputed FirstValid CSV", "precomputed SingleGrounded CSV").replace(
        "This is caption method 02 only;", "This is caption method 03 only;")
    helpers = "".join(changes.get(index, code[index]) for index in (8, 9, 11, 12, 14, 15))
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(prefix.SOURCE.read_bytes()).hexdigest()!r}\n"
    changes[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n"
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-caption-singlegrounded-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(prefix.source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(caption_method="single_grounded_control",
        caption_display_name="Florence-SingleGrounded", caption_csv_sha256=selected["sha256"],
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
                           ("single_grounded_control", TARGET)):
        if target.is_file():
            created[method] = target.relative_to(prefix.ROOT).as_posix()
    manifest.update(methods=catalog, created_training_notebooks=created, no_training_executed=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
    print("Bound CSV:", catalog[2]["filename"], "| SHA256:", catalog[2]["sha256"])


if __name__ == "__main__":
    main()
