"""Bind the same full_both classifier to Florence-FirstValid, caption method 02."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import hashlib
import json
from pathlib import Path

import build_dtd_caption_method01_prefix_notebook as prefix


FOLDER = prefix.FOLDER
TARGET = project_path('all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-02-florence-firstvalid.ipynb')


def build_notebook(catalog=None):
    catalog = prefix.inspect_caption_files(FOLDER) if catalog is None else catalog
    selected = catalog[1]
    if selected["generation_method"] != "first_valid_control" or selected["number"] != 2:
        raise ValueError("Method 02 requires the uploaded first_valid_control CSV")
    notebook = prefix.build_notebook(catalog)
    code = {index: prefix.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: f'''# DTD Caption Ablation 02: Florence-FirstValid + M-TACS Full-Both

Method 2 of the four caption-generation controls: **Florence-FirstValid** (`first_valid_control`).
For each image, the caption generator chose the first eligible 5-8-word span from its original
greedy Florence description. It did NOT rank spans by CLIP or add beam hypotheses for this control.
The uploaded CSV is already complete: 5,640 images, 1,880 per official split, with
{selected["stored_valid_total"]:,} stored-valid captions. Full coverage does not certify visual correctness or absence of aliases.

Compared with method 1 Florence-Prefix, the short-text extraction changes, and caption availability
can also change. Use coverage reports when interpreting downstream differences. This is a
natural-mask experiment, not an intervention holding every caption mask or selected context fixed.
The first-valid vs single-grounded control later uses the same span pool and availability.

The M-TACS classifier is unchanged from notebook 05 and method 1: query image + query caption + one
selected context image + its caption, both text fusion gates enabled and a visual+semantic selector.
Although caption selection did not use CLIP ranking, the classifier still uses its frozen CLIP
text encoder and semantic selector exactly as before. Losses, transforms, optimizer and train/test
functions are unchanged. No classifier or captioner architecture is added in this experiment.

Default: official DTD split 1, pool seed 42, training seed 42, K=1 and 50 epochs, no early stopping.
Candidate pool contains 376 training images; self-selection is excluded and no val/test images
enter the pool. Select the checkpoint by minimum validation loss, not test performance.
Batch retains notebook 05's VRAM policy (32 on >=80 GiB); keep the same GPU/batch for all four methods.

Attach `captions_first_valid_control.csv`, DTD, DINOv3 code/ViT-L weights and local CLIP, then Run All.
Leave `CAPTIONS_INPUT_PATH` blank to find the exact filename/hash in Inputs or set its attached path.
No Florence weights, reference captions, caption generation, fine-tuning or Internet are needed here.
The notebook has no project-local imports. It rejects wrong/changed CSVs instead of falling back to Prefix.

Policy remains `dtd_lexical_v2`; class-CLIP audit scores are diagnostic-only. Do not change policy or
regenerate captions to bypass an input error. This binding is for the uploaded natural-mask CSV,
not a matched-mask export. Synonyms/implicit class cues may remain; a valid caption is not ground truth.

Outputs, text cache and checkpoints are isolated in
`outputs/mtacs_dtd_split1_caption_02_florence_firstvalid_vitl16/`.
Train from scratch initially; resume only this method's own unfinished run. Never load method 1's
or historical notebook 05's classifier checkpoint. Metrics are not prefilled.
'''}
    changes[1] = code[1].replace("Florence-Prefix", "Florence-FirstValid")
    config = prefix.replace_once(code[2], "CAPTION_METHOD_EXPECTED = " + repr(catalog[0]),
        "CAPTION_METHOD_EXPECTED = " + repr(selected))
    config = config.replace("DTD_CAPTION_01_PREFIX_SEED", "DTD_CAPTION_02_FIRST_VALID_SEED").replace(
        "DTD_PREFIX_CAPTIONS_INPUT_PATH", "DTD_FIRST_VALID_CAPTIONS_INPUT_PATH").replace(
        "discover exact Prefix filename", "discover exact FirstValid filename")
    changes[2] = config
    changes[4] = code[4].replace("DTD_CAPTION_01_PREFIX_OUTPUT", "DTD_CAPTION_02_FIRST_VALID_OUTPUT").replace(
        "caption_01_florence_prefix_vitl16", "caption_02_florence_firstvalid_vitl16").replace(
        "DTD_CAPTION_01_PREFIX_EPOCHS", "DTD_CAPTION_02_FIRST_VALID_EPOCHS")
    changes[7] = "## Load the exact Florence-FirstValid CSV; audit both roles and cache CLIP vectors\n"
    changes[8] = prefix.replace_once(code[8], "This notebook requires the uploaded Florence-Prefix file;",
        "This notebook requires the uploaded Florence-FirstValid file;")
    changes[8] = prefix.replace_once(changes[8], "Attach the Prefix CSV and set its exact path.",
        "Attach the FirstValid CSV and set its exact path.")
    changes[18] = "## Train Florence-FirstValid full_both only: 50 epochs by default\n"
    changes[22] = "## Florence-FirstValid metrics, learning curves, gates and caption coverage\n"
    changes[23] = code[23].replace("caption_01_florence_prefix", "caption_02_florence_firstvalid").replace(
        "Florence-Prefix / full_both", "Florence-FirstValid / full_both").replace(
        "precomputed Prefix CSV", "precomputed FirstValid CSV").replace(
        "This is caption method 01 only;", "This is caption method 02 only;")
    helpers = "".join(changes.get(index, code[index]) for index in (8, 9, 11, 12, 14, 15))
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(prefix.SOURCE.read_bytes()).hexdigest()!r}\n"
    changes[6] += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n"
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = f"dtd-caption-firstvalid-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(prefix.source_of(cell))
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["dtd_ablation"].update(caption_method="first_valid_control",
        caption_display_name="Florence-FirstValid", caption_csv_sha256=selected["sha256"],
        captioner_fingerprint=selected["captioner_fingerprint"], caption_generation=False, default_epochs=50)
    return notebook


def main():
    catalog = prefix.inspect_caption_files(FOLDER)
    notebook = build_notebook(catalog)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    path = FOLDER / "caption_methods_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    created = manifest.get("created_training_notebooks", {})
    if prefix.TARGET.is_file():
        created["prefix_control"] = prefix.TARGET.relative_to(prefix.ROOT).as_posix()
    created["first_valid_control"] = TARGET.relative_to(prefix.ROOT).as_posix()
    manifest.update(methods=catalog, created_training_notebooks=created, no_training_executed=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
    print("Bound CSV:", catalog[1]["filename"], "| SHA256:", catalog[1]["sha256"])


if __name__ == "__main__":
    main()
