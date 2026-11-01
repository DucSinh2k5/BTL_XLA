"""Build an offline DTD caption experiment with frozen image-grounded selection."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import json
from pathlib import Path

from dtd_caption_generation_cells import HELPERS


ROOT = PROJECT_ROOT
TARGET = project_path('florence2-dtd-caption-grounded-rerank.ipynb')


def get_function(source, name):
    return ast.get_source_segment(source, next(node for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name))


def cell(kind, source):
    source = source.strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(outputs=[], execution_count=None)
    return result


utilities = (project_path('dtd_caption_finetune.py')).read_text(encoding="utf-8")
utility_names = ["atomic_csv", "atomic_json", "checkpoint_asset_identity", "file_sha256", "load_caption_manifest", "object_hash", "read_csv_rows"]
definitions = "\n\n".join(get_function(utilities, name) for name in utility_names)
cleanup = get_function((project_path('dtd_caption_screening.py')).read_text(encoding="utf-8"), "clean_dtd_caption")
core = (project_path('dtd_grounded_caption.py')).read_text(encoding="utf-8").replace(
    "from dtd_caption_screening import clean_dtd_caption", cleanup).replace(
    "from dtd_caption_finetune import " + ", ".join(utility_names), definitions)
assert "from dtd_caption_" not in core
worker = core + '\n\nif __name__ == "__main__":\n    worker_main(sys.argv[1])\n'
runtime = "\n\n".join(get_function(HELPERS, name) for name in (
    "florence_checkpoint", "offline_caption_wheels", "caption_runtime_environment"))
runtime += '''


def run_grounded_worker(worker_source, payload_path, runtime_dir, env):
    worker_path = Path(runtime_dir) / "generate_grounded_dtd.py"
    worker_path.write_text(worker_source, encoding="utf-8")
    log_path = Path(runtime_dir) / "generation.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(worker_path), str(payload_path)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Caption worker failed; inspect {log_path}. Completed per-image caches are preserved for resume.")
'''

cells = [
cell("markdown", """
# DTD Caption Experiment C: Frozen Florence + Image-Grounded Short Spans

This notebook creates a **third caption approach** for comparison with the original Florence CSV and the
decoder-LoRA CSV. It does **not** fine-tune Florence, CLIP or M-TACS. No annotation targets or PEFT wheels are needed.
It uses the same local Microsoft Florence-2-base and local full CLIP ViT-B/32 checkpoint already attached.

For each image, Florence produces its original greedy description plus three beam-search descriptions.
The notebook extracts existing 5-8-word spans within sentences, avoids some incomplete openings/endings,
and chooses a span by **CLIP image-to-text cosine similarity to that image**. It never creates a generic replacement
caption to hide a failure. If no candidate is eligible, it explicitly falls back to the original prefix and records it.
CLIP ranking can prefer generic/object-oriented or misleading spans; it is a hypothesis, not guaranteed improvement.

**Generation inputs are pixels plus the fixed Florence task token only.** No filename, category, correct class,
reference caption or test classification result is supplied to generation. The global class vocabulary is used
only after generation for the same `dtd_lexical_v2` eligibility checks and diagnostic text-to-class scores.
Text-to-class scores do not select the winning caption. Alias checks are diagnostic, not a stricter hidden filter.

The notebook uses official DTD split 1 (1,880 train / 1,880 val / 1,880 test). There is no classifier training here.
For downstream comparison, use identical M-TACS architecture, split, pool, seed, batch size and 50-epoch budget.
Do not select the caption method or change its settings based on repeated test F1.
"""),
cell("code", '''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import shutil
import subprocess
import sys
from pathlib import Path
from importlib.util import find_spec

required = [name for name in ("torch", "numpy", "PIL", "pandas") if find_spec(name) is None]
if required:
    raise RuntimeError("Missing offline packages: " + ", ".join(required))
print("Offline preflight passed. Transformers is loaded only in the isolated legacy worker; workers=0.")
'''),
cell("code", '''
DTD_ROOT = Path(os.getenv("DTD_ROOT", "/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd"))
DTD_SPLIT = int(os.getenv("DTD_SPLIT", "1"))
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("DTD_GROUNDED_CAPTION_OUTPUT", str(
    PROJECT_DIR / "outputs" / f"florence2_dtd_split{DTD_SPLIT}_grounded_caption_v1")))
FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
OFFLINE_ASSETS_ROOT = os.getenv("DTD_OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
SETTINGS = {"batch_size": 8, "num_beams": 3, "num_return_sequences": 3,
    "max_new_tokens": 48, "max_spans_per_hypothesis": 48, "clip_text_batch_size": 256}
SEED = 42
MAX_NEW_IMAGES = 0  # Smoke test: set 32; then return to 0 and rerun to finish all missing images.

# Optional exact attached paths. Never auto-select another file called captions.csv.
OLD_CAPTIONS_CSV = os.getenv("DTD_ORIGINAL_CAPTIONS_CSV", "")
LORA_CAPTIONS_CSV = os.getenv("DTD_LORA_CAPTIONS_CSV", "")
WRITE_MATCHED_MASK_CSVS = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("New experiment output:", OUTPUT_DIR)
'''),
cell("markdown", """
## Four Outputs to Explain Differences

| CSV | Florence proposals | Short-text choice | Purpose |
|---|---|---|---|
| `controls/prefix_control/captions.csv` | One greedy description | Original cleanup / first 8 words | Fresh procedural baseline |
| `controls/first_valid_control/captions.csv` | Same greedy description | First eligible span, no CLIP ranking | Effect of short-span extraction |
| `controls/single_grounded_control/captions.csv` | Same greedy description | Image-CLIP ranking of the same span pool | Effect of image-grounded ranking |
| `exports/captions.csv` | Greedy + 3 beam hypotheses | Same extraction + image-CLIP ranking | Main new caption approach |

First-valid vs single-grounded uses the same candidates and availability mask. Single-grounded vs main changes
the proposal pool. These are diagnostics you may run later; initially compare the main CSV to the two existing CSVs.
The fresh prefix control is not claimed byte-identical to a historical CSV: attached assets/runtime can differ.
Attach the actual old CSVs via the configuration above for an exact file-level comparison.
Three beam hypotheses are **caption alternatives for one image**, not three selected context images.
Downstream M-TACS still uses **K=1** in the matched experiment.

The outer policy stays `dtd_lexical_v2` (5-8 words, canonical class words screened, CLIP class score diagnostic only).
Selection can change validity/coverage by finding other eligible spans. Optional matched-mask CSVs later hold
availability fixed so changed coverage is not confused with changed text content. None of this certifies no semantic leakage.
"""),
cell("code", core),
cell("code", '''
caption_records, class_names = load_caption_manifest(DTD_ROOT, DTD_SPLIT)
validate_settings(SETTINGS)
for split in ("train", "val", "test"):
    print(split, sum(row["split"] == split for row in caption_records))
print("Total:", len(caption_records), "| classes:", len(class_names))
'''),
cell("markdown", """
## Offline Runtime and Assets

The attached Microsoft checkpoint is **legacy Florence**: use the same isolated Transformers **4.49.x** worker
as the earlier notebooks. A kernel with Transformers 5.x is fine; it is not downgraded or restarted.
Missing legacy packages are installed into this experiment's worker folder using attached offline wheels only.
There are no Internet/API requests, PEFT imports or DataLoader subprocesses.

This approach requires **both CLIP towers**, not a text-only checkpoint. Missing image/text weights are fatal,
not silently initialized. If CLIP's `preprocessor_config.json` is absent, the worker permits only standard
224px/patch32 CLIP and records use of Transformers 4.49's standard CLIPImageProcessor defaults. Otherwise it
uses the attached preprocessing config. Inspect `clip_preprocessing_used.json` when comparing asset provenance.

The protocol hashes images, model assets, code, settings and runtime versions. Resume keeps per-image atomic
transactions, so an interrupted run does not need to regenerate completed images. Changed protocols must use
a new output directory. Batch OOM halves the effective generation batch; other errors stop visibly.
"""),
cell("code", runtime + "\n\nGROUNDED_WORKER_SOURCE = " + repr(worker)),
cell("code", '''
model_dir, backend = florence_checkpoint(FLORENCE_MODEL_REF, [Path(OFFLINE_ASSETS_ROOT), Path("/kaggle/input")])
if backend != "legacy":
    raise ValueError("Use the attached legacy Microsoft Florence-2-base; this is not the LoRA adapter or native-community experiment")
clip_dir = Path(CLIP_MODEL_LOCAL_PATH).expanduser()
if not (clip_dir / "config.json").is_file():
    raise FileNotFoundError(f"Set CLIP_MODEL_LOCAL_PATH to the complete local CLIP ViT-B/32 folder: {clip_dir}")
if json.loads((clip_dir / "config.json").read_text(encoding="utf-8")).get("model_type") != "clip":
    raise ValueError("The supplied folder is not a full CLIP checkpoint")
RUNTIME_DIR = OUTPUT_DIR / "caption_runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
worker_env = caption_runtime_environment("legacy", OFFLINE_ASSETS_ROOT, RUNTIME_DIR)
payload = {"output": str(OUTPUT_DIR.resolve()), "records": caption_records, "dtd_split": DTD_SPLIT,
    "class_names": class_names, "florence": str(model_dir), "clip": str(clip_dir.resolve()),
    "settings": SETTINGS, "seed": SEED, "max_new_images": MAX_NEW_IMAGES,
    "code_sha256": object_hash(GROUNDED_WORKER_SOURCE)}
payload["settings"] = dict(SETTINGS, seed=SEED)
PAYLOAD_PATH = RUNTIME_DIR / "request.json"
atomic_json(PAYLOAD_PATH, payload)
run_grounded_worker(GROUNDED_WORKER_SOURCE, PAYLOAD_PATH, RUNTIME_DIR, worker_env)
generation_status = json.loads((OUTPUT_DIR / "generation_status.json").read_text(encoding="utf-8"))
print("Complete:", generation_status["complete"], "| cached images:", generation_status["rows"])
if not generation_status["complete"]:
    print("SMOKE TEST ONLY: preview CSVs are incomplete. Set MAX_NEW_IMAGES=0 and rerun this cell to resume.")
'''),
cell("markdown", """
## Inspect Caption Content and Coverage

`raw_caption` contains the actual selected short text, because M-TACS reads raw_caption before final_caption.
The unabridged Florence output is preserved in `source_long_caption`; every chosen span has its hypothesis,
sentence and word interval in the cleaned source sentence. `candidate_audit.jsonl` retains proposals and scores.
An interval of -1 denotes the original-prefix control/fallback rather than an extracted sentence span.

`image_text_similarity` ranks proposals against their image. `semantic_leakage_score` compares the selected text
to class prompts for audit only. They are distinct scores. CLIP alignment is optimized by this method, so a higher
score is **not independent evidence** of better captions or increased M-TACS F1. Check visible details yourself.
"""),
cell("code", '''
import pandas as pd
from IPython.display import display

report_dir = OUTPUT_DIR if generation_status["complete"] else OUTPUT_DIR / "preview_incomplete"
summary = pd.read_csv(report_dir / "caption_generation_summary.csv")
display(summary)
main_csv = Path(generation_status["csv_paths"]["multibeam_grounded"])
main_rows = read_csv_rows(main_csv)
display(pd.DataFrame(main_rows)[["split", "source_long_caption", "final_caption", "caption_valid",
                                "image_text_similarity", "alias_hit_diagnostic", "selection_fallback"]].head(10))
print("Main CSV:", main_csv)
print("These are generated captions, not reference annotations or human ground truth.")
'''),
cell("code", '''
def show_train_caption(index=0):
    train_records = [row for row in caption_records if row["split"] == "train"]
    if not 0 <= index < len(train_records):
        raise IndexError("Choose an official training-image index")
    record = train_records[index]
    cached = read_cached(OUTPUT_DIR, record, generation_status["identity"])
    if cached is None:
        print("This image is not generated yet; finish/resume generation first.")
        return
    image = load_rgb(record["image_path"])
    image.thumbnail((650, 450))
    display(image)
    print("Training review index:", index)
    for method in METHODS:
        row = cached["rows"][method]
        print(method, "|", row["final_caption"], "| valid:", row["caption_valid"])

show_train_caption(0)  # Change the index for image-grounded review; category labels are not displayed.
'''),
cell("markdown", """
## Compare to the Two Previous CSVs

Set the exact paths in `OLD_CAPTIONS_CSV` and `LORA_CAPTIONS_CSV`, then rerun this section.
Blank paths are skipped. Wrong, duplicate or incomplete DTD files stop with an error. Originals are read-only.
Reports compare the actual **dtd_lexical_v2 model inputs**, including legacy semantic-only reject recovery;
do not confuse a stored caption_valid column in an older schema with its effective M-TACS mask.

The matched-mask CSVs keep every image but disable text on the **same intersection** for all compared methods.
Original audit fields are retained separately. They are for screened/full_both experiments, not unscreened/redacted
variants which intentionally use different masks. Use separate M-TACS runs for natural masks vs matched masks.
Choose which comparisons to include before training; adding a CSV changes the intersection and requires a new
comparison output directory. High coverage, lexical cleanliness and CLIP similarity do not prove visual truth.
"""),
cell("code", '''
if not generation_status["complete"]:
    print("Skipping complete-dataset comparison: finish the smoke test/resume first.")
else:
    comparisons = {}
    if OLD_CAPTIONS_CSV:
        comparisons["original_old"] = OLD_CAPTIONS_CSV
    if LORA_CAPTIONS_CSV:
        comparisons["decoder_lora"] = LORA_CAPTIONS_CSV
    comparisons.update({method: path for method, path in generation_status["csv_paths"].items()})
    # A path-set-specific folder avoids accidentally mixing intersections from different comparison sets.
    comparison_tag = object_hash({key: file_sha256(path) for key, path in comparisons.items()})[:12]
    COMPARISON_DIR = OUTPUT_DIR / "comparisons" / comparison_tag
    comparison = compare_caption_files(comparisons, caption_records, class_names, COMPARISON_DIR, WRITE_MATCHED_MASK_CSVS)
    display(pd.DataFrame(comparison))
    print("Comparison reports:", COMPARISON_DIR)
    if WRITE_MATCHED_MASK_CSVS:
        print("Matched-mask CSVs:", COMPARISON_DIR / "matched_masks")
'''),
cell("markdown", """
## Downstream M-TACS Evaluation

Attach the completed main `exports/captions.csv` to **notebook 05 full_both**, train a fresh classifier and text cache,
and keep DTD split 1, seed/pool seed 42, batch 32, K=1 and 50 epochs. Set `GENERATE_CAPTIONS_IF_MISSING=False`;
do not let a classifier notebook regenerate missing rows using the old captioner. Use a new output directory.
Leave `CAPTION_INPUT_POLICY="dtd_lexical_v2"`. Select the classifier checkpoint by validation loss as before.

Historical single-seed test macro-F1 references: original full_both 0.809124; LoRA-caption full_both 0.806707;
visual_only 0.848168. These are observations from prior runs, not goals used to tune this caption selector.
The new method has **no measured F1 yet**. Higher F1 than the old full_both is an improvement for that run;
it is not automatically a robust gain or a win over visual_only. Confirm with paired seeds and locked evaluation.

To explain a difference, compare first-valid vs single-grounded, single-grounded vs main, caption validity,
alias hits, fallbacks, selected contexts and query/candidate gates. The common-mask CSVs help separate text-content
effects from changed availability; they do not hold the selected context fixed when the semantic selector is active.
No automatic test-score search or claim of guaranteed improvement is made.

Example additions to notebook 05's configuration cell (replace the attached CSV path):

```python
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_ATTACHED_DATASET/captions.csv"
CAPTION_INPUT_POLICY = "dtd_lexical_v2"
GENERATE_CAPTIONS_IF_MISSING = False
os.environ["DTD_FULL_BOTH_OUTPUT"] = "/kaggle/working/outputs/dtd_full_both_grounded_seed42"
```

Sources: [Microsoft Florence-2 model](https://huggingface.co/microsoft/Florence-2-base),
[Transformers 4.49 generation parameters](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/text_generation),
[CLIP image/text feature APIs](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/clip).
"""),
]
for index, item in enumerate(cells):
    item["id"] = f"dtd-grounded-caption-{index:02d}"
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {TARGET} ({len(cells)} cells)")
