"""Build a review-first, offline Florence-2 LoRA adaptation notebook for DTD."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import json
from pathlib import Path

from dtd_caption_generation_cells import HELPERS


ROOT = PROJECT_ROOT
TARGET = project_path('florence2-dtd-caption-finetune.ipynb')


def definition(source, name):
    tree = ast.parse(source)
    return ast.get_source_segment(source, next(node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name))


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
        ast.parse(source.strip())
    return result


cleanup = definition((project_path('dtd_caption_screening.py')).read_text(encoding="utf-8"), "clean_dtd_caption")
core = (project_path('dtd_caption_finetune.py')).read_text(encoding="utf-8").replace(
    "from dtd_caption_screening import clean_dtd_caption", cleanup)
worker = core + '\n\nif __name__ == "__main__":\n    caption_worker_main(sys.argv[1], sys.argv[2])\n'
runtime = "\n\n".join(definition(HELPERS, name) for name in (
    "florence_checkpoint", "offline_caption_wheels", "caption_runtime_environment"))
runtime = runtime.replace('prefixes = ("transformers-",', 'prefixes = ("peft-", "transformers-",')
runtime += '''


def lora_runtime_environment(asset_root, runtime_dir):
    env = caption_runtime_environment("legacy", asset_root, runtime_dir)
    target = Path(runtime_dir) / "legacy_packages"
    probe = ("import transformers, peft; assert transformers.__version__.startswith('4.49.'); "
             "assert peft.__version__ == '0.14.0', peft.__version__; "
             "from peft import LoraConfig, PeftModel, get_peft_model; "
             "print('Transformers', transformers.__version__, '| PEFT', peft.__version__)")
    checked = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    if checked.returncode == 0:
        print(checked.stdout.strip())
        return env
    wheels = offline_caption_wheels([Path(asset_root), Path("/kaggle/input")], Path(runtime_dir) / "wheels")
    if not wheels:
        raise FileNotFoundError("Attach the offline peft==0.14.0 wheel, plus compatible Florence wheels. "
                                "Reviewing captions does not require PEFT. No Internet is enabled.")
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--only-binary=:all:",
               "--no-deps", "--upgrade", "--target", str(target), "peft==0.14.0"]
    for folder in wheels:
        command.extend(["--find-links", folder])
    installed = subprocess.run(command, env=env, capture_output=True, text=True)
    if installed.returncode:
        raise RuntimeError("Offline PEFT setup failed: attach peft-0.14.0-py3-none-any.whl in a wheel folder "
                           "or offline_wheels.zip/wheels.zip. Kernel packages were not changed.\\n"
                           + installed.stdout[-2000:] + installed.stderr[-3000:])
    target_prefix = str(target)
    if not env.get("PYTHONPATH", "").split(os.pathsep)[0] == target_prefix:
        env["PYTHONPATH"] = target_prefix + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    checked = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError("PEFT/Florence worker imports still fail; attach the missing compatible offline dependency.\\n"
                           + checked.stderr[-4000:])
    print(checked.stdout.strip())
    return env


def run_caption_worker(action, payload_path, runtime_dir, env):
    log_path = Path(runtime_dir) / f"{action}.log"
    worker_path = Path(runtime_dir) / "finetune_florence_dtd.py"
    worker_path.write_text(CAPTION_FINETUNE_WORKER_SOURCE, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(worker_path), str(payload_path), action], env=env,
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
        raise RuntimeError(f"Caption {action} failed. Inspect {log_path}; completed checkpoints/CSV batches are preserved.")
'''

cells = [
cell("markdown", """
# DTD: Review Captions, Adapt Florence-2, Export New M-TACS Captions

This is a **new captioner experiment**, not another M-TACS ablation or an already trained model.
You currently have no approved caption targets. Start with the review template below; training deliberately
stops until real reviewed targets are provided. Old Florence captions are optional **unverified drafts**,
never automatically copied into the target or approved as ground truth.

The ready-to-run second stage uses supervised **decoder-only LoRA** on the original Microsoft Florence-2-base.
The vision encoder and all base weights remain frozen. Input is image pixels plus `<MORE_DETAILED_CAPTION>`;
supervision is a human-approved 5-8-word English description of visible texture. No category label, filename,
or class hint is placed in the prompt. A target is a reviewed reference description, not the DTD class label.

Only official DTD **train** images can supply references. Those references are split again into fit/holdout;
holdout token NLL selects the adapter. The official DTD val/test images are never used to optimize the captioner.
The frozen adapter later generates captions for all three splits. Do not select captioner settings using test F1.

Run cells in order, complete the review, then continue at the reference-validation cell. The notebook is
self-contained on Kaggle: no project Python files need to be attached. It never calls an external teacher/API
or downloads weights. Captioner training defaults to **5 epochs**, independently of the existing **50-epoch M-TACS** runs.
Lower NLL or more readable captions does not prove better downstream classification; that requires a new controlled run.
"""),
cell("code", '''
# Review needs no Transformers/PEFT installation and no GPU.
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

missing_review_packages = [name for name in ("PIL", "IPython") if find_spec(name) is None]
if missing_review_packages:
    raise RuntimeError("Missing image-review packages: " + ", ".join(missing_review_packages))
print("Offline review stage ready. Tokenizer parallelism is disabled; training/export loaders use workers=0.")
'''),
cell("code", '''
DTD_ROOT = Path(os.getenv("DTD_ROOT", "/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd"))
DTD_SPLIT = int(os.getenv("DTD_SPLIT", "1"))
PROJECT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUTPUT_DIR = Path(os.getenv("DTD_CAPTION_FINETUNE_OUTPUT", str(
    PROJECT_DIR / "outputs" / f"florence2_dtd_split{DTD_SPLIT}_caption_lora_v1")))
REVIEW_CSV = OUTPUT_DIR / "review" / "caption_references.csv"
REFERENCE_INPUT_CSV = os.getenv("DTD_CAPTION_REFERENCES", "")  # Optional previously reviewed CSV to import once.
DRAFT_CAPTIONS_CSV = os.getenv("DTD_DRAFT_CAPTIONS_CSV", "")  # Optional OLD captions.csv, for unverified drafts only.
REFERENCE_SAMPLE_PER_CLASS = 4  # 188 images by default; increase before creating the template for wider coverage.
MIN_APPROVED_REFERENCES = 50  # Safety floor, NOT a claim that 50 examples are sufficient for useful adaptation.
REFERENCE_VAL_FRACTION = 0.20
REVIEW_SEED = 42

FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF",
    "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
OFFLINE_ASSETS_ROOT = os.getenv("DTD_OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")
CLIP_MODEL_LOCAL_PATH = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
RUN_CLIP_AUDIT = True  # Diagnostic only, never a 0.30 semantic hard gate.
CAPTION_SETTINGS = {
    "seed": 42, "epochs": 5, "batch_size": 2, "accumulation": 4,
    "learning_rate": 5e-5, "weight_decay": 0.01,
    "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
    "max_target_tokens": 48, "max_new_tokens": 48, "export_batch_size": 16,
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Review template:", REVIEW_CSV)
print("New artifacts only:", OUTPUT_DIR)
'''),
cell("code", core),
cell("markdown", """
## 1. Create the Review Template

The default is 4 train images per class, 188 in total. Sampling uses classes only to cover the dataset;
the blind image viewer below does not show the class or filename. `target_caption` starts empty.
Existing template edits are preserved when this cell is rerun. Keep the output folder when leaving Kaggle.

Set `DTD_DRAFT_CAPTIONS_CSV` only to show an old generated caption as an unverified suggestion.
You can leave it empty and describe images directly. A teacher may suggest text from pixels alone,
but record its model/version and have a human inspect each accepted answer. No teacher is called here.
"""),
cell("code", '''
caption_records, dtd_class_names = load_caption_manifest(DTD_ROOT, DTD_SPLIT)
if REFERENCE_INPUT_CSV and not REVIEW_CSV.exists():
    source = Path(REFERENCE_INPUT_CSV)
    if not source.is_file():
        raise FileNotFoundError(f"Reviewed reference CSV not found: {source}")
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, REVIEW_CSV)
    atomic_json(REVIEW_CSV.parent / "import_source.json", {"source": str(source), "sha256": file_sha256(source)})
create_reference_template(caption_records, REVIEW_CSV, REFERENCE_SAMPLE_PER_CLASS, REVIEW_SEED, DRAFT_CAPTIONS_CSV)
review_rows = read_csv_rows(REVIEW_CSV)
for role in ("train", "val", "test"):
    print(role, sum(row["split"] == role for row in caption_records))
print("Review rows:", len(review_rows), "| approved:", sum(row.get("review_status") == "approved" for row in review_rows))
print("Review CSV:", REVIEW_CSV)

review_guide = """CAPTION REVIEW PROTOCOL v1
- Inspect only the displayed training-image pixels; do not infer a target from its category/filename.
- Write 5-8 English words describing visible material, color, local shapes, repetition or surface structure.
- Avoid boilerplate ('The image is', 'a close-up of'), unsupported objects and invisible properties.
- Describe what is visible, not a guess at the DTD class. Do not look up a label to construct the target.
- Targets need not match a single exact wording. Consistency and visual truth matter more than verbosity.
- Reject uncertain drafts: keep review_status=pending and explain uncertainty in notes.
- Approval requires target_caption, caption_source (human/human_corrected/teacher), reviewer_id,
  review_status=approved, and teacher_model when a teacher proposed the text.
- A teacher proposal is not ground truth until a human has inspected and accepted it.
- Direct class-word overlap is reported later; reference correctness and leakage screening are distinct.
- Do not use validation/test images or test classification scores to revise reference captions.
"""
(REVIEW_CSV.parent / "REVIEW_GUIDE.txt").write_text(review_guide, encoding="utf-8")
teacher_prompt = """Optional image-only teacher prompt (not executed by this notebook):
Describe only the visible texture in this image in 5-8 English words. Focus on material, color,
local shapes, spacing, repetition or surface structure. Do not name a texture category, infer
a dataset label, mention filenames or use generic image/close-up boilerplate. Do not invent details.
If the pixels do not support a confident description, return REVIEW_NEEDED instead.
Human review is required before this text may become a training target.
"""
(REVIEW_CSV.parent / "OPTIONAL_TEACHER_PROMPT.txt").write_text(teacher_prompt, encoding="utf-8")
print(review_guide)
'''),
cell("code", '''
from IPython.display import display
from PIL import Image, ImageOps

def show_caption_review(index=0):
    rows = read_csv_rows(REVIEW_CSV)
    if not 0 <= index < len(rows):
        raise IndexError(f"Review index must be in 0..{len(rows)-1}")
    row = rows[index]
    record = next(record for record in caption_records if record["image_id"] == row["image_id"])
    if record["split"] != "train":
        raise ValueError("The review viewer accepts official training images only")
    with Image.open(record["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((700, 500))
    display(image)
    print("Review index:", index, "| blind review_id:", row["review_id"], "| status:", row["review_status"])
    print("UNVERIFIED draft (optional):", row["draft_caption_unverified"] or "[none]")
    print("Your target:", row["target_caption"] or "[empty]")
    print("Reviewer:", row["reviewer_id"] or "[not reviewed]")
    return row["review_id"]

current_review_id = show_caption_review(0)
'''),
cell("markdown", """
Inspect another image with `current_review_id = show_caption_review(1)` (change the index).
After looking at the image, fill and run the following cell to save that review. The default does nothing;
there are no prefilled reference answers. Repeat for the template, then continue to section 2.
You may also edit the CSV fields directly, but the training guard checks approval/provenance/word count.

The blind ID prevents the viewer from displaying labels, but the CSV `image_id` contains the category folder
for joining records. Avoid reading it while writing targets. Review status is a human declaration;
software cannot verify visual truth or certify the absence of all semantic label clues.
"""),
cell("code", '''
REVIEWED_TARGET = ""  # Fill with your description of the image displayed above, 5-8 English words.
REVIEWER_ID = ""  # Your stable reviewer identifier, not a dataset label.
CAPTION_SOURCE = "human_corrected"  # human / human_corrected / teacher
TEACHER_MODEL = ""  # Required only for teacher proposals that you have personally reviewed.
if REVIEWED_TARGET.strip():
    save_caption_review(REVIEW_CSV, current_review_id, REVIEWED_TARGET, REVIEWER_ID, CAPTION_SOURCE, TEACHER_MODEL)
    print("Saved approved review:", current_review_id)
else:
    print("Nothing approved. Inspect an image and fill REVIEWED_TARGET and REVIEWER_ID before saving.")
'''),
cell("markdown", """
## 2. Validate References Before Any Fine-Tuning

This is the intentional stopping point on the first run. Review all 188 default examples for balanced
coverage if feasible. The minimum of 50 is only a guard; fewer or narrow-quality references may hurt the model.
Both fit and caption holdout are subsets of **official train**, disjoint by image ID. Official val/test remain untouched.
Do not change reference captions after examining holdout results repeatedly without creating a new protocol.
"""),
cell("code", '''
approved_pairs = approved_reference_pairs(REVIEW_CSV, caption_records, minimum=MIN_APPROVED_REFERENCES)
fit_pairs, holdout_pairs = split_reference_pairs(approved_pairs, REFERENCE_VAL_FRACTION, REVIEW_SEED)
print("Approved train-only references:", len(approved_pairs), "| fit:", len(fit_pairs), "| caption holdout:", len(holdout_pairs))
atomic_json(OUTPUT_DIR / "review" / "reference_split.json", {"fit": fit_pairs, "holdout": holdout_pairs})
lexical_overlap = sum(any(" " + normalized_caption(name) + " " in " " + normalized_caption(row["target_caption"]) + " "
                          for name in dtd_class_names) for row in approved_pairs)
print("References with direct DTD class-word overlap:", lexical_overlap,
      "(diagnostic; do not rewrite visually true references from class labels)")
'''),
cell("markdown", """
## 3. Offline Training Runtime and Locked Protocol

Requires a GPU, the **legacy Microsoft Florence-2-base** folder already attached at
`/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`, and compatible offline wheels.
The child process uses **Transformers 4.49.x + PEFT 0.14.0**; your kernel's Transformers 5.x is unchanged.
The old notebook input may not include PEFT: attach `peft-0.14.0-py3-none-any.whl` separately when needed.
Use compatible Python/platform wheels for Transformers, tokenizers, huggingface-hub and any missing dependency.
Wheel lookup is local only; this notebook never enables Internet or installs packages into the kernel environment.

Trainable parameters are strictly decoder attention `q_proj`/`v_proj` LoRA adapters, including decoder cross-attention.
References, image hashes, source code, model assets, settings and split are locked. Changed references/settings require
a **new output directory**. `last.pt` resumes the last completed epoch with optimizer/scheduler/scaler/RNG state;
an interrupted partial epoch is replayed. `best.pt` is selected only by caption holdout token NLL.
`best_adapter/` contains LoRA weights, not a standalone full Florence model.
"""),
cell("code", runtime + "\n\nCAPTION_FINETUNE_WORKER_SOURCE = " + repr(worker)),
cell("code", '''
# The reference guard is repeated so this cell cannot bypass manual review after a kernel restart.
approved_pairs = approved_reference_pairs(REVIEW_CSV, caption_records, minimum=MIN_APPROVED_REFERENCES)
fit_pairs, holdout_pairs = split_reference_pairs(approved_pairs, REFERENCE_VAL_FRACTION, REVIEW_SEED)
model_dir, backend = florence_checkpoint(FLORENCE_MODEL_REF, [Path(OFFLINE_ASSETS_ROOT), Path("/kaggle/input")])
if backend != "legacy":
    raise ValueError("This fine-tuning notebook expects the attached legacy Microsoft Florence checkpoint, not native community weights")
print("Hashing the base checkpoint and reviewed images to lock the experiment...")
protocol = {
    "schema": "dtd_caption_review_lora_v1", "dtd_split": DTD_SPLIT, "seed_reference_split": REVIEW_SEED,
    "reference_val_fraction": REFERENCE_VAL_FRACTION,
    "settings": CAPTION_SETTINGS, "fit": fit_pairs, "holdout": holdout_pairs,
    "base_assets": checkpoint_asset_identity(model_dir), "prompt": TASK,
    "source_sha256": object_hash(CAPTION_FINETUNE_WORKER_SOURCE),
    "reference_policy": "human_reviewed_train_only_5_to_8_words", "export_policy": "dtd_lexical_v2",
}
protocol_id = lock_caption_protocol(OUTPUT_DIR, protocol)
payload = {"model": str(model_dir), "output": str(OUTPUT_DIR.resolve()), "protocol": protocol,
    "settings": CAPTION_SETTINGS, "fit": fit_pairs, "holdout": holdout_pairs,
    "export_records": caption_records, "class_names": dtd_class_names,
    "clip": CLIP_MODEL_LOCAL_PATH, "run_clip_audit": RUN_CLIP_AUDIT}
validate_caption_payload(payload)
RUNTIME_DIR = OUTPUT_DIR / "caption_runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
PAYLOAD_PATH = RUNTIME_DIR / "request.json"
atomic_json(PAYLOAD_PATH, payload)
worker_env = lora_runtime_environment(OFFLINE_ASSETS_ROOT, RUNTIME_DIR)
print("Locked caption experiment:", protocol_id)
'''),
cell("code", '''
run_caption_worker("train", PAYLOAD_PATH, RUNTIME_DIR, worker_env)
training_summary = json.loads((OUTPUT_DIR / "training_complete.json").read_text(encoding="utf-8"))
display(training_summary)
'''),
cell("code", '''
import pandas as pd
history = pd.read_csv(OUTPUT_DIR / "history.csv")
display(history)
comparison = pd.read_csv(OUTPUT_DIR / "holdout_before_after.csv").fillna("")
display(comparison[["reference", "base_short_caption", "adapted_short_caption", "human_preference"]].head(12))
print("Inspect the holdout images and fill human_preference/review_notes in holdout_before_after.csv.")
print("Lower token NLL is not proof of visual correctness, leakage absence, or improved M-TACS F1.")
if training_summary["best_holdout_token_nll"] >= training_summary["baseline_holdout_token_nll"]:
    print("Warning: adapted holdout NLL does not beat the frozen base; inspect data/overfitting before using it downstream.")
'''),
cell("markdown", """
## 4. Export a New, Resumable `captions.csv`

Only after training completes: apply the fixed best adapter to all 5,640 images. The generation prompt uses pixels
and the same task token only. Dataset class names appear only in the **post-generation audit**, not caption generation.
The CSV uses **dtd_lexical_v2**: remove boilerplate, keep 5-8 words, reject direct class-term hits. CLIP maximum cosine
at 0.30 is **diagnostic only**; it never gates caption validity. This is not a certified semantic-leakage detector.
When `RUN_CLIP_AUDIT=False`, score fields remain empty with `semantic_audit_status=not_run`, not fabricated zeros.

Every successful batch is flushed; rerunning resumes missing image IDs. Model/adapter/policy/asset fingerprints
prevent mixing different captioners into a single CSV. Existing ablation captions/results are not overwritten.
Do not change decoding/policy based on test metrics. Adapter outputs are machine captions, not human ground truth.
"""),
cell("code", '''
run_caption_worker("export", PAYLOAD_PATH, RUNTIME_DIR, worker_env)
NEW_CAPTIONS_CSV = OUTPUT_DIR / "exports" / "captions.csv"
export_summary = json.loads((OUTPUT_DIR / "exports" / "export_summary.json").read_text(encoding="utf-8"))
display(export_summary)
exported = pd.read_csv(NEW_CAPTIONS_CSV)
display(exported.groupby("split").agg(images=("image_id", "size"), valid_captions=("caption_valid", "sum")))
print("New captions CSV:", NEW_CAPTIONS_CSV)
print("Next: attach this CSV to a separate M-TACS run and set DTD_CAPTIONS_INPUT_PATH to its attached path.")
print("Keep the same DTD split, seed, candidate pool, 50 epochs and dtd_lexical_v2 policy as the old comparison.")
print("Use a new M-TACS output/cache directory. Do not resume old classifier checkpoints or reuse old text embeddings.")
'''),
cell("markdown", """
## Interpretation and Sources

Use the before/after holdout table to check visual faithfulness and generic/repeated descriptions, not just NLL.
Then compare old vs new captions with the **same M-TACS configuration** and more than one seed when resources allow.
The existing ablation set must remain a frozen baseline. A captioner can become more fluent while classification worsens.

This notebook supplies a training pipeline and an empty review task; no GPU training or improvement claim is bundled.
Starting with 188 reviewed images is a small pilot, not a guaranteed adaptation recipe. An optional VLM teacher would
be a separate annotation step with image-only prompts and human acceptance; no teacher is silently invoked here.

Primary implementation references: [Florence fine-tuning example](https://huggingface.co/blog/finetune-florence2),
[Microsoft Florence model code](https://huggingface.co/microsoft/Florence-2-base/blob/main/modeling_florence2.py),
[PEFT 0.14 LoRA guide](https://huggingface.co/docs/peft/v0.14.0/en/quicktour).
"""),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
for index, item in enumerate(cells):
    item["id"] = f"dtd-caption-lora-{index:02d}"
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created {TARGET} ({len(cells)} cells)")
