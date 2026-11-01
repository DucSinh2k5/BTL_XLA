"""Apply the visual web memory repair to the full_both deployment notebook."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

from _project_bootstrap import PROJECT_ROOT
from build_sun397_mtacs_web_notebook import SOURCES, TARGET
from repair_sun397_tacs_web_notebook import repaired_training_source as visual_training_source, set_source


BACKUP = PROJECT_ROOT / "tmp/notebook_backups/sun397_mtacs_web.before_memory_repair.ipynb"
PATCH = PROJECT_ROOT / "src/training/sun397_mtacs_web_training_patch.py"


def repaired_training_source():
    source = SOURCES["train_sun397_mtacs_web.py"].read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    replaced = {"default_config", "validate_config", "train_epoch", "train_run"}
    found, edits = set(), []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in replaced:
            found.add(node.name)
            first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            if node.name in {"default_config", "validate_config"}:
                node.name = "_legacy_" + node.name
                replacement = ast.unparse(node) + "\n"
            else:
                replacement = ""
            edits.append((first, node.end_lineno, replacement))
    if found != replaced:
        raise ValueError("Legacy M-TACS interface changed; re-audit the repair")
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first:last] = [replacement]
    keys = {"tacs_visual_web": "tacs_visual_web.py", "train_sun397_tacs_web": "train_sun397_tacs_web.py",
            "mtacs_multimodal_web": "mtacs_multimodal_web.py", "sun397_caption_web": "sun397_caption_web.py",
            "training": "train_sun397_mtacs_web.py"}
    hashes = {key: hashlib.sha256(SOURCES[name].read_text(encoding="utf-8").encode()).hexdigest() for key, name in keys.items()}
    result = "".join(lines) + "\n\nLEGACY_MTACS_SOURCE_SHA256 = " + repr(hashes) + "\n\n" + PATCH.read_text(encoding="utf-8")
    compile(result, "train_sun397_mtacs_web.py", "exec")
    return result


def build_notebook(original=None):
    if original is None:
        original = json.loads(BACKUP.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(original)
    cells = notebook["cells"]
    if len(cells) < 21 or "TEXT_TABLE" not in "".join(cells[10]["source"]):
        raise ValueError("Not the checked standalone full_both notebook")
    config_code = "".join(cells[4]["source"])
    assignments = {t.id: ast.get_source_segment(config_code, node.value)
                   for node in ast.parse(config_code).body if isinstance(node, ast.Assign)
                   for t in node.targets if isinstance(t, ast.Name)}
    set_source(cells[0], """# SUN397 M-TACS full_both L/16: memory-bounded training and complete web export

This notebook applies the `tacs_web` memory repair to M-TACS while keeping its multimodal architecture and caption policy unchanged. **DINOv3 ViT-L/16, 384 pixels, K=1; query text + selected candidate text enter fusion; both captions also enter the learned visual/semantic selector.** No text branch is removed to gain speed. Florence and CLIP remain frozen and isolated from classifier training.

Logical batch64 is independent from GPU compute batch8 (smaller defaults on smaller GPUs). Gradient checkpointing stays on. The detached reward prepass covers the entire logical batch; selector RNG is replayed for policy gradients; task and policy losses are backpropagated per compute chunk. Selected image, text and validity always use the same context index, and both pixel and text Gumbel paths retain task gradients. The no-context reward baseline keeps the query caption but removes candidate image and caption.

The training pixel bank is copied once to GPU FP32 when free memory permits, with explicit CPU fallback. Values still represent the previous fp16 training pixels; no index-only estimator or backbone freezing is substituted. Candidate feature refresh uses parallel image loading; validation/test/export retain raw fp32 preprocessing. Initial batches report measured images/s and peak allocated VRAM. No fixed speedup or maximum GPU batch is promised without a real GPU measurement.

**Restart the Kaggle kernel before Run All.** The repaired output is separate from the old run. `RESUME_FROM_DIR` can point to previous Kaggle Output: compatible registered captions and CLIP vectors are copied byte-for-byte, including a partial CSV when no checkpoint exists. A checkpoint without its bound caption cache is rejected, not trained with regenerated replacement text. Complete valid caches do not regenerate all captions. Keep local Florence, CLIP, offline wheels and DINO assets attached for preflight and the complete web export.

Compatible last.pt/best.pt checkpoints migrate only after validating source, weights, classes, pool/order, caption/cache/settings hashes and fixed logical training configuration. Model, optimizer, scheduler, scaler, RNG, history and best validation state are retained, originals remain unchanged. Legacy logical batch size is preserved. Changing only COMPUTE_BATCH_SIZE on a repaired resume is allowed. The new stochastic order is not bitwise legacy replay; protocol.json records migration. With no completed checkpoint, training starts epoch1 explicitly.

Caption policy stays `sun397_lexical_clip_audit_v1`: strip boilerplate, require 5-8 words and screen direct SUN class terms; CLIP similarity is audit-only, not proof of leak-free text. Invalid captions retain zero vectors/gates. No test-label prompts, test-based threshold tuning, pool expansion or test-time training are introduced.

The final verified ZIP still contains the trained full_both model, frozen Florence/CLIP assets, pool images/captions, aligned visual/text banks, complete CSV/cache, source, inference adapter, metrics and actual visual attention readouts. Attention is last-block mean-head CLS-to-patch, not text attribution, causal proof, Grad-CAM or an extra trained model. The existing FastAPI frontend/backend are not modified here.

This remains a declared deployment adaptation, not an exact reproduction of the original paper: L/16/384, cached detached keys, validation holdout, one seed and existing augmentation choices are retained.
""")
    code = "".join(cells[2]["source"])
    code = code.replace("import os\n", 'import os\nos.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")\n', 1)
    set_source(cells[2], code)
    set_source(cells[3], """## 2 Inputs, resume source and compute budget

Keep the original SUN397, DINOv3 L/16, Florence, CLIP and offline-wheel Inputs. Correct the exact paths here. No CSV input is required for a fresh run. If continuing an old run, RESUME_FROM_DIR points to its full folder, including shared/, protocol.json, last.pt and best.pt; do not point to a ZIP. The new output has suffix `_repaired_v2` and does not overwrite the original.

LOGICAL_BATCH_SIZE controls reward normalization; COMPUTE_BATCH_SIZE controls GPU activation memory. Start at compute8 on the shown 96GB GPU and keep checkpointing on. Do not turn logical64 into one GPU batch64. You can lower compute batch and restart without changing the registered logical run. A legacy run retains its original logical/effective batch. Keep epoch horizon, seed, learning rates and caption settings fixed on resume.
""")
    paths = "\n".join(name + " = " + assignments[name] for name in (
        "DATASET_ROOT", "TRAIN_PARTITION", "TEST_PARTITION", "DINOV3_REPO_DIR", "DINOV3_WEIGHTS", "PROJECT_DIR"))
    frozen_paths = "\n".join(name + " = " + assignments[name] for name in (
        "FLORENCE_MODEL_DIR", "CLIP_MODEL_DIR", "OFFLINE_ASSETS_ROOT", "CAPTION_BATCH_SIZE", "CAPTION_MAX_NEW_TOKENS"))
    set_source(cells[4], paths + "\nLEGACY_OUTPUT_DIR = " + assignments["OUTPUT_DIR"] + '''
OUTPUT_DIR = Path(os.getenv("SUN397_MTACS_WEB_REPAIRED_OUTPUT", str(LEGACY_OUTPUT_DIR.with_name(LEGACY_OUTPUT_DIR.name + "_repaired_v2"))))
RESUME_FROM_DIR = Path(os.getenv("SUN397_MTACS_WEB_RESUME_FROM", str(LEGACY_OUTPUT_DIR)))
DEVICE = torch.device("cuda")
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
LOGICAL_BATCH_SIZE = 64
EFFECTIVE_BATCH_SIZE = 64
COMPUTE_BATCH_SIZE = int(os.getenv("SUN397_MTACS_WEB_COMPUTE_BATCH", "8" if GPU_GIB >= 80 else "4" if GPU_GIB >= 40 else "2"))
NUM_WORKERS = int(os.getenv("SUN397_MTACS_WEB_WORKERS", str(min(4, os.cpu_count() or 1))))
EPOCHS = int(os.getenv("SUN397_MTACS_WEB_EPOCHS", "100"))
SEED = 42
IMAGE_SIZE = 384
EXECUTION = {"compute_batch_size": COMPUTE_BATCH_SIZE, "pixel_bank_device": "auto",
             "gpu_reserve_gib": 20.0, "profile_batches": 3}
RUN_TRAINING = True
RUN_FINAL_TEST = True
RUN_EXPORT = True
''' + frozen_paths + '''
print("Repaired output:", OUTPUT_DIR)
print("Previous run/cache:", RESUME_FROM_DIR)
print("GPU:", torch.cuda.get_device_name(0), "| GiB:", round(GPU_GIB, 2))
print("Requested logical / compute batch:", LOGICAL_BATCH_SIZE, COMPUTE_BATCH_SIZE)
''')
    payload = {name: path.read_text(encoding="utf-8") for name, path in SOURCES.items()}
    payload["train_sun397_tacs_web.py"] = visual_training_source()
    payload["train_sun397_mtacs_web.py"] = repaired_training_source()
    set_source(cells[6], '''import json
RUNTIME_DIR = OUTPUT_DIR / "embedded_runtime_v2"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_SOURCES = ''' + repr(payload) + '''
for name, source in EMBEDDED_SOURCES.items():
    module_name = Path(name).stem
    if module_name in sys.modules:
        loaded = Path(sys.modules[module_name].__file__).resolve()
        if loaded != (RUNTIME_DIR / name).resolve() or loaded.read_text(encoding="utf-8") != source:
            raise RuntimeError("A different TACS/M-TACS runtime is loaded. Restart the kernel and Run All this notebook.")
    path = RUNTIME_DIR / name
    if path.is_file() and path.read_text(encoding="utf-8") != source:
        raise ValueError("Runtime changed under this repaired output; preserve the run and use another output directory")
    if not path.is_file():
        path.write_text(source, encoding="utf-8")
sys.path.insert(0, str(RUNTIME_DIR.resolve()))
import tacs_visual_web as visual
import train_sun397_tacs_web as common
import sun397_caption_web as caption_service
import mtacs_multimodal_web as mtacs
import train_sun397_mtacs_web as workflow
CONFIG = workflow.default_config()
CONFIG.update(epochs=EPOCHS, batch_size=LOGICAL_BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH_SIZE,
              workers=NUM_WORKERS, seed=SEED, image_size=IMAGE_SIZE, gradient_checkpointing=True,
              feature_batch_size=32 if GPU_GIB >= 40 else 8, eval_batch_size=8 if GPU_GIB >= 40 else 4)
CONFIG = common.config_for_resume(CONFIG, OUTPUT_DIR, RESUME_FROM_DIR)
EXECUTION["compute_batch_size"] = min(COMPUTE_BATCH_SIZE, CONFIG["batch_size"])
workflow.validate_config(CONFIG)
common.execution_options(CONFIG, EXECUTION)
common.seed_all(SEED)
SHARED_DIR = OUTPUT_DIR / "shared"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_SETTINGS = {"policy": caption_service.POLICY, "task": caption_service.PROMPT,
                    "max_new_tokens": CAPTION_MAX_NEW_TOKENS, "batch_size": CAPTION_BATCH_SIZE,
                    "precision": "float32", "semantic_audit_threshold": 0.30, "aliases": []}
CAPTION_SETTINGS = workflow.caption_settings_for_resume(CAPTION_SETTINGS, SHARED_DIR, RESUME_FROM_DIR)
display(CONFIG)
display(EXECUTION)
display(CAPTION_SETTINGS)
''')
    set_source(cells[9], """## 5 Reuse or generate registered captions and frozen CLIP vectors

First validate/copy the old shared cache into the new output. A complete cache is reused, not regenerated. A partial registered CSV can resume missing images when no trained checkpoint depends on it yet. Old checkpoint plus missing/mismatched cache is an error, not permission to create different captions for the same run. CSV, vectors, masks, image hashes, model assets and caption settings remain bound together.

Fresh runs still generate captions from pixels only, never class labels or image path hints. Caption policy is unchanged; no test-driven threshold tuning. Inspect valid query/candidate coverage before training. Caption workers are isolated and tokenizer parallelism is disabled. They release Florence/CLIP GPU memory before the classifier model is constructed.
""")
    code = "".join(cells[10]["source"])
    anchor = "TEXT_CACHE, CAPTION_ENV = workflow.prepare_captions("
    insertion = "workflow.stage_previous_caption_cache(RESUME_FROM_DIR, SHARED_DIR, caption_records, class_names, MODEL_ASSETS, CAPTION_SETTINGS)\n"
    if code.count(anchor) != 1:
        raise ValueError("Caption staging insertion point changed")
    set_source(cells[10], code.replace(anchor, insertion + anchor, 1))
    set_source(cells[13], """## 7 Memory-bounded full_both train and checked resume

Both query and selected-candidate text remain active in fusion and selector. Gumbel task gradients flow through selected pixels AND selected text vectors. The separate categorical policy reward uses that action's matching candidate image, caption and mask; its query-only baseline keeps the same query caption.

Reward normalization uses all examples in the logical batch, not each GPU chunk. Every chunk immediately backpropagates its correctly weighted task/policy loss. Selector RNG is replayed to match the policy that sampled actions. No failed/OOM batch is skipped. q_gate/c_gate/semantic_mix, throughput, pixel-bank device, peak VRAM and refresh/validation timings are saved. Zero gates remain correct for invalid captions.

Original checkpoints and cache are not overwritten. Compatible completed epochs migrate with optimizer/scheduler/scaler/RNG/best/history and source/cache provenance. Changing only compute chunk is allowed on repaired resume; logical batch, horizon, pool, seed, caption content and policy stay fixed. After OOM restart and lower COMPUTE_BATCH_SIZE, not LOGICAL_BATCH_SIZE.
""")
    code = "".join(cells[14]["source"])
    old_call = "CONFIG, ASSETS, OUTPUT_DIR, DEVICE)"
    if code.count(old_call) != 1:
        raise ValueError("M-TACS training call changed")
    set_source(cells[14], code.replace(old_call, "CONFIG, ASSETS, OUTPUT_DIR, DEVICE, execution=EXECUTION, resume_from=RESUME_FROM_DIR)", 1))
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            compile("".join(cell["source"]), "notebook_cell", "exec")
    notebook["metadata"].setdefault("tacs", {}).update(
        repair="memory_bounded_two_pass_v2", original_notebook_backup=BACKUP.relative_to(PROJECT_ROOT).as_posix())
    return notebook


def main():
    if not BACKUP.is_file():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TARGET, BACKUP)
    notebook = build_notebook()
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Repaired:", TARGET)
    print("Previous notebook preserved:", BACKUP)


if __name__ == "__main__":
    main()
