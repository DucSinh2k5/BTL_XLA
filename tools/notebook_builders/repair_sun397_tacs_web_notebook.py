"""Repair the user's interrupted web notebook without changing other experiments."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

from _project_bootstrap import PROJECT_ROOT


TARGET = PROJECT_ROOT / "tacs_web.ipynb"
BACKUP = PROJECT_ROOT / "tmp/notebook_backups/tacs_web.before_memory_repair.ipynb"
MODEL = PROJECT_ROOT / "src/models/tacs_visual_web.py"
BASE = PROJECT_ROOT / "src/training/train_sun397_tacs_web.py"
PATCH = PROJECT_ROOT / "src/training/sun397_tacs_web_training_patch.py"


def repaired_training_source():
    source = BASE.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    replacements = []
    replaced = {"default_config", "validate_config", "refresh_bank", "train_epoch", "train_run"}
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in replaced:
            found.add(node.name)
            first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            if node.name in {"default_config", "validate_config"}:
                node.name = "_legacy_" + node.name
                replacement = ast.unparse(node) + "\n"
            else:
                replacement = ""
            replacements.append((first, node.end_lineno, replacement))
    if found != replaced:
        raise ValueError("Base training interface changed; re-audit the repair before rebuilding")
    for first, last, replacement in sorted(replacements, reverse=True):
        lines[first:last] = [replacement]
    hashes = {"runtime": hashlib.sha256(MODEL.read_text(encoding="utf-8").encode()).hexdigest(),
              "training": hashlib.sha256(source.encode()).hexdigest()}
    result = "".join(lines) + "\n\nLEGACY_SOURCE_SHA256 = " + repr(hashes) + "\n\n" + PATCH.read_text(encoding="utf-8")
    compile(result, "train_sun397_tacs_web.py", "exec")
    return result


def set_source(cell, source):
    source = source.strip() + "\n"
    if cell["cell_type"] == "code":
        ast.parse(source)
    cell["source"] = source.splitlines(keepends=True)
    cell["id"] = hashlib.sha256(source.encode()).hexdigest()[:12]


def build_notebook(original):
    notebook = copy.deepcopy(original)
    cells = notebook["cells"]
    original_config = "".join(cells[4]["source"])
    assignments = {target.id: ast.get_source_segment(original_config, node.value)
                   for node in ast.parse(original_config).body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    if len(cells) < 19 or "EMBEDDED_SOURCES" not in "".join(cells[6]["source"]):
        raise ValueError("Not the checked visual SUN397 web notebook")
    set_source(cells[0], """# SUN397 visual TACS L/16: repaired memory-bounded training and web export

This repairs the interrupted `tacs_web.ipynb`, not the older caption experiments. The inference architecture is unchanged: shared DINOv3 ViT-L/16, 384 pixels, K=1, query/context patch concatenation, task CE with straight-through Gumbel selection and a categorical REINFORCE loss. No captions, labels-as-input, frozen backbone, or index-only task estimator are substituted.

The failed configuration used **GPU batch 64 with checkpointing disabled**, exhausting a 95 GiB GPU in the first task forward. Here **logical batch 64** is separate from **GPU compute batch 8**. Reward normalization uses the complete logical batch. A no-grad reward prepass followed by RNG-replayed selector gradients bounds activation memory to a compute chunk; this adds selector recomputation, not a new loss. Gradient checkpointing remains enabled.

The training pixel bank is uploaded once as float32 when free VRAM permits. Its values are the original fp16 training pixels converted to float32; the straight-through derivative is unchanged. CPU fallback is explicit and can be slower. Raw fp32 preprocessing is still used for validation, final test and the checkpoint-bound export feature bank. Feature refresh uses parallel image loading. Initial logical batches print actual throughput and peak allocated VRAM; no GPU speedup or maximum batch is guaranteed without measuring on your device.

**Restart the Kaggle kernel before Run All**, especially after an OOM. Old live models, optimizer tensors and traceback references are not freed by `empty_cache()` alone. There are no pip installs or Internet calls. Keep the same local SUN397 and DINOv3 Inputs.

The repaired run uses a new output directory. If the previous output contains a compatible `last.pt`, `best.pt` and `protocol.json`, it validates the original source/data/pool/config identity and copies all resume states into the new run, without changing the originals. Logical batch, epoch horizon, seed and learning settings are preserved. Changing only COMPUTE_BATCH_SIZE on a repaired resume is allowed. This execution order is not bitwise replay of the old training loop; migration provenance is recorded. If no completed epoch was saved, the run explicitly starts epoch 1. A notebook file alone does not contain checkpoint weights.

Best is selected by validation CE. Full trained weights, candidate images, feature bank, exact class order, vendored DINOv3 code, inference adapter, metrics and real attention readouts are exported and checked as before. Attention remains last-block mean-head CLS attention, not a separate network or a causal/class-specific explanation.

This remains a declared L/16 deployment adaptation, not an exact paper reproduction: detached epoch-refreshed candidate keys, a validation holdout, one seed and the existing augmentation choices are retained. The repair does not silently change those experiment choices.
""")
    preflight = "".join(cells[2]["source"])
    preflight = preflight.replace('import os\n', 'import os\nos.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")\n', 1)
    set_source(cells[2], preflight)
    set_source(cells[3], """## 2 Paths, logical batch and GPU compute batch

Keep L/16 and IMAGE_SIZE=384. LOGICAL_BATCH_SIZE=64 defines reward statistics and EFFECTIVE_BATCH_SIZE=64 defines optimizer accumulation. COMPUTE_BATCH_SIZE=8 only controls GPU activation memory; it can be 1, 2, 4, 8, 12 or 16 as memory permits, without requiring divisibility by 64. Start at 8 on the shown 96 GB GPU. Do not set it to 64 or disable checkpointing just because total VRAM is large.

Keep RESUME_FROM_DIR pointing to the previous run folder; on a new Kaggle session attach its saved Output and use its exact folder containing protocol.json, last.pt and best.pt. Do not point it at a ZIP or at the repaired output. The new OUTPUT_DIR keeps all previous results untouched. To measure another GPU compute chunk later, change only COMPUTE_BATCH_SIZE and restart the kernel. The registered logical experiment must stay fixed.
""")
    paths = "\n".join(name + " = " + assignments[name] for name in (
        "DATASET_ROOT", "TRAIN_PARTITION", "TEST_PARTITION", "DINOV3_REPO_DIR", "DINOV3_WEIGHTS", "PROJECT_DIR"))
    # Retain the user's exact mount paths, not the original template owner's path.
    old_output = assignments["OUTPUT_DIR"]
    set_source(cells[4], paths + '\nLEGACY_OUTPUT_DIR = ' + old_output + '''
OUTPUT_DIR = Path(os.getenv("SUN397_TACS_WEB_REPAIRED_OUTPUT", str(LEGACY_OUTPUT_DIR.with_name(LEGACY_OUTPUT_DIR.name + "_repaired_v2"))))
RESUME_FROM_DIR = Path(os.getenv("SUN397_TACS_WEB_RESUME_FROM", str(LEGACY_OUTPUT_DIR)))
DEVICE = torch.device("cuda")
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
LOGICAL_BATCH_SIZE = 64
EFFECTIVE_BATCH_SIZE = 64
COMPUTE_BATCH_SIZE = int(os.getenv("SUN397_TACS_WEB_COMPUTE_BATCH", "8" if GPU_GIB >= 40 else "2"))
NUM_WORKERS = int(os.getenv("SUN397_TACS_WEB_WORKERS", str(min(4, os.cpu_count() or 1))))
EPOCHS = int(os.getenv("SUN397_TACS_WEB_EPOCHS", "100"))
SEED = 42
IMAGE_SIZE = 384
EXECUTION = {"compute_batch_size": COMPUTE_BATCH_SIZE, "pixel_bank_device": "auto",
             "gpu_reserve_gib": 20.0, "profile_batches": 3}
RUN_TRAINING = True
RUN_FINAL_TEST = True
RUN_EXPORT = True
print("Output:", OUTPUT_DIR, "| previous run:", RESUME_FROM_DIR)
print("GPU:", torch.cuda.get_device_name(0), "| GiB:", round(GPU_GIB, 2))
print("Requested logical / compute batch:", LOGICAL_BATCH_SIZE, COMPUTE_BATCH_SIZE)
''')
    payload = {"tacs_visual_web.py": MODEL.read_text(encoding="utf-8"),
               "train_sun397_tacs_web.py": repaired_training_source()}
    set_source(cells[6], '''import hashlib
import json
RUNTIME_DIR = OUTPUT_DIR / "embedded_runtime_v2"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_SOURCES = ''' + repr(payload) + '''
for filename, source in EMBEDDED_SOURCES.items():
    module_name = Path(filename).stem
    if module_name in sys.modules:
        module_path = Path(sys.modules[module_name].__file__).resolve()
        expected = (RUNTIME_DIR / filename).resolve()
        if module_path != expected or module_path.read_text(encoding="utf-8") != source:
            raise RuntimeError("A previous training runtime is loaded. Restart the kernel, then Run All this repaired notebook.")
    path = RUNTIME_DIR / filename
    if path.is_file() and path.read_text(encoding="utf-8") != source:
        raise ValueError("Runtime source changed in the repaired output; preserve that run and use another output directory")
    if not path.is_file():
        path.write_text(source, encoding="utf-8")
sys.path.insert(0, str(RUNTIME_DIR.resolve()))
import tacs_visual_web as web_runtime
import train_sun397_tacs_web as workflow
CONFIG = workflow.default_config()
CONFIG.update(epochs=EPOCHS, batch_size=LOGICAL_BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH_SIZE,
              workers=NUM_WORKERS, seed=SEED, image_size=IMAGE_SIZE,
              feature_batch_size=32 if GPU_GIB >= 40 else 8, eval_batch_size=8 if GPU_GIB >= 40 else 4,
              gradient_checkpointing=True)
CONFIG = workflow.config_for_resume(CONFIG, OUTPUT_DIR, RESUME_FROM_DIR)
EXECUTION["compute_batch_size"] = min(COMPUTE_BATCH_SIZE, CONFIG["batch_size"])
workflow.validate_config(CONFIG)
workflow.execution_options(CONFIG, EXECUTION)
workflow.seed_all(SEED)
display(CONFIG)
display(EXECUTION)
''')
    set_source(cells[11], """## 6 Train, profile and safely continue previous epochs

The first pass samples policy actions without retaining autograd graphs, computes detached rewards in eval mode, and standardizes them over the entire logical batch. A second pass replays selector RNG for the same policy scores, then computes fresh hard Gumbel task selections, task CE and policy gradients in small GPU chunks. Each chunk immediately backpropagates its correctly weighted loss. Optimizer updates happen only at the effective-batch boundary. Both task straight-through and policy gradients are retained.

No OOM batch is skipped and no automatic mid-update halving is performed. If OOM still occurs, restart the kernel and lower COMPUTE_BATCH_SIZE, keeping logical batch and epoch horizon unchanged. The repaired checkpoint resumes from its last completed epoch. The progress bar counts logical batches, not GPU chunks. Profile lines show actual images/s and peak allocated GiB. The prior traceback failed at epoch 1 batch 0, so that failed attempt alone provides no completed epoch to recover.

Migration accepts only the checked legacy visual code and matching classes, pool, seed, assets and training settings. It preserves model/optimizer/scheduler/scaler/RNG/history and the prior best validation checkpoint. Previous sources and data are not overwritten. Runs that include old epochs are marked in protocol.json and migration.json; do not describe them as a fresh, bitwise identical run of the new loop.
""")
    training = "".join(cells[12]["source"])
    training = training.replace("CONFIG, ASSETS, OUTPUT_DIR, DEVICE)",
                                "CONFIG, ASSETS, OUTPUT_DIR, DEVICE, execution=EXECUTION, resume_from=RESUME_FROM_DIR)", 1)
    set_source(cells[12], training)
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            compile("".join(cell["source"]), "notebook_cell", "exec")
    notebook["metadata"].setdefault("tacs", {}).update(
        repair="memory_bounded_two_pass_v2", original_notebook_backup=BACKUP.relative_to(PROJECT_ROOT).as_posix(),
        note="Original outputs preserved in backup; architecture and inference runtime unchanged")
    return notebook


def main():
    if not BACKUP.is_file():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TARGET, BACKUP)
    original = json.loads(BACKUP.read_text(encoding="utf-8"))
    notebook = build_notebook(original)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Repaired:", TARGET)
    print("Original notebook and outputs preserved:", BACKUP)


if __name__ == "__main__":
    main()
