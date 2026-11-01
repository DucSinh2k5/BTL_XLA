"""Export the complete embedded-runtime cell with both inference tensor fixes."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from _project_bootstrap import PROJECT_ROOT


NOTEBOOK = PROJECT_ROOT / "notebooks/04_deployment/sun397/02_mtacs_full_both/sun397-mtacs-full-both-vitl16-train-export-web.ipynb"
TARGET = PROJECT_ROOT / "docs/guides/snippets/mtacs_cell03_embedded_runtime_inference_fixed.py"
PAIRS = (
    ("vectors[~mask] = 0", "vectors = vectors.masked_fill((~mask).unsqueeze(1), 0.0)"),
    ("vectors[~masks] = 0", "vectors = vectors.masked_fill((~masks).unsqueeze(1), 0.0)"),
)


def build_cell():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original = "".join(notebook["cells"][6]["source"])
    tree = ast.parse(original)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "EMBEDDED_SOURCES" for t in node.targets))
    payload = copy.deepcopy(ast.literal_eval(assignment.value))
    source = payload["sun397_caption_web.py"]
    for old, new in PAIRS:
        if source.count(old) == 1:
            source = source.replace(old, new, 1)
        elif source.count(new) != 1:
            raise ValueError("Caption masking implementation changed; re-audit the fix")
    compile(source, "sun397_caption_web.py", "exec")
    payload["sun397_caption_web.py"] = source
    result = '''# Replace the complete section 3 Embedded runtime code cell with this cell.
# It includes every original runtime source; nothing is omitted.
import importlib
import json

RUNTIME_DIR = OUTPUT_DIR / "embedded_runtime_v2"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_SOURCES = ''' + repr(payload) + '''

def _caption_tensor_fix(source):
    for old, new in (
        ("vectors[~mask] = 0", "vectors = vectors.masked_fill((~mask).unsqueeze(1), 0.0)"),
        ("vectors[~masks] = 0", "vectors = vectors.masked_fill((~masks).unsqueeze(1), 0.0)"),
    ):
        if source.count(old) == 1:
            source = source.replace(old, new, 1)
        elif source.count(new) != 1:
            raise ValueError("Unexpected caption code; refusing to rewrite it")
    return source


def _stage_embedded_runtime(name, source):
    path = RUNTIME_DIR / name
    module_name = Path(name).stem
    loaded = sys.modules.get(module_name)
    if loaded is not None and Path(loaded.__file__).resolve() != path.resolve():
        raise RuntimeError("Another TACS/M-TACS runtime is imported. Restart the kernel, then run this notebook in order.")
    if path.is_file() and path.read_text(encoding="utf-8") != source:
        previous = path.read_text(encoding="utf-8")
        if name != "sun397_caption_web.py" or _caption_tensor_fix(previous) != source:
            raise ValueError("Runtime differs beyond the two inference tensor fixes; preserve the run and inspect it")
        protected = [OUTPUT_DIR / name for name in ("protocol.json", "last.pt", "best.pt", "training_complete.json")]
        protected += [OUTPUT_DIR / "shared" / name for name in ("captions.csv", "caption_protocol.json", "text_embeddings.pt")]
        if any(item.exists() for item in protected):
            raise RuntimeError(
                "This output already contains a registered cache or training state. "
                "Keep it unchanged and select a NEW OUTPUT_DIR before running this fixed cell. "
                "Do not delete captions or bypass their source/hash guards."
            )
        backup = path.with_name("sun397_caption_web.before_inference_tensor_fix.py")
        if not backup.exists():
            backup.write_text(previous, encoding="utf-8")
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(path)
    elif not path.is_file():
        path.write_text(source, encoding="utf-8")
    if loaded is not None and module_name == "sun397_caption_web":
        importlib.invalidate_caches()
        importlib.reload(loaded)


for name, source in EMBEDDED_SOURCES.items():
    _stage_embedded_runtime(name, source)
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
print("Caption worker ready: both inference tensor masks are fixed")
'''
    compile(result, "mtacs_fixed_runtime_cell", "exec")
    return result


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(build_cell(), encoding="utf-8", newline="\n")
    print("Complete replacement cell:", TARGET)
