"""Offline caption-stage code embedded in the standalone DTD text notebooks."""
from __future__ import annotations

from project_paths import PROJECT_ROOT, project_path, project_glob

import ast
import json
from pathlib import Path


ROOT = PROJECT_ROOT

CONFIG = '''
GENERATE_CAPTIONS_IF_MISSING = True  # Reuse complete CSVs; otherwise create/resume with local models.
FLORENCE_MODEL_REF = os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base")
OFFLINE_ASSETS_ROOT = os.getenv("DTD_OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi")
CAPTION_BATCH_SIZE = int(os.getenv("DTD_CAPTION_BATCH_SIZE", "16"))
CAPTION_WORKERS = 0  # Caption workers stay in the child process; avoid tokenizer/fork warnings.
CAPTION_MAX_NEW_TOKENS = 48
CAPTION_PROMPT_VERSION = "florence2_detailed_v2"
'''

HELPERS = '''
def caption_progress(path, split_frames):
    frame = pd.read_csv(path).fillna("")
    required = {"image_id", "final_caption", "caption_valid", "leakage_detected",
                "semantic_leakage_detected", "semantic_leakage_score"}
    if not required.issubset(frame.columns):
        raise ValueError("Caption CSV is missing audit columns: " + str(sorted(required - set(frame.columns))))
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate caption image IDs; resolve the CSV before resuming")
    expected = {str(image_id) for part in split_frames.values() for image_id in part.image_id}
    if set(frame.image_id.astype(str)) - expected:
        raise ValueError("Partial caption CSV contains images outside this DTD dataset")
    return frame


def florence_checkpoint(model_ref, roots):
    def checked(path):
        path = Path(path)
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if config.get("model_type") != "florence2":
            return None
        indexes = list(path.glob("*.index.json"))
        weights = any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
        for index in indexes:
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
                weights |= bool(shards) and all((path / shard).is_file() for shard in shards)
            except (OSError, ValueError):
                continue
        if not weights:
            return None
        backend = "legacy" if "AutoModelForCausalLM" in config.get("auto_map", {}) else "native"
        if backend == "legacy" and not all((path / name).is_file() for name in (
                "configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py")):
            return None
        return path.resolve(), backend

    requested = Path(model_ref).expanduser()
    if requested.is_dir():
        result = checked(requested)
        if result:
            return result
        for config in sorted(requested.rglob("config.json")):
            result = checked(config.parent)
            if result:
                return result
        raise FileNotFoundError(f"Florence folder is incomplete: {requested}; need config, weights, processor/tokenizer and local model code.")
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in {"images", "__pycache__", ".git"})
            if len(Path(current).relative_to(root).parts) >= 8:
                dirs[:] = []
            if "config.json" in files:
                result = checked(current)
                if result:
                    return result
    raise FileNotFoundError(
        f"No complete local Florence-2 checkpoint found for {model_ref}. "
        "Attach notebook-mt-moi/Florence-2-base or set FLORENCE_MODEL_REF to the correct local folder. No download is attempted."
    )


def offline_caption_wheels(roots, destination):
    import zipfile
    destination = Path(destination)
    wheel_dirs = set()
    prefixes = ("transformers-", "tokenizers-", "huggingface_hub-", "accelerate-", "einops-", "timm-")
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in {"images", "__pycache__", ".git", "Florence-2-base"})
            if len(Path(current).relative_to(root).parts) >= 7:
                dirs[:] = []
            for name in sorted(files):
                path = Path(current) / name
                if name.endswith(".whl"):
                    wheel_dirs.add(str(path.parent))
                elif name.lower() in {"offline_wheels.zip", "wheels.zip"}:
                    with zipfile.ZipFile(path) as archive:
                        for member in archive.infolist():
                            filename = Path(member.filename.replace("\\\\", "/")).name
                            if not filename.endswith(".whl") or not filename.startswith(prefixes):
                                continue
                            destination.mkdir(parents=True, exist_ok=True)
                            target = destination / filename
                            if not target.is_file() or target.stat().st_size != member.file_size:
                                with archive.open(member) as source, target.open("wb") as handle:
                                    shutil.copyfileobj(source, handle)
                            wheel_dirs.add(str(destination))
    return sorted(wheel_dirs)


def caption_runtime_environment(backend, asset_root, runtime_dir):
    import subprocess
    import sys
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_MODULES_CACHE=str(runtime_dir / "hf_modules"), PYTHONUNBUFFERED="1")
    target = runtime_dir / "legacy_packages"
    probe = (
        "import transformers; "
        + ("assert transformers.__version__.startswith('4.49.'), transformers.__version__; " if backend == "legacy" else "")
        + "from transformers import AutoProcessor, AutoTokenizer, CLIPModel; "
        + ("from transformers import AutoModelForCausalLM; " if backend == "legacy" else
           "from transformers import Florence2ForConditionalGeneration; ")
        + "import einops, timm, accelerate; print(transformers.__version__)"
    )
    if target.is_dir():
        env["PYTHONPATH"] = str(target) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    if result.returncode == 0:
        print("Caption subprocess Transformers:", result.stdout.strip())
        return env
    if backend != "legacy":
        raise RuntimeError("The local native Florence checkpoint cannot load in this runtime. "
                           "Use a matching native Transformers installation or the attached legacy model + offline 4.49 wheels.\\n"
                           + result.stderr[-3000:])
    wheel_dirs = offline_caption_wheels([Path(asset_root), Path("/kaggle/input")], runtime_dir / "wheels")
    if not wheel_dirs:
        raise FileNotFoundError(
            "The attached Microsoft Florence checkpoint needs Transformers 4.49.x. "
            f"No offline wheels found under {asset_root}. Attach offline_wheels.zip/wheels.zip or a wheel folder "
            "with transformers==4.49.0, tokenizers 0.21.x and huggingface-hub <1 compatible with this Python version. "
            "The notebook never enables Internet or modifies the kernel's Transformers."
        )
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--only-binary=:all:",
               "--no-deps", "--upgrade", "--target", str(target), "transformers==4.49.0",
               "tokenizers>=0.21,<0.22", "huggingface-hub>=0.26,<1"]
    import importlib.util
    for module in ("einops", "timm", "accelerate"):
        if importlib.util.find_spec(module) is None:
            command.append(module)
    for directory in wheel_dirs:
        command.extend(["--find-links", directory])
    print("Preparing isolated Florence dependencies from attached offline wheels; the notebook kernel is unchanged.")
    installed = subprocess.run(command, env=env, capture_output=True, text=True)
    if installed.returncode:
        raise RuntimeError("Offline Florence dependency setup failed. Wheels must match Kaggle's Python/platform.\\n"
                           + installed.stdout[-2000:] + installed.stderr[-3000:])
    base_pythonpath = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(target) + (os.pathsep + base_pythonpath if base_pythonpath else "")
    checked = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError("The offline Florence environment still lacks a compatible dependency. "
                           "No kernel packages were changed.\\n" + checked.stderr[-4000:])
    print("Isolated caption subprocess Transformers:", checked.stdout.strip())
    return env


def run_local_dtd_caption_generation(split_frames, csv_path, clip_asset):
    import subprocess
    import sys
    if CAPTION_BATCH_SIZE < 1 or CAPTION_MAX_NEW_TOKENS < 1:
        raise ValueError("Caption batch size and token budget must be positive")
    local_model, backend = florence_checkpoint(FLORENCE_MODEL_REF, [Path(OFFLINE_ASSETS_ROOT), Path("/kaggle/input"),
        Path.home() / ".cache/huggingface/hub"])
    runtime_dir = Path(csv_path).parent / "caption_runtime"
    env = caption_runtime_environment(backend, OFFLINE_ASSETS_ROOT, runtime_dir)
    records = [{"image_id": str(row.image_id), "image_path": str(row.image_path), "split": split}
               for split, frame in split_frames.items() for row in frame.itertuples(index=False)]
    model_id = "microsoft/Florence-2-base" if backend == "legacy" else "florence-community/Florence-2-base"
    if Path(csv_path).is_file():
        old = caption_progress(csv_path, split_frames)
        if len(old):
            versions = set(old.get("prompt_version", pd.Series(dtype=str)))
            identities = set(old.get("vlm_model", pd.Series(dtype=str)))
            if versions != {CAPTION_PROMPT_VERSION} or len(identities) != 1:
                raise ValueError("Partial caption CSV has missing/mixed generation metadata; use a new output directory.")
            old_identity = str(next(iter(identities)))
            if old_identity != model_id and old_identity != str(local_model):
                raise ValueError("Partial CSV uses another Florence checkpoint; use its original model or a new output directory.")
            model_id = old_identity
    payload = {"records": records, "output": str(Path(csv_path).resolve()), "split": DTD_SPLIT,
               "florence": str(local_model), "clip": str(clip_asset), "backend": backend,
               "vlm_model": model_id, "prompt_version": CAPTION_PROMPT_VERSION,
               "threshold": CAPTION_SEMANTIC_THRESHOLD, "batch_size": CAPTION_BATCH_SIZE,
               "max_new_tokens": CAPTION_MAX_NEW_TOKENS, "class_names": list(class_names)}
    worker = runtime_dir / "generate_dtd_captions.py"
    worker.write_text(CAPTION_WORKER_SOURCE, encoding="utf-8")
    payload_path = runtime_dir / "request.json"
    write_json(payload_path, payload)
    print("Creating/resuming captions from image pixels only:", csv_path, flush=True)
    log_path = runtime_dir / "generation.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(worker), str(payload_path)], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if returncode:
        raise RuntimeError(f"Florence caption process exited with code {returncode}; log: {log_path}. "
                           "Completed CSV batches are preserved; fix the error above and rerun this cell to resume.")


def ensure_dtd_caption_csv(split_frames, explicit_path, cached_path, roots):
    cached_path = Path(cached_path)
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    source = find_dtd_caption_csv(split_frames, explicit_path, cached_path, roots,
                                 allow_partial=True, allow_missing=True)
    if source is not None and source.resolve() != cached_path.resolve():
        shutil.copy2(source, cached_path)
    complete = False
    if cached_path.is_file():
        progress = caption_progress(cached_path, split_frames)
        expected = {str(image_id) for frame in split_frames.values() for image_id in frame.image_id}
        complete = set(progress.image_id.astype(str)) == expected
    if not complete:
        if not GENERATE_CAPTIONS_IF_MISSING:
            raise FileNotFoundError("DTD captions are missing/incomplete. Enable GENERATE_CAPTIONS_IF_MISSING to create/resume them offline.")
        clip_asset = resolve_clip_path(CLIP_MODEL_LOCAL_PATH, [Path("/kaggle/input"), PROJECT_DIR])
        run_local_dtd_caption_generation(split_frames, cached_path, clip_asset)
    load_ablation_captions(cached_path, split_frames)
    if source is not None:
        write_json(cached_path.parent / "caption_source.json", {
            "source": str(source), "sha256": hashlib.sha256(cached_path.read_bytes()).hexdigest(),
            "generation_needed": not complete,
        })
    return cached_path
'''


def worker_source():
    notebook = json.loads((project_path('mtacs-train-captions-vitl16-dtd.ipynb')).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][7]["source"])
    start = source.index("CAPTION_CSV = OUT_PUT_DIR")
    definitions = source[:start].replace("TASK_PROMPT = CAPTION_TASK_PROMPT", 'TASK_PROMPT = "<MORE_DETAILED_CAPTION>"')
    # Paths have already been resolved and checked by the parent, so no dataset scan is needed here.
    tree = ast.parse(definitions)
    resolver = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_local_model")
    definitions = definitions.replace(ast.get_source_segment(definitions, resolver),
        'def resolve_local_model(model_ref, expected_type, extra_roots=()):\n'
        '    path = Path(model_ref)\n'
        '    return path if path.is_dir() else None')
    run = source[start:]
    record_start = run.index("caption_records = []")
    record_end = run.index("if len({row['image_id']", record_start)
    run = run[:record_start] + 'caption_records = payload["records"]\n' + run[record_end:]
    run = run.replace("CAPTION_CSV = OUT_PUT_DIR / 'captions.csv'", 'CAPTION_CSV = Path(payload["output"])')
    run = run.replace("print('Valid captions:'", "print('CSV legacy strict-audit valid captions:'")
    prefix = '''import json
import os
import sys
from pathlib import Path
import torch
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
FLORENCE_BACKEND = payload["backend"]
_transformers_version = transformers.__version__
FLORENCE_MODEL_REF = payload["florence"]
VLM_MODEL = payload["vlm_model"]
CLIP_MODEL_LOCAL_PATH = payload["clip"]
PROMPT_VERSION = payload["prompt_version"]
DTD_SPLIT = payload["split"]
CAPTION_BATCH_SIZE = payload["batch_size"]
CAPTION_WORKERS = 0
CAPTION_MAX_NEW_TOKENS = payload["max_new_tokens"]
CAPTION_SEMANTIC_THRESHOLD = payload["threshold"]
ENABLE_LEAKAGE_CHECK = True
class_names = payload["class_names"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
    result = prefix + "\n" + definitions + "\n" + run
    ast.parse(result)
    return result


def embedded_helpers():
    screening = (project_path('dtd_caption_screening.py')).read_text(encoding="utf-8")
    return HELPERS + "\n\n" + screening + "\n\nCAPTION_WORKER_SOURCE = " + repr(worker_source()) + "\n"
