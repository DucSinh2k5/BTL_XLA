"""Offline Florence-Prefix and CLIP service shared by SUN397 train and web.

Importing this module does not import Transformers. The legacy model runs in a
child process with its own pinned environment; the notebook kernel is unchanged.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import torch
from torch.nn import functional as F

import tacs_visual_web as visual


POLICY = "sun397_lexical_clip_audit_v1"
PROMPT = "<MORE_DETAILED_CAPTION>"
COLUMNS = ["image_id", "split", "image_sha256", "raw_caption", "final_caption", "caption_valid",
           "leakage_detected", "lexical_leakage_detected", "semantic_similarity_score", "semantic_nearest_class",
           "semantic_similarity_flag", "caption_policy", "captioner_assets_sha256", "clip_assets_sha256",
           "vlm_model", "prompt_version"]


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def forbidden_terms(classes, aliases=()):
    terms = set()
    for name in list(classes) + list(aliases):
        # SUN hierarchy qualifiers (indoor/outdoor) are not standalone classes.
        parts = [p for p in str(name).replace("\\", "/").split("/") if p]
        if parts and len(parts[0]) == 1:
            parts = parts[1:]
        if parts:
            terms.add(normalized(" ".join(parts)))
            terms.add(normalized(parts[0]))
    return sorted(filter(None, terms), key=lambda value: (-len(value.split()), value))


def clean_caption(raw, terms):
    text = str(raw).strip()
    prefix = re.compile(
        r"^(?:(?:the|this)\s+(?:image|photo|picture)\s+(?:is|shows|depicts)\s+|"
        r"(?:a|an)\s+(?:close[ -]?up|detailed)\s+(?:view|photo|image|picture)?\s*(?:of\s+)?|"
        r"(?:a|an|the)\s+)", re.IGNORECASE)
    for _ in range(8):
        updated = prefix.sub("", text, count=1).strip()
        if updated == text:
            break
        text = updated
    words = re.findall(r"[A-Za-z0-9]+", text)[:8]
    final = " ".join(words)
    probe = " " + normalized(final) + " "
    matches = [term for term in terms if f" {term} " in probe]
    return final, 5 <= len(words) <= 8 and not matches, matches


def tree_identity(folder):
    folder = Path(folder).resolve()
    ignored = {".cache", ".git", "__pycache__"}
    files = {p.relative_to(folder).as_posix(): visual.file_sha256(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and not ignored.intersection(p.relative_to(folder).parts) and p.suffix != ".pyc"}
    if not files:
        raise FileNotFoundError(f"Empty asset folder: {folder}")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def validate_assets(florence, clip):
    def complete(folder):
        folder = Path(folder).expanduser().resolve()
        config = visual.read_json(folder / "config.json")
        files = [p for p in folder.glob("*.safetensors")] + list(folder.glob("pytorch_model*.bin"))
        if not files:
            raise FileNotFoundError(f"Model weights not found in {folder}")
        for index in folder.glob("*.index.json"):
            for name in set(visual.read_json(index).get("weight_map", {}).values()):
                if not visual.bundle_path(folder, name).is_file():
                    raise FileNotFoundError(f"Missing model shard: {name}")
        if not (folder / "tokenizer_config.json").is_file():
            raise FileNotFoundError(f"Tokenizer config missing in {folder}")
        if not (folder / "tokenizer.json").is_file() and not all((folder / n).is_file() for n in ("vocab.json", "merges.txt")):
            raise FileNotFoundError(f"Tokenizer vocabulary missing in {folder}")
        return folder, config

    florence, fc = complete(florence)
    clip, cc = complete(clip)
    if fc.get("model_type") != "florence2" or "AutoModelForCausalLM" not in fc.get("auto_map", {}):
        raise ValueError("Use the original Microsoft legacy Florence-2-base, not native community/Qwen/LoRA weights")
    for name in ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py", "preprocessor_config.json"):
        if not (florence / name).is_file():
            raise FileNotFoundError(f"Florence asset missing: {name}")
    if cc.get("model_type") != "clip" or cc.get("projection_dim") != 512 or cc.get("vision_config", {}).get("patch_size") != 32:
        raise ValueError("Use a complete local CLIP ViT-B/32 (512-dimensional projection)")
    if (florence / "adapter_config.json").exists():
        raise ValueError("This deployment recipe uses frozen Florence-2-base, not a partial adapter")
    return {"florence": str(florence), "clip": str(clip), "florence_sha256": tree_identity(florence),
            "clip_sha256": tree_identity(clip)}


def wheel_directories(asset_root, destination):
    root, destination = Path(asset_root), Path(destination)
    if not root.is_dir():
        raise FileNotFoundError(f"Offline asset root not found: {root}")
    folders = set()
    for current, dirs, files in os.walk(root):
        dirs[:] = [n for n in dirs if n not in {".cache", ".git", "images", "__pycache__", "Florence-2-base"}]
        if len(Path(current).relative_to(root).parts) >= 6:
            dirs[:] = []
        for name in files:
            path = Path(current) / name
            if name.endswith(".whl"):
                folders.add(str(path.parent.resolve()))
            elif name.lower() in {"wheels.zip", "offline_wheels.zip"}:
                with zipfile.ZipFile(path) as archive:
                    for item in archive.infolist():
                        filename = Path(item.filename.replace("\\", "/")).name
                        if not filename.endswith(".whl"):
                            continue
                        destination.mkdir(parents=True, exist_ok=True)
                        target = destination / filename
                        if not target.is_file() or target.stat().st_size != item.file_size:
                            with archive.open(item) as source, target.open("wb") as output:
                                shutil.copyfileobj(source, output)
                        folders.add(str(destination.resolve()))
    return sorted(folders)


def caption_environment(runtime_dir, wheels_root=None):
    """Only a child gets PYTHONPATH. Never downgrade notebook/global packages."""
    runtime_dir = Path(runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / "legacy_packages"
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1",
               PYTHONUNBUFFERED="1", HF_MODULES_CACHE=str(runtime_dir / "hf_modules"))
    # Do not inherit a foreign notebook's Python package overlay.
    env.pop("PYTHONPATH", None)
    if target.is_dir():
        env["PYTHONPATH"] = str(target)
    probe = ("import transformers; assert transformers.__version__.startswith('4.49.'), transformers.__version__; "
             "from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, CLIPModel; "
             "import einops, timm, accelerate; print(transformers.__version__)")
    check = subprocess.run([sys.executable, "-c", probe], env=env, text=True, capture_output=True)
    if check.returncode == 0:
        return env
    directories = wheel_directories(wheels_root, runtime_dir / "wheels") if wheels_root else []
    if not directories:
        raise RuntimeError("Florence requires an isolated Transformers 4.49.x environment. Attach offline wheels compatible "
                           "with this Python/platform and set OFFLINE_ASSETS_ROOT; Internet remains off.\n" + check.stderr[-2000:])
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--only-binary=:all:", "--no-deps", "--upgrade",
               "--target", str(target), "transformers==4.49.0", "tokenizers>=0.21,<0.22", "huggingface-hub>=0.26,<1"]
    for name in ("einops", "timm", "accelerate"):
        if importlib.util.find_spec(name) is None:
            command.append(name)
    for directory in directories:
        command.extend(["--find-links", directory])
    installed = subprocess.run(command, env=env, text=True, capture_output=True)
    if installed.returncode:
        raise RuntimeError("Offline dependency setup failed; no notebook kernel packages were changed.\n" + installed.stderr[-4000:])
    env["PYTHONPATH"] = str(target)
    checked = subprocess.run([sys.executable, "-c", probe], env=env, text=True, capture_output=True)
    if checked.returncode:
        raise RuntimeError("Caption environment still has missing/incompatible dependencies.\n" + checked.stderr[-4000:])
    return env


def invoke_worker(payload, env, worker_file=None, work_dir=None):
    worker_file = Path(worker_file or __file__).resolve()
    # Temporary files belong in working/cache, never a read-only model Input.
    with tempfile.TemporaryDirectory(prefix="mtacs_caption_", dir=work_dir) as temporary:
        request = Path(temporary) / "request.json"
        result = Path(temporary) / "result.pt"
        visual.write_json(request, dict(payload, result=str(result)))
        process = subprocess.Popen([sys.executable, str(worker_file), "--request", str(request)], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = deque(maxlen=35)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                tail.append(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
        if code:
            raise RuntimeError("Caption worker failed; completed CSV batches remain resumable.\n" + "".join(tail))
        return torch.load(result, map_location="cpu", weights_only=True)


class CaptionEngine:
    def __init__(self, payload):
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, CLIPModel
        if not transformers.__version__.startswith("4.49."):
            raise RuntimeError("Legacy Florence worker must run with Transformers 4.49.x")
        self.settings = payload["settings"]
        if self.settings["policy"] != POLICY or self.settings["task"] != PROMPT or self.settings["precision"] != "float32":
            raise ValueError("Unsupported caption policy/task/precision")
        self.device = torch.device(payload.get("device", "cpu"))
        self.processor = AutoProcessor.from_pretrained(payload["florence"], trust_remote_code=True, local_files_only=True)
        model, loading = AutoModelForCausalLM.from_pretrained(payload["florence"], torch_dtype=torch.float32,
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager", output_loading_info=True)
        missing = [key for key in loading.get("missing_keys", []) if not key.endswith("position_ids")]
        if missing or loading.get("mismatched_keys"):
            raise ValueError(f"Florence weights did not load completely: {missing}")
        if not 180_000_000 <= sum(p.numel() for p in model.parameters()) <= 300_000_000:
            raise ValueError("Expected Florence-2-base scale, not large/partial/quantized weights")
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(payload["clip"], local_files_only=True)
        clip, loading = CLIPModel.from_pretrained(payload["clip"], torch_dtype=torch.float32,
                                                  local_files_only=True, output_loading_info=True)
        missing = [key for key in loading.get("missing_keys", []) if not key.endswith("position_ids")]
        if missing or loading.get("mismatched_keys"):
            raise ValueError(f"CLIP weights did not load completely: {missing}")
        if clip.config.projection_dim != 512:
            raise ValueError("CLIP text features must have 512 dimensions")
        self.clip = clip.to(self.device).eval().requires_grad_(False)
        self.terms = forbidden_terms(payload["classes"], self.settings.get("aliases", []))
        labels = [name.replace("/", " ").replace("_", " ") for name in payload["classes"]]
        labels = [re.sub(r"^[a-z] ", "", name) for name in labels]
        prompts = [template.format(name=n) for n in labels for template in ("a photo of {name}", "a scene showing {name}")]
        vectors = self.encode(prompts).reshape(len(labels), 2, -1).mean(1)
        self.class_vectors = F.normalize(vectors, dim=1)
        self.labels = labels

    @torch.inference_mode()
    def encode(self, texts):
        outputs = []
        for start in range(0, len(texts), 128):
            tokens = self.tokenizer(texts[start:start + 128], padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            features = self.clip.text_projection(self.clip.text_model(**tokens, return_dict=True).pooler_output)
            outputs.append(F.normalize(features.float(), dim=1).cpu())
        return torch.cat(outputs) if outputs else torch.empty((0, 512))

    @torch.inference_mode()
    def generate(self, images):
        inputs = self.processor(text=[PROMPT] * len(images), images=images, return_tensors="pt", padding=True)
        moved = {k: v.to(device=self.device, dtype=torch.float32 if k == "pixel_values" else v.dtype)
                 for k, v in inputs.items() if torch.is_tensor(v)}
        generated = self.model.generate(**moved, max_new_tokens=self.settings["max_new_tokens"], num_beams=1, do_sample=False)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)
        if len(decoded) != len(images):
            raise ValueError("Florence output count mismatch")
        result = []
        for text, image in zip(decoded, images):
            parsed = self.processor.post_process_generation(text, task=PROMPT, image_size=image.size)
            if not isinstance(parsed, dict) or not isinstance(parsed.get(PROMPT), str):
                raise ValueError("Florence did not return the requested caption field")
            result.append(parsed[PROMPT].strip())
        return result

    def audit(self, raw_captions):
        rows = []
        for raw in raw_captions:
            final, valid, matches = clean_caption(raw, self.terms)
            rows.append({"raw_caption": raw, "final_caption": final, "caption_valid": valid,
                         "leakage_detected": bool(matches), "lexical_leakage_detected": bool(matches)})
        vectors = self.encode([r["final_caption"] for r in rows])
        similarities = vectors @ self.class_vectors.T
        scores, indices = similarities.max(1)
        for row, score, index in zip(rows, scores, indices):
            row.update(semantic_similarity_score=float(score), semantic_nearest_class=self.labels[int(index)],
                       semantic_similarity_flag=float(score) >= self.settings["semantic_audit_threshold"], caption_policy=POLICY)
        mask = torch.tensor([r["caption_valid"] for r in rows], dtype=torch.bool)
        vectors[~mask] = 0
        return rows, vectors, mask


def load_csv(path):
    if not Path(path).is_file():
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("Existing captions.csv has another schema; do not append or silently reinterpret it")
        rows = list(reader)
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate caption image IDs")
    return {r["image_id"]: r for r in rows}


def truth(value):
    return str(value).lower() in {"true", "1", "yes"}


def caption_protocol(payload):
    return {"schema": 1, "policy": POLICY, "settings": payload["settings"], "assets": payload["asset_identity"],
            "classes": payload["classes"], "records": [{k: r[k] for k in ("image_id", "split", "image_sha256")} for r in payload["records"]],
            "source_sha256": visual.file_sha256(__file__)}


def worker(payload):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    mode = payload["mode"]
    engine = CaptionEngine(payload)
    if mode == "probe":
        image = visual.read_rgb(payload["image"])
        audited, vectors, valid = engine.audit(engine.generate([image]))
        torch.save({"rows": audited, "embeddings": vectors, "valid": valid}, payload["result"])
        return
    if mode == "predict":
        image = visual.read_rgb(payload["image"])
        rows, vectors, valid = engine.audit(engine.generate([image]))
        torch.save({"rows": rows, "embeddings": vectors, "valid": valid}, payload["result"])
        return
    if mode != "prepare":
        raise ValueError("Unknown caption worker mode")
    path = Path(payload["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol = caption_protocol(payload)
    protocol_path = path.parent / "caption_protocol.json"
    if protocol_path.is_file() and visual.read_json(protocol_path) != protocol:
        raise ValueError("Caption resume protocol changed; use a fresh output directory")
    if path.is_file() and not protocol_path.is_file():
        raise ValueError("Unregistered captions.csv; this notebook does not silently import external/past-policy CSVs")
    visual.write_json(protocol_path, protocol)
    existing = load_csv(path)
    expected = {r["image_id"]: r for r in payload["records"]}
    if set(existing) - set(expected):
        raise ValueError("Cached captions contain images outside this SUN397 split")
    for image_id, row in existing.items():
        if any(row[k] != expected[image_id][k] for k in ("split", "image_sha256")) or row["caption_policy"] != POLICY:
            raise ValueError("Cached caption image binding/policy mismatch")
    todo = [r for r in payload["records"] if r["image_id"] not in existing]
    batch_size = int(payload["settings"]["batch_size"])
    if batch_size < 1 or payload["settings"]["max_new_tokens"] < 1:
        raise ValueError("Invalid caption batch/token budget")
    offset = 0
    print(f"SUN397 captions: cached={len(existing)}, remaining={len(todo)}, output={path}", flush=True)
    while offset < len(todo):
        batch = todo[offset:offset + batch_size]
        try:
            images = [visual.read_rgb(r["image_path"]) for r in batch]
            audited, _, _ = engine.audit(engine.generate(images))
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            print(f"Caption batch reduced to {batch_size}", flush=True)
            continue
        header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if header:
                writer.writeheader()
            for record, row in zip(batch, audited):
                writer.writerow({**{k: record[k] for k in ("image_id", "split", "image_sha256")}, **row,
                    "captioner_assets_sha256": payload["asset_identity"]["florence_sha256"],
                    "clip_assets_sha256": payload["asset_identity"]["clip_sha256"],
                    "vlm_model": "microsoft/Florence-2-base", "prompt_version": "florence_prefix_greedy_v1"})
            handle.flush()
            os.fsync(handle.fileno())
        offset += len(batch)
        print(f"Caption progress {len(existing) + offset}/{len(expected)}", flush=True)
    completed = load_csv(path)
    ordered = [completed[r["image_id"]] for r in payload["records"]]
    masks = torch.tensor([truth(r["caption_valid"]) for r in ordered], dtype=torch.bool)
    vectors = engine.encode([r["final_caption"] for r in ordered])
    vectors[~masks] = 0
    if not masks.any():
        raise ValueError("No usable caption remains. Inspect caption audit before training; do not tune using test metrics")
    result = {"embeddings": vectors, "valid": masks, "image_ids": [r["image_id"] for r in ordered],
              "csv_sha256": visual.file_sha256(path), "policy": POLICY,
              "clip_sha256": payload["asset_identity"]["clip_sha256"], "captioner_sha256": payload["asset_identity"]["florence_sha256"]}
    torch.save(result, payload["result"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    worker(visual.read_json(args.request))


if __name__ == "__main__":
    main()
