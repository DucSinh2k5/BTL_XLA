"""Offline, resumable Qwen3-VL caption exports for the existing DTD ablations."""
from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

from dtd_caption_finetune import atomic_csv, atomic_json, checkpoint_asset_identity, file_sha256, load_caption_manifest, object_hash, read_csv_rows
from dtd_caption_screening import clean_dtd_caption
from dtd_grounded_caption import ALIASES, bool_value, load_rgb, normalize, term_hits, valid_short


METHOD_VERSION = "qwen3vl_dtd_short_caption_v1"
MODEL_IDS = {
    "qwen3_vl_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
}
CAPTION_PROMPT = (
    "Describe only the visible surface texture in 5 to 8 English words. "
    "Focus on visible colors, local shapes, spacing, repetition, and surface structure. "
    "Do not identify the texture category or guess the object, use, location, or hidden material. "
    "Start directly with descriptive words, not an article or an image introduction. "
    "Output only one short description, without a heading, explanation, or quotation marks."
)
COLUMNS = [
    "image_id", "split", "raw_caption", "final_caption", "caption_valid", "leakage_detected",
    "lexical_leakage_detected", "semantic_leakage_detected", "semantic_leakage_score",
    "semantic_leakage_class", "retry_count", "vlm_model", "prompt_version",
    "caption_input_policy", "semantic_audit_status", "captioner_fingerprint", "generation_method",
    "source_long_caption", "source_word_count", "word_count", "cleanup_changed_text",
    "alias_hit_diagnostic", "generation_token_limit_reached", "effective_batch_size", "audit_fingerprint",
]


def local_model_configs(roots):
    seen = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in {
                "images", ".git", "__pycache__", "node_modules", ".cache"})
            if len(Path(current).relative_to(root).parts) >= 9:
                dirs[:] = []
            if "config.json" not in files:
                continue
            path = Path(current).resolve()
            if str(path) not in seen:
                seen.add(str(path))
                yield path


def weight_files(path):
    path = Path(path).resolve()
    index = path / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        if not names:
            raise ValueError(f"Empty safetensors weight index: {index}")
        if any(not isinstance(name, str) or "/" in name or "\\" in name or Path(name).is_absolute() for name in names):
            raise ValueError(f"Weight index must reference shard filenames inside its model folder: {index}")
        files = [path / name for name in sorted(names)]
    else:
        files = [path / "model.safetensors"]
    for file in files:
        # HF snapshot files may be read-only symlinks into a sibling blobs directory.
        if not file.is_file() or file.suffix != ".safetensors":
            raise FileNotFoundError(f"Missing/unsafe local safetensors shard: {file}")
    return files


def safetensors_parameter_count(files):
    # Read headers only, not tensor contents; verify the attached model's actual scale.
    count = 0
    for path in files:
        with Path(path).open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            length = struct.unpack("<Q", raw_length)[0]
            if not 2 <= length <= 64 * 1024 * 1024 or length + 8 > Path(path).stat().st_size:
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(handle.read(length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor.get("dtype") not in {"F16", "BF16", "F32"}:
                raise ValueError("Use original floating-point weights, not quantized/FP8 weights, for this comparison")
            shape = tensor.get("shape", [])
            if not all(type(size) is int and size >= 0 for size in shape):
                raise ValueError(f"Invalid tensor shape: {path}: {name}")
            count += math.prod(shape)
    return count


def validate_qwen_folder(path, method):
    path = Path(path).expanduser().resolve()
    if method not in MODEL_IDS:
        raise ValueError(f"Unknown Qwen method: {method}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config"):
        raise ValueError("Expected original dense Qwen3-VL, not Qwen2.5, pure-text Qwen, MoE or a quantized model")
    if "thinking" in str(path).lower():
        raise ValueError("Attach the Instruct checkpoint, not Thinking")
    for name in ("tokenizer_config.json", "preprocessor_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing local processor/tokenizer asset: {path / name}")
    if not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {path}")
    tokenizer = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    template_json = path / "chat_template.json"
    json_template = json.loads(template_json.read_text(encoding="utf-8")).get("chat_template") if template_json.is_file() else None
    if not tokenizer.get("chat_template") and not (path / "chat_template.jinja").is_file() and not json_template:
        raise FileNotFoundError(f"Missing Qwen chat template in {path}")
    files = weight_files(path)
    count = safetensors_parameter_count(files)
    low, high = (7_000_000_000, 11_000_000_000) if "8b" in method else (28_000_000_000, 38_000_000_000)
    if not low <= count <= high:
        raise ValueError(f"Wrong checkpoint size for {MODEL_IDS[method]}: {count:,} stored parameters")
    return path, count, sum(file.stat().st_size for file in files)


def resolve_qwen_model(reference, method, roots):
    if str(reference).strip():
        path, _, _ = validate_qwen_folder(reference, method)
        return path
    needle = MODEL_IDS[method].split("/")[-1].lower()
    matches = [path for path in local_model_configs(roots) if needle in str(path).lower()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Set the exact local path for {MODEL_IDS[method]}. Found {len(matches)} named matches. "
            "Attach its complete Hugging Face folder as Kaggle Input; Florence weights are not Qwen weights. "
            "No download is attempted.")
    return validate_qwen_folder(matches[0], method)[0]


def resolve_clip_model(reference, roots):
    def is_clip(path):
        try:
            return json.loads((path / "config.json").read_text(encoding="utf-8")).get("model_type") == "clip"
        except (OSError, ValueError):
            return False
    if str(reference).strip() and Path(reference).is_dir():
        path = Path(reference).resolve()
        if not is_clip(path):
            raise ValueError(f"Not a full local CLIP checkpoint: {path}")
        return path
    matches = [path for path in local_model_configs(roots) if is_clip(path)]
    if len(matches) != 1:
        raise FileNotFoundError("Set CLIP_MODEL_LOCAL_PATH to the exact full CLIP ViT-B/32 folder")
    return matches[0]


def qwen_asset_identity(path):
    result = checkpoint_asset_identity(path)
    for template in sorted(Path(path).glob("*.jinja")):
        result[template.name] = file_sha256(template)
    return result


def fingerprint_records(records):
    return [{"image_id": row["image_id"], "split": row["split"],
             "sha256": file_sha256(row["image_path"])} for row in records]


def runtime_versions():
    result = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "tokenizers", "Pillow", "numpy"):
        result[name] = importlib.metadata.version(name)
    return result


def validate_settings(settings):
    for key in ("batch_size", "max_new_tokens", "min_pixels", "max_pixels", "audit_batch_size"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["min_pixels"] > settings["max_pixels"]:
        raise ValueError("min_pixels must not exceed max_pixels")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if settings["dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("Use the same bfloat16 or float16 dtype for both captioners")
    if not math.isfinite(settings["semantic_threshold"]) or not -1 <= settings["semantic_threshold"] <= 1:
        raise ValueError("semantic_threshold must be finite and in [-1, 1]")


def lock_protocol(path, protocol):
    path = Path(path)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError(f"Changed images/model/prompt/runtime/settings. Use a new output directory: {path.parent}")
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError(f"Refusing to adopt cached output without its protocol: {path.parent}")
        atomic_json(path, protocol)
    return object_hash(protocol)


def cache_path(directory, image_id):
    return Path(directory) / (object_hash(image_id) + ".json")


def cached_row(directory, record, identity):
    path = cache_path(directory, record["image_id"])
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("identity") != identity or result.get("row", {}).get("image_id") != record["image_id"]
            or result["row"].get("split") != record["split"]):
        raise ValueError(f"Wrong cached image/protocol: {path}")
    return result["row"]


def generation_row(record, response, method, identity, names, token_limit=False, batch_size=1):
    text = clean_dtd_caption(response)
    lexical = bool(term_hits(text, names))
    return {
        "image_id": record["image_id"], "split": record["split"], "raw_caption": text, "final_caption": text,
        "caption_valid": valid_short(text, names), "leakage_detected": lexical,
        "lexical_leakage_detected": lexical, "semantic_leakage_detected": False,
        "semantic_leakage_score": "", "semantic_leakage_class": "", "retry_count": 0,
        "vlm_model": MODEL_IDS[method], "prompt_version": METHOD_VERSION,
        "caption_input_policy": "dtd_lexical_v2", "semantic_audit_status": "pending",
        "captioner_fingerprint": identity, "generation_method": method,
        "source_long_caption": str(response), "source_word_count": len(normalize(response).split()),
        "word_count": len(normalize(text).split()), "cleanup_changed_text": str(response).strip() != text,
        "alias_hit_diagnostic": bool(term_hits(text, names, aliases=True)),
        "generation_token_limit_reached": bool(token_limit), "effective_batch_size": batch_size,
        "audit_fingerprint": "",
    }


def audited_row(row, score, class_name, identity, threshold):
    if not math.isfinite(score) or not -1.00001 <= score <= 1.00001 or not class_name:
        raise ValueError("Invalid CLIP class-audit result; do not fill missing scores with zero")
    result = dict(row)
    result.update(semantic_leakage_score=round(score, 6), semantic_leakage_class=class_name,
                  semantic_leakage_detected=score >= threshold,
                  semantic_audit_status="diagnostic_only", audit_fingerprint=identity)
    # dtd_lexical_v2 deliberately does not hard-reject text using class-CLIP similarity.
    return result


def caption_messages(images, prompt=CAPTION_PROMPT):
    return [{"role": "user", "content": [{"type": "image", "image": image},
            {"type": "text", "text": prompt}]} for image in images]


class FrozenQwenCaptioner:
    def __init__(self, path, settings):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.device = torch.device("cuda:0")
        self.settings = settings
        dtype = getattr(torch, settings["dtype"])
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU lacks BF16 support. Set dtype=float16 for BOTH models in a new output directory")
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
            min_pixels=settings["min_pixels"], max_pixels=settings["max_pixels"], use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.model, info = Qwen3VLForConditionalGeneration.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, dtype=dtype, device_map={"": str(self.device)},
            attn_implementation="sdpa", low_cpu_mem_usage=True, output_loading_info=True)
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise ValueError(f"Incomplete Qwen learned weights: {info}")
        self.model.eval().requires_grad_(False)

    def generate(self, images):
        import torch
        conversations = [[message] for message in caption_messages(images)]
        prompts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt",
            min_pixels=self.settings["min_pixels"], max_pixels=self.settings["max_pixels"])
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.settings["max_new_tokens"],
                do_sample=False, num_beams=1, use_cache=True, temperature=None, top_p=None, top_k=None)
        suffix = output[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        limits = [len(tokens) >= self.settings["max_new_tokens"] and not any(token in eos for token in tokens)
                  for tokens in suffix.cpu().tolist()]
        if len(texts) != len(images):
            raise ValueError("Qwen output count differs from input image count")
        return list(zip(texts, limits))


class ClipClassAuditor:
    def __init__(self, path, names):
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model, info = CLIPTextModelWithProjection.from_pretrained(str(path), local_files_only=True,
            output_loading_info=True)
        missing = [key for key in info.get("missing_keys", []) if not key.endswith("position_ids")]
        if missing or info.get("mismatched_keys"):
            raise ValueError(f"Incomplete CLIP text tower/projection: {missing}")
        self.model = self.model.to("cuda:0").eval().requires_grad_(False)
        self.names = names
        prompts = [template.format(name=name.replace("_", " ").replace("-", " ")) for name in names
                   for template in ("a photo of {name}", "a scene showing {name}")]
        self.class_vectors = torch.nn.functional.normalize(self.encode(prompts).reshape(len(names), 2, -1).mean(1), dim=1)

    def encode(self, texts):
        import torch
        tokens = self.tokenizer(list(texts), padding=True, truncation=True,
            max_length=self.model.config.max_position_embeddings, return_tensors="pt")
        tokens = {key: value.to("cuda:0") for key, value in tokens.items()}
        with torch.inference_mode():
            features = self.model(**tokens).text_embeds
        return torch.nn.functional.normalize(features.float(), dim=1)

    def score(self, texts):
        scores, indices = (self.encode(texts) @ self.class_vectors.T).max(dim=1)
        return [(float(score), self.names[index]) for score, index in zip(scores.cpu(), indices.cpu().tolist())]


def available_rows(directory, records, identity):
    return [row for record in records if (row := cached_row(directory, record, identity)) is not None]


def generate_records(payload, identity, captioner_factory=FrozenQwenCaptioner):
    import torch
    from tqdm.auto import tqdm
    output = Path(payload["output"])
    cache = output / "generated"
    records, settings = payload["records"], payload["settings"]
    todo = [row for row in records if cached_row(cache, row, identity) is None]
    cached_count = len(records) - len(todo)
    limit = payload.get("max_new_images", 0)
    if limit:
        todo = todo[:limit]
    started = time.perf_counter()
    print(f"{payload['method']}: cached={cached_count}, queued={len(todo)}", flush=True)
    if todo:
        model = captioner_factory(payload["model_path"], settings)
        current, offset = settings["batch_size"], 0
        progress = tqdm(total=len(todo), desc=payload["method"], unit="image")
        try:
            while offset < len(todo):
                group = todo[offset:offset + current]
                images = []
                try:
                    for row in group:
                        if file_sha256(row["image_path"]) != row["image_sha256"]:
                            raise ValueError(f"Image changed after manifest hashing: {row['image_id']}")
                        images.append(load_rgb(row["image_path"]))
                    responses = model.generate(images)
                    if len(responses) != len(group):
                        raise ValueError("Incorrect caption batch length; batch not committed")
                except torch.cuda.OutOfMemoryError:
                    if current == 1:
                        raise RuntimeError("Qwen OOM at batch 1. Free GPU memory or use a larger GPU; completed images are cached") from None
                    current = max(1, current // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"CUDA OOM: generation batch reduced to {current}; no image skipped", flush=True)
                    continue
                finally:
                    for image in images:
                        image.close()
                for record, (text, token_limit) in zip(group, responses):
                    row = generation_row(record, text, payload["method"], identity,
                        payload["class_names"], token_limit, len(group))
                    atomic_json(cache_path(cache, record["image_id"]), {"identity": identity, "row": row})
                offset += len(group)
                progress.update(len(group))
        finally:
            progress.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    rows = available_rows(cache, records, identity)
    atomic_csv(output / "captions_raw_partial.csv", rows, COLUMNS)
    atomic_json(output / "generation_status.json", {"identity": identity, "rows": len(rows),
                "expected": len(records), "complete": len(rows) == len(records),
                "new_images_this_invocation": len(todo), "elapsed_seconds_this_invocation": time.perf_counter() - started})


def audit_records(payload, generation_identity, audit_identity, auditor_factory=ClipClassAuditor):
    from tqdm.auto import tqdm
    output, records = Path(payload["output"]), payload["records"]
    raw_rows = available_rows(output / "generated", records, generation_identity)
    by_id = {row["image_id"]: row for row in raw_rows}
    cache = output / "audited"
    pending = [row for row in records if row["image_id"] in by_id and cached_row(cache, row, audit_identity) is None]
    if pending:
        auditor = auditor_factory(payload["clip_path"], payload["class_names"])
        size = payload["settings"]["audit_batch_size"]
        for start in tqdm(range(0, len(pending), size), desc="CLIP diagnostic audit"):
            group = pending[start:start + size]
            scores = auditor.score([by_id[row["image_id"]]["final_caption"] for row in group])
            if len(scores) != len(group):
                raise ValueError("CLIP audit batch length mismatch")
            for record, (score, name) in zip(group, scores):
                row = audited_row(by_id[record["image_id"]], score, name,
                    audit_identity, payload["settings"]["semantic_threshold"])
                atomic_json(cache_path(cache, record["image_id"]), {"identity": audit_identity, "row": row})
    rows = available_rows(cache, records, audit_identity)
    complete = len(rows) == len(records)
    destination = output / ("captions.csv" if complete else "captions_audited_partial.csv")
    atomic_csv(destination, rows, COLUMNS)
    status = {"complete": complete, "rows": len(rows), "expected": len(records),
              "generation_identity": generation_identity, "audit_identity": audit_identity,
              "csv_path": str(destination.resolve()), "csv_sha256": file_sha256(destination)}
    atomic_json(output / "status.json", status)
    print(f"Audited rows={len(rows)}/{len(records)}; complete={complete}; CSV={destination}", flush=True)
    return status


def worker_main(request_path, phase):
    import torch
    from transformers import Qwen3VLForConditionalGeneration  # Fail before loading weights on an old runtime.
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    validate_settings(payload["settings"])
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU for Qwen generation and CLIP audit")
    torch.manual_seed(payload["settings"]["seed"])
    torch.cuda.manual_seed_all(payload["settings"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    output = Path(payload["output"])
    if phase == "generate":
        print("Hashing complete Qwen assets; large checkpoint hashing can take several minutes", flush=True)
        path, params, byte_count = validate_qwen_folder(payload["model_path"], payload["method"])
        protocol = {"schema": 1, "method": payload["method"], "model_id": MODEL_IDS[payload["method"]],
            "model_assets": qwen_asset_identity(path), "stored_parameter_count": params,
            "stored_weight_bytes": byte_count,
            "prompt": CAPTION_PROMPT, "prompt_version": METHOD_VERSION,
            "dtd_split": payload["dtd_split"], "class_names": payload["class_names"],
            "records": [{key: row[key] for key in ("image_id", "split", "image_sha256")} for row in payload["records"]],
            "settings": payload["settings"], "runtime": runtime_versions(), "code_sha256": payload["code_sha256"],
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
        identity = lock_protocol(output / "generation_protocol.json", protocol)
        free, total = torch.cuda.mem_get_info(0)
        print(f"Stored parameters={params:,}; GPU={total / 1024**3:.1f} GiB; free={free / 1024**3:.1f} GiB", flush=True)
        raw_weight_bytes = params * 2
        missing = any(cached_row(output / "generated", record, identity) is None for record in payload["records"])
        if missing and free < raw_weight_bytes * 1.08:
            raise RuntimeError(f"Insufficient free VRAM even for weights plus a small reserve (~{raw_weight_bytes / 1024**3:.1f} GiB weights). "
                               "32B normally needs a large-memory GPU. No quantization/CPU offload is silently enabled")
        generate_records(payload, identity)
    elif phase == "audit":
        generation = json.loads((output / "generation_protocol.json").read_text(encoding="utf-8"))
        expected_records = [{key: row[key] for key in ("image_id", "split", "image_sha256")} for row in payload["records"]]
        if (generation["method"] != payload["method"] or generation["settings"] != payload["settings"]
                or generation["class_names"] != payload["class_names"] or generation["records"] != expected_records
                or generation["code_sha256"] != payload["code_sha256"]):
            raise ValueError("Audit request does not match the locked generation protocol")
        generation_identity = object_hash(generation)
        protocol = {"schema": 1, "generation_identity": generation_identity,
                    "clip_assets": qwen_asset_identity(payload["clip_path"]),
                    "threshold": payload["settings"]["semantic_threshold"], "policy": "dtd_lexical_v2",
                    "dtype": "float32", "runtime": runtime_versions(), "code_sha256": payload["code_sha256"]}
        audit_identity = lock_protocol(output / "audit" / "protocol.json", protocol)
        status = audit_records(payload, generation_identity, audit_identity)
        atomic_json(output / "export_manifest.json", dict(status, model_id=MODEL_IDS[payload["method"]],
            caption_input_policy="dtd_lexical_v2", method=payload["method"], prompt_version=METHOD_VERSION))
    else:
        raise ValueError("Worker phase must be generate or audit")


def worker_environment(runtime_packages=""):
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if runtime_packages:
        path = Path(runtime_packages).expanduser().resolve()
        if not (path / "transformers").is_dir():
            raise FileNotFoundError("QWEN_RUNTIME_SITE_PACKAGES must contain transformers/ and compatible dependencies")
        env["PYTHONPATH"] = str(path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def check_worker_runtime(env):
    code = (
        "import json, torch, torchvision, transformers, accelerate, tokenizers; "
        "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, CLIPTextModelWithProjection; "
        "print(json.dumps({'transformers':transformers.__version__, 'torch':torch.__version__, "
        "'cuda':torch.cuda.is_available(), 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Qwen3-VL worker import failed. Attach a compatible Transformers >=4.57 runtime and its dependencies; "
            "the old Florence Transformers 4.49 wheels cannot run Qwen3-VL. No packages were installed.\n" + result.stderr[-4000:])
    print(result.stdout.strip())
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if not info["cuda"]:
        raise RuntimeError("Qwen worker imports passed but no CUDA GPU is available. Enable a GPU accelerator")
    return result.stdout.strip()


def run_worker(worker_path, payload_path, phase, env, log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(worker_path), str(payload_path), phase], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
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
        raise RuntimeError(f"{phase} worker exited with code {code}. See {log_path}; committed image caches are preserved")


def summarize_exports(method_outputs, expected_records):
    summaries = []
    exports = {}
    expected = [(row["image_id"], row["split"]) for row in expected_records]
    for method, output in method_outputs.items():
        status_path = Path(output) / "status.json"
        if not status_path.is_file():
            print(f"{method}: not run; no complete CSV yet")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status["complete"]:
            print(f"{method}: incomplete ({status['rows']}/{status['expected']}); resume before training")
            continue
        path = Path(status["csv_path"])
        if file_sha256(path) != status["csv_sha256"]:
            raise ValueError(f"Export CSV changed after audit: {path}")
        rows = read_csv_rows(path, COLUMNS)
        if [(row["image_id"], row["split"]) for row in rows] != expected:
            raise ValueError(f"Wrong row order/image IDs/split or incomplete CSV: {path}")
        if any(row["generation_method"] != method or row["semantic_audit_status"] != "diagnostic_only"
               or row["caption_input_policy"] != "dtd_lexical_v2" for row in rows):
            raise ValueError(f"Wrong caption export provenance: {path}")
        exports[method] = rows
        for role in ("train", "val", "test"):
            selected = [row for row in rows if row["split"] == role]
            valid = [row for row in selected if bool_value(row["caption_valid"])]
            summaries.append({"method": method, "split": role, "images": len(selected), "valid": len(valid),
                "valid_rate": len(valid) / len(selected), "unique_captions": len({r["final_caption"] for r in valid}),
                "direct_class_hits": sum(bool_value(r["lexical_leakage_detected"]) for r in selected),
                "alias_hits_diagnostic": sum(bool_value(r["alias_hit_diagnostic"]) for r in selected),
                "semantic_flags_diagnostic": sum(bool_value(r["semantic_leakage_detected"]) for r in selected),
                "token_limit_hits": sum(bool_value(r["generation_token_limit_reached"]) for r in selected),
                "csv_sha256": status["csv_sha256"], "csv_path": str(path)})
    return summaries, exports
