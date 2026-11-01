"""Pixel-only ISIC captions and frozen CLIP text exports, in isolated workers."""
from __future__ import annotations

import csv
import gc
import hashlib
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

from dtd_caption_finetune import atomic_csv, atomic_json, checkpoint_asset_identity, file_sha256, object_hash, read_csv_rows
from dtd_caption_screening import clean_dtd_caption
from dtd_qwen_caption import (
    MODEL_IDS, FrozenQwenCaptioner, available_rows, cache_path, cached_row, caption_messages,
    check_worker_runtime, local_model_configs, lock_protocol, qwen_asset_identity, resolve_clip_model,
    resolve_qwen_model, run_worker, runtime_versions, safetensors_parameter_count,
    validate_qwen_folder, weight_files, worker_environment,
)


METHOD = "qwen3_vl_8b_instruct"
POLICY = "isic_visible_attributes_v1"
CAPTION_PROMPT = (
    "Describe the visible pigmented skin region in 5 to 8 English words. "
    "Mention only visible colors, pigmentation variation, outline, border appearance or surface details. "
    "Do not give a diagnosis, disease name, malignancy assessment, treatment, measurements or hidden facts. "
    "Do not output a segmentation mask, coordinates or instructions. "
    "Start directly with descriptive words, without an article or image introduction. "
    "Output only the short visual description."
)
CLINICAL_PATTERN = r"\b(?:melanoma|melanomas|melanocytic|nevus|nevi|naevus|naevi|benign|malignant|malignancy|cancer|cancerous|carcinoma|keratosis|keratoses|diagnosis|diagnosed)\b"
COLUMNS = ["image_id", "split", "raw_caption", "final_caption", "caption_valid", "leakage_detected",
           "clinical_term_detected", "word_count", "source_long_caption", "generation_token_limit_reached",
           "vlm_model", "prompt_version", "caption_input_policy", "captioner_fingerprint", "retry_count"]


def caption_policy(response, token_limit=False):
    caption = clean_dtd_caption(response)
    count = len(re.findall(r"[A-Za-z0-9]+", caption))
    clinical = bool(re.search(CLINICAL_PATTERN, str(response), re.I))
    valid = 5 <= count <= 8 and not clinical and not token_limit
    return caption, valid, clinical, count


class IsicQwenCaptioner(FrozenQwenCaptioner):
    def generate(self, images):
        import torch
        conversations = [[message] for message in caption_messages(images, prompt=CAPTION_PROMPT)]
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
            raise ValueError("Qwen response count mismatch")
        return list(zip(texts, limits))


def validate_caption_records(records):
    if not records or len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Caption image IDs must be nonempty and unique across splits")
    for row in records:
        if set(row) != {"image_id", "image_path", "image_sha256", "split"}:
            raise ValueError("Caption worker accepts pixels/identity only, never masks, class labels or clinical metadata")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unexpected split")
        if file_sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['image_id']}")


def make_generation_row(record, response, token_limit, identity):
    caption, valid, clinical, count = caption_policy(response, token_limit)
    return {"image_id": record["image_id"], "split": record["split"], "raw_caption": caption,
            "final_caption": caption, "caption_valid": valid, "leakage_detected": clinical,
            "clinical_term_detected": clinical, "word_count": count, "source_long_caption": response,
            "generation_token_limit_reached": token_limit, "vlm_model": MODEL_IDS[METHOD],
            "prompt_version": POLICY, "caption_input_policy": POLICY,
            "captioner_fingerprint": identity, "retry_count": 0}


def generate_isic_records(payload, identity, factory=IsicQwenCaptioner):
    import torch
    from PIL import Image, ImageOps
    from tqdm.auto import tqdm
    records, settings = payload["records"], payload["settings"]
    directory = Path(payload["output"])
    cache = directory / "generated"
    todo = [row for row in records if cached_row(cache, row, identity) is None]
    if payload.get("max_new_images", 0):
        todo = todo[:payload["max_new_images"]]
    print(f"ISIC captions: cached={len(records) - sum(cached_row(cache, r, identity) is None for r in records)}, queued={len(todo)}", flush=True)
    if todo:
        captioner = factory(payload["model_path"], settings)
        offset, batch_size = 0, settings["batch_size"]
        progress = tqdm(total=len(todo), desc="ISIC Qwen3-VL-8B", unit="image")
        try:
            while offset < len(todo):
                group = todo[offset:offset + batch_size]
                images = []
                try:
                    for record in group:
                        if file_sha256(record["image_path"]) != record["image_sha256"]:
                            raise ValueError("Image changed after hashing")
                        with Image.open(record["image_path"]) as source:
                            images.append(ImageOps.exif_transpose(source).convert("RGB"))
                    responses = captioner.generate(images)
                    if len(responses) != len(group):
                        raise ValueError("Caption batch is incomplete")
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise RuntimeError("Qwen OOM at batch 1; free GPU memory or use a larger GPU. Committed captions are resumable") from None
                    batch_size = max(1, batch_size // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                finally:
                    for image in images:
                        image.close()
                for record, (response, limited) in zip(group, responses):
                    row = make_generation_row(record, response, limited, identity)
                    atomic_json(cache_path(cache, record["image_id"]), {"identity": identity, "row": row})
                offset += len(group)
                progress.update(len(group))
        finally:
            progress.close()
            del captioner
            gc.collect()
            torch.cuda.empty_cache()
    rows = available_rows(cache, records, identity)
    complete = len(rows) == len(records)
    path = directory / ("captions.csv" if complete else "captions_partial.csv")
    atomic_csv(path, rows, COLUMNS)
    status = {"complete": complete, "rows": len(rows), "expected": len(records), "identity": identity,
              "csv_path": str(path.resolve()), "csv_sha256": file_sha256(path),
              "valid_by_split": {role: sum(bool(row["caption_valid"]) for row in rows if row["split"] == role)
                                 for role in ("train", "val", "test")}}
    atomic_json(directory / "status.json", status)
    print(status, flush=True)
    return status


def validated_export(payload):
    directory = Path(payload["output"])
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if not status["complete"]:
        raise RuntimeError("Caption smoke test/partial output cannot train segmentation; set MAX_NEW_CAPTION_IMAGES=0 and resume")
    path = directory / "captions.csv"
    if file_sha256(path) != status["csv_sha256"]:
        raise ValueError("Caption CSV changed after generation")
    rows = read_csv_rows(path, COLUMNS)
    expected = [(r["image_id"], r["split"]) for r in payload["records"]]
    if [(r["image_id"], r["split"]) for r in rows] != expected:
        raise ValueError("Wrong/missing/duplicate image IDs or split order in captions.csv")
    for row in rows:
        clean, valid, clinical, count = caption_policy(row["source_long_caption"], str(row["generation_token_limit_reached"]).lower() == "true")
        if row["final_caption"] != clean or row["caption_valid"].lower() != str(valid).lower() or int(row["word_count"]) != count:
            raise ValueError("Caption policy fields do not match their original Qwen response")
        if row["captioner_fingerprint"] != status["identity"] or row["caption_input_policy"] != POLICY or row["vlm_model"] != MODEL_IDS[METHOD]:
            raise ValueError("Wrong caption generation provenance")
    if not any(row["split"] == "train" and row["caption_valid"].lower() == "true" for row in rows):
        raise ValueError("No usable training captions; inspect CSV and policy, do not force masks/gates on")
    return rows, status


def validate_text_cache(cache, rows, csv_sha256, identity):
    import torch
    expected_valid = torch.tensor([row["caption_valid"].lower() == "true" for row in rows], dtype=torch.bool)
    vectors, valid = cache["embeddings"], cache["valid"]
    if (cache["identity"] != identity or cache["image_ids"] != [row["image_id"] for row in rows]
            or cache["caption_csv_sha256"] != csv_sha256 or vectors.ndim != 2 or len(vectors) != len(rows)
            or valid.dtype != torch.bool or not torch.equal(valid.cpu(), expected_valid)):
        raise ValueError("Wrong CLIP text cache identity/IDs/shape/masks")
    if not torch.isfinite(vectors).all() or torch.count_nonzero(vectors[~valid]):
        raise ValueError("Invalid captions must have finite zero embeddings")
    if valid.any() and not torch.allclose(vectors[valid].float().norm(dim=1), torch.ones(int(valid.sum())), atol=1e-4):
        raise ValueError("Valid CLIP caption vectors must have unit norm")


def encode_isic_texts(payload):
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    rows, status = validated_export(payload)
    output = Path(payload["output"])
    protocol = {"schema": 1, "caption_csv_sha256": status["csv_sha256"],
                "clip_assets": qwen_asset_identity(payload["clip_path"]), "policy": POLICY,
                "runtime": runtime_versions(), "code_sha256": payload["code_sha256"]}
    identity = lock_protocol(output / "text_cache" / "protocol.json", protocol)
    target = output / "text_cache" / "clip_text.pt"
    if target.is_file():
        cached = torch.load(target, map_location="cpu", weights_only=True)
        validate_text_cache(cached, rows, status["csv_sha256"], identity)
        atomic_json(output / "text_cache" / "status.json", {"identity": identity, "sha256": file_sha256(target),
                    "valid": int(cached["valid"].sum()), "total": len(rows), "shape": list(cached["embeddings"].shape)})
        print("Reusing CLIP text cache:", target, flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(payload["clip_path"], local_files_only=True)
    model, info = CLIPTextModelWithProjection.from_pretrained(payload["clip_path"], local_files_only=True,
                                                            output_loading_info=True)
    missing = [key for key in info.get("missing_keys", []) if not key.endswith("position_ids")]
    if missing or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete CLIP text weights: {missing}")
    model = model.to("cuda:0").eval().requires_grad_(False)
    valid = torch.tensor([row["caption_valid"].lower() == "true" for row in rows], dtype=torch.bool)
    vectors = torch.zeros((len(rows), model.config.projection_dim), dtype=torch.float32)
    positions = valid.nonzero().flatten().tolist()
    for start in range(0, len(positions), 128):
        group = positions[start:start + 128]
        tokens = tokenizer([rows[i]["final_caption"] for i in group], padding=True, truncation=True,
                           max_length=model.config.max_position_embeddings, return_tensors="pt")
        tokens = {key: value.to("cuda:0") for key, value in tokens.items()}
        with torch.inference_mode():
            values = torch.nn.functional.normalize(model(**tokens).text_embeds.float(), dim=1)
        vectors[group] = values.cpu()
    temporary = target.with_suffix(".pt.tmp")
    torch.save({"identity": identity, "image_ids": [row["image_id"] for row in rows],
                "embeddings": vectors, "valid": valid, "caption_csv_sha256": status["csv_sha256"]}, temporary)
    temporary.replace(target)
    atomic_json(output / "text_cache" / "status.json", {"identity": identity, "sha256": file_sha256(target),
                "valid": int(valid.sum()), "total": len(rows), "shape": list(vectors.shape)})
    print("CLIP vectors:", tuple(vectors.shape), "valid:", int(valid.sum()), flush=True)


def isic_worker_main(request_path, phase):
    import torch
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    validate_caption_records(payload["records"])
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU")
    settings = payload["settings"]
    if type(payload.get("max_new_images", 0)) is not int or payload.get("max_new_images", 0) < 0:
        raise ValueError("max_new_images must be a nonnegative integer")
    for key in ("batch_size", "max_new_tokens", "min_pixels", "max_pixels"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["min_pixels"] > settings["max_pixels"] or settings["dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("Invalid caption pixel budget/dtype")
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    if phase == "generate":
        path, count, _ = validate_qwen_folder(payload["model_path"], METHOD)
        protocol = {"schema": 1, "method": METHOD, "model_assets": qwen_asset_identity(path),
                    "parameters": count, "prompt": CAPTION_PROMPT, "policy": POLICY,
                    "images": [{key: row[key] for key in ("image_id", "split", "image_sha256")} for row in payload["records"]],
                    "settings": settings, "runtime": runtime_versions(), "code_sha256": payload["code_sha256"]}
        identity = lock_protocol(Path(payload["output"]) / "generation_protocol" / "protocol.json", protocol)
        if not (Path(payload["output"]) / "status.json").exists() and (Path(payload["output"]) / "captions.csv").exists():
            raise ValueError("Unregistered captions.csv; use a fresh output root")
        generate_isic_records(payload, identity)
    elif phase == "encode":
        encode_isic_texts(payload)
    else:
        raise ValueError("Phase must be generate or encode")
