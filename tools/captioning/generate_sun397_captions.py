#!/usr/bin/env python3
"""Generate resumable SUN397 visual-attribute captions with local Florence-2.

Example (RTX PRO 6000):
  python generate_sun397_captions.py --dataset-root F:/data/SUN397/SUN397 \
      --train-partition F:/data/sun397_other/Training_01.txt \
      --test-partition F:/data/sun397_other/Testing_01.txt \
      --model F:/models/Florence-2-base --batch-size 64 --workers 8 \
      --output captions.csv

For offline Kaggle, attach local Florence-2 and CLIP folders as Inputs and pass
their paths with --local-files-only. The script makes no external API calls.
Lexical plus CLIP similarity screening reduces leakage risk but cannot prove
that every synonym or implied category has been removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DEFAULT_DATASET_ROOT = Path(os.getenv("SUN397_ROOT", "/kaggle/input/datasets/hiuphmnhtrung/sun397/SUN397/SUN397"))
DEFAULT_TRAIN_PARTITION = Path(os.getenv("SUN397_TRAIN_PARTITION", "/kaggle/input/datasets/hiuphmnhtrung/sun397/sun397_other/Training_01.txt"))
DEFAULT_TEST_PARTITION = Path(os.getenv("SUN397_TEST_PARTITION", "/kaggle/input/datasets/hiuphmnhtrung/sun397/sun397_other/Testing_01.txt"))
DEFAULT_OUTPUT = Path(os.getenv("CAPTIONS_OUTPUT", "outputs/mtacs_sun397_visual_text_context/captions.csv"))
DEFAULT_CLIP_MODEL = os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit")
MODEL_ID = "microsoft/Florence-2-base"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"
PROMPT_VERSION = "florence2_detailed_v2"
CSV_COLUMNS = [
    "image_id", "split", "raw_caption", "final_caption", "caption_valid",
    "leakage_detected", "lexical_leakage_detected", "semantic_leakage_detected",
    "semantic_leakage_score", "semantic_leakage_class", "retry_count", "vlm_model", "prompt_version",
]
REQUIRED_COLUMNS = {"image_id", "split", "final_caption", "caption_valid", "leakage_detected", "retry_count"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_phrase(text: Any) -> str:
    text = str(text).replace("_", " ").replace("/", " ").replace("-", " ").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_class_names(dataset_root: Path) -> List[str]:
    for path in (dataset_root / "ClassName.txt", dataset_root.parent / "ClassName.txt", dataset_root / "classes.txt", dataset_root.parent / "classes.txt"):
        if path.is_file():
            names = [line.strip().lstrip("/").replace("\\", "/") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if names:
                return names
    raise FileNotFoundError(f"ClassName.txt not found under {dataset_root} or its parent")


def build_forbidden_terms(class_names: Sequence[str], extra_terms: Iterable[str] = ()) -> List[str]:
    terms: Set[str] = set()
    for name in class_names:
        normalized = normalize_phrase(name)
        leaf = normalize_phrase(PurePosixPath(name.replace("\\", "/")).name)
        if normalized:
            terms.add(normalized)
        if leaf:
            terms.add(leaf)
    terms.update(filter(None, (normalize_phrase(term) for term in extra_terms)))
    return sorted(terms, key=lambda value: (-len(value.split()), -len(value), value))


def audit_caption(raw_text: str, forbidden_terms: Sequence[str]) -> Tuple[str, bool, bool, List[str]]:
    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", str(raw_text))
    caption = " ".join(words[:8])
    normalized = f" {normalize_phrase(caption)} "
    matched = [term for term in forbidden_terms if f" {term} " in normalized]
    count = len(normalized.split())
    return caption, 5 <= count <= 8 and not matched, bool(matched), matched


def read_partition(path: Path, split: str, dataset_root: Path, image_root: Path, class_to_id: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip().replace("\\", "/").lstrip("/")
            if not raw:
                continue
            parts = list(PurePosixPath(raw).parts)
            if parts and parts[0].lower() in {"sun397", "images"}:
                parts = parts[1:]
            if len(parts) < 2:
                continue
            rel_path = PurePosixPath(*parts).as_posix()
            class_path = PurePosixPath(*parts[:-1]).as_posix()
            if class_path not in class_to_id:
                continue
            candidates = (image_root.joinpath(*parts), dataset_root.joinpath(*parts))
            image_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if image_path is None:
                missing += 1
                continue
            rows.append({"image_id": rel_path, "image_path": image_path, "class_id": class_to_id[class_path], "split": split})
    if missing:
        print(f"Warning: {missing:,} files listed in {path.name} are missing")
    return rows


def split_train_val(rows: Sequence[Dict[str, Any]], fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0 < fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    by_class: Dict[int, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[int(row["class_id"])].append(index)
    rng = np.random.default_rng(seed)
    val_indices: Set[int] = set()
    for indices in by_class.values():
        if len(indices) > 1:
            count = min(max(1, int(round(len(indices) * fraction))), len(indices) - 1)
            val_indices.update(int(i) for i in rng.choice(indices, size=count, replace=False))
    train, val = [], []
    for i, row in enumerate(rows):
        item = dict(row)
        item["split"] = "val" if i in val_indices else "train"
        (val if i in val_indices else train).append(item)
    return train, val


def load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    for path in (args.dataset_root, args.train_partition, args.test_partition):
        if not path.exists():
            raise FileNotFoundError(f"Required SUN397 path not found: {path}")
    names = load_class_names(args.dataset_root)
    class_to_id = {name.lstrip("/"): i for i, name in enumerate(names)}
    train = read_partition(args.train_partition, "train", args.dataset_root, args.image_root, class_to_id)
    test = read_partition(args.test_partition, "test", args.dataset_root, args.image_root, class_to_id)
    if not train or not test:
        raise RuntimeError("Empty partition. Verify the image root and partition paths.")
    train, val = split_train_val(train, args.val_fraction, args.seed)
    records = train + val + test
    if len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Duplicate image IDs were found across SUN397 partitions")
    print(f"Images: train={len(train):,}, val={len(val):,}, test={len(test):,}, total={len(records):,}")
    return records, names


def load_existing(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(CSV_COLUMNS) - fields
        if missing:
            raise ValueError(
                f"Existing CSV predates the semantic audit and lacks {sorted(missing)}. "
                "Choose a new --output path or pass --force to regenerate; do not append new rows to the old schema."
            )
        return {row["image_id"]: row for row in reader if row.get("image_id")}


def append_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


class CaptionDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        try:
            with Image.open(row["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            return row, image, None
        except Exception as exc:
            return row, None, f"{type(exc).__name__}: {exc}"


def collate_images(items):
    return items


def _model_type(path: Path) -> Optional[str]:
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(config.get("architectures", [])).lower()
    if model_type:
        return model_type
    if "florence2" in architectures or "florence" in architectures:
        return "florence2"
    return None


def resolve_local_model(model_ref: str, expected_type: str, extra_roots: Sequence[Path] = ()) -> Optional[Path]:
    """Find a model folder in Kaggle Inputs, notebook outputs, or HF cache."""
    requested = Path(model_ref).expanduser()
    candidates: List[Path] = []
    if requested.is_dir():
        candidates.append(requested)
        candidates.extend(path.parent for path in requested.rglob("config.json"))

    roots = list(extra_roots)
    roots.extend([
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.home() / ".cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots",
        Path.home() / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots",
    ])
    visited: Set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        # Kaggle Inputs can contain many images; only inspect model configs to
        # a modest directory depth, which covers attached model/notebook outputs.
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                dirs[:] = []
                continue
            if depth >= 6:
                dirs[:] = []
            if "config.json" in files:
                key = str(current_path.resolve())
                if key not in visited:
                    visited.add(key)
                    candidates.append(current_path)

    expected = expected_type.lower()
    matching = [path for path in candidates if _model_type(path) == expected]
    if not matching:
        return None
    # Prefer an explicit path, then folder names that identify the requested model.
    requested_name = "florence" if expected == "florence2" else "clip"
    matching.sort(key=lambda path: (0 if path == requested else 1, 0 if requested_name in str(path).lower() else 1, len(str(path))))
    return matching[0]


def load_florence(model_id: str, device: torch.device, local_files_only: bool):
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError('Install compatible dependencies: pip install "transformers==4.49.0" accelerate einops timm') from exc
    import transformers
    version = tuple(int(piece) for piece in transformers.__version__.split(".")[:2])
    if version >= (4, 50):
        raise RuntimeError(f"Florence-2 remote code requires Transformers 4.49.x here; found {transformers.__version__}. Pin to 4.49.0 and restart the Python runtime.")
    local_model = resolve_local_model(model_id, "florence2")
    if local_model is not None:
        model_id = str(local_model)
        print(f"Using local Florence-2 model: {model_id}")
    elif local_files_only:
        raise FileNotFoundError(
            "No local Florence-2 model was found in /kaggle/input, /kaggle/working, "
            "or the Hugging Face cache. The shown Kaggle inputs include SUN397 and CLIP, "
            "but Florence-2 must also be present locally (attached Model/Dataset or notebook output)."
        )
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, local_files_only=local_files_only,
    ).to(device).eval()
    return model, processor, dtype


def generate_batch(model, processor, images, device, dtype, max_new_tokens: int) -> List[str]:
    inputs = processor(text=[TASK_PROMPT] * len(images), images=images, return_tensors="pt", padding=True)
    moved = {key: value.to(device=device, dtype=dtype if key == "pixel_values" else value.dtype, non_blocking=True)
             for key, value in inputs.items() if torch.is_tensor(value)}
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        generated = model.generate(**moved, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    decoded = processor.batch_decode(generated, skip_special_tokens=False)
    outputs = []
    for text, image in zip(decoded, images):
        parsed = processor.post_process_generation(text, task=TASK_PROMPT, image_size=(image.width, image.height))
        outputs.append(str(parsed.get(TASK_PROMPT, text)).strip() if isinstance(parsed, dict) else str(text).strip())
    return outputs


def load_semantic_auditor(model_path: str, class_names: Sequence[str], device: torch.device, dtype: torch.dtype, local_files_only: bool):
    """Load local CLIP text tower and cache vectors for all SUN class names."""
    from transformers import AutoTokenizer, CLIPModel

    local_model = resolve_local_model(model_path, "clip")
    if local_model is not None:
        model_path = str(local_model)
        print(f"Using local CLIP auditor: {model_path}")
    elif local_files_only:
        raise FileNotFoundError(
            f"Local CLIP model not found for {model_path}. Attach its complete folder as Kaggle Input "
            "or set --clip-model to the correct path. Semantic audit is required by default."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    model = CLIPModel.from_pretrained(model_path, torch_dtype=dtype, local_files_only=local_files_only).to(device).eval()
    labels = [PurePosixPath(name.replace("\\", "/")).name.replace("_", " ").replace("-", " ") for name in class_names]
    prompts = [prompt.format(name=name) for name in labels for prompt in ("a photo of {name}", "a scene showing {name}")]
    vectors = encode_clip_texts(model, tokenizer, prompts, device, dtype, batch_size=128)
    vectors = vectors.reshape(len(labels), 2, -1).mean(dim=1)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    return model, tokenizer, vectors, labels


@torch.inference_mode()
def encode_clip_texts(model, tokenizer, texts: Sequence[str], device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    encoded = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True, return_tensors="pt")
        tokens = {key: value.to(device, non_blocking=True) for key, value in tokens.items()}
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            text_output = model.text_model(**tokens, return_dict=True)
            features = model.text_projection(text_output.pooler_output)
        encoded.append(torch.nn.functional.normalize(features.float(), dim=1).cpu())
    return torch.cat(encoded, dim=0) if encoded else torch.empty((0, model.config.projection_dim))


def semantic_audit_captions(
    captions: Sequence[str], model, tokenizer, class_vectors: torch.Tensor, class_labels: Sequence[str],
    device: torch.device, dtype: torch.dtype, threshold: float, batch_size: int,
):
    caption_vectors = encode_clip_texts(model, tokenizer, captions, device, dtype, batch_size).to(device)
    similarities = caption_vectors @ class_vectors.to(device).T
    scores, indices = similarities.max(dim=1)
    return [
        (float(score), str(class_labels[int(index)]), float(score) >= threshold)
        for score, index in zip(scores.cpu(), indices.cpu())
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--train-partition", type=Path, default=DEFAULT_TRAIN_PARTITION)
    parser.add_argument("--test-partition", type=Path, default=DEFAULT_TEST_PARTITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_ID, help="Florence-2 model ID or local model directory")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="Local CLIP model folder for semantic leakage audit")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Use only attached/local model files (default)")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads; requires Internet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--semantic-threshold", type=float, default=0.30,
                        help="Flag caption if CLIP cosine similarity to any SUN class prompt reaches this threshold")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: process up to N uncached images")
    parser.add_argument("--forbidden-terms-file", type=Path, default=None, help="Extra category aliases, one per line")
    parser.add_argument("--retry-invalid-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    if argv is None:
        # Jupyter/Colab injects `-f <kernel.json>` into sys.argv when a script
        # is pasted into a cell or invoked as a notebook entry point.
        argv = list(sys.argv[1:])
        if "-f" in argv:
            kernel_arg = argv.index("-f")
            del argv[kernel_arg:min(kernel_arg + 2, len(argv))]
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1 or args.clip_batch_size < 1 or args.workers < 0 or args.max_new_tokens < 1:
        raise ValueError("batch-size/max-new-tokens must be positive and workers nonnegative")
    if not -1.0 <= args.semantic_threshold <= 1.0:
        raise ValueError("--semantic-threshold must be in [-1, 1]")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.image_root = (args.image_root or args.dataset_root).expanduser().resolve()
    args.train_partition = args.train_partition.expanduser().resolve()
    args.test_partition = args.test_partition.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    records, class_names = load_records(args)
    extra_terms = []
    if args.forbidden_terms_file:
        extra_terms = [line.strip() for line in args.forbidden_terms_file.read_text(encoding="utf-8-sig").splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
    forbidden = build_forbidden_terms(class_names, extra_terms)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output.exists():
        output.unlink()
    existing = load_existing(output)
    done = {key for key, row in existing.items() if not args.retry_invalid_existing or str(row.get("caption_valid", "")).lower() in {"true", "1", "yes"}}
    todo = [row for row in records if row["image_id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"Device={device}; model={args.model}; starting batch={args.batch_size}; workers={args.workers}")
    print(f"Resume cache={len(done):,}; queued={len(todo):,}; output={output}")
    if not todo:
        print("All requested images already have captions.")
        return 0

    local_files_only = not args.allow_download
    model, processor, dtype = load_florence(args.model, device, local_files_only)
    clip_model, clip_tokenizer, class_vectors, class_labels = load_semantic_auditor(
        args.clip_model, class_names, device, dtype, local_files_only
    )
    print(f"Semantic audit: local CLIP; threshold={args.semantic_threshold:.2f}; class prompts={len(class_labels)}")
    loader = DataLoader(CaptionDataset(todo), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0, collate_fn=collate_images)
    effective_batch_size = args.batch_size
    start_time = time.perf_counter()
    done_in_dataset = sum(row["image_id"] in done for row in records)
    progress = tqdm(total=len(records), initial=done_in_dataset, desc="SUN397 captions", unit="image")
    valid_count = leakage_count = semantic_leakage_count = failures = 0
    for batch in loader:
        loaded = [(row, image) for row, image, error in batch if image is not None]
        batch_rows = []
        failures += len(batch) - len(loaded)
        for row, image, error in batch:
            if image is None:
                tqdm.write(f"Skipping unreadable {row['image_id']}: {error}")
                progress.update(1)
        captions: List[Optional[str]] = [None] * len(loaded)
        if loaded:
            current = effective_batch_size
            offset = 0
            while offset < len(loaded):
                group = loaded[offset:offset + current]
                try:
                    captions[offset:offset + len(group)] = generate_batch(
                        model, processor, [image for _, image in group], device, dtype, args.max_new_tokens
                    )
                    offset += len(group)
                except Exception as exc:
                    if current > 1:
                        current = max(1, current // 2)
                        effective_batch_size = current
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        tqdm.write(f"Batch reduced to {current} after inference error: {type(exc).__name__}: {exc}")
                    else:
                        failures += len(group)
                        tqdm.write(f"Skipping image after inference error: {type(exc).__name__}: {exc}")
                        offset += len(group)

        pending_rows = []
        pending_captions = []
        for (row, _image), raw in zip(loaded, captions):
            if raw is None:
                failures += 1
                progress.update(1)
                continue
            final, valid, leaked, _matched = audit_caption(raw, forbidden)
            pending_rows.append({
                "image_id": row["image_id"], "split": row["split"], "raw_caption": raw,
                "final_caption": final, "caption_valid": valid, "leakage_detected": leaked,
                "lexical_leakage_detected": leaked, "semantic_leakage_detected": False,
                "semantic_leakage_score": 0.0, "semantic_leakage_class": "",
                "retry_count": 0, "vlm_model": args.model, "prompt_version": PROMPT_VERSION,
            })
            pending_captions.append(final)
            valid_count += int(valid)
            leakage_count += int(leaked)
            progress.update(1)
        semantic_results = semantic_audit_captions(
            pending_captions, clip_model, clip_tokenizer, class_vectors, class_labels,
            device, dtype, args.semantic_threshold, args.clip_batch_size,
        )
        for row, (score, matched_class, semantic_leak) in zip(pending_rows, semantic_results):
            row["semantic_leakage_score"] = round(score, 6)
            row["semantic_leakage_class"] = matched_class
            row["semantic_leakage_detected"] = semantic_leak
            row["leakage_detected"] = bool(row["leakage_detected"] or semantic_leak)
            row["caption_valid"] = bool(row["caption_valid"] and not semantic_leak)
            semantic_leakage_count += int(semantic_leak)
            batch_rows.append(row)
        append_rows(output, batch_rows)
    progress.close()
    if args.retry_invalid_existing and output.is_file():
        deduplicated = load_existing(output)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(deduplicated.values())
        temporary.replace(output)
    final_rows = load_existing(output)
    invalid = sum(str(row.get("caption_valid", "")).lower() not in {"true", "1", "yes"} for row in final_rows.values())
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(f"Finished: rows={len(final_rows):,}, valid={len(final_rows)-invalid:,}, invalid={invalid:,}, lexical leaks={leakage_count:,}, semantic flags={semantic_leakage_count:,}, failures={failures:,}")
    print(f"This run: {len(todo):,} queued images in {elapsed/60:.1f} min ({len(todo)/elapsed:.2f} images/s); effective batch={effective_batch_size}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}; peak allocated={torch.cuda.max_memory_allocated(device)/1024**3:.1f} GiB")
    if len(final_rows) < len(records):
        print(f"{len(records)-len(final_rows):,} images remain. Rerun with the same output path to resume.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed rows are flushed. Rerun to resume.", file=sys.stderr)
        raise SystemExit(130)
