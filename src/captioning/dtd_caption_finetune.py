"""Review-first supervised Florence caption adaptation; embedded in a Kaggle notebook."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

from dtd_caption_screening import clean_dtd_caption


REFERENCE_COLUMNS = ["review_id", "image_id", "draft_caption_unverified", "target_caption",
                     "caption_source", "review_status", "reviewer_id", "teacher_model", "notes"]
EXPORT_COLUMNS = ["image_id", "split", "raw_caption", "final_caption", "caption_valid",
    "leakage_detected", "lexical_leakage_detected", "semantic_leakage_detected",
    "semantic_leakage_score", "semantic_leakage_class", "retry_count", "vlm_model", "prompt_version",
    "caption_input_policy", "semantic_audit_status", "captioner_fingerprint"]
TASK = "<MORE_DETAILED_CAPTION>"
PROMPT_VERSION = "florence2_dtd_decoder_lora_short_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv_rows(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing/duplicate CSV header: {path}")
        if not set(required).issubset(reader.fieldnames):
            raise ValueError(f"Missing required CSV columns: {sorted(set(required) - set(reader.fieldnames))}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed/incomplete CSV row: {path}; original file was not modified")
    return rows


def atomic_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalized_caption(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower().replace("_", " ")).strip()


def load_caption_manifest(root, split, expected_classes=47, per_class=40):
    root = Path(root).expanduser().resolve()
    if not 1 <= int(split) <= 10:
        raise ValueError("DTD_SPLIT must be in 1..10")
    images = (root / "images").resolve()
    names = sorted(path.name for path in images.iterdir() if path.is_dir())
    if len(names) != expected_classes:
        raise ValueError(f"Expected {expected_classes} DTD image folders; found {len(names)} at {images}")
    records, seen = [], set()
    for role in ("train", "val", "test"):
        counts = defaultdict(int)
        for raw in (root / "labels" / f"{role}{split}.txt").read_text(encoding="utf-8-sig").splitlines():
            text = raw.strip().replace("\\", "/")
            if not text:
                continue
            parts = PurePosixPath(text).parts
            if len(parts) != 2 or parts[0] not in names or any(part in {"..", "."} for part in parts):
                raise ValueError(f"Unexpected official DTD path: {text!r}")
            image_id = PurePosixPath(*parts).as_posix()
            path = images.joinpath(*parts).resolve()
            if not path.is_relative_to(images) or not path.is_file():
                raise FileNotFoundError(f"DTD image missing/outside images root: {path}")
            if image_id in seen:
                raise ValueError(f"Duplicate image or overlapping official splits: {image_id}")
            seen.add(image_id)
            counts[parts[0]] += 1
            records.append({"image_id": image_id, "image_path": str(path), "split": role})
        if set(counts) != set(names) or any(count != per_class for count in counts.values()):
            raise ValueError(f"Official {role}{split}.txt must contain {per_class} images/class")
    return records, names


def create_reference_template(records, output, per_class=4, seed=42, draft_csv=""):
    output = Path(output)
    if per_class < 1:
        raise ValueError("Review sample size per class must be positive")
    train = {row["image_id"]: row for row in records if row["split"] == "train"}
    if output.is_file():
        existing = read_csv_rows(output, REFERENCE_COLUMNS)
        if any(row.get("image_id") not in train for row in existing):
            raise ValueError("Existing review template has non-training images; use a separate output directory")
        return output  # Never overwrite manual review work on rerun.
    drafts = {}
    if draft_csv:
        for row in read_csv_rows(draft_csv):
            if row.get("image_id") in train:
                drafts[row["image_id"]] = row.get("raw_caption", "").strip() or row.get("final_caption", "")
    by_class = defaultdict(list)
    for image_id in train:
        by_class[image_id.split("/")[0]].append(image_id)
    rows = []
    for category in sorted(by_class):
        ordered = sorted(by_class[category], key=lambda key: object_hash([seed, key]))
        for image_id in ordered[:per_class]:
            rows.append({"review_id": object_hash(["blind-review", image_id])[:12], "image_id": image_id,
                "draft_caption_unverified": drafts.get(image_id, ""), "target_caption": "",
                "caption_source": "", "review_status": "pending", "reviewer_id": "",
                "teacher_model": "", "notes": ""})
    rows.sort(key=lambda row: object_hash(["review-order", seed, row["review_id"]]))
    atomic_csv(output, rows, REFERENCE_COLUMNS)
    return output


def validate_reference_text(text):
    text = " ".join(str(text).split())
    count = len(normalized_caption(text).split())
    if not 5 <= count <= 8:
        raise ValueError("Approved target_caption must contain 5-8 normalized English words; do not truncate references silently")
    if normalized_caption(clean_dtd_caption(text)) != normalized_caption(text):
        raise ValueError("Remove generic image/close-up openings from the reviewed target caption")
    return text


def save_caption_review(template, review_id, caption, reviewer_id, source="human_corrected", teacher_model=""):
    if not str(reviewer_id).strip():
        raise ValueError("Supply the human reviewer_id; approval is a provenance declaration, not an automated correctness check")
    if source not in {"human", "human_corrected", "teacher"}:
        raise ValueError("caption_source must be human, human_corrected or teacher")
    if source == "teacher" and not teacher_model.strip():
        raise ValueError("Record teacher_model for teacher captions, even after human review")
    target = validate_reference_text(caption)
    rows = read_csv_rows(template)
    matches = [row for row in rows if row.get("review_id") == review_id]
    if len(matches) != 1:
        raise ValueError("review_id must identify exactly one template row")
    matches[0].update(target_caption=target, reviewer_id=reviewer_id.strip(), caption_source=source,
                      review_status="approved", teacher_model=teacher_model.strip())
    atomic_csv(template, rows, REFERENCE_COLUMNS)


def approved_reference_pairs(path, records, minimum=50):
    train = {row["image_id"]: row for row in records if row["split"] == "train"}
    rows = read_csv_rows(path)
    required = {"image_id", "target_caption", "caption_source", "review_status", "reviewer_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Reference CSV must contain image_id,target_caption,caption_source,review_status,reviewer_id")
    seen, approved = set(), []
    for row in rows:
        image_id = row["image_id"].strip()
        if image_id not in train:
            raise ValueError(f"Only official TRAIN images may be reference examples: {image_id}")
        if image_id in seen:
            raise ValueError(f"Duplicate reference image_id: {image_id}")
        seen.add(image_id)
        if row["review_status"].strip().lower() != "approved":
            continue
        source = row["caption_source"].strip().lower()
        if source not in {"human", "human_corrected", "teacher"} or not row["reviewer_id"].strip():
            raise ValueError(f"Approved reference lacks human review provenance: {image_id}")
        if source == "teacher" and not row.get("teacher_model", "").strip():
            raise ValueError(f"Teacher caption requires teacher_model and human approval: {image_id}")
        approved.append({**train[image_id], "target_caption": validate_reference_text(row["target_caption"]),
            "caption_source": source, "reviewer_id": row["reviewer_id"].strip(),
            "teacher_model": row.get("teacher_model", ""), "image_sha256": file_sha256(train[image_id]["image_path"])})
    if len(approved) < minimum:
        raise ValueError(f"Only {len(approved)} approved caption references; need at least {minimum}. "
                         "Review the template and save captions first. Florence drafts are NOT ground truth; "
                         "no model was trained and no reference answers were invented.")
    return sorted(approved, key=lambda row: row["image_id"])


def split_reference_pairs(pairs, fraction=.20, seed=42):
    if not 0 < fraction < .5:
        raise ValueError("Reference validation fraction must be in (0, 0.5)")
    groups = defaultdict(list)
    for row in pairs:
        if row["split"] != "train":
            raise ValueError("Captioner reference train/holdout must both come from official DTD train")
        groups[row["image_id"].split("/")[0]].append(row)
    fit, holdout = [], []
    for group in groups.values():
        group = sorted(group, key=lambda row: object_hash([seed, row["image_id"]]))
        count = min(max(1, round(len(group) * fraction)), len(group) - 1) if len(group) > 1 else 0
        holdout.extend(group[:count])
        fit.extend(group[count:])
    if len(holdout) < 2 or not fit:
        raise ValueError("Need at least two held-out reference images; approve more training images per class")
    return sorted(fit, key=lambda row: row["image_id"]), sorted(holdout, key=lambda row: row["image_id"])


def checkpoint_asset_identity(model_dir):
    model_dir = Path(model_dir)
    files = sorted(path for path in model_dir.iterdir() if path.is_file() and
                   (path.suffix in {".py", ".json", ".safetensors", ".bin", ".txt"}))
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("Checkpoint has no weight files")
    return {path.name: file_sha256(path) for path in files}


def lock_caption_protocol(output, protocol):
    output = Path(output)
    path = output / "caption_training_protocol.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Changed caption references, assets, code or hyperparameters: choose a new output directory")
    if not path.is_file() and any((output / name).exists() for name in ("last.pt", "best.pt", "best_adapter")):
        raise ValueError("Existing checkpoint lacks this run's protocol; do not adopt another run's weights")
    atomic_json(path, protocol)
    return object_hash(protocol)


def seed_caption_run(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_caption_payload(payload):
    settings = payload["settings"]
    for name in ("epochs", "batch_size", "accumulation", "lora_rank", "lora_alpha",
                 "max_target_tokens", "max_new_tokens", "export_batch_size"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(settings["weight_decay"]) or settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= settings["lora_dropout"] < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    for key in ("settings", "fit", "holdout"):
        if payload[key] != payload["protocol"][key]:
            raise ValueError(f"Worker {key} does not match the locked training protocol")
    seen = set()
    for row in payload["fit"] + payload["holdout"]:
        if row["split"] != "train" or row["image_id"] in seen:
            raise ValueError("Reference sets must be disjoint and contain official training images only")
        seen.add(row["image_id"])
        validate_reference_text(row["target_caption"])
        if file_sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Reference image changed since review/protocol preparation")
    if not payload["fit"] or len(payload["holdout"]) < 2:
        raise ValueError("Prepare approved reference fit and holdout sets first")


class CaptionPairDataset:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from PIL import Image, ImageOps
        row = self.pairs[index]
        with Image.open(row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return image, row["target_caption"]


def caption_collate(items, processor, max_target_tokens=48):
    import torch
    images, targets = zip(*items)
    inputs = processor(text=[TASK] * len(images), images=list(images), padding=True, return_tensors="pt")
    tokens = processor.tokenizer(list(targets), padding=True, truncation=False, return_tensors="pt")
    if tokens["input_ids"].shape[1] > max_target_tokens:
        raise ValueError("Reference token sequence exceeds max_target_tokens; inspect targets rather than truncating them")
    labels = tokens["input_ids"].clone()
    labels[tokens["attention_mask"] == 0] = -100
    if not torch.any(labels != -100):
        raise ValueError("Empty reference-token batch")
    return {key: value for key, value in inputs.items() if torch.is_tensor(value)}, labels


def decoder_lora_targets(model):
    import torch
    targets = [name for name, module in model.named_modules() if name.startswith("language_model.")
               and ".decoder." in name and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No Florence decoder q_proj/v_proj modules found; do not silently adapt the vision encoder")
    return sorted(targets)


def load_caption_model(payload, device, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model
    processor = AutoProcessor.from_pretrained(payload["model"], trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(payload["model"], trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")
    if trainable:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        targets = decoder_lora_targets(model)
        settings = payload["settings"]
        # Generic PEFT preserves Florence's image-aware forward/generate signature.
        model = get_peft_model(model, LoraConfig(r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"], target_modules=targets, bias="none"))
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not names or any("lora_" not in name or ".decoder." not in name for name in names):
            raise ValueError("Unexpected trainable parameters; vision encoder/base model must remain frozen")
        atomic_json(Path(payload["output"]) / "trainable_parameters.json", {
            "targets": targets, "parameters": names,
            "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_count": sum(p.numel() for p in model.parameters())})
    else:
        model = PeftModel.from_pretrained(model, str(Path(payload["output"]) / "best_adapter"),
                                         is_trainable=False, local_files_only=True)
    return model.to(device), processor


def freeze_caption_vision(model):
    base = model.get_base_model()
    if hasattr(base, "vision_tower"):
        base.vision_tower.eval()


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()}


def restore_adapter(model, state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def atomic_torch_file(path, state):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def capture_caption_rng(generator):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_caption_rng(state, generator):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def move_caption_batch(batch, device):
    inputs, labels = batch
    return {key: value.to(device) for key, value in inputs.items()}, labels.to(device)


def caption_autocast(device):
    import torch
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


def evaluate_caption_nll(model, loader, device):
    import torch
    model.eval()
    total, tokens = 0., 0
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = move_caption_batch(batch, device)
            count = int((labels != -100).sum())
            with caption_autocast(device):
                loss = model(**inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite caption validation NLL")
            total += float(loss) * count
            tokens += count
    return total / max(tokens, 1)


def train_caption_epoch(model, loader, optimizer, scheduler, scaler, device, accumulation):
    import torch
    model.train()
    freeze_caption_vision(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_tokens = group_tokens = 0
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for index, batch in enumerate(loader):
        inputs, labels = move_caption_batch(batch, device)
        count = int((labels != -100).sum())
        with caption_autocast(device):
            loss = model(**inputs, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite caption training loss; last completed epoch is preserved")
        scaler.scale(loss * count).backward()
        total_loss += float(loss.detach()) * count
        total_tokens += count
        group_tokens += count
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(group_tokens)
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_tokens = 0
    return total_loss / max(total_tokens, 1)


def train_captioner(payload, device):
    import torch
    from functools import partial
    from torch.utils.data import DataLoader
    validate_caption_payload(payload)
    settings = payload["settings"]
    output = Path(payload["output"])
    seed_caption_run(settings["seed"])
    fingerprint = lock_caption_protocol(output, payload["protocol"])
    model, processor = load_caption_model(payload, device, trainable=True)
    generator = torch.Generator().manual_seed(settings["seed"])
    collate = partial(caption_collate, processor=processor, max_target_tokens=settings["max_target_tokens"])
    train_loader = DataLoader(CaptionPairDataset(payload["fit"]), batch_size=settings["batch_size"],
        shuffle=True, generator=generator, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(CaptionPairDataset(payload["holdout"]), batch_size=settings["batch_size"],
        shuffle=False, num_workers=0, collate_fn=collate)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    updates_per_epoch = math.ceil(len(train_loader) / settings["accumulation"])
    total_updates = updates_per_epoch * settings["epochs"]
    warmup = max(1, int(total_updates * .1))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(.01, .5 * (1 + math.cos(math.pi * (step - warmup) / max(total_updates - warmup, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    history, start, best_nll, best_epoch = [], 1, float("inf"), 0
    resume_path, best_path = output / "last.pt", output / "best.pt"
    if resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["identity"] != fingerprint:
            raise ValueError("Resume checkpoint belongs to another caption experiment")
        restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_nll, best_epoch = state["history"], state["best_nll"], state["best_epoch"]
        start = state["epoch"] + 1
        restore_caption_rng(state["rng"], generator)
        print("Resuming captioner at epoch", start, flush=True)
    # DataLoader iteration consumes RNG even without shuffling; preserve exact epoch-boundary resume.
    baseline_rng = capture_caption_rng(generator)
    with model.disable_adapter():
        baseline_nll = evaluate_caption_nll(model, val_loader, device)
    restore_caption_rng(baseline_rng, generator)
    for epoch in range(start, settings["epochs"] + 1):
        training = train_caption_epoch(model, train_loader, optimizer, scheduler, scaler, device, settings["accumulation"])
        validation = evaluate_caption_nll(model, val_loader, device)
        history.append({"epoch": epoch, "train_token_nll": training, "holdout_token_nll": validation})
        if validation < best_nll:
            best_nll, best_epoch = validation, epoch
            atomic_torch_file(best_path, {"identity": fingerprint, "epoch": epoch, "holdout_nll": validation,
                                         "adapter": adapter_state(model)})
        atomic_torch_file(resume_path, {"identity": fingerprint, "epoch": epoch, "adapter": adapter_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "best_nll": best_nll, "best_epoch": best_epoch, "rng": capture_caption_rng(generator)})
        atomic_csv(output / "history.csv", history, ["epoch", "train_token_nll", "holdout_token_nll"])
        print(f"Caption epoch {epoch}/{settings['epochs']}: train_nll={training:.4f}; holdout_nll={validation:.4f}", flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["identity"] != fingerprint:
        raise ValueError("Best adapter has the wrong identity")
    restore_adapter(model, best["adapter"])
    adapter_dir = output / "best_adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(output / "processor")
    comparisons = []
    for row in payload["holdout"]:
        with model.disable_adapter():
            before = generate_caption_texts(model, processor, [row], device, settings["max_new_tokens"])[0]
        after = generate_caption_texts(model, processor, [row], device, settings["max_new_tokens"])[0]
        comparisons.append({"image_id": row["image_id"], "reference": row["target_caption"],
            "base_raw_caption": before, "adapted_raw_caption": after,
            "base_short_caption": clean_dtd_caption(before), "adapted_short_caption": clean_dtd_caption(after),
            "human_preference": "", "review_notes": ""})
    comparison_path = output / "holdout_before_after.csv"
    if comparison_path.is_file():
        previous = {row["image_id"]: row for row in read_csv_rows(comparison_path)}
        for row in comparisons:
            old = previous.get(row["image_id"], {})
            if any(old.get(key) != value for key, value in row.items()
                   if key not in {"human_preference", "review_notes"}):
                raise ValueError("Existing holdout comparison differs; preserve manual reviews and use a new output directory")
        # Keep the original file, including human preferences, notes and any extra review columns.
    else:
        atomic_csv(comparison_path, comparisons, list(comparisons[0]))
    summary = {"identity": fingerprint, "epochs_completed": len(history), "best_epoch": best_epoch,
        "baseline_holdout_token_nll": baseline_nll, "best_holdout_token_nll": best_nll,
        "fit_images": len(payload["fit"]), "holdout_images": len(payload["holdout"]),
        "reference_origin": "human_reviewed_train_only", "downstream_f1_improvement_verified": False}
    atomic_json(output / "training_complete.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def generate_caption_texts(model, processor, rows, device, max_new_tokens):
    import torch
    from PIL import Image, ImageOps
    images = []
    for row in rows:
        with Image.open(row["image_path"]) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGB"))
    inputs = processor(text=[TASK] * len(images), images=images, padding=True, return_tensors="pt")
    moved = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    model.eval()
    with torch.inference_mode(), caption_autocast(device):
        generated = model.generate(**moved, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, use_cache=True)
    decoded = processor.batch_decode(generated, skip_special_tokens=False)
    if len(decoded) != len(rows):
        raise ValueError("Florence returned the wrong number of captions")
    results = []
    for text, image in zip(decoded, images):
        parsed = processor.post_process_generation(text, task=TASK, image_size=(image.width, image.height))
        results.append(str(parsed.get(TASK, text) if isinstance(parsed, dict) else text).strip())
    return results


class CaptionSemanticAuditor:
    def __init__(self, path, names, device):
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        if not Path(path).is_dir():
            raise FileNotFoundError("Attach local CLIP ViT-B/32 for diagnostic audit, or explicitly disable run_clip_audit")
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)
        self.model = CLIPTextModelWithProjection.from_pretrained(path, local_files_only=True).to(device).eval()
        self.names, self.device = names, device
        prompts = [template.format(name=name) for name in names
                   for template in ("a photo of {name}", "a scene showing {name}")]
        vectors = self.encode(prompts).reshape(len(names), 2, -1).mean(1)
        self.vectors = torch.nn.functional.normalize(vectors, dim=1)

    def encode(self, texts):
        import torch
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), 128):
                tokens = self.tokenizer(texts[start:start + 128], return_tensors="pt", padding=True,
                    truncation=True, max_length=self.model.config.max_position_embeddings).to(self.device)
                vectors.append(torch.nn.functional.normalize(self.model(**tokens).text_embeds.float(), dim=1))
        return torch.cat(vectors)

    def scores(self, texts):
        scores, indices = (self.encode(texts) @ self.vectors.T).max(1)
        return [(float(score), self.names[int(index)]) for score, index in zip(scores.cpu(), indices.cpu())]


def make_export_row(record, raw, identity, names, semantic=None):
    final = clean_dtd_caption(raw)
    normalized = " " + normalized_caption(final) + " "
    lexical = any(" " + normalized_caption(name) + " " in normalized for name in names)
    valid = 5 <= len(normalized.split()) <= 8 and not lexical
    score, matched = semantic if semantic is not None else ("", "")
    return {"image_id": record["image_id"], "split": record["split"], "raw_caption": raw,
        "final_caption": final, "caption_valid": valid, "leakage_detected": lexical,
        "lexical_leakage_detected": lexical, "semantic_leakage_detected": bool(semantic is not None and score >= .30),
        "semantic_leakage_score": score, "semantic_leakage_class": matched, "retry_count": 0,
        "vlm_model": "microsoft/Florence-2-base+dtd-decoder-lora:" + identity[:16],
        "prompt_version": PROMPT_VERSION, "caption_input_policy": "dtd_lexical_v2",
        "semantic_audit_status": "diagnostic_only" if semantic is not None else "not_run",
        "captioner_fingerprint": identity}


def export_caption_csv(payload, device):
    import torch
    output = Path(payload["output"])
    summary = json.loads((output / "training_complete.json").read_text(encoding="utf-8"))
    training_id = object_hash(payload["protocol"])
    if summary["identity"] != training_id or summary["epochs_completed"] != payload["settings"]["epochs"]:
        raise ValueError("Finish this caption training protocol before exporting all DTD captions")
    records = payload["export_records"]
    if len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Duplicate export image IDs")
    print("Hashing export images to verify dataset content across resumed sessions...", flush=True)
    inventory = [{"image_id": row["image_id"], "split": row["split"],
                  "sha256": file_sha256(row["image_path"])} for row in records]
    manifest = {"training": training_id, "best_adapter": checkpoint_asset_identity(output / "best_adapter"),
        "images": inventory, "prompt": TASK, "version": PROMPT_VERSION, "policy": "dtd_lexical_v2",
        "max_new_tokens": payload["settings"]["max_new_tokens"], "sampling": "greedy",
        "audit_class_names": list(payload["class_names"]),
        "clip": checkpoint_asset_identity(payload["clip"]) if payload["run_clip_audit"] else None,
        "clip_threshold": .30, "semantic_hard_gate": False}
    identity = object_hash(manifest)
    folder, csv_path = output / "exports", output / "exports" / "captions.csv"
    manifest_path = folder / "caption_generation_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Changed caption generation assets/options; use a new export directory, never mix CSV rows")
    if csv_path.exists() and not manifest_path.exists():
        raise ValueError("Existing captions.csv has no matching generation manifest; original file was not modified")
    atomic_json(manifest_path, manifest)
    existing = read_csv_rows(csv_path, EXPORT_COLUMNS) if csv_path.is_file() and csv_path.stat().st_size else []
    expected = {row["image_id"]: row["split"] for row in records}
    seen = set()
    for row in existing:
        if not set(EXPORT_COLUMNS).issubset(row) or row["image_id"] in seen or row["image_id"] not in expected:
            raise ValueError("Invalid schema, duplicate or unexpected image in export CSV")
        if row["captioner_fingerprint"] != identity or row["split"] != expected[row["image_id"]]:
            raise ValueError("Export CSV belongs to another captioner/split")
        seen.add(row["image_id"])
    todo = [row for row in records if row["image_id"] not in seen]
    print(f"Improved captions: cached={len(seen)}, remaining={len(todo)}, output={csv_path}", flush=True)
    if todo:
        model, processor = load_caption_model(payload, device, trainable=False)
        auditor = CaptionSemanticAuditor(payload["clip"], payload["class_names"], device) if payload["run_clip_audit"] else None
        size, offset = payload["settings"]["export_batch_size"], 0
        while offset < len(todo):
            group = todo[offset:offset + size]
            try:
                raw = generate_caption_texts(model, processor, group, device, payload["settings"]["max_new_tokens"])
                scores = auditor.scores([clean_dtd_caption(text) for text in raw]) if auditor else [None] * len(group)
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                size = max(1, size // 2)
                torch.cuda.empty_cache()
                print("Export batch reduced to", size, flush=True)
                continue
            if len(raw) != len(group) or len(scores) != len(group):
                raise ValueError("Caption/audit batch count mismatch; no rows from this batch were appended")
            rows = [make_export_row(row, text, identity, payload["class_names"], score)
                    for row, text, score in zip(group, raw, scores)]
            header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
                if header:
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            offset += len(group)
            print(f"Exported {len(seen) + offset}/{len(records)}", flush=True)
    final = read_csv_rows(csv_path)
    if {row["image_id"] for row in final} != set(expected):
        raise ValueError("Export is incomplete; rerun export to resume missing images")
    atomic_json(folder / "export_summary.json", {"rows": len(final), "csv_sha256": file_sha256(csv_path),
        "captioner_fingerprint": identity, "policy": "dtd_lexical_v2",
        "valid_captions": sum(row["caption_valid"] == "True" for row in final),
        "human_ground_truth": False, "downstream_improvement_verified": False})
    return csv_path


def caption_worker_main(payload_path, action):
    import torch
    import transformers
    import peft
    if not transformers.__version__.startswith("4.49.") or peft.__version__ != "0.14.0":
        raise RuntimeError("This legacy Florence training notebook uses Transformers 4.49.x and PEFT 0.14.0 in its isolated worker")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for real Florence training/export; CPU mocks are only for tests")
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    if checkpoint_asset_identity(payload["model"]) != payload["protocol"]["base_assets"]:
        raise ValueError("Base Florence assets changed after protocol preparation; do not reuse this run")
    device = torch.device("cuda")
    if action == "train":
        train_captioner(payload, device)
    elif action == "export":
        export_caption_csv(payload, device)
    else:
        raise ValueError("Worker action must be train or export")
