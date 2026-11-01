"""SUN397 visual TACS training and verified, self-contained web export."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import tacs_visual_web as runtime


def default_config():
    return {
        "schema": runtime.SCHEMA, "dataset": "SUN397", "method": "TACS_visual",
        "backbone": "dinov3_vitl16", "feature_dim": 1024, "num_classes": 397,
        "pretrained_variant": "LVD1689M_8aa4cbdd", "rope_training": "native_per_block_resampling",
        "image_size": 384, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "resize": "PIL_bicubic_stretch_square", "head_dropout": 0.1,
        "context_top_k": 1, "seed": 42, "val_fraction": 0.1, "pool_fraction": 0.2,
        "epochs": 100, "batch_size": 2, "effective_batch_size": 64, "workers": 0,
        "feature_batch_size": 8, "eval_batch_size": 8, "st_bank_chunk": 32, "gradient_checkpointing": True,
        "backbone_lr": 3e-6, "head_lr": 1e-4, "weight_decay": 0.05,
        "warmup_epochs": 5, "gumbel_tau": 0.1, "policy_weight": 1.0,
        "entropy_weight": 0.0, "label_smoothing": 0.0, "grad_clip": 1.0,
        "utility": "shared_cls_dot_product", "candidate_bank_update": "epoch_start_and_before_validation",
        "candidate_training_pixels": "cpu_float16_with_exact_st_weight_gradient",
        "train_augmentation": "PIL_random_resized_crop_scale_0.7_1.0_ratio_0.75_1.333_horizontal_flip_0.5",
        "inference_precision": "float32", "attention_method": runtime.ATTENTION_METHOD,
        "caption_used": False, "context_label_prior": False,
    }


def validate_config(config):
    if config["backbone"] != "dinov3_vitl16" or config["feature_dim"] != 1024:
        raise ValueError("This notebook is ViT-L/16 only")
    if config["num_classes"] != 397 or config["context_top_k"] != 1 or config["caption_used"]:
        raise ValueError("Expected SUN397 visual TACS, K=1, no captions")
    if config["image_size"] < 32 or config["image_size"] % 16:
        raise ValueError("Image size must be a positive multiple of patch size 16")
    if config["batch_size"] < 2:
        raise ValueError("Use train batch_size >=2 for within-batch standardized policy rewards")
    if config["effective_batch_size"] % config["batch_size"]:
        raise ValueError("effective_batch_size must be divisible by batch_size")
    if config["epochs"] < 1 or not 0 < config["val_fraction"] < 1 or not 0 < config["pool_fraction"] < 1:
        raise ValueError("Invalid epoch/split/pool settings")
    if config["gumbel_tau"] <= 0 or config["st_bank_chunk"] < 1 or config["workers"] < 0:
        raise ValueError("Invalid Gumbel/loader settings")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def environment():
    versions = {}
    for name in ("torch", "numpy", "Pillow", "tqdm", "torchvision", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": sys.platform, "packages": versions,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def csv_write(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError("Cannot infer columns for an empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def source_tree_identity(repo):
    repo = Path(repo)
    files = {p.relative_to(repo).as_posix(): runtime.file_sha256(p)
             for p in sorted(repo.rglob("*.py")) if ".git" not in p.parts and "__pycache__" not in p.parts}
    if "hubconf.py" not in files or not any(p.startswith("dinov3/") for p in files):
        raise ValueError("Incomplete DINOv3 repository")
    if not any(p.is_file() and p.name.upper().startswith(("LICENSE", "NOTICE")) for p in repo.iterdir()):
        raise FileNotFoundError("Attach complete DINOv3 source including original LICENSE/NOTICE files before training")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def resolve_weights(reference):
    path = Path(reference).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        matches = [p for p in path.rglob("*.pth") if "vitl16" in p.name.lower() and "plus" not in p.name.lower()]
        if len(matches) == 1:
            return matches[0].resolve()
        raise ValueError(f"Expected one ViT-L/16 .pth in {path}; found {len(matches)}. Set the exact filename.")
    raise FileNotFoundError(f"Attach local DINOv3 ViT-L/16 weights and set DINOV3_WEIGHTS: {path}")


def _relative_id(raw):
    raw = str(raw).strip().replace("\\", "/").lstrip("/")
    parts = list(PurePosixPath(raw).parts)
    if parts and parts[0].lower() in {"sun397", "images"}:
        parts = parts[1:]
    if len(parts) < 3 or ".." in parts or ":" in raw:
        raise ValueError(f"Unexpected SUN397 relative path: {raw}")
    return PurePosixPath(*parts).as_posix()


def load_sun397(dataset_root, train_partition, test_partition, config):
    root = Path(dataset_root).expanduser().resolve()
    name_file = next((p for p in (root / "ClassName.txt", root.parent / "ClassName.txt") if p.is_file()), None)
    if name_file is None:
        raise FileNotFoundError(f"ClassName.txt not found at {root} or its parent")
    classes = [line.strip().lstrip("/").replace("\\", "/") for line in name_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(classes) != config["num_classes"] or len(set(classes)) != len(classes):
        raise ValueError("Expected 397 unique SUN397 classes, preserving ClassName.txt order")
    class_to_id = {name: i for i, name in enumerate(classes)}

    def partition(path, split):
        path = Path(path).expanduser().resolve()
        rows = []
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            if not raw.strip():
                continue
            image_id = _relative_id(raw)
            class_name = str(PurePosixPath(image_id).parent)
            if class_name not in class_to_id:
                raise ValueError(f"Unknown class in official partition: {image_id}")
            image_path = root.joinpath(*PurePosixPath(image_id).parts).resolve()
            if not image_path.is_relative_to(root) or not image_path.is_file():
                raise FileNotFoundError(f"Missing SUN397 image: {image_path}")
            rows.append({"image_id": image_id, "image_path": str(image_path), "class_id": class_to_id[class_name],
                         "class_name": class_name, "split": split})
        ids = [row["image_id"] for row in rows]
        counts = np.bincount([row["class_id"] for row in rows], minlength=len(classes))
        if len(ids) != len(classes) * 50 or len(set(ids)) != len(ids) or not np.all(counts == 50):
            raise ValueError(f"{path} must contain 50 distinct images per class (19,850 images)")
        return rows

    official_train = partition(train_partition, "train")
    test = partition(test_partition, "test")
    if {r["image_id"] for r in official_train} & {r["image_id"] for r in test}:
        raise ValueError("Official train/test image IDs overlap")
    rng = np.random.default_rng(config["seed"])
    val_positions = set()
    for class_id in range(len(classes)):
        positions = [i for i, row in enumerate(official_train) if row["class_id"] == class_id]
        count = max(1, min(len(positions) - 1, round(len(positions) * config["val_fraction"])))
        val_positions.update(int(i) for i in rng.choice(positions, count, replace=False))
    train, val = [], []
    for i, row in enumerate(official_train):
        item = dict(row)
        item["split"] = "val" if i in val_positions else "train"
        (val if i in val_positions else train).append(item)
    count = int(len(train) * config["pool_fraction"])
    pool_positions = np.random.default_rng(config["seed"]).choice(len(train), count, replace=False)
    candidates = [dict(train[int(i)], position=position) for position, i in enumerate(pool_positions)]
    if count < 2 or {r["image_id"] for r in candidates} & {r["image_id"] for r in val + test}:
        raise ValueError("Candidate pool must be a nonempty subset of actual training images only")
    return classes, train, val, test, candidates


def augment(image, config):
    image = runtime.read_rgb(image)
    width, height = image.size
    area = width * height
    box = None
    for _ in range(10):
        target = area * random.uniform(0.7, 1.0)
        ratio = math.exp(random.uniform(math.log(0.75), math.log(4 / 3)))
        crop_w, crop_h = round(math.sqrt(target * ratio)), round(math.sqrt(target / ratio))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            x, y = random.randint(0, width - crop_w), random.randint(0, height - crop_h)
            box = (x, y, x + crop_w, y + crop_h)
            break
    if box is None:
        side = min(width, height)
        x, y = (width - side) // 2, (height - side) // 2
        box = (x, y, x + side, y + side)
    image = image.crop(box)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return runtime.preprocess(image, config)


class QueryDataset(Dataset):
    def __init__(self, rows, config, training=False):
        self.rows, self.config, self.training = rows, config, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            image = augment(row["image_path"], self.config) if self.training else runtime.preprocess(row["image_path"], self.config)
        except Exception as exc:
            raise RuntimeError(f"Unreadable image {row['image_id']}: {exc}") from exc
        return {"image": image, "label": row["class_id"], "image_id": row["image_id"]}


def worker_seed(_worker_id):
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def make_loader(rows, config, training=False):
    kwargs = {"num_workers": config["workers"], "pin_memory": torch.cuda.is_available(), "worker_init_fn": worker_seed}
    if config["workers"]:
        kwargs.update(persistent_workers=False, prefetch_factor=2)
    size = config["batch_size"] if training else config["eval_batch_size"]
    return DataLoader(QueryDataset(rows, config, training), batch_size=size, shuffle=training,
                      drop_last=training, generator=torch.Generator().manual_seed(config["seed"]), **kwargs)


def build_image_bank(candidates, config):
    count, size = len(candidates), config["image_size"]
    bank = torch.empty((count, 3, size, size), dtype=torch.float16, device="cpu")
    for index, row in enumerate(tqdm(candidates, desc="Candidate pixels on CPU")):
        bank[index].copy_(runtime.preprocess(row["image_path"], config))
    return bank


@torch.no_grad()
def refresh_bank(model, candidates, config, device, image_bank=None, amp=False):
    was_training = model.training
    model.eval()
    chunks = []
    for start in range(0, len(candidates), config["feature_batch_size"]):
        group = candidates[start:start + config["feature_batch_size"]]
        # Evaluate raw images in fp32; the optional fp16 pixel bank is training-only.
        images = torch.stack([runtime.preprocess(r["image_path"], config) for r in group]).to(device)
        with autocast_context(device, amp):
            features = model.encode_selector(images)
        chunks.append(features.detach().float().cpu())
    model.train(was_training)
    bank = torch.cat(chunks)
    if not torch.isfinite(bank).all():
        raise FloatingPointError("Candidate features contain NaN/Inf")
    return bank


def autocast_context(device, enabled=True):
    device = torch.device(device)
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def classification_metrics(truth, predictions, classes):
    counts = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(counts, (np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)), 1)
    tp = counts.diagonal()
    denominator = counts.sum(0) + counts.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes, dtype=float), where=denominator != 0)
    return {"accuracy": float(tp.sum() / max(1, counts.sum())), "macro_f1": float(f1.mean())}


def rng_state(loader):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def restore_rng(saved, loader):
    random.setstate(saved["python"])
    state = saved["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    torch.set_rng_state(saved["torch"].cpu())
    if torch.cuda.is_available() and saved["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda"]])
    loader.generator.set_state(saved["loader"].cpu())


def run_identity(config, classes, frames, assets):
    ids = {key: [(r["image_id"], int(r["class_id"])) for r in value] for key, value in frames.items()}
    payload = {"config": config, "classes": classes, "splits_and_pool": ids, "assets": assets,
               "model_source_sha256": runtime.file_sha256(Path(runtime.__file__)), "training_source_sha256": runtime.file_sha256(__file__)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def train_epoch(model, loader, feature_bank, pixel_bank, candidates, optimizer, scaler, config, device, epoch):
    model.train()
    bank = feature_bank.to(device)
    positions = {row["image_id"]: i for i, row in enumerate(candidates)}
    accumulate = config["effective_batch_size"] // config["batch_size"]
    stats = {key: 0.0 for key in ("loss", "ce_loss", "policy_loss", "reward", "entropy")}
    truth, predictions = [], []
    optimizer_steps = skipped_steps = 0
    optimizer.zero_grad(set_to_none=True)
    steps = len(loader)
    if steps < 1:
        raise ValueError("Training loader is empty")
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch:03d} train")):
        query, labels = batch["image"].to(device), batch["label"].to(device)
        excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"]], device=device)
        with autocast_context(device):
            scores, _ = model.score(query, bank, excluded)
            weights = F.gumbel_softmax(scores, tau=config["gumbel_tau"], hard=True, dim=1)
            context = runtime.straight_through_images(weights, pixel_bank, config["st_bank_chunk"])
            logits = model(query, context)
            ce = F.cross_entropy(logits.float(), labels, label_smoothing=config["label_smoothing"])
            distribution = torch.distributions.Categorical(logits=scores)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()
        # Reward passes use the same eval-mode task network and detached outputs.
        model.eval()
        with torch.no_grad(), autocast_context(device):
            baseline = model(query)
            reward_context = pixel_bank.index_select(0, actions.cpu()).to(device, dtype=torch.float32)
            paired = model(query, reward_context)
            reward = (F.cross_entropy(baseline.float(), labels, reduction="none") -
                      F.cross_entropy(paired.float(), labels, reduction="none")).detach()
            advantage = ((reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-6)).detach()
        model.train()
        policy = -(log_probs * advantage).mean()
        loss = ce + config["policy_weight"] * policy - config["entropy_weight"] * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}")
        group_start = (step // accumulate) * accumulate
        divisor = min(accumulate, steps - group_start)
        scaler.scale(loss / divisor).backward()
        if (step + 1) % accumulate == 0 or step + 1 == steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
            skipped_steps += int(skipped)
            optimizer_steps += int(not skipped)
            optimizer.zero_grad(set_to_none=True)
        count = len(labels)
        for key, value in (("loss", loss), ("ce_loss", ce), ("policy_loss", policy), ("reward", reward.mean()), ("entropy", entropy)):
            stats[key] += float(value.detach()) * count
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
    if not optimizer_steps:
        raise FloatingPointError("Every optimizer step overflowed; inspect AMP/learning rates before resuming")
    return {**{k: v / len(truth) for k, v in stats.items()}, **classification_metrics(truth, predictions, config["num_classes"]),
            "optimizer_steps": optimizer_steps, "amp_skipped_steps": skipped_steps}


@torch.no_grad()
def evaluate(model, loader, bank, pixel_bank, candidates, config, device, label="validation", amp=False):
    model.eval()
    features = bank.to(device)
    positions = {row["image_id"]: i for i, row in enumerate(candidates)}
    rows, loss_sum = [], 0.0
    for batch in tqdm(loader, desc=label):
        query, labels = batch["image"].to(device), batch["label"].to(device)
        excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"]], device=device)
        with autocast_context(device, amp):
            scores, _ = model.score(query, features, excluded)
            indices = scores.argmax(1)
            # Raw files + identical eval transform, matching exported inference.
            context = torch.stack([runtime.preprocess(candidates[int(i)]["image_path"], config) for i in indices.cpu()]).to(device)
            logits = model(query, context)
            loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        for j, image_id in enumerate(batch["image_id"]):
            pos = int(indices[j])
            rows.append({"image_id": image_id, "ground_truth": int(labels[j]), "prediction": int(logits[j].argmax()),
                         "candidate_position": pos, "candidate_id": candidates[pos]["image_id"]})
    if not rows:
        raise ValueError("Evaluation loader is empty")
    metrics = classification_metrics([r["ground_truth"] for r in rows], [r["prediction"] for r in rows], config["num_classes"])
    return {**metrics, "loss": loss_sum / len(rows), "images": len(rows)}, rows


def train_run(model, classes, train, val, candidates, config, assets, output, device):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config, classes, {"train": train, "val": val, "pool": candidates}, assets)
    protocol = output / "protocol.json"
    if protocol.is_file() and runtime.read_json(protocol)["identity"] != identity:
        raise ValueError("Output directory belongs to another run. Do not change seed/epochs/pool/source on resume.")
    source_hashes = {"runtime": runtime.file_sha256(runtime.__file__), "training": runtime.file_sha256(__file__)}
    runtime.write_json(protocol, {"identity": identity, "config": config, "assets": assets,
                                  "source_sha256": source_hashes, "environment": environment()})
    csv_write(output / "train_split.csv", train)
    csv_write(output / "validation_split.csv", val)
    csv_write(output / "candidate_pool.csv", candidates)
    runtime.write_json(output / "class_names.json", classes)
    train_loader, val_loader = make_loader(train, config, True), make_loader(val, config)
    pixel_bank = build_image_bank(candidates, config)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["backbone_lr"]},
        {"params": model.head.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    def lr_scale(epoch):
        warmup = min(config["warmup_epochs"], config["epochs"])
        if warmup and epoch < warmup:
            return 0.2 + 0.8 * epoch / warmup
        progress = (epoch - warmup) / max(1, config["epochs"] - warmup)
        return 0.01 + 0.99 * (1 + math.cos(math.pi * min(1, progress))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported())
    last_path, best_path = output / "last.pt", output / "best.pt"
    history, best_loss, start_epoch = [], math.inf, 1
    if last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        if last["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        restore_rng(last["rng"], train_loader)
        history, best_loss, start_epoch = last["history"], last["best_val_loss"], last["epoch"] + 1
        del last
        if not best_path.is_file():
            raise FileNotFoundError("Resume requires this run's best.pt as well as last.pt")
        print(f"Resuming after completed epoch {start_epoch - 1}; horizon stays {config['epochs']}")
    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        bank = refresh_bank(model, candidates, config, device, amp=True)
        train_metrics = train_epoch(model, train_loader, bank, pixel_bank, candidates, optimizer, scaler, config, device, epoch)
        bank = refresh_bank(model, candidates, config, device, amp=True)
        val_metrics, _ = evaluate(model, val_loader, bank, pixel_bank, candidates, config, device, amp=True)
        improved = val_metrics["loss"] < best_loss
        best_loss = min(best_loss, val_metrics["loss"])
        history.append({"epoch": epoch, **{"train_" + k: v for k, v in train_metrics.items()},
                        **{"val_" + k: v for k, v in val_metrics.items()}, "seconds": time.perf_counter() - started,
                        "backbone_lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        base = {"schema": runtime.SCHEMA, "identity": identity, "epoch": epoch, "model_state_dict": model_state,
                "config": config, "class_names": classes, "val_metrics": val_metrics}
        if improved:
            atomic_torch_save(best_path, base)
        atomic_torch_save(last_path, {**base, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                                     "scaler_state_dict": scaler.state_dict(), "rng": rng_state(train_loader),
                                     "history": history, "best_val_loss": best_loss})
        csv_write(output / "history.csv", history)
        print(f"epoch={epoch:03d}/{config['epochs']} val_loss={val_metrics['loss']:.6f} val_f1={val_metrics['macro_f1']:.6f} best={improved}")
        del model_state, base
    if not history or history[-1]["epoch"] != config["epochs"]:
        raise RuntimeError("Training budget did not complete; final test/export are blocked")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    if best["identity"] != identity:
        raise ValueError("Best checkpoint belongs to another run")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    runtime.write_json(output / "training_complete.json", {"identity": identity, "epochs_completed": config["epochs"],
                                                         "best_epoch": best["epoch"], "best_sha256": runtime.file_sha256(best_path)})
    return best, history


def copy_vendor(repo, destination):
    repo, destination = Path(repo), Path(destination)
    package = repo / "dinov3"
    if not package.is_dir() or not (repo / "hubconf.py").is_file():
        raise FileNotFoundError("Cannot export incomplete DINOv3 source")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.pth", "*.pt", "*.safetensors", ".cache")
    destination.mkdir(parents=True, exist_ok=False)
    # Backbone inference needs these upstream modules, not long training YAML
    # trees or detection/segmentation imports. Preserve selected sources verbatim.
    vendored = destination / "dinov3"
    vendored.mkdir()
    shutil.copy2(package / "__init__.py", vendored / "__init__.py")
    for name in ("models", "layers", "utils"):
        shutil.copytree(package / name, vendored / name, ignore=ignore)
    (vendored / "hub").mkdir()
    for name in ("__init__.py", "backbones.py", "utils.py"):
        shutil.copy2(package / "hub" / name, vendored / "hub" / name)
    licenses = [p for p in repo.iterdir() if p.is_file() and (p.name.upper().startswith(("LICENSE", "NOTICE")))]
    if not licenses:
        raise FileNotFoundError("Attach the original DINOv3 source with its license/notice files")
    for path in licenses:
        shutil.copy2(path, destination / path.name)


def export_bundle(model, best, candidates, classes, config, repo, output, val_example, test_metrics, test_predictions,
                  model_factory=runtime.build_model, require_training_complete=True):
    """Only call after loading best.pt; always rebuild the final feature bank."""
    output = Path(output).resolve()
    if require_training_complete:
        complete = runtime.read_json(output / "training_complete.json")
        if complete["identity"] != best["identity"] or complete["best_epoch"] != best["epoch"]:
            raise ValueError("Training/best/export identity mismatch")
        if complete["best_sha256"] != runtime.file_sha256(output / "best.pt"):
            raise ValueError("Best checkpoint changed after completion")
    if classes != best["class_names"] or config != best["config"]:
        raise ValueError("Export arguments do not match the best checkpoint")
    protocol = runtime.read_json(output / "protocol.json")
    if protocol["identity"] != best["identity"] or protocol["config"] != config:
        raise ValueError("Export protocol does not match the trained run")
    source_hashes = {"runtime": runtime.file_sha256(runtime.__file__), "training": runtime.file_sha256(__file__)}
    if protocol["source_sha256"] != source_hashes:
        raise ValueError("Runtime/training source changed after training")
    expected_source = protocol["assets"].get("dino_source_sha256")
    if expected_source is not None and source_tree_identity(repo) != expected_source:
        raise ValueError("Export DINOv3 source differs from the training source")
    with (output / "candidate_pool.csv").open(encoding="utf-8", newline="") as handle:
        trained_pool = [(r["image_id"], int(r["class_id"])) for r in csv.DictReader(handle)]
    if trained_pool != [(r["image_id"], int(r["class_id"])) for r in candidates]:
        raise ValueError("Export candidate pool/order differs from the trained pool")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    device = next(model.parameters()).device
    export_root = output / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    target = export_root / "sun397_tacs_visual_vitl16_web_bundle"
    archive = target.with_suffix(".zip")
    best_sha = runtime.file_sha256(output / "best.pt")
    if target.exists() or archive.exists():
        if target.is_dir() and archive.is_file() and (export_root / "download_manifest.json").is_file():
            saved_manifest = runtime.verify_bundle(target)
            download = runtime.read_json(export_root / "download_manifest.json")
            if (saved_manifest["run_identity"] == best["identity"]
                    and saved_manifest.get("training_checkpoint_sha256") == best_sha
                    and download["zip"] == archive.name and download["bytes"] == archive.stat().st_size
                    and download["sha256"] == runtime.file_sha256(archive)):
                print(f"Reusing this run's verified deployment ZIP: {archive}")
                return archive
        raise FileExistsError("Export target exists but cannot be verified as this run's complete bundle. Preserve it and inspect exports/.")
    parameter_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    image_bytes = sum(Path(row["image_path"]).stat().st_size for row in candidates)
    if shutil.disk_usage(export_root).free < 2 * (parameter_bytes + image_bytes) + 1024 ** 3:
        raise OSError("Not enough disk space for bundle plus ZIP; preserve/move last.pt before exporting")
    stage = export_root / (".staging_" + uuid.uuid4().hex)
    stage.mkdir()
    portable_config = dict(config, candidate_count=len(candidates), inference_precision="float32")
    runtime.write_json(stage / "config.json", portable_config)
    runtime.write_json(stage / "class_names.json", classes)
    atomic_torch_save(stage / "model.pt", {"schema": runtime.SCHEMA, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                          "class_names": classes, "config": portable_config, "epoch": best["epoch"], "run_identity": best["identity"]})
    checkpoint_sha = runtime.file_sha256(stage / "model.pt")
    pool = []
    for position, row in enumerate(tqdm(candidates, desc="Bundle candidate images")):
        name = "candidates/" + row["image_id"]
        destination = runtime.bundle_path(stage, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image_path"], destination)
        pool.append({"position": position, "image_id": row["image_id"], "file": name, "class_id": int(row["class_id"]),
                     "class_name": classes[int(row["class_id"])], "sha256": runtime.file_sha256(destination)})
    runtime.write_json(stage / "candidate_pool.json", pool)
    csv_write(stage / "candidate_pool.csv", pool)
    final_bank = refresh_bank(model, candidates, config, device, amp=False)
    atomic_torch_save(stage / "candidate_features.pt", {"features": final_bank, "image_ids": [r["image_id"] for r in pool],
                                                      "checkpoint_sha256": checkpoint_sha, "precision": "float32"})
    copy_vendor(repo, stage / "vendor/dinov3")
    (stage / "runtime").mkdir()
    shutil.copy2(Path(runtime.__file__), stage / "runtime/tacs_visual_web.py")
    (stage / "provenance").mkdir()
    shutil.copy2(Path(__file__), stage / "provenance/training_source.py")
    runtime.write_json(stage / "environment.json", environment())
    runtime.write_json(stage / "metrics.json", {"test": test_metrics, "validation_best": best["val_metrics"], "best_epoch": best["epoch"],
                                              "run_identity": best["identity"], "test_precision": "float32"})
    csv_write(stage / "test_predictions.csv", test_predictions)
    if (output / "history.csv").is_file():
        shutil.copy2(output / "history.csv", stage / "history.csv")
    shutil.copy2(output / "protocol.json", stage / "protocol.json")
    for name in ("train_split.csv", "validation_split.csv", "test_split.csv"):
        if (output / name).is_file():
            shutil.copy2(output / name, stage / "provenance" / name)
    (stage / "examples").mkdir()
    example_path = stage / "examples" / ("query" + Path(val_example["image_path"]).suffix.lower())
    shutil.copy2(val_example["image_path"], example_path)
    runtime.write_json(stage / "attention_config.json", {
        "method": runtime.ATTENTION_METHOD, "layer": "last_transformer_block", "heads": "mean",
        "query_classifier": "query patch slice of pair-input classifier CLS attention",
        "candidate_selector": "selected candidate standalone selector-encoder CLS attention",
        "raw_maps_included": True, "class_conditional": False, "causal_explanation": False,
        "no_context_comparison_model_included": False,
        "note": "No separate attention network is needed; maps are recomputed by the trained backbone. Not an exact replication of every Figure 4 visualization choice.",
    })
    versions = environment()["packages"]
    requirements = [f"{name}=={versions[name].split('+')[0]}" for name in ("torch", "numpy", "Pillow", "tqdm")]
    (stage / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    (stage / "runtime/verify_export.py").write_text(COLD_VERIFY_SOURCE, encoding="utf-8")
    (stage / "README.md").write_text(
        "# SUN397 visual TACS ViT-L/16 web bundle\n\n"
        "Full trained weights, exact class order, candidate images and bank, vendored DINOv3 backbone code and runtime.\n"
        "No SUN397 training mount or pretrained weight file is needed after extraction.\n"
        "Install matching PyTorch for your OS/CUDA and the listed dependencies in a separate environment.\n"
        "No Florence, CLIP or captions are used in this visual bundle.\n\n"
        "```bash\npython runtime/tacs_visual_web.py --bundle . --image examples/" + example_path.name + " --output prediction.json\n```\n\n"
        "For FastAPI, import BundlePredictor from runtime/tacs_visual_web.py, load once and call predict(file_bytes).\n"
        "It returns model/prediction/top_k and evidence.candidate plus query_classifier/candidate_selector PNG data URLs.\n"
        "Do not call the CPU/GPU predictor concurrently without a lock. API wiring in the existing web is a separate step.\n"
        "Attention is last-block mean-head CLS attention, not Grad-CAM or a class-conditioned causal claim.\n"
        "There is no separately trained no-context model, text captioner or textual feature naming in this bundle.\n"
        "Keep original DINOv3/model/dataset license terms; do not assume public redistribution rights.\n",
        encoding="utf-8")
    # The manifest is refreshed after offline reload/parity and explanation checks.
    def manifest():
        files = {p.relative_to(stage).as_posix(): {"bytes": p.stat().st_size, "sha256": runtime.file_sha256(p)}
                 for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
        runtime.write_json(stage / "manifest.json", {"schema": runtime.SCHEMA, "status": "ready", "files": files,
                                                   "run_identity": best["identity"], "model_sha256": checkpoint_sha,
                                                   "training_checkpoint_sha256": best_sha})
    manifest()
    cold_reload = model_factory is runtime.build_model
    if cold_reload:
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, str(stage / "runtime/verify_export.py"), "--bundle", str(stage),
                                  "--image", str(example_path), "--image-id", val_example["image_id"], "--device", str(device)],
                                 cwd=stage, env=env, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError("Cold offline bundle reload failed:\n" + process.stdout[-4000:] + process.stderr[-8000:])
        result = runtime.read_json(stage / "examples/prediction.json")
        reloaded_logits = torch.load(stage / "examples/reload_logits.pt", map_location="cpu", weights_only=True)
    else:
        # Dependency injection is used by CPU unit tests, never the notebook's production export.
        predictor = runtime.BundlePredictor(stage, device=str(device), model_factory=model_factory)
        result = predictor.predict(example_path, explain=True, image_id=val_example["image_id"])
        reloaded_logits = predictor.last_logits
        del predictor
    position = result["evidence"]["candidate"]["position"]
    with torch.no_grad():
        query = runtime.preprocess(example_path, config)[None].to(device)
        reference_scores, _ = model.score(query, final_bank.to(device))
        example_sha = runtime.file_sha256(example_path)
        for i, row in enumerate(pool):
            if row["sha256"] == example_sha or row["image_id"] == val_example["image_id"]:
                reference_scores[:, i] = -torch.inf
        if int(reference_scores.argmax(1)) != position:
            raise ValueError("Export reload selected another candidate than the best checkpoint")
        reference = model(query, runtime.preprocess(candidates[position]["image_path"], config)[None].to(device)).float().cpu()
    error = float((reference - reloaded_logits).abs().max())
    if not torch.allclose(reference, reloaded_logits, atol=1e-4, rtol=1e-4):
        raise ValueError(f"Export reload logits disagree with best checkpoint: max_abs={error}")
    for role, item in result["evidence"]["attention"].items():
        if item["method"] != runtime.ATTENTION_METHOD or not np.isfinite(np.asarray(item["raw_grid"])).all():
            raise ValueError(f"Invalid exported attention: {role}")
    runtime.write_json(stage / "examples/prediction.json", result)
    runtime.write_json(stage / "smoke_test.json", {"passed": True, "offline_local_reload": True, "cold_subprocess_reload": cold_reload,
                                                  "strict_state_dict": True,
                                                  "max_abs_logit_error": error, "query_image_id": val_example["image_id"],
                                                  "selected_candidate_id": result["evidence"]["candidate"]["image_id"],
                                                  "attention_roles": list(result["evidence"]["attention"]),
                                                  "flat_attention_roles": [k for k, v in result["evidence"]["attention"].items() if v["status"] != "ready"]})
    manifest()
    runtime.verify_bundle(stage)
    stage.rename(target)
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
        for path in sorted(target.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                handle.write(path, arcname=target.name + "/" + path.relative_to(target).as_posix())
    with zipfile.ZipFile(temporary) as handle:
        bad = handle.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
    temporary.replace(archive)
    runtime.write_json(export_root / "download_manifest.json", {"zip": archive.name, "bytes": archive.stat().st_size,
                                                              "sha256": runtime.file_sha256(archive), "bundle": target.name,
                                                              "model": config["backbone"], "schema": runtime.SCHEMA})
    print(f"Verified deployment ZIP: {archive}")
    return archive


COLD_VERIFY_SOURCE = '''"""Cold-process inference check using only extracted bundle assets."""
import argparse
import sys
from pathlib import Path
import torch
from tacs_visual_web import BundlePredictor, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--image-id", required=True)
parser.add_argument("--device", required=True)
args = parser.parse_args()
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
predictor = BundlePredictor(args.bundle, args.device)
import dinov3
if not Path(dinov3.__file__).resolve().is_relative_to((args.bundle / "vendor/dinov3").resolve()):
    raise RuntimeError("Reload used an external DINOv3 source instead of the bundle")
plain = predictor.predict(args.image, explain=False, image_id=args.image_id)
plain_logits = predictor.last_logits.clone()
explained = predictor.predict(args.image, explain=True, image_id=args.image_id)
if plain["evidence"]["candidate"]["image_id"] != explained["evidence"]["candidate"]["image_id"]:
    raise RuntimeError("Explanation changed context selection")
if not torch.allclose(plain_logits, predictor.last_logits, atol=1e-5, rtol=1e-5):
    raise RuntimeError("Attention capture changed model logits")
write_json(args.bundle / "examples/prediction.json", explained)
torch.save(predictor.last_logits, args.bundle / "examples/reload_logits.pt")
print("Cold bundle inference, both real attention maps and logits parity passed")
'''
