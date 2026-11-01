"""Train/evaluate/export the same ISIC segmentation model used for web inference."""
from __future__ import annotations

from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import isic_segmentation_web as net
from isic_segmentation_web import io_utils as io


def default_config(mode="tacs"):
    return {"schema": net.SCHEMA, "mode": mode, "backbone": "dinov3_vitl16", "feature_dim": 1024,
        "image_size": 448, "cross_heads": 16, "decoder_channels": 128, "text_dim": 512,
        "context_top_k": 1, "pool_fraction": .20, "seed": 42, "epochs": 100,
        "batch_size": 2, "effective_batch_size": 8, "eval_batch_size": 4, "feature_batch_size": 8,
        "workers": 0, "lr": 1e-4, "weight_decay": 1e-4, "warmup_epochs": 5,
        "gumbel_tau": .1, "policy_weight": .5, "entropy_weight": 0., "grad_clip": 1.,
        "gradient_checkpointing": True, "threshold": .5, "pool_token_dtype": "float16",
        "mean": [.485, .456, .406], "std": [.229, .224, .225], "patience": 0,
        "strict_official_counts": True}


def validate_config(config):
    if config["mode"] not in {"tacs", "mtacs"} or config["context_top_k"] != 1:
        raise ValueError("This experiment supports TACS / M-TACS full_both, K=1")
    if config["image_size"] < 32 or config["image_size"] % 16:
        raise ValueError("image_size must be a multiple of 16")
    if config["feature_dim"] % config["cross_heads"] or config["decoder_channels"] < 8:
        raise ValueError("Invalid attention heads/decoder width")
    for name in ("epochs", "batch_size", "effective_batch_size", "eval_batch_size", "feature_batch_size"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["effective_batch_size"] < 2 or config["effective_batch_size"] % config["batch_size"]:
        raise ValueError("Use an effective batch >=2 divisible by the physical batch")
    if config["pool_fraction"] != .2 or config["threshold"] != .5:
        raise ValueError("Protocol fixes the training pool to 20% and the threshold to 0.5")
    if config["gumbel_tau"] <= 0 or config["workers"] < 0:
        raise ValueError("Invalid temperature/worker count")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing an empty manifest/metrics table")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def pixel_hash(image):
    return hashlib.sha256(np.asarray(image).tobytes() + str(image.size).encode()).hexdigest()


def _file_map(folder, masks=False):
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        match = re.search(r"ISIC_\d+", path.stem, re.I)
        if not match:
            continue
        identity = match[0].upper()
        if identity in result:
            raise ValueError(f"Duplicate image/mask identity {identity} in {folder}")
        result[identity] = path.resolve()
    return result


def resolve_dataset_root(reference):
    if str(reference).strip():
        root = Path(reference).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"ISIC_ROOT does not exist: {root}")
        return root
    matches = []
    for current, dirs, _files in os.walk("/kaggle/input"):
        dirs[:] = [name for name in dirs if name.lower() not in {"images", "masks", ".cache", ".git"}]
        if len(Path(current).parts) > 9:
            dirs[:] = []
        names = {name.lower() for name in dirs}
        if {"train", "val", "test"}.issubset(names) or {"train", "valid", "test"}.issubset(names):
            matches.append(Path(current))
        elif (any(name.lower().startswith("isic-2017_training_data") for name in dirs)
              and any(name.lower().startswith("isic-2017_validation_data") for name in dirs)
              and any(name.lower().startswith("isic-2017_test") and "data" in name.lower() for name in dirs)):
            matches.append(Path(current))
    if len(matches) != 1:
        raise FileNotFoundError(f"Set ISIC_ROOT to one exact ISIC-2017 dataset root; found {matches}")
    return matches[0].resolve()


def resolve_dino_assets(repo_reference, weights_reference, config):
    repo, weights = Path(repo_reference).expanduser(), Path(weights_reference).expanduser()
    filename = f'{config["backbone"]}_pretrain_lvd1689m-{net.BACKBONES[config["backbone"]][3]}.pth'
    if weights.is_dir():
        matches = list(weights.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(f"Need one exact {filename} in {weights}")
        weights = matches[0]
    if not repo.is_dir() or not weights.is_file():
        repos, checkpoints = [], []
        for current, dirs, files in os.walk("/kaggle/input"):
            dirs[:] = [name for name in dirs if name.lower() not in {"images", "masks", ".cache", ".git", "__pycache__"}]
            if len(Path(current).parts) > 10:
                dirs[:] = []
            if "backbones.py" in files and Path(current).parts[-2:] == ("dinov3", "hub"):
                repos.append(Path(current).parents[1])
            if filename in files:
                checkpoints.append(Path(current) / filename)
        if not repo.is_dir():
            if len(repos) != 1:
                raise FileNotFoundError(f"Set DINOV3_REPO_DIR explicitly; found {repos}")
            repo = repos[0]
        if not weights.is_file():
            if len(checkpoints) != 1:
                raise FileNotFoundError(f"Set DINOV3_WEIGHTS to {filename}; found {checkpoints}")
            weights = checkpoints[0]
    if not (repo / "dinov3/hub/backbones.py").is_file() or weights.name != filename:
        raise ValueError("Wrong DINOv3 source/checkpoint; refusing another backbone")
    return repo.resolve(), weights.resolve()


def load_isic(reference, config, groups_csv=""):
    root = resolve_dataset_root(reference)
    frames = {}
    counts = {"train": 2000, "val": 150, "test": 600}
    all_ids, cross_split_hashes = set(), {}
    for role, aliases, official in (("train", ("train", "training"), "Training"),
                                     ("val", ("val", "valid", "validation"), "Validation"),
                                     ("test", ("test", "testing"), "Test")):
        children = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
        prepared = [children[name] for name in aliases if name in children]
        if len(prepared) == 1:
            base = prepared[0]
            images, masks = base / "images", base / "masks"
        elif not prepared:
            versions = [official, "Test_v2"] if role == "test" else [official]
            pairs = [(root / f"ISIC-2017_{version}_Data", root / f"ISIC-2017_{version}_Part1_GroundTruth")
                     for version in versions]
            found = [pair for pair in pairs if pair[0].is_dir() and pair[1].is_dir()]
            if len(found) > 1:
                raise ValueError(f"Ambiguous original ISIC split folders: {found}")
            images, masks = found[0] if found else pairs[0]
        else:
            raise ValueError(f"Ambiguous {role} split folders: {prepared}")
        if not images.is_dir() or not masks.is_dir():
            raise FileNotFoundError(f"Need {role} image/mask folders: {images} and {masks}")
        image_map, mask_map = _file_map(images), _file_map(masks, True)
        if not image_map or set(image_map) != set(mask_map):
            raise ValueError(f"Unpaired {role} image/mask IDs: images={len(image_map)}, masks={len(mask_map)}")
        if config["strict_official_counts"] and len(image_map) != counts[role]:
            raise ValueError(f"Official ISIC2017 {role} should have {counts[role]} images; found {len(image_map)}")
        rows = []
        for image_id, image_path in tqdm(image_map.items(), desc=f"Verify ISIC {role}"):
            if image_id in all_ids:
                raise ValueError(f"Image ID overlaps official splits: {image_id}")
            all_ids.add(image_id)
            with Image.open(image_path) as source:
                if source.getexif().get(274, 1) not in {None, 1}:
                    raise ValueError("Dataset image has EXIF rotation; align its mask explicitly before using it")
                image = source.convert("RGB")
            with Image.open(mask_map[image_id]) as mask:
                if mask.size != image.size:
                    raise ValueError(f"Image/mask dimensions differ: {image_id}")
                values = np.unique(np.asarray(mask.convert("L")))
                if not set(values).issubset({0, 1, 255}):
                    raise ValueError(f"Expected binary Part1 lesion mask for {image_id}")
            digest = pixel_hash(image)
            if digest in cross_split_hashes and cross_split_hashes[digest] != role:
                raise ValueError(f"Identical image pixels occur in different splits: {image_id}")
            cross_split_hashes[digest] = role
            rows.append({"image_id": image_id, "image_path": str(image_path), "mask_path": str(mask_map[image_id]),
                "split": role, "image_sha256": io.file_sha256(image_path), "mask_sha256": io.file_sha256(mask_map[image_id]),
                "pixel_sha256": digest, "group_id": ""})
        frames[role] = rows
    if groups_csv:
        with Path(groups_csv).open(encoding="utf-8-sig", newline="") as handle:
            groups = list(csv.DictReader(handle))
        group_map = {row["image_id"]: row["group_id"].strip() for row in groups}
        if len(group_map) != len(groups) or set(group_map) != all_ids or not all(group_map.values()):
            raise ValueError("Group metadata must supply one nonempty group_id per ISIC image_id")
        groups_seen = {}
        for role, rows in frames.items():
            for row in rows:
                row["group_id"] = group_map[row["image_id"]]
                if row["group_id"] in groups_seen and groups_seen[row["group_id"]] != role:
                    raise ValueError("Patient/group overlaps official splits; resolve the protocol before training")
                groups_seen[row["group_id"]] = role
    rng = np.random.default_rng(config["seed"])
    positions = sorted(rng.choice(len(frames["train"]), max(2, round(.2 * len(frames["train"]))), replace=False).tolist())
    pool = [dict(frames["train"][index]) for index in positions]
    return frames, pool


def allowed_candidates(rows, pool, device):
    allowed = [[q["image_id"] != c["image_id"] and q["pixel_sha256"] != c["pixel_sha256"]
                and not (q.get("group_id") and q["group_id"] == c.get("group_id")) for c in pool] for q in rows]
    mask = torch.tensor(allowed, dtype=torch.bool, device=device)
    if not mask.any(1).all():
        raise ValueError("Self/group exclusion leaves a query without context")
    return mask


class SegmentationDataset(Dataset):
    def __init__(self, rows, config, training=False):
        self.rows, self.config, self.training = rows, config, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = io.read_rgb(row["image_path"])
        with Image.open(row["mask_path"]) as source:
            mask = source.convert("L")
        if self.training:
            scale = random.uniform(.8, 1.) ** .5
            width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
            left, top = random.randint(0, image.width - width), random.randint(0, image.height - height)
            box = (left, top, left + width, top + height)
            image, mask = image.crop(box), mask.crop(box)
            for method in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if random.random() < .5:
                    image, mask = image.transpose(method), mask.transpose(method)
        pixels = io.preprocess(image, self.config)
        mask = mask.resize((self.config["image_size"],) * 2, Image.Resampling.NEAREST)
        target = torch.from_numpy((np.asarray(mask).copy() > 0).astype(np.float32))[None]
        return pixels, target, row


def collate(items):
    images, masks, rows = zip(*items)
    return torch.stack(images), torch.stack(masks), list(rows)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(rows, config, training=False, generator=None):
    return DataLoader(SegmentationDataset(rows, config, training),
        batch_size=config["batch_size"] if training else config["eval_batch_size"], shuffle=training,
        num_workers=config["workers"], collate_fn=collate, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


@torch.no_grad()
def build_bank(model, pool, config, destination, backbone_sha):
    identity = object_hash({"backbone": backbone_sha, "image_size": config["image_size"],
        "pool": [(r["image_id"], r["image_sha256"]) for r in pool], "dtype": config["pool_token_dtype"]})
    path = Path(destination)
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["image_ids"] != [r["image_id"] for r in pool]:
            raise ValueError("Candidate cache identity changed; preserve the old run and use another output")
        return saved
    device = next(model.parameters()).device
    features, tokens = [], []
    model.eval()
    for start in tqdm(range(0, len(pool), config["feature_batch_size"]), desc="Frozen candidate bank"):
        images = torch.stack([io.preprocess(r["image_path"], config) for r in pool[start:start + config["feature_batch_size"]]]).to(device)
        initial = model.prepare(images)
        values, _ = model.encode_selector_tokens(initial)
        tokens.append(initial.cpu().to(getattr(torch, config["pool_token_dtype"])))
        features.append(values.cpu())
    saved = {"identity": identity, "image_ids": [r["image_id"] for r in pool],
             "tokens": torch.cat(tokens), "features": torch.cat(features)}
    atomic_torch(path, saved)
    return saved


class TextTable:
    def __init__(self, path):
        self.data = torch.load(path, map_location="cpu", weights_only=True)
        self.positions = {image_id: index for index, image_id in enumerate(self.data["image_ids"])}
        if len(self.positions) != len(self.data["image_ids"]):
            raise ValueError("Duplicate text IDs")
        values, valid = self.data["embeddings"], self.data["valid"]
        if not torch.isfinite(values).all() or torch.count_nonzero(values[~valid]):
            raise ValueError("Invalid captions must be finite zero vectors")

    def gather(self, rows, device):
        positions = [self.positions[row["image_id"]] for row in rows]
        return self.data["embeddings"][positions].to(device), self.data["valid"][positions].to(device)


def amp_context(device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _selection_inputs(model, rows, pool, bank, table, device):
    allowed = allowed_candidates(rows, pool, device)
    q_text = q_valid = c_text = c_valid = None
    if table is not None:
        q_text, q_valid = table.gather(rows, device)
        c_text, c_valid = table.gather(pool, device)
    return allowed, q_text, q_valid, c_text, c_valid


@torch.no_grad()
def evaluate(model, loader, pool, bank, config, table=None, desc="validation"):
    model.eval()
    device = next(model.parameters()).device
    totals = dict(loss=0., dice=0., iou=0., q_gate=0., c_gate=0.)
    predictions = []
    count = 0
    for images, masks, rows in tqdm(loader, desc=desc, leave=False):
        images, masks = images.to(device), masks.to(device)
        initial = model.prepare(images)
        global_query, _ = model.encode_selector_tokens(initial)
        allowed, qt, qv, ct, cv = _selection_inputs(model, rows, pool, bank, table, device)
        scores = model.score(global_query, bank["features"].to(device), allowed, qt, qv, ct, cv)
        positions = scores.argmax(1)
        context = bank["tokens"][positions.cpu()].to(device, dtype=torch.float32)
        logits, details = model.decode(initial, context, qt, qv,
            ct[positions] if ct is not None else None, cv[positions] if cv is not None else None)
        losses = net.segmentation_loss(logits, masks)
        dice, iou = net.binary_metrics(logits, masks, config["threshold"])
        for key, values in {"loss": losses, "dice": dice, "iou": iou, **details}.items():
            if key in totals:
                totals[key] += float(values.sum())
        count += len(rows)
        for index, row in enumerate(rows):
            predictions.append({"image_id": row["image_id"], "dice": float(dice[index]), "iou": float(iou[index]),
                "context_id": pool[int(positions[index])]["image_id"], "context_position": int(positions[index]),
                "q_gate": float(details["q_gate"][index]), "c_gate": float(details["c_gate"][index])})
    return {key: value / count for key, value in totals.items()}, predictions


def train_epoch(model, loader, pool, bank, table, optimizer, scaler, config, epoch):
    model.train()
    device = next(model.parameters()).device
    totals = {key: 0. for key in ("loss", "task_loss", "policy_loss", "reward", "dice", "iou", "q_gate", "c_gate")}
    count = 0
    iterator = iter(loader)
    steps = math.ceil(config["effective_batch_size"] / config["batch_size"])
    progress = tqdm(total=len(loader.dataset), desc=f"Epoch {epoch:03d} train", unit="image")
    while True:
        group = []
        for _ in range(steps):
            batch = next(iterator, None)
            if batch is None:
                break
            images, masks, rows = batch
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                initial = model.prepare(images)
                global_query, _ = model.encode_selector_tokens(initial)
                allowed, qt, qv, ct, cv = _selection_inputs(model, rows, pool, bank, table, device)
                scores = model.score(global_query, bank["features"].to(device), allowed, qt, qv, ct, cv)
                actions = torch.distributions.Categorical(logits=scores).sample()
                context = bank["tokens"][actions.cpu()].to(device, dtype=torch.float32)
                with amp_context(device):
                    # Query text is identical in BOTH reward terms. Only context
                    # image + its own caption is removed from the baseline.
                    alone, _ = model.decode(initial, None, qt, qv)
                    paired, _ = model.decode(initial, context, qt, qv,
                        ct[actions] if ct is not None else None, cv[actions] if cv is not None else None)
                reward = net.segmentation_loss(alone, masks) - net.segmentation_loss(paired, masks)
            group.append((initial.detach(), global_query.detach(), masks, rows, allowed, qt, qv, ct, cv, actions, reward))
        if not group:
            break
        # Standardize over the EFFECTIVE batch, not a physical batch of size 1.
        # No model/optimizer update occurs between policy sampling and log_prob.
        rewards = torch.cat([item[-1] for item in group])
        advantages = ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)).detach()
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for initial, global_query, masks, rows, allowed, qt, qv, ct, cv, actions, reward in group:
            scores = model.score(global_query, bank["features"].to(device), allowed, qt, qv, ct, cv)
            policy = torch.distributions.Categorical(logits=scores)
            weights = torch.nn.functional.gumbel_softmax(scores, tau=config["gumbel_tau"], hard=True, dim=1)
            context = net.select_bank(weights, bank["tokens"])
            c_text = weights @ ct if ct is not None else None
            c_valid = weights @ cv.float() if cv is not None else None
            with amp_context(device):
                logits, details = model.decode(initial, context, qt, qv, c_text, c_valid)
                task_loss = net.segmentation_loss(logits, masks).mean()
            policy_loss = -(policy.log_prob(actions) * advantages[offset:offset + len(rows)]).mean()
            loss = task_loss + config["policy_weight"] * policy_loss - config["entropy_weight"] * policy.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective; no checkpoint exported")
            scaler.scale(loss * len(rows) / len(rewards)).backward()
            dice, iou = net.binary_metrics(logits.detach(), masks, config["threshold"])
            for key, value in {"loss": loss, "task_loss": task_loss, "policy_loss": policy_loss,
                "reward": reward.mean(), "dice": dice.mean(), "iou": iou.mean(),
                "q_gate": details["q_gate"].mean(), "c_gate": details["c_gate"].mean()}.items():
                totals[key] += float(value.detach()) * len(rows)
            offset += len(rows)
            count += len(rows)
            progress.update(len(rows))
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["grad_clip"], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        del group
    progress.close()
    return {key: value / count for key, value in totals.items()}


def run_training(model, frames, pool, bank, config, output, identity, table=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "protocol.json"
    if protocol.is_file() and io.read_json(protocol)["identity"] != identity:
        raise ValueError("Run identity/config/data/source changed; do not overwrite or reuse old checkpoints")
    io.write_json(protocol, {"identity": identity, "config": config})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"], weight_decay=config["weight_decay"])
    def schedule(epoch):
        if epoch < config["warmup_epochs"]:
            return (epoch + 1) / max(1, config["warmup_epochs"])
        t = (epoch - config["warmup_epochs"]) / max(1, config["epochs"] - config["warmup_epochs"])
        return .5 * (1 + math.cos(math.pi * min(t, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    generator = torch.Generator().manual_seed(config["seed"])
    history, start, best, bad = [], 1, -1., 0
    if (output / "last.pt").is_file():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Wrong resume checkpoint identity")
        model.restore_trainable(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, start, best, bad = saved["history"], saved["epoch"] + 1, saved["best_dice"], saved["bad_epochs"]
        random.setstate(saved["python_rng"])
        np_state = saved["numpy_rng"]
        np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["loader_rng"])
    train_loader = make_loader(frames["train"], config, True, generator)
    val_loader = make_loader(frames["val"], config)
    for epoch in range(start, config["epochs"] + 1):
        if config["patience"] and bad >= config["patience"]:
            break
        train = train_epoch(model, train_loader, pool, bank, table, optimizer, scaler, config, epoch)
        val, _ = evaluate(model, val_loader, pool, bank, config, table)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}})
        if val["dice"] > best:
            best, bad = val["dice"], 0
            atomic_torch(output / "best.pt", {"identity": identity, "epoch": epoch, "config": config,
                "validation": val, "state": model.trainable_state()})
        else:
            bad += 1
        scheduler.step()
        numpy_rng = np.random.get_state()
        atomic_torch(output / "last.pt", {"identity": identity, "epoch": epoch, "best_dice": best, "bad_epochs": bad,
            "state": model.trainable_state(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "history": history, "python_rng": random.getstate(),
            "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [], "loader_rng": generator.get_state()})
        csv_write(output / "history.csv", history)
        print(f'{config["mode"]} epoch={epoch} val_dice={val["dice"]:.4f} val_iou={val["iou"]:.4f} '
              f'q_gate={val["q_gate"]:.4f} c_gate={val["c_gate"]:.4f}', flush=True)
    best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if best_state["identity"] != identity:
        raise ValueError("Best checkpoint belongs to another run")
    model.restore_trainable(best_state["state"])
    model.eval()
    return history, best_state


def source_identity(folder):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): io.file_sha256(p) for p in sorted(folder.rglob("*.py"))
             if not {".git", "__pycache__", ".cache"}.intersection(p.relative_to(folder).parts)}
    return object_hash(files)


def export_bundle(model, frames, pool, bank, config, output, repo, weights, runtime_dir, metrics,
                  *, caption_assets=None, caption_output=None, cold_verify=True, factory=None):
    """Ready is written only after strict state/key/hash, reload and attention parity."""
    output = Path(output)
    # Only completion metadata is needed here, not a second CPU copy of all
    # Adam moments. Restore tensor storage on meta to keep export RAM bounded.
    last = torch.load(output / "last.pt", map_location="meta", weights_only=True)
    if last["epoch"] < config["epochs"] and not (config["patience"] and last["bad_epochs"] >= config["patience"]):
        raise RuntimeError("Training is incomplete; resume before final deployment export")
    best_path = output / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.restore_trainable(best["state"])
    model.eval()
    root = output / "exports" / f'isic2017_{config["mode"]}_vitl16_448_web_bundle'
    root.mkdir(parents=True, exist_ok=True)
    io.write_json(root / "export_status.json", {"status": "staging"})
    if (root / "manifest.json").exists():
        (root / "manifest.json").unlink()
    (root / "models").mkdir(exist_ok=True)
    (root / "runtime").mkdir(exist_ok=True)
    io.write_json(root / "config.json", config)
    state = model.trainable_state()
    is_selector = lambda key: key.startswith("selector.") or key == "semantic_mix_logit"
    atomic_torch(root / "models/selector.pt", {key: value for key, value in state.items() if is_selector(key)})
    atomic_torch(root / "models/segmentor.pt", {key: value for key, value in state.items() if not is_selector(key)})
    backbone_file = "models/" + Path(weights).name
    shutil.copy2(weights, root / backbone_file)
    atomic_torch(root / "candidate_bank.pt", bank)
    shutil.copytree(repo, root / "vendor/dinov3", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".cache", "*.pyc", "*.pth", "*.pt", "*.ipynb", "data", "outputs"))
    for name in ("isic_segmentation_web.py", "tacs_visual_web.py", "train_isic_segmentation_web.py",
                 "isic_caption_web.py", "isic_caption_core.py"):
        path = Path(runtime_dir) / name
        if path.is_file():
            shutil.copy2(path, root / "runtime" / name)
    exported_pool = []
    for index, row in enumerate(tqdm(pool, desc="Export real context images")):
        target = root / "candidate_images" / (row["image_id"] + Path(row["image_path"]).suffix.lower())
        target.parent.mkdir(exist_ok=True)
        # Preserve real source bytes; converting high-resolution dermoscopy
        # JPEGs to RGB PNG unnecessarily expands the deployment by gigabytes.
        shutil.copy2(row["image_path"], target)
        exported_pool.append({"image_id": row["image_id"], "position": index, "image": target.relative_to(root).as_posix(),
            "pixel_sha256": row["pixel_sha256"], "group_id": row.get("group_id", ""), "split": "train"})
    io.write_json(root / "candidate_pool.json", exported_pool)
    gt_index, caption_index = {}, {}
    for role, rows in frames.items():
        portable = []
        for row in rows:
            target = root / "ground_truth_masks" / f'{row["image_id"]}.png'
            target.parent.mkdir(exist_ok=True)
            with Image.open(row["mask_path"]) as image:
                Image.fromarray((np.asarray(image.convert("L")) > 0).astype(np.uint8) * 255).save(target)
            gt_index[row["pixel_sha256"]] = {"image_id": row["image_id"], "split": role,
                "mask": target.relative_to(root).as_posix(), "group_id": row.get("group_id", "")}
            caption_index[row["pixel_sha256"]] = row["image_id"]
            portable.append({key: value for key, value in row.items() if key not in {"image_path", "mask_path"}})
        csv_write(root / f"{role}_split.csv", portable)
    io.write_json(root / "ground_truth_index.json", gt_index)
    example = frames["val"][0]
    (root / "example").mkdir(exist_ok=True)
    io.read_rgb(example["image_path"]).save(root / "example/query.png")
    shutil.copy2(output / "history.csv", root / "history.csv")
    io.write_json(root / "metrics.json", metrics)
    io.write_json(root / "training_identity.json", {"identity": best["identity"], "epoch": best["epoch"],
        "best_checkpoint_sha256": io.file_sha256(best_path), "validation": best["validation"],
        "patient_groups_available": bool(frames["train"][0].get("group_id")), "source": source_identity(runtime_dir)})
    if config["mode"] == "mtacs":
        if not caption_assets or not caption_output:
            raise ValueError("M-TACS export requires actual Florence/CLIP assets and registered captions")
        for key in ("florence", "clip"):
            shutil.copytree(caption_assets[key], root / "caption_models" / key, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".cache", ".git", "__pycache__", "*.pyc"))
        caption_output = Path(caption_output)
        shutil.copy2(caption_output / "captions.csv", root / "captions.csv")
        shutil.copy2(caption_output / "text_cache/clip_text.pt", root / "text_embeddings.pt")
        with (root / "captions.csv").open(encoding="utf-8", newline="") as handle:
            lookup = {row["image_id"]: row for row in csv.DictReader(handle)}
        io.write_json(root / "caption_lookup.json", lookup)
        shutil.copy2(caption_output / "web_caption_settings.json", root / "caption_settings.json")
        for directory in ("generation_protocol", "text_cache"):
            for name in ("protocol.json", "status.json"):
                path = caption_output / directory / name
                if path.is_file():
                    target = root / "caption_provenance" / directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        wheels = caption_output / "caption_runtime/wheels"
        if wheels.is_dir():
            shutil.copytree(wheels, root / "offline_wheels", dirs_exist_ok=True)
        from isic_caption_web import probe_export
        fresh_caption_check = probe_export(root, "cpu", caption_assets["offline_assets_root"], output / "export_caption_runtime")
        io.write_json(root / "example/fresh_caption_check.json", fresh_caption_check)
    versions = {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                "device": str(next(model.parameters()).device)}
    io.write_json(root / "environment.json", versions)
    (root / "requirements.txt").write_text("torch>=2.5\nnumpy\nPillow\ntqdm\n# M-TACS caption child only:\ntransformers==4.49.0\naccelerate\neinops\ntimm\n", encoding="utf-8")
    readme = f'''# ISIC2017 {config["mode"]}: complete segmentation deployment bundle

Backbone: {config["backbone"]}, frozen; image size {config["image_size"]}; one context from the fixed training-only pool.
Model parts: models/ official DINO checkpoint + selector.pt + segmentor.pt. All three are required.
The segmentor component includes EVERY per-block context module, DPT, and M-TACS query/candidate text projectors/gates.
candidate_bank.pt is aligned exactly to candidate_pool.json and the actual candidate_images/.
Ground-truth masks are DISPLAY/EVALUATION annotations, never context/model inputs.

Python: add runtime/ to sys.path, import isic_segmentation_web, construct Predictor(bundle_path, device).
Use .predict(image) for the mask, probability array, overlay, selected candidate and attention.
For new M-TACS images, pass caption_engine=BundledCaptionEngine(bundle_path, device) from isic_caption_web
in a separate compatible Transformers 4.49 worker. Cached known images can use verified pixel bindings.
CLI: python runtime/isic_segmentation_web.py --bundle . --image example/query.png --output prediction --device cpu
Add --captions for a new M-TACS image when running inside the compatible caption environment.
The environment must supply compatible PyTorch/Pillow/NumPy; this ZIP does not contain a Python interpreter.
Original model/source/dataset licenses remain applicable. Vendor source includes its original license files.

Attention is last-block CLS-to-patch, averaged over heads. It is not Grad-CAM or a causal/diagnostic claim.
The selector map describes its frozen image encoder; the learned utility projections and semantic score are not token explanations.
The binary mask is the model's lesion-region prediction, not a diagnosis or a calibrated probability of malignancy.
This is a declared reconstruction of the published architecture, with an explicitly implemented lightweight DPT
and learned projection utility head. It is not a claim of byte-identical unpublished author implementation.
Paper settings were S/16, 256, 3 seeds. This run's exact settings and metrics are in config.json / metrics.json.
Training best.pt/last.pt remain OUTSIDE this ZIP; preserve the full run directory to resume training.
'''
    (root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {"schema": net.SCHEMA, "status": "staged", "mode": config["mode"], "backbone_file": backbone_file,
                "caption_pixel_index": caption_index if config["mode"] == "mtacs" else {}, "files": {}}
    def bind_files():
        manifest["files"] = {p.relative_to(root).as_posix(): io.file_sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in {"manifest.json", "export_status.json"} and "__pycache__" not in p.parts}
        io.write_json(root / "manifest.json", manifest)
    bind_files()
    # Independent model construction + split checkpoint reload, then attention
    # invariance and a cold-process replay without original Input paths.
    predictor = net.Predictor(root, "cpu", factory=factory, _allow_staged=True)
    plain = predictor.predict(root / "example/query.png", return_attention=False)
    explained = predictor.predict(root / "example/query.png", return_attention=True)
    if not torch.allclose(plain["logits"], explained["logits"], atol=1e-5, rtol=1e-5):
        raise ValueError("Attention capture changed segmentation logits")
    # Compare against the trained in-memory model on the SAME quantized bank.
    device = next(model.parameters()).device
    images = io.preprocess(root / "example/query.png", config)[None].to(device)
    qinitial = model.prepare(images)
    qglobal, _ = model.encode_selector_tokens(qinitial)
    table = TextTable(root / "text_embeddings.pt") if config["mode"] == "mtacs" else None
    allowed, qt, qv, ct, cv = _selection_inputs(model, [example], pool, bank, table, device)
    with torch.no_grad():
        scores = model.score(qglobal, bank["features"].to(device), allowed, qt, qv, ct, cv)
        chosen = scores.argmax(1)
        original, _ = model.decode(qinitial, bank["tokens"][chosen.cpu()].to(device, dtype=torch.float32), qt, qv,
            ct[chosen] if ct is not None else None, cv[chosen] if cv is not None else None)
    if int(chosen) != plain["evidence"]["candidate"]["position"] or not torch.allclose(original.cpu(), plain["logits"], atol=5e-4, rtol=5e-4):
        raise ValueError("Reloaded export differs from the trained best model")
    atomic_torch(root / "example/expected_logits.pt", plain["logits"])
    np.save(root / "example/probabilities.npy", explained.pop("probabilities"))
    explained.pop("logits")
    io.write_json(root / "example/prediction.json", explained)
    del predictor
    bind_files()
    if cold_verify:
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH", None)
        command = [sys.executable, str(root / "runtime/isic_segmentation_web.py"), "--bundle", str(root),
            "--image", str(root / "example/query.png"), "--output", str(output / "cold_reload_check"), "--device", "cpu", "--verify-staged"]
        completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError("Cold offline reload failed: " + completed.stderr[-5000:])
        cold = torch.load(output / "cold_reload_check/logits.pt", map_location="cpu", weights_only=True)
        if not torch.allclose(cold, plain["logits"], atol=1e-5, rtol=1e-5):
            raise ValueError("Cold-process mask logits differ")
    manifest["status"] = "ready" if cold_verify else "fixture_unverified"
    manifest["verification"] = {"state_reload": True, "attention_parity": True, "trained_model_parity": True,
                                "cold_offline_reload": cold_verify}
    bind_files()
    io.write_json(root / "export_status.json", {"status": manifest["status"]})
    if not cold_verify:
        return root, None
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive.with_suffix(".zip.tmp"), "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zipped.write(path, path.relative_to(root.parent).as_posix())
    archive.with_suffix(".zip.tmp").replace(archive)
    io.write_json(output / "export_receipt.json", {"bundle": str(root), "zip": str(archive),
        "zip_sha256": io.file_sha256(archive), "best_sha256": io.file_sha256(best_path), "status": "ready"})
    return root, archive
