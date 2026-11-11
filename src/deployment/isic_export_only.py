"""Recover an ISIC TACS checkpoint snapshot without training or changing its protocol."""
from __future__ import annotations

import base64
import csv
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import torch

VERSION = "isic_tacs_snapshot_export_v1"
RUNTIME_FILES = ("isic_segmentation_web.py", "tacs_visual_web.py", "train_isic_segmentation_web.py")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(folder):
    folder = Path(folder)
    return object_hash({p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*.py"))
                        if not {".git", "__pycache__", ".cache"}.intersection(p.relative_to(folder).parts)})


def copy_verified(source, target):
    source, target = Path(source), Path(target)
    digest = sha(source)
    if target.is_file():
        if sha(target) != digest:
            raise ValueError(f"Existing export artifact differs: {target}. Preserve it and select another OUTPUT_DIR.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    if sha(temporary) != digest:
        raise ValueError(f"Copy checksum mismatch: {source}")
    temporary.replace(target)


def discover_run(reference="", roots=(Path("/kaggle/input"),), *, mode="tacs"):
    explicit = str(reference).strip()
    root = Path(explicit).expanduser() if explicit else None
    if root is not None:
        if not root.exists():
            raise FileNotFoundError(f"SOURCE_RUN_DIR does not exist: {root}")
        if root.is_file():
            if root.name != "best.pt":
                raise ValueError("SOURCE_RUN_DIR must be a saved run directory or best.pt")
            root = root.parent
        if (root / "best.pt").is_file():
            return root.resolve()
        roots = (root,)
    found = []
    for search in roots:
        if not Path(search).is_dir():
            continue
        for current, dirs, files in os.walk(search):
            dirs[:] = [name for name in dirs if name not in {".git", ".cache", "__pycache__", "images", "masks", "caption_runtime", "vendor"}]
            if len(Path(current).relative_to(search).parts) >= 9:
                dirs[:] = []
            if {"best.pt", "protocol.json", "run_binding.json"}.issubset(files):
                protocol = read_json(Path(current) / "protocol.json")
                if protocol.get("config", {}).get("mode") == mode and protocol["config"].get("schema") == "isic2017_segmentation_web_v1":
                    found.append(Path(current).resolve())
    found = sorted(set(found))
    if len(found) != 1:
        raise FileNotFoundError(f"Set SOURCE_RUN_DIR to the exact saved ISIC {mode} run. Found: {found}")
    return found[0]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not re.fullmatch(r"ISIC_\d+", row.get("image_id", "")):
            raise ValueError(f"Unsafe or unexpected ISIC ID in {path}: {row.get('image_id')}")
        for key in ("image_sha256", "mask_sha256", "pixel_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row.get(key, "")):
                raise ValueError(f"Missing/invalid {key} for {row['image_id']}")
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError(f"Empty/duplicate image IDs in {path}")
    return rows


def validate_metadata(run, *, production=True, mode="tacs"):
    """Uses meta storage: best/last optimizer tensors are never materialized here."""
    run = Path(run)
    for name in ("best.pt", "protocol.json", "run_binding.json", "candidate_pool.csv", "train_split.csv", "val_split.csv", "test_split.csv"):
        if not (run / name).is_file():
            raise FileNotFoundError(f"Missing saved sidecar: {run / name}")
    protocol, binding = read_json(run / "protocol.json"), read_json(run / "run_binding.json")
    checkpoint = torch.load(run / "best.pt", map_location="meta", weights_only=True)
    if not {"state", "config", "identity", "epoch", "validation"}.issubset(checkpoint):
        raise ValueError("Unsupported best.pt schema; do not rename keys or guess the model")
    config = checkpoint["config"]
    if (config != protocol.get("config") or config != binding.get("config")
            or checkpoint["identity"] != protocol.get("identity") or object_hash(binding) != checkpoint["identity"]):
        raise ValueError("best.pt / protocol / run_binding belong to different runs")
    if mode not in {"tacs", "mtacs"} or config.get("mode") != mode or config.get("schema") != "isic2017_segmentation_web_v1" or config.get("context_top_k") != 1:
        raise ValueError(f"Expected ISIC {mode} segmentation, K=1")
    if production and (config.get("backbone") != "dinov3_vitl16" or config.get("feature_dim") != 1024 or config.get("image_size") != 448):
        raise ValueError("Expected the saved ViT-L/16 448 segmentation architecture")
    if not checkpoint["state"] or any(key.startswith("backbone.") for key in checkpoint["state"]):
        raise ValueError("Expected learned segmentation/selector weights and a separate frozen DINO checkpoint")
    frames = {role: read_rows(run / f"{role}_split.csv") for role in ("train", "val", "test")}
    for role, rows in frames.items():
        compact = [{key: row[key] for key in ("image_id", "image_sha256", "mask_sha256", "group_id")} for row in rows]
        if compact != binding["splits"].get(role) or any(row["split"] != role for row in rows):
            raise ValueError(f"Saved {role} split differs from the immutable training binding")
    combined = sum(frames.values(), [])
    if len({r["image_id"] for r in combined}) != len(combined):
        raise ValueError("Saved splits overlap by image ID")
    pixel_records = {}
    for row in combined:
        same = pixel_records.setdefault(row["pixel_sha256"], [])
        if same and same[0]["split"] != row["split"]:
            raise ValueError("Identical image pixels overlap across saved splits")
        same.append(row)
    pool = read_rows(run / "candidate_pool.csv")
    train = {r["image_id"]: r for r in frames["train"]}
    if [r["image_id"] for r in pool] != binding.get("pool_ids") or len(pool) < 2:
        raise ValueError("Candidate order/identity changed")
    if any(r != train.get(r["image_id"]) for r in pool):
        raise ValueError("Candidates must be exact saved training records")
    expected_count = round(len(train) * config["pool_fraction"])
    if len(pool) != expected_count:
        raise ValueError("The recorded candidate pool is not the saved 20% training pool")
    history = []
    if (run / "history.csv").is_file():
        with (run / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in history]
        if epochs != list(range(1, len(history) + 1)):
            raise ValueError("History epochs are incomplete/out of order")
        selected = next((r for r in history if int(r["epoch"]) == int(checkpoint["epoch"])), None)
        if selected is None or abs(float(selected["val_dice"]) - checkpoint["validation"]["dice"]) > 1e-7:
            raise ValueError("Best validation score/epoch does not match history.csv")
    last_meta = None
    if (run / "last.pt").is_file():
        last = torch.load(run / "last.pt", map_location="meta", weights_only=True)
        if last.get("identity") != checkpoint["identity"]:
            raise ValueError("last.pt belongs to another run")
        last_meta = {key: last.get(key) for key in ("epoch", "bad_epochs", "best_dice")}
        if history and int(last["epoch"]) != int(history[-1]["epoch"]):
            raise ValueError("last.pt and history.csv have different completed epochs")
        del last
    completed = int(last_meta["epoch"]) if last_meta else int(history[-1]["epoch"]) if history else None
    if completed is not None and completed < int(checkpoint["epoch"]):
        raise ValueError("Best checkpoint is newer than the recorded training progress")
    finished = None if completed is None else completed >= config["epochs"] or bool(
        config.get("patience") and last_meta and last_meta["bad_epochs"] >= config["patience"])
    snapshot = {"training_complete": finished, "requested_epochs": config["epochs"], "completed_epochs": completed,
                "best_epoch": int(checkpoint["epoch"]), "validation_from_checkpoint": checkpoint["validation"],
                "progress_source": "last.pt_meta" if last_meta else "history.csv" if history else "best_epoch_only",
                "split_counts": {key: len(rows) for key, rows in frames.items()}, "pool_count": len(pool),
                "repeated_pixels_within_split": [{"split": rows[0]["split"], "image_ids": [r["image_id"] for r in rows]}
                    for rows in pixel_records.values() if len(rows) > 1],
                "snapshot_note": "Use exact saved splits, including the original run's duplicate exclusions; do not resplit or resample."}
    del checkpoint
    return config, binding, frames, pool, snapshot


def stage_runtime(run, output, fallback, expected):
    folder = Path(output) / "export_runtime"
    folder.mkdir(parents=True, exist_ok=True)
    original = Path(run) / "embedded_runtime"
    supplied = sorted(p.name for p in original.glob("*.py")) if original.is_dir() else []
    if supplied:
        if source_identity(original) != expected:
            raise ValueError("Original embedded_runtime no longer matches this trained run")
        sources = {name: (original / name).read_bytes() for name in supplied}
    else:
        sources = {name: value.encode("utf-8") for name, value in fallback.items()}
    if not set(RUNTIME_FILES).issubset(sources):
        raise FileNotFoundError("Attach the original embedded_runtime files with best.pt")
    if object_hash({name: hashlib.sha256(value).hexdigest() for name, value in sorted(sources.items())}) != expected:
        raise ValueError("Embedded fallback is not the exact original runtime for this checkpoint")
    for name, content in sources.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Unsafe runtime filename")
        compile(content, name, "exec")
        path = folder / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Export runtime changed; use a fresh OUTPUT_DIR")
        if not path.exists():
            path.write_bytes(content)
    if source_identity(folder) != expected:
        raise ValueError("Staged runtime contains unregistered source")
    sys.path.insert(0, str(folder.resolve()))
    for name in RUNTIME_FILES:
        loaded = sys.modules.get(Path(name).stem)
        if loaded and Path(loaded.__file__).resolve() != (folder / name).resolve():
            raise RuntimeError("Another ISIC runtime is imported. Restart the kernel before running this notebook.")
    net = importlib.import_module("isic_segmentation_web")
    work = importlib.import_module("train_isic_segmentation_web")
    return folder, net, work


def relocate_frames(frames, dataset, work, io):
    dataset = work.resolve_dataset_root(dataset)
    index = {}
    for current, dirs, files in os.walk(dataset):
        dirs[:] = [n for n in dirs if not n.startswith('.')]
        for name in files:
            if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                match = re.search(r"ISIC_\d+", name, re.I)
                if match:
                    index.setdefault(match[0].upper(), []).append(Path(current) / name)
    relocated, digests = {}, {}
    def file_digest(path):
        if path not in digests:
            digests[path] = sha(path)
        return digests[path]
    for role, rows in frames.items():
        result = []
        for row in work.tqdm(rows, desc=f"Verify saved {role} images/masks"):
            found = {}
            for field, digest in (("image_path", row["image_sha256"]), ("mask_path", row["mask_sha256"])):
                previous = Path(row[field])
                candidates = ([previous] if previous.is_file() and previous.resolve().is_relative_to(dataset) else [])
                candidates += sorted(index.get(row["image_id"], []))
                unique = dict.fromkeys(p.resolve() for p in candidates)
                match = next((p for p in unique if file_digest(p) == digest), None)
                if match is None:
                    raise FileNotFoundError(f"No content-matching {field} for {row['image_id']} below {dataset}")
                found[field] = str(match)
            image = io.read_rgb(found["image_path"])
            if work.pixel_hash(image) != row["pixel_sha256"]:
                raise ValueError("Saved pixel identity mismatch: " + row["image_id"])
            with Image.open(found["mask_path"]) as mask:
                if mask.size != image.size or not set(np.unique(np.asarray(mask.convert("L"))).tolist()).issubset({0, 1, 255}):
                    raise ValueError("Invalid binary mask/alignment: " + row["image_id"])
            result.append(dict(row, **found))
        relocated[role] = result
    return dataset, relocated


def prepare(source_run, dataset_root, dino_repo, dino_weights, output, fallback_sources,
            *, device="cuda", feature_batch_size=4, run_full_test=False, factory=None, mode="tacs"):
    run = discover_run(source_run, mode=mode)
    output = Path(output).expanduser().resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("OUTPUT_DIR must be separate from the read-only saved run")
    if int(feature_batch_size) < 1:
        raise ValueError("FEATURE_BATCH_SIZE must be positive")
    config, binding, frames, pool, snapshot = validate_metadata(run, production=factory is None, mode=mode)
    output.mkdir(parents=True, exist_ok=True)
    runtime, net, work = stage_runtime(run, output, fallback_sources, binding["runtime"])
    repo, weights = work.resolve_dino_assets(dino_repo, dino_weights, config) if factory is None else (Path(dino_repo).resolve(), Path(dino_weights).resolve())
    if sha(weights) != binding["assets"]["backbone_sha256"] or source_identity(repo) != binding["assets"]["dino_source"]:
        raise ValueError("DINO source/frozen pretrained weights differ from the trained checkpoint binding")
    dataset, frames = relocate_frames(frames, dataset_root, work, net.io_utils)
    by_id = {row["image_id"]: row for row in frames["train"]}
    pool = [by_id[row["image_id"]] for row in pool]
    best_sha = sha(run / "best.pt")
    identity = {"exporter": VERSION if mode == "tacs" else "isic_mtacs_snapshot_export_v1", "run_identity": object_hash(binding), "source_best_sha256": best_sha,
                "runtime_identity": binding["runtime"], "assets": binding["assets"],
                "source_sidecars": {name: sha(run / name) for name in ("protocol.json", "run_binding.json", "candidate_pool.csv", "train_split.csv", "val_split.csv", "test_split.csv")},
                "run_full_test": bool(run_full_test), "production": factory is None}
    lock = output / "snapshot_binding.json"
    if lock.exists() and read_json(lock) != identity:
        raise ValueError("This export directory belongs to another checkpoint/options; choose a new OUTPUT_DIR")
    write_json(lock, identity)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Enable a GPU or select DEVICE='cpu'.")
    stage = output / f"bundle/isic2017_{mode}_vitl16_448_web_bundle"
    write_json(output / "snapshot_summary.json", snapshot)
    print(json.dumps(snapshot, indent=2))
    return {"run": run, "output": output, "stage": stage, "runtime": runtime, "net": net, "work": work,
            "config": config, "binding": binding, "identity": identity, "snapshot": snapshot,
            "frames": frames, "pool": pool, "repo": repo, "weights": weights, "dataset": dataset,
            "device": device, "feature_batch_size": int(feature_batch_size), "run_full_test": bool(run_full_test), "factory": factory}


def validate_bank(bank, pool, config, backbone_sha, prefix):
    identity = object_hash({"backbone": backbone_sha, "image_size": config["image_size"],
                           "pool": [(r["image_id"], r["image_sha256"]) for r in pool], "dtype": config["pool_token_dtype"]})
    tokens, features = bank["tokens"], bank["features"]
    if bank.get("identity") != identity or bank.get("image_ids") != [r["image_id"] for r in pool]:
        raise ValueError("Candidate bank does not match the frozen backbone/preprocessing/original pool order")
    if tokens.shape != (len(pool), prefix + (config["image_size"] // 16)**2, config["feature_dim"]) or features.shape != (len(pool), config["feature_dim"]):
        raise ValueError("Candidate bank token/CLS shape mismatch")
    if tokens.dtype != getattr(torch, config["pool_token_dtype"]) or features.dtype != torch.float32:
        raise ValueError("Candidate bank precision differs from training")
    for start in range(0, len(pool), 8):
        if not torch.isfinite(tokens[start:start + 8]).all() or not torch.isfinite(features[start:start + 8]).all():
            raise ValueError("Candidate bank contains nonfinite values")


def stage_manifest(ctx, status="staged", verification=None):
    stage = ctx["stage"]
    files = {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob("*"))
             if p.is_file() and p.name not in {"manifest.json", "export_status.json"} and "__pycache__" not in p.parts and not p.name.endswith('.tmp')}
    manifest = {"schema": ctx["net"].SCHEMA, "mode": ctx["config"]["mode"], "status": status,
                "backbone_file": "models/" + ctx["weights"].name, "caption_pixel_index": ctx.get("caption_pixel_index", {}),
                "training_snapshot": ctx["snapshot"], "verification": verification or {}, "files": files}
    write_json(stage / "manifest.json", manifest)
    write_json(stage / "export_status.json", {"status": status, "training_complete": ctx["snapshot"]["training_complete"]})


def build(ctx):
    """Restore best once, reuse/rebuild the frozen bank, and stage portable assets."""
    net, work, config = ctx["net"], ctx["work"], ctx["config"]
    if config["mode"] == "mtacs" and ("text_table" not in ctx or "stage_extra" not in ctx):
        raise ValueError("M-TACS requires the caption-aware exporter and its original trained text cache")
    stage, output = ctx["stage"], ctx["output"]
    stage.mkdir(parents=True, exist_ok=True)
    # Any retry revalidates the export instead of trusting an old ready marker.
    (stage / "manifest.json").unlink(missing_ok=True)
    write_json(stage / "export_status.json", {"status": "staging"})
    print("Restoring the original frozen backbone + all learned best.pt parameters (no training)")
    model = (ctx["factory"](config) if ctx["factory"] else net.build_model(ctx["repo"], config, ctx["weights"])).to(ctx["device"]).eval()
    saved = torch.load(ctx["run"] / "best.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != ctx["identity"]["run_identity"] or sha(ctx["run"] / "best.pt") != ctx["identity"]["source_best_sha256"]:
        raise ValueError("best.pt changed after preflight")
    model.restore_trainable(saved["state"])
    is_selector = lambda key: key.startswith("selector.") or key == "semantic_mix_logit"
    for name, chosen in (("selector.pt", True), ("segmentor.pt", False)):
        work.atomic_torch(stage / "models" / name, {k: v for k, v in saved["state"].items() if is_selector(k) == chosen})
    del saved
    copy_verified(ctx["weights"], stage / "models" / ctx["weights"].name)
    bank_path = ctx["run"] / "shared/candidate_bank.pt"
    if bank_path.is_file():
        print("Reusing the original frozen candidate token/CLS bank")
        bank = torch.load(bank_path, map_location="cpu", weights_only=True, mmap=True)
    else:
        bank_path = output / "rebuilt_candidate_bank.pt"
        print("Rebuilding only the frozen candidate bank from the SAME saved pool")
        bank = work.build_bank(model, ctx["pool"], dict(config, feature_batch_size=ctx["feature_batch_size"]), bank_path,
                               ctx["binding"]["assets"]["backbone_sha256"])
    validate_bank(bank, ctx["pool"], config, ctx["binding"]["assets"]["backbone_sha256"], model.prefix)
    copy_verified(bank_path, stage / "candidate_bank.pt")
    write_json(stage / "config.json", config)
    for p in ctx["runtime"].glob("*.py"):
        copy_verified(p, stage / "runtime" / p.name)
    shutil.copytree(ctx["repo"], stage / "vendor/dinov3", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__", "*.pyc", "*.pth", "*.pt", "*.ipynb"))
    if source_identity(stage / "vendor/dinov3") != ctx["binding"]["assets"]["dino_source"]:
        raise ValueError("Exported DINO source differs from the saved source identity")
    exported_pool = []
    for position, row in enumerate(ctx["pool"]):
        path = stage / "candidate_images" / (row["image_id"] + Path(row["image_path"]).suffix.lower())
        copy_verified(row["image_path"], path)
        exported_pool.append({"image_id": row["image_id"], "position": position, "image": path.relative_to(stage).as_posix(),
                              "pixel_sha256": row["pixel_sha256"], "group_id": row["group_id"], "split": "train"})
    write_json(stage / "candidate_pool.json", exported_pool)
    gt, pixel_groups = {}, {}
    for role, rows in ctx["frames"].items():
        portable = []
        for row in work.tqdm(rows, desc=f"Export {role} ground-truth masks"):
            path = stage / "ground_truth_masks" / (row["image_id"] + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(row["mask_path"]) as image:
                Image.fromarray((np.asarray(image.convert("L")) > 0).astype(np.uint8) * 255).save(path)
            gt[row["pixel_sha256"]] = {"image_id": row["image_id"], "split": role, "mask": path.relative_to(stage).as_posix(), "group_id": row["group_id"]}
            pixel_groups.setdefault(row["pixel_sha256"], []).append(row)
            portable.append({k: v for k, v in row.items() if k not in {"image_path", "mask_path"}})
        work.csv_write(stage / f"{role}_split.csv", portable)
    ambiguities = []
    for digest, rows in pixel_groups.items():
        if len(rows) < 2:
            continue
        groups = {row["group_id"] for row in rows}
        masks = {sha(stage / "ground_truth_masks" / (row["image_id"] + ".png")) for row in rows}
        record = {"pixel_sha256": digest, "split": rows[0]["split"], "image_ids": [r["image_id"] for r in rows]}
        if len(masks) > 1:
            if any(groups):
                raise ValueError("Conflicting ground-truth aliases with group metadata need a group-aware ambiguity adapter")
            gt.pop(digest)
            record["automatic_ground_truth"] = "omitted_conflicting_masks"
        else:
            if len(groups) > 1:
                raise ValueError("Identical-pixel aliases have conflicting group metadata")
            gt[digest]["image_id"] = None
            gt[digest]["image_ids"] = record["image_ids"]
            record["automatic_ground_truth"] = "same_mask_image_id_ambiguous"
        ambiguities.append(record)
    # Preserve every recorded test row and mask. Ambiguous uploaded pixels may
    # not silently pick one of several different annotations as ground truth.
    write_json(stage / "ground_truth_ambiguities.json", ambiguities)
    write_json(stage / "ground_truth_index.json", gt)
    source_folder = stage / "provenance/original_run"
    for name in ("protocol.json", "run_binding.json", "candidate_pool.csv", "train_split.csv", "val_split.csv", "test_split.csv", "history.csv"):
        if (ctx["run"] / name).is_file():
            copy_verified(ctx["run"] / name, source_folder / name)
    write_json(stage / "training_snapshot.json", ctx["snapshot"])
    write_json(stage / "export_provenance.json", ctx["identity"])
    write_json(stage / "attention_config.json", {"method": net.ATTENTION_METHOD, "grid": [config["image_size"] // 16]*2,
        "roles": ["query_segmentor", "candidate_selector"], "separate_attention_model": False,
        "explanation": "Last-block mean-head CLS-to-patch self-attention, not Grad-CAM or causal attribution"})
    if ctx.get("stage_extra"):
        ctx["stage_extra"](ctx)
    metrics = {"dataset": "ISIC2017", "task": "binary_segmentation", "mode": config["mode"], "threshold": config["threshold"],
        "validation_from_checkpoint": ctx["snapshot"]["validation_from_checkpoint"], "best_epoch": ctx["snapshot"]["best_epoch"],
        "training_complete": ctx["snapshot"]["training_complete"], "completed_epochs": ctx["snapshot"]["completed_epochs"],
        "split_counts": ctx["snapshot"]["split_counts"], "test": None, "test_status": "not_run_export_only"}
    if ctx["run_full_test"]:
        values, rows = work.evaluate(model, work.make_loader(ctx["frames"]["test"], dict(config, workers=0, eval_batch_size=1)),
            ctx["pool"], bank, config, table=ctx.get("text_table"), desc="Optional test / recovered best checkpoint")
        metrics.update(test=values, test_status="evaluated_snapshot", aggregation="mean_per_image_at_network_resolution")
        work.csv_write(stage / "test_predictions.csv", rows)
    write_json(stage / "metrics.json", metrics)
    example = ctx.get("example_record", ctx["frames"]["val"][0])
    example_path = stage / "example" / ("query" + Path(example["image_path"]).suffix.lower())
    copy_verified(example["image_path"], example_path)
    with torch.no_grad():
        initial = model.prepare(net.io_utils.preprocess(example["image_path"], config)[None].to(ctx["device"]))
        global_query, _ = model.encode_selector_tokens(initial)
        allowed = work.allowed_candidates([example], ctx["pool"], ctx["device"])
        q_text = q_valid = c_text = c_valid = None
        if ctx.get("text_table") is not None:
            q_text, q_valid = ctx["text_table"].gather([example], ctx["device"])
            c_text, c_valid = ctx["text_table"].gather(ctx["pool"], ctx["device"])
        scores = model.score(global_query, bank["features"].to(ctx["device"]), allowed, q_text, q_valid, c_text, c_valid)
        position = int(scores.argmax(1))
        logits, _ = model.decode(initial, bank["tokens"][position:position + 1].to(ctx["device"], dtype=torch.float32),
            q_text, q_valid, c_text[position:position + 1] if c_text is not None else None,
            c_valid[position:position + 1] if c_valid is not None else None)
    work.atomic_torch(stage / "example/expected_logits.pt", logits.detach().cpu())
    write_json(stage / "example/reference.json", {"context_position": position, "context_id": ctx["pool"][position]["image_id"],
        "query_id": example["image_id"], "checkpoint_sha256": ctx["identity"]["source_best_sha256"]})
    if not (stage / "requirements.txt").exists():
        (stage / "requirements.txt").write_text("torch>=2.5\nnumpy\nPillow\ntqdm\n", encoding="utf-8")
    ctx.update(example_path=example_path)
    del model, bank, logits, initial, global_query, allowed, scores, q_text, q_valid, c_text, c_valid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stage_manifest(ctx)
    print("All model/data artifacts staged. Ready will be set only after independent reload + attention/logit verification.")
    return ctx


COLD_SOURCE = r'''import base64, json, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
root, image, device = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
sys.path.insert(0, str(root / "runtime"))
import isic_segmentation_web as net
print("Cold loading original DINO + selector.pt + segmentor.pt", flush=True)
predictor = net.Predictor(root, device, _allow_staged=True)
plain = predictor.predict(image, return_attention=False)
explained = predictor.predict(image, return_attention=True)
expected = torch.load(root / "example/expected_logits.pt", map_location="cpu", weights_only=True)
reference = net.io_utils.read_json(root / "example/reference.json")
if not torch.allclose(plain["logits"], explained["logits"], atol=1e-5, rtol=1e-5):
    raise ValueError("Attention capture changed logits")
if not torch.allclose(expected, plain["logits"], atol=5e-4, rtol=5e-4):
    raise ValueError("Split exported weights differ from the loaded best checkpoint")
candidate = explained["evidence"]["candidate"]
if candidate["position"] != reference["context_position"] or candidate["image_id"] != reference["context_id"]:
    raise ValueError("Export changed the selected context")
for role in ["query_segmentor", "candidate_selector"]:
    attention = explained["evidence"]["attention"][role]
    if attention["grid_shape"] != [predictor.config["image_size"]//16]*2 or not np.isfinite(attention["grid"]).all():
        raise ValueError("Missing/invalid attention grid")
    (root / "example" / (role + ".png")).write_bytes(base64.b64decode(attention["image_url"].split(",",1)[1]))
torch.save(explained.pop("logits"), root / "example/reloaded_logits.pt")
np.save(root / "example/probabilities.npy", explained.pop("probabilities"))
for role in ["mask", "overlay"]:
    (root / "example" / (role + ".png")).write_bytes(base64.b64decode(explained["prediction"][role + "_url"].split(",",1)[1]))
net.io_utils.write_json(root / "example/prediction.json", explained)
net.io_utils.write_json(root / "smoke_test.json", {"state_reload":True, "trained_model_parity":True,
    "attention_parity":True, "selected_context_parity":True, "cold_offline_reload":True,
    "device":device, "benchmark_metric":False})
print("Independent offline checkpoint/attention reload passed", flush=True)
'''


def verify_and_publish(ctx, *, device=None):
    stage = ctx["stage"]
    verification_device = str(device or ctx["device"])
    if ctx["factory"] is None:
        script = ctx["output"] / "verify_isic_snapshot.py"
        script.write_text(ctx.get("cold_source", COLD_SOURCE), encoding="utf-8")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH", None)
        command = [sys.executable, str(script), str(stage), str(ctx["example_path"]), verification_device]
        log_path = ctx["output"] / "cold_reload.log"
        print(f"Independent offline reload on {verification_device}; progress/error log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            raise RuntimeError("Cold reload failed; staged artifacts remain available.\n" + log_path.read_text('utf-8')[-5000:])
        print(log_path.read_text("utf-8")[-1600:])
        smoke = read_json(stage / "smoke_test.json")
        status = "ready"
    else:
        # Small explicit fixtures can test the real exporter; they are never deployable models.
        predictor = ctx["net"].Predictor(stage, "cpu", factory=ctx["factory"], _allow_staged=True)
        plain = predictor.predict(ctx["example_path"], return_attention=False)
        explained = predictor.predict(ctx["example_path"], return_attention=True)
        reference = read_json(stage / "example/reference.json")
        expected = torch.load(stage / "example/expected_logits.pt", map_location="cpu", weights_only=True)
        if (not torch.allclose(expected, plain["logits"], atol=5e-4, rtol=5e-4)
                or not torch.allclose(plain["logits"], explained["logits"], atol=1e-5, rtol=1e-5)
                or explained["evidence"]["candidate"]["position"] != reference["context_position"]):
            raise ValueError("Fixture checkpoint/attention reload failed")
        explained.pop("logits")
        np.save(stage / "example/probabilities.npy", explained.pop("probabilities"))
        write_json(stage / "example/prediction.json", explained)
        smoke = {"state_reload": True, "trained_model_parity": True, "attention_parity": True,
                 "cold_offline_reload": False, "benchmark_metric": False, "fixture": True}
        write_json(stage / "smoke_test.json", smoke)
        status = "fixture_unverified"
    stage_manifest(ctx, status, smoke)
    write_json(ctx["output"] / "export_receipt.json", {"status": status, "bundle": str(stage),
        "source_best_sha256": ctx["identity"]["source_best_sha256"], "training_snapshot": ctx["snapshot"]})
    return stage


class _SplitArchive:
    """Unseekable ZIP sink: never creates an additional multi-GB full ZIP."""
    def __init__(self, folder, name, limit):
        self.folder, self.name, self.limit = Path(folder), name, limit
        self.offset, self.handle, self.current_size = 0, None, 0
        self.files, self.digest = [], hashlib.sha256()

    def seekable(self):
        return False

    def tell(self):
        return self.offset

    def write(self, data):
        data = memoryview(data)
        written = len(data)
        self.digest.update(data)
        while data:
            if self.handle is None or self.current_size == self.limit:
                if self.handle:
                    self.handle.close()
                path = self.folder / f"{self.name}.part{len(self.files) + 1:03d}"
                self.files.append(path)
                self.handle, self.current_size = path.open("wb"), 0
            size = min(len(data), self.limit - self.current_size)
            self.handle.write(data[:size])
            self.current_size += size
            self.offset += size
            data = data[size:]
        return written

    def flush(self):
        if self.handle:
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None


JOIN_SOURCE = r'''from pathlib import Path
import hashlib, json, sys
folder = Path(__file__).resolve().parent
info = json.loads((folder / "download_manifest.json").read_text("utf-8"))
target = Path(sys.argv[1]) if len(sys.argv)>1 else folder / info["archive_name"]
if target.exists():
    raise FileExistsError("Preserve the existing ZIP or choose another output: " + str(target))
temporary = target.with_name(target.name + ".tmp")
full = hashlib.sha256()
with temporary.open("wb") as out:
    for item in info["parts"]:
        path = folder / item["name"]
        if path.parent != folder or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing/wrong archive part: " + item["name"])
        part = hashlib.sha256()
        with path.open("rb") as src:
            for block in iter(lambda:src.read(4*1024**2), b""):
                part.update(block); full.update(block); out.write(block)
        if part.hexdigest() != item["sha256"]:
            raise ValueError("Corrupt part: " + item["name"])
if full.hexdigest() != info["archive_sha256"]:
    raise ValueError("Joined archive checksum mismatch")
temporary.replace(target)
print("Verified ZIP:", target)
print("Extract it to obtain the complete ISIC web bundle.")
'''


def package_downloads(ctx, *, part_mib=256):
    stage, output = ctx["stage"], ctx["output"]
    if read_json(stage / "manifest.json")["status"] != "ready":
        raise ValueError("Only a verified production bundle may be packaged for deployment")
    ctx["net"].verify_bundle(stage)
    if type(part_mib) is not int or part_mib < 16:
        raise ValueError("DOWNLOAD_PART_MIB must be an integer >=16")
    folder = output / "downloads"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "download_manifest.json").unlink(missing_ok=True)
    archive_name = stage.name + ".zip"
    sink = _SplitArchive(folder, archive_name, part_mib * 1024**2)
    try:
        with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".tmp"):
                    archive.write(path, path.relative_to(stage.parent).as_posix())
    finally:
        sink.close()
    parts = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sink.files]
    write_json(folder / "download_manifest.json", {"archive_name": archive_name, "archive_sha256": sink.digest.hexdigest(),
        "part_size_mib": part_mib, "parts": parts, "bundle": stage.name})
    (folder / "join_downloads.py").write_text(JOIN_SOURCE, encoding="utf-8")
    print(f"Download {len(parts)} ZIP parts, download_manifest.json and join_downloads.py from {folder}")
    print("On Windows: python join_downloads.py (checks hashes and joins the ZIP; no Torch required).")
    return folder
