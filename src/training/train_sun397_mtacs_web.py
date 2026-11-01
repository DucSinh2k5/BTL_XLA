"""Full-both SUN397 training and complete offline multimodal deployment export."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

import tacs_visual_web as visual
import train_sun397_tacs_web as common
import mtacs_multimodal_web as runtime
import sun397_caption_web as captions


def default_config():
    return {**common.default_config(), "schema": runtime.SCHEMA, "method": "M_TACS_full_both", "caption_used": True,
            "text_dim": 512, "caption_policy": captions.POLICY, "text_fusion": "two_gated_additive_residuals_shared_projection",
            "utility": "eligible_visual_z_plus_sigmoid_mix_caption_cosine_z", "gate_initial_bias": -1.0,
            "policy_reward": "query_caption_baseline_ce_minus_paired_image_and_caption_ce"}


def validate_config(config):
    common.validate_config({**config, "caption_used": False})
    if (config["schema"] != runtime.SCHEMA or config["method"] != "M_TACS_full_both" or not config["caption_used"]
            or config["text_dim"] != 512 or config["caption_policy"] != captions.POLICY):
        raise ValueError("Expected L/16 SUN397 full_both, K=1, fixed Florence-Prefix/CLIP policy")


def source_hashes():
    return {module.__name__: visual.file_sha256(module.__file__) for module in (visual, common, runtime, captions)} | {
        "training": visual.file_sha256(__file__)}


def run_identity(config, classes, frames, assets):
    return common.run_identity(config, classes, frames, dict(assets, mtacs_sources=source_hashes()))


class TextTable:
    def __init__(self, cache, csv_path, records, config):
        self.cache = cache
        self.rows = captions.load_csv(csv_path)
        if (cache["csv_sha256"] != visual.file_sha256(csv_path) or cache["policy"] != config["caption_policy"]
                or len(set(cache["image_ids"])) != len(cache["image_ids"])):
            raise ValueError("Text cache CSV/policy/ID binding mismatch")
        expected = {r["image_id"] for r in records}
        if set(cache["image_ids"]) != expected or set(self.rows) != expected:
            raise ValueError("Caption coverage must exactly match train/validation/test SUN397 IDs")
        if cache["embeddings"].shape != (len(expected), config["text_dim"]) or cache["valid"].shape != (len(expected),):
            raise ValueError("Text cache shape mismatch")
        if not torch.isfinite(cache["embeddings"]).all() or torch.count_nonzero(cache["embeddings"][~cache["valid"].bool()]):
            raise ValueError("Nonfinite or nonzero masked text embedding")
        norms = cache["embeddings"][cache["valid"].bool()].norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
            raise ValueError("Valid CLIP vectors must be unit normalized")
        self.index = {image_id: i for i, image_id in enumerate(cache["image_ids"])}
        for row in records:
            cached = self.rows[row["image_id"]]
            if cached["split"] != row["split"] or cached["caption_policy"] != config["caption_policy"]:
                raise ValueError("Caption split/policy mismatch")
            if cached["clip_assets_sha256"] != cache["clip_sha256"] or cached["captioner_assets_sha256"] != cache["captioner_sha256"]:
                raise ValueError("Caption rows and vector cache use different frozen text model assets")
            if captions.truth(cached["caption_valid"]) != bool(cache["valid"][self.index[row["image_id"]]]):
                raise ValueError("Caption CSV and embedding masks differ")

    def gather(self, image_ids, device):
        positions = torch.tensor([self.index[str(image_id)] for image_id in image_ids], dtype=torch.long)
        return self.cache["embeddings"][positions].to(device, dtype=torch.float32), self.cache["valid"][positions].to(device, dtype=torch.bool)


def prepare_captions(records, classes, model_assets, settings, shared, device, offline_assets_root, env=None):
    shared = Path(shared)
    shared.mkdir(parents=True, exist_ok=True)
    payload = {"mode": "prepare", "records": records, "classes": classes, "settings": settings,
               "florence": model_assets["florence"], "clip": model_assets["clip"], "asset_identity": model_assets,
               "output": str((shared / "captions.csv").resolve()), "device": str(device)}
    env = env or captions.caption_environment(shared / "caption_runtime", offline_assets_root)
    if all((shared / name).is_file() for name in ("text_embeddings.pt", "captions.csv", "caption_protocol.json", "caption_settings.json")):
        if visual.read_json(shared / "caption_protocol.json") != captions.caption_protocol(payload):
            raise ValueError("Caption cache protocol differs from this run")
        cached = torch.load(shared / "text_embeddings.pt", map_location="cpu", weights_only=True)
        if cached["csv_sha256"] == visual.file_sha256(shared / "captions.csv"):
            print("Reusing complete registered caption/vector cache")
            return cached, env
    # The worker owns CSV append/fsync; only publish the complete cache afterward.
    cache = captions.invoke_worker(payload, env, work_dir=shared)
    common.atomic_torch_save(shared / "text_embeddings.pt", cache)
    visual.write_json(shared / "caption_settings.json", {k: payload[k] for k in ("classes", "settings", "asset_identity")})
    return cache, env


def train_epoch(model, loader, bank, pixels, candidates, table, config, optimizer, scaler, device, epoch):
    model.train()
    features = bank.to(device)
    candidate_text, candidate_valid = table.gather([r["image_id"] for r in candidates], device)
    positions = {r["image_id"]: i for i, r in enumerate(candidates)}
    accumulate = config["effective_batch_size"] // config["batch_size"]
    optimizer.zero_grad(set_to_none=True)
    truth, predictions = [], []
    stats = {key: 0.0 for key in ("loss", "ce_loss", "policy_loss", "reward", "q_gate", "c_gate")}
    successful = skipped = 0
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch:03d} M-TACS train")):
        query, labels = batch["image"].to(device), batch["label"].to(device)
        text, valid = table.gather(batch["image_id"], device)
        excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"]], device=device)
        with common.autocast_context(device):
            scores, _ = model.score(query, features, excluded, text, valid, candidate_text, candidate_valid)
            weights = F.gumbel_softmax(scores, tau=config["gumbel_tau"], hard=True, dim=1)
            context = visual.straight_through_images(weights, pixels, config["st_bank_chunk"])
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                selected_text = weights.float() @ candidate_text
            selected_valid = candidate_valid[weights.argmax(1)]
            logits, details = model(query, context, text, valid, selected_text, selected_valid, return_details=True)
            ce = F.cross_entropy(logits.float(), labels, label_smoothing=config["label_smoothing"])
            distribution = torch.distributions.Categorical(logits=scores)
            actions = distribution.sample()
            log_probs, entropy = distribution.log_prob(actions), distribution.entropy().mean()
        model.eval()
        with torch.no_grad(), common.autocast_context(device):
            baseline = model(query, query_text=text, query_valid=valid)
            paired = model(query, pixels.index_select(0, actions.cpu()).to(device, dtype=torch.float32), text, valid,
                           candidate_text[actions], candidate_valid[actions])
            reward = F.cross_entropy(baseline.float(), labels, reduction="none") - F.cross_entropy(paired.float(), labels, reduction="none")
            advantage = ((reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-6)).detach()
        model.train()
        policy = -(log_probs * advantage).mean()
        loss = ce + config["policy_weight"] * policy - config["entropy_weight"] * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite M-TACS loss")
        group_start = step // accumulate * accumulate
        divisor = min(accumulate, len(loader) - group_start)
        scaler.scale(loss / divisor).backward()
        if (step + 1) % accumulate == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.is_enabled() and scaler.get_scale() < previous
            successful += int(not overflow)
            skipped += int(overflow)
            optimizer.zero_grad(set_to_none=True)
        count = len(labels)
        for key, value in (("loss", loss), ("ce_loss", ce), ("policy_loss", policy), ("reward", reward.mean()),
                           ("q_gate", details["q_gate"].mean()), ("c_gate", details["c_gate"].mean())):
            stats[key] += float(value.detach()) * count
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
    if not successful:
        raise FloatingPointError("No optimizer step completed; inspect loader and AMP overflows")
    return {**{k: v / len(truth) for k, v in stats.items()}, **common.classification_metrics(truth, predictions, config["num_classes"]),
            "optimizer_steps": successful, "amp_skipped_steps": skipped, "semantic_mix": float(model.semantic_mix_logit.sigmoid().detach())}


@torch.no_grad()
def evaluate(model, loader, bank, candidates, table, config, device, label="M-TACS validation", amp=False):
    model.eval()
    features = bank.to(device)
    candidate_text, candidate_valid = table.gather([r["image_id"] for r in candidates], device)
    mapping = {r["image_id"]: i for i, r in enumerate(candidates)}
    rows, loss_sum = [], 0.0
    for batch in tqdm(loader, desc=label):
        query, labels = batch["image"].to(device), batch["label"].to(device)
        text, valid = table.gather(batch["image_id"], device)
        excluded = torch.tensor([mapping.get(i, -1) for i in batch["image_id"]], device=device)
        with common.autocast_context(device, amp):
            scores, _ = model.score(query, features, excluded, text, valid, candidate_text, candidate_valid)
            indices = scores.argmax(1)
            context = torch.stack([visual.preprocess(candidates[int(i)]["image_path"], config) for i in indices.cpu()]).to(device)
            logits, details = model(query, context, text, valid, candidate_text[indices], candidate_valid[indices], return_details=True)
            loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        for j, image_id in enumerate(batch["image_id"]):
            position = int(indices[j])
            rows.append({"image_id": image_id, "ground_truth": int(labels[j]), "prediction": int(logits[j].argmax()),
                         "candidate_position": position, "candidate_id": candidates[position]["image_id"],
                         "query_caption_valid": bool(valid[j]), "candidate_caption_valid": bool(candidate_valid[position]),
                         "q_gate": float(details["q_gate"][j]), "c_gate": float(details["c_gate"][j])})
    if not rows:
        raise ValueError("Empty evaluation loader")
    metrics = common.classification_metrics([r["ground_truth"] for r in rows], [r["prediction"] for r in rows], config["num_classes"])
    return {**metrics, "loss": loss_sum / len(rows), "images": len(rows),
            "q_gate": sum(r["q_gate"] for r in rows) / len(rows), "c_gate": sum(r["c_gate"] for r in rows) / len(rows),
            "semantic_mix": float(model.semantic_mix_logit.sigmoid())}, rows


def train_run(model, classes, train, val, candidates, table, config, assets, output, device):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config, classes, {"train": train, "val": val, "pool": candidates}, assets)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file() and visual.read_json(protocol_path)["identity"] != identity:
        raise ValueError("Output directory belongs to another run/config/source/caption CSV")
    visual.write_json(protocol_path, {"identity": identity, "config": config, "assets": assets,
                                     "source_sha256": source_hashes(), "environment": common.environment()})
    for name, rows in (("train_split.csv", train), ("validation_split.csv", val), ("candidate_pool.csv", candidates)):
        common.csv_write(output / name, rows)
    visual.write_json(output / "class_names.json", classes)
    train_loader, val_loader = common.make_loader(train, config, True), common.make_loader(val, config)
    pixel_bank = common.build_image_bank(candidates, config)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": config["backbone_lr"]},
                                  {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": config["head_lr"]}],
                                 weight_decay=config["weight_decay"])
    def lr_scale(epoch):
        warmup = min(config["warmup_epochs"], config["epochs"])
        if warmup and epoch < warmup:
            return 0.2 + 0.8 * epoch / warmup
        progress = (epoch - warmup) / max(1, config["epochs"] - warmup)
        return 0.01 + 0.99 * (1 + math.cos(math.pi * min(1, progress))) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported())
    history, best_loss, start = [], math.inf, 1
    if (output / "last.pt").is_file():
        last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if last["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        common.restore_rng(last["rng"], train_loader)
        history, best_loss, start = last["history"], last["best_val_loss"], last["epoch"] + 1
        del last
        best_check = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        if best_check["identity"] != identity or best_check["epoch"] >= start:
            raise ValueError("Resume best checkpoint does not match last.pt")
        del best_check
        print(f"Resume after epoch {start - 1}; fixed horizon={config['epochs']}")
    for epoch in range(start, config["epochs"] + 1):
        started = time.perf_counter()
        bank = common.refresh_bank(model, candidates, config, device, amp=True)
        train_metrics = train_epoch(model, train_loader, bank, pixel_bank, candidates, table, config, optimizer, scaler, device, epoch)
        bank = common.refresh_bank(model, candidates, config, device, amp=True)
        val_metrics, _ = evaluate(model, val_loader, bank, candidates, table, config, device, amp=True)
        improved = val_metrics["loss"] < best_loss
        best_loss = min(best_loss, val_metrics["loss"])
        history.append({"epoch": epoch, **{"train_" + k: v for k, v in train_metrics.items()},
                        **{"val_" + k: v for k, v in val_metrics.items()}, "seconds": time.perf_counter() - started})
        scheduler.step()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        base = {"schema": runtime.SCHEMA, "identity": identity, "config": config, "class_names": classes,
                "epoch": epoch, "val_metrics": val_metrics, "model_state_dict": state}
        if improved:
            common.atomic_torch_save(output / "best.pt", base)
        common.atomic_torch_save(output / "last.pt", {**base, "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(), "rng": common.rng_state(train_loader),
            "history": history, "best_val_loss": best_loss})
        common.csv_write(output / "history.csv", history)
        print(f"epoch={epoch:03d}/{config['epochs']} val_f1={val_metrics['macro_f1']:.6f} val_loss={val_metrics['loss']:.6f} "
              f"q_gate={val_metrics['q_gate']:.3f} c_gate={val_metrics['c_gate']:.3f} semantic_mix={val_metrics['semantic_mix']:.3f}")
        del base, state
    if not history or history[-1]["epoch"] != config["epochs"]:
        raise RuntimeError("Complete the configured epoch budget before final test/export")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if best["identity"] != identity:
        raise ValueError("Wrong best checkpoint")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    visual.write_json(output / "training_complete.json", {"identity": identity, "epochs_completed": config["epochs"],
        "best_epoch": best["epoch"], "best_sha256": visual.file_sha256(output / "best.pt")})
    return best, history


def copy_assets(source, target):
    source, target = Path(source), Path(target)
    ignore = shutil.ignore_patterns(".git", ".cache", "__pycache__", "*.pyc")
    shutil.copytree(source, target, ignore=ignore)
    if captions.tree_identity(source) != captions.tree_identity(target):
        raise ValueError("Caption/CLIP assets changed during copy")


def publish_zip(target, archive, identity):
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
        for path in sorted(target.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                handle.write(path, target.name + "/" + path.relative_to(target).as_posix())
    with zipfile.ZipFile(temporary) as handle:
        bad = handle.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
    temporary.replace(archive)
    visual.write_json(archive.parent / "download_manifest.json", {"zip": archive.name, "sha256": visual.file_sha256(archive),
        "bytes": archive.stat().st_size, "run_identity": identity, "schema": runtime.SCHEMA})


def export_bundle(model, best, candidates, classes, table, config, repo, output, val_example, metrics, predictions,
                  model_assets, offline_assets_root, model_factory=runtime.build_model, caption_runner=None):
    output = Path(output).resolve()
    complete = visual.read_json(output / "training_complete.json")
    protocol = visual.read_json(output / "protocol.json")
    best_sha = visual.file_sha256(output / "best.pt")
    if (complete["identity"] != best["identity"] or complete["best_sha256"] != best_sha
            or complete["epochs_completed"] != config["epochs"] or complete["best_epoch"] != best["epoch"]
            or best["config"] != config or best["class_names"] != classes or protocol["source_sha256"] != source_hashes()
            or protocol["assets"].get("caption_assets") != model_assets):
        raise ValueError("Training completion/best/config/source mismatch")
    with (output / "candidate_pool.csv").open(encoding="utf-8", newline="") as handle:
        original = [(r["image_id"], int(r["class_id"])) for r in csv.DictReader(handle)]
    if original != [(r["image_id"], int(r["class_id"])) for r in candidates]:
        raise ValueError("Export pool/order differs from training")
    if protocol["assets"].get("dino_source_sha256") not in (None, common.source_tree_identity(repo)):
        raise ValueError("DINOv3 source changed after training")
    shared = output / "shared"
    for key, name in (("caption_csv_sha256", "captions.csv"), ("text_cache_sha256", "text_embeddings.pt")):
        if protocol["assets"].get(key) != visual.file_sha256(shared / name):
            raise ValueError("Caption CSV/embedding cache changed after training")
    if protocol["assets"].get("caption_settings_sha256") != visual.file_sha256(shared / "caption_settings.json"):
        raise ValueError("Caption inference settings changed after training")
    for key in ("florence", "clip"):
        if captions.tree_identity(model_assets[key]) != model_assets[key + "_sha256"]:
            raise ValueError("Frozen text model assets changed after training")
    export_root = output / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    target = export_root / "sun397_mtacs_full_both_vitl16_web_bundle"
    archive = target.with_suffix(".zip")
    if target.exists() or archive.exists():
        if target.is_dir() and archive.is_file() and (export_root / "download_manifest.json").is_file():
            manifest = runtime.verify_bundle(target)
            saved = visual.read_json(export_root / "download_manifest.json")
            if (manifest["run_identity"] == best["identity"] and manifest["training_checkpoint_sha256"] == best_sha
                    and saved["sha256"] == visual.file_sha256(archive) and saved["bytes"] == archive.stat().st_size):
                return archive
        raise FileExistsError("Partial or mismatched export exists. Preserve and inspect exports/ rather than overwrite it")
    parameter_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    assets_bytes = sum(p.stat().st_size for key in ("florence", "clip") for p in Path(model_assets[key]).rglob("*") if p.is_file())
    image_bytes = sum(Path(r["image_path"]).stat().st_size for r in candidates)
    if shutil.disk_usage(export_root).free < 2 * (parameter_bytes + assets_bytes + image_bytes) + 2 * 1024**3:
        raise OSError("Insufficient free disk for the full M-TACS bundle plus ZIP. Preserve last.pt and free space first")
    stage = export_root / (".stage_" + uuid.uuid4().hex[:12])
    stage.mkdir()
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    device = next(model.parameters()).device
    portable = dict(config, candidate_count=len(candidates))
    visual.write_json(stage / "config.json", portable)
    visual.write_json(stage / "class_names.json", classes)
    common.atomic_torch_save(stage / "model.pt", {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": portable, "class_names": classes, "epoch": best["epoch"], "run_identity": best["identity"]})
    model_sha = visual.file_sha256(stage / "model.pt")
    pool = []
    for position, record in enumerate(tqdm(candidates, desc="Export candidate images and captions")):
        name = "candidates/" + record["image_id"]
        path = visual.bundle_path(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image_path"], path)
        pool.append({"position": position, "image_id": record["image_id"], "file": name,
                     "sha256": visual.file_sha256(path), "class_id": record["class_id"], "class_name": classes[record["class_id"]]})
    visual.write_json(stage / "candidate_pool.json", pool)
    common.csv_write(stage / "candidate_pool.csv", pool)
    final_bank = common.refresh_bank(model, candidates, config, device, amp=False)
    common.atomic_torch_save(stage / "candidate_features.pt", {"features": final_bank, "image_ids": [r["image_id"] for r in pool], "checkpoint_sha256": model_sha})
    texts, valid = table.gather([r["image_id"] for r in pool], "cpu")
    common.atomic_torch_save(stage / "candidate_text.pt", {"embeddings": texts, "valid": valid, "image_ids": [r["image_id"] for r in pool],
        "checkpoint_sha256": model_sha, "csv_sha256": visual.file_sha256(shared / "captions.csv")})
    for name in ("captions.csv", "text_embeddings.pt", "caption_protocol.json", "caption_settings.json"):
        shutil.copy2(shared / name, stage / name)
    for key in ("florence", "clip"):
        copy_assets(model_assets[key], stage / "models" / key)
    (stage / "offline_wheels").mkdir()
    if offline_assets_root and Path(offline_assets_root).is_dir():
        directories = captions.wheel_directories(offline_assets_root, shared / "caption_runtime/wheels")
        prefixes = ("transformers-", "tokenizers-", "huggingface_hub-", "accelerate-", "einops-", "timm-", "safetensors-")
        for directory in directories:
            for path in Path(directory).glob("*.whl"):
                if path.name.startswith(prefixes):
                    destination = stage / "offline_wheels" / path.name
                    if destination.is_file() and visual.file_sha256(destination) != visual.file_sha256(path):
                        raise ValueError("Conflicting offline wheel filenames")
                    shutil.copy2(path, destination)
    common.copy_vendor(repo, stage / "vendor/dinov3")
    (stage / "runtime").mkdir()
    for module in (visual, runtime, captions):
        shutil.copy2(module.__file__, stage / "runtime" / Path(module.__file__).name)
    (stage / "provenance").mkdir()
    for module in (common, sys.modules[__name__]):
        shutil.copy2(module.__file__, stage / "provenance" / Path(module.__file__).name)
    for name in ("train_split.csv", "validation_split.csv", "test_split.csv", "history.csv", "protocol.json"):
        if (output / name).is_file():
            shutil.copy2(output / name, stage / name)
    common.csv_write(stage / "test_predictions.csv", predictions)
    visual.write_json(stage / "metrics.json", {"test": metrics, "validation_best": best["val_metrics"], "best_epoch": best["epoch"], "precision": "float32"})
    visual.write_json(stage / "environment.json", common.environment())
    visual.write_json(stage / "attention_config.json", {"method": visual.ATTENTION_METHOD,
        "query_classifier": "last-block CLS attention to query patches in paired visual backbone",
        "candidate_selector": "standalone selected candidate encoder CLS attention",
        "text_attribution": False, "causal_explanation": False, "class_conditional": False, "extra_attention_model": False})
    (stage / "requirements.txt").write_text("torch\nnumpy\nPillow\ntqdm\ntransformers==4.49.0\ntokenizers>=0.21,<0.22\nhuggingface-hub>=0.26,<1\naccelerate\neinops\ntimm\n", encoding="utf-8")
    (stage / "README.md").write_text(BUNDLE_README, encoding="utf-8")
    (stage / "runtime/verify_export.py").write_text(COLD_VERIFY_SOURCE, encoding="utf-8")
    (stage / "examples").mkdir()
    example = stage / "examples" / ("query" + Path(val_example["image_path"]).suffix.lower())
    shutil.copy2(val_example["image_path"], example)
    def manifest():
        files = {p.relative_to(stage).as_posix(): {"bytes": p.stat().st_size, "sha256": visual.file_sha256(p)} for p in sorted(stage.rglob("*"))
                 if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
        visual.write_json(stage / "manifest.json", {"schema": runtime.SCHEMA, "status": "ready", "run_identity": best["identity"],
            "training_checkpoint_sha256": best_sha, "model_sha256": model_sha, "files": files})
    manifest()
    production = model_factory is runtime.build_model and caption_runner is None
    if production:
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, str(stage / "runtime/verify_export.py"), "--bundle", str(stage),
            "--image", str(example), "--image-id", val_example["image_id"], "--device", str(device)], env=env,
            cwd=stage, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError("Cold M-TACS export check failed:\n" + process.stdout[-3000:] + process.stderr[-6000:])
        result = visual.read_json(stage / "examples/prediction.json")
        reload = torch.load(stage / "examples/reload.pt", map_location="cpu", weights_only=True)
    else:
        predictor = runtime.BundlePredictor(stage, str(device), model_factory, caption_runner)
        result = predictor.predict(example, image_id=val_example["image_id"], use_caption_cache=False)
        reload = {"logits": predictor.last_logits, "text": predictor.last_query_embedding, "valid": predictor.last_query_valid}
        del predictor
    # Compare the exact fresh caption vector returned by the exported pipeline,
    # not an assumed equality between batch-generated and fresh greedy captions.
    with torch.no_grad():
        query = visual.preprocess(example, config)[None].to(device)
        text, mask = reload["text"].to(device), reload["valid"].to(device)
        excluded = [i for i, r in enumerate(pool) if r["image_id"] == val_example["image_id"] or r["sha256"] == visual.file_sha256(example)]
        if len(excluded) > 1:
            raise ValueError("Export example duplicates several pool images; select another held-out validation example")
        position_tensor = torch.tensor([excluded[0] if excluded else -1], device=device)
        scores, _ = model.score(query, final_bank.to(device), position_tensor, text, mask, texts.to(device), valid.to(device))
        position = int(scores.argmax(1))
        if position != result["evidence"]["candidate"]["position"]:
            raise ValueError("Reload candidate selection differs from the best checkpoint")
        reference = model(query, visual.preprocess(candidates[position]["image_path"], config)[None].to(device),
                          text, mask, texts[position:position + 1].to(device), valid[position:position + 1].to(device)).float().cpu()
    if not torch.allclose(reference, reload["logits"], atol=1e-4, rtol=1e-4):
        raise ValueError("Reload M-TACS logits do not match the best checkpoint")
    roles = result["evidence"]["attention"]
    if set(roles) != {"query_classifier", "candidate_selector"}:
        raise ValueError("Both real attention roles are required")
    visual.write_json(stage / "examples/prediction.json", result)
    visual.write_json(stage / "smoke_test.json", {"passed": True, "cold_subprocess_reload": production,
        "fresh_florence_clip_integration": production, "test_injected_services": not production,
        "max_abs_logit_error": float((reference - reload["logits"]).abs().max()), "attention_roles": list(roles),
        "source_is_bundle_only": production, "caption_policy": config["caption_policy"]})
    manifest()
    runtime.verify_bundle(stage)
    stage.rename(target)
    publish_zip(target, archive, best["identity"])
    print(f"Verified complete M-TACS ZIP: {archive}")
    return archive


BUNDLE_README = '''# SUN397 M-TACS full_both ViT-L/16

Full trained model, frozen Microsoft Florence-2-base, frozen CLIP ViT-B/32,
tokenizers/processors/model code, candidate images, caption CSV, aligned visual
and text banks, real attention and inference code are included. No Kaggle model
mount is needed after extraction. Keep original model/data license terms.

Install compatible OS/CUDA PyTorch and caption dependencies. Requirements and
the recorded environment describe dependencies, not a cross-platform installer.
Attached offline wheels, when available, match their original Python/platform;
Linux Kaggle wheels cannot install Windows tokenizers. Prepare a compatible
Transformers 4.49 environment or supply matching offline wheels. Caption workers
use an isolated overlay and do not downgrade the web/notebook kernel.

Run from the bundle directory:
python runtime/mtacs_multimodal_web.py --bundle . --image YOUR_IMAGE.jpg --output prediction.json

Use --fresh-caption to force the full Florence -> audit -> CLIP path. Otherwise
known identical image bytes may reuse verified cached text. Novel images always
use real Florence, never fabricated/stub captions. The Python BundlePredictor
returns query caption, selected candidate image/caption, top classes, q/c gates
and real query-classifier/candidate-selector attention PNG data URLs and grids.
Load once and serialize web calls with a lock. Existing FastAPI wiring is a
separate step. For both TACS/M-TACS processes, use one compatible DINOv3 source.

Policy sun397_lexical_clip_audit_v1: remove boilerplate, keep first 5-8 readable
words, reject direct SUN class terms, encode accepted text only. CLIP cosine
to class prompts is an audit signal, not proof of label leakage and not an
automatic semantic rejection. A masked caption is still shown but contributes
zero text to selector/fusion; its image remains eligible. No test-label prompt,
test-set fine-tuning or online continual learning is performed.

Attention is visual last-block mean-head CLS attention, not text attribution,
not Grad-CAM, and not a causal explanation. No separate attention model is needed.
This new deployment adaptation is not automatically comparable to historical
SUN397 checkpoints trained with a different caption/split/architecture policy.
'''


COLD_VERIFY_SOURCE = '''import argparse
from pathlib import Path
import torch
import tacs_visual_web as visual
from mtacs_multimodal_web import BundlePredictor

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
    raise RuntimeError("DINOv3 did not load from the bundle")
predictor.predict(args.image, image_id=args.image_id, explain=False)
plain = predictor.last_logits.clone()
cached = predictor.predict(args.image, image_id=args.image_id, explain=True)
if not torch.allclose(plain, predictor.last_logits, atol=1e-5, rtol=1e-5):
    raise RuntimeError("Attention capture changed cached-query logits")
fresh = predictor.predict(args.image, image_id=args.image_id, explain=True, use_caption_cache=False)
if fresh["evidence"]["query_caption"]["source"] != "fresh_florence":
    raise RuntimeError("Fresh caption generation was not exercised")
visual.write_json(args.bundle / "examples/cached_prediction.json", cached)
visual.write_json(args.bundle / "examples/prediction.json", fresh)
torch.save({"logits": predictor.last_logits, "text": predictor.last_query_embedding,
            "valid": predictor.last_query_valid}, args.bundle / "examples/reload.pt")
print("Cold M-TACS inference with real fresh Florence/CLIP and both attention maps passed")
'''
