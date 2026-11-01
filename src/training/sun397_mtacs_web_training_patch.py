"""Memory-bounded full_both functions assembled into the standalone M-TACS trainer.

The multimodal inference model and caption service are deliberately unchanged.
Shared execution helpers come from the already-tested visual repair engine.
"""


def default_config():
    config = _legacy_default_config()
    config.update(batch_size=64, effective_batch_size=64, workers=4,
                  gradient_checkpointing=True, feature_batch_size=32, eval_batch_size=8,
                  st_bank_chunk=128, training_engine="memory_bounded_two_pass_v2",
                  candidate_training_pixels="fp16_quantized_values_exact_st_auto_cpu_or_gpu_storage")
    return config


def validate_config(config):
    _legacy_validate_config(config)
    if config.get("training_engine") != "memory_bounded_two_pass_v2" or not config["gradient_checkpointing"]:
        raise ValueError("Use the memory-bounded full_both engine with checkpointing enabled")


def caption_settings_for_resume(requested, shared, resume_from=None):
    folders = [Path(shared)]
    if resume_from is not None:
        folders.append(Path(resume_from) / "shared")
    for folder in folders:
        path = folder / "caption_protocol.json"
        if not path.is_file():
            continue
        saved = visual.read_json(path)["settings"]
        if {k: v for k, v in saved.items() if k != "batch_size"} != {k: v for k, v in requested.items() if k != "batch_size"}:
            raise ValueError("Caption policy/task/token budget/audit settings changed; keep the registered values on resume")
        print("Reusing registered caption settings, batch:", saved["batch_size"])
        return dict(saved)
    return requested


def stage_previous_caption_cache(source, shared, records, classes, model_assets, settings):
    """Reuse only a registered, image-bound cache; do not regenerate checkpoint text."""
    shared = Path(shared)
    shared.mkdir(parents=True, exist_ok=True)
    payload = {"records": records, "classes": classes, "asset_identity": model_assets, "settings": settings}
    expected = captions.caption_protocol(payload)
    expected_rows = {row["image_id"]: row for row in records}

    def check_rows(path):
        rows = captions.load_csv(path)
        for image_id, row in rows.items():
            if image_id not in expected_rows or any(row[k] != expected_rows[image_id][k] for k in ("split", "image_sha256")):
                raise ValueError("Previous caption cache has another image/split binding")
            if (row["caption_policy"] != settings["policy"]
                    or row["captioner_assets_sha256"] != model_assets["florence_sha256"]
                    or row["clip_assets_sha256"] != model_assets["clip_sha256"]):
                raise ValueError("Previous caption policy or frozen model identity differs")
        return rows

    if (shared / "caption_protocol.json").is_file():
        if visual.read_json(shared / "caption_protocol.json") != expected:
            raise ValueError("Repaired caption cache protocol differs; keep original caption settings and assets")
        rows = check_rows(shared / "captions.csv")
        print(f"Keeping this repaired run's caption progress: {len(rows):,}/{len(records):,}")
        return len(rows)
    if (shared / "captions.csv").exists():
        raise ValueError("Unregistered CSV in repaired output; do not silently reinterpret old captions")
    if source is None:
        return 0
    source = Path(source).resolve()
    previous = source / "shared"
    required = ("captions.csv", "text_embeddings.pt", "caption_protocol.json", "caption_settings.json")
    has_checkpoint = (source / "last.pt").is_file() or (source / "best.pt").is_file()
    if has_checkpoint and not all((previous / name).is_file() for name in required):
        raise FileNotFoundError("Previous checkpoint requires its complete shared caption/vector cache; do not generate replacement text")
    if not (previous / "caption_protocol.json").is_file():
        if (previous / "captions.csv").is_file():
            raise ValueError("Previous captions.csv is missing its registration protocol")
        print("No previous registered captions; generation will start from images")
        return 0
    if previous.resolve() == shared.resolve():
        raise ValueError("Use a new repaired output directory, not the previous shared directory")
    if visual.read_json(previous / "caption_protocol.json") != expected:
        raise ValueError("Previous caption cache settings/assets/source/records differ; refusing automatic regeneration for resume")
    rows = check_rows(previous / "captions.csv")
    if has_checkpoint:
        protocol = visual.read_json(source / "protocol.json")
        for key, name in (("caption_csv_sha256", "captions.csv"), ("text_cache_sha256", "text_embeddings.pt"),
                          ("caption_settings_sha256", "caption_settings.json")):
            if protocol["assets"].get(key) != visual.file_sha256(previous / name):
                raise ValueError("Previous checkpoint text-cache hash does not match: " + name)
    # An interrupted caption run can legitimately lack the final vector cache.
    # Copy only known data files, not an old Python environment or worker state.
    for name in required:
        old, target = previous / name, shared / name
        if not old.is_file():
            continue
        if target.exists() and visual.file_sha256(target) != visual.file_sha256(old):
            raise ValueError("Conflicting staged caption file: " + name)
    for name in required:
        old, target = previous / name, shared / name
        if old.is_file() and not target.exists():
            temporary = target.with_name(target.name + ".tmp")
            shutil.copy2(old, temporary)
            temporary.replace(target)
    print(f"Copied registered caption progress: {len(rows):,}/{len(records):,}; original files unchanged")
    return len(rows)


def _legacy_mtacs_identity(config, classes, frames, assets):
    return common._legacy_identity(config, classes, frames, dict(assets, mtacs_sources=LEGACY_MTACS_SOURCE_SHA256))


def migrate_legacy_run(source, output, config, classes, frames, assets):
    output = Path(output)
    if source is None or (output / "protocol.json").is_file():
        return None
    source = Path(source).resolve()
    if not (source / "protocol.json").is_file():
        print("No prior M-TACS training protocol; starting epoch 1 with the registered caption cache")
        return None
    if output.resolve() == source:
        raise ValueError("Use a separate repaired OUTPUT_DIR")
    old = visual.read_json(source / "protocol.json")
    if old.get("source_sha256") != LEGACY_MTACS_SOURCE_SHA256:
        raise ValueError("Previous run is not the checked legacy full_both visual/text implementation")
    old_config = old["config"]
    allowed = {"gradient_checkpointing", "workers", "feature_batch_size", "eval_batch_size",
               "st_bank_chunk", "candidate_training_pixels"}
    expected = {**old_config, "training_engine": "memory_bounded_two_pass_v2"}
    for key in allowed:
        expected[key] = config[key]
    if expected != config:
        raise ValueError("Resume cannot change model/text policy, logical batch, seed, learning settings or epoch horizon")
    old_identity = _legacy_mtacs_identity(old_config, classes, frames, assets)
    if old["identity"] != old_identity or old["assets"] != assets:
        raise ValueError("Legacy classes/pool/order/weights/caption-cache identity mismatch")
    last_path, best_path = source / "last.pt", source / "best.pt"
    if not last_path.is_file():
        if best_path.is_file():
            raise FileNotFoundError("Legacy best.pt exists but last.pt is required for optimizer resume")
        print("No completed legacy epoch checkpoint; starting epoch 1")
        return {"source": str(source), "from_identity": old_identity, "completed_epochs": 0}
    if not best_path.is_file():
        raise FileNotFoundError("Legacy resume also requires best.pt")
    if any((output / name).exists() for name in ("last.pt", "best.pt", "migration.json")):
        raise FileExistsError("Incomplete migration in repaired output; preserve and inspect it before retrying")
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    for item in (last, best):
        if item["identity"] != old_identity or item["config"] != old_config or item["class_names"] != classes:
            raise ValueError("Legacy checkpoint/config/class mapping mismatch")
    if not last.get("history") or last["history"][-1]["epoch"] != last["epoch"] or not 1 <= best["epoch"] <= last["epoch"] <= config["epochs"]:
        raise ValueError("Invalid legacy checkpoint epoch/history")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng"):
        if key not in last:
            raise ValueError("Legacy last.pt lacks " + key)
    audit = {"source": str(source), "from_identity": old_identity, "completed_epochs": last["epoch"],
             "legacy_sources": LEGACY_MTACS_SOURCE_SHA256,
             "legacy_checkpoints": {name: visual.file_sha256(source / name) for name in ("last.pt", "best.pt")},
             "caption_csv_sha256": assets["caption_csv_sha256"], "text_cache_sha256": assets["text_cache_sha256"],
             "note": "All optimizer and frozen-caption bindings retained; repaired stochastic order is not bitwise legacy replay"}
    identity = run_identity(config, classes, frames, assets)
    output.mkdir(parents=True, exist_ok=True)
    for name, item in (("best.pt", best), ("last.pt", last)):
        item.update(identity=identity, config=dict(config), migration=audit)
        common.atomic_torch_save(output / name, item)
    visual.write_json(output / "migration.json", audit)
    fields = list(dict.fromkeys(key for row in last["history"] for key in row))
    common.csv_write(output / "history.csv", last["history"], fields=fields)
    print(f"Copied compatible M-TACS epoch {last['epoch']} into repaired output; previous run untouched")
    return audit


def _reward_pass(model, query, labels, text, valid, actions, pixels, candidate_text, candidate_valid, device):
    model.eval()
    try:
        with torch.no_grad(), common.autocast_context(device):
            baseline = model(query, query_text=text, query_valid=valid)
            paired = model(query, common._selected_pixels(pixels, actions, device), text, valid,
                           candidate_text[actions], candidate_valid[actions])
            rewards = F.cross_entropy(baseline.float(), labels, reduction="none") - F.cross_entropy(paired.float(), labels, reduction="none")
        return rewards.detach().cpu()
    finally:
        model.train()


def train_epoch(model, loader, bank, pixels, candidates, table, config, optimizer, scaler, device, epoch, execution=None):
    model.train()
    model.gradient_checkpointing = True
    device = torch.device(device)
    options = common.execution_options(config, execution)
    compute = options["compute_batch_size"]
    features = bank.to(device)
    candidate_text, candidate_valid = table.gather([r["image_id"] for r in candidates], device)
    positions = {r["image_id"]: i for i, r in enumerate(candidates)}
    accumulate = config["effective_batch_size"] // config["batch_size"]
    optimizer.zero_grad(set_to_none=True)
    truth, predictions = [], []
    stats = {key: 0.0 for key in ("loss", "ce_loss", "policy_loss", "reward", "entropy", "q_gate", "c_gate")}
    successful = skipped = 0
    if not len(loader):
        raise ValueError("Empty M-TACS training loader")
    started_epoch = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} M-TACS / logical={config['batch_size']} compute={compute}")
    for step, batch in enumerate(progress):
        profile = step < options["profile_batches"]
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        count = len(batch["label"])
        actions_list, rewards, states, sampled_scores = [], [], [], []
        for offset in range(0, count, compute):
            end = min(offset + compute, count)
            query = batch["image"][offset:end].to(device, non_blocking=True)
            labels = batch["label"][offset:end].to(device, non_blocking=True)
            text, valid = table.gather(batch["image_id"][offset:end], device)
            excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"][offset:end]], device=device)
            states.append(common._selector_rng(device))
            with torch.no_grad(), common.autocast_context(device):
                scores, _ = model.score(query, features, excluded, text, valid, candidate_text, candidate_valid)
                actions = torch.distributions.Categorical(logits=scores).sample()
            sampled_scores.append(scores.detach().cpu())
            actions_list.append(actions.cpu())
            rewards.append(_reward_pass(model, query, labels, text, valid, actions, pixels, candidate_text, candidate_valid, device))
            del query, labels, text, valid, excluded, scores, actions
        all_rewards = torch.cat(rewards)
        advantages = common.policy_advantage(all_rewards)
        divisor = min(accumulate, len(loader) - step // accumulate * accumulate)
        for chunk_index, offset in enumerate(range(0, count, compute)):
            end = min(offset + compute, count)
            query = batch["image"][offset:end].to(device, non_blocking=True)
            labels = batch["label"][offset:end].to(device, non_blocking=True)
            text, valid = table.gather(batch["image_id"][offset:end], device)
            excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"][offset:end]], device=device)
            with common.autocast_context(device):
                with common._replay_selector_rng(states[chunk_index], device):
                    scores, _ = model.score(query, features, excluded, text, valid, candidate_text, candidate_valid)
                if not torch.allclose(scores.detach(), sampled_scores[chunk_index].to(device), atol=1e-4, rtol=1e-4):
                    raise RuntimeError("M-TACS selector RNG replay differs from the sampled policy")
                weights = F.gumbel_softmax(scores, tau=config["gumbel_tau"], hard=True, dim=1)
                chunk_size = len(pixels) if pixels.device.type == "cuda" else config["st_bank_chunk"]
                context = visual.straight_through_images(weights, pixels, chunk_size)
                # Use the same hard selection for pixels, text and validity.
                # Do not detach the text matmul: CE also trains the selector here.
                with torch.autocast(device_type=device.type, enabled=False):
                    selected_text = weights.float() @ candidate_text
                selected_valid = candidate_valid[weights.argmax(1)]
                logits, details = model(query, context, text, valid, selected_text, selected_valid, return_details=True)
                ce = F.cross_entropy(logits.float(), labels, label_smoothing=config["label_smoothing"])
                distribution = torch.distributions.Categorical(logits=scores)
                log_probs = distribution.log_prob(actions_list[chunk_index].to(device))
                policy = -(log_probs * advantages[offset:end].to(device)).mean()
                entropy = distribution.entropy().mean()
                loss = ce + config["policy_weight"] * policy - config["entropy_weight"] * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite full_both loss")
            scaler.scale(loss * ((end - offset) / count) / divisor).backward()
            for key, value in (("loss", loss), ("ce_loss", ce), ("policy_loss", policy), ("entropy", entropy),
                               ("q_gate", details["q_gate"].mean()), ("c_gate", details["c_gate"].mean())):
                stats[key] += float(value.detach()) * (end - offset)
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
            del query, labels, text, valid, excluded, scores, weights, context, selected_text, selected_valid
            del logits, details, ce, distribution, log_probs, policy, entropy, loss
        truth.extend(batch["label"].tolist())
        stats["reward"] += float(all_rewards.sum())
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
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            tqdm.write(f"Profile M-TACS batch {step+1}: {seconds:.2f}s, {count/seconds:.2f} images/s, peak={peak:.2f} GiB, pixel bank={pixels.device}")
        del actions_list, rewards, states, sampled_scores, all_rewards, advantages
    if not successful:
        raise FloatingPointError("No M-TACS optimizer step completed")
    elapsed = max(time.perf_counter() - started_epoch, 1e-9)
    return {**{k: v / len(truth) for k, v in stats.items()}, **common.classification_metrics(truth, predictions, config["num_classes"]),
            "optimizer_steps": successful, "amp_skipped_steps": skipped,
            "semantic_mix": float(model.semantic_mix_logit.sigmoid().detach()),
            "images_per_second": len(truth) / elapsed, "compute_batch_size": compute,
            "pixel_bank_device": str(pixels.device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0}


def train_run(model, classes, train, val, candidates, table, config, assets, output, device, execution=None, resume_from=None):
    validate_config(config)
    options = common.execution_options(config, execution)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {"train": train, "val": val, "pool": candidates}
    identity = run_identity(config, classes, frames, assets)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file() and visual.read_json(protocol_path)["identity"] != identity:
        raise ValueError("Output belongs to another run/config/source/caption CSV")
    for key, name in (("caption_csv_sha256", "captions.csv"), ("text_cache_sha256", "text_embeddings.pt"),
                      ("caption_settings_sha256", "caption_settings.json")):
        if assets.get(key) != visual.file_sha256(output / "shared" / name):
            raise ValueError("Registered caption cache changed before training: " + name)
    if table.cache["csv_sha256"] != assets["caption_csv_sha256"]:
        raise ValueError("TextTable belongs to another caption CSV")
    migration = migrate_legacy_run(resume_from, output, config, classes, frames, assets)
    if migration is None and protocol_path.is_file():
        migration = visual.read_json(protocol_path).get("migration")
    visual.write_json(protocol_path, {"identity": identity, "config": config, "assets": assets,
                                     "source_sha256": source_hashes(), "environment": common.environment(),
                                     "execution": options, "migration": migration})
    for name, rows in (("train_split.csv", train), ("validation_split.csv", val), ("candidate_pool.csv", candidates)):
        common.csv_write(output / name, rows)
    visual.write_json(output / "class_names.json", classes)
    train_loader, val_loader = common.make_loader(train, config, True), common.make_loader(val, config)
    model.gradient_checkpointing = True
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
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        if last["identity"] != identity or last["config"] != config:
            raise ValueError("Resume identity/config mismatch")
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        common.restore_rng(last["rng"], train_loader)
        history, best_loss, start = last["history"], last["best_val_loss"], last["epoch"] + 1
        del last
        best_check = torch.load(best_path, map_location="cpu", weights_only=True)
        if best_check["identity"] != identity or best_check["epoch"] >= start:
            raise ValueError("Best checkpoint does not match last.pt")
        del best_check
        print(f"Resume after epoch {start-1}; horizon stays {config['epochs']}")
    print("M-TACS logical / GPU compute batch:", config["batch_size"], options["compute_batch_size"], "| checkpointing: on")
    pixels = common.prepare_training_pixels(candidates, config, device, options) if start <= config["epochs"] else None
    try:
        for epoch in range(start, config["epochs"] + 1):
            started = time.perf_counter()
            bank = common.refresh_bank(model, candidates, config, device, amp=True)
            first_refresh = time.perf_counter() - started
            train_metrics = train_epoch(model, train_loader, bank, pixels, candidates, table, config, optimizer, scaler, device, epoch, options)
            refresh_started = time.perf_counter()
            bank = common.refresh_bank(model, candidates, config, device, amp=True)
            second_refresh = time.perf_counter() - refresh_started
            validation_started = time.perf_counter()
            val_metrics, _ = evaluate(model, val_loader, bank, candidates, table, config, device, amp=True)
            validation_seconds = time.perf_counter() - validation_started
            improved = val_metrics["loss"] < best_loss
            best_loss = min(best_loss, val_metrics["loss"])
            history.append({"epoch": epoch, **{"train_"+k: v for k, v in train_metrics.items()},
                            **{"val_"+k: v for k, v in val_metrics.items()}, "seconds": time.perf_counter()-started,
                            "backbone_lr": optimizer.param_groups[0]["lr"], "refresh_seconds": first_refresh+second_refresh,
                            "validation_seconds": validation_seconds, "training_engine": config["training_engine"]})
            scheduler.step()
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            base = {"schema": runtime.SCHEMA, "identity": identity, "config": config, "class_names": classes,
                    "epoch": epoch, "val_metrics": val_metrics, "model_state_dict": state, "migration": migration}
            if improved:
                common.atomic_torch_save(best_path, base)
            common.atomic_torch_save(last_path, {**base, "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "rng": common.rng_state(train_loader), "history": history, "best_val_loss": best_loss})
            fields = list(dict.fromkeys(key for row in history for key in row))
            common.csv_write(output / "history.csv", history, fields=fields)
            print(f"epoch={epoch:03d}/{config['epochs']} val_f1={val_metrics['macro_f1']:.6f} val_loss={val_metrics['loss']:.6f} "
                  f"q_gate={val_metrics['q_gate']:.3f} c_gate={val_metrics['c_gate']:.3f} "
                  f"semantic_mix={val_metrics['semantic_mix']:.3f} images/s={train_metrics['images_per_second']:.2f}")
            del base, state, bank
    except torch.cuda.OutOfMemoryError as exc:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("M-TACS CUDA OOM: lower COMPUTE_BATCH_SIZE, restart kernel and Run All. Keep logical batch/cache/horizon fixed; resume from the last completed epoch. No failed batch was skipped.") from exc
    finally:
        del pixels, optimizer, scheduler
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    if not history or history[-1]["epoch"] != config["epochs"]:
        raise RuntimeError("Finish the configured budget before final test/export")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    if best["identity"] != identity:
        raise ValueError("Wrong best checkpoint")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    visual.write_json(output / "training_complete.json", {"identity": identity, "epochs_completed": config["epochs"],
        "best_epoch": best["epoch"], "best_sha256": visual.file_sha256(best_path)})
    return best, history
