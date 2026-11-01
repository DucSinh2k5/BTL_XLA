"""Memory-bounded visual training functions injected by the notebook repair builder.

The builder retains the original inference model and export implementation. These
functions execute in that training module's namespace, not as a separate runtime.
"""


def default_config():
    config = _legacy_default_config()
    config.update(batch_size=64, effective_batch_size=64, workers=4,
                  gradient_checkpointing=True, feature_batch_size=32,
                  eval_batch_size=8, st_bank_chunk=128,
                  training_engine="memory_bounded_two_pass_v2",
                  candidate_training_pixels="fp16_quantized_values_exact_st_auto_cpu_or_gpu_storage")
    return config


def validate_config(config):
    _legacy_validate_config(config)
    if config.get("training_engine") != "memory_bounded_two_pass_v2":
        raise ValueError("Use the repaired training engine configuration")
    if not config.get("gradient_checkpointing"):
        raise ValueError("Keep gradient_checkpointing=True; logical batch is not GPU compute batch")


def execution_options(config, execution=None):
    options = {"compute_batch_size": min(8, config["batch_size"]),
               "pixel_bank_device": "auto", "gpu_reserve_gib": 20.0,
               "profile_batches": 3}
    supplied = execution or {}
    if set(supplied) - set(options):
        raise ValueError("Unknown execution options: " + str(sorted(set(supplied) - set(options))))
    options.update(supplied)
    if not isinstance(options["compute_batch_size"], int) or not 1 <= options["compute_batch_size"] <= config["batch_size"]:
        raise ValueError("compute_batch_size must be in 1..logical batch_size")
    if options["pixel_bank_device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("pixel_bank_device must be auto, cpu, or cuda")
    if options["gpu_reserve_gib"] < 0 or options["profile_batches"] < 0:
        raise ValueError("Invalid profiling or GPU reserve")
    return options


def config_for_resume(requested, output, resume_from=None):
    """Preserve the logical experiment, including legacy reward batch size."""
    saved = None
    for folder in (Path(output), Path(resume_from) if resume_from else None):
        if folder is not None and (folder / "protocol.json").is_file():
            saved = runtime.read_json(folder / "protocol.json")["config"]
            break
    if saved is None:
        return requested
    mutable = {"gradient_checkpointing", "workers", "feature_batch_size", "eval_batch_size",
               "st_bank_chunk", "training_engine", "batch_size", "effective_batch_size", "candidate_training_pixels"}
    for key, value in saved.items():
        if key not in mutable and requested.get(key) != value:
            raise ValueError(f"Cannot resume after changing {key}: {value!r} -> {requested.get(key)!r}")
    # GPU chunking is independent; do not silently change reward normalization
    # from a legacy logical batch of 8 to a new logical batch of 64.
    result = dict(requested)
    for key in ("batch_size", "effective_batch_size"):
        result[key] = saved[key]
    if saved.get("training_engine") == "memory_bounded_two_pass_v2":
        result = dict(saved)
    print("Registered logical/effective batch:", result["batch_size"], result["effective_batch_size"])
    return result


def _legacy_identity(config, classes, frames, assets):
    ids = {key: [(r["image_id"], int(r["class_id"])) for r in value] for key, value in frames.items()}
    payload = {"config": config, "classes": classes, "splits_and_pool": ids, "assets": assets,
               "model_source_sha256": LEGACY_SOURCE_SHA256["runtime"],
               "training_source_sha256": LEGACY_SOURCE_SHA256["training"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def migrate_legacy_run(source, output, config, classes, frames, assets):
    """Copy checked checkpoints to a new run; never rewrite the legacy run."""
    output = Path(output)
    if (output / "protocol.json").is_file() or source is None:
        return None
    source = Path(source).resolve()
    if not (source / "protocol.json").is_file():
        print("No previous run protocol found; starting epoch 1")
        return None
    if source == output.resolve():
        raise ValueError("Repair requires a separate OUTPUT_DIR; keep the original run untouched")
    protocol = runtime.read_json(source / "protocol.json")
    if protocol.get("source_sha256") != LEGACY_SOURCE_SHA256:
        raise ValueError("Previous source is not the checked legacy visual TACS implementation")
    old_config = protocol["config"]
    allowed = {"gradient_checkpointing", "workers", "feature_batch_size", "eval_batch_size", "st_bank_chunk", "candidate_training_pixels"}
    expected_config = {**old_config, "training_engine": "memory_bounded_two_pass_v2"}
    for key in allowed:
        expected_config[key] = config[key]
    if expected_config != config:
        raise ValueError("Legacy resume may not change model/loss, logical batch, seed, horizon, or pool settings")
    old_identity = _legacy_identity(old_config, classes, frames, assets)
    if protocol["identity"] != old_identity or protocol["assets"] != assets:
        raise ValueError("Legacy dataset/pool/order/assets/config identity does not match this run")
    last_path, best_path = source / "last.pt", source / "best.pt"
    if not last_path.is_file():
        if best_path.is_file():
            raise FileNotFoundError("best.pt exists but last.pt is missing; optimizer resume requires both")
        print("Previous run saved no completed epoch; starting epoch 1, not resuming nonexistent weights")
        return {"source": str(source), "from_identity": old_identity, "completed_epochs": 0,
                "note": "Previous run failed before its first epoch checkpoint"}
    if not best_path.is_file():
        raise FileNotFoundError("Legacy resume requires best.pt as well as last.pt")
    if any((output / name).exists() for name in ("last.pt", "best.pt", "migration.json")):
        raise FileExistsError("Incomplete migration in the new output; preserve and inspect it before retrying")
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    for item in (last, best):
        if item["identity"] != old_identity or item["config"] != old_config or item["class_names"] != classes:
            raise ValueError("Legacy checkpoint/protocol/class order mismatch")
    if not last.get("history") or last["history"][-1]["epoch"] != last["epoch"] or not 1 <= best["epoch"] <= last["epoch"] <= config["epochs"]:
        raise ValueError("Invalid legacy epoch/history ordering")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng"):
        if key not in last:
            raise ValueError("Legacy last.pt lacks resume state: " + key)
    audit = {"source": str(source), "from_identity": old_identity,
             "completed_epochs": last["epoch"],
             "legacy_checkpoints": {name: runtime.file_sha256(source / name) for name in ("last.pt", "best.pt")},
             "legacy_source_sha256": LEGACY_SOURCE_SHA256,
             "note": "Optimizer/scheduler/scaler/RNG and best weights retained; new stochastic execution order is not bitwise legacy replay"}
    identity = run_identity(config, classes, frames, assets)
    output.mkdir(parents=True, exist_ok=True)
    for name, item in (("best.pt", best), ("last.pt", last)):
        item.update(identity=identity, config=dict(config), migration=audit)
        atomic_torch_save(output / name, item)
    runtime.write_json(output / "migration.json", audit)
    fields = list(dict.fromkeys(key for row in last["history"] for key in row))
    csv_write(output / "history.csv", last["history"], fields=fields)
    print(f"Copied checked legacy epoch {last['epoch']} into {output}; originals remain unchanged")
    return audit


def prepare_training_pixels(candidates, config, device, execution):
    cpu_bank = build_image_bank(candidates, config)
    device = torch.device(device)
    use_gpu = device.type == "cuda" and execution["pixel_bank_device"] != "cpu"
    if execution["pixel_bank_device"] == "cuda" and device.type != "cuda":
        raise ValueError("CUDA pixel bank requested without a CUDA device")
    if use_gpu:
        required = cpu_bank.numel() * 4
        free, _ = torch.cuda.mem_get_info(device)
        use_gpu = free >= required + int(execution["gpu_reserve_gib"] * 1024**3)
        if execution["pixel_bank_device"] == "cuda" and not use_gpu:
            raise MemoryError("Not enough free VRAM for the float32 pixel bank plus training reserve")
    if use_gpu:
        try:
            bank = cpu_bank.to(device=device, dtype=torch.float32)
        except torch.cuda.OutOfMemoryError:
            if execution["pixel_bank_device"] == "cuda":
                raise
            torch.cuda.empty_cache()
            bank = cpu_bank
    else:
        bank = cpu_bank
    print(f"Training pixel bank: {bank.device}, {bank.dtype}, {bank.numel()*bank.element_size()/1024**3:.2f} GiB")
    if bank.device.type == "cpu":
        print("CPU fallback: full straight-through gradients retained, but bank transfers can be slow")
    return bank


def refresh_bank(model, candidates, config, device, image_bank=None, amp=False):
    # Keep raw fp32 preprocessing for checkpoint-bound validation/export features.
    was_training = model.training
    model.eval()
    options = {**config, "eval_batch_size": config["feature_batch_size"]}
    chunks = []
    try:
        with torch.no_grad():
            for batch in tqdm(make_loader(candidates, options), desc="Refresh candidate features", leave=False):
                images = batch["image"].to(device, non_blocking=True)
                with autocast_context(device, amp):
                    features = model.encode_selector(images)
                chunks.append(features.detach().float().cpu())
    finally:
        model.train(was_training)
    bank = torch.cat(chunks)
    if not torch.isfinite(bank).all():
        raise FloatingPointError("Candidate features contain NaN/Inf")
    return bank


def _selector_rng(device):
    device = torch.device(device)
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


@contextlib.contextmanager
def _replay_selector_rng(state, device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state["cpu"])
        if devices:
            torch.cuda.set_rng_state(state["cuda"], device)
        yield


def policy_advantage(rewards):
    return ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)).detach()


def _selected_pixels(bank, actions, device):
    return bank.index_select(0, actions.to(bank.device)).to(device=device, dtype=torch.float32, non_blocking=True)


def _reward_pass(model, query, labels, actions, pixels, device):
    model.eval()
    try:
        with torch.no_grad(), autocast_context(device):
            baseline = model(query)
            paired = model(query, _selected_pixels(pixels, actions, device))
            reward = F.cross_entropy(baseline.float(), labels, reduction="none") - F.cross_entropy(paired.float(), labels, reduction="none")
        return reward.detach().cpu()
    finally:
        model.train()


def train_epoch(model, loader, feature_bank, pixel_bank, candidates, optimizer, scaler, config, device, epoch, execution=None):
    model.train()
    model.gradient_checkpointing = True
    device = torch.device(device)
    options = execution_options(config, execution)
    compute = options["compute_batch_size"]
    bank = feature_bank.to(device)
    positions = {row["image_id"]: i for i, row in enumerate(candidates)}
    accumulate = config["effective_batch_size"] // config["batch_size"]
    stats = {key: 0.0 for key in ("loss", "ce_loss", "policy_loss", "reward", "entropy")}
    truth, predictions = [], []
    successful = skipped = 0
    optimizer.zero_grad(set_to_none=True)
    if not len(loader):
        raise ValueError("Training loader is empty")
    started_epoch = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} train / logical={config['batch_size']} compute={compute}")
    for step, batch in enumerate(progress):
        profile = step < options["profile_batches"]
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        count = len(batch["label"])
        actions_list, rewards, states, sampled_scores = [], [], [], []
        # No autograd graphs survive this prepass. Only small score tensors and
        # RNG states are retained so policy log-probabilities can be replayed.
        for offset in range(0, count, compute):
            end = min(offset + compute, count)
            query = batch["image"][offset:end].to(device, non_blocking=True)
            labels = batch["label"][offset:end].to(device, non_blocking=True)
            excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"][offset:end]], device=device)
            states.append(_selector_rng(device))
            with torch.no_grad(), autocast_context(device):
                scores, _ = model.score(query, bank, excluded)
                actions = torch.distributions.Categorical(logits=scores).sample()
            sampled_scores.append(scores.detach().cpu())
            actions_list.append(actions.cpu())
            rewards.append(_reward_pass(model, query, labels, actions, pixel_bank, device))
            del query, labels, excluded, scores, actions
        all_rewards = torch.cat(rewards)
        advantages = policy_advantage(all_rewards)
        divisor = min(accumulate, len(loader) - (step // accumulate) * accumulate)
        chunk_index = 0
        for offset in range(0, count, compute):
            end = min(offset + compute, count)
            query = batch["image"][offset:end].to(device, non_blocking=True)
            labels = batch["label"][offset:end].to(device, non_blocking=True)
            excluded = torch.tensor([positions.get(i, -1) for i in batch["image_id"][offset:end]], device=device)
            with autocast_context(device):
                with _replay_selector_rng(states[chunk_index], device):
                    scores, _ = model.score(query, bank, excluded)
                # Score replay must represent the policy that sampled each action.
                if not torch.allclose(scores.detach(), sampled_scores[chunk_index].to(device), atol=1e-4, rtol=1e-4):
                    raise RuntimeError("Selector RNG replay mismatch; refusing inconsistent policy gradients")
                weights = F.gumbel_softmax(scores, tau=config["gumbel_tau"], hard=True, dim=1)
                chunk_size = len(pixel_bank) if pixel_bank.device.type == "cuda" else config["st_bank_chunk"]
                context = runtime.straight_through_images(weights, pixel_bank, chunk_size)
                logits = model(query, context)
                ce = F.cross_entropy(logits.float(), labels, label_smoothing=config["label_smoothing"])
                distribution = torch.distributions.Categorical(logits=scores)
                log_probs = distribution.log_prob(actions_list[chunk_index].to(device))
                policy = -(log_probs * advantages[offset:end].to(device)).mean()
                entropy = distribution.entropy().mean()
                loss = ce + config["policy_weight"] * policy - config["entropy_weight"] * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch}, logical batch {step}")
            scaler.scale(loss * ((end - offset) / count) / divisor).backward()
            for key, value in (("loss", loss), ("ce_loss", ce), ("policy_loss", policy), ("entropy", entropy)):
                stats[key] += float(value.detach()) * (end - offset)
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
            del query, labels, excluded, scores, weights, context, logits, ce, distribution, log_probs, policy, entropy, loss
            chunk_index += 1
        truth.extend(batch["label"].tolist())
        stats["reward"] += float(all_rewards.sum())
        if (step + 1) % accumulate == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.is_enabled() and scaler.get_scale() < previous_scale
            skipped += int(overflow)
            successful += int(not overflow)
            optimizer.zero_grad(set_to_none=True)
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            tqdm.write(f"Profile batch {step+1}: {seconds:.2f}s, {count/seconds:.2f} images/s, peak allocated={peak:.2f} GiB, pixel bank={pixel_bank.device}")
        del actions_list, rewards, states, sampled_scores, all_rewards, advantages
    if not successful:
        raise FloatingPointError("Every optimizer step overflowed")
    elapsed = max(time.perf_counter() - started_epoch, 1e-9)
    return {**{k: v / len(truth) for k, v in stats.items()}, **classification_metrics(truth, predictions, config["num_classes"]),
            "optimizer_steps": successful, "amp_skipped_steps": skipped,
            "images_per_second": len(truth) / elapsed, "compute_batch_size": compute,
            "pixel_bank_device": str(pixel_bank.device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0}


def train_run(model, classes, train, val, candidates, config, assets, output, device, execution=None, resume_from=None):
    validate_config(config)
    options = execution_options(config, execution)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {"train": train, "val": val, "pool": candidates}
    identity = run_identity(config, classes, frames, assets)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file() and runtime.read_json(protocol_path)["identity"] != identity:
        raise ValueError("Output directory belongs to another run. Keep the registered logical configuration")
    migration = migrate_legacy_run(resume_from, output, config, classes, frames, assets)
    if migration is None and protocol_path.is_file():
        migration = runtime.read_json(protocol_path).get("migration")
    protocol = {"identity": identity, "config": config, "assets": assets,
                "source_sha256": {"runtime": runtime.file_sha256(runtime.__file__), "training": runtime.file_sha256(__file__)},
                "environment": environment(), "execution": options, "migration": migration}
    runtime.write_json(protocol_path, protocol)
    csv_write(output / "train_split.csv", train)
    csv_write(output / "validation_split.csv", val)
    csv_write(output / "candidate_pool.csv", candidates)
    runtime.write_json(output / "class_names.json", classes)
    train_loader, val_loader = make_loader(train, config, True), make_loader(val, config)
    model.gradient_checkpointing = True
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
    history, best_loss, start_epoch = [], math.inf, 1
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        if last["identity"] != identity or last["config"] != config:
            raise ValueError("Resume identity/config mismatch")
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        restore_rng(last["rng"], train_loader)
        history, best_loss, start_epoch = last["history"], last["best_val_loss"], last["epoch"] + 1
        del last
        if not best_path.is_file():
            raise FileNotFoundError("Resume requires best.pt as well as last.pt")
        print(f"Resuming after epoch {start_epoch-1}; horizon remains {config['epochs']}")
    print("Logical batch:", config["batch_size"], "| GPU compute batch:", options["compute_batch_size"], "| checkpointing: on")
    pixels = prepare_training_pixels(candidates, config, device, options) if start_epoch <= config["epochs"] else None
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            started = time.perf_counter()
            bank = refresh_bank(model, candidates, config, device, amp=True)
            first_refresh_seconds = time.perf_counter() - started
            train_metrics = train_epoch(model, train_loader, bank, pixels, candidates, optimizer, scaler, config, device, epoch, options)
            refresh_started = time.perf_counter()
            bank = refresh_bank(model, candidates, config, device, amp=True)
            second_refresh_seconds = time.perf_counter() - refresh_started
            validation_started = time.perf_counter()
            val_metrics, _ = evaluate(model, val_loader, bank, pixels, candidates, config, device, amp=True)
            validation_seconds = time.perf_counter() - validation_started
            improved = val_metrics["loss"] < best_loss
            best_loss = min(best_loss, val_metrics["loss"])
            history.append({"epoch": epoch, **{"train_"+k: v for k, v in train_metrics.items()},
                            **{"val_"+k: v for k, v in val_metrics.items()}, "seconds": time.perf_counter()-started,
                            "backbone_lr": optimizer.param_groups[0]["lr"], "training_engine": config["training_engine"],
                            "refresh_seconds": first_refresh_seconds + second_refresh_seconds,
                            "validation_seconds": validation_seconds})
            scheduler.step()
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            base = {"schema": runtime.SCHEMA, "identity": identity, "epoch": epoch, "model_state_dict": state,
                    "config": config, "class_names": classes, "val_metrics": val_metrics, "migration": migration}
            if improved:
                atomic_torch_save(best_path, base)
            atomic_torch_save(last_path, {**base, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                                         "scaler_state_dict": scaler.state_dict(), "rng": rng_state(train_loader),
                                         "history": history, "best_val_loss": best_loss})
            fields = list(dict.fromkeys(key for row in history for key in row))
            csv_write(output / "history.csv", history, fields=fields)
            print(f"epoch={epoch:03d}/{config['epochs']} val_loss={val_metrics['loss']:.6f} val_f1={val_metrics['macro_f1']:.6f} images/s={train_metrics['images_per_second']:.2f} best={improved}")
            del state, base, bank
    except torch.cuda.OutOfMemoryError as exc:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("CUDA OOM: reduce COMPUTE_BATCH_SIZE (not LOGICAL_BATCH_SIZE), restart kernel, then Run All. The last completed epoch remains resumable; no failed batch was skipped.") from exc
    finally:
        del pixels, optimizer, scheduler
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    if not history or history[-1]["epoch"] != config["epochs"]:
        raise RuntimeError("Training budget is incomplete; final test/export are blocked")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    runtime.write_json(output / "training_complete.json", {"identity": identity, "epochs_completed": config["epochs"],
                                                         "best_epoch": best["epoch"], "best_sha256": runtime.file_sha256(best_path)})
    return best, history
