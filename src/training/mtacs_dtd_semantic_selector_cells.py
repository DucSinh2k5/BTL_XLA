"""Image-to-candidate-caption selector and validation interventions for study04."""
from __future__ import annotations


def semantic_selector_config():
    return {
        "method": "aligned_query_image_candidate_caption_residual_v1",
        "max_weight": SELECTOR_SEMANTIC_MAX_WEIGHT,
        "visual_warmup_epochs": SELECTOR_VISUAL_WARMUP_EPOCHS,
        "ramp_epochs": SELECTOR_SEMANTIC_RAMP_EPOCHS,
        "semantic_z_clip": SELECTOR_SEMANTIC_Z_CLIP,
        "score": "negative_shared_alignment_diagonal_mahalanobis_distance",
        "normalization": "semantic_z_over_valid_nonself_candidates; original_visual_z_unchanged",
        "fallback": "zero_semantic_residual_for_invalid_or_constant_or_insufficient_captions",
        "weight": "max_weight * sigmoid(existing_semantic_mix_logit) * checkpoint_epoch_ramp",
        "query_caption": False, "candidate_scope": "training_pool_only",
        "transport_at_inference": False,
    }


def validate_semantic_selector_settings():
    validate_alignment_settings()
    if not np.isfinite(SELECTOR_SEMANTIC_MAX_WEIGHT) or not 0 <= SELECTOR_SEMANTIC_MAX_WEIGHT <= 1:
        raise ValueError("SELECTOR_SEMANTIC_MAX_WEIGHT must be finite and in [0, 1]")
    if SELECTOR_VISUAL_WARMUP_EPOCHS < 0 or SELECTOR_SEMANTIC_RAMP_EPOCHS < 1:
        raise ValueError("Selector warmup must be nonnegative and ramp must be positive")
    if not np.isfinite(SELECTOR_SEMANTIC_Z_CLIP) or SELECTOR_SEMANTIC_Z_CLIP <= 0:
        raise ValueError("Semantic score clip must be finite and positive")
    if ALIGNMENT_WEIGHT <= 0 and SELECTOR_SEMANTIC_MAX_WEIGHT > 0:
        raise ValueError("The cross-modal selector requires positive own-image/caption alignment supervision")


def make_semantic_selector_model_class(base_class, visual_class):
    aligned_class = make_mltm_alignment_model_class(base_class, visual_class)

    class DTDAlignedSemanticSelector(aligned_class):
        """Retain study03 fusion/loss; use its learned metric for a bounded retrieval residual."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_buffer("semantic_selector_epoch", torch.tensor(0, dtype=torch.long))

        def semantic_selector_weight(self):
            epoch = int(self.semantic_selector_epoch.item())
            ramp = min(1.0, max(0.0, (epoch - SELECTOR_VISUAL_WARMUP_EPOCHS)
                                / SELECTOR_SEMANTIC_RAMP_EPOCHS))
            return SELECTOR_SEMANTIC_MAX_WEIGHT * ramp * torch.sigmoid(self.selector.semantic_mix_logit)

        def select_context(self, query_images, feature_bank, q_text, q_valid, banks,
                           excluded_positions, semantic=False, sample=False):
            weight = self.semantic_selector_weight()
            if not semantic or SELECTOR_SEMANTIC_MAX_WEIGHT == 0 or float(weight.detach()) == 0:
                # Exact study03 path during warmup and for the zero-weight control.
                details = super().select_context(query_images, feature_bank, q_text, q_valid, banks,
                                                 excluded_positions, semantic=False, sample=sample)
                details["semantic_weight"] = weight.detach() * 0
                return details
            if len(feature_bank) != len(CANDIDATE_IDS):
                raise ValueError("Selector feature bank does not match the training candidate order")
            visual_raw, query_features = self.score_full_candidate_pool(query_images, feature_bank)
            visual = visual_raw.float()
            visual = (visual - visual.mean(1, keepdim=True)) / visual.std(
                1, keepdim=True, unbiased=False).clamp_min(FUSION_NORM_EPS)
            positions = excluded_positions.to(visual.device, dtype=torch.long)
            if positions.shape != (len(visual),) or ((positions < -1) | (positions >= len(feature_bank))).any():
                raise ValueError("Invalid self-exclusion positions")
            allowed = torch.ones_like(visual, dtype=torch.bool)
            in_pool = positions >= 0
            allowed[torch.arange(len(visual), device=visual.device)[in_pool], positions[in_pool]] = False
            if not allowed.any(1).all():
                raise ValueError("No candidate remains after self-exclusion")
            candidate = banks.get("selector_c", banks["c"])
            available = candidate["valid"].to(visual.device, dtype=torch.bool)
            if available.shape != (len(feature_bank),) or len(candidate["embeddings"]) != len(feature_bank):
                raise ValueError("Selector captions do not match the candidate bank")
            eligible = torch.where(available)[0]
            residual = torch.zeros_like(visual)
            if len(eligible):
                with torch.autocast(device_type=query_features.device.type, enabled=False):
                    images = F.normalize(query_features.float(), dim=1, eps=FUSION_NORM_EPS)
                    captions = alignment_text_features(self, candidate["embeddings"].index_select(
                        0, eligible.to(candidate["embeddings"].device)))
                    diagonal = self.alignment_diagonal()
                    squared = ((images.square() * diagonal).sum(1)[:, None]
                               + (captions.square() * diagonal).sum(1)[None, :]
                               - 2 * (images * diagonal) @ captions.T)
                    raw = -(squared.clamp_min(0) + ALIGNMENT_DISTANCE_EPS).sqrt()
                    if not torch.isfinite(raw).all():
                        raise ValueError("Non-finite image-caption selector distances")
                    raw = torch.zeros_like(visual).index_copy(1, eligible, raw)
                    valid = allowed & available[None, :]
                    count = valid.sum(1, keepdim=True)
                    mean = raw.masked_fill(~valid, 0).sum(1, keepdim=True) / count.clamp_min(1)
                    centered = (raw - mean).masked_fill(~valid, 0)
                    variance = centered.square().sum(1, keepdim=True) / count.clamp_min(1)
                    usable = (count >= 2) & (variance > FUSION_NORM_EPS ** 2)
                    residual = (centered / variance.clamp_min(FUSION_NORM_EPS ** 2).sqrt()).clamp(
                        -SELECTOR_SEMANTIC_Z_CLIP, SELECTOR_SEMANTIC_Z_CLIP)
                    residual = residual.masked_fill(~(valid & usable), 0)
            scores = (visual + weight * residual).masked_fill(~allowed, -torch.inf)
            distribution = torch.distributions.Categorical(logits=scores)
            indices = distribution.sample() if sample else scores.argmax(1)
            return {"indices": indices, "log_prob": distribution.log_prob(indices),
                    "entropy": distribution.entropy().mean(), "semantic_weight": weight.detach(),
                    "semantic_scores": residual, "selector_scores": scores,
                    "visual_indices": visual.masked_fill(~allowed, -torch.inf).argmax(1)}

    return DTDAlignedSemanticSelector


def collect_selector_validation_interventions(network, loader, banks, feature_bank, seed):
    if banks["settings"]["query_fusion"] or not banks["settings"]["semantic_selector"]:
        raise ValueError("Study04 diagnostics require no query caption and an enabled semantic selector")
    allowed_ids = set(SPLIT_FRAMES["val"].image_id.astype(str))
    shuffled, assignment = make_validation_caption_shuffle(banks, seed)
    modes = {
        "aligned": banks,
        "visual_selector_only": {**banks, "settings": {**banks["settings"], "semantic_selector": False}},
        "selector_text_shuffled": {**banks, "selector_c": shuffled["c"]},
    }
    frames, probabilities, metrics = {}, {}, {}
    old_training = network.training

    def validation_batches():
        seen = set()
        for batch in loader:
            ids = [str(image_id) for image_id in batch["image_id"]]
            if len(set(ids)) != len(ids) or set(ids) - allowed_ids or seen.intersection(ids):
                raise ValueError("Selector diagnostics received duplicate or non-validation query IDs")
            seen.update(ids)
            yield batch

    try:
        with torch.inference_mode():
            for name, mode_banks in modes.items():
                values, frame, probs = evaluate_ablation(network, validation_batches(), mode_banks, feature_bank)
                if (frame.image_id.duplicated().any() or set(frame.image_id) != allowed_ids
                        or not torch.isfinite(probs).all()):
                    raise ValueError("Selector diagnostics require exact validation coverage and finite probabilities")
                positions = frame.selected_context_position.to_numpy(dtype=int)
                selector_c = mode_banks.get("selector_c", mode_banks["c"])
                frame = frame.assign(intervention=name,
                    selector_caption_source_id=[selector_c["source_ids"][i] for i in positions],
                    selector_caption=[selector_c["texts"][i] for i in positions],
                    semantic_selector_enabled=name != "visual_selector_only",
                    semantic_weight=float(network.semantic_selector_weight()) if name != "visual_selector_only" else 0.0)
                frames[name], probabilities[name], metrics[name] = frame, probs, values
    finally:
        network.train(old_training)
    baseline, base_prob = frames["aligned"], probabilities["aligned"]
    summary = []
    for name, frame in frames.items():
        if frame.image_id.tolist() != baseline.image_id.tolist():
            raise ValueError("Validation diagnostic order changed between interventions")
        probs = probabilities[name]
        summary.append({"intervention": name, "split": "val", "query_count": len(frame),
            "scope": "same_checkpoint_selector_intervention_not_retrained_ablation",
            **metrics[name], **paired_f1_interval(baseline, frame, BOOTSTRAP_REPEATS, seed),
            "context_changed_rate": float((frame.selected_context_id != baseline.selected_context_id).mean()),
            "mean_abs_probability_change": float((probs - base_prob).abs().mean()),
            "semantic_weight": float(frame.semantic_weight.iloc[0]),
            "shuffle_testable": int(assignment.caption_available.sum()) >= 2})
    return pd.DataFrame(summary), pd.concat(frames.values(), ignore_index=True), assignment


def run_selector_validation_interventions(spec, seed):
    if not is_run_completed(spec, seed):
        raise RuntimeError("Complete the declared study04 epoch budget before selector diagnostics")
    run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
    checkpoint = run_dir / "best.pt"
    completed = read_json(run_dir / "completed.json")
    target = run_dir / "validation_selector_diagnostics"
    files = {"summary": target / "validation_selector_interventions.csv",
             "predictions": target / "validation_selector_predictions.csv",
             "assignments": target / "selector_caption_shuffle_assignments.csv"}
    metadata = {"schema": "dtd_aligned_semantic_selector_validation_v1", "identity": run_identity(spec, seed),
        "best_epoch": completed["best_epoch"], "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns, "selector": semantic_selector_config(),
        "shuffle_seed": DIAGNOSTIC_SHUFFLE_SEED + seed, "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "split": "val", "workers": 0, "fusion_caption_policy": "aligned_own_caption_of_reselected_image"}
    if (target / "metadata.json").is_file():
        if read_json(target / "metadata.json") != metadata:
            raise ValueError("Cached selector diagnostics belong to another checkpoint/settings")
        if all(path.is_file() for path in files.values()):
            return pd.read_csv(files["summary"])
    network = feature_bank = banks = loader = None
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (state["identity"] != run_identity(spec, seed) or state["class_names"] != class_names
                or int(state["epoch"]) != int(completed["best_epoch"])):
            raise ValueError("Wrong study04 checkpoint identity, class mapping or epoch")
        network = create_ablation_model(seed)
        network.load_state_dict(state["model"], strict=True)
        if int(network.semantic_selector_epoch) != int(state["epoch"]):
            raise ValueError("Selector ramp epoch does not match the best checkpoint")
        del state
        feature_bank, banks = refresh_ablation_bank(network), make_ablation_banks(spec, seed)
        loader = make_norm_validation_loader(seed)
        summary, predictions, assignments = collect_selector_validation_interventions(
            network, loader, banks, feature_bank, DIAGNOSTIC_SHUFFLE_SEED + seed)
        summary = summary.assign(experiment=spec["name"], seed=seed, best_epoch=completed["best_epoch"],
                                 study_stage=ARCHITECTURE_STUDY["stage"])
        target.mkdir(parents=True, exist_ok=True)
        for key, frame in (("summary", summary), ("predictions", predictions), ("assignments", assignments)):
            frame.to_csv(files[key], index=False)
        write_json(target / "metadata.json", metadata)
        return summary
    finally:
        del network, feature_bank, banks, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
