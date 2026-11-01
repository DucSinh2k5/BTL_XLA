"""m-LTM-inspired auxiliary alignment cells embedded into the standalone DTD notebook."""
from __future__ import annotations


def alignment_protocol_config():
    return {
        "method": "mltm_inspired_image_caption_v1", "weight": ALIGNMENT_WEIGHT,
        "warmup_epochs": ALIGNMENT_WARMUP_EPOCHS, "batch_size": ALIGNMENT_BATCH_SIZE,
        "entropic_epsilon": ALIGNMENT_OT_EPSILON, "sinkhorn_iterations": ALIGNMENT_SINKHORN_ITERS,
        "max_marginal_error": ALIGNMENT_MAX_MARGINAL_ERROR, "sample_seed": ALIGNMENT_SAMPLE_SEED,
        "metric": "positive_trace_normalized_diagonal_mahalanobis",
        "metric_floor": ALIGNMENT_METRIC_FLOOR, "distance_epsilon": ALIGNMENT_DISTANCE_EPS,
        "image_anchor": "detached_epoch_start_training_candidate_backbone_feature_bank",
        "text_encoder": "shared_fusion_linear_layernorm_l2_without_dropout",
        "target": "uniform_same_canonical_caption_blocks_or_diagonal",
        "scope": "eligible_training_pool_only", "partial_ot": False,
        "paper": "https://arxiv.org/abs/2405.10084",
    }


def validate_alignment_settings():
    if not np.isfinite(ALIGNMENT_WEIGHT) or ALIGNMENT_WEIGHT < 0:
        raise ValueError("ALIGNMENT_WEIGHT must be finite and nonnegative")
    if ALIGNMENT_WARMUP_EPOCHS < 0 or ALIGNMENT_BATCH_SIZE < 2 or ALIGNMENT_SINKHORN_ITERS < 1:
        raise ValueError("Alignment requires nonnegative warmup, batch size >=2 and positive Sinkhorn iterations")
    for value in (ALIGNMENT_OT_EPSILON, ALIGNMENT_MAX_MARGINAL_ERROR, ALIGNMENT_METRIC_FLOOR, ALIGNMENT_DISTANCE_EPS):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Alignment numerical tolerances must be finite and positive")
    if ALIGNMENT_SAMPLE_SEED < 0 or any(seed < 0 for seed in TRAIN_SEEDS):
        raise ValueError("Alignment sampling requires nonnegative seeds")


def make_mltm_alignment_model_class(base_class, visual_class):
    norm_class = make_norm_calibrated_model_class(base_class, visual_class)

    class DTDMLTMAlignmentModel(norm_class):
        """Identical inference/fusion; extra diagonal metric is used by the training loss only."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.alignment_log_diagonal = nn.Parameter(torch.zeros(FEATURE_DIM))

        def alignment_diagonal(self):
            positive = F.softplus(self.alignment_log_diagonal.float()) + ALIGNMENT_METRIC_FLOOR
            return positive / positive.mean()

    return DTDMLTMAlignmentModel


def uniform_log_sinkhorn(cost, epsilon, iterations):
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] < 2:
        raise ValueError("Alignment transport requires a square minibatch with at least two pairs")
    if epsilon <= 0 or iterations < 1 or not torch.isfinite(cost).all():
        raise ValueError("Invalid entropic transport inputs")
    log_kernel = -cost / epsilon
    log_mass = -math.log(cost.shape[0])
    u, v = torch.zeros_like(cost[:, 0]), torch.zeros_like(cost[0])
    for _ in range(iterations):
        u = log_mass - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_mass - torch.logsumexp(log_kernel + u[:, None], dim=0)
    return log_kernel + u[:, None] + v[None, :]


def caption_block_target(texts, device, dtype):
    keys = [normalized_text(text) for text in texts]
    if not keys or any(not key for key in keys):
        raise ValueError("Alignment captions must be nonempty")
    groups = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    codes = torch.tensor([groups[key] for key in keys], device=device)
    positives = codes[:, None].eq(codes[None, :])
    target = positives.to(dtype) / (len(keys) * positives.sum(1, keepdim=True))
    return target, positives, len(groups)


def validate_alignment_pool(banks):
    settings = banks["settings"]
    if settings["name"] != "candidate_only" or settings["query_fusion"] or settings["semantic_selector"]:
        raise ValueError("Alignment must not enable query text or the semantic selector")
    if not settings["candidate_fusion"] or banks["c"]["source_ids"] != CANDIDATE_IDS:
        raise ValueError("Alignment needs correctly paired candidate images and their own captions")
    train_ids = set(SPLIT_FRAMES["train"].image_id.astype(str))
    if len(set(CANDIDATE_IDS)) != len(CANDIDATE_IDS) or not set(CANDIDATE_IDS).issubset(train_ids):
        raise ValueError("Alignment candidate pool must contain unique training images only")
    valid = banks["c"]["valid"].detach().cpu().bool().numpy()
    positions = np.flatnonzero(valid)
    keys = {normalized_text(banks["c"]["texts"][int(i)]) for i in positions}
    if len(positions) < 2 or len(keys) < 2 or "" in keys:
        raise ValueError("Alignment needs at least two eligible training captions with distinct nonempty descriptions")
    return positions


def alignment_batch_positions(banks, training_seed, epoch, step):
    eligible = validate_alignment_pool(banks)
    rng = np.random.default_rng(np.random.SeedSequence([ALIGNMENT_SAMPLE_SEED, training_seed, epoch, step]))
    selected = rng.choice(eligible, size=min(ALIGNMENT_BATCH_SIZE, len(eligible)), replace=False)
    return torch.tensor(selected, dtype=torch.long, device=banks["c"]["embeddings"].device)


def alignment_text_features(network, embeddings):
    # Reuse the actual fusion parameters, skipping dropout for a deterministic alignment target.
    features = embeddings.detach().float()
    for layer in network.context_text_projection:
        if not isinstance(layer, nn.Dropout):
            features = layer(features)
    return F.normalize(features.float(), dim=1, eps=FUSION_NORM_EPS)


def alignment_transport_objective(network, image_features, text_embeddings, texts):
    if (image_features.ndim != 2 or text_embeddings.ndim != 2 or len(texts) < 2
            or len(image_features) != len(texts) or len(text_embeddings) != len(texts)):
        raise ValueError("Alignment requires equally sized minibatches of own-image/caption pairs")
    with torch.autocast(device_type=text_embeddings.device.type, enabled=False):
        images = image_features.detach().to(text_embeddings.device, dtype=torch.float32).clone()
        if not torch.isfinite(images).all() or (images.norm(dim=1) <= FUSION_NORM_EPS).any():
            raise ValueError("Alignment image anchors are zero or non-finite")
        images = F.normalize(images, dim=1, eps=FUSION_NORM_EPS)
        captions = alignment_text_features(network, text_embeddings)
        diagonal = network.alignment_diagonal()
        squared = (images.square() * diagonal).sum(1)[:, None] + (captions.square() * diagonal).sum(1)[None, :]
        squared = squared - 2 * (images * diagonal) @ captions.T
        cost = (squared.clamp_min(0) + ALIGNMENT_DISTANCE_EPS).sqrt()
        log_plan = uniform_log_sinkhorn(cost, ALIGNMENT_OT_EPSILON, ALIGNMENT_SINKHORN_ITERS)
        target, positives, groups = caption_block_target(texts, log_plan.device, log_plan.dtype)
        plan = log_plan.exp()
        marginal_error = torch.maximum((plan.sum(0) - 1 / len(texts)).abs().max(),
                                       (plan.sum(1) - 1 / len(texts)).abs().max()) * len(texts)
        if not torch.isfinite(log_plan).all() or float(marginal_error.detach()) > ALIGNMENT_MAX_MARGINAL_ERROR:
            raise RuntimeError("Alignment Sinkhorn did not converge within the declared tolerance; inspect epsilon/iterations in a new protocol")
        kl = (target * (target.clamp_min(1e-30).log() - log_plan)).sum().clamp_min(0)
        return {"loss": kl, "positive_mass": (plan * positives).sum().detach(),
                "marginal_error": marginal_error.detach(),
                "pair_cosine": (images * captions).sum(1).mean().detach(),
                "pairs": kl.new_tensor(len(texts)), "unique_captions": kl.new_tensor(groups),
                "informative": kl.new_tensor(float(groups > 1)),
                "cost": cost.detach(), "plan": plan.detach(), "positives": positives.detach()}


def candidate_alignment_loss(network, banks, feature_bank, epoch, step):
    positions = alignment_batch_positions(banks, network.alignment_training_seed, epoch, step)
    if feature_bank.ndim != 2 or len(feature_bank) != len(CANDIDATE_IDS) or feature_bank.shape[1] != FEATURE_DIM:
        raise ValueError("Alignment image bank does not match candidate order/dimension")
    images = feature_bank.index_select(0, positions.to(feature_bank.device))
    embeddings = banks["c"]["embeddings"].index_select(0, positions)
    texts = [banks["c"]["texts"][int(position)] for position in positions.cpu()]
    return alignment_transport_objective(network, images, embeddings, texts)


def write_alignment_manifest(banks, run_dir):
    eligible = set(int(position) for position in validate_alignment_pool(banks))
    rows = [{"candidate_position": i, "image_id": image_id,
             "caption_source_id": banks["c"]["source_ids"][i], "alignment_eligible": i in eligible,
             "canonical_caption_group": normalized_text(banks["c"]["texts"][i]) if i in eligible else ""}
            for i, image_id in enumerate(CANDIDATE_IDS)]
    pd.DataFrame(rows).to_csv(Path(run_dir) / "alignment_pair_manifest.csv", index=False)
    write_json(Path(run_dir) / "alignment_config.json", alignment_protocol_config())


def training_pool_alignment_summary(network, banks, feature_bank):
    eligible = validate_alignment_pool(banks)
    positions = torch.tensor(eligible, dtype=torch.long, device=banks["c"]["embeddings"].device)
    was_training = network.training
    network.eval()
    try:
        with torch.inference_mode():
            result = alignment_transport_objective(network,
                feature_bank.index_select(0, positions.to(feature_bank.device)),
                banks["c"]["embeddings"].index_select(0, positions),
                [banks["c"]["texts"][int(i)] for i in eligible])
            plan, positive = result["plan"], result["positives"]
            rows = torch.arange(len(eligible), device=plan.device)
            columns = plan.argmax(1)
            diagonal = network.alignment_diagonal()
            return pd.DataFrame([{"scope": "training_pool_fit_not_held_out_retrieval", "alignment_enabled": ALIGNMENT_WEIGHT > 0,
                "eligible_pairs": len(eligible), "unique_captions": int(result["unique_captions"]),
                "transport_kl": float(result["loss"]), "positive_mass": float(result["positive_mass"]),
                "marginal_error": float(result["marginal_error"]), "mean_pair_cosine": float(result["pair_cosine"]),
                "caption_group_top1": float(positive[rows, columns].float().mean()),
                "exact_source_top1": float(rows.eq(columns).float().mean()),
                "metric_diagonal_min": float(diagonal.min()), "metric_diagonal_max": float(diagonal.max()),
                "metric_condition_ratio": float(diagonal.max() / diagonal.min())}])
    finally:
        network.train(was_training)
