"""Embedded notebook helpers; runtime globals are supplied by the DTD notebook."""
from __future__ import annotations


def validate_norm_output_directory(path):
    path = Path(path)
    protocol_path = path / "protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (previous.get("schema") != "dtd_prefix_norm_fusion_v1"
                or previous.get("architecture_study", {}).get("stage") != ARCHITECTURE_STUDY["stage"]):
            raise ValueError("Output belongs to another experiment; study02 must not overwrite study01 or other runs")
    elif path.is_dir() and any(path.glob("runs/*/seed_*/*.pt")):
        raise ValueError("Output has checkpoints but no study02 protocol; choose a new output directory")


def make_norm_calibrated_model_class(base_class, visual_class):
    original_class = make_ablation_model_class(base_class, visual_class)

    class DTDNormCalibratedModel(original_class):
        """Same parameters/selector; normalize candidate text before gate and fusion."""

        def forward_ablated(self, query_images, context_images, q_text, q_valid, c_text, c_valid, settings):
            if context_images is None:
                visual = self.query_norm(self._pool_backbone_features(query_images))
            else:
                _, visual, _ = visual_class.forward_selected_context(
                    self, query_images, context_images, return_features=True)
            q_features = self.text_projection(q_text.float())
            c_projected = self.context_text_projection(c_text.float())
            c_features = F.normalize(c_projected.float(), dim=1, eps=FUSION_NORM_EPS)
            q_gate = torch.sigmoid(self.text_gate(torch.cat([visual.float(), q_features], 1))).squeeze(1)
            c_gate = torch.sigmoid(self.context_text_gate(torch.cat([visual.float(), c_features], 1))).squeeze(1)
            q_gate = q_gate * q_valid.to(q_gate.dtype) * bool(settings["query_fusion"])
            c_gate = c_gate * c_valid.to(c_gate.dtype) * bool(settings["candidate_fusion"])
            fused = self.mtacs_norm(visual.float() + q_gate[:, None] * q_features + c_gate[:, None] * c_features)
            result = {"logits": self.mtacs_classifier(fused), "features": F.normalize(fused, dim=1),
                      "query_gate": q_gate, "candidate_gate": c_gate}
            if getattr(self, "capture_fusion_diagnostics", False):
                visual_norm = visual.detach().float().norm(dim=1)
                weighted_norm = (c_gate[:, None] * c_features).detach().float().norm(dim=1)
                result["fusion_diagnostics"] = {
                    "visual_norm": visual_norm,
                    "candidate_raw_projection_norm": c_projected.detach().float().norm(dim=1),
                    "candidate_fusion_norm": c_features.detach().float().norm(dim=1),
                    "weighted_candidate_norm": weighted_norm,
                    "weighted_text_visual_ratio": weighted_norm / visual_norm.clamp_min(FUSION_NORM_EPS),
                }
            return result

    return DTDNormCalibratedModel


def make_validation_caption_shuffle(banks, seed):
    candidate = banks["c"]
    if candidate["source_ids"] != CANDIDATE_IDS:
        raise ValueError("Expected aligned training-pool caption IDs before validation shuffle")
    valid = candidate["valid"].detach().cpu().bool().numpy()
    positions = np.arange(len(CANDIDATE_IDS))
    eligible = np.flatnonzero(valid)
    if len(eligible) >= 2:
        order = np.random.default_rng(seed).permutation(eligible)
        positions[order] = np.roll(order, 1)
    device_positions = torch.tensor(positions, dtype=torch.long, device=candidate["embeddings"].device)
    shuffled = {**banks, "c": {**candidate,
        "embeddings": candidate["embeddings"].index_select(0, device_positions),
        "valid": candidate["valid"].clone(),
        "source_ids": [candidate["source_ids"][int(i)] for i in positions],
        "texts": [candidate["texts"][int(i)] for i in positions],
    }}
    assignment = pd.DataFrame({
        "candidate_position": np.arange(len(positions)), "candidate_image_id": CANDIDATE_IDS,
        "caption_available": valid, "original_caption_source_id": candidate["source_ids"],
        "shuffled_caption_source_id": shuffled["c"]["source_ids"],
        "source_changed": positions != np.arange(len(positions)),
        "text_changed": [a != b for a, b in zip(candidate["texts"], shuffled["c"]["texts"])],
    })
    return shuffled, assignment


def collect_norm_validation_interventions(network, loader, banks, feature_bank, seed):
    if banks["settings"]["name"] != "candidate_only" or banks["settings"]["semantic_selector"]:
        raise ValueError("Norm diagnostics require candidate_only with an image-only selector")
    if banks["settings"]["query_fusion"] or not banks["settings"]["candidate_fusion"]:
        raise ValueError("Norm diagnostics must not enable query text or disable candidate fusion")
    allowed_ids = set(SPLIT_FRAMES["val"].image_id.astype(str))
    shuffled, assignment = make_validation_caption_shuffle(banks, seed)
    modes = ("aligned", "text_off", "text_shuffled")
    logits = {mode: [] for mode in modes}
    rows, seen = [], set()
    old_capture = getattr(network, "capture_fusion_diagnostics", False)
    old_training = network.training
    network.eval()
    network.capture_fusion_diagnostics = True
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Validation text interventions", leave=False):
                ids = [str(image_id) for image_id in batch["image_id"]]
                if len(set(ids)) != len(ids) or set(ids) - allowed_ids or seen.intersection(ids):
                    raise ValueError("Validation diagnostics received duplicate or non-validation query IDs")
                seen.update(ids)
                labels = batch["label"].cpu().tolist()
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    aligned, _, indices, inputs = run_forward(network, batch, banks, feature_bank)
                    query, q_text, q_valid, c_text, c_valid, context = inputs
                    outputs = {"aligned": aligned,
                        "text_off": network.forward_ablated(query, context, q_text, q_valid,
                            c_text, torch.zeros_like(c_valid), banks["settings"]),
                        "text_shuffled": network.forward_ablated(query, context, q_text, q_valid,
                            shuffled["c"]["embeddings"][indices], shuffled["c"]["valid"][indices], banks["settings"]),
                    }
                positions = indices.cpu().tolist()
                for mode, output in outputs.items():
                    values = output["logits"].detach().float().cpu()
                    diagnostics = output["fusion_diagnostics"]
                    if not torch.isfinite(values).all() or any(not torch.isfinite(v).all() for v in diagnostics.values()):
                        raise ValueError("Non-finite logits or fusion diagnostics")
                    logits[mode].append(values)
                    probabilities = values.softmax(1)
                    mode_candidates = shuffled["c"] if mode == "text_shuffled" else banks["c"]
                    for j, image_id in enumerate(ids):
                        position = int(positions[j])
                        active = mode != "text_off" and bool(mode_candidates["valid"][position])
                        prediction = int(probabilities[j].argmax())
                        rows.append({"intervention": mode, "image_id": image_id, "ground_truth": int(labels[j]),
                            "prediction": prediction, "correct": prediction == int(labels[j]),
                            "confidence": float(probabilities[j].max()),
                            "selected_context_position": position, "selected_context_id": CANDIDATE_IDS[position],
                            "candidate_caption_source_id": mode_candidates["source_ids"][position],
                            "candidate_caption": mode_candidates["texts"][position] if active else "",
                            "candidate_caption_available": bool(banks["c"]["valid"][position]),
                            "candidate_caption_valid": active,
                            "query_gate": float(output["query_gate"][j]),
                            "candidate_gate": float(output["candidate_gate"][j]),
                            **{name: float(value[j]) for name, value in diagnostics.items()}})
    finally:
        network.capture_fusion_diagnostics = old_capture
        network.train(old_training)
    if seen != allowed_ids:
        raise ValueError("Validation diagnostics did not cover the exact official validation query IDs")
    predictions = pd.DataFrame(rows)
    by_mode = {mode: predictions.loc[predictions.intervention.eq(mode)].reset_index(drop=True) for mode in modes}
    aligned_rows = by_mode["aligned"]
    aligned_logits = torch.cat(logits["aligned"])
    aligned_prob = aligned_logits.softmax(1)
    summaries, norm_rows = [], []
    norm_names = ["visual_norm", "candidate_raw_projection_norm", "candidate_fusion_norm",
                  "weighted_candidate_norm", "weighted_text_visual_ratio", "candidate_gate"]
    shuffle_testable = int(assignment.caption_available.sum()) >= 2
    for mode in modes:
        frame = by_mode[mode]
        values = torch.cat(logits[mode])
        probabilities = values.softmax(1)
        paired = paired_f1_interval(aligned_rows, frame, BOOTSTRAP_REPEATS, seed)
        summaries.append({"intervention": mode, "split": "val", "query_count": len(frame),
            "shuffle_testable": shuffle_testable,
            **metric_scores(frame.ground_truth, frame.prediction),
            "loss": float(F.cross_entropy(values, torch.tensor(frame.ground_truth.to_numpy(), dtype=torch.long),
                                          label_smoothing=LABEL_SMOOTHING)),
            "candidate_caption_available_rate": float(frame.candidate_caption_available.mean()),
            "candidate_caption_rate": float(frame.candidate_caption_valid.mean()),
            "mean_query_gate": float(frame.query_gate.mean()),
            "mean_candidate_gate": float(frame.candidate_gate.mean()),
            "mean_weighted_text_visual_ratio": float(frame.weighted_text_visual_ratio.mean()),
            "mean_abs_logit_change": float((values - aligned_logits).abs().mean()),
            "mean_abs_probability_change": float((probabilities - aligned_prob).abs().mean()),
            "mean_kl_from_aligned": float((aligned_prob * (
                aligned_prob.clamp_min(1e-8).log() - probabilities.clamp_min(1e-8).log())).sum(1).mean()),
            **paired})
        for subset, selected in (("all", frame), ("caption_available", frame.loc[frame.candidate_caption_available]),
                                 ("active_caption", frame.loc[frame.candidate_caption_valid])):
            for name in norm_names:
                array = selected[name].to_numpy(dtype=float)
                norm_rows.append({"intervention": mode, "subset": subset, "metric": name, "count": len(array),
                    "mean": float(array.mean()) if len(array) else None,
                    "p05": float(np.quantile(array, .05)) if len(array) else None,
                    "median": float(np.median(array)) if len(array) else None,
                    "p95": float(np.quantile(array, .95)) if len(array) else None,
                    "max": float(array.max()) if len(array) else None})
    return pd.DataFrame(summaries), predictions, pd.DataFrame(norm_rows), assignment


def run_norm_validation_interventions(spec, seed):
    if not is_run_completed(spec, seed):
        raise RuntimeError("Finish this notebook's declared training budget before validation interventions")
    run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    completed = read_json(run_dir / "completed.json")
    target = run_dir / "validation_norm_diagnostics"
    metadata_path = target / "metadata.json"
    files = {"summary": target / "validation_interventions.csv",
             "predictions": target / "validation_intervention_predictions.csv",
             "norms": target / "validation_norm_summary.csv",
             "shuffle": target / "validation_caption_shuffle_assignments.csv"}
    metadata = {"schema": "dtd_norm_fusion_validation_v1", "identity": run_identity(spec, seed),
                "best_epoch": completed["best_epoch"], "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
                "shuffle_seed": DIAGNOSTIC_SHUFFLE_SEED, "effective_shuffle_seed": DIAGNOSTIC_SHUFFLE_SEED + seed,
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "norm_epsilon": FUSION_NORM_EPS, "split": "val", "diagnostic_loader_workers": 0,
                "context_policy": "same selected image indices for aligned/off/shuffled within each batch"}
    if metadata_path.is_file():
        if read_json(metadata_path) != metadata:
            raise ValueError("Cached validation diagnostics use another checkpoint/settings; preserve them and use a new output directory")
        if all(path.is_file() for path in files.values()):
            print("Reusing validation diagnostics:", target)
            return pd.read_csv(files["summary"])
    network = bank = banks = loader = None
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["identity"] != run_identity(spec, seed) or state["class_names"] != class_names:
            raise ValueError("Wrong checkpoint identity or class mapping for validation diagnostics")
        if int(state["epoch"]) != int(completed["best_epoch"]):
            raise ValueError("Best checkpoint epoch disagrees with the completed run")
        network = create_ablation_model(seed)
        network.load_state_dict(state["model"], strict=True)
        del state
        bank, banks = refresh_ablation_bank(network), make_ablation_banks(spec, seed)
        loader = make_norm_validation_loader(seed)
        summary, predictions, norms, assignments = collect_norm_validation_interventions(
            network, loader, banks, bank, DIAGNOSTIC_SHUFFLE_SEED + seed)
        summary = summary.assign(experiment=spec["name"], seed=seed, best_epoch=completed["best_epoch"],
                                 study_stage=ARCHITECTURE_STUDY["stage"])
        target.mkdir(parents=True, exist_ok=True)
        for key, frame in (("summary", summary), ("predictions", predictions), ("norms", norms), ("shuffle", assignments)):
            frame.to_csv(files[key], index=False)
        write_json(metadata_path, metadata)
        return summary
    finally:
        del network, bank, banks, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_norm_validation_loader(seed):
    return DataLoader(DTDImageQueryDataset(SPLIT_FRAMES["val"], eval_transform), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda",
                      generator=torch.Generator().manual_seed(seed))
