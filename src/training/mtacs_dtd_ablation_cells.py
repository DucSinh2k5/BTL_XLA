"""Code embedded by build_mtacs_dtd_ablation_notebook.py; no model runs on import."""

# %% utilities
import copy
import gc
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def stable_hash(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_ablation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_scores(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(NUM_CLASSES)),
                                   average="macro", zero_division=0)),
    }


def paired_f1_interval(reference, alternative, repeats=200, seed=123):
    a = reference.set_index("image_id").sort_index()
    b = alternative.set_index("image_id").sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.ground_truth, b.ground_truth):
        raise ValueError("Paired comparison requires the same image IDs and labels")
    labels = a.ground_truth.to_numpy()
    base, changed = a.prediction.to_numpy(), b.prediction.to_numpy()
    delta = metric_scores(labels, changed)["macro_f1"] - metric_scores(labels, base)["macro_f1"]
    result = {"delta_macro_f1": delta, "changed_prediction_rate": float(np.mean(base != changed)),
              "ci95_low": None, "ci95_high": None}
    if repeats:
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        differences = []
        for _ in range(repeats):
            indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            differences.append(metric_scores(labels[indices], changed[indices])["macro_f1"]
                               - metric_scores(labels[indices], base[indices])["macro_f1"])
        result["ci95_low"], result["ci95_high"] = [float(x) for x in np.quantile(differences, [0.025, 0.975])]
    return result


# %% captions
def normalized_text(text):
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(text).replace("_", " ").lower()).strip()


def bool_value(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


DTD_EXTRA_TERMS = {
    "banded": ["bands", "banding"], "blotchy": ["blotches", "blotch"],
    "braided": ["braid", "braids"], "bubbly": ["bubbles", "bubble"],
    "bumpy": ["bumps", "bump"], "chequered": ["checkered", "checkerboard"],
    "cobwebbed": ["cobwebs", "cobweb"], "cracked": ["cracks", "crack"],
    "crosshatched": ["crosshatched", "cross hatching"], "crystalline": ["crystals", "crystal"],
    "dotted": ["dots", "polka dots"], "fibrous": ["fibers", "fibres"],
    "flecked": ["flecks"], "freckled": ["freckles"], "frilly": ["frills"],
    "gauzy": ["gauze"], "gridded": ["grid", "grids"], "grooved": ["grooves"],
    "honeycombed": ["honeycomb"], "interlaced": ["interlacing"],
    "knitted": ["knit", "knitting"], "lacelike": ["lace", "lace like"],
    "lined": ["parallel lines"], "marbled": ["marbling"], "meshed": ["mesh"],
    "paisley": [], "perforated": ["perforations"], "pitted": ["pits"],
    "pleated": ["pleats"], "polka-dotted": ["polka dotted", "polka dots"],
    "porous": ["pores"], "potholed": ["potholes"], "scaly": ["scales"],
    "smeared": ["smears"], "spiralled": ["spiraled", "spirals", "spiral"],
    "sprinkled": ["sprinkles"], "stained": ["stains"], "stratified": ["strata"],
    "striped": ["stripes", "stripe"], "studded": ["studs"],
    "swirly": ["swirls", "swirl"], "veined": ["veins"],
    "waffled": ["waffle"], "woven": ["weave", "weaving"],
    "wrinkled": ["wrinkles"], "zigzagged": ["zigzag", "zig zag"],
}


def caption_term_pattern(names, extra_terms=()):
    import re
    terms = {normalized_text(name) for name in names}
    terms.update(normalized_text(term) for name in names for term in DTD_EXTRA_TERMS.get(name, []))
    terms.update(normalized_text(term) for term in extra_terms)
    terms.discard("")
    ordered = sorted(terms, key=lambda term: (-len(term.split()), -len(term), term))
    if not ordered:
        raise ValueError("No category terms available for the redaction diagnostic")
    expressions = [r"[\W_]+".join(re.escape(word) for word in term.split()) for term in ordered]
    return re.compile(r"(?<!\w)(?:" + "|".join(expressions) + r")(?!\w)", flags=re.IGNORECASE)


def load_ablation_captions(path, split_frames):
    frame = pd.read_csv(path).fillna("")
    required = {"image_id", "final_caption", "caption_valid", "leakage_detected",
                "semantic_leakage_detected", "semantic_leakage_score"}
    if not required.issubset(frame.columns):
        raise ValueError("Caption CSV is missing audit columns: " + str(sorted(required - set(frame.columns))))
    frame["image_id"] = frame.image_id.astype(str)
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate caption image IDs; resolve the input CSV before this experiment")
    expected = pd.concat([part[["image_id"]].assign(split=split) for split, part in split_frames.items()],
                         ignore_index=True)
    if expected.image_id.duplicated().any():
        raise ValueError("Official train/val/test partitions overlap")
    missing = set(expected.image_id) - set(frame.image_id)
    if missing:
        raise ValueError(f"Caption CSV is incomplete: {len(missing)} missing DTD images")
    # Split membership comes from DTD files, so one image-only CSV can be reused across official splits.
    return expected.merge(frame.drop(columns="split", errors="ignore"), on="image_id", validate="one_to_one")


def prepare_caption_views(frame, names, threshold, extra_terms=()):
    import re
    result = frame.copy()
    pattern = caption_term_pattern(names, extra_terms)
    texts = [str(text).strip() for text in result.final_caption]
    normalized = [normalized_text(text) for text in texts]
    exact = {normalized_text(name) for name in names}
    matched = [pattern.findall(text) for text in normalized]
    redacted = [re.sub(r"\s+", " ", pattern.sub(" ", text)).strip() for text in texts]
    word_count = np.array([len(text.split()) for text in normalized])
    eligible = (word_count >= 5) & (word_count <= 8)
    semantic = pd.to_numeric(result.semantic_leakage_score, errors="coerce").to_numpy(dtype=float)
    lexical = np.array([any(f" {term} " in f" {text} " for term in exact) for text in normalized])
    screened = (eligible & result.caption_valid.map(bool_value).to_numpy()
                & ~result.leakage_detected.map(bool_value).to_numpy() & ~lexical
                & ~result.semantic_leakage_detected.map(bool_value).to_numpy()
                & np.isfinite(semantic) & (semantic < threshold))
    # Match raw/redacted coverage; removing all words must not turn a failed caption into a valid one.
    matched_mask = eligible & np.array([bool(normalized_text(text)) for text in redacted])
    views = {
        "screened": {"texts": texts, "valid": screened},
        "unscreened": {"texts": texts, "valid": matched_mask},
        "redacted": {"texts": redacted, "valid": matched_mask.copy()},
        "term_only": {"texts": [" ".join(terms) for terms in matched],
                      "valid": eligible & np.array([bool(terms) for terms in matched])},
    }
    for view in views.values():
        view["image_ids"] = result.image_id.astype(str).tolist()
    result["normalized_caption"] = normalized
    result["redacted_caption"] = redacted
    result["matched_class_terms"] = [" | ".join(terms) for terms in matched]
    result["exact_class_term_hit"] = lexical
    result["alias_or_class_term_hit"] = [bool(terms) for terms in matched]
    result["screened_valid"] = screened
    result["raw_redacted_matched_valid"] = matched_mask
    result["word_count"] = word_count
    return result, views


def deranged_sources(image_ids, valid, split_by_id, candidate_ids, seed):
    rng = np.random.default_rng(seed)
    candidate_ids = set(candidate_ids)
    sources = np.arange(len(image_ids))
    groups = {}
    for position, image_id in enumerate(image_ids):
        if bool(valid[position]):
            # Never borrow a held-out caption for a train image, or a non-pool caption for a pool image.
            key = (split_by_id[image_id], image_id in candidate_ids)
            groups.setdefault(key, []).append(position)
    singletons = []
    for positions in groups.values():
        if len(positions) < 2:
            singletons.extend(image_ids[i] for i in positions)
            continue
        order = rng.permutation(positions)
        sources[order] = np.roll(order, 1)
    return sources, singletons


# %% text_banks
def encode_view_cache(views, clip_path, cache_path):
    fingerprint = stable_hash({"schema": 1, "clip_path": str(clip_path),
        "clip_config": hashlib.sha256((Path(clip_path) / "config.json").read_bytes()).hexdigest(),
        "clip_weights": [(path.name, path.stat().st_size, path.stat().st_mtime_ns)
                         for path in sorted(Path(clip_path).iterdir()) if path.suffix in {".safetensors", ".bin"}],
        "views": {key: {"ids": value["image_ids"], "texts": value["texts"],
                        "valid": value["valid"].tolist()} for key, value in views.items()}})
    if Path(cache_path).is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") == fingerprint:
            for name, view in cached["views"].items():
                if not torch.isfinite(view["embeddings"]).all():
                    raise ValueError(f"Non-finite CLIP cache: {name}")
                if torch.count_nonzero(view["embeddings"][~view["valid"]]):
                    raise ValueError(f"Invalid captions have nonzero vectors: {name}")
            return cached
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    encoder = CLIPTextModelWithProjection.from_pretrained(str(clip_path), local_files_only=True).to(DEVICE).eval()
    all_texts = sorted({text for view in views.values() for text, valid in zip(view["texts"], view["valid"]) if valid})
    vectors = {}
    try:
        with torch.inference_mode():
            for offset in tqdm(range(0, len(all_texts), 128), desc="CLIP caption views"):
                texts = all_texts[offset:offset + 128]
                tokens = tokenizer(texts, padding=True, truncation=True,
                    max_length=encoder.config.max_position_embeddings, return_tensors="pt").to(DEVICE)
                embeddings = encoder(**tokens).text_embeds.float().cpu()
                vectors.update(zip(texts, embeddings))
        dimension = int(encoder.config.projection_dim)
        cached_views = {}
        for name, view in views.items():
            matrix = torch.zeros((len(view["image_ids"]), dimension), dtype=torch.float32)
            for i, (text, valid) in enumerate(zip(view["texts"], view["valid"])):
                if valid:
                    matrix[i] = vectors[text]
            cached_views[name] = {**view, "valid": torch.as_tensor(view["valid"], dtype=torch.bool),
                                  "embeddings": matrix}
        cached = {"fingerprint": fingerprint, "embedding_dim": dimension, "views": cached_views}
        temporary = Path(cache_path).with_suffix(".tmp")
        torch.save(cached, temporary)
        temporary.replace(cache_path)
        return cached
    finally:
        del encoder, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_ablation_banks(spec, seed, override=None):
    settings = {**spec, **(override or {})}
    view = TEXT_CACHE["views"][settings["view"]]
    mask_view = TEXT_CACHE["views"][settings.get("mask_view", settings["view"])]
    ids = view["image_ids"]
    if mask_view["image_ids"] != ids:
        raise ValueError("Counterfactual masks and caption views must have identical image ordering")
    lookup = {image_id: position for position, image_id in enumerate(ids)}
    candidate_positions = torch.tensor([lookup[image_id] for image_id in CANDIDATE_IDS], dtype=torch.long)
    banks = {"settings": settings, "q_index": lookup, "singletons": {}}
    for role, offset in (("q", 1009), ("c", 2017)):
        indices = np.arange(len(ids))
        if settings.get(role + "_transform", "aligned") == "shuffle":
            indices, singletons = deranged_sources(ids, mask_view["valid"], SPLIT_BY_ID, CANDIDATE_IDS, seed + offset)
            banks["singletons"][role] = singletons
        matrix = view["embeddings"][torch.as_tensor(indices)].clone()
        valid = mask_view["valid"].clone()  # Preserve the recipient's exact availability pattern.
        if settings.get(role + "_transform") == "zero":
            matrix.zero_()
            valid.zero_()
        if role == "c":
            matrix = matrix[candidate_positions].to(DEVICE)
            valid = valid[candidate_positions].to(DEVICE)
            indices = indices[candidate_positions.numpy()]
        banks[role] = {"embeddings": matrix, "valid": valid,
                      "source_ids": [ids[int(i)] for i in indices],
                      "texts": [view["texts"][int(i)] for i in indices]}
    return banks


def batch_caption_inputs(banks, image_ids):
    positions = torch.tensor([banks["q_index"][str(image_id)] for image_id in image_ids], dtype=torch.long)
    return (banks["q"]["embeddings"][positions].to(DEVICE, non_blocking=True),
            banks["q"]["valid"][positions].to(DEVICE, non_blocking=True))


# %% model
def make_ablation_model_class(base_class, visual_class):
    class DTDAblationModel(base_class):
        """All configurations share this class, parameter count, and classifier head."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context_text_projection = copy.deepcopy(self.text_projection)
            self.context_text_gate = copy.deepcopy(self.text_gate)
            self.selector.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))

        def select_context(self, query_images, feature_bank, q_text, q_valid, banks,
                           excluded_positions, semantic=False, sample=False):
            scores, _ = self.score_full_candidate_pool(query_images, feature_bank)
            scores = scores.float()
            scores = (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
            if semantic:
                text_scores = F.normalize(q_text.float(), dim=1) @ F.normalize(banks["c"]["embeddings"].float(), dim=1).T
                valid = q_valid[:, None] & banks["c"]["valid"][None, :]
                count = valid.sum(1, keepdim=True).clamp_min(1)
                mean = (text_scores * valid).sum(1, keepdim=True) / count
                variance = ((text_scores - mean).square() * valid).sum(1, keepdim=True) / count
                text_scores = (text_scores - mean) / variance.sqrt().clamp_min(1e-6) * valid
                scores = scores + torch.sigmoid(self.selector.semantic_mix_logit) * text_scores
            positions = excluded_positions.to(scores.device)
            in_pool = positions >= 0
            if in_pool.any():
                scores = scores.clone()
                scores[torch.arange(len(scores), device=scores.device)[in_pool], positions[in_pool]] = -torch.inf
            if not torch.isfinite(scores).any(1).all():
                raise ValueError("No candidate remains after self-exclusion")
            distribution = torch.distributions.Categorical(logits=scores)
            indices = distribution.sample() if sample else scores.argmax(1)
            return {"indices": indices, "log_prob": distribution.log_prob(indices),
                    "entropy": distribution.entropy().mean()}

        def forward_ablated(self, query_images, context_images, q_text, q_valid, c_text, c_valid, settings):
            if context_images is None:
                visual = self.query_norm(self._pool_backbone_features(query_images))
            else:
                _, visual, _ = visual_class.forward_selected_context(self, query_images, context_images, return_features=True)
            q_features = self.text_projection(q_text.float())
            c_features = self.context_text_projection(c_text.float())
            q_gate = torch.sigmoid(self.text_gate(torch.cat([visual.float(), q_features], 1))).squeeze(1)
            c_gate = torch.sigmoid(self.context_text_gate(torch.cat([visual.float(), c_features], 1))).squeeze(1)
            q_gate = q_gate * q_valid.to(q_gate.dtype) * bool(settings["query_fusion"])
            c_gate = c_gate * c_valid.to(c_gate.dtype) * bool(settings["candidate_fusion"])
            fused = self.mtacs_norm(visual.float() + q_gate[:, None] * q_features + c_gate[:, None] * c_features)
            return {"logits": self.mtacs_classifier(fused), "features": F.normalize(fused, dim=1),
                    "query_gate": q_gate, "candidate_gate": c_gate}
    return DTDAblationModel


class DTDImageQueryDataset(Dataset):
    def __init__(self, annotations, transform):
        self.rows = annotations.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        from PIL import Image
        row = self.rows.iloc[position]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
        return {"query_image": self.transform(image), "label": int(row.class_id), "image_id": str(row.image_id)}


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_ablation_loaders(seed):
    common = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
              "pin_memory": DEVICE.type == "cuda", "worker_init_fn": seed_worker}
    return {
        split: DataLoader(DTDImageQueryDataset(frame, train_transform if split == "train" else eval_transform),
                          shuffle=split == "train", generator=torch.Generator().manual_seed(seed), **common)
        for split, frame in SPLIT_FRAMES.items()
    }


def candidate_images_at(indices):
    return candidate_image_bank.index_select(0, indices.to(candidate_image_bank.device)).to(DEVICE, non_blocking=True)


def run_forward(network, batch, banks, feature_bank, training=False, fixed_positions=None):
    settings = banks["settings"]
    query = batch["query_image"].to(DEVICE, non_blocking=True)
    q_text, q_valid = batch_caption_inputs(banks, batch["image_id"])
    excluded = torch.tensor([CANDIDATE_POSITION.get(str(i), -1) for i in batch["image_id"]], device=DEVICE)
    if fixed_positions is None:
        details = network.select_context(query, feature_bank, q_text, q_valid, banks, excluded,
                                         semantic=settings["semantic_selector"], sample=training)
        indices = details["indices"]
    else:
        indices = torch.tensor([fixed_positions[str(i)] for i in batch["image_id"]], device=DEVICE)
        if ((indices < 0) | (indices >= len(CANDIDATE_IDS))).any() or (indices == excluded).any():
            raise ValueError("Invalid fixed candidate positions")
        details = None
    context = candidate_images_at(indices)
    c_text, c_valid = banks["c"]["embeddings"][indices], banks["c"]["valid"][indices]
    output = network.forward_ablated(query, context, q_text, q_valid, c_text, c_valid, settings)
    return output, details, indices, (query, q_text, q_valid, c_text, c_valid, context)


# %% training
def ablation_supcon(features, labels):
    logits = F.normalize(features.float(), dim=1) @ F.normalize(features.float(), dim=1).T / SUPCON_TEMPERATURE
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    valid = positive.sum(1) > 0
    if not valid.any():
        return features.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), 1, keepdim=True)
    mean = log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1)
    return -mean[valid].mean()


def train_ablation_epoch(network, loader, banks, feature_bank, optimizer, scaler, epoch):
    network.train()
    for parameter in network.backbone.parameters():
        parameter.requires_grad_(epoch > FREEZE_BACKBONE_EPOCHS)
    labels_all, predictions = [], []
    totals = {key: 0.0 for key in ("loss", "reward", "policy_loss", "query_gate", "candidate_gate")}
    count = 0
    for batch in tqdm(loader, desc=f"train epoch {epoch}", leave=False):
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            output, details, _, inputs = run_forward(network, batch, banks, feature_bank, training=True)
            ce = F.cross_entropy(output["logits"].float(), labels, label_smoothing=LABEL_SMOOTHING)
            supcon = ablation_supcon(output["features"], labels)
            query, q_text, q_valid, c_text, c_valid, context = inputs
            # Evaluate both reward branches without dropout; candidate image/text vary together.
            network.eval()
            with torch.no_grad():
                baseline = network.forward_ablated(query, None, q_text, q_valid, c_text,
                                                  torch.zeros_like(c_valid), banks["settings"])["logits"]
                contextual = network.forward_ablated(query, context, q_text, q_valid, c_text,
                                                    c_valid, banks["settings"])["logits"]
                rewards = (F.cross_entropy(baseline.float(), labels, reduction="none")
                           - F.cross_entropy(contextual.float(), labels, reduction="none"))
                advantages = ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
                              if len(rewards) > 1 else rewards)
            network.train()
            policy = -(details["log_prob"] * advantages.detach()).mean()
            loss = ce + SUPCON_WEIGHT * supcon + TACS_POLICY_WEIGHT * policy - TACS_ENTROPY_WEIGHT * details["entropy"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; stop rather than write an invalid checkpoint")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        n = len(labels)
        count += n
        labels_all.extend(labels.cpu().tolist())
        predictions.extend(output["logits"].detach().argmax(1).cpu().tolist())
        for key, value in (("loss", loss), ("policy_loss", policy), ("reward", rewards.mean()),
                           ("query_gate", output["query_gate"].mean()),
                           ("candidate_gate", output["candidate_gate"].mean())):
            totals[key] += float(value.detach()) * n
    return {**metric_scores(labels_all, predictions), **{key: value / count for key, value in totals.items()}}


@torch.inference_mode()
def evaluate_ablation(network, loader, banks, feature_bank, fixed_positions=None):
    network.eval()
    rows, probabilities = [], []
    loss_sum = 0.0
    for batch in tqdm(loader, desc="evaluate", leave=False):
        labels = batch["label"].to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            output, _, indices, inputs = run_forward(network, batch, banks, feature_bank, fixed_positions=fixed_positions)
        probs = output["logits"].float().softmax(1).cpu()
        probabilities.append(probs)
        loss_sum += float(F.cross_entropy(output["logits"].float(), labels, reduction="sum",
                                          label_smoothing=LABEL_SMOOTHING))
        for j, image_id in enumerate(batch["image_id"]):
            q_position, c_position = banks["q_index"][str(image_id)], int(indices[j])
            prediction, truth = int(probs[j].argmax()), int(labels[j])
            rows.append({"image_id": str(image_id), "ground_truth": truth, "prediction": prediction,
                "correct": prediction == truth, "confidence": float(probs[j].max()),
                "selected_context_position": c_position, "selected_context_id": CANDIDATE_IDS[c_position],
                "query_caption_source_id": banks["q"]["source_ids"][q_position],
                "candidate_caption_source_id": banks["c"]["source_ids"][c_position],
                "query_caption": banks["q"]["texts"][q_position],
                "candidate_caption": banks["c"]["texts"][c_position],
                "query_caption_valid": bool(inputs[2][j]), "candidate_caption_valid": bool(inputs[4][j]),
                "query_gate": float(output["query_gate"][j]), "candidate_gate": float(output["candidate_gate"][j])})
    frame = pd.DataFrame(rows)
    metrics = {**metric_scores(frame.ground_truth, frame.prediction), "loss": loss_sum / len(frame),
        "query_caption_rate": float(frame.query_caption_valid.mean()),
        "candidate_caption_rate": float(frame.candidate_caption_valid.mean()),
        "mean_query_gate": float(frame.query_gate.mean()), "mean_candidate_gate": float(frame.candidate_gate.mean())}
    return metrics, frame, torch.cat(probabilities)


def create_ablation_model(seed):
    seed_ablation(seed)
    return AblationModel(DINOV3_REPO_DIR, DINOV3_MODEL_NAME, resolve_dinov3_weights(), NUM_CLASSES,
                         FEATURE_DIM, allow_feature_fallback=False, clip_dim=CLIP_EMBED_DIM).to(DEVICE)


def optimizer_for(network):
    groups = []
    for prefix, learning_rate, decay in (("backbone", BACKBONE_LR, BACKBONE_WEIGHT_DECAY),
                                          ("selector", TACS_SELECTOR_LR, TACS_SELECTOR_WEIGHT_DECAY),
                                          ("heads", HEAD_LR, HEAD_WEIGHT_DECAY)):
        parameters = [p for name, p in network.named_parameters()
                      if (name.startswith(prefix + ".") if prefix != "heads"
                          else not name.startswith(("backbone.", "selector.")))]
        groups.append({"params": parameters, "lr": learning_rate, "weight_decay": decay})
    return torch.optim.AdamW(groups)


def refresh_ablation_bank(network):
    return refresh_candidate_feature_bank(network, candidate_image_bank).clone().to(DEVICE)


def run_identity(spec, seed):
    return {"protocol": PROTOCOL_FINGERPRINT, "spec": spec, "training_seed": seed}


def checkpoint_cpu(network, optimizer, scheduler, scaler, history, best_loss, best_epoch, bad_epochs, identity, loaders):
    return {"identity": identity, "epoch": history[-1]["epoch"], "history": history,
            "model": {key: value.detach().cpu() for key, value in network.state_dict().items()},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_loss": best_loss, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_loader_rng": loaders["train"].generator.get_state()}


def atomic_torch_save(value, path):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_one_configuration(spec, seed):
    run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(spec, seed)
    config_path, finished = run_dir / "config.json", run_dir / "completed.json"
    if config_path.is_file() and read_json(config_path) != identity:
        raise ValueError(f"Changed experiment at {run_dir}; choose another output directory")
    write_json(config_path, identity)
    if finished.is_file():
        print("Already trained:", spec["name"], seed)
        return run_dir
    network = create_ablation_model(seed)
    loaders, banks = make_ablation_loaders(seed), make_ablation_banks(spec, seed)
    write_json(run_dir / "caption_assignment_manifest.json", {
        "view": spec["view"], "singletons_not_deranged": banks["singletons"],
        "query_transform": spec["q_transform"], "candidate_transform": spec["c_transform"],
        "query_fusion": spec["query_fusion"], "candidate_fusion": spec["candidate_fusion"],
        "semantic_selector": spec["semantic_selector"],
    })
    assignments = []
    for role, ids in (("q", TEXT_CACHE["views"][spec["view"]]["image_ids"]), ("c", CANDIDATE_IDS)):
        assignments.extend({"role": role, "image_id": image_id, "caption_source_id": source_id,
                            "caption_available": bool(valid)}
                           for image_id, source_id, valid in zip(ids, banks[role]["source_ids"], banks[role]["valid"].cpu()))
    pd.DataFrame(assignments).to_csv(run_dir / "caption_source_assignments.csv", index=False)
    optimizer = optimizer_for(network)
    def learning_rate_factor(epoch):
        if epoch < WARMUP_EPOCHS:
            return 0.2 + 0.8 * epoch / max(WARMUP_EPOCHS, 1)
        return max(0.001, 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS)
                                             / max(EPOCHS - WARMUP_EPOCHS, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history, best_loss, best_epoch, bad_epochs, start_epoch = [], float("inf"), 0, 0, 1
    if RESUME_ABLATIONS and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint belongs to another experiment")
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_loss, best_epoch, bad_epochs = state["history"], state["best_loss"], state["best_epoch"], state["bad_epochs"]
        start_epoch = state["epoch"] + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        loaders["train"].generator.set_state(state["train_loader_rng"])
        del state
    elif last_path.is_file() or best_path.is_file():
        raise ValueError(f"Unfinished existing run: {run_dir}; enable RESUME_ABLATIONS or choose a new directory")
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            if bad_epochs >= PATIENCE:
                break
            bank = refresh_ablation_bank(network)
            training = train_ablation_epoch(network, loaders["train"], banks, bank, optimizer, scaler, epoch)
            del bank
            # The backbone changed during training; validation must use its current candidate features.
            bank = refresh_ablation_bank(network)
            validation, _, _ = evaluate_ablation(network, loaders["val"], banks, bank)
            del bank
            scheduler.step()
            improved = validation["loss"] < best_loss - 1e-6
            if improved:
                best_loss, best_epoch, bad_epochs = validation["loss"], epoch, 0
            else:
                bad_epochs += 1
            history.append({"epoch": epoch, **{"train_" + key: value for key, value in training.items()},
                            **{"val_" + key: value for key, value in validation.items()}})
            if improved:
                atomic_torch_save({"identity": identity, "epoch": epoch, "validation": validation,
                    "class_names": class_names,
                    "model": {key: value.detach().cpu() for key, value in network.state_dict().items()}}, best_path)
            atomic_torch_save(checkpoint_cpu(network, optimizer, scheduler, scaler, history, best_loss,
                                            best_epoch, bad_epochs, identity, loaders), last_path)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            print(f'{spec["name"]} seed={seed} epoch={epoch} val_f1={validation["macro_f1"]:.4f} '
                  f'val_loss={validation["loss"]:.4f} q_gate={validation["mean_query_gate"]:.3f} '
                  f'c_gate={validation["mean_candidate_gate"]:.3f}')
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        network.load_state_dict(best["model"])
        del best
        bank = refresh_ablation_bank(network)
        validation, predictions, _ = evaluate_ablation(network, loaders["val"], banks, bank)
        predictions.to_csv(run_dir / "val_predictions.csv", index=False)
        write_json(run_dir / "val_metrics.json", validation)
        write_json(finished, {"identity": identity, "best_epoch": best_epoch, "best_val_loss": best_loss})
        if not KEEP_LAST_CHECKPOINT and last_path.is_file():
            last_path.unlink()  # Only this run's redundant resume file; keep its best weights.
        return run_dir
    finally:
        del network, optimizer, scheduler, scaler, loaders, banks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %% diagnostics
def caption_label_diagnostics(split="val"):
    if split == "test" and not (OUT_PUT_DIR / "locked_test_plan.json").is_file():
        raise ValueError("Lock the complete evaluation plan before computing test label-word diagnostics")
    frame = SPLIT_FRAMES[split][["image_id", "class_id"]].merge(caption_audit, on="image_id", validate="one_to_one")
    exact_hits, alias_hits = [], []
    for row in frame.itertuples(index=False):
        label = class_names[int(row.class_id)]
        exact_hits.append(f" {normalized_text(label)} " in f" {row.normalized_caption} ")
        alias_hits.append(bool(caption_term_pattern([label]).search(row.normalized_caption)))
    frame["true_class_name_in_caption"] = exact_hits
    frame["true_class_name_or_alias_in_caption"] = alias_hits
    columns = ["class_id", "image_id", "true_class_name_in_caption", "true_class_name_or_alias_in_caption", "screened_valid"]
    frame[columns].to_csv(OUT_PUT_DIR / f"caption_label_diagnostics_{split}.csv", index=False)
    result = {"split": split, "n": len(frame), "true_class_name_rate": float(np.mean(exact_hits)),
              "true_class_or_alias_rate": float(np.mean(alias_hits))}
    write_json(OUT_PUT_DIR / f"caption_label_summary_{split}.json", result)
    return result


def caption_probes():
    rows = []
    view = TEXT_CACHE["views"]["unscreened"]
    position = {image_id: i for i, image_id in enumerate(view["image_ids"])}
    train_pos = [position[str(i)] for i in train_df.image_id]
    val_pos = [position[str(i)] for i in val_df.image_id]
    mask_features = np.column_stack([TEXT_CACHE["views"][key]["valid"].numpy().astype(float)
                                    for key in ("screened", "unscreened", "term_only")])
    matrices = {key: item["embeddings"].numpy() for key, item in TEXT_CACHE["views"].items()}
    matrices["availability_mask_only"] = mask_features
    for name, matrix in matrices.items():
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
        probe.fit(matrix[train_pos], train_df.class_id.to_numpy())
        prediction = probe.predict(matrix[val_pos])
        metrics = metric_scores(val_df.class_id.to_numpy(), prediction)
        rows.append({"probe": name, **metrics, "split": "val", "n": len(val_pos)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_PUT_DIR / "caption_probes_val.csv", index=False)
    return result


def validation_interventions(spec, seed):
    run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
    network = create_ablation_model(seed)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if best["identity"] != run_identity(spec, seed):
        raise ValueError("Wrong checkpoint for validation interventions")
    network.load_state_dict(best["model"])
    del best
    loader = make_ablation_loaders(seed)["val"]
    bank = refresh_ablation_bank(network)
    base_banks = make_ablation_banks(spec, seed)
    _, base, _ = evaluate_ablation(network, loader, base_banks, bank)
    fixed = dict(zip(base.image_id, base.selected_context_position))
    interventions = {
        "zero_query_fixed_context": {"q_transform": "zero"},
        "zero_candidate_fixed_context": {"c_transform": "zero"},
        "zero_both_fixed_context": {"q_transform": "zero", "c_transform": "zero"},
        "shuffle_query_fixed_context": {"q_transform": "shuffle"},
        "shuffle_candidate_fixed_context": {"c_transform": "shuffle"},
        "shuffle_both_fixed_context": {"q_transform": "shuffle", "c_transform": "shuffle"},
        "redact_both_fixed_context": {"view": "redacted", "mask_view": spec["view"]},
        "zero_both_reselect": {"q_transform": "zero", "c_transform": "zero"},
        "shuffle_both_reselect": {"q_transform": "shuffle", "c_transform": "shuffle"},
    }
    rows = []
    try:
        for name, override in interventions.items():
            banks = make_ablation_banks(spec, seed, override)
            metrics, predictions, _ = evaluate_ablation(network, loader, banks, bank,
                                                        fixed_positions=fixed if "fixed_context" in name else None)
            paired = paired_f1_interval(base, predictions, BOOTSTRAP_REPEATS, seed)
            changed_context = float(np.mean(base.selected_context_id.to_numpy() != predictions.selected_context_id.to_numpy()))
            rows.append({"experiment": spec["name"], "seed": seed, "intervention": name,
                         **metrics, **paired, "changed_context_rate": changed_context})
            predictions.to_csv(run_dir / f"val_{name}.csv", index=False)
        result = pd.DataFrame(rows)
        result.to_csv(run_dir / "val_interventions.csv", index=False)
        return result
    finally:
        del network, bank, loader, base_banks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_locked_test():
    plan = {"protocol": PROTOCOL_FINGERPRINT, "runs": []}
    for spec in SELECTED_ABLATIONS:
        for seed in TRAIN_SEEDS:
            run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
            if not (run_dir / "completed.json").is_file():
                raise RuntimeError("Complete every declared training run before opening test")
            completed = read_json(run_dir / "completed.json")
            if completed["identity"] != run_identity(spec, seed):
                raise ValueError("Completed run no longer matches the locked protocol")
            plan["runs"].append({"experiment": spec["name"], "seed": seed,
                                 "best_epoch": completed["best_epoch"]})
    lock = OUT_PUT_DIR / "locked_test_plan.json"
    if lock.is_file() and read_json(lock) != plan:
        raise ValueError("Test plan changed after evaluation; use a separate experiment directory")
    write_json(lock, plan)
    caption_label_diagnostics("test")
    rows = []
    for spec in SELECTED_ABLATIONS:
        for seed in TRAIN_SEEDS:
            run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
            target = run_dir / "test_metrics.json"
            if target.is_file():
                rows.append({"experiment": spec["name"], "seed": seed, **read_json(target)})
                continue
            network = create_ablation_model(seed)
            state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            if state["identity"] != run_identity(spec, seed):
                raise ValueError("Wrong best checkpoint for test")
            network.load_state_dict(state["model"])
            del state
            bank = loader = banks = None
            try:
                bank = refresh_ablation_bank(network)
                loader, banks = make_ablation_loaders(seed)["test"], make_ablation_banks(spec, seed)
                metrics, predictions, _ = evaluate_ablation(network, loader, banks, bank)
                predictions.to_csv(run_dir / "test_predictions.csv", index=False)
                write_json(target, metrics)
                rows.append({"experiment": spec["name"], "seed": seed, **metrics})
            finally:
                del network, bank, loader, banks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    result.to_csv(OUT_PUT_DIR / "ablation_test_results.csv", index=False)
    return result


def summarize_ablation(split="val"):
    rows, paired = [], []
    for spec in SELECTED_ABLATIONS:
        for seed in TRAIN_SEEDS:
            run_dir = RUNS_DIR / spec["name"] / f"seed_{seed}"
            path = run_dir / f"{split}_metrics.json"
            if not path.is_file():
                continue
            rows.append({"experiment": spec["name"], "seed": seed, **read_json(path)})
            reference = RUNS_DIR / "visual_only" / f"seed_{seed}" / f"{split}_predictions.csv"
            alternative = run_dir / f"{split}_predictions.csv"
            if reference.is_file() and alternative.is_file():
                paired.append({"experiment": spec["name"], "seed": seed,
                    **paired_f1_interval(pd.read_csv(reference), pd.read_csv(alternative), BOOTSTRAP_REPEATS, seed)})
    results = pd.DataFrame(rows)
    results.to_csv(OUT_PUT_DIR / f"ablation_{split}_results.csv", index=False)
    if not results.empty:
        aggregate = results.groupby("experiment").agg(
            runs=("seed", "count"), accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
        aggregate.to_csv(OUT_PUT_DIR / f"ablation_{split}_mean_std.csv")
    pd.DataFrame(paired).to_csv(OUT_PUT_DIR / f"ablation_{split}_paired_vs_visual.csv", index=False)
    return results
