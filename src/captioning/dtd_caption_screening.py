"""Explicit caption-input policies embedded in the standalone DTD ablations."""


def clean_dtd_caption(text):
    import re
    text = str(text).strip()
    for _ in range(3):
        before = text
        text = re.sub(r"^(?:(?:the|this)\s+)?(?:image|picture|photo(?:graph)?)\s+"
                      r"(?:is|shows|depicts|features|contains)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:(?:a|an|the)\s+)?close[ -]up(?:\s+view)?\s+(?:of|showing)\s+",
                      "", text, flags=re.I)
        text = re.sub(r"^(?:a|an|the)(?:\s+|$)", "", text, flags=re.I)
        if text == before:
            break
    words = re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?", text)
    selected = []
    normalized_count = 0
    for word in words:
        count = len(re.findall(r"[A-Za-z0-9]+", word))
        if normalized_count + count > 8:
            break
        selected.append(word)
        normalized_count += count
    return " ".join(selected)


def prepare_standalone_caption_views(frame, names, threshold, policy):
    if policy not in {"legacy_strict", "dtd_lexical_v2"}:
        raise ValueError("Caption policy must be legacy_strict or dtd_lexical_v2")
    source = frame.copy()
    if policy == "legacy_strict":
        result, views = prepare_caption_views(source, names, threshold)
        result["semantic_score_matches_input"] = True
    else:
        original_text = source.final_caption.astype(str)
        original_valid = source.caption_valid.map(bool_value).to_numpy()
        original_semantic = source.semantic_leakage_detected.map(bool_value).to_numpy()
        original_scores = pd.to_numeric(source.semantic_leakage_score, errors="coerce").to_numpy(dtype=float)
        raw = source.get("raw_caption", original_text).fillna("").astype(str)
        raw = raw.where(raw.str.strip().ne(""), original_text)
        source["final_caption"] = raw.map(clean_dtd_caption)
        result, views = prepare_caption_views(source, names, threshold)
        # Recover legacy semantic-only rejects, not unexplained failures/invalid source captions.
        recoverable = original_valid | (original_semantic & np.isfinite(original_scores))
        eligible = result.word_count.between(5, 8).to_numpy()
        screened = eligible & ~result.exact_class_term_hit.to_numpy() & recoverable
        result["legacy_final_caption"] = original_text
        result["legacy_caption_valid"] = original_valid
        result["legacy_semantic_gate_recovered"] = screened & ~original_valid
        result["semantic_score_matches_input"] = source.final_caption.eq(original_text).to_numpy()
        result["screened_valid"] = screened
        views["screened"]["valid"] = screened
    result["input_caption_policy"] = policy
    result["semantic_hard_gate"] = policy == "legacy_strict"
    return result, views


def caption_filter_diagnostics(audit, threshold, output_dir, prefix="caption"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = audit.loc[audit.split.eq("train")].copy()
    scores = pd.to_numeric(train.semantic_leakage_score, errors="coerce")
    rows = {
        "caption_count": len(train),
        "word_count_outside_5_8": int((~train.word_count.between(5, 8)).sum()),
        "direct_class_term_hit": int(train.exact_class_term_hit.sum()),
        "csv_caption_valid_false": int((~train.caption_valid.map(bool_value)).sum()),
        "csv_leakage_flag": int(train.leakage_detected.map(bool_value).sum()),
        "csv_semantic_flag": int(train.semantic_leakage_detected.map(bool_value).sum()),
        "semantic_score_missing_or_nonfinite": int((~np.isfinite(scores)).sum()),
        "semantic_score_at_or_above_threshold": int((scores >= threshold).sum()),
        "usable_model_captions": int(train.screened_valid.sum()),
        "unique_model_captions": int(train.final_caption.nunique()),
    }
    counts = pd.DataFrame([{"condition": key, "count": value} for key, value in rows.items()])
    counts.to_csv(output_dir / f"{prefix}_filter_diagnostics_train.csv", index=False)
    scores.describe(percentiles=[.05, .25, .50, .75, .95]).to_csv(
        output_dir / f"{prefix}_semantic_scores_train.csv", header=["score"])
    columns = [name for name in ("image_id", "raw_caption", "legacy_final_caption", "final_caption",
               "word_count", "exact_class_term_hit", "semantic_leakage_score", "semantic_score_matches_input",
               "screened_valid", "input_caption_policy") if name in train]
    samples = train[columns].head(12)
    samples.to_csv(output_dir / f"{prefix}_samples_train.csv", index=False)
    print("Training-only caption diagnostics; rejection counts overlap:")
    display(counts)
    print("Stored CLIP score range:", scores.min(), "..", scores.max(), "| median:", scores.median())
    if "semantic_score_matches_input" in train and not train.semantic_score_matches_input.all():
        print("Stored CLIP scores refer to the original CSV text where semantic_score_matches_input=False; "
              "they are legacy diagnostics, not scores for the cleaned model input.")
    if not train.screened_valid.any():
        display(samples)
        print("No usable training captions. Do not force masks/gates on or tune thresholds using test metrics.")
    return counts
