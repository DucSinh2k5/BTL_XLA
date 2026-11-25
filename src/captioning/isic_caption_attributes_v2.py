"""Extract short visible-attribute spans from Florence responses, without labels.

This is deterministic post-processing, not a trained captioner or ground truth.
    Every output word comes from one cleaned source sentence. Missing attributes
produce an empty/masked caption instead of an invented description.
"""
from __future__ import annotations


def attribute_caption_details(response):
    import re
    colors = set("brown black pink pinkish red reddish purple purplish blue bluish yellow yellowish white whitish beige tan gray grey orange multicolored".split())
    shapes = set("round rounded circular oval irregular irregularly shaped asymmetric asymmetrical elongated angular curved patch patches spot spots circle circles blob blobs".split())
    boundaries = set("border borders edge edges outline outlines boundary boundaries rim perimeter jagged blurred defined sharp distinct".split())
    textures = set("rough smooth flaky dry raised flat wrinkled granular speckled dotted dots speckles lines network reticular scaly cracked bumpy fuzzy hairy hairs streak streaks".split())
    sizes = set("small tiny large little thin thick narrow wide central center centre scattered clustered surrounding dark darker light lighter pale uneven".split())
    stop = set("a an the of to in on at from for with and or but is are was were be been being that which this these those it its their there as than by has have appears appear seems seem looks look very slightly covered larger smaller".split())
    filler = re.compile(r"\b(?:person['’]s|person|image|photo|picture|specifically|appears|seems|depicts|shows|petri|dish|container|bowl|substance|object)\b", re.I)
    candidates = []
    seen = set()
    for sentence in re.split(r"[.!?;\n]+", str(response)):
        sentence = sentence.strip().lower()
        sentence = re.sub(r"^(?:in|at)\s+(?:the\s+)?cent(?:er|re)\s+of\s+(?:the\s+)?(?:skin|image|photo|picture)\s*,?\s*(?:there\s+(?:is|are)\s+)?", "central ", sentence)
        sentence = re.sub(r"^(?:(?:the|this)\s+)?(?:image|photo|picture)\s+(?:shows|is|depicts|features)\s+", "", sentence)
        sentence = re.sub(r"^(?:a\s+)?close[ -]up(?:\s+view)?\s+of\s+", "", sentence)
        sentence = re.sub(r"^(?:the\s+)?skin\s+(?:appears|seems)\s+to\s+be\s+", "", sentence)
        sentence = re.sub(r"^(?:the\s+)?(?:substance|surface|area)\s+is\s+", "", sentence)
        sentence = re.sub(r"\b(?:a|an|the)\b", "", sentence)
        sentence = re.sub(r"\bin (?:the )?colou?r\b", "", sentence)
        sentence = re.sub(r"\b(?:that|which) (?:is|are)\b", "", sentence)
        sentence = re.sub(r"\band (?:has|have)(?: a)?\b", "and", sentence)
        words = re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", sentence)
        # A compact noun phrase may move attributes before their own source
        # noun. Never borrow a colour/texture from another sentence/referent.
        noun = next((w for w in words if w in {"area", "patch", "spot", "circle", "surface"}), None)
        if noun:
            noun_colors = [w for w in words if w in colors]
            shape_adjectives = [w for w in words if w in {"round", "rounded", "circular", "oval", "irregular", "asymmetric", "elongated"}]
            size = next((w for w in words if w in {"small", "tiny", "large", "thin", "thick"}), None)
            tone = next((w for w in words if w in {"dark", "light", "pale"}), None)
            composed = [w for w in [size, "central" if "central" in words else None, tone,
                        noun_colors[0] if noun_colors else None,
                        shape_adjectives[0] if shape_adjectives else None, noun] if w]
            unambiguous = (len(set(noun_colors)) == 1
                and sum(w in {"area", "patch", "spot", "circle", "surface"} for w in words) == 1
                and not set(words) & {"background", "next", "beside", "near", "surrounding", "around", "than", "while"})
            if unambiguous and len(composed) == len(set(composed)) and 5 <= len(composed) <= 8 and noun_colors and (shape_adjectives or noun in shapes):
                caption = " ".join(composed)
                tokens = set(composed)
                groups = [tokens & g for g in (colors, shapes, boundaries, textures, sizes)]
                info = set.union(*groups)
                score = (3 * len(groups[0]) + 6 * len(groups[1]) + 5 * len(groups[2]) + 2 * len(groups[3])
                         + .5 * len(groups[4]) + 2 * len(info) / len(composed) + 1.)
                if caption not in seen:
                    candidates.append({"caption": caption, "score": round(score, 8), "attributes": sorted(info)})
                    seen.add(caption)
        for start in range(len(words)):
            if words[start] in stop:
                continue
            for width in range(5, 9):
                selected = words[start:start + width]
                if len(selected) != width or selected[-1] in stop:
                    continue
                caption = " ".join(selected)
                if re.search(r"\b(?:there is|there are|it is|appears to|skin there)\b", caption):
                    continue
                if caption in seen or not 5 <= len(re.findall(r"[A-Za-z0-9]+", caption)) <= 8 or filler.search(caption):
                    continue
                if selected[-1] in colors | sizes | textures:
                    continue  # Do not cut immediately before the described noun.
                tokens = set(selected)
                groups = [tokens & group for group in (colors, shapes, boundaries, textures, sizes)]
                # Generic skin/texture-only fragments must not pass by word count.
                if sum(bool(g) for g in groups) < 2 or not any(groups[:3]):
                    continue
                info = set.union(*groups)
                score = (3 * len(groups[0]) + 6 * len(groups[1]) + 5 * len(groups[2])
                         + 2 * len(groups[3]) + .5 * len(groups[4]) + 2 * len(info) / len(selected)
                         - .7 * sum(w in stop for w in selected))
                candidates.append({"caption": caption, "score": round(score, 8), "attributes": sorted(info)})
                seen.add(caption)
    candidates.sort(key=lambda x: (-x["score"], len(x["caption"].split()), x["caption"]))
    best = candidates[0] if candidates else {"caption": "", "score": 0., "attributes": []}
    return {**best, "candidate_count": len(candidates), "reason": "extractive_attributes" if candidates else "no_informative_span"}


def attribute_caption(response):
    return attribute_caption_details(response)["caption"]


def prior_caption_registration(payload):
    """Validate reusable long responses against original image/model hashes."""
    import csv
    import json
    from pathlib import Path
    reference = str(payload.get("prior_captions_source", "")).strip()
    if not reference:
        return None, {}
    root = Path(reference).expanduser().resolve()
    if root.is_file():
        root = root.parent
    candidates = [root, root / "shared/caption_generation"]
    found = []
    for folder in candidates:
        csv_path = folder / "captions.csv"
        for protocol_path, status_path in [
            (folder / "generation_protocol/protocol.json", folder / "status.json"),
            (folder / "caption_provenance/generation_protocol/protocol.json", folder / "caption_provenance/status.json"),
        ]:
            if csv_path.is_file() and protocol_path.is_file() and status_path.is_file():
                found.append((csv_path, protocol_path, status_path))
    if len(found) != 1:
        raise FileNotFoundError("LONG_CAPTIONS_SOURCE must be the old bundle or shared/caption_generation, with CSV, generation protocol and status")
    csv_path, protocol_path, status_path = found[0]
    protocol, status = json.loads(protocol_path.read_text("utf-8")), json.loads(status_path.read_text("utf-8"))
    if not status.get("complete") or file_sha256(csv_path) != status.get("csv_sha256") or object_hash(protocol) != status.get("identity"):
        raise ValueError("Original caption CSV/protocol/status is incomplete or changed")
    if protocol.get("model_assets") != checkpoint_asset_identity(payload["model_path"]):
        raise ValueError("Reuse only long responses produced by this same frozen Florence checkpoint")
    image_binding = {r["image_id"]: r for r in protocol["images"]}
    if len(image_binding) != len(protocol["images"]):
        raise ValueError("Duplicate original image IDs")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {r["image_id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate caption rows")
    reused = {}
    for record in payload["records"]:
        bound, row = image_binding.get(record["image_id"]), lookup.get(record["image_id"])
        if bound is None or row is None:
            continue  # Generate only genuinely missing images.
        if (bound.get("image_sha256") != record["image_sha256"] or bound.get("split") != record["split"]
                or row.get("split") != record["split"] or row.get("captioner_fingerprint") != status["identity"]):
            raise ValueError("Original long caption belongs to another image/split: " + record["image_id"])
        if not row.get("source_long_caption"):
            continue
        reused[record["image_id"]] = row
    identity = {"csv_sha256": status["csv_sha256"], "generation_identity": status["identity"],
                "reusable_images": len(reused), "operation": "reextract_original_response_not_human_ground_truth"}
    return identity, reused


def generate_attribute_records(payload, identity, factory):
    """Seed a NEW per-image cache; never change the old CSV or old checkpoint."""
    from pathlib import Path
    registration, source = prior_caption_registration(payload)
    reused = 0
    directory = Path(payload["output"]) / "generated"
    for record in payload["records"]:
        if record["image_id"] not in source or cached_row(directory, record, identity) is not None:
            continue
        original = source[record["image_id"]]
        limited = str(original["generation_token_limit_reached"]).lower() == "true"
        row = make_generation_row(record, original["source_long_caption"], limited, identity)
        atomic_json(cache_path(directory, record["image_id"]), {"identity": identity, "row": row})
        reused += 1
    print("Re-extracted long responses in NEW cache:", reused, flush=True)
    return generate_prefix_records(payload, identity, factory=factory)
