"""Frozen Florence proposals and image-grounded short-span selection for DTD."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from dtd_caption_screening import clean_dtd_caption
from dtd_caption_finetune import atomic_csv, atomic_json, checkpoint_asset_identity, file_sha256, load_caption_manifest, object_hash, read_csv_rows


TASK = "<MORE_DETAILED_CAPTION>"
METHOD_VERSION = "florence_dtd_grounded_spans_v1"
METHODS = ("prefix_control", "first_valid_control", "single_grounded_control", "multibeam_grounded")
BASE_COLUMNS = ["image_id", "split", "raw_caption", "final_caption", "caption_valid", "leakage_detected",
    "lexical_leakage_detected", "semantic_leakage_detected", "semantic_leakage_score",
    "semantic_leakage_class", "retry_count", "vlm_model", "prompt_version"]
COLUMNS = BASE_COLUMNS + ["caption_input_policy", "semantic_audit_status", "captioner_fingerprint",
    "generation_method", "source_long_caption", "source_hypothesis", "source_sentence", "source_word_start",
    "source_word_end", "image_text_similarity", "alias_hit_diagnostic", "selection_fallback"]
EDGE_WORDS = set("a an the this that these those it its is are was were be been being and or but with of to in on at by for from into as has have had can will which there s t".split())
BAD_START = EDGE_WORDS | {"image", "photo", "picture", "photograph", "shows", "showing", "depicts"}
# Same diagnostic vocabulary as the existing DTD ablations; never a new eligibility gate.
ALIASES = {
    "banded": ["bands", "banding"], "blotchy": ["blotches", "blotch"], "braided": ["braid", "braids"],
    "bubbly": ["bubbles", "bubble"], "bumpy": ["bumps", "bump"], "chequered": ["checkered", "checkerboard"],
    "cobwebbed": ["cobwebs", "cobweb"], "cracked": ["cracks", "crack"],
    "crosshatched": ["crosshatched", "cross hatching"], "crystalline": ["crystals", "crystal"],
    "dotted": ["dots", "polka dots"], "fibrous": ["fibers", "fibres"], "flecked": ["flecks"],
    "freckled": ["freckles"], "frilly": ["frills"], "gauzy": ["gauze"], "gridded": ["grid", "grids"],
    "grooved": ["grooves"], "honeycombed": ["honeycomb"], "interlaced": ["interlacing"],
    "knitted": ["knit", "knitting"], "lacelike": ["lace", "lace like"], "lined": ["parallel lines"],
    "marbled": ["marbling"], "meshed": ["mesh"], "paisley": [], "perforated": ["perforations"],
    "pitted": ["pits"], "pleated": ["pleats"], "polka-dotted": ["polka dotted", "polka dots"],
    "porous": ["pores"], "potholed": ["potholes"], "scaly": ["scales"], "smeared": ["smears"],
    "spiralled": ["spiraled", "spirals", "spiral"], "sprinkled": ["sprinkles"], "stained": ["stains"],
    "stratified": ["strata"], "striped": ["stripes", "stripe"], "studded": ["studs"],
    "swirly": ["swirls", "swirl"], "veined": ["veins"], "waffled": ["waffle"],
    "woven": ["weave", "weaving"], "wrinkled": ["wrinkles"], "zigzagged": ["zigzag", "zig zag"],
}


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).replace("_", " ").lower()).strip()


def bool_value(value):
    return str(value).lower() in {"true", "yes", "1"}


def term_hits(text, names, aliases=False):
    padded = " " + normalize(text) + " "
    terms = set(normalize(name) for name in names)
    if aliases:
        terms.update(normalize(term) for name in names for term in ALIASES.get(name, []))
    return sorted(term for term in terms if term and " " + term + " " in padded)


def valid_short(text, names):
    return 5 <= len(normalize(text).split()) <= 8 and not term_hits(text, names)


def strip_opening(text):
    for _ in range(3):
        before = text
        text = re.sub(r"^(?:(?:the|this)\s+)?(?:image|picture|photo(?:graph)?)\s+"
                      r"(?:is|shows|depicts|features|contains)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:(?:a|an|the)\s+)?close[ -]up(?:\s+view)?\s+(?:of|showing)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:a|an|the)(?:\s+|$)", "", text, flags=re.I)
        if text == before:
            break
    return text.strip()


def candidate_spans(raw, names, limit=48, hypothesis=0):
    """Extract substrings only; do not invent replacements or cross sentence boundaries."""
    candidates, seen = [], set()
    sentences = re.split(r"(?<=[.!?;:])\s+", str(raw).strip())
    for sentence_index, sentence in enumerate(sentences):
        words = re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?", strip_opening(sentence))
        for start in range(len(words)):
            for length in (8, 7, 6, 5):
                selected = words[start:start + length]
                if len(selected) != length:
                    continue
                text = " ".join(selected)
                key = normalize(text)
                if (key in seen or normalize(selected[0]) in BAD_START or normalize(selected[-1]) in EDGE_WORDS
                        or not valid_short(text, names) or clean_dtd_caption(text) != text):
                    continue
                tokens = key.split()
                if len(set(tokens)) < 4 or max(Counter(tokens).values()) > 2:
                    continue
                seen.add(key)
                candidates.append({"text": text, "long_caption": raw, "hypothesis": hypothesis,
                    "sentence": sentence_index, "word_start": start, "word_end": start + length})
    if len(candidates) > limit:
        indices = [round(i * (len(candidates) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
        candidates = [candidates[index] for index in indices]
    return candidates


def proposal_sets(raw_proposals, names, span_limit):
    prefix = {"text": clean_dtd_caption(raw_proposals[0]), "long_caption": raw_proposals[0],
              "hypothesis": 0, "sentence": -1, "word_start": 0, "word_end": -1}
    all_spans, seen = [], set()
    single = candidate_spans(raw_proposals[0], names, span_limit, 0)
    for index, raw in enumerate(raw_proposals):
        for candidate in candidate_spans(raw, names, span_limit, index):
            key = normalize(candidate["text"])
            if key not in seen:
                seen.add(key)
                all_spans.append(candidate)
    return prefix, single, all_spans


def select_candidate(candidates, scores, fallback):
    if not candidates:
        return fallback, True
    # Stable ordering for exact ties; class-audit scores are deliberately NOT consulted.
    return max(candidates, key=lambda candidate: round(scores[candidate["text"]]["image_cosine"], 7)), False


def row_for(record, selected, score, method, identity, names, fallback=False):
    text = selected["text"]
    if clean_dtd_caption(text) != text:
        raise ValueError("Export would change text when M-TACS cleans raw_caption; inspect this candidate")
    lexical = bool(term_hits(text, names))
    semantic = score.get("class_cosine", "") if score else ""
    return {"image_id": record["image_id"], "split": record["split"], "raw_caption": text, "final_caption": text,
        "caption_valid": valid_short(text, names), "leakage_detected": lexical,
        "lexical_leakage_detected": lexical, "semantic_leakage_detected": semantic != "" and semantic >= .30,
        "semantic_leakage_score": semantic, "semantic_leakage_class": score.get("class_name", "") if score else "",
        "retry_count": 0, "vlm_model": "microsoft/Florence-2-base+frozen-grounded-selection:" + identity[:16],
        "prompt_version": METHOD_VERSION, "caption_input_policy": "dtd_lexical_v2",
        "semantic_audit_status": "diagnostic_only" if score else "not_run_empty_caption",
        "captioner_fingerprint": identity, "generation_method": method,
        "source_long_caption": selected["long_caption"], "source_hypothesis": selected["hypothesis"],
        "source_sentence": selected["sentence"], "source_word_start": selected["word_start"],
        "source_word_end": selected["word_end"], "image_text_similarity": score.get("image_cosine", "") if score else "",
        "alias_hit_diagnostic": bool(term_hits(text, names, aliases=True)), "selection_fallback": fallback}


def build_image_result(record, proposals, scores, names, settings, identity):
    prefix, single, multiple = proposal_sets(proposals, names, settings["max_spans_per_hypothesis"])
    one_best, one_fallback = select_candidate(single, scores, prefix)
    multi_best, multi_fallback = select_candidate(multiple, scores, prefix)
    selected = {"prefix_control": (prefix, False), "first_valid_control": (single[0] if single else prefix, not single),
                "single_grounded_control": (one_best, one_fallback), "multibeam_grounded": (multi_best, multi_fallback)}
    rows = {method: row_for(record, candidate, scores.get(candidate["text"]), method, identity, names, fallback)
            for method, (candidate, fallback) in selected.items()}
    audit = [{**candidate, **scores[candidate["text"]],
              "alias_hits": term_hits(candidate["text"], names, aliases=True)} for candidate in multiple]
    return {"identity": identity, "image_id": record["image_id"], "split": record["split"],
            "proposals": proposals, "rows": rows, "candidate_audit": audit}


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def model_autocast(device):
    import torch
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


class FrozenFlorence:
    def __init__(self, path, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.device = device
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True,
            torch_dtype=self.dtype, low_cpu_mem_usage=True, attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)

    def generate(self, images, beams, returns, tokens):
        import torch
        inputs = self.processor(text=[TASK] * len(images), images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype if key == "pixel_values" else value.dtype)
                  for key, value in inputs.items() if torch.is_tensor(value)}
        with torch.inference_mode(), model_autocast(self.device):
            output = self.model.generate(**inputs, max_new_tokens=tokens, num_beams=beams,
                num_return_sequences=returns, do_sample=False)
        decoded = self.processor.batch_decode(output, skip_special_tokens=False)
        if len(decoded) != len(images) * returns:
            raise ValueError("Wrong Florence beam output count; no batch rows were cached")
        results = []
        for index, image in enumerate(images):
            group = []
            for text in decoded[index * returns:(index + 1) * returns]:
                parsed = self.processor.post_process_generation(text, task=TASK, image_size=(image.width, image.height))
                group.append(str(parsed.get(TASK, text) if isinstance(parsed, dict) else text).strip())
            results.append(group)
        return results


class ImageTextRanker:
    def __init__(self, path, names, device, text_batch_size=256):
        import torch
        from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel
        self.device, self.names, self.text_batch_size = device, names, text_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model, loading = CLIPModel.from_pretrained(path, local_files_only=True, output_loading_info=True)
        missing = [key for key in loading.get("missing_keys", []) if not key.endswith("position_ids")]
        if missing or loading.get("mismatched_keys"):
            raise ValueError(f"A complete CLIP image AND text checkpoint is required; missing weights: {missing[:8]}")
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        if (Path(path) / "preprocessor_config.json").is_file():
            self.image_processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
            self.preprocessing_source = "checkpoint_preprocessor_config"
        else:
            size = self.model.config.vision_config.image_size
            if size != 224 or self.model.config.vision_config.patch_size != 32:
                raise ValueError("Missing CLIP image preprocessing config for a non-ViT-B/32 checkpoint")
            self.image_processor = CLIPImageProcessor()
            self.preprocessing_source = "transformers_4_49_standard_clip_vit_b32_defaults"
            print("CLIP preprocessor_config.json absent: using recorded standard CLIP ViT-B/32 preprocessing defaults", flush=True)
        prompts = [template.format(name=name.replace("_", " ").replace("-", " ")) for name in names
                   for template in ("a photo of {name}", "a scene showing {name}")]
        vectors = self.encode_text(prompts).reshape(len(names), 2, -1).mean(1)
        self.class_vectors = torch.nn.functional.normalize(vectors, dim=1)

    def encode_text(self, texts):
        import torch
        result = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.text_batch_size):
                tokens = self.tokenizer(texts[start:start + self.text_batch_size], padding=True,
                    truncation=True, max_length=self.model.config.text_config.max_position_embeddings, return_tensors="pt")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                with model_autocast(self.device):
                    features = self.model.get_text_features(**tokens)
                result.append(torch.nn.functional.normalize(features.float(), dim=1))
        return torch.cat(result)

    def score(self, images, text_groups):
        import torch
        unique = list(dict.fromkeys(text for group in text_groups for text in group if text.strip()))
        if not unique:
            return [{} for _ in images]
        texts = self.encode_text(unique)
        positions = {text: index for index, text in enumerate(unique)}
        pixels = self.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode(), model_autocast(self.device):
            features = self.model.get_image_features(pixel_values=pixels)
        features = torch.nn.functional.normalize(features.float(), dim=1)
        image_scores = features @ texts.T
        class_scores, class_indices = (texts @ self.class_vectors.T).max(1)
        image_scores, class_scores, class_indices = image_scores.cpu().tolist(), class_scores.cpu().tolist(), class_indices.cpu().tolist()
        result = []
        for i, group in enumerate(text_groups):
            result.append({text: {"image_cosine": image_scores[i][positions[text]],
                "class_cosine": class_scores[positions[text]],
                "class_name": self.names[class_indices[positions[text]]]} for text in group if text.strip()})
        return result


def validate_settings(settings):
    for key in ("batch_size", "num_beams", "num_return_sequences", "max_new_tokens", "max_spans_per_hypothesis", "clip_text_batch_size"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["num_beams"] < 2 or settings["num_return_sequences"] > settings["num_beams"]:
        raise ValueError("Use beam search with 1 <= num_return_sequences <= num_beams and num_beams >= 2")


def cache_path(output, image_id):
    return Path(output) / "image_cache" / (object_hash(image_id) + ".json")


def read_cached(output, record, identity):
    path = cache_path(output, record["image_id"])
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached["identity"] != identity or cached["image_id"] != record["image_id"] or cached["split"] != record["split"]:
        raise ValueError("Wrong experiment/image in cache; original cache was not modified")
    if set(cached["rows"]) != set(METHODS):
        raise ValueError("Incomplete per-image transaction in cache")
    for method, row in cached["rows"].items():
        if (set(row) != set(COLUMNS) or row["image_id"] != record["image_id"] or row["split"] != record["split"]
                or row["captioner_fingerprint"] != identity or row["generation_method"] != method):
            raise ValueError("Invalid cached row schema/identity")
    return cached


def run_manifest(payload):
    import torch
    import transformers
    validate_settings(payload["settings"])
    records = payload["records"]
    if not records or len({r["image_id"] for r in records}) != len(records):
        raise ValueError("Missing/duplicate dataset records")
    return {"version": METHOD_VERSION, "task": TASK, "settings": payload["settings"], "seed": payload["seed"],
        "code_sha256": payload["code_sha256"], "dtd_split": payload["dtd_split"],
        "class_names_for_post_generation_audit": payload["class_names"],
        "base_florence_assets": checkpoint_asset_identity(payload["florence"]),
        "clip_assets": checkpoint_asset_identity(payload["clip"]),
        "image_preprocessing": "checkpoint_config_if_present_else_standard_clip_vit_b32_defaults",
        "images": [{"image_id": r["image_id"], "split": r["split"], "sha256": file_sha256(r["image_path"])} for r in records],
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
            "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported()},
        "canonical_policy": "dtd_lexical_v2", "semantic_audit_threshold": .30,
        "semantic_hard_gate": False, "trainable_parameters": 0}


def export_cached(payload, identity):
    output = Path(payload["output"])
    cached = [read_cached(output, r, identity) for r in payload["records"]]
    complete = all(item is not None for item in cached)
    available = [item for item in cached if item is not None]
    folder = output if complete else output / "preview_incomplete"
    paths = {}
    for method in METHODS:
        path = folder / "exports" / "captions.csv" if method == "multibeam_grounded" else folder / "controls" / method / "captions.csv"
        atomic_csv(path, [item["rows"][method] for item in available], COLUMNS)
        paths[method] = str(path)
    audit_path = folder / "candidate_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in available:
            handle.write(json.dumps({"image_id": item["image_id"], "split": item["split"],
                "raw_proposals": item["proposals"], "candidates": item["candidate_audit"]}, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(audit_path)
    summaries = []
    for method in METHODS:
        for role in ("train", "val", "test"):
            rows = [item["rows"][method] for item in available if item["split"] == role]
            if not rows:
                continue
            summaries.append({"method": method, "split": role, "images": len(rows),
                "valid": sum(row["caption_valid"] for row in rows),
                "valid_rate": sum(row["caption_valid"] for row in rows) / len(rows),
                "unique_captions": len({row["final_caption"] for row in rows}),
                "canonical_hits": sum(row["lexical_leakage_detected"] for row in rows),
                "alias_hits_diagnostic": sum(row["alias_hit_diagnostic"] for row in rows),
                "fallbacks": sum(row["selection_fallback"] for row in rows),
                "mean_image_text_score": sum(row["image_text_similarity"] for row in rows if row["image_text_similarity"] != "") /
                    max(1, sum(row["image_text_similarity"] != "" for row in rows))})
    if summaries:
        atomic_csv(folder / "caption_generation_summary.csv", summaries, list(summaries[0]))
    summary = {"identity": identity, "complete": complete, "rows": len(available), "expected": len(cached),
        "csv_paths": paths, "csv_sha256": {key: file_sha256(path) for key, path in paths.items()},
        "optimized_clip_score_is_not_independent_quality_evidence": True,
        "downstream_improvement_verified": False, "human_ground_truth": False}
    atomic_json(output / "generation_status.json", summary)
    return summary


def run_generation(payload, device, florence_factory=FrozenFlorence, ranker_factory=ImageTextRanker):
    import torch
    output = Path(payload["output"])
    manifest = run_manifest(payload)
    identity = object_hash(manifest)
    manifest_path = output / "generation_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Changed assets/code/dataset/settings/runtime: choose a new output directory")
    if not manifest_path.is_file() and ((output / "image_cache").exists() or (output / "exports").exists()):
        raise ValueError("Existing generated artifacts have no matching manifest; use another output directory")
    atomic_json(manifest_path, manifest)
    pending = [r for r in payload["records"] if read_cached(output, r, identity) is None]
    limit = payload.get("max_new_images", 0)
    if limit < 0:
        raise ValueError("max_new_images must be >= 0; zero processes all missing images")
    if limit:
        pending = pending[:limit]
    print(f"Grounded captions: remaining this run={len(pending)}; output={output}", flush=True)
    if pending:
        generator = florence_factory(payload["florence"], device)
        ranker = ranker_factory(payload["clip"], payload["class_names"], device, payload["settings"]["clip_text_batch_size"])
        atomic_json(output / "clip_preprocessing_used.json", {"source": ranker.preprocessing_source,
                    "config": ranker.image_processor.to_dict()})
        size, offset = payload["settings"]["batch_size"], 0
        while offset < len(pending):
            group = pending[offset:offset + size]
            try:
                images = [load_rgb(r["image_path"]) for r in group]
                greedy = generator.generate(images, 1, 1, payload["settings"]["max_new_tokens"])
                beams = generator.generate(images, payload["settings"]["num_beams"], payload["settings"]["num_return_sequences"], payload["settings"]["max_new_tokens"])
                if (len(greedy) != len(group) or len(beams) != len(group) or any(len(p) != 1 for p in greedy)
                        or any(len(p) != payload["settings"]["num_return_sequences"] for p in beams)):
                    raise ValueError("Wrong image/hypothesis count from Florence")
                proposals = [one + many for one, many in zip(greedy, beams)]
                text_groups = []
                for raw in proposals:
                    prefix, single, multiple = proposal_sets(raw, payload["class_names"], payload["settings"]["max_spans_per_hypothesis"])
                    text_groups.append(list(dict.fromkeys([prefix["text"]] + [item["text"] for item in multiple])))
                scores = ranker.score(images, text_groups)
                if len(scores) != len(group) or any(set(score) != {t for t in texts if t.strip()} for score, texts in zip(scores, text_groups)):
                    raise ValueError("Wrong CLIP score count/keys; no batch rows were cached")
                if any(not math.isfinite(value["image_cosine"]) or not math.isfinite(value["class_cosine"])
                       for score in scores for value in score.values()):
                    raise ValueError("Non-finite CLIP score")
                transactions = [build_image_result(r, p, s, payload["class_names"], payload["settings"], identity)
                                for r, p, s in zip(group, proposals, scores)]
            except torch.cuda.OutOfMemoryError:
                if len(group) == 1:
                    raise
                size = max(1, len(group) // 2)
                torch.cuda.empty_cache()
                print("Caption batch reduced after CUDA OOM to", size, flush=True)
                continue
            for record, result in zip(group, transactions):
                atomic_json(cache_path(output, record["image_id"]), result)
            offset += len(group)
            print(f"Completed {offset}/{len(pending)} newly queued images", flush=True)
    return export_cached(payload, identity)


def worker_main(payload_path):
    import torch
    import transformers
    if not transformers.__version__.startswith("4.49."):
        raise RuntimeError("Use the isolated Transformers 4.49.x worker for the attached legacy Florence model")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator for this Florence + full CLIP generation worker")
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    torch.manual_seed(payload["seed"])
    torch.cuda.manual_seed_all(payload["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    summary = run_generation(payload, torch.device("cuda"))
    print(json.dumps(summary, indent=2), flush=True)


def effective_dtd_v2(row, names):
    text = clean_dtd_caption(str(row.get("raw_caption", "")).strip() or row["final_caption"])
    try:
        finite_score = math.isfinite(float(row["semantic_leakage_score"]))
    except (TypeError, ValueError):
        finite_score = False
    recoverable = bool_value(row["caption_valid"]) or (bool_value(row["semantic_leakage_detected"]) and finite_score)
    return text, bool(valid_short(text, names) and recoverable)


def compare_caption_files(paths, records, names, output_dir, write_matched_masks=True):
    """Read existing CSVs without changing them; compare the actual dtd_lexical_v2 model inputs."""
    expected = {r["image_id"]: r["split"] for r in records}
    if not paths or len(expected) != len(records):
        raise ValueError("Provide named complete CSVs and distinct dataset records")
    methods, source_hashes = {}, {}
    for label, path in paths.items():
        if not re.fullmatch(r"[a-z0-9_]+", label):
            raise ValueError("Comparison labels must be lowercase letters/digits/underscores")
        rows = read_csv_rows(path, ["image_id", "split", "final_caption", "caption_valid", "leakage_detected",
                                    "semantic_leakage_detected", "semantic_leakage_score"])
        mapped = {row["image_id"]: row for row in rows}
        if len(mapped) != len(rows) or set(mapped) != set(expected) or any(row["split"] != expected[key] for key, row in mapped.items()):
            raise ValueError(f"{label}: incomplete, duplicate or wrong official DTD split; original CSV was not modified")
        methods[label] = {key: {"row": mapped[key], "text": effective_dtd_v2(mapped[key], names)[0],
                                "valid": effective_dtd_v2(mapped[key], names)[1]} for key in expected}
        source_hashes[label] = file_sha256(path)
    output_dir = Path(output_dir)
    signatures = {"csv_sha256": source_hashes, "policy": "dtd_lexical_v2", "names": names,
                  "image_ids_and_splits": expected, "comparison_version": 1}
    lock = output_dir / "comparison_manifest.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != signatures:
        raise ValueError("Comparison inputs changed; choose a new comparison output directory")
    atomic_json(lock, signatures)
    common = {key: all(method[key]["valid"] for method in methods.values()) for key in expected}
    summaries, details = [], []
    first_label = next(iter(methods))
    for label, method in methods.items():
        for role in ("train", "val", "test"):
            entries = [entry for key, entry in method.items() if expected[key] == role]
            if entries:
                summaries.append({"method": label, "split": role, "images": len(entries),
                    "valid": sum(entry["valid"] for entry in entries),
                    "valid_rate": sum(entry["valid"] for entry in entries) / len(entries),
                    "unique_input_texts": len({entry["text"] for entry in entries}),
                    "valid_with_alias_hit": sum(entry["valid"] and bool(term_hits(entry["text"], names, True)) for entry in entries),
                    "common_valid": sum(common[key] for key in expected if expected[key] == role),
                    "text_changed_vs_first": sum(method[key]["text"] != methods[first_label][key]["text"] for key in expected if expected[key] == role)})
        for key, entry in method.items():
            details.append({"method": label, "image_id": key, "split": expected[key], "model_input_text": entry["text"],
                            "valid": entry["valid"], "common_valid": common[key]})
        if write_matched_masks:
            masked = []
            for record in records:
                key = record["image_id"]
                row = dict(method[key]["row"])
                row.update(original_caption_valid=row["caption_valid"], original_semantic_flag=row["semantic_leakage_detected"],
                    original_semantic_score=row["semantic_leakage_score"], caption_valid=common[key],
                    caption_input_policy="dtd_lexical_v2", comparison_mask="common_valid_intersection")
                if not common[key]:
                    # Prevent legacy semantic-only recovery from switching a deliberately masked caption back on.
                    row.update(semantic_leakage_detected=False, semantic_leakage_score="",
                               semantic_audit_status="original_audit_retained_in_original_columns")
                masked.append(row)
            columns = list(dict.fromkeys(key for row in masked for key in row))
            atomic_csv(output_dir / "matched_masks" / label / "captions.csv", masked, columns)
            assert all(effective_dtd_v2(row, names)[1] == common[row["image_id"]] for row in masked)
    atomic_csv(output_dir / "caption_comparison_summary.csv", summaries, list(summaries[0]))
    atomic_csv(output_dir / "caption_comparison_details.csv", details, list(details[0]))
    atomic_csv(output_dir / "common_valid_mask.csv", [{"image_id": key, "split": role, "caption_valid": common[key]}
               for key, role in expected.items()], ["image_id", "split", "caption_valid"])
    return summaries
