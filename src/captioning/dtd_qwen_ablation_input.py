"""Read-only binding of a precomputed Qwen3-VL-8B DTD CSV to notebook 05."""

import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from dtd_caption_screening import clean_dtd_caption


QWEN8B_EXPECTED = {
    "method": "qwen3_vl_8b_instruct", "display_name": "Qwen3-VL-8B-Instruct",
    "model": "Qwen/Qwen3-VL-8B-Instruct", "prompt_version": "qwen3vl_dtd_short_caption_v1",
    "policy": "dtd_lexical_v2", "semantic_audit_status": "diagnostic_only",
}
QWEN8B_REQUIRED = {
    "image_id", "split", "raw_caption", "final_caption", "caption_valid", "leakage_detected",
    "lexical_leakage_detected", "semantic_leakage_detected", "semantic_leakage_score",
    "semantic_leakage_class", "retry_count", "vlm_model", "prompt_version", "caption_input_policy",
    "semantic_audit_status", "captioner_fingerprint", "generation_method", "audit_fingerprint",
    "source_long_caption",
}


def qwen8b_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qwen8b_bool(value):
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no", "y", "n"}:
        raise ValueError(f"Invalid CSV boolean: {value!r}")
    return normalized in {"true", "1", "yes", "y"}


def qwen8b_normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).replace("_", " ").lower()).strip()


def verify_qwen8b_caption_csv(path, split_frames, class_names, expected_sha256=""):
    path = Path(path)
    digest = qwen8b_file_sha256(path)
    if expected_sha256 and digest != str(expected_sha256).lower():
        raise ValueError("Qwen8B CSV SHA256 differs from EXPECTED_CAPTIONS_SHA256")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not QWEN8B_REQUIRED.issubset(fields):
            raise ValueError("Qwen8B CSV has missing/duplicate columns. Use the complete audited export from the Qwen caption notebook; "
                             "do not use captions_raw_partial.csv or remove provenance columns")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Qwen8B CSV is empty or has a malformed/incomplete row")
    expected = {str(image_id): split for split, frame in split_frames.items() for image_id in frame.image_id}
    ids = [row["image_id"] for row in rows]
    if (len(expected) != sum(len(frame) for frame in split_frames.values()) or len(set(ids)) != len(ids)
            or set(ids) != set(expected) or any(row["split"] != expected[row["image_id"]] for row in rows)):
        raise ValueError("Qwen8B CSV is incomplete or does not match the official DTD train/val/test image IDs and splits")
    for column, key in (("generation_method", "method"), ("vlm_model", "model"),
                        ("prompt_version", "prompt_version"), ("caption_input_policy", "policy"),
                        ("semantic_audit_status", "semantic_audit_status")):
        if {row[column] for row in rows} != {QWEN8B_EXPECTED[key]}:
            raise ValueError(f"Unexpected {column}: this notebook accepts the Qwen3-VL-8B-Instruct export only, not Florence/32B/partial outputs")
    fingerprints = {}
    for column in ("captioner_fingerprint", "audit_fingerprint"):
        values = {row[column] for row in rows}
        if len(values) != 1 or not re.fullmatch(r"[a-f0-9]{64}", next(iter(values))):
            raise ValueError(f"Missing/mixed {column}; use one complete Qwen8B generation/audit run")
        fingerprints[column] = next(iter(values))
    terms = {qwen8b_normalized(name) for name in class_names}
    split_counts, valid_counts = {}, {}
    for row in rows:
        text = row["final_caption"]
        if row["raw_caption"] != text or clean_dtd_caption(text) != text:
            raise ValueError(f"Qwen8B model input would change during cleanup: {row['image_id']}; use the unmodified export")
        normalized = qwen8b_normalized(text)
        lexical = any(" " + term + " " in " " + normalized + " " for term in terms if term)
        eligible = 5 <= len(normalized.split()) <= 8 and not lexical
        if (qwen8b_bool(row["caption_valid"]) != eligible
                or qwen8b_bool(row["lexical_leakage_detected"]) != lexical
                or qwen8b_bool(row["leakage_detected"]) != lexical):
            raise ValueError(f"Qwen8B mask/lexical flags do not match dtd_lexical_v2: {row['image_id']}")
        qwen8b_bool(row["semantic_leakage_detected"])  # Diagnostic flag is NOT an eligibility gate.
        try:
            score = float(row["semantic_leakage_score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Missing CLIP audit score for {row['image_id']}") from error
        if not math.isfinite(score) or not -1.00001 <= score <= 1.00001 or row["semantic_leakage_class"] not in class_names:
            raise ValueError(f"Invalid CLIP class-audit result for {row['image_id']}")
        role = row["split"]
        split_counts[role] = split_counts.get(role, 0) + 1
        valid_counts[role] = valid_counts.get(role, 0) + int(eligible)
    return {"method": QWEN8B_EXPECTED["method"], "display_name": QWEN8B_EXPECTED["display_name"],
            "vlm_model": QWEN8B_EXPECTED["model"], "prompt_version": QWEN8B_EXPECTED["prompt_version"],
            "policy": QWEN8B_EXPECTED["policy"], "sha256": digest, "rows": len(rows),
            "split_counts": split_counts, "stored_valid_by_split": valid_counts, **fingerprints}


def looks_like_qwen8b_csv(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            first = next(csv.DictReader(handle), {})
        return first.get("generation_method") == QWEN8B_EXPECTED["method"] and first.get("vlm_model") == QWEN8B_EXPECTED["model"]
    except (OSError, UnicodeError, csv.Error):
        return False


def resolve_qwen8b_caption_csv(explicit_path, split_frames, class_names, expected_sha256="", local_candidates=(), roots=()):
    if str(explicit_path).strip():
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CAPTIONS_INPUT_PATH not found: {path}. Set its exact attached CSV path")
        return path.resolve(), verify_qwen8b_caption_csv(path, split_frames, class_names, expected_sha256)
    candidates = {str(Path(path).resolve()): Path(path).resolve() for path in local_candidates if Path(path).is_file()}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            depth = len(Path(current).relative_to(root).parts)
            dirs[:] = sorted(name for name in dirs if name not in {"images", ".git", "__pycache__", "node_modules", ".cache"}) if depth < 8 else []
            for name in sorted(files):
                if name.lower().endswith(".csv"):
                    path = (Path(current) / name).resolve()
                    candidates[str(path)] = path
    matches, failures = {}, []
    for path in sorted(candidates.values(), key=lambda item: (len(str(item)), str(item))):
        if not looks_like_qwen8b_csv(path):
            continue
        try:
            binding = verify_qwen8b_caption_csv(path, split_frames, class_names, expected_sha256)
        except (OSError, ValueError) as error:
            failures.append(f"{path}: {error}")
            continue
        matches.setdefault(binding["sha256"], (path, binding))
    if not matches:
        details = "\n" + "\n".join(failures[:3]) if failures else ""
        raise FileNotFoundError("No complete audited Qwen3-VL-8B CSV found. Attach captions.csv or captions_qwen3vl_8b_instruct.csv "
            "and set CAPTIONS_INPUT_PATH. Caption generation and fallback to Florence are disabled." + details)
    if len(matches) != 1:
        raise ValueError("Multiple different Qwen8B CSVs found. Set CAPTIONS_INPUT_PATH explicitly; no automatic choice between runs is made")
    return next(iter(matches.values()))


def stage_qwen8b_caption_csv(explicit_path, cached_path, split_frames, class_names, policy, generate,
                             expected_sha256="", local_candidates=(), roots=()):
    if generate or policy != "dtd_lexical_v2":
        raise ValueError("This experiment requires precomputed Qwen8B captions and dtd_lexical_v2; generation is disabled")
    source, binding = resolve_qwen8b_caption_csv(explicit_path, split_frames, class_names,
        expected_sha256, local_candidates, roots)
    cached = Path(cached_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.is_file() and qwen8b_file_sha256(cached) != binding["sha256"]:
        raise ValueError("Output directory already contains another caption CSV. Choose a new output directory; nothing was overwritten")
    lock = cached.parent / "caption_source.json"
    if lock.is_file():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        if any(previous.get(key) != binding[key] for key in ("sha256", "method", "captioner_fingerprint", "audit_fingerprint")):
            raise ValueError("Changed Qwen8B input in an existing run. Use a new output directory; old checkpoints/caches are not reused")
    if not cached.is_file() and cached.resolve() != source.resolve():
        temporary = cached.with_suffix(cached.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(cached)
    verify_qwen8b_caption_csv(cached, split_frames, class_names, binding["sha256"])
    provenance = {**binding, "source": str(source), "generation_needed": False}
    temporary = lock.with_suffix(lock.suffix + ".tmp")
    temporary.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(lock)
    print("Bound caption method:", binding["display_name"], "| CSV:", source)
    print("Caption CSV SHA256:", binding["sha256"])
    return cached, binding
