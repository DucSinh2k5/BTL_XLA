#!/usr/bin/env python3
"""Create a separate SUN397 caption CSV with exact class-name redaction.

This is deterministic post-processing of the deployed Florence-2 raw captions.
It does not call a model, invent words, regenerate embeddings, or train a model.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/deployment/sun397/mtacs_full_both/captions.csv"
CLASSES = ROOT / "artifacts/deployment/sun397/mtacs_full_both/class_names.json"
OUTPUT = ROOT / "data/captions/sun397/captions_lexical_redaction_v2.csv"
POLICY = "sun397_lexical_redaction_v2"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
PREFIX_RE = re.compile(
    r"^(?:(?:the|this)\s+(?:image|photo|picture)\s+(?:is|shows|depicts)\s+|"
    r"(?:a|an)\s+(?:close[ -]?up|detailed)\s+(?:view|photo|image|picture)?\s*(?:of\s+)?|"
    r"(?:a|an|the)\s+)",
    re.IGNORECASE,
)
FUNCTION_WORDS = {
    "a", "an", "the", "of", "with", "in", "on", "at", "to", "and", "or", "for", "from", "by",
    "is", "are", "was", "were", "has", "have", "into", "over", "under",
    "this", "that", "these", "those", "there", "it", "they", "which", "who", "as", "be", "been",
}
WEAK_BOUNDARY_WORDS = FUNCTION_WORDS | {
    "appears", "located", "made", "reads", "showing", "shows", "depicts", "surrounded", "covered",
    "supported", "consists", "features", "contains", "includes", "positioned", "arranged", "built",
    "situated", "seems", "looks", "lies", "stands", "sits", "extends", "featuring", "having",
}
INCOMPLETE_ENDINGS = {
    "arched", "pointed", "pitched", "sloping", "steep", "ornate", "intricate", "wooden", "painted",
    "covered", "located", "surrounded", "arranged", "filled", "featuring", "including",
}


def words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(str(text))]


def normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def class_phrases(class_names: list[str]) -> list[list[str]]:
    phrases: set[tuple[str, ...]] = set()
    for class_name in class_names:
        parts = [part for part in class_name.replace("\\", "/").split("/") if part]
        # SUN397 prefixes each top-level folder with a one-letter shard.
        if parts and len(parts[0]) == 1 and parts[0].isalpha():
            parts = parts[1:]
        normalized = [tuple(normalize_words(part)) for part in parts]
        normalized = [part for part in normalized if part]
        if not normalized:
            continue
        full = tuple(token for part in normalized for token in part)
        root = normalized[0]
        for phrase in {full, root}:
            phrases.add(phrase)
            if phrase:
                last = phrase[-1]
                if len(last) > 2 and last.endswith("y") and last[-2] not in "aeiou":
                    plural = last[:-1] + "ies"
                elif last.endswith(("s", "x", "ch", "sh")):
                    plural = last + "es"
                else:
                    plural = last + "s"
                phrases.add((*phrase[:-1], plural))
    return [list(phrase) for phrase in sorted(phrases, key=lambda phrase: (-len(phrase), phrase))]


def build_phrase_trie(phrases: list[list[str]]) -> dict:
    root: dict = {}
    for phrase in phrases:
        node = root
        for token in phrase:
            node = node.setdefault(token, {})
        node[""] = True
    return root


def redact(tokens: list[str], phrase_trie: dict) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        node = phrase_trie
        cursor = index
        longest_match_end = index
        while cursor < len(tokens) and tokens[cursor] in node:
            node = node[tokens[cursor]]
            cursor += 1
            if "" in node:
                longest_match_end = cursor
        if longest_match_end > index:
            index = longest_match_end
        else:
            result.append(tokens[index])
            index += 1
    return result


def safe_segments(tokens: list[str], phrase_trie: dict) -> list[list[str]]:
    """Split at exact class terms so chosen snippets never bridge a redaction gap."""
    segments: list[list[str]] = []
    current: list[str] = []
    index = 0
    while index < len(tokens):
        node = phrase_trie
        cursor = index
        longest_match_end = index
        while cursor < len(tokens) and tokens[cursor] in node:
            node = node[tokens[cursor]]
            cursor += 1
            if "" in node:
                longest_match_end = cursor
        if longest_match_end > index:
            if current:
                segments.append(current)
                current = []
            index = longest_match_end
        else:
            current.append(tokens[index])
            index += 1
    if current:
        segments.append(current)
    return segments


def strip_boilerplate(text: str) -> str:
    cleaned = str(text).strip()
    for _ in range(8):
        updated = PREFIX_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def make_caption(raw: str, phrase_trie: dict) -> tuple[str, bool, int]:
    # Extract a contiguous snippet from a sentence after removing exact class
    # names. Never join text across a removed class phrase; if no readable
    # 5–8-word snippet remains, leave the caption masked instead of padding it.
    source_tokens = words(raw)
    candidates: list[tuple[float, int, list[str]]] = []
    sentence_list = re.split(r"(?<=[.!?])\s+", strip_boilerplate(raw))
    for sentence_index, sentence in enumerate(sentence_list):
        tokens = words(sentence)
        for segment in safe_segments(tokens, phrase_trie):
            left, right = 0, len(segment)
            while left < right and segment[left] in FUNCTION_WORDS:
                left += 1
            while right > left and segment[right - 1] in FUNCTION_WORDS:
                right -= 1
            segment = segment[left:right]
            for start in range(len(segment)):
                if segment[start] in FUNCTION_WORDS:
                    continue
                for end in range(start + 5, min(start + 8, len(segment)) + 1):
                    snippet = segment[start:end]
                    if (snippet[-1] in WEAK_BOUNDARY_WORDS | INCOMPLETE_ENDINGS
                            or snippet[0] in WEAK_BOUNDARY_WORDS):
                        continue
                    if any(token in {"is", "are", "was", "were", "has", "have", "appears", "located", "reads"} for token in snippet):
                        continue
                    content_count = sum(token not in FUNCTION_WORDS for token in snippet)
                    # Prefer readable snippets with fewer grammatical fillers,
                    # then earlier sentences/phrases for reproducibility.
                    score = content_count - 0.35 * (len(snippet) - content_count)
                    candidates.append((score, -sentence_index, snippet))
    if not candidates:
        # Do not stitch fragments across deleted class names. The unmodified
        # source remains available in raw_caption for manual review.
        return "", False, len(source_tokens) - len(redact(source_tokens, phrase_trie))
    _, _, chosen = max(candidates, key=lambda item: (item[0], item[1], len(item[2])))
    caption = " ".join(chosen)
    valid = 5 <= len(chosen) <= 8 and redact(chosen, phrase_trie) == chosen
    return caption, valid, len(source_tokens) - len(redact(source_tokens, phrase_trie))


def main() -> None:
    if not SOURCE.is_file() or not CLASSES.is_file():
        raise FileNotFoundError(f"Required SUN397 caption source or class list is missing under {ROOT}")
    class_names = json.loads(CLASSES.read_text(encoding="utf-8"))
    phrases = class_phrases(class_names)
    phrase_trie = build_phrase_trie(phrases)
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    required = {"image_id", "split", "raw_caption", "final_caption", "caption_valid", "leakage_detected"}
    if not required.issubset(fields):
        raise ValueError(f"Input CSV is missing required columns: {sorted(required - set(fields))}")
    if len(rows) != 39_700 or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 39,700 unique SUN397 rows, found {len(rows)}")

    counts: Counter[str] = Counter()
    for row in rows:
        caption, valid, redacted_count = make_caption(row.get("raw_caption", ""), phrase_trie)
        row["final_caption"] = caption
        row["caption_valid"] = str(valid)
        # Exact class-name phrases were removed from the full raw text before
        # selecting this caption; a short caption is masked for length only.
        row["leakage_detected"] = "False"
        if "lexical_leakage_detected" in fields:
            row["lexical_leakage_detected"] = "False"
        # Old CLIP scores describe the old caption and the old uncalibrated
        # threshold. Clear them instead of presenting them as a new audit.
        for column in ("semantic_similarity_score", "semantic_nearest_class", "semantic_similarity_flag"):
            if column in fields:
                row[column] = ""
        if "caption_policy" in fields:
            row["caption_policy"] = POLICY
        counts["valid" if valid else "masked"] += 1
        counts["class_tokens_removed"] += redacted_count

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT}")
    print(f"Rows: {len(rows):,}; valid: {counts['valid']:,}; masked: {counts['masked']:,}")
    print(f"Direct class-name tokens removed: {counts['class_tokens_removed']:,}")
    print("CLIP semantic fields were cleared; this CSV has not received a new semantic audit.")
    print("Caption text comes from existing raw Florence-2 output; no model was trained or called.")


if __name__ == "__main__":
    main()
