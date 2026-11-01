"""Check image-grounded annotation alignment, lexical rules and the training guard."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEADER = ["review_id", "image_id", "draft_caption_unverified", "target_caption", "caption_source",
          "review_status", "reviewer_id", "teacher_model", "notes"]
TEACHER_ID = "openai-codex:model-id-unavailable"
# Word-boundary inflections, not substring tests (e.g. 'material' must not match 'mat').
ROOTS = re.compile(r"\b(?:bands?|banded|banding|blotch(?:y|es|ed)?|braid(?:s|ed|ing)?|"
    r"bubbl(?:e|es|ed|ing|y)|bump(?:s|y|ed|ing)?|check(?:er(?:ed|s|board)?|ered)?|chequer(?:ed|s)?|"
    r"cobweb(?:s|bed)?|web(?:s|bed|bing)?|crack(?:s|ed|ing)?|crosshatch(?:ed|es|ing)?|"
    r"crystal(?:s|line|lized|lization)?|dot(?:s|ted|ting)?|fibr(?:e|es|ous)|fiber(?:s)?|"
    r"fleck(?:s|ed)?|freckl(?:e|es|ed)|frill(?:s|y|ed)?|gauz(?:e|y)|grid(?:s)?|"
    r"groov(?:e|es|ed|ing)|honeycomb(?:s|ed)?|interlac(?:e|es|ed|ing)|knit(?:s|ted|ting)?|"
    r"lac(?:e|es|y|elike)|lin(?:e|es|ed|ing)|marbl(?:e|es|ed|ing)|mat(?:s|ted|ting)?|"
    r"mesh(?:es|ed|ing)?|paisley|perforat(?:e|ed|es|ing|ion|ions)|pit(?:s|ted|ting)?|"
    r"pleat(?:s|ed|ing)?|polka|por(?:e|es|ous)|pothol(?:e|es|ed)|scal(?:e|es|ed|y|ing)|"
    r"smear(?:s|ed|ing)?|spiral(?:s|led|ed|ling|ing)?|sprinkl(?:e|es|ed|ing)|"
    r"stain(?:s|ed|ing)?|stratif(?:ied|y|ication)|strat(?:a|um)|strip(?:e|es|ed|ing|s)|"
    r"stud(?:s|ded|ding)?|swirl(?:s|ed|ing)?|vein(?:s|ed|ing)?|waffl(?:e|es|ed|ing)|"
    r"weav(?:e|es|ing)|woven|wrinkl(?:e|es|ed|ing)|zigzag(?:s|ged|ging)?)\b", re.I)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == HEADER
        rows = list(reader)
    assert all(None not in row and all(value is not None for value in row.values()) for row in rows)
    return rows


def validate(final=False):
    originals = read_rows(HERE / "original_caption_references.csv")
    captions = json.loads((HERE / "captions.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "image_manifest.json").read_text(encoding="utf-8"))
    assert set(captions) == {str(index) for index in range(1, 189)}
    assert len(originals) == len(manifest["rows"]) == 188
    names = {row["image_id"].split("/")[0] for row in originals}
    assert len(names) == 47
    errors, categories = [], defaultdict(list)
    for index, (original, image) in enumerate(zip(originals, manifest["rows"]), 1):
        assert image["index"] == index and image["review_id"] == original["review_id"]
        assert image["image_id"] == original["image_id"] and image["official_split"] == "train1"
        assert hashlib.sha256(Path(image["path"]).read_bytes()).hexdigest() == image["sha256"]
        caption = captions[str(index)]
        words = re.findall(r"[A-Za-z0-9]+", caption)
        hits = ROOTS.findall(caption)
        if not 5 <= len(words) <= 8 or hits or not re.fullmatch(r"[A-Za-z0-9 ]+", caption):
            errors.append({"row": index, "caption": caption, "count": len(words), "forbidden": hits})
        if re.match(r"^(a|an|the|this)\b", caption, flags=re.I):
            errors.append({"row": index, "boilerplate": caption})
        normalized = " " + caption.lower() + " "
        assert not any(" " + name.replace("-", " ") + " " in normalized for name in names)
        categories[original["image_id"].split("/")[0]].append(caption)
    assert not errors, json.dumps(errors, indent=2)
    assert len(set(captions.values())) == 188
    assert all(len(values) == len(set(values)) == 4 for values in categories.values())
    summary = {"rows": 188, "unique_captions": 188, "classes": 47, "per_class": 4,
        "word_counts": dict(Counter(len(re.findall(r"[A-Za-z0-9]+", text)) for text in captions.values())),
        "direct_class_or_root_hits": 0, "unreadable_images": 0, "image_bytes_verified": 188,
        "official_train1_only": True, "header_and_row_order_preserved": None,
        "exact_model_id_available": False, "teacher_model_recorded": TEACHER_ID,
        "annotation_origin": "AI visual annotation requested by ducsinh",
        "reviewer_id_semantics": "ducsinh is the requesting owner who authorized AI annotation, not an assertion of per-image human inspection",
        "approved_semantics": "AI annotation completed and structurally validated, not independently human verified",
        "independent_human_review_performed": False, "human_gold_standard": False,
        "semantic_label_leakage_absence_certified": False,
        "source_sha256": manifest["source_sha256"]}
    if final:
        output = ROOT / "caption_references.csv"
        completed = read_rows(output)
        assert len(completed) == 188
        for index, (old, new) in enumerate(zip(originals, completed), 1):
            for key in ("review_id", "image_id", "draft_caption_unverified"):
                assert old[key] == new[key], (index, key)
            assert new["target_caption"] == captions[str(index)]
            assert new["caption_source"] == "teacher" and new["review_status"] == "approved"
            assert new["reviewer_id"] == "ducsinh" and new["teacher_model"] == TEACHER_ID and new["notes"] == ""
        sys.path.insert(0, str(ROOT))
        from dtd_caption_finetune import approved_reference_pairs, split_reference_pairs
        records = [{"image_id": row["image_id"], "image_path": row["path"], "split": "train"}
                   for row in manifest["rows"]]
        approved = approved_reference_pairs(output, records, minimum=50)
        fit, holdout = split_reference_pairs(approved)
        assert len(approved) == 188 and len(fit) == 141 and len(holdout) == 47
        summary.update(header_and_row_order_preserved=True, training_reference_guard_passed=True,
                       fit_images=len(fit), holdout_images=len(holdout),
                       output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        report = ROOT / "caption_references.annotation_report.json"
        report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    validate(final="--final" in sys.argv)
