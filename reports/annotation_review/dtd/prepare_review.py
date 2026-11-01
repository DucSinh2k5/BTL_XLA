"""Prepare blind inspection sheets without changing the source annotation CSV."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "caption_references.csv"
DATASET = ROOT / "archive (23)" / "dtd"
OUTPUT = Path(__file__).resolve().parent
HEADER = ["review_id", "image_id", "draft_caption_unverified", "target_caption", "caption_source",
          "review_status", "reviewer_id", "teacher_model", "notes"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == HEADER
        rows = list(reader)
    assert len(rows) == 188 and len({row["image_id"] for row in rows}) == 188
    assert len({row["review_id"] for row in rows}) == 188
    counts = Counter(row["image_id"].split("/")[0] for row in rows)
    assert len(counts) == 47 and set(counts.values()) == {4}
    official_train = set((DATASET / "labels" / "train1.txt").read_text().splitlines())
    assert all(row["image_id"] in official_train for row in rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    backup = OUTPUT / "original_caption_references.csv"
    if not backup.exists():
        shutil.copy2(SOURCE, backup)
    else:
        assert SOURCE.read_bytes() == backup.read_bytes(), "Source changed since preparation; do not overwrite the original backup"
    manifest, images = [], []
    image_root = (DATASET / "images").resolve()
    for index, row in enumerate(rows, 1):
        parts = PurePosixPath(row["image_id"]).parts
        assert len(parts) == 2 and ".." not in parts
        path = image_root.joinpath(*parts).resolve()
        assert path.is_relative_to(image_root) and path.is_file()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        images.append(image)
        manifest.append({"index": index, "review_id": row["review_id"], "image_id": row["image_id"],
            "path": str(path), "sha256": sha256(path), "size": list(image.size), "official_split": "train1"})
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    for start in range(0, len(rows), 12):
        sheet = Image.new("RGB", (1020, 1264), "#f4f4f4")
        draw = ImageDraw.Draw(sheet)
        for offset, image in enumerate(images[start:start + 12]):
            x, y = (offset % 3) * 340, (offset // 3) * 316
            draw.text((x + 12, y + 5), f"#{start + offset + 1:03d}", fill="black", font=font)
            thumbnail = ImageOps.contain(image, (324, 272), method=Image.Resampling.LANCZOS)
            sheet.paste(thumbnail, (x + 8 + (324 - thumbnail.width) // 2,
                                   y + 35 + (272 - thumbnail.height) // 2))
        sheet.save(OUTPUT / f"sheet_{start // 12 + 1:02d}.jpg", quality=96)
    (OUTPUT / "image_manifest.json").write_text(json.dumps({"source_sha256": sha256(SOURCE),
        "dataset": str(DATASET.resolve()), "rows": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "classes": len(counts), "images_readable": len(images),
                      "official_train_only": True, "sheets": (len(rows) + 11) // 12,
                      "source_sha256": sha256(SOURCE)}, indent=2))


if __name__ == "__main__":
    prepare()
