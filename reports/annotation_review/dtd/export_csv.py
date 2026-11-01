"""Serialize Artifact Tool-authored string cells with Python's standard CSV writer."""
import csv
import hashlib
import json
from pathlib import Path

from validate_annotations import HEADER, TEACHER_ID, validate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = ROOT / "caption_references.csv"
backup = HERE / "original_caption_references.csv"
assert source.read_bytes() == backup.read_bytes(), "Source changed; do not overwrite it"
validate()
values = json.loads((HERE / "workbook_values.json").read_text(encoding="utf-8"))
captions = json.loads((HERE / "captions.json").read_text(encoding="utf-8"))
with backup.open(encoding="utf-8-sig", newline="") as handle:
    originals = list(csv.DictReader(handle))
assert values[0] == HEADER and len(values) == 189
for index, (cells, original) in enumerate(zip(values[1:], originals), 1):
    assert len(cells) == 9 and all(isinstance(value, str) for value in cells)
    assert cells[:3] == [original[key] for key in HEADER[:3]]
    assert cells[3:] == [captions[str(index)], "teacher", "approved", "ducsinh", TEACHER_ID, ""]
temporary = source.with_suffix(".csv.annotation.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    csv.writer(handle, lineterminator="\r\n").writerows(values)
with temporary.open(encoding="utf-8", newline="") as handle:
    assert list(csv.reader(handle)) == values
assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(backup.read_bytes()).digest()
temporary.replace(source)
validate(final=True)
