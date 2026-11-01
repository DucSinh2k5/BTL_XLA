import csv
import hashlib
import json

import pytest

from sun397_demo.labels import LabelCatalog, compare, sha256


def bundle(tmp_path, captions=True, conflicting_hash=False):
    root = tmp_path / "bundle"
    root.mkdir()
    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".csv"):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, list(value[0]))
                writer.writeheader()
                writer.writerows(value)
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
    write("class_names.json", ["a/abbey", "b/bedroom"])
    train = {"image_id": "a/abbey/train.jpg", "class_id": 0, "class_name": "a/abbey", "split": "train"}
    val = {"image_id": "b/bedroom/val.jpg", "class_id": 1, "class_name": "b/bedroom", "split": "val"}
    write("provenance/original_run/train_split.csv", [train])
    write("provenance/original_run/validation_split.csv", [val])
    train_sha, val_sha = hashlib.sha256(b"train").hexdigest(), hashlib.sha256(b"val").hexdigest()
    write("candidate_pool.json", [{"position": 0, **{k:v for k,v in train.items() if k != "split"}, "sha256": train_sha}])
    if captions:
        write("captions.csv", [{"image_id": row["image_id"], "split": row["split"], "image_sha256": digest}
              for row, digest in [(train, train_sha), (val, train_sha if conflicting_hash else val_sha)]])
    files = {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in root.rglob("*") if p.is_file()}
    write("manifest.json", {"files": files})
    return root, train_sha, val_sha


def test_hashes_bind_to_official_split_labels_even_when_caption_text_is_absent(tmp_path):
    root, train_sha, val_sha = bundle(tmp_path)
    catalog = LabelCatalog({"mtacs": root})
    catalog.load()
    assert not catalog.errors
    assert catalog.lookup(val_sha) == {"class_id": 1, "label": "b/bedroom", "image_id": "b/bedroom/val.jpg",
                                      "split": "val", "source": "dataset_sha256", "verified": True, "status": "verified"}
    assert catalog.lookup(train_sha)["class_id"] == 0
    assert catalog.lookup("unknown")["label"] is None


def test_visual_bundle_alone_only_claims_labels_for_hashes_it_contains(tmp_path):
    root, train_sha, val_sha = bundle(tmp_path, captions=False)
    catalog = LabelCatalog({"tacs": root})
    catalog.load()
    assert catalog.lookup(train_sha)["verified"]
    assert catalog.lookup(val_sha)["verified"] is False


def test_tampered_annotations_are_not_used(tmp_path):
    root, train_sha, _ = bundle(tmp_path)
    path = root / "class_names.json"
    path.write_text('["b/bedroom", "a/abbey"]')
    catalog = LabelCatalog({"mtacs": root})
    catalog.load()
    assert catalog.errors
    assert catalog.lookup(train_sha)["verified"] is False


def test_identical_content_with_conflicting_annotations_is_ambiguous(tmp_path):
    root, train_sha, _ = bundle(tmp_path, conflicting_hash=True)
    catalog = LabelCatalog({"mtacs": root})
    catalog.load()
    assert catalog.lookup(train_sha)["status"] == "ambiguous"
    assert catalog.lookup(train_sha)["label"] is None


def test_context_label_requires_the_selected_image_and_position(tmp_path):
    root, train_sha, _ = bundle(tmp_path)
    catalog = LabelCatalog({"tacs": root})
    catalog.load()
    result = {"prediction": {"class_id": 1}, "evidence": {"candidate": {"image_id": "a/abbey/train.jpg", "position": 0}}}
    enriched = catalog.enrich("tacs", result, catalog.lookup(train_sha))
    assert enriched["evidence"]["candidate"]["label"] == "a/abbey"
    assert enriched["comparison"]["status"] == "incorrect"
    result["evidence"]["candidate"]["position"] = 1
    with pytest.raises(ValueError):
        catalog.enrich("tacs", result, catalog.lookup(train_sha))


@pytest.mark.parametrize("predicted,verified,status", [(0,True,"correct"),(1,True,"incorrect"),
                                                       (0,False,"matches_reference"),(1,False,"differs_from_reference")])
def test_ground_truth_comparison_distinguishes_user_reference(predicted, verified, status):
    assert compare(predicted, {"class_id": 0, "verified": verified})["status"] == status
