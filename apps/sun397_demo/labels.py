"""Verified dataset labels for display only; never supplied to model inference."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unknown(status="unknown"):
    return {"class_id": None, "label": None, "image_id": None, "split": None,
            "source": None, "verified": False, "status": status}


class LabelCatalog:
    def __init__(self, bundles):
        self.bundles = bundles
        self.classes = []
        self.by_hash = {}
        self.pools = {}
        self.errors = []
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        classes, records, hashes, pools, errors = None, {}, {}, {}, []
        for model, root in self.bundles.items():
            root = Path(root)
            try:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                def verified(name):
                    path = root / name
                    entry = manifest["files"][name]
                    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                        raise ValueError(f"Label metadata checksum mismatch: {name}")
                    return path
                names = json.loads(verified("class_names.json").read_text(encoding="utf-8"))
                if not names or len(set(names)) != len(names):
                    raise ValueError("Invalid class list")
                if classes is not None and classes != names:
                    raise ValueError("Class orders differ across bundles")
                local_records = {}
                for filename in ("train_split.csv", "validation_split.csv", "test_split.csv"):
                    rel = "provenance/original_run/" + filename
                    if rel not in manifest["files"]:
                        continue
                    with verified(rel).open(encoding="utf-8-sig", newline="") as handle:
                        for row in csv.DictReader(handle):
                            class_id, image_id = int(row["class_id"]), row["image_id"]
                            if not 0 <= class_id < len(names) or names[class_id] != row["class_name"]:
                                raise ValueError("Split class mapping mismatch")
                            if image_id in local_records or image_id.rsplit("/", 1)[0] != names[class_id]:
                                raise ValueError("Split image identity mismatch")
                            local_records[image_id] = {"class_id": class_id, "label": names[class_id],
                                                       "image_id": image_id, "split": row["split"]}
                pool = json.loads(verified("candidate_pool.json").read_text(encoding="utf-8"))
                local_pool, local_hashes = {}, []
                for position, item in enumerate(pool):
                    class_id = int(item["class_id"])
                    if (item["position"] != position or not 0 <= class_id < len(names)
                            or names[class_id] != item["class_name"]):
                        raise ValueError("Pool class/order mismatch")
                    record = {"class_id": class_id, "label": names[class_id],
                              "image_id": item["image_id"], "split": "train"}
                    if item["image_id"] in local_records and local_records[item["image_id"]] != record:
                        raise ValueError("Candidate is not in the recorded training split")
                    local_records[item["image_id"]] = record
                    local_pool[item["image_id"]] = dict(record, position=position)
                    local_hashes.append((item["sha256"], item["image_id"]))
                if "captions.csv" in manifest["files"]:
                    with verified("captions.csv").open(encoding="utf-8-sig", newline="") as handle:
                        seen = set()
                        for row in csv.DictReader(handle):
                            image_id = row["image_id"]
                            if (image_id in seen or image_id not in local_records
                                    or row["split"] != local_records[image_id]["split"]):
                                raise ValueError("Caption hash/split binding mismatch")
                            seen.add(image_id)
                            local_hashes.append((row["image_sha256"], image_id))
                for image_id, record in local_records.items():
                    if image_id in records and records[image_id] != record:
                        raise ValueError("Conflicting dataset annotations")
                for digest, image_id in local_hashes:
                    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                        raise ValueError("Invalid image content digest")
                classes = names
                records.update(local_records)
                pools[model] = local_pool
                for digest, image_id in local_hashes:
                    hashes.setdefault(digest, set()).add(image_id)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"{model}: {exc}")
                LOG.error("Cannot load label catalog: %s", errors[-1])
        self.classes = classes or []
        self.by_hash = {digest: [records[image_id] for image_id in sorted(ids)] for digest, ids in hashes.items()}
        self.pools, self.errors, self.loaded = pools, errors, True

    def lookup(self, digest):
        matches = self.by_hash.get(digest, [])
        if not matches:
            return unknown()
        if len({item["class_id"] for item in matches}) != 1:
            return unknown("ambiguous")
        record = dict(matches[0])
        if len(matches) > 1:
            record["image_id"] = None
            if len({item["split"] for item in matches}) > 1:
                record["split"] = None
        return dict(record, source="dataset_sha256", verified=True, status="verified")

    def reference(self, digest, class_id=None):
        actual = self.lookup(digest)
        if class_id is None:
            return actual
        if not 0 <= class_id < len(self.classes):
            raise ValueError("Nhãn đối chiếu không thuộc danh sách lớp SUN397.")
        if actual["verified"]:
            if actual["class_id"] != class_id:
                raise ValueError("Nhãn nhập không khớp ground truth SUN397 đã xác minh của ảnh.")
            return actual
        return {"class_id": class_id, "label": self.classes[class_id], "image_id": None, "split": None,
                "source": "user_provided", "verified": False, "status": "user_provided"}

    def enrich(self, model, result, ground_truth):
        # Deliberately performed in the API after prediction; no labels reach the worker.
        candidate = result.get("evidence", {}).get("candidate", {})
        record = self.pools.get(model, {}).get(candidate.get("image_id"))
        if record is None or candidate.get("position") != record["position"]:
            raise ValueError("Selected candidate does not match its recorded pool identity")
        candidate.update(class_id=record["class_id"], label=record["label"], class_name=record["label"],
                         label_source="saved_candidate_pool")
        result["ground_truth"] = ground_truth
        result["comparison"] = compare(result["prediction"]["class_id"], ground_truth)
        return result


def compare(prediction_id, reference):
    if reference["class_id"] is None:
        return {"status": "unavailable", "is_match": None, "verified_ground_truth": False}
    match = int(prediction_id) == reference["class_id"]
    if reference["verified"]:
        status = "correct" if match else "incorrect"
    else:
        status = "matches_reference" if match else "differs_from_reference"
    return {"status": status, "is_match": match, "verified_ground_truth": reference["verified"]}
