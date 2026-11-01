"""Read notebook sources/outputs to inventory report evidence; execute no cells."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "reports/paper_readiness"
CONFIG_NAMES = {
    "SEED", "TRAIN_SEEDS", "DTD_SPLIT", "DINOV3_MODEL_NAME", "NUM_CLASSES", "IMAGE_SIZE", "IMG_SIZE",
    "EPOCHS", "NUM_EPOCHS", "BATCH_SIZE", "VAL_FRACTION", "TRAIN_ON_FULL_TRAIN", "CONTEXT_TOP_K",
    "CANDIDATE_POOL_FRACTION", "TACS_CANDIDATES_PER_QUERY", "EXPERIMENT_MODE", "PATIENCE",
    "SUPCON_WEIGHT", "LABEL_SMOOTHING", "ENABLE_TEXT_CONTEXT", "TACS_POLICY_WEIGHT",
}


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def text(value):
    return value if isinstance(value, str) else "".join(value or [])


def source_evidence(source, cell_index):
    assignments = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Notebook magics are not Python statements; keep line numbers stable.
        sanitized = "\n".join("" if re.match(r"^\s*[!%]", line) else line for line in source.splitlines())
        try:
            tree = ast.parse(sanitized)
            source = sanitized
        except SyntaxError:
            return assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in CONFIG_NAMES:
                assignments.append({"name": target.id, "expression": ast.get_source_segment(source, node.value),
                                    "cell_index": cell_index, "line": node.lineno})
    return assignments


def audit(path):
    raw = path.read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    sources = []
    records = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
               "configurations": [], "metric_lines": [], "tables": [], "errors": [],
               "runtime_lines": [], "cells_with_outputs": 0, "image_outputs": 0}
    stored_outputs = [cell.get("outputs", []) for cell in nb["cells"] if cell.get("outputs")]
    records["saved_outputs_sha256"] = hashlib.sha256(json.dumps(stored_outputs, sort_keys=True).encode()).hexdigest()
    metric_re = re.compile(r"(?:test[_ ](?:accuracy|macro[_ -]?f1|top.?1|loss|dice|iou|soft[_ ]dice|soft[_ ]iou)|Test (?:top-1 accuracy|macro-F1|accuracy|Dice|IoU)|best (?:validation|epoch)|Early stopping)", re.I)
    runtime_re = re.compile(r"^(?:DEVICE|Device|GPU memory|Official train|Standard train|Actual train|Training images|Validation images|Standard test|Test images|Candidate pool|Train on full train|Seed|SEED)", re.I)
    for index, cell in enumerate(nb["cells"]):
        source = text(cell.get("source"))
        sources.append(source)
        records["configurations"].extend(source_evidence(source, index))
        outputs = cell.get("outputs", [])
        records["cells_with_outputs"] += bool(outputs)
        # IOPub may split a single print across stream messages: concatenate by stream name.
        streams = {}
        plain = []
        for output in outputs:
            if output.get("output_type") == "stream":
                name = output.get("name", "stdout")
                streams[name] = streams.get(name, "") + text(output.get("text"))
            if output.get("output_type") == "error":
                records["errors"].append({"cell_index": index, "type": output.get("ename"), "message": output.get("evalue")})
            data = output.get("data", {})
            if "text/plain" in data:
                plain.append(text(data["text/plain"]))
            records["image_outputs"] += "image/png" in data or "image/jpeg" in data
            if "text/html" in data:
                parser = Tables()
                parser.feed(text(data["text/html"]))
                for table in parser.tables:
                    header = " ".join(" ".join(row) for row in table[:2])
                    if re.search(r"accuracy|macro_f1|dice|iou|test_|best_epoch", header, re.I):
                        records["tables"].append({"cell_index": index, "rows": table[:12]})
        for output_text in list(streams.values()) + plain:
            for line in output_text.replace("\r", "\n").splitlines():
                if "train_loss=" not in line and metric_re.search(line):
                    records["metric_lines"].append({"cell_index": index, "text": line[:800]})
                if runtime_re.search(line):
                    records["runtime_lines"].append({"cell_index": index, "text": line[:800]})
    full = "\n".join(sources)
    records["source_indicators"] = {
        "val_copy_of_train": bool(re.search(r"val_df\s*=\s*train_df(?:\.copy\(\))?", full)),
        "val_copy_of_test": bool(re.search(r"val_df\s*=\s*test_df(?:\.copy\(\))?", full)),
        "gumbel_softmax_present": "gumbel_softmax" in full,
        "policy_loss_present": "policy_loss" in full,
        "candidate_self_exclusion_text_present": any(name in full for name in ("excluded_positions", "exclude_self", "forbidden_indices")),
        "full_train_monitor_subset_pattern": "train_df = full_train_df.reset_index(drop=True)" in full and "val_df = monitor_val_df.reset_index(drop=True)" in full,
    }
    records["source_caution"] = "Static indicators/config declarations are review leads, not proof of the actual executed runtime or complete method equivalence."
    return records


def main():
    paths = sorted(set((ROOT / "notebooks").rglob("*.ipynb")) | set((ROOT / "experiments").rglob("*.ipynb")))
    rows = [audit(path) for path in paths]
    DESTINATION.mkdir(parents=True, exist_ok=True)
    output = DESTINATION / "notebook_evidence_inventory.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Read {len(rows)} notebooks without executing cells; wrote {output.relative_to(ROOT)}")
    for row in rows:
        if "/01_paper_reproduction/" not in row["path"] and "/query_text/sun397/" not in row["path"]:
            continue
        final = [item["text"] for item in row["metric_lines"] if "test" in item["text"].lower()]
        unique = list(dict.fromkeys(final))
        print(row["path"], "| outputs", row["cells_with_outputs"], "| errors", [e["type"] for e in row["errors"]],
              "|", " / ".join(unique[-7:]))


if __name__ == "__main__":
    main()
