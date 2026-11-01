"""Read saved notebook outputs without executing any notebook/model code."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET = Path(__file__).resolve().parent
NAMES = ["visual-only", "query-only", "candidate-only", "both-fusion", "full-both",
         "full-shuffle-both", "full-unscreened", "full-redacted"]


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def scalar(value):
    if value in {"True", "False"}:
        return value == "True"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", value):
        return float(value)
    return value


def metric_rows(table):
    if not table or not {"experiment", "macro_f1", "accuracy"}.issubset(table[0]):
        return []
    header = table[0]
    if any(len(row) != len(header) for row in table[1:]):
        raise ValueError("Metric table has a truncated or unexpected layout")
    return [{name: scalar(value) for name, value in zip(header, row) if name}
            for row in table[1:]]


def extract(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    data = {"notebook": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tables": [], "images": [], "errors": [], "assignments": {}, "definitions": {},
            "training_epochs": [], "report_metrics": [], "test_metrics": []}
    streams = []
    for cell_index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        data["assignments"][target.id] = ast.unparse(node.value)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                data["definitions"][node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") == "stream":
                streams.append("".join(output.get("text", [])))
            if output.get("output_type") == "error":
                data["errors"].append({"cell": cell_index, "type": output.get("ename"),
                                       "message": output.get("evalue")})
            display = output.get("data", {})
            if "text/html" in display:
                parser = TableParser()
                parser.feed("".join(display["text/html"]))
                for table in parser.tables:
                    data["tables"].append({"cell": cell_index, "output": output_index, "rows": table})
                    rows = metric_rows(table)
                    if "validation_summary = summarize_ablation" in source:
                        data["report_metrics"].extend(rows)
                    elif source.startswith("run_status = ablation_run_status()"):
                        data["test_metrics"].extend(rows)
            if "image/png" in display:
                image_path = TARGET / f"{path.stem}_cell{cell_index}_output{output_index}.png"
                encoded = display["image/png"]
                if isinstance(encoded, list):
                    encoded = "".join(encoded)
                image_path.write_bytes(base64.b64decode(encoded, validate=False))
                data["images"].append(str(image_path.resolve()))
    stream = "\n".join(streams)
    pattern = (r"(?P<mode>[a-z_]+) seed=(?P<seed>\d+) epoch=(?P<epoch>\d+) "
               r"val_f1=(?P<val_f1>[\d.]+) val_loss=(?P<val_loss>[\d.]+) "
               r"q_gate=(?P<q_gate>[\d.]+) c_gate=(?P<c_gate>[\d.]+)"
               r"(?: sem_mix=(?P<sem_mix>[\d.]+))?")
    for match in re.finditer(pattern, stream):
        row = {key: scalar(value) for key, value in match.groupdict().items() if value is not None}
        data["training_epochs"].append(row)
    data["runtime_notes"] = [line for line in stream.splitlines() if any(phrase in line for phrase in (
        "GPU memory:", "Only ", "epochs; seed(s):", "Standard train images:", "Actual train images:",
        "Validation images:", "Standard test images:", "Candidate pool:", "Train on full train:",
        "Training each configuration", "Stored CLIP score range:", "cached=", "Checkpoint", "Resume",
        "Caption input policy:", "Text preprocessing:", "Unscreened query/pool vectors:"))]
    data["worker_warning_counts"] = {phrase: stream.count(phrase) for phrase in (
        "can only test a child process", "Bad file descriptor", "semaphore or lock released too many times",
        "The current process just got forked", "MISSING", "MISMATCH")}
    data["streams_file"] = str((TARGET / f"{path.stem}_streams.txt").resolve())
    Path(data["streams_file"]).write_text(stream, encoding="utf-8")
    assert len(data["report_metrics"]) == 2, path.name
    assert len(data["test_metrics"]) == 1, path.name
    assert not data["errors"], data["errors"]
    assert {row["split"] for row in data["report_metrics"]} == {"val", "test"}
    assert all(row["status"] == "completed" and row["epochs_trained"] == 50
               and row["epoch_limit"] == 50 and row["lr_schedule_epochs"] == 50
               and row["seed"] == 42 for row in data["report_metrics"]), path.name
    epochs = data["training_epochs"]
    observed_epochs = [row["epoch"] for row in epochs]
    assert observed_epochs == sorted(set(observed_epochs)) and all(1 <= epoch <= 50 for epoch in observed_epochs), path.name
    data["missing_epoch_log_lines"] = sorted(set(range(1, 51)) - set(observed_epochs))
    report_test = next(row for row in data["report_metrics"] if row["split"] == "test")
    test = data["test_metrics"][0]
    for key in ("accuracy", "macro_f1", "loss", "mean_query_gate", "mean_candidate_gate"):
        assert abs(report_test[key] - test[key]) < .000051, (path.name, key)
    data["test_correct_count_from_displayed_accuracy"] = round(test["accuracy"] * 1880)
    assert abs(data["test_correct_count_from_displayed_accuracy"] / 1880 - test["accuracy"]) < .00000051
    validation = next(row for row in data["report_metrics"] if row["split"] == "val")
    assert min(epochs, key=lambda row: row["val_loss"])["epoch"] == validation["best_epoch"], path.name
    return data


def main():
    TARGET.mkdir(exist_ok=True)
    records = {name.replace("-", "_"): extract(ROOT / (name + ".ipynb")) for name in NAMES}
    visual = records["visual_only"]["test_metrics"][0]
    for name, data in records.items():
        test = data["test_metrics"][0]
        data["delta_test_accuracy_pp"] = 100 * (test["accuracy"] - visual["accuracy"])
        data["delta_test_macro_f1_pp"] = 100 * (test["macro_f1"] - visual["macro_f1"])
        print(name, "| test:", test, "| epoch count:", len(data["training_epochs"]))
    result = TARGET / "saved_output_evidence.json"
    result.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved evidence:", result)


if __name__ == "__main__":
    main()
