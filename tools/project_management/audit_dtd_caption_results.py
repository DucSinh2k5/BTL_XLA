"""Extract saved DTD caption-method results without executing notebook cells."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .audit_report_readiness import Tables, text
else:
    from audit_report_readiness import Tables, text


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("experiments/completed/caption_ablation/dtd")
MANIFEST = RESULTS / "relocation_manifest.json"
EVIDENCE = Path("reports/caption_ablation/dtd/caption_method_output_evidence.json")
OBSERVED_DATE = "2026-10-24"
RUNS = [
    ("caption-cai-tien-prefix.ipynb", "01_florence_prefix", "florence-prefix", "captions_prefix_control.csv"),
    ("caption-cai-tien-first-valid.ipynb", "02_florence_firstvalid", "florence-firstvalid", "captions_first_valid_control.csv"),
    ("caption-cai-tien-single.ipynb", "03_florence_singlegrounded", "florence-singlegrounded", "captions_single_grounded_control.csv"),
    ("caption-cai-tien-first-multi.ipynb", "04_florence_multigrounded", "florence-multigrounded", "captions_multibeam_grounded.csv"),
    ("qwen_8b.ipynb", "05_qwen3_vl_8b", "qwen3-vl-8b", None),
    ("qwen_32b.ipynb", "06_qwen3_vl_32b", "qwen3-vl-32b", None),
]


def target_path(run):
    return RESULTS / run[1] / f"mtacs-dtd-full-both-{run[2]}-split1-seed42-50ep.ipynb"


def read_notebook(path):
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def output_hash(nb):
    outputs = [cell["outputs"] for cell in nb["cells"] if cell.get("outputs")]
    return hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest()


def saved_tables_and_streams(nb):
    tables, streams, errors = [], [], []
    for index, cell in enumerate(nb["cells"]):
        outputs = cell.get("outputs", [])
        streams.append("".join(text(o.get("text")) for o in outputs if o.get("output_type") == "stream"))
        for output in outputs:
            if output.get("output_type") == "error":
                errors.append({"cell_index": index, "name": output.get("ename"), "message": output.get("evalue")})
            parser = Tables()
            parser.feed(text(output.get("data", {}).get("text/html")))
            tables.extend({"cell_index": index, "rows": table} for table in parser.tables)
    return tables, "\n".join(streams), errors


def metric_rows(tables):
    # Dedicated full-precision tables are distinct from the rounded final report.
    found = {"val": [], "test": []}
    for table in tables:
        rows = table["rows"]
        if not rows or not {"accuracy", "macro_f1", "loss"}.issubset(rows[0]) or "split" in rows[0]:
            continue
        header = rows[0]
        for values in rows[1:]:
            if len(values) != len(header):
                raise ValueError("Metric table has a malformed row")
            row = {key: value for key, value in zip(header, values) if key}
            found["val" if "status" in header else "test"].append(row)
    if any(len(rows) != 1 for rows in found.values()):
        raise ValueError("Expected one dedicated validation and one test metric row")
    return found["val"][0], found["test"][0]


def definition_hashes(nb):
    definitions = {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join("" if line.lstrip().startswith(("!", "%")) else line
                           for line in text(cell.get("source")).splitlines())
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    return definitions


def audit_result(path, csv_path=None):
    raw, nb = read_notebook(path)
    tables, streams, errors = saved_tables_and_streams(nb)
    if errors:
        raise ValueError(f"Saved notebook errors: {errors}")
    val, test = metric_rows(tables)
    if val["status"] != "completed" or any(int(val[key]) != 50 for key in ("epochs_trained", "epoch_limit", "lr_schedule_epochs")):
        raise ValueError("Run did not complete its 50-epoch schedule")
    if any(row["experiment"] != "full_both" or row["seed"] != "42" for row in (val, test)):
        raise ValueError("Unexpected architecture or training seed")
    logs = [{"epoch": int(epoch), "val_f1": float(f1), "val_loss": float(loss)}
            for epoch, f1, loss in re.findall(r"full_both seed=42 epoch=(\d+) val_f1=([\d.]+) val_loss=([\d.]+)", streams)]
    if [row["epoch"] for row in logs] != list(range(1, 51)):
        raise ValueError("Expected 50 distinct consecutive epoch logs")
    best_log = logs[int(val["best_epoch"]) - 1]
    if abs(best_log["val_loss"] - float(val["loss"])) > 0.000051 or best_log["val_loss"] != min(row["val_loss"] for row in logs):
        raise ValueError("Best checkpoint table does not match minimum logged validation loss")
    for table in tables:
        rows = table["rows"]
        if not rows or not {"split", "macro_f1", "accuracy", "loss"}.issubset(rows[0]):
            continue
        for values in rows[1:]:
            row = dict(zip(rows[0], values))
            reference = val if row["split"] == "val" else test
            for metric in ("accuracy", "macro_f1", "loss"):
                if abs(float(reference[metric]) - float(row[metric])) > 0.000051:
                    raise ValueError("Rounded summary disagrees with dedicated metrics")
    metadata = nb["metadata"]["dtd_ablation"]
    runtime_hashes = set(re.findall(r"Caption CSV SHA256:\s*([a-f0-9]{64})", streams))
    declared = metadata.get("caption_csv_sha256", "")
    if re.fullmatch(r"[a-f0-9]{64}", declared):
        runtime_hashes.add(declared)
    if len(runtime_hashes) != 1:
        raise ValueError("Missing or inconsistent caption CSV hash")
    csv_sha = runtime_hashes.pop()
    if csv_path is not None and hashlib.sha256(csv_path.read_bytes()).hexdigest() != csv_sha:
        raise ValueError(f"Local caption CSV hash mismatch: {csv_path}")
    coverage = {}
    pool = None
    for table in tables:
        rows = table["rows"]
        if rows and rows[0] == ["", "count", "valid_captions", "valid_rate", "class_term_rate"]:
            for row in rows[2:]:
                coverage[row[0]] = int(row[2])
        if rows and rows[0] == ["", "", "count", "valid_captions", "valid_rate"]:
            # Parse only the explicit candidate row; query rows use HTML rowspan.
            for row in rows[2:]:
                if row[0] == "candidate_pool":
                    pool = {"count": int(row[2]), "valid_captions": int(row[3]), "valid_rate": float(row[4])}
    if set(coverage) != {"train", "val", "test"}:
        raise ValueError("Missing per-split caption coverage table")
    return {
        "method": metadata["caption_method"], "name": metadata["caption_display_name"],
        "source_notebook": path.relative_to(ROOT).as_posix(),
        "notebook_sha256": hashlib.sha256(raw).hexdigest(), "saved_outputs_sha256": output_hash(nb),
        "csv_sha256": csv_sha, "csv_sha_verified_against_local_input": csv_path is not None,
        "csv_provenance": "runtime/metadata and local bytes" if csv_path else "runtime stdout only; Qwen CSV not independently provided locally",
        "metadata": metadata, "validation_metrics": val, "test_metrics": test,
        "correct_count_inferred_from_displayed_accuracy": round(float(test["accuracy"]) * 1880),
        "training_logs": logs, "saved_error_outputs": errors,
        "runtime_lines": [line for line in streams.splitlines() if re.match(
            r"^(DEVICE|GPU memory|Standard train|Actual train|Validation images|Standard test|Candidate pool|Train on full train|Caption CSV SHA256|Caption CSV|TACS selected|Full candidate)", line)],
        "stored_valid_by_split": coverage, "stored_valid_total": sum(coverage.values()),
        "candidate_pool_coverage": pool,
        "tables": [t for t in tables if t["rows"] and any(key in t["rows"][0] for key in ("macro_f1", "best_epoch", "valid_captions"))],
        "definitions_sha256": definition_hashes(nb),
    }


def prepare_relocation():
    if (ROOT / MANIFEST).exists():
        raise FileExistsError("Preserve the existing relocation manifest")
    entries = []
    for run in RUNS:
        old, new = ROOT / "notebooks" / run[0], ROOT / target_path(run)
        if not old.resolve().is_relative_to(ROOT.resolve()) or not new.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Relocation escapes project root")
        if new.exists():
            raise FileExistsError(new)
        raw, nb = read_notebook(old)
        entries.append({"old": old.relative_to(ROOT).as_posix(), "new": new.relative_to(ROOT).as_posix(),
                        "sha256_before": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                        "saved_outputs_sha256_before": output_hash(nb)})
    target = ROOT / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"date": OBSERVED_DATE, "notebook_content_changed": False,
                                 "verified_after_move": False, "files": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(entries)} byte-preserving file relocations; no notebook moved or edited")


def update_evidence():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        raw, nb = read_notebook(ROOT / entry["new"])
        if hashlib.sha256(raw).hexdigest() != entry["sha256_before"] or output_hash(nb) != entry["saved_outputs_sha256_before"] or len(raw) != entry["bytes"]:
            raise ValueError(f"Notebook preservation check failed: {entry['new']}")
        if (ROOT / entry["old"]).exists():
            raise ValueError("Old filename still exists after relocation")
    evidence = json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8"))
    original_comparison = evidence.get("four_florence_source_ast_comparison", evidence["source_ast_comparison"])
    runs = [audit_result(ROOT / target_path(run), ROOT / "data/captions/dtd/grounded_controls" / run[3] if run[3] else None) for run in RUNS]
    definitions = [row["definitions_sha256"] for row in runs]
    shared = set.intersection(*(set(row) for row in definitions))
    identical = sorted(name for name in shared if len({row[name] for row in definitions}) == 1)
    visual_path = ROOT / "experiments/completed/text_context_ablation/dtd/visual-only.ipynb"
    raw, visual = read_notebook(visual_path)
    val, test = metric_rows(saved_tables_and_streams(visual)[0])
    evidence.update({
        "date": OBSERVED_DATE, "schema": "dtd_caption_method_outputs_v2", "new_runs": runs,
        "four_florence_source_ast_comparison": original_comparison,
        "source_ast_comparison": {"scope": "all six new caption-method runs", "shared_definitions": len(shared),
                                  "identical_definitions": len(identical), "different_definitions": sorted(shared - set(identical)),
                                  "identical_definition_names": identical},
        "visual_control_reference": {"source_notebook": visual_path.relative_to(ROOT).as_posix(),
                                     "notebook_sha256": hashlib.sha256(raw).hexdigest(),
                                     "validation_metrics": val, "test_metrics": test,
                                     "comparison_status": "Historical architecture control with visual context; not a caption method or No-Context."},
        "relocation_manifest": MANIFEST.as_posix(),
        "limits": ["One seed and one official split only; no statistical superiority established.",
                   "No full per-image prediction CSVs or checkpoint/pool artifacts independently audited.",
                   "Source equality does not prove loaded model-weight hashes or every runtime package identical.",
                   "Captioner/prompt/extraction policy and natural text availability differ across families; not a matched-mask scale-only test.",
                   "Florence CSV hashes independently match local files; Qwen CSV hashes are recorded from saved runtime stdout only.",
                   "Six notebooks were relocated byte-for-byte; no cells executed, outputs cleared, or training rerun."]})
    (ROOT / EVIDENCE).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["verified_after_move"] = True
    (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for run in runs:
        print(run["name"], "best_epoch=" + run["validation_metrics"]["best_epoch"], "test=" + str(run["test_metrics"]))
    print("Verified notebook preservation and updated evidence; common AST definitions:", len(identical), "/", len(shared))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-relocation", action="store_true")
    args = parser.parse_args()
    prepare_relocation() if args.prepare_relocation else update_evidence()
