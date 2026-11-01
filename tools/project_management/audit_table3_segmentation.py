"""Read saved ISIC segmentation results; never execute cells or load checkpoints."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .audit_report_readiness import audit, text
else:
    from audit_report_readiness import audit, text


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tacs_isic_segement_448.ipynb"
DESTINATION = ROOT / "reports/paper_tables/table_03_segmentation_evidence.json"
CONFIG_KEYS = {
    "SEED", "DINOV3_MODEL_NAME", "FEATURE_DIM", "PATCH_SIZE", "IMAGE_SIZE", "EPOCHS",
    "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "WARMUP_EPOCHS", "PATIENCE",
    "CANDIDATE_POOL_FRACTION", "TACS_GUMBEL_TAU", "TACS_POLICY_WEIGHT", "TACS_ENTROPY_WEIGHT",
    "NUM_CROSS_ATTENTION_LAYERS", "CROSS_ATTENTION_HEADS", "THRESHOLD",
    "DICE_LOSS_WEIGHT", "BCE_LOSS_WEIGHT",
}


def stream_text(cell):
    return "".join(text(output.get("text")) for output in cell.get("outputs", [])
                   if output.get("output_type") == "stream")


def parse_test_metrics(stream):
    result = {}
    for name in ("dice", "iou"):
        matches = re.findall(rf"^test_{name}:\s*(\d+\.\d+)\s*\((\d+\.\d+)%\)\s*$", stream, re.M)
        if len(matches) != 1:
            raise ValueError(f"Expected one explicitly printed test_{name}; do not use paper constants or example rows")
        value, percent = matches[0]
        if not 0 <= float(value) <= 1 or abs(float(value) * 100 - float(percent)) > 0.0051:
            raise ValueError("Inconsistent displayed metric/percentage")
        result[f"test_{name}"] = {"value_displayed": float(value), "decimal_places": len(value.split('.')[1]),
                                  "percent_displayed": float(percent)}
    return result


def extract_evidence(path=SOURCE):
    path = path.resolve()
    raw = path.read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    generic = audit(path)
    if generic["errors"]:
        raise ValueError("Notebook contains saved error outputs")
    cells = nb["cells"]
    test_cells = [(i, cell) for i, cell in enumerate(cells)
                  if cell.get("cell_type") == "code" and
                  "test_metrics, test_rows = evaluate_test_with_rows(test_loader)" in text(cell.get("source"))]
    if len(test_cells) != 1:
        raise ValueError("Expected one identifiable final test evaluation cell")
    index, test_cell = test_cells[0]
    test_stream = stream_text(test_cell)
    metrics = parse_test_metrics(test_stream)
    combined = "\n".join(stream_text(cell) for cell in cells)
    config = {}
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                           for line in text(cell.get("source")).splitlines())
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in CONFIG_KEYS:
                        config[target.id] = ast.literal_eval(node.value)
    logs = []
    for line in combined.splitlines():
        match = re.match(r"^Epoch (\d+)/(\d+) \|", line)
        if match:
            logs.append({"epoch": int(match[1]), "epoch_limit": int(match[2]), "line": line,
                         **{key: float(value) for key, value in re.findall(
                             r"\b(train_loss|train_dice|train_iou|val_loss|val_dice|val_iou)=([\d.]+)", line)}})
    stopping = re.findall(r"Early stopping at epoch (\d+)\. Best val dice=([\d.]+) at epoch (\d+)\.", combined)
    if len(stopping) != 1:
        raise ValueError("Expected the saved early-stopping summary")
    stopped, best_dice, best = stopping[0]
    stopped, best = int(stopped), int(best)
    if [row["epoch"] for row in logs] != list(range(1, stopped + 1)):
        raise ValueError("Training logs are incomplete or duplicated")
    best_row = logs[best - 1]
    if best_row["val_dice"] != float(best_dice) or best_row["val_dice"] != max(row["val_dice"] for row in logs):
        raise ValueError("Best validation Dice disagrees with training logs")
    checkpoint = re.findall(r"^Loaded checkpoint:\s*(.+)$", test_stream, re.M)
    if len(checkpoint) != 1 or not checkpoint[0].strip().endswith("best_val_checkpoint.pt"):
        raise ValueError("Final test did not report loading the best validation checkpoint")
    sizes = {}
    for label, key in (("Train", "train"), ("Val", "val"), ("Test", "test")):
        found = re.findall(rf"^{label} images:\s*([\d,]+)\s*$", combined, re.M)
        if len(found) != 1:
            raise ValueError(f"Missing runtime {label} image count")
        sizes[key] = int(found[0].replace(",", ""))
    pools = re.findall(r"^Candidate pool:\s*([\d,]+)/([\d,]+) train images \((\d+)%\)\s*$", combined, re.M)
    if len(pools) != 1:
        raise ValueError("Missing runtime candidate pool count")
    pool_size, pool_train, pool_percent = (int(value.replace(",", "")) for value in pools[0])
    if pool_train != sizes["train"] or pool_size != round(pool_train * config["CANDIDATE_POOL_FRACTION"]) or pool_percent != round(config["CANDIDATE_POOL_FRACTION"] * 100):
        raise ValueError("Candidate pool counts disagree with source configuration")
    paper = next((ROOT / "docs/paper").glob("*.pdf"))
    return {
        "schema": "tacs_table3_segmentation_saved_outputs_v1", "audit_date": "2026-10-24",
        "source_notebook": path.relative_to(ROOT).as_posix(),
        "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "saved_outputs_sha256": generic["saved_outputs_sha256"],
        "source_notebook_modified": False, "cells_executed_by_auditor": False,
        "table_mapping": {"table": 3, "row_family": "TACS", "dataset": "ISIC 2017", "columns": ["Dice", "IoU"],
                          "report_label": "TACS reimplementation variant, 448, final-token cross-attention + DPT-lite",
                          "exact_paper_reproduction": False},
        "metrics": metrics, "test_loss": None,
        "metric_definition_from_source": "Binary foreground Dice/IoU at sigmoid probability > 0.5, computed per resized image and averaged across test images; not soft metrics or global pixel aggregation.",
        "test_cell_index_zero_based": index, "test_stdout": test_stream,
        "split_sizes_from_runtime": sizes, "candidate_pool_size_from_runtime": pool_size,
        "configuration_from_saved_source": config,
        "training": {"epoch_limit": config["EPOCHS"], "epochs_observed": stopped, "best_epoch": best,
                     "selection_criterion": "maximum validation hard Dice", "best_validation_row_displayed": best_row,
                     "checkpoint_loaded_from_stdout": checkpoint[0].strip(), "logs": logs},
        "paper_reference": {"path": paper.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
                            "table_pdf_page_one_based": 6, "implementation_pdf_page_one_based": 5,
                            "isic2017_tacs": {"dice_percent": 85.0, "iou_percent": 76.5},
                            "checked_by": "local PDF text extraction and visual inspection of Table 3"},
        "limits": ["Test Dice/IoU are rounded stdout values, not full-precision metrics.json values.",
                   "Only seed 42 observed; paper reports three-seed averages.",
                   "Resolution 448 rather than paper 256; final-token cross-attention and DPT-lite rather than blockwise cross-attention and lightweight DPT.",
                   "Test masks and predictions are evaluated at resized 448 resolution in source.",
                   "No standalone metrics.json, checkpoint, complete prediction CSV or split/patient manifests independently audited.",
                   "Training code masks exact self candidates; patient-associated exclusions are not independently verified.",
                   "Only five displayed per-image predictions; their repeated context ID does not establish whole-test selector collapse.",
                   "No same-protocol No-Context/Frozen baseline results supplied with this run."],
        "generic_output_audit": generic,
    }


def main():
    evidence = extract_evidence()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in ("source_notebook", "metrics", "split_sizes_from_runtime", "table_mapping")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
