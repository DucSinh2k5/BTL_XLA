"""Compare saved ISIC visual/text segmentation evidence; execute no notebook cells."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .audit_report_readiness import Tables, audit, text
    from .audit_table3_segmentation import CONFIG_KEYS, extract_evidence, parse_test_metrics, stream_text
else:
    from audit_report_readiness import Tables, audit, text
    from audit_table3_segmentation import CONFIG_KEYS, extract_evidence, parse_test_metrics, stream_text


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "isic-segmen-captions.ipynb"
TEMPLATE = ROOT / "notebooks/02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/mtacs-isic2017-448-qwen3-vl-8b-full-both.ipynb"
DESTINATION = ROOT / "reports/paper_tables/isic_visual_vs_qwen8b_comparison_evidence.json"


def source_identity(nb):
    definitions, config, expressions, constants = {}, {}, {}, {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                           for line in text(cell.get("source")).splitlines())
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in CONFIG_KEYS | {"NUM_WORKERS", "DECODER_DROPOUT", "GRAD_CLIP_NORM", "TACS_SELECTOR_DROPOUT", "TACS_SELECTOR_TEMPERATURE"}:
                            config[target.id] = ast.literal_eval(node.value)
                        if target.id == "CAPTION_SETTINGS":
                            expressions[target.id] = ast.dump(node.value, include_attributes=False)
                        if target.id in {"ISIC_WORKER_SOURCE", "TRAINING_CODE_SHA256", "BASELINE_DICE", "BASELINE_IOU"}:
                            constants[target.id] = ast.literal_eval(node.value)
    return definitions, config, expressions, constants


def parse_full_both_logs(stream):
    records = []
    for line in stream.splitlines():
        match = re.match(r"^full_both epoch=(\d+) ", line)
        if match:
            row = {key: float(value) for key, value in re.findall(
                r"\b(val_dice|val_iou|q_gate|c_gate|semantic_mix)=([\d.]+)", line)}
            row["epoch"] = int(match[1])
            row["best_epoch"] = int(re.search(r"\bbest=(\d+)", line)[1])
            bad = re.search(r"\bbad=(\d+)/(\d+)", line)
            row.update(bad_epochs=int(bad[1]), patience=int(bad[2]), line=line)
            records.append(row)
    stopped = re.findall(r"^Early stopping at epoch (\d+)\s*$", stream, re.M)
    if len(stopped) != 1 or [row["epoch"] for row in records] != list(range(1, int(stopped[0]) + 1)):
        raise ValueError("Incomplete/duplicated full_both training logs")
    best = records[-1]["best_epoch"]
    if records[best - 1]["val_dice"] != max(row["val_dice"] for row in records):
        raise ValueError("Reported best epoch does not maximize displayed validation Dice")
    if records[-1]["bad_epochs"] != records[-1]["patience"]:
        raise ValueError("Early-stop summary does not match patience")
    return records, records[best - 1]


def section(nb, name):
    found = [(index, cell) for index, cell in enumerate(nb["cells"]) if cell.get("metadata", {}).get("section") == name]
    if len(found) != 1:
        raise ValueError(f"Expected one section: {name}")
    return found[0]


def html_tables(cell):
    result = []
    for output in cell.get("outputs", []):
        parser = Tables()
        parser.feed(text(output.get("data", {}).get("text/html")))
        result.extend(parser.tables)
    return result


def comparison_evidence():
    raw = SOURCE.read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    generic = audit(SOURCE)
    if generic["errors"]:
        raise ValueError("Executed caption notebook contains saved errors")
    visual = extract_evidence()
    baseline = json.loads((ROOT / visual["source_notebook"]).read_text(encoding="utf-8"))
    definitions, config, expressions, constants = source_identity(nb)
    template_defs, template_config, template_expressions, template_constants = source_identity(template)
    base_defs, base_config, _, _ = source_identity(baseline)
    if definitions != template_defs or config != template_config or expressions != template_expressions or constants != template_constants:
        raise ValueError("Model/training definitions, configuration or embedded worker differ from the reviewed template")
    test_index, test_cell = section(nb, "test_and_comparison")
    metrics = parse_test_metrics(stream_text(test_cell))
    _, train_cell = section(nb, "train_and_resume")
    logs, best = parse_full_both_logs(stream_text(train_cell))
    _, coverage_cell = section(nb, "aligned_text_banks")
    coverage_tables = [t for t in html_tables(coverage_cell) if t[0] == ["", "role", "images", "usable_captions", "usable_rate"]]
    if len(coverage_tables) != 1:
        raise ValueError("Missing caption coverage table")
    coverage = {row[1]: {"images": int(row[2]), "usable_captions": int(row[3]), "usable_rate_displayed": float(row[4])}
                for row in coverage_tables[0][1:]}
    csv_hashes = re.findall(r"^Caption CSV SHA256:\s*([a-f0-9]{64})\s*$", stream_text(coverage_cell), re.M)
    if len(csv_hashes) != 1:
        raise ValueError("Missing caption CSV runtime hash")
    _, worker_cell = section(nb, "generate_and_encode")
    worker_stream = stream_text(worker_cell)
    statuses = [ast.literal_eval(line) for line in worker_stream.splitlines() if line.startswith("{'complete':")]
    if len(statuses) != 1 or not statuses[0]["complete"] or statuses[0]["rows"] != 2750 or statuses[0]["csv_sha256"] != csv_hashes[0]:
        raise ValueError("Caption export status is incomplete or inconsistent")
    for role, size in visual["split_sizes_from_runtime"].items():
        if coverage[role]["images"] != size or coverage[role]["usable_captions"] != statuses[0]["valid_by_split"][role]:
            raise ValueError("Caption coverage disagrees with split/export status")
    if coverage["candidate_pool"]["images"] != visual["candidate_pool_size_from_runtime"]:
        raise ValueError("Candidate pool coverage count differs from visual run")
    raw_comparison = [t for t in html_tables(test_cell) if t[0] == ["", "method", "test_dice", "test_iou", "new_run", "dice_delta_percentage_points", "iou_delta_percentage_points"]]
    if len(raw_comparison) != 1:
        raise ValueError("Missing two-run comparison table")
    records = [dict(zip(raw_comparison[0][0][1:], row[1:])) for row in raw_comparison[0][1:]]
    if len(records) != 2:
        raise ValueError("Expected visual reference and new result")
    if [row["new_run"] for row in records] != ["False", "True"]:
        raise ValueError("Unexpected visual/caption result row identity")
    deltas = {}
    for metric in ("dice", "iou"):
        before = visual["metrics"][f"test_{metric}"]["value_displayed"]
        after = metrics[f"test_{metric}"]["value_displayed"]
        delta = (after - before) * 100
        if abs(float(records[0][f"test_{metric}"]) - before) > 0.00000051 or abs(float(records[1][f"test_{metric}"]) - after) > 0.00000051:
            raise ValueError("Comparison table disagrees with independent test stdout")
        if abs(float(records[1][f"{metric}_delta_percentage_points"]) - delta) > 0.000051:
            raise ValueError("Wrong metric delta / unit")
        deltas[metric] = {"from_rounded_metric_stdout_percentage_points": round(delta, 6),
                          "comparison_table_displayed_percentage_points": float(records[1][f"{metric}_delta_percentage_points"])}
    same_config = sorted(key for key in config.keys() & base_config.keys() if config[key] == base_config[key])
    differences = {key: [base_config.get(key), config.get(key)] for key in config.keys() & base_config.keys() if config[key] != base_config[key]}
    shared = set(definitions) & set(base_defs)
    baseline_streams = "\n".join(stream_text(cell) for cell in baseline["cells"])
    new_streams = "\n".join(stream_text(cell) for cell in nb["cells"])
    baseline_params = int(re.findall(r"^Trainable params:\s*([\d,]+)\s*$", baseline_streams, re.M)[0].replace(",", ""))
    new_params = int(re.findall(r"Full-both segmentation trainable params:\s*([\d,]+)", new_streams)[0].replace(",", ""))
    return {
        "schema": "isic_visual_vs_qwen8b_saved_outputs_v1", "audit_date": "2026-10-25",
        "source_notebook": SOURCE.relative_to(ROOT).as_posix(), "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "saved_outputs_sha256": generic["saved_outputs_sha256"], "source_notebook_modified": False, "cells_executed_by_auditor": False,
        "visual_reference": {"source_notebook": visual["source_notebook"], "notebook_sha256": visual["notebook_sha256"],
            "metrics": visual["metrics"], "epochs_observed": visual["training"]["epochs_observed"],
            "best_epoch": visual["training"]["best_epoch"], "best_validation_row_displayed": visual["training"]["best_validation_row_displayed"]},
        "caption_run": {"metrics": metrics, "test_cell_index_zero_based": test_index, "test_stdout": stream_text(test_cell),
            "test_loss": None, "epochs_observed": len(logs), "best_epoch_from_logs": best["epoch"],
            "best_validation_row_displayed": best, "training_logs": logs, "coverage": coverage,
            "caption_csv_sha256_from_runtime": csv_hashes[0], "caption_csv_independently_audited": False,
            "generation_status_from_runtime": statuses[0], "configuration_from_source": config,
            "training_protocol_id_from_runtime": re.findall(r"Locked training protocol:\s*([a-f0-9]{64})", "\n".join(stream_text(c) for c in nb["cells"])),
            "metadata": nb["metadata"].get("isic_full_both"),
            "raw_runtime_lines": [line for line in worker_stream.replace("\r", "\n").splitlines()
                                  if line.startswith(('{"transformers"', 'ISIC captions:', 'CLIP vectors:'))]},
        "comparison": {"deltas": deltas, "raw_table_rows": records, "same_source_config_keys": same_config,
            "trainable_parameters_from_stdout": {"visual": baseline_params, "caption": new_params, "additional": new_params - baseline_params},
            "different_source_config_values": differences,
            "common_baseline_definitions": len(shared),
            "different_common_baseline_definitions": sorted(name for name in shared if definitions[name] != base_defs[name]),
            "reviewed_template_shared_definitions": len(definitions), "reviewed_template_definitions_identical": definitions == template_defs,
            "embedded_worker_sha256": hashlib.sha256(constants["ISIC_WORKER_SOURCE"].encode()).hexdigest(),
            "embedded_worker_matches_template": constants["ISIC_WORKER_SOURCE"] == template_constants["ISIC_WORKER_SOURCE"]},
        "limits": ["One training seed only; no statistical significance or universal text benefit/harm established.",
            "Baseline is the separately audited historical notebook, not rerun alongside this caption experiment.",
            "Source/runtime image counts agree; full split/pool and old/new weight-byte identities are not independently compared.",
            "Caption CSV, text vectors, protocol, full predictions and checkpoints were not separately supplied for independent audit.",
            "Text gates are validation values at epoch 26, not printed best-checkpoint test gates or causal contribution percentages.",
            "100% caption policy coverage is not human-verified correctness or proof of no shortcut/leakage.",
            "Test Dice/IoU are six-decimal stdout; baseline uses four decimals; comparison-table deltas use more internal precision.",
            "Text fusion, semantic retrieval, parameter count and training trajectory change together."],
        "generic_output_audit": generic,
    }


def main():
    evidence = comparison_evidence()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caption_metrics": evidence["caption_run"]["metrics"],
        "deltas": evidence["comparison"]["deltas"], "best_epoch": evidence["caption_run"]["best_epoch_from_logs"]}, indent=2))


if __name__ == "__main__":
    main()
