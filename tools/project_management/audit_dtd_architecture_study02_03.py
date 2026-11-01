"""Audit saved study02/03 notebooks and structured exports without executing cells/checkpoints."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/multimodal_architecture_ablation/dtd"
SOURCES = [
    ("02", "stydu2_da_train.ipynb", "dtd_study02_study03_results_20261026_050338/study02_norm_fusion",
     "02_candidate_only_norm_fusion/dtd/mtacs-dtd-vitl16-prefix-candidate-only-norm-fusion.ipynb"),
    ("03", "study3_datrain.ipynb", "dtd_study03_results_20261026_050358/study03_mltm_alignment",
     "03_candidate_only_mltm_alignment/dtd/mtacs-dtd-vitl16-prefix-candidate-only-mltm-alignment.ipynb"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification_metrics(frame):
    truth = [int(r["ground_truth"]) for r in frame]
    pred = [int(r["prediction"]) for r in frame]
    support, predicted, tp = Counter(truth), Counter(pred), Counter(a for a, b in zip(truth, pred) if a == b)
    labels = sorted(set(truth) | set(pred))
    return {"accuracy": sum(a == b for a, b in zip(truth, pred)) / len(truth),
            "macro_f1": sum(2 * tp[c] / (support[c] + predicted[c]) for c in labels) / len(labels)}


def definitions(notebook):
    result = {}
    for i, cell in enumerate(notebook["cells"][:26]):
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                result[f"{i}:{node.name}"] = ast.dump(node, include_attributes=False)
    return result


def audit():
    evidence = {"schema": "dtd_architecture_study02_03_saved_outputs_v1", "observed_date": "2026-10-26",
                "executes_notebook_cells": False, "loads_checkpoint_pickle": False, "studies": {}}
    results, interventions, protocols, predictions = [], [], {}, {}
    for study, name, folder, template in SOURCES:
        notebook_path, output = ROOT / name, ROOT / folder
        notebook = read_json(notebook_path)
        original = read_json(ROOT / "notebooks/02_improvements/multimodal_architecture_ablation" / template)
        protocol = read_json(output / "protocol.json")
        protocols[study] = protocol
        run = output / "runs/candidate_only/seed_42"
        history = rows(run / "history.csv")
        completed = read_json(run / "completed.json")
        assert [int(r["epoch"]) for r in history] == list(range(1, 51))
        assert completed["best_epoch"] == int(min(history, key=lambda r: float(r["val_loss"]))["epoch"])
        assert sha256(output / "shared/captions.csv") == protocol["caption_csv_sha256"]
        assert definitions(notebook) == definitions(original)
        errors, stderr_tracebacks = [], []
        for i, cell in enumerate(notebook["cells"]):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    errors.append({"cell": i, "type": out.get("ename"), "message": out.get("evalue")})
                text = "".join(out.get("text", []))
                if out.get("name") == "stderr" and any(t in text for t in ["Traceback", "Exception", "Bad file descriptor"]):
                    stderr_tracebacks.append({"cell": i, "text": text})
        item = {"notebook": name, "notebook_sha256": sha256(notebook_path), "output": folder,
                "template_definitions_ast_identical": True, "notebook_error_outputs": errors,
                "stderr_tracebacks": stderr_tracebacks, "extra_notebook_cells": len(notebook["cells"]) - 26,
                "epochs": len(history), "best_epoch": completed["best_epoch"],
                "environment": read_json(output / "environment.json"),
                "caption_sha256": protocol["caption_csv_sha256"],
                "candidate_pool_size": len(protocol["candidate_ids"]),
                "completed": completed, "history_first": history[0], "history_last": history[-1],
                "checkpoint": {"path": (run / "best.pt").relative_to(ROOT).as_posix(),
                               "bytes": (run / "best.pt").stat().st_size}, "metrics": {}}
        for split in ["val", "test"]:
            frame = rows(run / f"{split}_predictions.csv")
            expected = set(protocol["split_image_ids"][split])
            assert len(frame) == len({r["image_id"] for r in frame}) == len(expected) == 1880
            assert {r["image_id"] for r in frame} == expected
            assert {r["selected_context_id"] for r in frame} <= set(protocol["candidate_ids"])
            assert all(float(r["query_gate"]) == 0 and r["query_caption_valid"] == "False" for r in frame)
            assert all(r["candidate_caption_source_id"] == r["selected_context_id"] for r in frame)
            metric = read_json(run / f"{split}_metrics.json")
            for key, value in classification_metrics(frame).items():
                assert math.isclose(metric[key], value, abs_tol=1e-12)
            item["metrics"][split] = metric
            predictions[(study, split)] = {r["image_id"]: r for r in frame}
            results.append({"study": study, "split": split, "epochs": len(history),
                            "best_epoch": completed["best_epoch"], **metric})
        intervention_rows = rows(run / "validation_norm_diagnostics/validation_intervention_predictions.csv")
        for row in rows(output / "validation_interventions.csv"):
            subset = [r for r in intervention_rows if r["intervention"] == row["intervention"]]
            assert len(subset) == 1880
            for key, value in classification_metrics(subset).items():
                assert math.isclose(float(row[key]), value, abs_tol=1e-12)
            interventions.append({"study": study, **row})
        item["fusion_interventions"] = rows(output / "validation_interventions.csv")
        if study == "03":
            item["alignment_fit"] = rows(output / "training_pool_alignment_summary.csv")[0]
        evidence["studies"][study] = item
    same_keys = ["split_image_ids", "candidate_ids", "training", "preprocessing", "caption_csv_sha256",
                 "caption_input_policy", "text_scope", "class_names", "weights_bytes"]
    evidence["protocol_matches"] = {k: protocols["02"][k] == protocols["03"][k] for k in same_keys}
    assert all(evidence["protocol_matches"].values())
    evidence["weight_bytes_identity_proven"] = False
    evidence["runtime_versions_match"] = evidence["studies"]["02"]["environment"] == evidence["studies"]["03"]["environment"]
    evidence["paired_study03_vs_study02"] = {}
    for split in ["val", "test"]:
        left, right = predictions[("02", split)], predictions[("03", split)]
        assert all(left[i]["ground_truth"] == right[i]["ground_truth"] for i in left)
        improved = sum(a["correct"] == "False" and right[i]["correct"] == "True" for i, a in left.items())
        harmed = sum(a["correct"] == "True" and right[i]["correct"] == "False" for i, a in left.items())
        n = improved + harmed
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(improved, harmed) + 1)) / 2 ** n) if n else 1.
        evidence["paired_study03_vs_study02"][split] = {
            "improved": improved, "harmed": harmed, "mcnemar_exact_two_sided_p": p,
            "prediction_changed": sum(a["prediction"] != right[i]["prediction"] for i, a in left.items()),
            "context_changed": sum(a["selected_context_id"] != right[i]["selected_context_id"] for i, a in left.items()),
            "note": "paired correctness diagnostic, not a macro-F1 significance test or training-seed uncertainty"}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "study02_03_output_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    for filename, table in [("study02_03_results.csv", results), ("study02_03_validation_interventions.csv", interventions)]:
        with (REPORT / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps({"metrics": results, "paired": evidence["paired_study03_vs_study02"],
                      "protocol_matches": evidence["protocol_matches"],
                      "runtime_versions_match": evidence["runtime_versions_match"]}, indent=2))


if __name__ == "__main__":
    audit()
