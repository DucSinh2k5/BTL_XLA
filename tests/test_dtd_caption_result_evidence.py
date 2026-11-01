import hashlib
import json
import re
import pytest

from tools.project_management import audit_dtd_caption_results as audit


def test_rounded_final_table_cannot_replace_dedicated_metrics():
    val, test = audit.metric_rows([
        {"rows": [["", "status", "macro_f1", "accuracy", "loss"], ["0", "completed", "0.811412", "0.813298", "1.301636"]]},
        {"rows": [["", "macro_f1", "accuracy", "loss"], ["0", "0.821445", "0.822872", "1.275098"]]},
        {"rows": [["", "split", "macro_f1", "accuracy", "loss"], ["0", "test", "0.8214", "0.8229", "1.2751"]]},
    ])
    assert val["macro_f1"] == "0.811412"
    assert test["macro_f1"] == "0.821445"


def test_missing_test_metrics_are_not_fabricated():
    with pytest.raises(ValueError, match="one dedicated"):
        audit.metric_rows([])


@pytest.mark.parametrize("run", audit.RUNS, ids=lambda row: row[2])
def test_relocated_evidence_matches_saved_notebook(run):
    evidence = json.loads((audit.ROOT / audit.EVIDENCE).read_text(encoding="utf-8"))
    path = audit.ROOT / audit.target_path(run)
    csv_path = audit.ROOT / "data/captions/dtd/grounded_controls" / run[3] if run[3] else None
    actual = audit.audit_result(path, csv_path)
    saved = next(row for row in evidence["new_runs"] if row["method"] == actual["method"])
    assert actual == saved
    assert len(actual["training_logs"]) == 50
    assert actual["csv_sha_verified_against_local_input"] == (run[3] is not None)


def test_all_six_moves_preserve_every_notebook_byte():
    manifest = json.loads((audit.ROOT / audit.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["verified_after_move"] is True
    assert len(manifest["files"]) == 6
    for entry in manifest["files"]:
        path = audit.ROOT / entry["new"]
        raw, nb = audit.read_notebook(path)
        assert hashlib.sha256(raw).hexdigest() == entry["sha256_before"]
        assert audit.output_hash(nb) == entry["saved_outputs_sha256_before"]
        assert len(raw) == entry["bytes"]
        assert not (audit.ROOT / entry["old"]).exists()


def test_report_and_result_index_links_resolve():
    documents = [audit.ROOT / audit.RESULTS / "README.md",
                 audit.ROOT / "reports/caption_ablation/dtd/KET_QUA_SO_SANH_PHUONG_PHAP_CAPTION_DTD.md"]
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            assert (document.parent / target).resolve().exists(), (document, target)


def test_best_method_and_qwen_gap_are_reported_in_correct_units():
    evidence = json.loads((audit.ROOT / audit.EVIDENCE).read_text(encoding="utf-8"))
    rows = {row["method"]: row for row in evidence["new_runs"]}
    q8, q32 = rows["qwen3_vl_8b_instruct"], rows["qwen3_vl_32b_instruct"]
    assert max(rows.values(), key=lambda r: float(r["test_metrics"]["macro_f1"])) == q8
    assert (float(q8["test_metrics"]["macro_f1"]) - float(q32["test_metrics"]["macro_f1"])) * 100 == pytest.approx(0.0657)
    assert q8["correct_count_inferred_from_displayed_accuracy"] - q32["correct_count_inferred_from_displayed_accuracy"] == 2
    assert evidence["source_ast_comparison"]["identical_definitions"] == 57
