import hashlib
import json
import re

import pytest

from tools.project_management import audit_report_readiness as generic
from tools.project_management import audit_table3_segmentation as audit


def test_test_metrics_do_not_read_the_printed_paper_reference():
    stream = "paper TACS ISIC2017 Dice/IoU: 0.8500 / 0.7650\ntest_dice: 0.8560 (85.60%)\ntest_iou: 0.7738 (77.38%)\n"
    result = audit.parse_test_metrics(stream)
    assert result["test_dice"]["value_displayed"] == 0.8560
    assert result["test_iou"]["percent_displayed"] == 77.38
    assert result["test_dice"]["decimal_places"] == 4


def test_paper_reference_alone_cannot_fill_project_metrics():
    with pytest.raises(ValueError, match="Expected one explicitly printed"):
        audit.parse_test_metrics("paper TACS ISIC2017 Dice/IoU: 0.8500 / 0.7650\n")


def test_inconsistent_metric_percentage_fails():
    with pytest.raises(ValueError, match="Inconsistent"):
        audit.parse_test_metrics("test_dice: 0.8560 (87.31%)\ntest_iou: 0.7738 (77.38%)\n")


def test_split_stream_segmentation_metrics_are_recognized(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "ROOT", tmp_path)
    path = tmp_path / "run.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["forbidden_indices = None\n"],
        "outputs": [{"output_type": "stream", "name": "stdout", "text": "test_dice: 0."},
                    {"output_type": "stream", "name": "stdout", "text": "8560 (85.60%)\ntest_iou: 0.7738 (77.38%)\n"}]}]}), encoding="utf-8")
    result = generic.audit(path)
    assert len(result["metric_lines"]) == 2
    assert result["source_indicators"]["candidate_self_exclusion_text_present"] is True


def test_saved_notebook_and_evidence_agree_without_notebook_edits():
    expected = json.loads(audit.DESTINATION.read_text(encoding="utf-8"))
    assert audit.extract_evidence() == expected
    assert hashlib.sha256(audit.SOURCE.read_bytes()).hexdigest() == "052b4edf0110f2caf6d1b2e9a1d8619dfdfb57d56d97c8768553b9a566215599"
    assert expected["split_sizes_from_runtime"] == {"train": 2000, "val": 150, "test": 600}
    assert expected["training"]["epochs_observed"] == 64
    assert expected["training"]["best_epoch"] == 49
    assert expected["table_mapping"]["columns"] == ["Dice", "IoU"]
    assert expected["table_mapping"]["exact_paper_reproduction"] is False
    assert expected["test_loss"] is None


def test_table3_values_and_source_links_are_valid():
    report = audit.ROOT / "reports/paper_tables/TABLE_03_SEGMENTATION.md"
    content = report.read_text(encoding="utf-8")
    assert "**85.60** | **77.38**" in content
    assert "+0.60 | +0.88" in content
    for target in re.findall(r"\]\(([^)]+)\)", content):
        assert (report.parent / target.strip("<>")).resolve().exists(), target
