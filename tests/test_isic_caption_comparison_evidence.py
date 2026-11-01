import hashlib
import json
import re

import pytest

from tools.project_management import audit_isic_caption_comparison as comparison


def test_new_test_metrics_are_lower_than_independently_read_visual_reference():
    data = comparison.comparison_evidence()
    assert data['caption_run']['metrics']['test_dice']['value_displayed'] == .848689
    assert data['caption_run']['metrics']['test_iou']['value_displayed'] == .765407
    assert data['visual_reference']['metrics']['test_dice']['value_displayed'] == .856
    assert data['visual_reference']['metrics']['test_iou']['value_displayed'] == .7738
    assert data['comparison']['deltas']['dice']['from_rounded_metric_stdout_percentage_points'] == pytest.approx(-.7311)
    assert data['comparison']['deltas']['iou']['from_rounded_metric_stdout_percentage_points'] == pytest.approx(-.8393)


def test_epoch_gates_and_coverage_are_validation_observations_not_test_inventions():
    data = comparison.comparison_evidence()
    run = data['caption_run']
    assert run['epochs_observed'] == 41 and run['best_epoch_from_logs'] == 26
    assert run['best_validation_row_displayed']['val_dice'] == .8755
    assert run['best_validation_row_displayed']['q_gate'] == .1315
    assert run['best_validation_row_displayed']['c_gate'] == .0048
    assert run['best_validation_row_displayed']['semantic_mix'] == .5026
    assert run['coverage']['candidate_pool']['usable_captions'] == 400
    assert all(row['usable_rate_displayed'] == 1 for row in run['coverage'].values())
    assert run['test_loss'] is None and not run['caption_csv_independently_audited']


def test_comparison_does_not_modify_notebooks_and_json_matches_source():
    data = comparison.comparison_evidence()
    assert json.loads(comparison.DESTINATION.read_text(encoding='utf-8')) == data
    assert hashlib.sha256(comparison.SOURCE.read_bytes()).hexdigest() == '32779936c8509b0d45c201d6139d06dcea2bfcc591a8409e7ed2240fa5758847'
    assert data['visual_reference']['notebook_sha256'] == '052b4edf0110f2caf6d1b2e9a1d8619dfdfb57d56d97c8768553b9a566215599'
    assert data['comparison']['different_source_config_values'] == {}
    assert data['comparison']['common_baseline_definitions'] == 22
    assert data['comparison']['different_common_baseline_definitions'] == []
    assert data['comparison']['reviewed_template_shared_definitions'] == 63
    assert data['comparison']['embedded_worker_matches_template'] is True
    assert data['comparison']['trainable_parameters_from_stdout']['additional'] == 397571


def test_incomplete_training_logs_are_not_marked_complete():
    with pytest.raises(ValueError, match='Incomplete'):
        comparison.parse_full_both_logs('full_both epoch=001 val_dice=0.8 val_iou=0.7 q_gate=0.1 c_gate=0.01 semantic_mix=0.5 best=001 bad=0/15\nEarly stopping at epoch 41\n')


def test_comparison_report_links_and_actual_table3_row():
    report = comparison.ROOT / 'reports/paper_tables/SO_SANH_ISIC_VISUAL_VA_QWEN8B.md'
    table3 = comparison.ROOT / 'reports/paper_tables/TABLE_03_SEGMENTATION.md'
    assert '**84.87** | **76.54**' in table3.read_text(encoding='utf-8')
    for path in (report, table3):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            assert (path.parent / target.strip('<>')).resolve().exists(), (path, target)
