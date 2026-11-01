import json

from tools.project_management import audit_report_readiness as audit


def test_notebook_magics_do_not_hide_the_seed():
    evidence = audit.source_evidence('%pip install unused\nSEED = 43\nEPOCHS = 100\n', 2)
    assert [(row['name'], row['expression'], row['line']) for row in evidence] == [
        ('SEED', '43', 2), ('EPOCHS', '100', 3)]


def test_split_stream_messages_are_joined_without_executing_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    path = tmp_path / 'run.ipynb'
    path.write_text(json.dumps({'cells': [{'cell_type': 'code',
        'source': ['SEED = 42\n', 'raise RuntimeError("do not execute")\n'],
        'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['test_accuracy: 0.']},
                    {'output_type': 'stream', 'name': 'stdout', 'text': ['9360\n']}]}]}), encoding='utf-8')
    result = audit.audit(path)
    assert result['metric_lines'] == [{'cell_index': 0, 'text': 'test_accuracy: 0.9360'}]
    assert result['errors'] == []
    assert result['cells_with_outputs'] == 1


def test_html_tables_are_read_as_structured_rows():
    parser = audit.Tables()
    parser.feed('<table><tr><th>method</th><th>accuracy</th></tr>'
                '<tr><td>baseline</td><td>0.9</td></tr></table>')
    assert parser.tables == [[['method', 'accuracy'], ['baseline', '0.9']]]
