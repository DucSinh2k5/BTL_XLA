
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import copy
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

import mtacs_dtd_ablation_cells as original
from test_dtd_text_ablation import spec, toy_runtime


NOTEBOOK = project_path("notebookb7a9dba884.ipynb")


def sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {index: "".join(cell["source"]) for index, cell in enumerate(notebook["cells"])}


def runtime(toy_runtime):
    namespace = dict(vars(original))
    code = sources()
    exec(compile(code[15], "notebook_training", "exec"), namespace)
    exec(compile(code[16], "notebook_evaluation", "exec"), namespace)
    namespace.update({
        "os": os, "display": lambda value: None,
        "LR_SCHEDULE_EPOCHS": 100, "EARLY_STOPPING": False,
        "KEEP_LAST_CHECKPOINT": True, "DTD_SPLIT": 1,
        "RUN_VALIDATION_INTERVENTIONS": True, "RUN_FINAL_TEST": True,
        "SELECTED_ABLATIONS": [
            spec("visual_only", query_fusion=False, candidate_fusion=False, semantic_selector=False),
            spec(),
        ],
        "ABLATION_NAMES": ["visual_only", "full_both"],
    })
    return namespace


def test_all_notebook_code_compiles_and_defaults_are_50_epochs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")
    code = sources()
    assert 'DTD_ABLATION_EPOCHS", "50"' in code[5]
    assert "EARLY_STOPPING = False" in code[4]
    assert code[3].index('os.environ["TOKENIZERS_PARALLELISM"] = "false"') < code[3].index("import importlib")
    assert "max(LR_SCHEDULE_EPOCHS - WARMUP_EPOCHS, 1)" in code[15]
    assert "if EARLY_STOPPING and bad_epochs >= PATIENCE" in code[15]


def test_protocol_reuses_only_a_shortened_epoch_limit(toy_runtime):
    ns = runtime(toy_runtime)
    ns["EPOCHS"] = 50
    saved = {"schema": "test", "training": {"EPOCHS": 100, "BATCH_SIZE": 4}}
    original.write_json(ns["OUT_PUT_DIR"] / "protocol.json", saved)
    requested = copy.deepcopy(saved)
    requested["training"]["EPOCHS"] = 50
    actual = ns["resolve_training_protocol"](requested)
    assert actual == saved
    assert original.stable_hash(actual) == original.stable_hash(saved)
    requested["training"]["BATCH_SIZE"] = 8
    with pytest.raises(ValueError, match="another protocol"):
        ns["resolve_training_protocol"](requested)
    requested = copy.deepcopy(saved)
    requested["training"]["EPOCHS"] = ns["EPOCHS"] = 101
    with pytest.raises(ValueError):
        ns["resolve_training_protocol"](requested)


def test_resume_epoch_35_to_50_keeps_the_original_lr_and_rng(toy_runtime):
    ns = runtime(toy_runtime)
    configuration = ns["SELECTED_ABLATIONS"][0]
    ns["EPOCHS"] = 100
    ns["PATIENCE"] = 1
    original_train = ns["train_ablation_epoch"]

    def interrupt_at_36(network, loader, banks, bank, optimizer, scaler, epoch):
        if epoch == 36:
            raise RuntimeError("synthetic interruption at epoch 36")
        return original_train(network, loader, banks, bank, optimizer, scaler, epoch)

    ns["train_ablation_epoch"] = interrupt_at_36
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        ns["train_one_configuration"](configuration, 42)
    run_dir = ns["RUNS_DIR"] / "visual_only" / "seed_42"
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 35
    identity = state["identity"]
    assert (run_dir / "best.pt").is_file()
    del state

    ns["EPOCHS"] = 50
    resumed_epochs = []

    def resume(network, loader, banks, bank, optimizer, scaler, epoch):
        resumed_epochs.append(epoch)
        return original_train(network, loader, banks, bank, optimizer, scaler, epoch)

    ns["train_ablation_epoch"] = resume
    ns["train_one_configuration"](configuration, 42)
    assert resumed_epochs == list(range(36, 51))
    actual = pd.read_csv(run_dir / "history.csv")
    assert actual.epoch.tolist() == list(range(1, 51))
    completed = original.read_json(run_dir / "completed.json")
    assert completed["identity"] == identity
    assert completed["epochs_trained"] == 50
    assert completed["training_budget"] == {
        "epoch_limit": 50, "lr_schedule_epochs": 100, "early_stopping": False,
    }
    assert ns["is_run_completed"](configuration, 42)
    timestamp = (run_dir / "best.pt").stat().st_mtime_ns
    ns["train_one_configuration"](configuration, 42)
    assert (run_dir / "best.pt").stat().st_mtime_ns == timestamp

    # A resumed 100-epoch schedule must match its uninterrupted first 50 epochs.
    ns["RUNS_DIR"] = ns["OUT_PUT_DIR"] / "reference_runs"
    ns["train_ablation_epoch"] = original_train
    reference_dir = ns["train_one_configuration"](configuration, 42)
    expected = pd.read_csv(reference_dir / "history.csv")
    pd.testing.assert_frame_equal(actual, expected, atol=1e-7, rtol=1e-7)


def test_empty_and_partial_reports_do_not_open_test_or_load_text_checkpoints(toy_runtime):
    ns = runtime(toy_runtime)
    full_dir = ns["RUNS_DIR"] / "full_both" / "seed_42"
    full_dir.mkdir(parents=True)
    (full_dir / "best.pt").write_bytes(b"not loaded for a partial run")
    for index in (23, 25, 27):
        exec(compile(sources()[index], f"cell_{index}", "exec"), ns)
    assert not (ns["OUT_PUT_DIR"] / "locked_test_plan.json").exists()
    for name in ("validation_interventions.csv", "ablation_val_results.csv",
                 "ablation_val_paired_vs_visual.csv", "ablation_test_results.csv"):
        frame = pd.read_csv(ns["OUT_PUT_DIR"] / name)
        assert frame.empty
        assert len(frame.columns) > 0
    report = (ns["OUT_PUT_DIR"] / "ablation_report.md").read_text(encoding="utf-8")
    assert "No completed metrics yet" in report
    assert "partial" in report


def test_completed_text_run_supports_interventions_and_locks_budget(toy_runtime):
    ns = runtime(toy_runtime)
    for configuration in ns["SELECTED_ABLATIONS"]:
        ns["train_one_configuration"](configuration, 42)
    exec(compile(sources()[23], "intervention_cell", "exec"), ns)
    assert len(ns["interventions"]) == 9
    assert len(ns["summarize_ablation"]("val")) == 2
    assert len(ns["evaluate_locked_test"]()) == 2
    ns["EPOCHS"] = 3
    with pytest.raises(ValueError, match="locked"):
        ns["ensure_training_budget"]()


def test_loader_disables_tokenizer_parallelism_before_worker_creation(toy_runtime, monkeypatch):
    ns = runtime(toy_runtime)
    node = next(node for node in ast.parse(sources()[14]).body
                if isinstance(node, ast.FunctionDef) and node.name == "make_ablation_loaders")
    calls = []

    def loader(*args, **kwargs):
        calls.append(os.environ.get("TOKENIZERS_PARALLELISM"))
        return object()

    ns.update({"NUM_WORKERS": 2, "BATCH_SIZE": 2, "train_transform": None, "eval_transform": None,
               "DTDImageQueryDataset": lambda frame, transform: frame, "DataLoader": loader})
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "loader", "exec"), ns)
    ns["make_ablation_loaders"](42)
    assert calls == ["false", "false", "false"]
