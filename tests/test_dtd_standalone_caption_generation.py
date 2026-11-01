
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import io
import json
import os
import subprocess
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

import dtd_caption_generation_cells as captions
from test_dtd_caption_generation import caption_runtime
from test_dtd_query_only_notebook import audited_rows, runtime
from test_dtd_text_ablation import toy_runtime


def test_embedded_worker_is_standalone_and_has_no_training_or_label_inputs():
    source = captions.worker_source()
    ast.parse(source)
    assert 'caption_records = payload["records"]' in source
    assert "train_df" not in source and "candidate_df" not in source
    assert "local_files_only=True" in source and "CAPTION_WORKERS = 0" in source
    assert 'CAPTION_CSV = Path(payload["output"])' in source
    assert "offset += len(group)" in source
    assert "except torch.cuda.OutOfMemoryError" in source
    assert "append_rows(CAPTION_CSV, batch_rows)" in source


def test_worker_generates_audited_csv_and_preserves_completed_batches_on_resume(caption_runtime):
    ns, _, calls = caption_runtime
    frames = {split: ns[f"{split}_df"] for split in ("train", "val", "test")}
    output = ns["OUT_PUT_DIR"] / "shared" / "captions.csv"
    ns["payload"] = {"output": str(output), "records": [
        {"image_id": row.image_id, "image_path": row.image_path, "split": split}
        for split, frame in frames.items() for row in frame.itertuples(index=False)]}
    generation = captions.worker_source().split('CAPTION_CSV = Path(payload["output"])', 1)[1]
    generation = 'CAPTION_CSV = Path(payload["output"])' + generation
    original = ns["generate_batch"]
    batch_calls = []

    def interrupted(*args, **kwargs):
        batch_calls.append(True)
        if len(batch_calls) == 2:
            raise RuntimeError("test interruption")
        return original(*args, **kwargs)

    ns["generate_batch"] = interrupted
    with pytest.raises(RuntimeError, match="test interruption"):
        exec(generation, ns)
    before = ns["load_existing"](output)
    assert len(before) == 2
    ns["generate_batch"] = original
    exec(generation, ns)
    after = ns["load_existing"](output)
    assert len(after) == 6 and calls["generated"] == 6
    assert all(after[key] == value for key, value in before.items())
    assert all(row["caption_valid"] == "True" for row in after.values())


def test_fresh_standalone_csv_stage_invokes_captioner_then_skips_it_when_complete(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    frames = ns["SPLIT_FRAMES"]
    cached = tmp_path / "shared" / "captions.csv"
    calls = []
    ns.update({"PROJECT_DIR": tmp_path, "resolve_clip_path": lambda *args: tmp_path / "clip"})

    def generate(parts, output, clip):
        assert parts is frames and clip == tmp_path / "clip"
        calls.append(output)
        audited_rows(parts).to_csv(output, index=False)

    ns["run_local_dtd_caption_generation"] = generate
    assert ns["ensure_dtd_caption_csv"](frames, "", cached, []) == cached
    assert len(calls) == 1
    ns["FLORENCE_MODEL_REF"] = "missing-model"
    assert ns["ensure_dtd_caption_csv"](frames, "", cached, []) == cached
    assert len(calls) == 1


def test_partial_csv_resumes_and_reuse_only_setting_still_fails_clearly(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    frames = ns["SPLIT_FRAMES"]
    cached = tmp_path / "shared" / "captions.csv"
    cached.parent.mkdir()
    rows = audited_rows(frames)
    rows.iloc[:2].to_csv(cached, index=False)
    ns["GENERATE_CAPTIONS_IF_MISSING"] = False
    with pytest.raises(FileNotFoundError, match="missing/incomplete"):
        ns["ensure_dtd_caption_csv"](frames, "", cached, [])
    ns["GENERATE_CAPTIONS_IF_MISSING"] = True
    ns.update({"PROJECT_DIR": tmp_path, "resolve_clip_path": lambda *args: tmp_path / "clip"})

    def resume(parts, output, clip):
        existing = pd.read_csv(output)
        assert existing.image_id.tolist() == rows.iloc[:2].image_id.tolist()
        rows.to_csv(output, index=False)

    ns["run_local_dtd_caption_generation"] = resume
    ns["ensure_dtd_caption_csv"](frames, "", cached, [])
    assert len(pd.read_csv(cached)) == len(rows)


def test_complete_external_csv_skips_all_florence_assets_and_rejects_different_cache(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    frames = ns["SPLIT_FRAMES"]
    source = tmp_path / "captions.csv"
    rows = audited_rows(frames)
    rows.to_csv(source, index=False)
    cached = tmp_path / "shared" / "captions.csv"

    def forbidden(*args):
        raise AssertionError("A complete CSV must never touch Florence or wheel setup")

    ns["run_local_dtd_caption_generation"] = forbidden
    ns["resolve_clip_path"] = forbidden
    ns["ensure_dtd_caption_csv"](frames, str(source), cached, [])
    assert cached.read_bytes() == source.read_bytes()
    rows.assign(final_caption="changed text").to_csv(source, index=False)
    with pytest.raises(ValueError, match="differ"):
        ns["ensure_dtd_caption_csv"](frames, str(source), cached, [])


def test_checkpoint_discovery_checks_weights_and_detects_legacy_format(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    model = tmp_path / "notebook-mt-moi" / "Florence-2-base"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "florence2",
        "auto_map": {"AutoModelForCausalLM": "modeling_florence2.Florence2ForConditionalGeneration"}}))
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ns["florence_checkpoint"](str(model), [])
    (model / "model.safetensors").write_bytes(b"test only")
    for name in ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py"):
        (model / name).write_text("# test only\n")
    path, backend = ns["florence_checkpoint"](str(model), [])
    assert path == model.resolve() and backend == "legacy"
    assert ns["florence_checkpoint"]("microsoft/Florence-2-base", [tmp_path]) == (path, backend)


def test_wheel_archives_extract_only_allowed_wheels_without_path_traversal(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    assets = tmp_path / "input"
    assets.mkdir()
    with zipfile.ZipFile(assets / "offline_wheels.zip", "w") as archive:
        archive.writestr("../../transformers-4.49.0-py3-none-any.whl", b"test wheel")
        archive.writestr("torch-9.0.0-whatever.whl", b"not extracted")
        archive.writestr("../../danger.py", b"not extracted")
    destination = tmp_path / "working" / "wheels"
    directories = ns["offline_caption_wheels"]([assets], destination)
    assert directories == [str(destination)]
    assert [path.name for path in destination.iterdir()] == ["transformers-4.49.0-py3-none-any.whl"]
    assert not (tmp_path / "danger.py").exists()


def test_legacy_runtime_uses_only_local_wheels_in_separate_target_without_mutating_kernel(toy_runtime, tmp_path, monkeypatch):
    ns = runtime(toy_runtime)
    assets = tmp_path / "input"
    assets.mkdir()
    (assets / "transformers-4.49.0-py3-none-any.whl").write_bytes(b"test wheel")
    calls = []
    old_environment = dict(os.environ)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="Transformers 5.x does not match legacy")
        if "pip" in command:
            assert "--no-index" in command and "--target" in command and "--no-deps" in command
            assert "--only-binary=:all:" in command
            assert "transformers==4.49.0" in command and "tokenizers>=0.21,<0.22" in command
            assert "huggingface-hub>=0.26,<1" in command
            return SimpleNamespace(returncode=0, stdout="offline installed", stderr="")
        assert kwargs["env"]["PYTHONPATH"].startswith(str(tmp_path / "runtime" / "legacy_packages"))
        return SimpleNamespace(returncode=0, stdout="4.49.0", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    env = ns["caption_runtime_environment"]("legacy", assets, tmp_path / "runtime")
    assert len(calls) == 3
    assert env["TOKENIZERS_PARALLELISM"] == "false" and env["HF_HUB_OFFLINE"] == "1"
    assert dict(os.environ) == old_environment


def test_missing_or_incompatible_offline_wheels_report_actionable_errors(toy_runtime, tmp_path, monkeypatch):
    ns = runtime(toy_runtime)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="missing matching wheel"))
    ns["offline_caption_wheels"] = lambda *args: []
    with pytest.raises(FileNotFoundError, match="No offline wheels"):
        ns["caption_runtime_environment"]("legacy", tmp_path, tmp_path / "runtime")
    ns["offline_caption_wheels"] = lambda *args: [str(tmp_path)]
    with pytest.raises(RuntimeError, match="must match Kaggle's Python/platform"):
        ns["caption_runtime_environment"]("legacy", tmp_path, tmp_path / "runtime")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_caption_child_logs_progress_and_never_receives_per_image_labels(toy_runtime, tmp_path, monkeypatch, capsys, exit_code):
    ns = runtime(toy_runtime)
    frames = {split: frame.assign(image_path=[str(tmp_path / (image_id.replace("/", "_") + ".png"))
                                             for image_id in frame.image_id])
              for split, frame in ns["SPLIT_FRAMES"].items()}
    csv = tmp_path / "shared" / "captions.csv"
    csv.parent.mkdir()
    ns["florence_checkpoint"] = lambda *args: (tmp_path / "Florence-2-base", "legacy")

    def environment(backend, assets, runtime_dir):
        runtime_dir.mkdir(parents=True)
        return {"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"}

    ns["caption_runtime_environment"] = environment

    class Child:
        def __init__(self, command, **kwargs):
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
            worker = captions.Path(command[1])
            ast.parse(worker.read_text(encoding="utf-8"))
            payload = json.loads(captions.Path(command[2]).read_text(encoding="utf-8"))
            assert payload["florence"] == str(tmp_path / "Florence-2-base")
            assert len(payload["records"]) == 18
            assert all(set(record) == {"image_id", "image_path", "split"} for record in payload["records"])
            self.stdout = io.StringIO("DTD captions: batch 1 flushed\n")

        def wait(self):
            return exit_code

    monkeypatch.setattr(subprocess, "Popen", Child)
    if exit_code:
        with pytest.raises(RuntimeError, match="Completed CSV batches are preserved"):
            ns["run_local_dtd_caption_generation"](frames, csv, tmp_path / "clip")
    else:
        ns["run_local_dtd_caption_generation"](frames, csv, tmp_path / "clip")
    assert "batch 1 flushed" in capsys.readouterr().out
    assert "batch 1 flushed" in (csv.parent / "caption_runtime" / "generation.log").read_text()
