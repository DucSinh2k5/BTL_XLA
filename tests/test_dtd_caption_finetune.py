
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import copy
import json
import os
import runpy
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import dtd_caption_finetune as ft


ROOT = PROJECT_ROOT
NOTEBOOK = project_path('florence2-dtd-caption-finetune.ipynb')
TARGET_TEXT = "Fine pale lines cross dark rough surfaces"


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "dtd"
    (root / "labels").mkdir(parents=True)
    for role in ("train", "val", "test"):
        ids = []
        for category in ("banded", "bubbly"):
            (root / "images" / category).mkdir(parents=True, exist_ok=True)
            for index in range(4):
                image_id = f"{category}/{role}_{index}.png"
                Image.new("RGB", (8, 8), (50 + index, 100, 150)).save(root / "images" / image_id)
                ids.append(image_id)
        (root / "labels" / f"{role}1.txt").write_text("\n".join(ids), encoding="utf-8")
    records, names = ft.load_caption_manifest(root, 1, expected_classes=2, per_class=4)
    return root, records, names


def approved(data, tmp_path):
    _, records, _ = data
    path = ft.create_reference_template(records, tmp_path / "references.csv", per_class=4)
    rows = ft.read_csv_rows(path)
    for row in rows:
        ft.save_caption_review(path, row["review_id"], TARGET_TEXT, "reviewer-test")
    return path, ft.approved_reference_pairs(path, records, minimum=2)


@pytest.fixture
def payload(data, tmp_path):
    _, pairs = approved(data, tmp_path)
    fit, holdout = ft.split_reference_pairs(pairs)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"synthetic test weights")
    settings = dict(seed=42, epochs=3, batch_size=2, accumulation=2, learning_rate=.01,
        weight_decay=0., lora_rank=2, lora_alpha=4, lora_dropout=.05,
        max_target_tokens=48, max_new_tokens=48, export_batch_size=3)
    protocol = {"settings": settings, "fit": fit, "holdout": holdout, "base_assets": ft.checkpoint_asset_identity(model)}
    return dict(output=str(tmp_path / "run"), protocol=protocol, settings=settings,
        fit=fit, holdout=holdout, model=str(model), class_names=data[2], export_records=data[1],
        clip="unused", run_clip_audit=False)


def test_official_manifest_is_balanced_disjoint_and_uses_verified_images(data):
    _, rows, names = data
    assert names == ["banded", "bubbly"]
    assert len(rows) == 24 and len({row["image_id"] for row in rows}) == 24
    assert all(set(row) == {"image_id", "image_path", "split"} for row in rows)


@pytest.mark.parametrize("bad", ["banded/../outside.png", "/banded/file.png", "unknown/file.png"])
def test_manifest_rejects_unsafe_or_unexpected_paths(data, bad):
    root, _, _ = data
    (root / "labels/train1.txt").write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        ft.load_caption_manifest(root, 1, expected_classes=2, per_class=4)


def test_manifest_rejects_cross_split_overlap(data):
    root, _, _ = data
    (root / "labels/val1.txt").write_bytes((root / "labels/train1.txt").read_bytes())
    with pytest.raises(ValueError, match="overlapping"):
        ft.load_caption_manifest(root, 1, expected_classes=2, per_class=4)


def test_template_never_converts_drafts_into_answers_and_preserves_review(data, tmp_path):
    _, records, _ = data
    draft = tmp_path / "old.csv"
    ft.atomic_csv(draft, [dict(row, raw_caption="Unverified model answer", final_caption="old shortened") for row in records],
                  ["image_id", "split", "image_path", "raw_caption", "final_caption"])
    output = ft.create_reference_template(records, tmp_path / "review.csv", per_class=2, draft_csv=draft)
    rows = ft.read_csv_rows(output)
    assert len(rows) == 4
    assert all(row["target_caption"] == row["reviewer_id"] == row["caption_source"] == "" for row in rows)
    assert all(row["review_status"] == "pending" and row["draft_caption_unverified"] == "Unverified model answer" for row in rows)
    assert all("train" in row["image_id"] for row in rows)
    ft.save_caption_review(output, rows[0]["review_id"], TARGET_TEXT, "r1")
    before = output.read_bytes()
    ft.create_reference_template(records, output, per_class=4, draft_csv=draft)
    assert output.read_bytes() == before


def test_template_sampling_is_deterministic(data, tmp_path):
    records = data[1]
    first = ft.create_reference_template(records, tmp_path / "a.csv", per_class=2)
    second = ft.create_reference_template(list(reversed(records)), tmp_path / "b.csv", per_class=2)
    assert first.read_bytes() == second.read_bytes()


def test_empty_targets_stop_before_training(data, tmp_path):
    path = ft.create_reference_template(data[1], tmp_path / "pending.csv")
    with pytest.raises(ValueError, match="NOT ground truth"):
        ft.approved_reference_pairs(path, data[1], minimum=2)


@pytest.mark.parametrize("caption", ["few words", "The image is a close up of a", "one two three four five six seven eight nine"])
def test_targets_are_checked_without_silent_truncation(caption):
    with pytest.raises(ValueError):
        ft.validate_reference_text(caption)


def test_approval_requires_human_and_teacher_provenance(data, tmp_path):
    path = ft.create_reference_template(data[1], tmp_path / "review.csv")
    review_id = ft.read_csv_rows(path)[0]["review_id"]
    with pytest.raises(ValueError, match="reviewer_id"):
        ft.save_caption_review(path, review_id, TARGET_TEXT, "")
    with pytest.raises(ValueError, match="teacher_model"):
        ft.save_caption_review(path, review_id, TARGET_TEXT, "r1", source="teacher")
    ft.save_caption_review(path, review_id, TARGET_TEXT, "r1", source="teacher", teacher_model="declared-test-teacher")
    pairs = ft.approved_reference_pairs(path, data[1], minimum=1)
    assert pairs[0]["caption_source"] == "teacher" and pairs[0]["image_sha256"]


def test_official_val_or_test_cannot_be_reference_even_when_pending(data, tmp_path):
    path = ft.create_reference_template(data[1], tmp_path / "review.csv")
    rows = ft.read_csv_rows(path)
    rows[0]["image_id"] = next(row["image_id"] for row in data[1] if row["split"] == "test")
    ft.atomic_csv(path, rows, ft.REFERENCE_COLUMNS)
    with pytest.raises(ValueError, match="Only official TRAIN"):
        ft.approved_reference_pairs(path, data[1], minimum=1)


def test_reference_fit_holdout_are_disjoint_deterministic_train_only(data, tmp_path):
    _, pairs = approved(data, tmp_path)
    fit, holdout = ft.split_reference_pairs(pairs)
    assert len(fit) == 6 and len(holdout) == 2
    assert not {row["image_id"] for row in fit} & {row["image_id"] for row in holdout}
    assert all(row["split"] == "train" for row in fit + holdout)
    assert (fit, holdout) == ft.split_reference_pairs(list(reversed(pairs)))


def test_protocol_refuses_changed_references_or_settings(payload):
    output, protocol = payload["output"], payload["protocol"]
    fingerprint = ft.lock_caption_protocol(output, protocol)
    assert fingerprint == ft.lock_caption_protocol(output, protocol)
    other = copy.deepcopy(protocol)
    other["settings"]["epochs"] += 1
    with pytest.raises(ValueError, match="Changed"):
        ft.lock_caption_protocol(output, other)


def test_payload_rechecks_train_disjointness_and_image_content(payload):
    ft.validate_caption_payload(payload)
    bad = copy.deepcopy(payload)
    bad["holdout"][0]["split"] = "test"  # deepcopy retains shared protocol references in the fixture.
    with pytest.raises(ValueError, match="official training"):
        ft.validate_caption_payload(bad)
    Path(payload["fit"][0]["image_path"]).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="image changed"):
        ft.validate_caption_payload(payload)


class FakeProcessor:
    def tokenizer(self, targets, **kwargs):
        sizes = [len(text.split()) + 2 for text in targets]
        ids = torch.zeros((len(targets), max(sizes)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, size in enumerate(sizes):
            ids[index, :size] = torch.arange(size)  # EOS=0 can be a real attended token.
            mask[index, :size] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def __call__(self, text, images, **kwargs):
        assert all(token == ft.TASK for token in text)
        return {"input_ids": torch.ones((len(images), 1), dtype=torch.long),
                "pixel_values": torch.ones((len(images), 3, 2, 2))}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "processor_config.json").write_text("{}")


def test_collator_masks_padding_by_attention_not_eos_id():
    _, labels = ft.caption_collate([(Image.new("RGB", (2, 2)), TARGET_TEXT),
                                  (Image.new("RGB", (2, 2)), "fine pale small rough flat surfaces")], FakeProcessor())
    assert labels[0, 0] == 0 and labels[1, -1] == -100
    with pytest.raises(ValueError, match="exceeds"):
        ft.caption_collate([(Image.new("RGB", (2, 2)), TARGET_TEXT)], FakeProcessor(), max_target_tokens=2)


def test_lora_targets_are_strictly_decoder_attention():
    linear = torch.nn.Linear(2, 2)
    model = SimpleNamespace(named_modules=lambda: [
        ("language_model.model.decoder.layers.0.self_attn.q_proj", linear),
        ("language_model.model.decoder.layers.0.encoder_attn.v_proj", linear),
        ("language_model.model.encoder.layers.0.self_attn.q_proj", linear),
        ("vision_tower.blocks.0.q_proj", linear)])
    assert ft.decoder_lora_targets(model) == [
        "language_model.model.decoder.layers.0.encoder_attn.v_proj",
        "language_model.model.decoder.layers.0.self_attn.q_proj"]
    with pytest.raises(ValueError, match="No Florence decoder"):
        ft.decoder_lora_targets(torch.nn.Linear(2, 2))


def test_generic_peft_loader_freezes_base_and_adapts_only_decoder(payload, monkeypatch):
    base = torch.nn.Module()
    base.vision_tower = torch.nn.Linear(2, 2)
    base.language_model = torch.nn.Module()
    base.language_model.model = torch.nn.Module()
    for role in ("encoder", "decoder"):
        tower = torch.nn.Module()
        tower.attn = torch.nn.Module()
        tower.attn.q_proj = torch.nn.Linear(2, 2)
        tower.attn.v_proj = torch.nn.Linear(2, 2)
        setattr(base.language_model.model, role, tower)
    calls = {}

    def auto_model(path, **kwargs):
        calls["load"] = kwargs
        return base

    def peft_config(**kwargs):
        calls["lora"] = kwargs
        return kwargs

    def adapt(model, config):
        assert all(not p.requires_grad for p in model.parameters())
        for name in config["target_modules"]:
            model.get_submodule(name).register_parameter("lora_A", torch.nn.Parameter(torch.ones(2, 2)))
        return model

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=auto_model),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeProcessor())))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=peft_config, PeftModel=SimpleNamespace(), get_peft_model=adapt))
    model, _ = ft.load_caption_model(payload, torch.device("cpu"), trainable=True)
    assert "task_type" not in calls["lora"]
    assert all(".decoder." in name for name in calls["lora"]["target_modules"])
    assert calls["load"]["local_files_only"] and calls["load"]["attn_implementation"] == "eager"
    assert all("lora_" in name and ".decoder." in name for name, p in model.named_parameters() if p.requires_grad)


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(.10))
        self.vision_tower = torch.nn.Identity()
        self.adapter_enabled = True

    def get_base_model(self):
        return self

    @contextmanager
    def disable_adapter(self):
        old, self.adapter_enabled = self.adapter_enabled, False
        try:
            yield
        finally:
            self.adapter_enabled = old

    def forward(self, labels, **inputs):
        selected = labels[labels != -100].float() / 100
        prediction = self.lora_weight if self.adapter_enabled else selected.new_tensor(0.)
        if self.training:
            prediction = prediction + torch.rand_like(selected) / 100
        return SimpleNamespace(loss=((prediction - selected) ** 2).mean())

    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.lora_weight.detach(), Path(path) / "adapter_model.bin")
        (Path(path) / "adapter_config.json").write_text("{}")


@pytest.fixture
def tiny_worker(monkeypatch):
    monkeypatch.setattr(ft, "load_caption_model", lambda *args, **kwargs: (TinyAdapter(), FakeProcessor()))
    monkeypatch.setattr(ft, "adapter_state", lambda model: {"weight": model.lora_weight.detach().cpu().clone()})

    def restore(model, state):
        with torch.no_grad():
            model.lora_weight.copy_(state["weight"])

    monkeypatch.setattr(ft, "restore_adapter", restore)
    monkeypatch.setattr(ft, "generate_caption_texts", lambda model, processor, rows, *args: [TARGET_TEXT] * len(rows))


def test_training_cpu_harness_saves_adapter_history_and_holdout_comparison(payload, tiny_worker):
    summary = ft.train_captioner(payload, torch.device("cpu"))
    assert summary["epochs_completed"] == 3 and not summary["downstream_f1_improvement_verified"]
    output = Path(payload["output"])
    assert (output / "best_adapter/adapter_model.bin").is_file()
    assert len(ft.read_csv_rows(output / "history.csv")) == 3
    comparisons = ft.read_csv_rows(output / "holdout_before_after.csv")
    assert len(comparisons) == 2 and all(row["human_preference"] == "" for row in comparisons)
    comparisons[0]["human_preference"] = "base"
    comparisons[0]["review_notes"] = "Human review work must survive reruns"
    ft.atomic_csv(output / "holdout_before_after.csv", comparisons, list(comparisons[0]))
    reviewed = (output / "holdout_before_after.csv").read_bytes()
    complete = ft.train_captioner(payload, torch.device("cpu"))
    assert complete == summary
    assert (output / "holdout_before_after.csv").read_bytes() == reviewed


def test_interrupted_training_resumes_same_epoch_boundary_rng_and_optimizer(payload, tiny_worker, monkeypatch):
    first_payload = copy.deepcopy(payload)
    first_payload["output"] += "_uninterrupted"
    ft.train_captioner(first_payload, torch.device("cpu"))
    original = ft.train_caption_epoch
    calls = []

    def stop_after_one(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise KeyboardInterrupt("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(ft, "train_caption_epoch", stop_after_one)
    with pytest.raises(KeyboardInterrupt):
        ft.train_captioner(payload, torch.device("cpu"))
    assert torch.load(Path(payload["output"]) / "last.pt", weights_only=False)["epoch"] == 1
    monkeypatch.setattr(ft, "train_caption_epoch", original)
    ft.train_captioner(payload, torch.device("cpu"))
    resumed = torch.load(Path(payload["output"]) / "last.pt", weights_only=False)
    reference = torch.load(Path(first_payload["output"]) / "last.pt", weights_only=False)
    assert resumed["history"] == reference["history"]
    assert torch.equal(resumed["adapter"]["weight"], reference["adapter"]["weight"])


def test_token_weighted_gradient_accumulation_handles_partial_last_group(monkeypatch):
    class Model(TinyAdapter):
        def forward(self, labels, **inputs):
            target = labels[labels != -100].float() / 10
            return SimpleNamespace(loss=((self.lora_weight - target) ** 2).mean())

    model, expected = Model(), Model()
    batches = [({}, torch.tensor([[1, 2, -100]])), ({}, torch.tensor([[3, -100, -100]])),
               ({}, torch.tensor([[4, 5, 6]]))]
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ft.train_caption_epoch(model, batches, opt, scheduler, scaler, torch.device("cpu"), accumulation=2)
    expected_opt = torch.optim.SGD(expected.parameters(), lr=.1)
    for tokens in ([1, 2, 3], [4, 5, 6]):
        expected_opt.zero_grad()
        expected(labels=torch.tensor([tokens])).loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
        expected_opt.step()
    assert torch.equal(model.lora_weight, expected.lora_weight)
    assert scheduler.last_epoch == 2


def test_export_policy_preserves_new_raw_text_and_uses_clip_diagnostic_only():
    row = ft.make_export_row({"image_id": "banded/a.png", "split": "test"}, TARGET_TEXT,
                              "x" * 64, ["banded", "bubbly"], semantic=(.99, "banded"))
    assert row["raw_caption"] == TARGET_TEXT and row["caption_valid"]
    assert row["semantic_leakage_detected"] and not row["leakage_detected"]
    no_audit = ft.make_export_row(row, TARGET_TEXT, "x", ["banded"])
    assert no_audit["semantic_leakage_score"] == "" and no_audit["semantic_audit_status"] == "not_run"
    leaked = ft.make_export_row(row, "Banded pale lines cover rough gray surfaces", "x", ["banded"])
    assert not leaked["caption_valid"] and leaked["lexical_leakage_detected"]


def prepare_export(payload):
    output = Path(payload["output"])
    adapter = output / "best_adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"test adapter")
    ft.atomic_json(output / "training_complete.json", {"identity": ft.object_hash(payload["protocol"]),
                                                      "epochs_completed": payload["settings"]["epochs"]})


def test_export_resume_generates_only_missing_images_and_complete_csv_skips_models(payload, tiny_worker, monkeypatch):
    prepare_export(payload)
    original, calls = ft.generate_caption_texts, []

    def fail_second(*args):
        calls.append(len(args[2]))
        if len(calls) == 2:
            raise RuntimeError("interrupted export")
        return original(*args)

    monkeypatch.setattr(ft, "generate_caption_texts", fail_second)
    with pytest.raises(RuntimeError, match="interrupted export"):
        ft.export_caption_csv(payload, torch.device("cpu"))
    csv_path = Path(payload["output"]) / "exports/captions.csv"
    saved = ft.read_csv_rows(csv_path)
    assert len(saved) == 3
    monkeypatch.setattr(ft, "generate_caption_texts", original)
    assert ft.export_caption_csv(payload, torch.device("cpu")) == csv_path
    completed = ft.read_csv_rows(csv_path)
    assert len(completed) == 24 and completed[:3] == saved
    assert set(completed[0]) == set(ft.EXPORT_COLUMNS)

    def forbidden(*args, **kwargs):
        raise AssertionError("Complete export must not load Florence")

    monkeypatch.setattr(ft, "load_caption_model", forbidden)
    for row in payload["export_records"]:
        path = Path(row["image_path"])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert ft.export_caption_csv(payload, torch.device("cpu")) == csv_path


def test_export_rejects_changed_assets_and_unidentified_legacy_csv(payload, tiny_worker):
    prepare_export(payload)
    path = ft.export_caption_csv(payload, torch.device("cpu"))
    before = path.read_bytes()
    adapter = Path(payload["output"]) / "best_adapter/adapter_model.safetensors"
    adapter.write_bytes(b"different adapter")
    with pytest.raises(ValueError, match="Changed caption generation"):
        ft.export_caption_csv(payload, torch.device("cpu"))
    assert path.read_bytes() == before
    (path.parent / "caption_generation_manifest.json").unlink()
    with pytest.raises(ValueError, match="no matching generation manifest"):
        ft.export_caption_csv(payload, torch.device("cpu"))


def test_export_header_only_wrong_schema_is_rejected_without_append(payload, tiny_worker):
    prepare_export(payload)
    path = ft.export_caption_csv(payload, torch.device("cpu"))
    path.write_text("image_id,split\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Missing required CSV columns"):
        ft.export_caption_csv(payload, torch.device("cpu"))
    assert path.read_bytes() == before


def test_notebook_is_reproducible_self_contained_and_does_not_change_ablations():
    old = {path: path.read_bytes() for path in project_glob("mtacs-train-vitl16-dtd-ablation-*.ipynb")}
    assert len(old) == 8
    before = NOTEBOOK.read_bytes()
    runpy.run_path(str(project_path('build_florence_dtd_caption_finetune_notebook.py')))
    assert NOTEBOOK.read_bytes() == before
    assert all(path.read_bytes() == contents for path, contents in old.items())
    notebook = json.loads(before)
    sources = []
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            source = "".join(item["source"])
            ast.parse(source)
            sources.append(source)
            assert item["execution_count"] is None and item["outputs"] == []
            assert "from dtd_caption_" not in source
    combined = "\n".join(sources)
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in combined
    assert "num_workers=0" in combined and "target_caption\": \"\"" in combined
    assert combined.index("approved_pairs = approved_reference_pairs") < combined.index("worker_env = lora_runtime_environment")
    ns = {}
    exec(sources[2], ns)  # Embedded core can be imported without Transformers or PEFT.
    assert ns["validate_reference_text"](TARGET_TEXT) == TARGET_TEXT


def test_notebook_review_stage_runs_with_empty_targets_and_stops_before_dependencies(data, tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(item["source"]) for item in notebook["cells"] if item["cell_type"] == "code"]
    ns = {"Path": Path, "os": os, "json": json, "shutil": __import__("shutil")}
    exec(sources[2], ns)
    output = tmp_path / "review_stage"
    ns.update(DTD_ROOT=data[0], DTD_SPLIT=1, OUTPUT_DIR=output,
        REVIEW_CSV=output / "review/caption_references.csv", REFERENCE_INPUT_CSV="", DRAFT_CAPTIONS_CSV="",
        REFERENCE_SAMPLE_PER_CLASS=4, REVIEW_SEED=42, MIN_APPROVED_REFERENCES=50, REFERENCE_VAL_FRACTION=.2)
    ns["load_caption_manifest"] = lambda *args: (data[1], data[2])
    exec(sources[3], ns)
    assert len(ft.read_csv_rows(ns["REVIEW_CSV"])) == 8
    with pytest.raises(ValueError, match="Only 0 approved"):
        exec(next(source for source in sources if source.startswith("approved_pairs = approved_reference_pairs")), ns)
    assert not (output / "last.pt").exists() and not (output / "caption_runtime").exists()


def runtime_namespace(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(item["source"]) for item in notebook["cells"]
                  if item["cell_type"] == "code" and "def lora_runtime_environment" in "".join(item["source"]))
    ns = {"Path": Path, "os": os, "json": json, "sys": sys, "shutil": __import__("shutil"),
          "subprocess": SimpleNamespace(run=None), "__name__": "test"}
    exec(source, ns)
    ast.parse(ns["CAPTION_FINETUNE_WORKER_SOURCE"])
    return ns


def test_runtime_installs_only_offline_peft_in_child_target(tmp_path):
    ns = runtime_namespace(tmp_path)
    original = os.environ.get("PYTHONPATH")
    calls = []
    ns["caption_runtime_environment"] = lambda *args: {"TOKENIZERS_PARALLELISM": "false"}
    ns["offline_caption_wheels"] = lambda *args: [str(tmp_path)]

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0, stdout="ok", stderr="no peft")

    ns["subprocess"].run = run
    env = ns["lora_runtime_environment"](tmp_path, tmp_path / "runtime")
    install = calls[1]
    assert "peft==0.14.0" in install and "--no-index" in install and "--no-deps" in install
    assert install[install.index("--target") + 1] == str(tmp_path / "runtime/legacy_packages")
    assert env["TOKENIZERS_PARALLELISM"] == "false" and "legacy_packages" in env["PYTHONPATH"]
    assert os.environ.get("PYTHONPATH") == original


def test_offline_wheel_zip_includes_peft_but_not_arbitrary_members(tmp_path):
    ns = runtime_namespace(tmp_path)
    archive = tmp_path / "offline_wheels.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/peft-0.14.0-py3-none-any.whl", b"test")
        handle.writestr("../../unexpected.py", b"not a wheel")
    dirs = ns["offline_caption_wheels"]([tmp_path], tmp_path / "extracted")
    assert dirs and (tmp_path / "extracted/peft-0.14.0-py3-none-any.whl").is_file()
    assert not (tmp_path / "unexpected.py").exists()


def test_no_offline_peft_wheels_has_actionable_error_and_does_not_install(tmp_path):
    ns = runtime_namespace(tmp_path)
    ns["caption_runtime_environment"] = lambda *args: {}
    ns["offline_caption_wheels"] = lambda *args: []
    commands = []

    def probe(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="no peft")

    ns["subprocess"].run = probe
    with pytest.raises(FileNotFoundError, match="peft==0.14.0"):
        ns["lora_runtime_environment"](tmp_path, tmp_path)
    assert len(commands) == 1 and "pip" not in commands[0]
