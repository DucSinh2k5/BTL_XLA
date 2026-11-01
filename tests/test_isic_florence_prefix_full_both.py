"""CPU-only checks for the caption-backend swap; no pretrained weights/downloads."""
import ast
import hashlib
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import build_isic_florence_prefix_full_both_notebook as builder
import build_isic_qwen8b_full_both_notebook as control
import isic_florence_prefix as backend


TEXT = "dark brown skin with uneven pale edges"


@pytest.fixture
def core():
    namespace = {"__name__": "embedded_florence_test"}
    exec(builder.embed_caption_core(), namespace)
    return namespace


def sections(notebook):
    return {cell["metadata"]["section"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_notebook_is_current_standalone_and_has_separate_output():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert len({cell["id"] for cell in notebook["cells"]}) == 19
    cells = sections(notebook)
    assert list(cells).index("generate_and_encode") < list(cells).index("multimodal_model")
    for setting in ("IMAGE_SIZE = 448", "EPOCHS = 100", "PATIENCE = 15", "BATCH_SIZE = 8",
                    "CANDIDATE_POOL_FRACTION = 0.20", "mtacs_isic2017_448_florence_prefix_full_both"):
        assert setting in cells["configuration"]
    assert "QWEN_8B_MODEL_PATH" not in cells["configuration"]
    assert "notebook-mt-moi/Florence-2-base" in cells["configuration"]
    assert notebook["metadata"]["isic_full_both"]["context_k"] == 1
    assert notebook["metadata"]["isic_full_both"]["caption_method"] == "florence_prefix_control"
    assert "Florence results are measured by this run only" in cells["test_policy"]


def test_shared_training_model_dataset_and_loss_are_unchanged():
    new, old = sections(builder.build_notebook()), sections(control.build_notebook())
    for name in ("standard_splits", "visual_transforms", "aligned_text_banks", "architecture", "multimodal_model",
                 "frozen_visual_banks", "unchanged_loss_and_loaders", "train_and_resume", "visualize_query_context_text_masks"):
        assert new[name] == old[name], name
    new_protocol = "\n".join(new["locked_protocol_and_evaluation"].splitlines()[1:])
    old_protocol = "\n".join(old["locked_protocol_and_evaluation"].splitlines()[1:])
    assert new_protocol.replace("isic2017_florence_prefix_full_both_448", "isic2017_qwen8b_full_both_448") == old_protocol


def test_original_executed_notebooks_and_qwen_template_are_untouched():
    expected = {"tacs_isic_segement_448.ipynb": "052b4edf0110f2caf6d1b2e9a1d8619dfdfb57d56d97c8768553b9a566215599",
                "isic-segmen-captions.ipynb": "32779936c8509b0d45c201d6139d06dcea2bfcc591a8409e7ed2240fa5758847"}
    for name, digest in expected.items():
        assert hashlib.sha256((builder.PROJECT_ROOT / name).read_bytes()).hexdigest() == digest
    assert json.loads(control.TARGET.read_text(encoding="utf-8")) == control.build_notebook()


def test_caption_core_embeds_no_project_or_qwen_model_imports(core):
    source = builder.embed_caption_core()
    assert core["MODEL_IDS"] == {"florence_prefix_control": "microsoft/Florence-2-base"}
    assert core["CAPTION_PROMPT"] == "<MORE_DETAILED_CAPTION>"
    for forbidden in ("FrozenQwenCaptioner", "Qwen3VLForConditionalGeneration", "validate_qwen_folder", "resolve_qwen_model"):
        assert forbidden not in source
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("dtd_", "isic_"))
                   for node in ast.walk(ast.parse(source)))


@pytest.mark.parametrize("response,limited,valid,clinical", [
    ("The image is a close up of " + TEXT, False, True, False),
    (TEXT, True, False, False),
    ("brown skin", False, False, False),
    (TEXT + " melanoma", False, False, True),
    ("striped brown region with dotted pale border", False, True, False),
])
def test_prefix_cleanup_keeps_isic_policy_and_checks_full_response(core, response, limited, valid, clinical):
    caption, actual, found, count = core["caption_policy"](response, limited)
    assert (actual, found) == (valid, clinical)
    assert count <= 8
    if actual:
        assert count >= 5 and not caption.lower().startswith(("a ", "an ", "the "))


@pytest.fixture
def payload(tmp_path, core):
    records = []
    for role in ("train", "val", "test"):
        path = tmp_path / (role + ".png")
        Image.new("RGB", (16, 16), (120, 60, 40)).save(path)
        records.append({"image_id": role, "image_path": str(path), "split": role,
                        "image_sha256": core["file_sha256"](path)})
    return {"records": records, "settings": {"batch_size": 2, "max_new_tokens": 48, "seed": 42},
            "output": str(tmp_path / "captions"), "model_path": "mock", "max_new_images": 0}


def test_generation_resume_oom_and_provenance(core, payload):
    seen = []

    class FakeFlorence:
        def __init__(self, path, settings):
            pass

        def generate(self, images):
            if len(images) > 1:
                raise torch.cuda.OutOfMemoryError("synthetic")
            seen.append(len(images))
            return [("The image shows " + TEXT, False) for _ in images]

    payload["max_new_images"] = 1
    result = core["generate_isic_records"](payload, "identity", FakeFlorence)
    assert not result["complete"] and not (Path(payload["output"]) / "captions.csv").exists()
    with pytest.raises(RuntimeError, match="partial"):
        core["validated_export"](payload)
    payload["max_new_images"] = 0
    core["generate_isic_records"](payload, "identity", FakeFlorence)
    rows, status = core["validated_export"](payload)
    assert status["complete"] and len(rows) == 3 and sum(seen) == 3
    core["generate_isic_records"](payload, "identity", FakeFlorence)
    assert sum(seen) == 3
    assert all(row["vlm_model"] == backend.MODEL_ID and row["final_caption"] == TEXT for row in rows)
    assert all(row["source_long_caption"] == "The image shows " + TEXT for row in rows)
    bad = dict(payload["records"][0], mask_path="not allowed")
    with pytest.raises(ValueError, match="never masks"):
        core["validate_caption_records"]([bad])


def test_pipeline_reuses_existing_generation_and_clip_code_with_only_backend_labels_changed():
    new, old = builder.embed_caption_core(), control.embed_caption_core()
    for name in ("caption_policy", "validate_caption_records", "validate_text_cache", "make_generation_row", "encode_isic_texts"):
        assert builder.definition(new, name) == builder.definition(old, name), name
    adapted = builder.definition(new, "generate_isic_records")
    adapted = adapted.replace("IsicFlorencePrefixCaptioner", "IsicQwenCaptioner")
    adapted = adapted.replace("ISIC Florence-Prefix", "ISIC Qwen3-VL-8B").replace("Florence OOM", "Qwen OOM")
    assert adapted == builder.definition(old, "generate_isic_records")


@pytest.mark.parametrize("sequence,limit", [([2, 7, 8, 9], True), ([2, 7, 2, 1], False)])
def test_florence_generate_is_greedy_pixel_only_and_excludes_decoder_start_eos(sequence, limit):
    observed = {}

    class Processor:
        def __call__(self, **kwargs):
            observed["processor"] = kwargs
            return {"input_ids": torch.ones((1, 3), dtype=torch.long), "pixel_values": torch.ones(1, 3, 8, 8)}

        def batch_decode(self, tokens, **kwargs):
            return ["raw sequence"]

        def post_process_generation(self, text, **kwargs):
            observed["post"] = kwargs
            return {backend.TASK_PROMPT: "The image shows " + TEXT}

    class Model:
        generation_config = SimpleNamespace(eos_token_id=2)

        def generate(self, **kwargs):
            observed["generate"] = kwargs
            return torch.tensor([sequence])

    captioner = backend.IsicFlorencePrefixCaptioner.__new__(backend.IsicFlorencePrefixCaptioner)
    captioner.device, captioner.dtype = torch.device("cpu"), torch.float32
    captioner.settings = {"max_new_tokens": 3}
    captioner.processor, captioner.model = Processor(), Model()
    with Image.new("RGB", (12, 10)) as image:
        assert captioner.generate([image]) == [("The image shows " + TEXT, limit)]
        assert observed["processor"]["images"] == [image]
    assert observed["processor"]["text"] == ["<MORE_DETAILED_CAPTION>"]
    assert observed["generate"]["num_beams"] == 1 and observed["generate"]["do_sample"] is False
    assert observed["post"]["image_size"] == (12, 10)


@pytest.fixture
def fake_checkpoint(tmp_path):
    config = {"model_type": "florence2", "auto_map": {"AutoModelForCausalLM": "modeling_florence2.Florence2ForConditionalGeneration"}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py",
                 "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    header = json.dumps({"weights": {"shape": [230_000_000], "dtype": "F16"}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return tmp_path


def test_checkpoint_validates_original_base_not_native_large_or_adapter(fake_checkpoint):
    path, count = backend.validate_florence_prefix_folder(fake_checkpoint)
    assert path == fake_checkpoint and count == 230_000_000
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="LoRA"):
        backend.validate_florence_prefix_folder(path)
    (path / "adapter_config.json").unlink()
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = {}
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="legacy"):
        backend.validate_florence_prefix_folder(path)


def test_legacy_setup_is_offline_target_only_and_leaves_parent_environment_unchanged(core, tmp_path, monkeypatch):
    calls = []
    parent_env = dict(os.environ)

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0, stdout="4.49.0", stderr="modern parent")

    monkeypatch.setattr(core["subprocess"], "run", run)
    core["offline_caption_wheels"] = lambda roots, dest: [str(tmp_path / "wheels")]
    env = core["caption_runtime_environment"]("legacy", tmp_path, tmp_path / "runtime")
    command = calls[1][0]
    assert "--no-index" in command and "--no-deps" in command and "--target" in command
    assert "transformers==4.49.0" in command and "tokenizers>=0.21,<0.22" in command
    assert str(tmp_path / "runtime" / "legacy_packages") in command
    assert str(tmp_path / "runtime" / "legacy_packages") in env["PYTHONPATH"]
    assert env["TOKENIZERS_PARALLELISM"] == "false" and env["HF_HUB_OFFLINE"] == "1"
    assert dict(os.environ) == parent_env
    launch = sections(builder.build_notebook())["generate_and_encode"]
    assert "'generate', florence_env" in launch and "'encode', clip_env" in launch
    assert "clip_env = dict(os.environ)" in launch


def test_encode_phase_does_not_require_legacy_florence_runtime(core, payload, tmp_path, monkeypatch):
    called = []
    request = tmp_path / "request.json"
    core["atomic_json"](request, payload)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    core["validate_florence_prefix_folder"] = lambda path: pytest.fail("encode must not load Florence weights")
    core["florence_worker_main"](request, "encode", generate_records=lambda *args: pytest.fail("wrong phase"),
                                 encode_texts=lambda value: called.append(value))
    assert called == [payload]


def test_comparison_contains_measured_historical_values_but_no_new_florence_test_value():
    notebook = builder.build_notebook()
    cells = sections(notebook)
    assert "'test_dice': 0.848689" in cells["test_and_comparison"]
    assert "'test_iou': 0.765407" in cells["test_and_comparison"]
    assert "'Florence-Prefix full_both 448', 'test_dice': metrics['test_dice']" in cells["test_and_comparison"]
    assert "'ISIC Florence-Prefix full_both 448'" in cells["test_and_comparison"]
    assert "historical seed42" in cells["test_and_comparison"]
    assert "dice_delta_vs_qwen8b_pp" in cells["test_and_comparison"]
