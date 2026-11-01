"""CPU checks for the Qwen export contract; no model downloads or GPU inference."""
import ast
import copy
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import build_qwen3vl_dtd_caption_notebook as builder
import dtd_caption_screening as screening
import dtd_qwen_caption as qwen
from project_paths import PROJECT_ROOT


SETTINGS = {"batch_size": 2, "dtype": "bfloat16", "min_pixels": 65536, "max_pixels": 262144,
            "max_new_tokens": 48, "seed": 42, "audit_batch_size": 2, "semantic_threshold": .30}
TEXT = "rough gray grains cover pale uneven surface"


@pytest.fixture
def payload(tmp_path):
    records = []
    for role in ("train", "val", "test"):
        for index in range(2):
            path = tmp_path / f"{role}_{index}.png"
            Image.new("RGB", (12, 12), (index * 80, 30, 40)).save(path)
            records.append({"image_id": f"banded/{role}_{index}.png", "image_path": str(path),
                            "split": role, "image_sha256": qwen.file_sha256(path)})
    return {"records": records, "class_names": ["banded", "striped"], "settings": dict(SETTINGS),
            "method": "qwen3_vl_8b_instruct", "model_path": "unused-local-test", "clip_path": "unused-clip",
            "output": str(tmp_path / "output"), "max_new_images": 0}


class Captioner:
    calls = []

    def __init__(self, path, settings):
        pass

    def generate(self, images):
        self.calls.append(len(images))
        return [(TEXT, False) for _ in images]


class Auditor:
    def __init__(self, path, names):
        pass

    def score(self, texts):
        return [(.95, "banded") for text in texts]


def test_notebook_is_standalone_offline_and_has_two_ordered_methods():
    notebook = builder.build_notebook()
    assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 19
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    for cell, source in zip(notebook["cells"], sources):
        if cell["cell_type"] == "code":
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert 'TOKENIZERS_PARALLELISM' in sources[1]
    assert '"dtype": "bfloat16"' in sources[2] and 'MAX_NEW_IMAGES = 0' in sources[2]
    assert 'run_caption_method("qwen3_vl_8b_instruct"' in sources[10]
    assert 'run_caption_method("qwen3_vl_32b_instruct"' in sources[12]
    assert "from dtd_" not in builder.embedded_core()
    namespace = {"__name__": "embedded_test"}
    exec(builder.embedded_core(), namespace)
    assert namespace["CAPTION_PROMPT"] == qwen.CAPTION_PROMPT
    assert namespace["clean_dtd_caption"]("The image shows " + TEXT) == TEXT
    assert namespace["term_hits"]("red stripes cover rough dark fabric", ["striped"], aliases=True) == ["stripes"]
    nodes = ast.parse(sources[8])
    assignment = next(node for node in nodes.body if isinstance(node, ast.Assign))
    worker = ast.literal_eval(assignment.value)
    ast.parse(worker)
    assert "worker_main(sys.argv[1], sys.argv[2])" in worker
    assert "No test accuracy or F1 has been measured" in sources[17]


def test_generated_file_is_up_to_date_and_outputs_are_empty():
    saved = json.loads(builder.TARGET.read_text(encoding="utf-8"))
    assert saved == builder.build_notebook()


def test_clean_input_is_stable_and_full_response_is_preserved():
    record = {"image_id": "banded/a.png", "split": "train"}
    response = "The image is a close-up of a " + TEXT + " beside large white stones."
    row = qwen.generation_row(record, response, "qwen3_vl_8b_instruct", "identity", ["banded"])
    assert row["source_long_caption"] == response
    assert row["raw_caption"] == row["final_caption"] == screening.clean_dtd_caption(response)
    assert 5 <= row["word_count"] <= 8 and row["caption_valid"]
    assert row["cleanup_changed_text"] and row["semantic_audit_status"] == "pending"
    assert row["semantic_leakage_score"] == ""  # Missing audits are not fabricated as zero.


@pytest.mark.parametrize("text, valid, alias", [
    (TEXT, True, False),
    ("red stripes cover rough dark fabric", True, True),
    ("striped red fabric covers rough dark stone", False, True),
    ("gray surface", False, False),
    ("", False, False),
])
def test_policy_rejects_canonical_names_but_keeps_aliases_diagnostic(text, valid, alias):
    row = qwen.generation_row({"image_id": "x", "split": "train"}, text,
        "qwen3_vl_32b_instruct", "identity", ["striped"])
    assert row["caption_valid"] == valid and row["alias_hit_diagnostic"] == alias
    audited = qwen.audited_row(row, .99, "striped", "audit", .30)
    assert audited["caption_valid"] == valid
    assert audited["leakage_detected"] == row["leakage_detected"]
    assert audited["semantic_leakage_detected"] and audited["semantic_audit_status"] == "diagnostic_only"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 1.2, -1.2])
def test_nonfinite_or_invalid_audits_stop(score):
    with pytest.raises(ValueError, match="CLIP"):
        qwen.audited_row({}, score, "banded", "audit", .30)


def test_export_round_trips_existing_dtd_lexical_v2_masks():
    # Embed the original caption-view functions without importing the full training stack.
    source = (PROJECT_ROOT / "src/training/mtacs_dtd_ablation_cells.py").read_text(encoding="utf-8")
    names = ["normalized_text", "bool_value", "DTD_EXTRA_TERMS", "caption_term_pattern", "prepare_caption_views"]
    ns = {"np": np, "pd": pd}
    exec("\n\n".join(builder.definition(source, name) for name in names), ns)
    exec(builder.definition((PROJECT_ROOT / "src/captioning/dtd_caption_screening.py").read_text(encoding="utf-8"),
                            "prepare_standalone_caption_views"), ns)
    ns["clean_dtd_caption"] = screening.clean_dtd_caption
    rows = []
    for index, text in enumerate((TEXT, "red stripes cover rough dark fabric", "striped red fabric covers rough dark stone", "gray")):
        row = qwen.generation_row({"image_id": str(index), "split": "train"}, text,
            "qwen3_vl_8b_instruct", "identity", ["striped"])
        rows.append(qwen.audited_row(row, .95, "striped", "audit", .30))
    audit, views = ns["prepare_standalone_caption_views"](pd.DataFrame(rows), ["striped"], .30, "dtd_lexical_v2")
    assert views["screened"]["valid"].tolist() == [True, True, False, False]
    assert audit.final_caption.tolist() == [row["final_caption"] for row in rows]
    assert audit.semantic_score_matches_input.all()


def test_prompt_contains_pixels_and_fixed_instruction_not_paths_or_category():
    image = Image.new("RGB", (4, 4))
    messages = qwen.caption_messages([image])
    assert messages == [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": qwen.CAPTION_PROMPT}]}]
    assert all(word not in qwen.CAPTION_PROMPT.lower() for word in ("banded", "striped", "train1", "dtd", "image_id"))


def test_qwen_batch_trims_padded_prompts_before_decoding():
    seen = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            return ["rendered"] * len(messages)

        def __call__(self, **kwargs):
            seen["processor"] = kwargs
            return {"input_ids": torch.ones((2, 4), dtype=torch.long),
                    "attention_mask": torch.ones((2, 4), dtype=torch.long),
                    "token_type_ids": torch.zeros((2, 4), dtype=torch.long)}

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[19, 7], [21, 22]]
            return [TEXT, TEXT]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=7)

        def generate(self, **kwargs):
            seen["generate"] = kwargs
            return torch.cat([kwargs["input_ids"], torch.tensor([[19, 7], [21, 22]])], dim=1)

    model = qwen.FrozenQwenCaptioner.__new__(qwen.FrozenQwenCaptioner)
    model.device = torch.device("cpu")
    model.settings = dict(SETTINGS, max_new_tokens=2)
    model.processor, model.model = Processor(), Model()
    results = model.generate([Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))])
    assert results == [(TEXT, False), (TEXT, True)]
    assert seen["generate"]["do_sample"] is False and seen["generate"]["num_beams"] == 1
    assert "token_type_ids" not in seen["generate"]
    assert seen["processor"]["max_pixels"] == SETTINGS["max_pixels"]


def test_smoke_test_resume_and_audit_only_publish_complete_exports(payload):
    Captioner.calls = []
    payload["max_new_images"] = 2
    qwen.generate_records(payload, "generation", Captioner)
    status = qwen.audit_records(payload, "generation", "audit", Auditor)
    assert not status["complete"] and status["rows"] == 2
    assert not (Path(payload["output"]) / "captions.csv").exists()
    payload["max_new_images"] = 0
    qwen.generate_records(payload, "generation", Captioner)
    status = qwen.audit_records(payload, "generation", "audit", Auditor)
    assert status["complete"] and status["rows"] == 6 and sum(Captioner.calls) == 6
    rows = qwen.read_csv_rows(status["csv_path"])
    assert [row["image_id"] for row in rows] == [row["image_id"] for row in payload["records"]]
    assert all(qwen.bool_value(row["caption_valid"]) and qwen.bool_value(row["semantic_leakage_detected"]) for row in rows)
    qwen.generate_records(payload, "generation", Captioner)
    qwen.audit_records(payload, "generation", "audit", Auditor)
    assert sum(Captioner.calls) == 6  # No repeated generation on resume.
    summary, exports = qwen.summarize_exports({payload["method"]: payload["output"]}, payload["records"])
    assert len(summary) == 3 and len(exports[payload["method"]]) == 6
    assert all(row["valid_rate"] == 1 for row in summary)


def test_non_oom_failure_is_visible_and_completed_images_are_resumable(payload):
    class Interrupted(Captioner):
        call_count = 0

        def generate(self, images):
            self.call_count += 1
            if self.call_count == 2:
                raise RuntimeError("synthetic inference error")
            return super().generate(images)

    with pytest.raises(RuntimeError, match="synthetic inference error"):
        qwen.generate_records(payload, "generation", Interrupted)
    assert len(qwen.available_rows(Path(payload["output"]) / "generated", payload["records"], "generation")) == 2
    Captioner.calls = []
    qwen.generate_records(payload, "generation", Captioner)
    assert sum(Captioner.calls) == 4


def test_oom_reduces_batch_without_skipping_or_duplicating_images(payload):
    class SmallGPU(Captioner):
        def generate(self, images):
            if len(images) > 1:
                raise torch.cuda.OutOfMemoryError("synthetic OOM")
            return super().generate(images)

    Captioner.calls = []
    payload["settings"]["batch_size"] = 4
    qwen.generate_records(payload, "generation", SmallGPU)
    rows = qwen.available_rows(Path(payload["output"]) / "generated", payload["records"], "generation")
    assert len(rows) == 6 and sum(Captioner.calls) == 6
    assert all(row["effective_batch_size"] == 1 for row in rows)


def test_changed_image_and_cross_method_cache_are_rejected(payload):
    payload["records"][0]["image_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Image changed"):
        qwen.generate_records(payload, "generation", Captioner)
    record = payload["records"][0]
    cache = Path(payload["output"]) / "generated"
    qwen.atomic_json(qwen.cache_path(cache, record["image_id"]), {
        "identity": "wrong-method", "row": record})
    with pytest.raises(ValueError, match="Wrong cached"):
        qwen.cached_row(cache, record, "generation")


def test_protocol_identity_rejects_changes_and_unowned_existing_output(tmp_path):
    path = tmp_path / "run/protocol.json"
    protocol = {"prompt": "locked", "model": "8B"}
    identity = qwen.lock_protocol(path, protocol)
    assert identity == qwen.lock_protocol(path, copy.deepcopy(protocol))
    with pytest.raises(ValueError, match="Changed"):
        qwen.lock_protocol(path, dict(protocol, model="32B"))
    path.unlink()
    (path.parent / "captions.csv").write_text("old output", encoding="utf-8")
    with pytest.raises(ValueError, match="without its protocol"):
        qwen.lock_protocol(path, protocol)


def fake_checkpoint(path, parameters=8_000_000_000):
    path.mkdir(parents=True)
    for name, value in {
        "config.json": {"model_type": "qwen3_vl"},
        "tokenizer_config.json": {"chat_template": "test-only-template"},
        "preprocessor_config.json": {}, "tokenizer.json": {},
    }.items():
        (path / name).write_text(json.dumps(value), encoding="utf-8")
    # Synthetic header only: no model is loaded from this test fixture.
    header = json.dumps({"weight": {"shape": [parameters], "dtype": "BF16", "data_offsets": [0, 0]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def test_asset_resolution_checks_model_scale_and_never_substitutes_another_model(tmp_path):
    path = fake_checkpoint(tmp_path / "Qwen3-VL-8B-Instruct")
    assert qwen.resolve_qwen_model("", "qwen3_vl_8b_instruct", [tmp_path]) == path
    assert qwen.validate_qwen_folder(path, "qwen3_vl_8b_instruct")[1] == 8_000_000_000
    with pytest.raises(ValueError, match="Wrong checkpoint size"):
        qwen.validate_qwen_folder(path, "qwen3_vl_32b_instruct")
    with pytest.raises(FileNotFoundError, match="32B-Instruct"):
        qwen.resolve_qwen_model("", "qwen3_vl_32b_instruct", [tmp_path])
    (path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="shard"):
        qwen.validate_qwen_folder(path, "qwen3_vl_8b_instruct")


def test_missing_safetensors_shard_and_thinking_are_not_accepted(tmp_path):
    path = fake_checkpoint(tmp_path / "Qwen3-VL-8B-Instruct")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"missing": "missing.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        qwen.validate_qwen_folder(path, "qwen3_vl_8b_instruct")
    thinking = fake_checkpoint(tmp_path / "Qwen3-VL-8B-Thinking")
    with pytest.raises(ValueError, match="Thinking"):
        qwen.validate_qwen_folder(thinking, "qwen3_vl_8b_instruct")


def test_chat_template_changes_are_fingerprinted(tmp_path):
    path = fake_checkpoint(tmp_path / "model")
    template = path / "chat_template.jinja"
    template.write_text("first", encoding="utf-8")
    old = qwen.qwen_asset_identity(path)
    template.write_text("second", encoding="utf-8")
    assert old["chat_template.jinja"] != qwen.qwen_asset_identity(path)["chat_template.jinja"]


def test_official_separate_chat_template_json_is_accepted_and_hashed(tmp_path):
    path = fake_checkpoint(tmp_path / "Qwen3-VL-8B-Instruct")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="chat template"):
        qwen.validate_qwen_folder(path, "qwen3_vl_8b_instruct")
    template = path / "chat_template.json"
    template.write_text(json.dumps({"chat_template": "official-format-template"}), encoding="utf-8")
    assert qwen.validate_qwen_folder(path, "qwen3_vl_8b_instruct")[1] == 8_000_000_000
    assert "chat_template.json" in qwen.qwen_asset_identity(path)


def test_empty_separate_chat_template_does_not_bypass_validation(tmp_path):
    path = fake_checkpoint(tmp_path / "Qwen3-VL-8B-Instruct")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="chat template"):
        qwen.validate_qwen_folder(path, "qwen3_vl_8b_instruct")


@pytest.mark.parametrize("key,value", [("batch_size", 0), ("max_pixels", -1), ("seed", -5),
                                      ("dtype", "int4"), ("semantic_threshold", float("nan"))])
def test_bad_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        qwen.validate_settings(dict(SETTINGS, **{key: value}))


def test_offline_environment_is_fixed_before_child_tokenizer_imports():
    env = qwen.worker_environment()
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    with pytest.raises(FileNotFoundError, match="site-packages|transformers"):
        qwen.worker_environment("this-runtime-folder-does-not-exist")


def test_clip_audit_is_text_to_class_not_image_caption_reranking():
    auditor = qwen.ClipClassAuditor.__new__(qwen.ClipClassAuditor)
    auditor.names = ["banded", "striped"]
    auditor.class_vectors = torch.eye(2)
    auditor.encode = lambda texts: torch.tensor([[.6, .8], [1., 0.]])
    result = auditor.score([TEXT, TEXT])
    assert result[0][1] == "striped" and result[0][0] == pytest.approx(.8)
    assert result[1] == (1., "banded")


def test_native_qwen_loader_is_offline_sdpa_and_rejects_missing_learned_weights(monkeypatch):
    seen = {}
    model = SimpleNamespace(eval=lambda: SimpleNamespace(requires_grad_=lambda enabled: model))

    def load_processor(*args, **kwargs):
        seen["processor"] = kwargs
        return SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    def load_model(*args, **kwargs):
        seen["model"] = kwargs
        return model, seen.get("loading", {})

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    captioner = qwen.FrozenQwenCaptioner("local-only", SETTINGS)
    assert captioner.processor.tokenizer.padding_side == "left"
    assert seen["processor"]["local_files_only"] and not seen["processor"]["trust_remote_code"]
    assert seen["model"]["local_files_only"] and seen["model"]["attn_implementation"] == "sdpa"
    assert seen["model"]["dtype"] == torch.bfloat16 and seen["model"]["device_map"] == {"": "cuda:0"}
    seen["loading"] = {"missing_keys": ["model.language_model.layers.0.self_attn.q_proj.weight"]}
    with pytest.raises(ValueError, match="Incomplete Qwen"):
        qwen.FrozenQwenCaptioner("local-only", SETTINGS)


def test_runtime_preflight_reports_missing_gpu_and_missing_native_qwen(monkeypatch):
    result = SimpleNamespace(returncode=0, stdout=json.dumps({"cuda": False}), stderr="")
    monkeypatch.setattr(qwen.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        qwen.check_worker_runtime(qwen.worker_environment())
    result.returncode, result.stderr = 1, "ImportError: Qwen3VLForConditionalGeneration"
    with pytest.raises(RuntimeError, match="4.49 wheels cannot run Qwen3-VL"):
        qwen.check_worker_runtime(qwen.worker_environment())


def test_weight_index_cannot_reference_paths_outside_model_directory(tmp_path):
    path = fake_checkpoint(tmp_path / "model")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="shard filenames"):
        qwen.weight_files(path)
