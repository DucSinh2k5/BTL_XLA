"""Exercise caption-only notebook, real policy, retry selection and interrupted resume on CPU."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/notebook_builders"))
import build_isic_caption_firstvalid_v3_notebook as builder


def core_module():
    module = types.ModuleType("test_firstvalid_core")
    exec(compile(builder.runtime_source(), module.__name__, "exec"), module.__dict__)
    return module


def fixture_payload(tmp_path, core):
    source, model = tmp_path / "old_bundle", tmp_path / "florence"
    source.mkdir(); model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixture-no-model-loading")
    records = []
    for i, role in enumerate(("train", "train", "val", "test")):
        image = tmp_path / "dataset" / (f"ISIC_{i:07d}.png")
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (20, 20), (i, 30, 20)).save(image)
        records.append({"image_id": image.stem, "split": role, "image_path": str(image),
                        "image_sha256": core.file_sha256(image)})
    protocol = {"method": "florence_extractive_attributes_v2", "policy": core.POLICY,
                "model_assets": core.checkpoint_asset_identity(model),
                "images": [{k: r[k] for k in ("image_id", "split", "image_sha256")} for r in records]}
    identity = core.object_hash(protocol)
    original = ["A small dark brown circular area.",
                "A large light brown circular area. This could be melanoma.",
                "The image is a close-up of skin. The skin appears to be.",
                "The image shows skin."]
    rows = [core.make_generation_row(r, s, False, identity) for r, s in zip(records, original)]
    core.atomic_csv(source / "captions.csv", rows, core.COLUMNS)
    core.atomic_json(source / "caption_provenance/generation_protocol/protocol.json", protocol)
    core.atomic_json(source / "caption_provenance/status.json", {
        "complete": True, "identity": identity, "csv_sha256": core.file_sha256(source / "captions.csv")})
    payload = {"records": records, "model_path": str(model), "prior_captions_source": str(source),
               "output": str(tmp_path / "new"), "settings": {
                   "seed": 42, "device": "cpu", "precision": "float32", "batch_size": 4,
                   "max_new_tokens": 96, "retry_greedy_tokens": 128, "retry_beam_tokens": 192,
                   "retry_beams": 4, "retry_returns": 3}}
    return payload, protocol, rows


def scripted_factory(calls, interrupt_beam=False):
    class Captioner:
        def __init__(self, *args):
            calls.append(("load",))

        def generate_candidates(self, images, *, max_new_tokens, num_beams, num_return_sequences):
            ids = [image.getpixel((0, 0))[0] for image in images]
            calls.append((num_beams, ids))
            if num_beams > 1 and interrupt_beam:
                raise RuntimeError("simulated interruption after greedy commit")
            output = []
            for i in ids:
                if num_beams == 1:
                    responses = ["A rough uneven dark brown surface." if i == 1 else "The image shows skin."]
                elif i == 2:
                    # First response contains a banned diagnosis despite valid attributes.
                    responses = ["A small dark brown circular area. This could be melanoma.",
                                 "A large light brown circular area.", "A small dark brown circular area."]
                else:
                    responses = ["The image shows skin."] * num_return_sequences
                output.append([{"response": text, "token_limit": False, "new_tokens": 20, "rank": rank}
                               for rank, text in enumerate(responses)])
            return output
    return Captioner


def test_caption_only_notebook_is_self_contained_and_does_not_require_segmentation_assets():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text("utf8")) == notebook
    assert len(notebook["cells"]) == 11
    sections = {c["metadata"]["section"]: "".join(c["source"]) for c in notebook["cells"]}
    assert "CAPTIONS_ONLY = True" in sections["configuration"]
    assert "ENCODE_CLIP_TEXT = False" in sections["configuration"]
    assert "no_segmentation_training" in notebook["metadata"]["isic_caption_firstvalid_v3"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    core = core_module()
    assert core.METHOD == builder.METHOD and core.POLICY == "isic_visible_attributes_v2"
    assert "train_one_epoch" not in builder.runtime_source()
    assert "DINOV3_REPO_DIR" not in "".join(sections.values())
    assert "TOKENIZERS_PARALLELISM" in sections["configuration"]


def test_retry_preserves_valid_baseline_filters_clinical_response_and_reports_mask_change(tmp_path):
    core = core_module()
    payload, protocol, old_rows = fixture_payload(tmp_path, core)
    source = Path(payload["prior_captions_source"])
    before = {str(p.relative_to(source)): core.file_sha256(p) for p in source.rglob("*") if p.is_file()}
    calls = []
    identity = core.object_hash({"settings": payload["settings"], "images": protocol["images"]})
    result = core.generate_firstvalid_records(payload, identity, scripted_factory(calls))
    rows, status = core.validated_export(payload)
    assert result == status
    assert (status["baseline_v2_masked"], status["recovered_from_v2"], status["final_masked"]) == (3, 2, 1)
    assert calls == [("load",), (1, [1, 2, 3]), (4, [2, 3])]
    assert rows[0]["final_caption"] == old_rows[0]["final_caption"]
    assert rows[0]["retry_count"] == "0" and rows[1]["selection_origin"] == "greedy_retry"
    assert rows[2]["selection_origin"] == "beam_retry" and rows[2]["retry_count"] == "4"
    assert rows[2]["final_caption"] == "large light brown circular area"
    assert rows[3]["caption_valid"] == "False" and rows[3]["selection_origin"] == "still_masked_after_retries"
    trace = json.loads(core.firstvalid_audit_path(payload["output"], rows[2]["image_id"]).read_text("utf8"))
    assert trace["rounds"][1]["candidates"][0]["row"]["clinical_term_detected"]
    assert not trace["rounds"][1]["candidates"][0]["row"]["caption_valid"]
    report = core.write_firstvalid_reports(payload["output"])
    assert report["filters_relaxed"] is False and report["segmentation_training_performed"] is False
    assert report["mask_comparison"][0]["baseline_v2_masked"] == 3
    assert report["mask_comparison"][0]["firstvalid_masked"] == 1
    assert before == {str(p.relative_to(source)): core.file_sha256(p) for p in source.rglob("*") if p.is_file()}
    # A completed run resumes without loading Florence again.
    def forbidden(*args):
        raise AssertionError("No model loading for completed retries")
    assert core.generate_firstvalid_records(payload, identity, forbidden) == result
    changed = dict(payload, records=[dict(payload["records"][0], mask_path="forbidden")])
    with pytest.raises(ValueError, match="pixels/identity only"):
        core.validate_caption_records(changed["records"])


def test_interrupted_beam_resumes_at_pending_round_and_repairs_final_atomic_commit(tmp_path):
    core = core_module()
    payload, _, _ = fixture_payload(tmp_path, core)
    calls = []
    with pytest.raises(RuntimeError, match="simulated interruption"):
        core.generate_firstvalid_records(payload, "same-locked-identity", scripted_factory(calls, True))
    assert calls == [("load",), (1, [1, 2, 3]), (4, [2, 3])]
    # Simulate interruption between completed trace commit and final row commit.
    core.cache_path(Path(payload["output"]) / "generated", payload["records"][1]["image_id"]).unlink()
    resumed = []
    result = core.generate_firstvalid_records(payload, "same-locked-identity", scripted_factory(resumed))
    assert resumed == [("load",), (4, [2, 3])]
    assert result["rows"] == 4 and result["recovered_from_v2"] == 2
    with pytest.raises(ValueError, match="another image/protocol"):
        core.generate_firstvalid_records(payload, "changed-identity", scripted_factory([]))


def test_source_discovery_image_hashes_and_notebook_download_cells(tmp_path, monkeypatch):
    core = core_module()
    payload, protocol, _ = fixture_payload(tmp_path, core)
    source, recovered_protocol, status = core.resolve_prior_source(payload["prior_captions_source"], [])
    assert recovered_protocol == protocol
    recovered = core.recover_image_records(protocol, str(tmp_path / "dataset"), [])
    assert recovered == payload["records"]
    # Even if another copy exists, equal CSVs are one source, not ambiguous methods.
    import shutil
    shutil.copytree(source, tmp_path / "copied_bundle")
    assert core.resolve_prior_source("", [tmp_path])[2]["csv_sha256"] == status["csv_sha256"]
    changed = json.loads(json.dumps(protocol))
    changed["images"][0]["image_sha256"] = "0" * 64
    with pytest.raises(FileNotFoundError, match="matching hash"):
        core.recover_image_records(changed, str(tmp_path / "dataset"), [])
    core.generate_firstvalid_records(payload, "integration-identity", scripted_factory([]))
    core.write_firstvalid_reports(payload["output"])
    output = Path(payload["output"])
    runtime = output / "caption_firstvalid_runtime.py"
    runtime.write_text(builder.runtime_source(), encoding="utf8")
    core.atomic_json(output / "generation_protocol/protocol.json", {"test": "protocol-fixture"})
    captions, caption_status = core.validated_export(payload)
    scope = {"core": core, "OUTPUT_DIR": output, "SETTINGS": payload["settings"],
             "CAPTION_STATUS": caption_status, "SOURCE_STATUS": status, "ENCODE_CLIP_TEXT": False}
    # Notebook display is supplied by Kaggle; only replace rendering on this CPU test host.
    display_module = types.ModuleType("IPython.display")
    display_module.FileLink = str
    display_module.display = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    exec(builder.DOWNLOADS, scope)
    with zipfile.ZipFile(output / "isic2017_captions_firstvalid_v3.zip") as handle:
        assert "captions.csv" in handle.namelist()
        assert "caption_firstvalid_runtime.py" in handle.namelist()
        assert "generation_protocol/protocol.json" in handle.namelist()
        assert len([p for p in handle.namelist() if p.startswith("retry_audit/")]) == len(captions)
        assert not any(p.endswith(".safetensors") for p in handle.namelist())


def test_grouped_beam_outputs_and_forced_eos_length_cap():
    import torch
    core = core_module()
    captioner = core.IsicFlorenceFirstValidCaptioner.__new__(core.IsicFlorenceFirstValidCaptioner)
    captioner.device = torch.device("cpu"); captioner.dtype = torch.float32
    class Processor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(2, 3, 2, 2)}
        def batch_decode(self, outputs, **kwargs):
            return [f"candidate{i}" for i in range(len(outputs))]
        def post_process_generation(self, text, **kwargs):
            return {core.TASK_PROMPT: text}
    class Model:
        generation_config = types.SimpleNamespace(eos_token_id=2, pad_token_id=1)
        def generate(self, **kwargs):
            # decoder_start=2; first caption ends early + pads, next hits cap with forced EOS.
            return torch.tensor([[2, 7, 2, 1, 1], [2, 7, 8, 9, 2], [2, 8, 2, 1, 1], [2, 8, 9, 10, 2]])
    captioner.model = Model(); captioner.processor = Processor()
    groups = captioner.generate_candidates([Image.new("RGB", (2, 2)) for _ in range(2)],
        max_new_tokens=4, num_beams=2, num_return_sequences=2)
    assert [[r["response"] for r in group] for group in groups] == [["candidate0", "candidate1"], ["candidate2", "candidate3"]]
    assert [[r["token_limit"] for r in group] for group in groups] == [[False, True], [False, True]]
