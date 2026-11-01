
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast
import copy
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import dtd_grounded_caption as grounded


ROOT = PROJECT_ROOT
NOTEBOOK = project_path('florence2-dtd-caption-grounded-rerank.ipynb')
RAW = "The image is a close up of a rough pale surface with dense dark specks. Fine gray grains cover uneven pale material."
BEAM = "Bright pale fibers form dense vertical columns beside scattered dark patches."
NAMES = ["banded", "bubbly"]


def scoring(texts):
    return {text: {"image_cosine": (sum(ord(char) for char in text) % 997) / 1000,
                   "class_cosine": .95, "class_name": "banded"} for text in texts if text.strip()}


def test_spans_keep_budget_and_reject_canonical_terms_and_dangling_edges():
    raw = "The image is a banded surface with pale colors. Dense red pebbles cover dark rough stone."
    spans = grounded.candidate_spans(raw, NAMES)
    assert spans and all(grounded.valid_short(span["text"], NAMES) for span in spans)
    assert all("banded" not in span["text"].lower() for span in spans)
    assert all(span["text"].split()[0].lower() not in grounded.BAD_START for span in spans)
    assert all(span["text"].split()[-1].lower() not in grounded.EDGE_WORDS for span in spans)
    assert all(span["sentence"] in (0, 1) for span in spans)
    for span in spans:
        sentence = grounded.re.split(r"(?<=[.!?;:])\s+", raw)[span["sentence"]]
        words = grounded.re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?", grounded.strip_opening(sentence))
        assert " ".join(words[span["word_start"]:span["word_end"]]) == span["text"]


def test_spans_normalized_apostrophe_word_count_and_deterministic_limit():
    spans = grounded.candidate_spans("Stone's pale rough surfaces carry dense dark mineral specks across large uneven patches", NAMES, limit=3)
    assert len(spans) == 3 and all(5 <= len(grounded.normalize(item["text"]).split()) <= 8 for item in spans)
    assert spans == grounded.candidate_spans("Stone's pale rough surfaces carry dense dark mineral specks across large uneven patches", NAMES, limit=3)


def test_single_and_multi_pools_deduplicate_and_control_availability_matches():
    prefix, single, multiple = grounded.proposal_sets([RAW, RAW, BEAM], NAMES, 48)
    assert prefix["text"] == grounded.clean_dtd_caption(RAW)
    assert len({grounded.normalize(item["text"]) for item in multiple}) == len(multiple)
    assert all(item in multiple for item in single)
    scores = scoring([prefix["text"]] + [item["text"] for item in multiple])
    result = grounded.build_image_result({"image_id": "x/a.png", "split": "train"}, [RAW, RAW, BEAM], scores,
                                        NAMES, {"max_spans_per_hypothesis": 48}, "hash")
    rows = result["rows"]
    assert set(rows) == set(grounded.METHODS)
    assert rows["first_valid_control"]["caption_valid"] == rows["single_grounded_control"]["caption_valid"]
    assert rows["first_valid_control"]["final_caption"] == single[0]["text"]
    assert all(row["raw_caption"] == row["final_caption"] == grounded.clean_dtd_caption(row["raw_caption"]) for row in rows.values())
    assert all(row["source_long_caption"] in (RAW, BEAM) for row in rows.values())


def test_selection_uses_image_scores_not_class_scores_and_has_stable_ties():
    candidates = [{"text": "rough pale grains cover dark stone"}, {"text": "thin green strands form dense bundles"}]
    scores = {candidates[0]["text"]: {"image_cosine": .1, "class_cosine": .99},
              candidates[1]["text"]: {"image_cosine": .8, "class_cosine": .01}}
    selected, fallback = grounded.select_candidate(candidates, scores, {"text": "fallback"})
    assert selected == candidates[1] and not fallback
    scores[candidates[0]["text"]]["image_cosine"] = .8
    assert grounded.select_candidate(candidates, scores, {})[0] == candidates[0]


def test_no_eligible_spans_records_invalid_fallback_without_fabricated_caption():
    record = {"image_id": "banded/a.jpg", "split": "train"}
    scores = scoring(["banded surface"])
    result = grounded.build_image_result(record, ["banded surface"], scores, NAMES,
                                        {"max_spans_per_hypothesis": 48}, "hash")
    row = result["rows"]["multibeam_grounded"]
    assert row["final_caption"] == "banded surface" and not row["caption_valid"] and row["selection_fallback"]


def test_class_audit_score_never_changes_new_csv_validity():
    candidate = {"text": "rough pale grains cover dark stone", "long_caption": RAW,
                 "hypothesis": 0, "sentence": 0, "word_start": 0, "word_end": 6}
    row = grounded.row_for({"image_id": "x/a", "split": "test"}, candidate,
        {"image_cosine": .2, "class_cosine": .99, "class_name": "banded"}, "multibeam_grounded", "hash", NAMES)
    assert row["caption_valid"] and row["semantic_leakage_detected"] and not row["leakage_detected"]
    assert row["semantic_audit_status"] == "diagnostic_only"


def test_florence_multiple_return_sequences_are_grouped_by_image_without_label_prompts():
    seen = {}

    class Processor:
        def __call__(self, text, images, **kwargs):
            seen["prompts"] = text
            return {"input_ids": torch.ones((len(images), 2), dtype=torch.long),
                    "pixel_values": torch.ones((len(images), 3, 4, 4))}

        def batch_decode(self, output, **kwargs):
            return [str(index) for index in range(output.shape[0])]

        def post_process_generation(self, text, **kwargs):
            return {grounded.TASK: text}

    class Model:
        def generate(self, **kwargs):
            seen["kwargs"] = kwargs
            return torch.ones((6, 4), dtype=torch.long)

    instance = grounded.FrozenFlorence.__new__(grounded.FrozenFlorence)
    instance.processor, instance.model = Processor(), Model()
    instance.device, instance.dtype = torch.device("cpu"), torch.float32
    results = instance.generate([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))], 3, 3, 48)
    assert results == [["0", "1", "2"], ["3", "4", "5"]]
    assert seen["prompts"] == [grounded.TASK] * 2
    assert seen["kwargs"]["num_return_sequences"] == 3 and not seen["kwargs"]["do_sample"]


def test_ranker_aligns_each_image_with_its_own_texts():
    ranker = grounded.ImageTextRanker.__new__(grounded.ImageTextRanker)
    ranker.device, ranker.names = torch.device("cpu"), ["alpha", "beta"]
    ranker.class_vectors = torch.eye(2)
    ranker.encode_text = lambda texts: torch.tensor([[1., 0.] if text == "red" else [0., 1.] for text in texts])
    ranker.image_processor = lambda **kwargs: {"pixel_values": torch.eye(2)}
    ranker.model = SimpleNamespace(get_image_features=lambda pixel_values: pixel_values)
    scores = ranker.score([object(), object()], [["red", "green"], ["red", "green"]])
    assert scores[0]["red"]["image_cosine"] == 1 and scores[0]["green"]["image_cosine"] == 0
    assert scores[1]["green"]["image_cosine"] == 1 and scores[1]["red"]["image_cosine"] == 0


@pytest.mark.parametrize("loading", [
    {"missing_keys": ["vision_model.embeddings.patch_embedding.weight"]},
    {"missing_keys": ["text_projection.weight"]},
    {"missing_keys": [], "mismatched_keys": [("visual_projection.weight", (2, 2), (3, 3))]},
])
def test_ranker_rejects_missing_or_mismatched_learned_clip_weights(monkeypatch, loading):
    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        CLIPImageProcessor=object,
        CLIPModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: (object(), loading)),
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(ValueError, match="complete CLIP image AND text"):
        grounded.ImageTextRanker("local-test-only", NAMES, torch.device("cpu"))


@pytest.fixture
def payload(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="4.49.0"))
    records = []
    for role in ("train", "val", "test"):
        for index in range(2):
            image = tmp_path / f"{role}_{index}.png"
            Image.new("RGB", (8, 8), (index * 80, 100, 150)).save(image)
            records.append({"image_id": f"alpha/{role}_{index}.png", "image_path": str(image), "split": role})
    for name in ("florence", "clip"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"CPU test-only fake weights")
    return {"output": str(tmp_path / "output"), "records": records, "dtd_split": 1, "class_names": NAMES,
            "florence": str(tmp_path / "florence"), "clip": str(tmp_path / "clip"), "code_sha256": "test",
            "seed": 42, "max_new_images": 0, "settings": {"batch_size": 2, "num_beams": 3,
            "num_return_sequences": 3, "max_new_tokens": 48, "max_spans_per_hypothesis": 12, "clip_text_batch_size": 256}}


class Generator:
    calls = []

    def __init__(self, *args):
        pass

    def generate(self, images, beams, returns, tokens):
        self.calls.append((len(images), beams))
        return [[RAW if beams == 1 else BEAM] * returns for _ in images]


class Ranker:
    def __init__(self, *args):
        self.preprocessing_source = "test_only"
        self.image_processor = SimpleNamespace(to_dict=lambda: {"test": True})

    def score(self, images, text_groups):
        return [scoring(texts) for texts in text_groups]


def run(payload, generator=Generator, ranker=Ranker):
    return grounded.run_generation(payload, torch.device("cpu"), generator, ranker)


def test_smoke_then_resume_keeps_completed_caches_and_exports_complete_csvs(payload):
    payload["max_new_images"] = 2
    summary = run(payload)
    assert not summary["complete"] and summary["rows"] == 2
    assert "preview_incomplete" in summary["csv_paths"]["multibeam_grounded"]
    assert not (Path(payload["output"]) / "exports/captions.csv").exists()
    first_cache = grounded.cache_path(payload["output"], payload["records"][0]["image_id"])
    before = first_cache.read_bytes()
    payload["max_new_images"] = 0
    Generator.calls = []
    complete = run(payload)
    assert complete["complete"] and complete["rows"] == 6 and first_cache.read_bytes() == before
    assert sum(count for count, beams in Generator.calls if beams == 1) == 4
    for path in complete["csv_paths"].values():
        rows = grounded.read_csv_rows(path)
        assert len(rows) == 6 and list(rows[0]) == grounded.COLUMNS
        assert [row["image_id"] for row in rows] == [row["image_id"] for row in payload["records"]]
    assert not complete["human_ground_truth"] and not complete["downstream_improvement_verified"]


def test_complete_generation_does_not_load_models_and_changed_assets_are_rejected(payload):
    completed = run(payload)

    def forbidden(*args):
        raise AssertionError("Complete cache must skip model loading")

    assert run(payload, forbidden, forbidden) == completed
    (Path(payload["clip"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed assets"):
        run(payload)


def test_interruption_preserves_finished_images_and_resume_does_not_repeat_them(payload):
    class Interrupted(Generator):
        count = 0

        def generate(self, images, beams, returns, tokens):
            self.count += 1
            if self.count == 3:
                raise KeyboardInterrupt("test interruption")
            return super().generate(images, beams, returns, tokens)

    with pytest.raises(KeyboardInterrupt):
        run(payload, Interrupted)
    assert len(list((Path(payload["output"]) / "image_cache").glob("*.json"))) == 2
    Generator.calls = []
    assert run(payload)["complete"]
    assert sum(count for count, beams in Generator.calls if beams == 1) == 4


def test_wrong_score_keys_stop_without_committing_any_rows(payload):
    class Missing(Ranker):
        def score(self, images, texts):
            return [{} for _ in images]

    with pytest.raises(ValueError, match="Wrong CLIP score"):
        run(payload, ranker=Missing)
    assert not (Path(payload["output"]) / "image_cache").exists()


def test_cuda_oom_backoff_retries_smaller_groups_and_other_errors_are_not_hidden(payload):
    class OOM(Generator):
        def generate(self, images, beams, returns, tokens):
            if len(images) > 1:
                raise torch.cuda.OutOfMemoryError("test only")
            return super().generate(images, beams, returns, tokens)

    assert run(payload, OOM)["complete"]


def actual_view_runtime():
    notebook = json.loads((project_path('mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb')).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][8]["source"])
    ns = {"pd": pd, "np": np}
    for node in ast.parse(source).body:
        if ((isinstance(node, ast.FunctionDef) and node.name in {"normalized_text", "bool_value", "caption_term_pattern", "prepare_caption_views"})
                or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DTD_EXTRA_TERMS" for t in node.targets))):
            exec(ast.get_source_segment(source, node), ns)
    exec((project_path('dtd_caption_screening.py')).read_text(encoding="utf-8"), ns)
    return ns


def test_effective_policy_matches_actual_mtacs_including_legacy_semantic_recovery():
    rows = []
    for valid, semantic, score in (("True", "True", ".99"), ("False", "True", ".99"), ("False", "False", ""),
                                   ("False", "True", "nan"), ("True", "False", "")):
        rows.append({"image_id": str(len(rows)), "split": "train", "raw_caption": RAW,
            "final_caption": "The image is a close up of", "caption_valid": valid, "leakage_detected": True,
            "semantic_leakage_detected": semantic, "semantic_leakage_score": score})
    ns = actual_view_runtime()
    audit, views = ns["prepare_standalone_caption_views"](pd.DataFrame(rows), NAMES, .3, "dtd_lexical_v2")
    ours = [grounded.effective_dtd_v2(row, NAMES) for row in rows]
    assert [pair[0] for pair in ours] == views["screened"]["texts"]
    assert [pair[1] for pair in ours] == views["screened"]["valid"].tolist()


def test_comparison_reports_and_common_masks_preserve_originals_and_block_recovery(payload):
    status = run(payload)
    prefix = Path(status["csv_paths"]["prefix_control"])
    rows = grounded.read_csv_rows(prefix)
    rows[0].update(raw_caption="banded surface with pale colors and large markings", caption_valid="False",
                   semantic_leakage_detected="True", semantic_leakage_score=".99")
    old = Path(payload["output"]) / "old.csv"
    grounded.atomic_csv(old, rows, grounded.COLUMNS)
    before = old.read_bytes()
    folder = Path(payload["output"]) / "comparison"
    reports = grounded.compare_caption_files({"old": old, "new": status["csv_paths"]["multibeam_grounded"]},
        payload["records"], NAMES, folder)
    assert old.read_bytes() == before and reports
    common = grounded.read_csv_rows(folder / "common_valid_mask.csv")
    for label in ("old", "new"):
        path = folder / "matched_masks" / label / "captions.csv"
        masked = grounded.read_csv_rows(path)
        assert not grounded.effective_dtd_v2(masked[0], NAMES)[1]
        assert masked[0]["semantic_leakage_score"] == "" and masked[0]["original_semantic_score"]
        _, actual = actual_view_runtime()["prepare_standalone_caption_views"](pd.DataFrame(masked), NAMES, .3, "dtd_lexical_v2")
        assert actual["screened"]["valid"].tolist() == [grounded.bool_value(row["caption_valid"]) for row in common]


def test_comparison_rejects_wrong_split_and_duplicate_or_missing_rows(payload):
    status = run(payload)
    path = Path(status["csv_paths"]["multibeam_grounded"])
    rows = grounded.read_csv_rows(path)
    rows[0]["split"] = "test"
    grounded.atomic_csv(path, rows, grounded.COLUMNS)
    with pytest.raises(ValueError, match="wrong official DTD split"):
        grounded.compare_caption_files({"new": path}, payload["records"], NAMES, Path(payload["output"]) / "comparison")


def test_new_notebook_is_self_contained_reproducible_and_preserves_previous_experiments():
    previous = [project_path('florence2-dtd-caption-finetune.ipynb'), project_path('caption-cai-tien.ipynb')]
    previous += list(project_glob("mtacs-train-vitl16-dtd-ablation-*.ipynb"))
    before = {path: path.read_bytes() for path in previous}
    notebook_before = NOTEBOOK.read_bytes()
    runpy.run_path(str(project_path('build_florence_dtd_grounded_caption_notebook.py')))
    assert NOTEBOOK.read_bytes() == notebook_before
    assert all(path.read_bytes() == value for path, value in before.items())
    notebook = json.loads(notebook_before)
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
            assert item["outputs"] == [] and item["execution_count"] is None
            assert "from dtd_" not in "".join(item["source"])
    core = "".join(notebook["cells"][4]["source"])
    ns = {}
    exec(core, ns)
    assert ns["object_hash"]({"a": 1}) == grounded.object_hash({"a": 1})
    assert ns["candidate_spans"](RAW, NAMES)
    assert "PeftModel" not in core and '"dtd_lexical_v2"' in core


@pytest.mark.parametrize("key,value", [("batch_size", 0), ("num_beams", 1), ("num_return_sequences", 4), ("max_spans_per_hypothesis", -1)])
def test_invalid_settings_fail_early(payload, key, value):
    settings = dict(payload["settings"], **{key: value})
    with pytest.raises(ValueError):
        grounded.validate_settings(settings)
