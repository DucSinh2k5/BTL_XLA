
from project_paths import PROJECT_ROOT, project_path, project_glob
import ast

import pandas as pd
import pytest
import torch

import build_mtacs_dtd_candidate_only_notebook as candidate
import build_mtacs_dtd_query_only_notebook as query
import dtd_caption_screening as screening
import mtacs_dtd_ablation_cells as original
from test_dtd_query_only_notebook import code_cells, runtime
from test_dtd_text_ablation import toy_runtime


def namespace():
    ns = dict(vars(original))
    exec(query.captions.embedded_helpers(), ns)
    return ns


def failed_legacy_rows():
    return pd.DataFrame([
        {"image_id": "train/0", "split": "train",
         "raw_caption": "The image is a close-up of a pale fabric with many fine dark marks across its surface.",
         "final_caption": "The image is a close up of a", "caption_valid": False,
         "leakage_detected": True, "semantic_leakage_detected": True, "semantic_leakage_score": .913577},
        {"image_id": "train/1", "split": "train",
         "raw_caption": "The image shows a close-up of a striped fabric with dark horizontal lines.",
         "final_caption": "The image shows a close up of a", "caption_valid": False,
         "leakage_detected": True, "semantic_leakage_detected": True, "semantic_leakage_score": .914},
        {"image_id": "train/2", "split": "train",
         "raw_caption": "The image is a close-up of a pale rough fabric with many small marks.",
         "final_caption": "The image is a close up of a", "caption_valid": False,
         "leakage_detected": False, "semantic_leakage_detected": False, "semantic_leakage_score": .1},
    ])


def test_cleaning_removes_only_opening_before_truncating_and_never_invents_words():
    assert screening.clean_dtd_caption("The image is a close-up of a pale rough material with tiny dark marks.") == (
        "pale rough material with tiny dark marks")
    assert screening.clean_dtd_caption("This photograph shows a close up of the striped fabric with long dark lines.") == (
        "striped fabric with long dark lines")
    assert screening.clean_dtd_caption("The image is a close up of a") == ""
    assert screening.clean_dtd_caption("Small marks on the image are pale") == "Small marks on the image are pale"
    possessive = screening.clean_dtd_caption("The image is a close-up of a bird's pale feathers with small dark marks.")
    assert len(original.normalized_text(possessive).split()) <= 8


def test_legacy_strict_reproduces_old_mask_and_lexical_v2_keeps_direct_class_screening():
    ns = namespace()
    rows = failed_legacy_rows()
    before = rows.copy(deep=True)
    strict, strict_views = ns["prepare_standalone_caption_views"](rows, ["striped", "bumpy"], .3, "legacy_strict")
    old, old_views = original.prepare_caption_views(rows, ["striped", "bumpy"], .3)
    torch.testing.assert_close(torch.as_tensor(strict_views["screened"]["valid"]),
                               torch.as_tensor(old_views["screened"]["valid"]))
    assert not strict.screened_valid.any()
    clean, clean_views = ns["prepare_standalone_caption_views"](rows, ["striped", "bumpy"], .3, "dtd_lexical_v2")
    assert clean.screened_valid.tolist() == [True, False, False]
    assert clean.exact_class_term_hit.tolist() == [False, True, False]
    assert clean.legacy_semantic_gate_recovered.tolist() == [True, False, False]
    assert not clean.semantic_hard_gate.any()
    assert not clean.semantic_score_matches_input.any()
    assert clean_views["screened"]["texts"] == clean.final_caption.tolist()
    assert clean_views["screened"]["valid"].tolist() == clean.screened_valid.tolist()
    pd.testing.assert_frame_equal(rows, before)
    assert clean.semantic_leakage_score.tolist() == rows.semantic_leakage_score.tolist()


def test_lexical_policy_does_not_tune_cutoff_or_use_per_image_labels():
    ns = namespace()
    rows = failed_legacy_rows()
    first, _ = ns["prepare_standalone_caption_views"](rows, ["striped", "bumpy"], .3, "dtd_lexical_v2")
    changed = rows.assign(semantic_leakage_score=[.6, .9, .1], class_id=[999, -2, 34])
    second, _ = ns["prepare_standalone_caption_views"](changed, ["striped", "bumpy"], .95, "dtd_lexical_v2")
    assert first.screened_valid.tolist() == second.screened_valid.tolist()
    assert first.final_caption.tolist() == second.final_caption.tolist()
    with pytest.raises(ValueError, match="Caption policy"):
        ns["prepare_standalone_caption_views"](rows, ["striped"], .3, "auto_tune_from_test")


def test_diagnostics_are_training_only_and_preserve_input_csv(tmp_path):
    ns = namespace()
    rows = failed_legacy_rows()
    heldout = rows.iloc[[0]].assign(image_id="heldout/0", split="test", raw_caption="heldout secret content")
    rows = pd.concat([rows, heldout], ignore_index=True)
    audit, _ = ns["prepare_standalone_caption_views"](rows, ["striped", "bumpy"], .3, "dtd_lexical_v2")
    ns["display"] = lambda value: None
    original_file = tmp_path / "captions.csv"
    rows.to_csv(original_file, index=False)
    before = original_file.read_bytes()
    counts = ns["caption_filter_diagnostics"](audit, .3, tmp_path)
    counts = counts.set_index("condition")["count"]
    assert counts["caption_count"] == 3 and counts["usable_model_captions"] == 1
    samples = (tmp_path / "caption_samples_train.csv").read_text()
    assert "heldout" not in samples
    assert original_file.read_bytes() == before


def test_new_input_policy_is_recorded_in_each_notebook_protocol():
    for builder in (query, candidate):
        nb = builder.build_notebook()
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        assert '"caption_input_policy": CAPTION_INPUT_POLICY' in "".join(nb["cells"][17]["source"])
        assert "caption_model_inputs.csv" in "".join(nb["cells"][9]["source"])


def test_query_stage_can_reuse_legacy_zero_valid_csv_without_loading_florence(toy_runtime, tmp_path):
    ns = runtime(toy_runtime)
    rows = []
    example = failed_legacy_rows().iloc[0].to_dict()
    for split, frame in ns["SPLIT_FRAMES"].items():
        rows.extend({**example, "image_id": image_id, "split": split} for image_id in frame.image_id)
    source = tmp_path / "zero_valid.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "config.json").write_text('{"model_type":"clip","projection_dim":512}')
    ns.update({"PROJECT_DIR": tmp_path, "DTD_SPLIT": 1, "CAPTIONS_INPUT_PATH": str(source),
               "CLIP_MODEL_LOCAL_PATH": str(clip), "CAPTION_INPUT_POLICY": "dtd_lexical_v2"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing complete CSV must not invoke Florence")

    def encode(views, *args):
        return {"embedding_dim": 512, "fingerprint": "new-texts", "views": {
            key: {**view, "embeddings": torch.zeros(len(view["texts"]), 512)} for key, view in views.items()}}

    ns["run_local_dtd_caption_generation"] = forbidden
    ns["encode_view_cache"] = encode
    exec(code_cells()[9], ns)
    assert ns["caption_audit"].screened_valid.all()
    assert all("close up" not in text.lower() for text in ns["TEXT_CACHE"]["views"]["screened"]["texts"])
    assert ns["CAPTION_CSV"].read_bytes() == source.read_bytes()
    assert (ns["SHARED_DIR"] / "caption_model_inputs.csv").is_file()
