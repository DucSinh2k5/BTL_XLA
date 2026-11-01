import ast
import hashlib
import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import build_dtd_prefix_candidate_baseline_notebook as builder
from test_dtd_candidate_only_notebook import runtime as candidate_runtime
from test_dtd_text_ablation import toy_runtime


@lru_cache(maxsize=1)
def code_cells():
    return {i: builder.prefix.source_of(cell) for i, cell in enumerate(builder.build_notebook()["cells"])}


def baseline_runtime(toy_runtime):
    ns = candidate_runtime(toy_runtime)
    exec(code_cells()[2], ns)
    ns.update(CLIP_EMBED_DIM=4, PROTOCOL_FINGERPRINT="prefix-baseline-cpu-test",
              CAPTION_BINDING={"method": "prefix_control", "sha256": builder.expected_prefix()["sha256"]})
    return ns


def test_generated_notebook_matches_builder_preserves_sources_and_keeps_learning_recipe():
    sources = [builder.SOURCE, builder.prefix.TARGET, builder.MANIFEST,
               builder.MANIFEST.parent / builder.expected_prefix()["filename"]]
    before = {path: path.read_bytes() for path in sources}
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == notebook
    assert all(path.read_bytes() == value for path, value in before.items())
    original = builder.candidate.build_notebook()
    assert len(notebook["cells"]) == 24
    assert len({cell["id"] for cell in notebook["cells"]}) == 24
    for i in (11, 12, 14, 15, 19, 21):
        assert notebook["cells"][i]["source"] == original["cells"][i]["source"]
    all_code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = builder.prefix.source_of(cell)
            ast.parse(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert "import build_" not in source and "from project_paths" not in source
            all_code.append(source)
    combined = "\n".join(all_code)
    ast.parse(combined)
    assert "load_florence" not in combined and "ensure_dtd_caption_csv" not in combined
    assert "subprocess" not in combined and "pip install" not in combined
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' in code_cells()[1]
    assert 'DTD_PREFIX_CANDIDATE_BASELINE_EPOCHS", "50"' in code_cells()[4]
    assert "architecture_01_prefix_candidate_only_vitl16" in code_cells()[4]
    assert "IMAGE_SIZE = 384" in code_cells()[4] and "CONTEXT_TOP_K = 1" in code_cells()[4]
    normalized = code_cells()[4].replace("DTD_PREFIX_CANDIDATE_BASELINE_OUTPUT", "DTD_CANDIDATE_ONLY_OUTPUT").replace(
        "architecture_01_prefix_candidate_only_vitl16", "ablation_03_candidate_only_vitl16").replace(
        "DTD_PREFIX_CANDIDATE_BASELINE_EPOCHS", "DTD_CANDIDATE_ONLY_EPOCHS").replace(
        "/datasets/nguyenducxu/dinov3-code/", "/datasets/quctriunguyn/dinov3-code/").replace(
        "/datasets/nguyenducxu/dinov3-vitl16/", "/datasets/quctriunguyn/dinov3-vitl16/")
    assert normalized == builder.prefix.source_of(original["cells"][4])
    metadata = notebook["metadata"]["dtd_ablation"]
    assert metadata["experiment"] == "candidate_only" and not metadata["caption_generation"]
    assert metadata["candidate_fusion"] and not metadata["query_fusion"] and not metadata["semantic_selector"]
    assert metadata["caption_csv_sha256"] == builder.expected_prefix()["sha256"]


def test_configuration_isolated_from_previous_experiments(monkeypatch):
    for name in ("DTD_PREFIX_CANDIDATE_BASELINE_SEED", "DTD_PREFIX_CAPTIONS_INPUT_PATH", "DTD_SPLIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DTD_CANDIDATE_ONLY_SEED", "123")
    monkeypatch.setenv("DTD_CAPTIONS_INPUT_PATH", "/wrong/other_caption.csv")
    monkeypatch.setenv("DTD_CAPTION_INPUT_POLICY", "legacy_strict")
    monkeypatch.setenv("DTD_ABLATIONS", "full_both")
    ns = {"os": os}
    exec(code_cells()[2], ns)
    assert ns["TRAIN_SEEDS"] == [42] and ns["CAPTIONS_INPUT_PATH"] == ""
    assert not ns["GENERATE_CAPTIONS_IF_MISSING"] and not ns["EARLY_STOPPING"]
    assert ns["CAPTION_INPUT_POLICY"] == "dtd_lexical_v2"
    mode, = ns["SELECTED_ABLATIONS"]
    assert mode["name"] == "candidate_only" and mode["candidate_fusion"]
    assert not mode["query_fusion"] and not mode["semantic_selector"]
    assert not any(ns["ARCHITECTURE_STUDY"][key] for key in
                   ("alignment", "partial_ot", "semantic_selector", "token_fusion", "aswp"))
    monkeypatch.setenv("DTD_SPLIT", "2")
    with pytest.raises(ValueError, match="split1"):
        exec(code_cells()[2], {"os": os})


def test_real_prefix_csv_encodes_only_training_pool_and_reuses_cache(tmp_path, monkeypatch):
    ns = {"os": os}
    exec(code_cells()[2], ns)
    exec(code_cells()[6], ns)
    exec(code_cells()[8], ns)
    expected = builder.expected_prefix()
    source = builder.MANIFEST.parent / expected["filename"]
    rows = pd.read_csv(source).fillna("")
    frames = {split: rows.loc[rows.split.eq(split), ["image_id"]].copy()
              for split in ("train", "val", "test")}
    positions = np.sort(np.random.default_rng(42).choice(1880, size=376, replace=False))
    candidates = frames["train"].iloc[positions].reset_index(drop=True)
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "config.json").write_text(json.dumps({"model_type": "clip", "projection_dim": 512}))
    (clip / "model.safetensors").write_bytes(b"test-only placeholder")
    calls = {"loads": 0, "texts": []}

    class Tokens(dict):
        def to(self, device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            return cls()

        def __call__(self, texts, **kwargs):
            calls["texts"].extend(texts)
            return Tokens(input_ids=torch.ones((len(texts), 1), dtype=torch.long))

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=77, projection_dim=512)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            calls["loads"] += 1
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids):
            return SimpleNamespace(text_embeds=torch.ones(len(input_ids), 512))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPTokenizer=Tokenizer, CLIPTextModelWithProjection=Encoder))
    ns.update(OUT_PUT_DIR=tmp_path, PROJECT_DIR=tmp_path, train_df=frames["train"],
              val_df=frames["val"], test_df=frames["test"], candidate_df=candidates,
              CAPTIONS_INPUT_PATH=str(source), CLIP_MODEL_LOCAL_PATH=str(clip),
              DEVICE=torch.device("cpu"), class_names=sorted({x.split("/")[0] for x in rows.image_id}),
              display=lambda *args: None)
    exec(code_cells()[9], ns)
    audit = ns["caption_audit"]
    assert len(audit) == 376 and set(audit.split) == {"train"}
    assert set(audit.image_id) == set(candidates.image_id)
    eligible = audit.loc[audit.screened_valid, "final_caption"]
    assert 0 < len(eligible) < 376
    assert set(calls["texts"]) == set(eligible) and len(calls["texts"]) == len(set(eligible))
    cached = tmp_path / "shared/captions.csv"
    assert cached.read_bytes() == source.read_bytes()
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == expected["sha256"]
    view = ns["TEXT_CACHE"]["views"]["screened"]
    assert view["embeddings"].shape == (376, 512)
    assert view["embeddings"][~view["valid"]].count_nonzero() == 0
    assert view["embeddings"][view["valid"]].gt(0).all()
    banks = ns["make_ablation_banks"](ns["SELECTED_ABLATIONS"][0], 42)
    assert banks["c"]["source_ids"] == candidates.image_id.tolist()
    assert banks["q"]["embeddings"].shape == (5640, 512)
    assert banks["q"]["embeddings"].count_nonzero() == 0 and not banks["q"]["valid"].any()
    q_text, q_valid = ns["batch_caption_inputs"](banks, frames["test"].image_id[:2])
    assert q_text.count_nonzero() == 0 and not q_valid.any()
    exec(code_cells()[9], ns)
    assert calls["loads"] == 1
    renamed = tmp_path / "uploaded_captions.csv"
    shutil.copy2(source, renamed)
    assert ns["resolve_method_caption_csv"](str(renamed), expected) == renamed.resolve()
    other = builder.MANIFEST.parent / "captions_first_valid_control.csv"
    with pytest.raises(ValueError, match="Wrong or modified"):
        ns["verify_method_caption_csv"](other, frames, expected)
    assert (tmp_path / "caption_method.json").is_file()
    assert (tmp_path / "architecture_study.json").is_file()


def test_baseline_actual_train_resume_final_test_and_reports(toy_runtime, monkeypatch):
    ns = baseline_runtime(toy_runtime)
    exec(code_cells()[19], ns)
    run_dir = ns["RUNS_DIR"] / "candidate_only" / "seed_42"
    assert len(pd.read_csv(run_dir / "history.csv")) == 2
    stamp = (run_dir / "best.pt").stat().st_mtime_ns
    exec(code_cells()[19], ns)
    assert (run_dir / "best.pt").stat().st_mtime_ns == stamp
    exec(code_cells()[21], ns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update(plt=plt, audit_summary=pd.DataFrame({"valid_rate": [1.0]}, index=["train"]))
    exec(code_cells()[23], ns)
    result = pd.read_csv(ns["OUT_PUT_DIR"] / "architecture_01_prefix_candidate_only_results.csv")
    assert set(result.split) == {"val", "test"}
    assert result.mean_query_gate.eq(0).all() and result.mean_candidate_gate.gt(0).all()
    assert result.caption_method.eq("prefix_control").all() and result.candidate_text_enabled.all()
    assert not result[["query_text_enabled", "semantic_selector_enabled", "alignment_enabled",
                       "token_fusion_enabled", "aswp_enabled"]].any().any()
    report = (ns["OUT_PUT_DIR"] / "architecture_01_prefix_candidate_only_report.md").read_text(encoding="utf-8")
    assert "## Candidate-pool caption coverage" in report and "## Caption provenance" in report
    assert "not human-verified ground truth" in report
    assert (ns["OUT_PUT_DIR"] / "architecture_01_prefix_candidate_only_training.png").is_file()
    plt.close("all")
