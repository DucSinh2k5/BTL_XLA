
from project_paths import PROJECT_ROOT, project_path, project_glob
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

import build_mtacs_dtd_ablation_notebook as builder


@pytest.fixture
def caption_runtime(tmp_path):
    source = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    definitions, generation = builder.caption_components(source["cells"])
    namespace = {"CAPTION_TASK_PROMPT": "<MORE_DETAILED_CAPTION>"}
    exec(definitions, namespace)
    frames = {}
    for split in ("train", "val", "test"):
        rows = []
        for i, label in enumerate(("striped", "bumpy")):
            image_path = tmp_path / f"{split}_{i}.png"
            Image.new("RGB", (12, 10), color=(40 + i, 80, 120)).save(image_path)
            rows.append({"image_id": f"{label}/{split}_{i}.png", "image_path": str(image_path)})
        frames[split] = pd.DataFrame(rows)
    calls = {"model_loads": 0, "generated": 0}

    def fake_florence(*args, **kwargs):
        calls["model_loads"] += 1
        return object(), object(), torch.float32

    def fake_generate(model, processor, images, *args):
        assert all(isinstance(image, Image.Image) for image in images)
        calls["generated"] += len(images)
        return ["A rough surface with pale and dark marks" for _ in images]

    namespace.update({
        "OUT_PUT_DIR": tmp_path / "ablation", "PROJECT_DIR": tmp_path, "DTD_SPLIT": 1,
        "CAPTIONS_INPUT_PATH": "", "GENERATE_CAPTIONS_IF_MISSING": True, "shutil": shutil,
        "train_df": frames["train"], "val_df": frames["val"], "test_df": frames["test"],
        "DEVICE": SimpleNamespace(type="cuda"), "CLIP_MODEL_LOCAL_PATH": str(tmp_path / "clip"),
        "VLM_MODEL": "microsoft/Florence-2-base", "FLORENCE_MODEL_REF": "microsoft/Florence-2-base",
        "PROMPT_VERSION": "florence2_detailed_v2", "CAPTION_MAX_NEW_TOKENS": 48,
        "CAPTION_WORKERS": 0, "CAPTION_BATCH_SIZE": 2, "CAPTION_SEMANTIC_THRESHOLD": .30,
        "ENABLE_LEAKAGE_CHECK": True, "class_names": ["striped", "bumpy"],
        "resolve_local_model": lambda *args, **kwargs: tmp_path / "clip",
        "load_florence": fake_florence,
        "load_semantic_auditor": lambda *args, **kwargs: (object(), object(), torch.zeros(2, 4), ["striped", "bumpy"]),
        "generate_batch": fake_generate,
        "semantic_audit_captions": lambda captions, *args, **kwargs: [(.1, "striped", False) for _ in captions],
    })
    return namespace, generation, calls


def caption_stage(namespace, generation):
    exec(builder.CAPTION_SETUP, namespace)
    exec(builder.CAPTION_STATUS, namespace)
    if namespace["NEEDS_CAPTION_GENERATION"]:
        exec(generation, namespace)


def test_fresh_run_creates_audited_csv_when_no_input_exists(caption_runtime):
    namespace, generation, calls = caption_runtime
    caption_stage(namespace, generation)
    rows = namespace["load_existing"](namespace["CAPTION_CSV"])
    assert len(rows) == 6
    assert calls == {"model_loads": 1, "generated": 6}
    assert all(row["vlm_model"] == "microsoft/Florence-2-base" for row in rows.values())
    assert all(row["caption_valid"] == "True" and row["semantic_leakage_score"] == "0.1" for row in rows.values())
    assert namespace["CAPTION_CSV"].parent == namespace["OUT_PUT_DIR"] / "shared"


def test_complete_csv_skips_captioner_even_when_model_reference_changes(caption_runtime):
    namespace, generation, calls = caption_runtime
    caption_stage(namespace, generation)
    before = namespace["CAPTION_CSV"].read_bytes()
    namespace["VLM_MODEL"] = "another/default/path"
    caption_stage(namespace, generation)
    assert namespace["NEEDS_CAPTION_GENERATION"] is False
    assert calls == {"model_loads": 1, "generated": 6}
    assert namespace["CAPTION_CSV"].read_bytes() == before


def test_interrupted_generation_resumes_only_missing_images(caption_runtime):
    namespace, generation, calls = caption_runtime
    original_generate = namespace["generate_batch"]
    invocations = []

    def interrupted(*args, **kwargs):
        invocations.append(True)
        if len(invocations) == 2:
            raise RuntimeError("synthetic interruption")
        return original_generate(*args, **kwargs)

    namespace["generate_batch"] = interrupted
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        caption_stage(namespace, generation)
    completed = namespace["load_existing"](namespace["CAPTION_CSV"])
    assert len(completed) == 2
    namespace["generate_batch"] = original_generate
    caption_stage(namespace, generation)
    resumed = namespace["load_existing"](namespace["CAPTION_CSV"])
    assert len(resumed) == 6
    assert calls["generated"] == 6
    assert all(resumed[key] == row for key, row in completed.items())


def test_incomplete_cache_requires_explicit_reuse_only_override(caption_runtime):
    namespace, generation, _ = caption_runtime
    caption_stage(namespace, generation)
    path = namespace["CAPTION_CSV"]
    frame = pd.read_csv(path)
    frame.iloc[:1].to_csv(path, index=False)
    namespace["GENERATE_CAPTIONS_IF_MISSING"] = False
    with pytest.raises(FileNotFoundError, match="incomplete: 1/6"):
        caption_stage(namespace, generation)


def test_generation_reports_legacy_checkpoint_version_mismatch_before_loading(tmp_path):
    source = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    definitions, _ = builder.caption_components(source["cells"])
    namespace = {"CAPTION_TASK_PROMPT": "<MORE_DETAILED_CAPTION>"}
    exec(definitions, namespace)
    namespace.update({"FLORENCE_BACKEND": "native", "_transformers_version": "5.0.0",
        "resolve_local_model": lambda reference, expected, **kwargs:
            tmp_path / "Florence-2-base" if expected == "florence2_legacy" else None})
    with pytest.raises(RuntimeError, match="Use Transformers 4.49.x"):
        namespace["load_florence"]("microsoft/Florence-2-base", torch.device("cpu"), True)
