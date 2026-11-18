"""Study06 label-aware candidate text: notebook build + helper logic on toys (no sklearn needed)."""
import ast
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import build_dtd_prefix_label_context_notebook as builder

DIM, CLIP = 8, 4


@lru_cache(maxsize=1)
def notebook():
    return builder.build_notebook()


def code(index):
    return builder.baseline.prefix.source_of(notebook()["cells"][index])


class ToyStudy01(nn.Module):
    """Has every attribute the study01 fusion uses; the visual path is a fixed linear map."""

    def __init__(self, *args, clip_dim=CLIP, **kwargs):
        super().__init__()
        self.backbone = nn.Linear(12, DIM)
        self.query_norm = nn.LayerNorm(DIM)
        self.text_projection = nn.Sequential(nn.Linear(clip_dim, DIM), nn.LayerNorm(DIM))
        self.context_text_projection = nn.Sequential(nn.Linear(clip_dim, DIM), nn.LayerNorm(DIM))
        self.text_gate = nn.Sequential(nn.Linear(2 * DIM, DIM), nn.GELU(), nn.Linear(DIM, 1))
        self.context_text_gate = nn.Sequential(nn.Linear(2 * DIM, DIM), nn.GELU(), nn.Linear(DIM, 1))
        self.mtacs_norm = nn.LayerNorm(DIM)
        self.mtacs_classifier = nn.Linear(DIM, 3)

    def _pool_backbone_features(self, images):
        return self.backbone(images.reshape(len(images), -1))

    def forward_selected_context(self, query, context, return_features=False):
        features = F.normalize(self._pool_backbone_features(query) + self._pool_backbone_features(context), dim=1)
        return None, features, None


@pytest.fixture
def ns():
    namespace = {"torch": torch, "nn": nn, "F": F, "Path": Path, "json": json, "FUSION_NORM_EPS": 1e-6,
                 "make_ablation_model_class": lambda base, visual: base,
                 "ARCHITECTURE_STUDY": {"stage": builder.STAGE}}
    exec(builder.HELPERS.read_text(encoding="utf-8"), namespace)
    torch.manual_seed(0)
    return namespace


MODE = {"name": "candidate_only", "query_fusion": False, "candidate_fusion": True, "semantic_selector": False}


def test_notebook_matches_builder_and_wires_label_context():
    nb = notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == nb
    assert len(nb["cells"]) == 27 and len({c["id"] for c in nb["cells"]}) == 27
    ast.parse("\n".join(builder.baseline.prefix.source_of(c) for c in nb["cells"] if c["cell_type"] == "code"))
    assert all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert "_PREFIX_CSV_GZ_B64" in code(9)
    assert "make_ablation_banks = make_label_context_banks(make_ablation_banks" in code(10)
    assert "make_label_context_model_class(MTACSModel" in code(18)
    assert '"schema": "dtd_prefix_label_context_v1"' in code(18)
    assert "label_context_enabled=True" in code(26)
    assert builder.PREFIX + "_vitl16" in code(4)


def test_configuration_keeps_query_text_off(monkeypatch):
    monkeypatch.delenv("DTD_PREFIX_LABEL_CONTEXT_SEED", raising=False)
    monkeypatch.delenv("DTD_SPLIT", raising=False)
    config = {"os": os}
    exec(code(2), config)
    mode, = config["SELECTED_ABLATIONS"]
    assert mode["candidate_fusion"] and not mode["query_fusion"] and not mode["semantic_selector"]
    assert config["ARCHITECTURE_STUDY"]["label_context"] and "{label}" in config["LABEL_CONTEXT_TEMPLATE"]


def test_texts_use_candidate_label_and_only_valid_captions(ns):
    view = {"image_ids": ["a.jpg", "b.jpg"], "texts": ["red stripes", "noise"], "valid": np.array([True, False])}
    texts = ns["candidate_context_texts"](["b.jpg", "a.jpg"], [1, 0], ["banded_x", "dotted"], view,
                                          "a photo of a {label} texture")
    assert texts == ["a photo of a dotted texture", "a photo of a banded x texture, red stripes"]
    with pytest.raises(ValueError, match="align"):
        ns["candidate_context_texts"](["a.jpg"], [0, 1], ["x", "y"], view, "{label}")


def test_banks_replace_only_candidate_text(ns):
    original = {"settings": MODE, "q": {"valid": torch.zeros(3, dtype=torch.bool)},
                "c": {"embeddings": torch.zeros(2, CLIP), "valid": torch.tensor([True, False])}}
    label_view = {"image_ids": ["a", "b"], "texts": ["t1", "t2"], "embeddings": torch.ones(2, CLIP),
                  "valid": torch.ones(2, dtype=torch.bool)}
    banks = ns["make_label_context_banks"](lambda *a, **k: dict(original), label_view, ["a", "b"], "cpu")(MODE, 42)
    assert banks["q"] is original["q"] and banks["settings"] is MODE
    assert banks["c"]["valid"].all() and torch.equal(banks["c"]["embeddings"], torch.ones(2, CLIP))
    assert banks["c"]["source_ids"] == ["a", "b"] and banks["c"]["texts"] == ["t1", "t2"]
    with pytest.raises(ValueError, match="order"):
        ns["make_label_context_banks"](None, label_view, ["b", "a"], "cpu")


def test_model_keeps_study01_math_and_exports_diagnostics(ns):
    network = ns["make_label_context_model_class"](ToyStudy01, ToyStudy01)().eval()
    query, context = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)
    q_text, c_text = torch.randn(2, CLIP), torch.randn(2, CLIP)
    on, off = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    out = network.forward_ablated(query, context, q_text, off, c_text, on, MODE)
    _, visual, _ = ToyStudy01.forward_selected_context(network, query, context)
    c = network.context_text_projection(c_text)
    gate = torch.sigmoid(network.context_text_gate(torch.cat([visual, c], 1))).squeeze(1)
    expected = network.mtacs_classifier(network.mtacs_norm(visual + gate[:, None] * c))
    torch.testing.assert_close(out["logits"], expected)
    assert out["query_gate"].eq(0).all()
    masked = network.forward_ablated(query, context, q_text, off, c_text, off, MODE)
    assert masked["candidate_gate"].eq(0).all()
    network.capture_fusion_diagnostics = True
    diag = network.forward_ablated(query, context, q_text, off, c_text, on, MODE)["fusion_diagnostics"]
    assert set(diag) == {"visual_norm", "candidate_raw_projection_norm", "candidate_fusion_norm",
                         "weighted_candidate_norm", "weighted_text_visual_ratio"}
