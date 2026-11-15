"""Study05 in-ViT text token: notebook build + token logic on a toy DINOv3-shaped model (no sklearn needed)."""
import ast
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import build_dtd_prefix_text_token_notebook as builder

PREFIX_TOKENS, PATCHES, DIM = 2, 4, 8


@lru_cache(maxsize=1)
def notebook():
    return builder.build_notebook()


def code(index):
    return builder.baseline.prefix.source_of(notebook()["cells"][index])


class ToyPairVisual(nn.Module):
    """CLS+register prefix, 4 patches per image, RoPE covering only trailing patch tokens (like DINOv3)."""

    def __init__(self, *args, clip_dim=4, **kwargs):
        super().__init__()
        self.patch = nn.Linear(3, DIM)
        self.prefix = nn.Parameter(torch.randn(PREFIX_TOKENS, DIM))
        self.block = nn.TransformerEncoderLayer(DIM, 2, 16, dropout=0.0, batch_first=True)
        self.mtacs_norm = nn.LayerNorm(DIM)
        self.mtacs_classifier = nn.Linear(DIM, 3)
        self.seen = []

    def _prefix_token_count(self):
        return PREFIX_TOKENS

    def _prepare_tokens_with_rope(self, images):
        patches = self.patch(images.flatten(2).transpose(1, 2))
        return torch.cat([self.prefix.expand(len(images), -1, -1), patches], 1), torch.zeros(PATCHES, DIM)

    def _repeat_rope_for_query_context(self, rope):
        return rope.repeat_interleave(2, dim=-2)

    def _run_blocks(self, tokens, rope):
        self.seen.append({"length": tokens.shape[1], "no_rope_prefix": tokens.shape[1] - rope.shape[-2],
                          "text_slot": tokens[:, PREFIX_TOKENS].detach().clone()})
        return self.block(tokens)

    def _forward_token_concat_features(self, query_images, context_images):
        q, rope = self._prepare_tokens_with_rope(query_images)
        c, _ = self._prepare_tokens_with_rope(context_images)
        pair = torch.cat([q[:, :PREFIX_TOKENS], q[:, PREFIX_TOKENS:], c[:, PREFIX_TOKENS:]], 1)
        return self._run_blocks(pair, self._repeat_rope_for_query_context(rope))[:, 0]

    def forward_selected_context(self, query_images, context_images, return_features=False):
        features = F.normalize(self._forward_token_concat_features(query_images, context_images), dim=1)
        return self.mtacs_classifier(features), features, None


def make_ablation_model_class(base_class, visual_class):
    class Ablation(base_class):
        def forward_ablated(self, *args, **kwargs):
            return {"fallback": True}
    return Ablation


@pytest.fixture
def ns():
    namespace = {"torch": torch, "nn": nn, "F": F, "Path": Path, "json": json, "CLIP_EMBED_DIM": 4,
                 "FEATURE_DIM": DIM, "TEXT_TOKEN_INIT_SCALE": 0.0, "FUSION_NORM_EPS": 1e-6,
                 "make_ablation_model_class": make_ablation_model_class,
                 "ARCHITECTURE_STUDY": {"stage": builder.STAGE}}
    exec(builder.HELPERS.read_text(encoding="utf-8"), namespace)
    torch.manual_seed(0)
    return namespace


def model(ns):
    return ns["make_text_token_model_class"](ToyPairVisual, ToyPairVisual)().eval()


MODE = {"name": "candidate_only", "query_fusion": False, "candidate_fusion": True, "semantic_selector": False}


def inputs(batch=2):
    return (torch.randn(batch, 3, 2, 2), torch.randn(batch, 3, 2, 2), torch.randn(batch, 4),
            torch.zeros(batch, dtype=torch.bool), torch.randn(batch, 4), torch.ones(batch, dtype=torch.bool))


def test_notebook_matches_builder_parses_and_declares_token_fusion():
    nb = notebook()
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == nb
    assert len(nb["cells"]) == 27 and len({c["id"] for c in nb["cells"]}) == 27
    combined = [builder.baseline.prefix.source_of(c) for c in nb["cells"] if c["cell_type"] == "code"]
    ast.parse("\n".join(combined))
    assert all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code")
    assert "make_text_token_model_class(MTACSModel" in code(18)
    assert '"schema": "dtd_prefix_text_token_v1"' in code(18)
    assert builder.PREFIX + "_vitl16" in code(4) and "CONTEXT_TOP_K = 1" in code(4)
    assert "token_fusion_enabled=True" in code(26)
    assert "_PREFIX_CSV_GZ_B64" in code(9) and "stage_method_caption_csv(" in code(10)
    assert nb["metadata"]["dtd_ablation"]["architecture_study"] == builder.STAGE


def test_embedded_caption_cell_writes_exact_prefix_csv(tmp_path, monkeypatch):
    config = {"os": os}
    exec(code(2), config)
    expected = config["CAPTION_METHOD_EXPECTED"]
    monkeypatch.chdir(tmp_path)
    ns = {"CAPTIONS_INPUT_PATH": "/kaggle/input/dtd-images-folder", "CAPTION_METHOD_EXPECTED": expected}
    exec(code(9), ns)
    written = Path(ns["CAPTIONS_INPUT_PATH"])
    assert written == tmp_path / expected["filename"]
    assert hashlib.sha256(written.read_bytes()).hexdigest() == expected["sha256"]
    attached = tmp_path / "attached.csv"
    attached.write_bytes(b"kept as given")
    ns = {"CAPTIONS_INPUT_PATH": str(attached), "CAPTION_METHOD_EXPECTED": expected}
    exec(code(9), ns)
    assert ns["CAPTIONS_INPUT_PATH"] == str(attached)


def test_configuration_enables_only_the_token(monkeypatch):
    monkeypatch.delenv("DTD_PREFIX_TEXT_TOKEN_SEED", raising=False)
    monkeypatch.delenv("DTD_SPLIT", raising=False)
    config = {"os": os}
    exec(code(2), config)
    assert config["TRAIN_SEEDS"] == [42] and config["TEXT_TOKEN_INIT_SCALE"] == 0.0
    mode, = config["SELECTED_ABLATIONS"]
    assert mode["candidate_fusion"] and not mode["query_fusion"] and not mode["semantic_selector"]
    study = config["ARCHITECTURE_STUDY"]
    assert study["token_fusion"] and not any(study[k] for k in ("alignment", "partial_ot", "semantic_selector", "aswp"))


def test_output_guard_rejects_other_studies(ns, tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps(
        {"schema": "dtd_prefix_norm_fusion_v1", "architecture_study": {"stage": "02_prefix_candidate_only_norm_fusion"}}))
    with pytest.raises(ValueError, match="another experiment"):
        ns["validate_text_token_output_directory"](tmp_path)
    (tmp_path / "protocol.json").write_text(json.dumps(
        {"schema": "dtd_prefix_text_token_v1", "architecture_study": {"stage": builder.STAGE}}))
    ns["validate_text_token_output_directory"](tmp_path)


def test_token_sits_after_prefix_without_rope_and_starts_neutral(ns):
    network = model(ns)
    query, context, q_text, q_valid, c_text, c_valid = inputs()
    aligned = network.forward_ablated(query, context, q_text, q_valid, c_text, c_valid, MODE)
    seen = network.seen[-1]
    assert seen["length"] == PREFIX_TOKENS + 1 + 2 * PATCHES
    assert seen["no_rope_prefix"] == PREFIX_TOKENS + 1
    off = network.forward_ablated(query, context, q_text, q_valid, c_text * 20, ~c_valid, MODE)
    torch.testing.assert_close(aligned["logits"], off["logits"])  # zero scale == zero null token at init
    assert aligned["candidate_gate"].eq(0).all() and aligned["query_gate"].eq(0).all()


def test_caption_changes_output_once_scale_moves_and_scale_gets_gradient(ns):
    network = model(ns)
    query, context, q_text, q_valid, c_text, c_valid = inputs()
    network.forward_ablated(query, context, q_text, q_valid, c_text, c_valid, MODE)["logits"].sum().backward()
    assert network.text_token_scale.grad is not None and network.text_token_scale.grad.abs() > 0
    with torch.no_grad():
        network.text_token_scale.fill_(1.0)
    aligned = network.forward_ablated(query, context, q_text, q_valid, c_text, c_valid, MODE)
    off = network.forward_ablated(query, context, q_text, q_valid, c_text, ~c_valid, MODE)
    torch.testing.assert_close(network.seen[-1]["text_slot"], torch.zeros(2, DIM))  # null token for text_off
    assert not torch.allclose(aligned["logits"], off["logits"])
    torch.testing.assert_close(aligned["candidate_gate"], torch.ones(2))
    network.capture_fusion_diagnostics = True
    diagnostics = network.forward_ablated(query, context, q_text, q_valid, c_text, c_valid, MODE)["fusion_diagnostics"]
    assert set(diagnostics) == {"visual_norm", "candidate_raw_projection_norm", "candidate_fusion_norm",
                                "weighted_candidate_norm", "weighted_text_visual_ratio"}
    assert diagnostics["weighted_text_visual_ratio"].gt(0).all()


def test_no_context_and_query_text_paths(ns):
    network = model(ns)
    query, context, q_text, q_valid, c_text, c_valid = inputs()
    assert network.forward_ablated(query, None, q_text, q_valid, c_text, c_valid, MODE)["fallback"]
    with pytest.raises(ValueError, match="query captions"):
        network.forward_ablated(query, context, q_text, q_valid, c_text, c_valid, {**MODE, "query_fusion": True})
    assert not any(name.startswith(("backbone.", "selector.")) for name, _ in network.named_parameters()
                   if name.startswith(("text_token", "null_text")))
