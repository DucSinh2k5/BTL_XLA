"""Embedded notebook helpers for study05; runtime globals are supplied by the DTD notebook."""
from __future__ import annotations


def validate_text_token_output_directory(path):
    path = Path(path)
    protocol_path = path / "protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (previous.get("schema") != "dtd_prefix_text_token_v1"
                or previous.get("architecture_study", {}).get("stage") != ARCHITECTURE_STUDY["stage"]):
            raise ValueError("Output belongs to another experiment; study05 must not overwrite earlier studies")
    elif path.is_dir() and any(path.glob("runs/*/seed_*/*.pt")):
        raise ValueError("Output has checkpoints but no study05 protocol; choose a new output directory")


def make_text_token_model_class(base_class, visual_class):
    original_class = make_ablation_model_class(base_class, visual_class)

    class DTDTextTokenModel(original_class):
        """Candidate caption enters the paired ViT as one extra prefix token (no RoPE) instead of a late residual."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.text_token_projection = nn.Sequential(
                nn.Linear(CLIP_EMBED_DIM, FEATURE_DIM), nn.LayerNorm(FEATURE_DIM))
            # Zero scale + zero null token: at step 0 every query sees the same constant extra token.
            self.text_token_scale = nn.Parameter(torch.tensor(float(TEXT_TOKEN_INIT_SCALE)))
            self.null_text_token = nn.Parameter(torch.zeros(FEATURE_DIM))
            self._pending_text_tokens = None
            self._pending_patch_norm = None

        def candidate_text_tokens(self, c_text, c_valid):
            projected = self.text_token_projection(c_text.float())
            tokens = projected * self.text_token_scale
            valid = c_valid.to(tokens.device, dtype=torch.bool)[:, None]
            return torch.where(valid, tokens, self.null_text_token.to(tokens.dtype).expand_as(tokens)), projected

        def _forward_token_concat_features(self, query_images, context_images):
            text = self._pending_text_tokens
            if text is None:
                return super()._forward_token_concat_features(query_images, context_images)
            query_tokens, query_rope = self._prepare_tokens_with_rope(query_images)
            context_tokens, _ = self._prepare_tokens_with_rope(context_images)
            if text.shape != (len(query_tokens), query_tokens.shape[-1]):
                raise ValueError("Text tokens must be one FEATURE_DIM vector per query/context pair (K=1)")
            prefix = self._prefix_token_count()
            # DINOv3 rotates only the trailing patch tokens, so a token placed after CLS/registers gets no RoPE.
            pair_tokens = torch.cat([query_tokens[:, :prefix], text[:, None].to(query_tokens.dtype),
                                     query_tokens[:, prefix:], context_tokens[:, prefix:]], dim=1)
            if getattr(self, "capture_fusion_diagnostics", False):
                self._pending_patch_norm = query_tokens[:, prefix:].detach().float().norm(dim=-1).mean(1)
            pair_rope = self._repeat_rope_for_query_context(query_rope)
            return self._run_blocks(pair_tokens, pair_rope)[:, 0]

        def forward_ablated(self, query_images, context_images, q_text, q_valid, c_text, c_valid, settings):
            if context_images is None or not settings["candidate_fusion"]:
                return super().forward_ablated(query_images, context_images, q_text, q_valid,
                                               c_text, c_valid, settings)
            if settings["query_fusion"]:
                raise ValueError("Study05 keeps query captions disabled")
            tokens, projected = self.candidate_text_tokens(c_text, c_valid)
            self._pending_text_tokens = tokens
            try:
                _, visual, _ = visual_class.forward_selected_context(
                    self, query_images, context_images, return_features=True)
            finally:
                self._pending_text_tokens = None
            fused = self.mtacs_norm(visual.float())
            valid = c_valid.to(fused.device, dtype=fused.dtype)
            # Reported "candidate gate" = |token scale| on active captions; not comparable to the late-fusion gate.
            candidate_gate = self.text_token_scale.detach().abs().to(fused.dtype) * valid
            result = {"logits": self.mtacs_classifier(fused), "features": F.normalize(fused, dim=1),
                      "query_gate": torch.zeros_like(valid), "candidate_gate": candidate_gate}
            if getattr(self, "capture_fusion_diagnostics", False):
                token_norm = tokens.detach().float().norm(dim=1)
                patch_norm = self._pending_patch_norm
                self._pending_patch_norm = None
                if patch_norm is None:
                    raise ValueError("Patch-token norms were not captured for this pair")
                result["fusion_diagnostics"] = {
                    "visual_norm": visual.detach().float().norm(dim=1),
                    "candidate_raw_projection_norm": projected.detach().float().norm(dim=1),
                    "candidate_fusion_norm": token_norm,
                    "weighted_candidate_norm": token_norm * valid.float(),
                    "weighted_text_visual_ratio": token_norm * valid.float() / patch_norm.clamp_min(FUSION_NORM_EPS),
                }
            return result

    return DTDTextTokenModel
