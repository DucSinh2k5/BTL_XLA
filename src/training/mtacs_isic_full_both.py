"""Text-conditioned selector and dense feature fusion for the ISIC TACS variant."""
import torch
from torch import nn
import torch.nn.functional as F


def text_for_indices(weights, embeddings, valid):
    # Straight-through weights must remain attached on the segmentation-loss path.
    weights = weights.float()
    return weights @ embeddings.float(), weights @ valid.float()


def make_isic_full_both_class(base_class):
    class MultimodalISICTACS(base_class):
        def __init__(self, *args, text_dim=512, **kwargs):
            super().__init__(*args, **kwargs)
            feature_dim = kwargs.get("feature_dim", 384)
            self.query_text_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, feature_dim), nn.GELU())
            self.candidate_text_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, feature_dim), nn.GELU())
            self.query_text_gate = nn.Linear(feature_dim * 2, 1)
            self.candidate_text_gate = nn.Linear(feature_dim * 2, 1)
            nn.init.zeros_(self.query_text_gate.weight)
            nn.init.zeros_(self.candidate_text_gate.weight)
            nn.init.constant_(self.query_text_gate.bias, -4.0)
            nn.init.constant_(self.candidate_text_gate.bias, -4.0)
            self.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))

        def select_full_both(self, query_global, candidate_global, q_text, q_valid, c_bank, c_valid,
                             forbidden_indices=None, sample=False, tau=0.1):
            scores = self.selector.score(query_global, candidate_global).float()
            semantic = F.normalize(q_text.float(), dim=1) @ F.normalize(c_bank.float(), dim=1).T
            valid = q_valid.bool()[:, None] & c_valid.bool()[None, :]
            scores = scores + torch.sigmoid(self.semantic_mix_logit) * semantic * valid
            if forbidden_indices is not None:
                forbidden_indices = forbidden_indices.to(scores.device)
                in_pool = forbidden_indices >= 0
                scores = scores.clone()
                scores[torch.arange(len(scores), device=scores.device)[in_pool], forbidden_indices[in_pool]] = -torch.inf
            if not torch.isfinite(scores).any(dim=1).all():
                raise ValueError("No candidate remains after self-exclusion")
            if sample:
                weights = F.gumbel_softmax(scores, tau=tau, hard=True, dim=1)
            else:
                weights = F.one_hot(scores.argmax(1), scores.shape[1]).to(scores.dtype)
            return {"selector_scores": scores, "selection_weights": weights,
                    "selected_indices": weights.argmax(1),
                    "selector_entropy": torch.distributions.Categorical(logits=scores).entropy().mean()}

        def fuse_text(self, tokens, text, valid, candidate=False):
            projection = self.candidate_text_projection if candidate else self.query_text_projection
            gate_layer = self.candidate_text_gate if candidate else self.query_text_gate
            features = projection(text.float())[:, None, :].expand(-1, tokens.shape[1], -1)
            gate = torch.sigmoid(gate_layer(torch.cat([tokens.float(), features], dim=-1)))
            gate = gate * valid.to(gate.dtype)[:, None, None]
            return tokens.float() + gate * features, gate.mean(dim=(1, 2))

        def decode_full_both(self, query_tokens, context_tokens, q_text, q_valid, c_text, c_valid):
            query, q_gate = self.fuse_text(query_tokens, q_text, q_valid)
            context, c_gate = self.fuse_text(context_tokens, c_text, c_valid, candidate=True)
            logits = self.decode_with_context_tokens(query, context)
            return logits, q_gate, c_gate

        def query_only_with_text(self, query_tokens, q_text, q_valid):
            query, _ = self.fuse_text(query_tokens, q_text, q_valid)
            return self.forward_query_only_from_tokens(query)

    return MultimodalISICTACS
