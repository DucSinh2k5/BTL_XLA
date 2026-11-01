"""Create a separate M-TACS notebook that also fuses selected context captions."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import json
from pathlib import Path


ROOT = PROJECT_ROOT
SOURCE = project_path('mtacs-train-captions-vitl16-multimodal-top3.ipynb')
TARGET = project_path('mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb')


def source_of(cell):
    return "".join(cell.get("source", []))


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def replace_once(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {name}, found {count}")
    return text.replace(old, new, 1)


def replace_block(text, start, end, replacement, name):
    first = text.find(start)
    last = text.find(end, first + len(start)) if first >= 0 else -1
    if first < 0 or last < 0:
        raise RuntimeError(f"Could not locate {name}")
    return text[:first] + replacement + text[last:]


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

# Keep this experiment isolated from the existing top-3 visual/semantic run.
config = source_of(cells[3])
config = replace_once(config, 'EXPERIMENT_MODE = "mtacs_dual_top3"',
                      'EXPERIMENT_MODE = "mtacs_dual_top3_context_text"', "experiment mode")
config = config.replace("mtacs_dual_top3\n", "mtacs_dual_top3 | mtacs_dual_top3_context_text\n")
set_source(cells[3], config)

settings = source_of(cells[5])
settings = replace_once(settings,
    "mtacs_sun397_visual_text_context_vitl16_dual_top3",
    "mtacs_sun397_visual_text_context_vitl16_dual_top3_context_text", "output directory")
set_source(cells[5], settings)

intro = source_of(cells[0])
intro = intro.replace(
    "# M-TACS: visual + semantic top-3 context",
    "# M-TACS: visual + query/context semantic top-3 context",
)
intro = intro.replace(
    "The visual TACS selector remains image-only; text is consumed only by the downstream branch.",
    "This variant scores candidates using visual and CLIP-caption similarity; selected context images and valid context captions are both used downstream.",
)
intro = intro.replace(
    "Only visual context changes between branches.",
    "In this variant, selected visual contexts and their captions change together while the query caption stays fixed.",
)
intro = intro.replace(
    "the selector and TACS policy are not given captions, CLIP embeddings, or labels.",
    "the selector receives query/candidate caption embeddings for semantic scoring; task labels supervise only the query prediction.",
)
intro += (
    "\n\nThis variant also sends the selected candidate captions into the downstream fusion block. "
    "The query caption remains active. Candidate captions participate both in selector scoring and "
    "in the context-text fusion; invalid captions are masked while their candidate images remain usable."
)
set_source(cells[0], intro)

model = source_of(cells[13])
model = replace_block(
    model,
    "    def select_full_pool(\n",
    "    def forward_selected_context(self, query_images, selected_context_images, return_features=False):\n",
    '''    def select_full_pool(
        self, query_images, candidate_feature_bank, query_text, query_text_valid,
        candidate_text_bank, candidate_text_valid, query_candidate_positions=None,
        use_gumbel=True,
    ):
        visual_scores, query_features = self.score_full_candidate_pool(query_images, candidate_feature_bank)
        visual_scores = visual_scores.float()
        candidate_excluded = torch.zeros_like(visual_scores, dtype=torch.bool)
        if query_candidate_positions is not None:
            positions = query_candidate_positions.to(visual_scores.device, dtype=torch.long).reshape(-1)
            in_pool = (positions >= 0) & (positions < visual_scores.shape[1])
            candidate_excluded = F.one_hot(
                positions.clamp(0, visual_scores.shape[1] - 1), visual_scores.shape[1]
            ).bool() & in_pool.unsqueeze(1)
        visual_valid = ~candidate_excluded
        visual_count = visual_valid.sum(dim=1, keepdim=True).clamp_min(1)
        visual_mean = (visual_scores * visual_valid).sum(dim=1, keepdim=True) / visual_count
        visual_var = (((visual_scores - visual_mean) ** 2) * visual_valid).sum(dim=1, keepdim=True) / visual_count
        visual_z = ((visual_scores - visual_mean) / visual_var.sqrt().clamp_min(1e-6)) * visual_valid

        query_text = F.normalize(query_text.float(), dim=1)
        candidate_text_bank = F.normalize(candidate_text_bank.float().to(query_text.device), dim=1)
        semantic_raw = query_text @ candidate_text_bank.T
        semantic_valid = (
            query_text_valid.bool().unsqueeze(1)
            & candidate_text_valid.bool().to(query_text.device).unsqueeze(0)
        ) & ~candidate_excluded
        semantic_count = semantic_valid.sum(dim=1, keepdim=True).clamp_min(1)
        semantic_mean = (semantic_raw * semantic_valid).sum(dim=1, keepdim=True) / semantic_count
        semantic_var = (((semantic_raw - semantic_mean) ** 2) * semantic_valid).sum(dim=1, keepdim=True) / semantic_count
        semantic_z = ((semantic_raw - semantic_mean) / semantic_var.sqrt().clamp_min(1e-6)) * semantic_valid

        semantic_mix = torch.sigmoid(self.selector.semantic_mix_logit)
        selector_scores = visual_z + semantic_mix * semantic_z
        selector_scores = selector_scores.masked_fill(candidate_excluded, -torch.inf)

        k = min(CONTEXT_TOP_K, selector_scores.shape[1])
        if k < 1:
            raise RuntimeError("Candidate pool is empty")
        selected_steps, log_prob_steps, entropy_steps = [], [], []
        if self.training and use_gumbel:
            remaining = selector_scores
            for _ in range(k):
                dist = torch.distributions.Categorical(logits=remaining / max(TACS_GUMBEL_TAU, 1e-6))
                action = dist.sample()
                selected_steps.append(action)
                log_prob_steps.append(dist.log_prob(action))
                entropy_steps.append(dist.entropy())
                remaining = remaining.scatter(1, action[:, None], -torch.inf)
            selection_log_probs = torch.stack(log_prob_steps, dim=1)
            entropy = torch.stack(entropy_steps, dim=1).sum(dim=1).mean()
        else:
            selected = selector_scores.topk(k, dim=1).indices
            selected_steps = [selected[:, index] for index in range(k)]
            selection_log_probs = selector_scores.new_zeros((selector_scores.shape[0], k))
            entropy = torch.distributions.Categorical(logits=selector_scores).entropy().mean()

        selected_indices = torch.stack(selected_steps, dim=1)
        selected_scores = selector_scores.gather(1, selected_indices)
        context_weights = torch.softmax(selected_scores, dim=1)
        details = {
            "selector_scores": selector_scores,
            "selected_indices": selected_indices,
            "selected_scores": selected_scores,
            "context_weights": context_weights,
            "selection_log_probs": selection_log_probs,
            "selector_entropy": entropy,
            "semantic_mix": semantic_mix.detach(),
            "query_features": query_features,
        }
        self.active_fusion_mode = "dual_visual_semantic_top3_with_context_text"
        return details

''',
    "top-3 multimodal selector",
)

# Candidate captions are aligned by image_id, so gather the exact selected captions.
model = replace_once(
    model,
    "        self.text_gate = nn.Sequential(\n            nn.Linear(FEATURE_DIM * 2, FEATURE_DIM),\n            nn.GELU(),\n            nn.Linear(FEATURE_DIM, 1),\n        )\n",
    "        self.text_gate = nn.Sequential(\n            nn.Linear(FEATURE_DIM * 2, FEATURE_DIM),\n            nn.GELU(),\n            nn.Linear(FEATURE_DIM, 1),\n        )\n"
    "        self.context_text_gate = nn.Sequential(\n            nn.Linear(FEATURE_DIM * 2, FEATURE_DIM),\n            nn.GELU(),\n            nn.Linear(FEATURE_DIM, 1),\n        )\n",
    "context caption gate",
)
model = replace_block(
    model,
    "    def _fuse_text(self, visual_features, text_embeddings, text_valid_mask):\n",
    "    def forward_query_with_text(\n",
    '''    def _fuse_text(
        self, visual_features, text_embeddings, text_valid_mask,
        context_text_embeddings=None, context_text_valid_mask=None, context_weights=None,
    ):
        query_text_features = self.text_projection(text_embeddings.float())
        query_gate = torch.sigmoid(self.text_gate(torch.cat(
            [visual_features.float(), query_text_features], dim=1
        )))
        query_valid = text_valid_mask.to(query_gate.device, dtype=query_gate.dtype).view(-1, 1)
        query_gate = query_gate * query_valid
        fused = visual_features.float() + query_gate * query_text_features

        context_gate = torch.zeros_like(query_gate)
        context_valid = torch.zeros_like(query_valid)
        if context_text_embeddings is not None and context_text_valid_mask is not None:
            batch_size, context_count, text_dim = context_text_embeddings.shape
            projected = self.text_projection(
                context_text_embeddings.float().reshape(batch_size * context_count, text_dim)
            ).reshape(batch_size, context_count, -1)
            context_valid_items = context_text_valid_mask.to(
                projected.device, dtype=torch.bool
            )
            weights = (
                torch.ones_like(context_valid_items, dtype=projected.dtype)
                if context_weights is None
                else context_weights.to(projected.device, dtype=projected.dtype)
            )
            weights = weights * context_valid_items.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            context_features = (projected * weights.unsqueeze(-1)).sum(dim=1)
            context_valid = context_valid_items.any(dim=1, keepdim=True).to(projected.dtype)
            context_gate = torch.sigmoid(self.context_text_gate(torch.cat(
                [visual_features.float(), context_features], dim=1
            ))) * context_valid
            fused = fused + context_gate * context_features

        fused = self.mtacs_norm(fused)
        logits = self.mtacs_classifier(fused)
        reported_gate = (query_gate + context_gate) / (query_valid + context_valid).clamp_min(1.0)
        return logits, fused, reported_gate.squeeze(1)

    def forward_selected_context_with_text(
        self, query_images, selected_context_images, text_embeddings, text_valid_mask,
        context_text_embeddings=None, context_text_valid_mask=None, context_weights=None,
        return_features=False,
    ):
        _, visual_features, attention = super().forward_selected_context(
            query_images, selected_context_images, return_features=True
        )
        logits, fused_features, gate = self._fuse_text(
            visual_features, text_embeddings, text_valid_mask,
            context_text_embeddings, context_text_valid_mask, context_weights,
        )
        if return_features:
            return logits, F.normalize(fused_features.float(), dim=1), attention, gate
        return logits, gate

''',
    "query plus selected context text fusion",
)
set_source(cells[13], model)

bank_cell = source_of(cells[13])
bank_cell = replace_once(
    bank_cell,
    "candidate_text_valid = torch.tensor([entry[1] for entry in candidate_text_entries], dtype=torch.bool, device=DEVICE)\n",
    "candidate_text_valid = torch.tensor([entry[1] for entry in candidate_text_entries], dtype=torch.bool, device=DEVICE)\n"
    "candidate_labels_device = candidate_labels_tensor.to(DEVICE, non_blocking=True)\n"
    "candidate_image_ids = candidate_df['image_id'].astype(str).tolist()\n",
    "candidate image IDs",
)
set_source(cells[13], bank_cell)

helpers = source_of(cells[14])
helpers = replace_once(
    helpers,
    "def gather_bank_labels(indices):\n",
    """candidate_position_by_id = {image_id: position for position, image_id in enumerate(candidate_image_ids)}


def query_candidate_positions(image_ids):
    return torch.tensor(
        [candidate_position_by_id.get(str(image_id), -1) for image_id in image_ids],
        dtype=torch.long,
        device=DEVICE,
    )


def gather_bank_text(indices):
    shape = tuple(indices.shape) if torch.is_tensor(indices) else tuple(torch.as_tensor(indices).shape)
    positions = _candidate_positions(indices).to(candidate_text_bank.device)
    embeddings = candidate_text_bank.index_select(0, positions).reshape(*shape, -1)
    valid = candidate_text_valid.index_select(0, positions).reshape(*shape)
    return embeddings.to(DEVICE, non_blocking=True), valid.to(DEVICE, non_blocking=True)


def gather_bank_labels(indices):
""",
    "candidate text lookup helpers",
)
set_source(cells[14], helpers)

training = source_of(cells[16])
training = replace_once(
    training,
    'VALID_MODES = {"no_context", "original_tacs", "text_only", "mtacs", "mtacs_shuffled", "mtacs_dual_top3"}',
    'VALID_MODES = {"no_context", "original_tacs", "text_only", "mtacs", "mtacs_shuffled", "mtacs_dual_top3", "mtacs_dual_top3_context_text"}',
    "valid mode set",
)
training = replace_once(
    training,
    'return mode in {"original_tacs", "mtacs", "mtacs_shuffled", "mtacs_dual_top3"}',
    'return mode in {"original_tacs", "mtacs", "mtacs_shuffled", "mtacs_dual_top3", "mtacs_dual_top3_context_text"}',
    "selector modes",
)
training = replace_block(
    training,
    "def mode_logits(network, mode, query_images, text_embeddings, text_valid, context_images=None, return_features=False):\n",
    "def mode_uses_visual_selector(mode):\n",
    '''def mode_logits(
    network, mode, query_images, text_embeddings, text_valid, context_images=None,
    context_text_embeddings=None, context_text_valid=None, context_weights=None,
    return_features=False,
):
    if mode == "no_context":
        logits = network.forward_query_only(query_images)
        return (logits, None, None) if return_features else (logits, None)
    if mode == "original_tacs":
        logits, features, attention = network.forward_selected_context(
            query_images, context_images, return_features=True
        )
        return (logits, features, None) if return_features else (logits, None)
    if mode == "text_only":
        logits, gate = network.forward_query_with_text(query_images, text_embeddings, text_valid)
        return (logits, None, gate) if return_features else (logits, gate)
    if mode == "mtacs_dual_top3_context_text":
        logits, features, attention, gate = network.forward_selected_context_with_text(
            query_images, context_images, text_embeddings, text_valid,
            context_text_embeddings, context_text_valid, context_weights,
            return_features=True,
        )
        return (logits, features, gate) if return_features else (logits, gate)
    logits, features, attention, gate = network.forward_selected_context_with_text(
        query_images, context_images, text_embeddings, text_valid, return_features=True
    )
    return (logits, features, gate) if return_features else (logits, gate)


''',
    "mode logits multimodal branch",
)
training = replace_once(
    training,
    "details = network.select_full_pool(\n                    query, bank_device, text, text_valid, candidate_text_bank, candidate_text_valid, use_gumbel=True\n                )\n                selected_images = gather_bank_images(details[\"selected_indices\"])\n",
    "details = network.select_full_pool(\n"
    "                    query, bank_device, text, text_valid, candidate_text_bank, candidate_text_valid,\n"
    "                    query_candidate_positions(batch[\"image_id\"]), use_gumbel=True,\n"
    "                )\n"
    "                selected_images = gather_bank_images(details[\"selected_indices\"])\n"
    "                selected_context_text, selected_context_text_valid = gather_bank_text(details[\"selected_indices\"])\n"
    "                context_weights = details[\"context_weights\"]\n",
    "training selector/gather call",
)
training = replace_once(
    training,
    "network, EXPERIMENT_MODE, query, text, text_valid, selected_images, return_features=True\n",
    "network, EXPERIMENT_MODE, query, text, text_valid, selected_images,\n"
    "                selected_context_text if EXPERIMENT_MODE == \"mtacs_dual_top3_context_text\" else None,\n"
    "                selected_context_text_valid if EXPERIMENT_MODE == \"mtacs_dual_top3_context_text\" else None,\n"
    "                context_weights if EXPERIMENT_MODE == \"mtacs_dual_top3_context_text\" else None,\n"
    "                return_features=True,\n",
    "training multimodal forward",
)
training = replace_once(
    training,
    "                # Keep identical caption and mask on both reward branches.\n                # The reward therefore measures visual-context utility only.\n",
    "                # Query caption stays fixed; selected context images and captions vary together.\n",
    "reward description",
)
training = replace_once(
    training,
    "                        context_logits, _ = network.forward_selected_context_with_text(\n                            query, policy_context_images, text, text_valid\n                        )\n",
    "                        if EXPERIMENT_MODE == \"mtacs_dual_top3_context_text\":\n"
    "                            policy_context_text, policy_context_text_valid = gather_bank_text(details[\"selected_indices\"])\n"
    "                            context_logits, _ = network.forward_selected_context_with_text(\n"
    "                                query, policy_context_images, text, text_valid,\n"
    "                                policy_context_text, policy_context_text_valid, details[\"context_weights\"],\n"
    "                            )\n"
    "                        else:\n"
    "                            context_logits, _ = network.forward_selected_context_with_text(\n"
    "                                query, policy_context_images, text, text_valid\n"
    "                            )\n",
    "context reward forward",
)
training = replace_once(
    training,
    'details = network.select_full_pool(query, bank_device, text, text_valid, candidate_text_bank, candidate_text_valid, use_gumbel=False) if mode_uses_visual_selector(mode) else None\n',
    'details = network.select_full_pool(\n'
    '            query, bank_device, text, text_valid, candidate_text_bank, candidate_text_valid,\n'
    '            query_candidate_positions(batch["image_id"]), use_gumbel=False,\n'
    '        ) if mode_uses_visual_selector(mode) else None\n',
    "evaluation selector/self mask",
)
training = replace_once(
    training,
    '        context = gather_bank_images(details["selected_indices"]) if details is not None else None\n',
    '        context = gather_bank_images(details["selected_indices"]) if details is not None else None\n'
    '        context_text, context_text_valid = gather_bank_text(details["selected_indices"]) if details is not None else (None, None)\n'
    '        context_weights = details["context_weights"] if details is not None else None\n',
    "evaluation context text gather",
)
training = replace_once(
    training,
    '            logits, gate = mode_logits(network, mode, query, text, text_valid, context)\n',
    '            logits, gate = mode_logits(\n'
    '                network, mode, query, text, text_valid, context,\n'
    '                context_text if mode == "mtacs_dual_top3_context_text" else None,\n'
    '                context_text_valid if mode == "mtacs_dual_top3_context_text" else None,\n'
    '                context_weights if mode == "mtacs_dual_top3_context_text" else None,\n'
    '            )\n',
    "evaluation multimodal forward",
)
training = replace_once(
    training,
    '            context_ids = [str(candidate_df.iloc[pos]["image_id"]) for pos in positions]\n            context_fnames = [str(candidate_df.iloc[pos]["fname"]) for pos in positions]\n',
    '            context_ids = [str(candidate_df.iloc[pos]["image_id"]) for pos in positions]\n'
    '            context_fnames = [str(candidate_df.iloc[pos]["fname"]) for pos in positions]\n'
    '            context_captions = [str(caption_text_by_id.get(context_id, "")) for context_id in context_ids]\n',
    "evaluation context captions",
)
training = replace_once(
    training,
    '                "selected_context_fnames": "|".join(context_fnames),\n',
    '                "selected_context_fnames": "|".join(context_fnames),\n'
    '                "selected_context_captions": " | ".join(context_captions),\n',
    "prediction audit captions",
)
set_source(cells[16], training)

# Keep ablation comparison and metadata explicit about this distinct method.
if len(cells) > 18 and cells[18].get("cell_type") == "code":
    comparison = source_of(cells[18])
    comparison = replace_once(
        comparison,
        '"mtacs_dual_top3")',
        '"mtacs_dual_top3", "mtacs_dual_top3_context_text")',
        "comparison mode list",
    )
    set_source(cells[18], comparison)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

notebook["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {TARGET}")
