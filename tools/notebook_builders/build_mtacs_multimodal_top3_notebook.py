"""Create a separate SUN397 experiment with top-3 visual+semantic contexts."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import copy
import json
from pathlib import Path


ROOT = PROJECT_ROOT
SOURCE = project_path('mtacs-train-captions-vitl16.ipynb')
TARGET = project_path('mtacs-train-captions-vitl16-multimodal-top3.ipynb')


def cell_source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence of {label}, found {count}")
    return source.replace(old, new, 1)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

# Give the experiment a distinct output/checkpoint directory and select K=3.
config = cell_source(cells[3])
config = replace_once(config, 'EXPERIMENT_MODE = "mtacs"', 'EXPERIMENT_MODE = "mtacs_dual_top3"', "experiment mode")
config = replace_once(config, 'mtacs_shuffled\n', 'mtacs_shuffled | mtacs_dual_top3\n', "mode comment")
config = replace_once(config, 'PROMPT_VERSION = "florence2_detailed_v1"', 'PROMPT_VERSION = "florence2_detailed_v2"', "caption prompt version")
set_source(cells[3], config)

settings = cell_source(cells[5])
settings = replace_once(settings, "OUT_PUT_DIR = PROJECT_DIR / 'outputs' / 'mtacs_sun397_visual_text_context_vitl16'",
                        "OUT_PUT_DIR = PROJECT_DIR / 'outputs' / 'mtacs_sun397_visual_text_context_vitl16_dual_top3'", "output dir")
settings = replace_once(settings, "CONTEXT_TOP_K = 1", "CONTEXT_TOP_K = 3", "context K")
settings = replace_once(settings, "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 32, 128, 64",
                        "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 16, 128, 64", "96-GiB top-3 batch")
settings = replace_once(settings, "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 16, 64, 32",
                        "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 8, 64, 32", "40-GiB top-3 batch")
settings = replace_once(settings, "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 8, 32, 16",
                        "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 4, 32, 16", "20-GiB top-3 batch")
settings = replace_once(settings, "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 4, 16, 8",
                        "BATCH_SIZE, RETRIEVAL_BATCH_SIZE, CANDIDATE_FEATURE_BANK_BATCH_SIZE = 2, 16, 8", "small-GPU top-3 batch")
settings = replace_once(settings, "print(f'TACS selected context per query: {CONTEXT_TOP_K}')",
                        "print(f'Dual-modality selected contexts per query: {CONTEXT_TOP_K}')", "K status")
set_source(cells[5], settings)

model_cell = cell_source(cells[13])
model_cell = replace_once(
    model_cell,
    "        self.logit_scale = nn.Parameter(torch.tensor(1.0))\n",
    "        self.logit_scale = nn.Parameter(torch.tensor(1.0))\n        self.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))\n",
    "learnable semantic mix",
)
old_select = '''    def select_full_pool(self, query_images, candidate_feature_bank, use_gumbel=True):
        selector_scores, query_features = self.score_full_candidate_pool(query_images, candidate_feature_bank)
        if self.training and use_gumbel:
            selection_weights = F.gumbel_softmax(
                selector_scores,
                tau=TACS_GUMBEL_TAU,
                hard=True,
                dim=1,
            )
            selected_indices = selection_weights.argmax(dim=1)
        else:
            selected_indices = selector_scores.argmax(dim=1)
            selection_weights = F.one_hot(selected_indices, num_classes=selector_scores.size(1)).to(selector_scores.dtype)
        details = {
            'selector_scores': selector_scores,
            'selection_weights': selection_weights,
            'selected_indices': selected_indices,
            'query_features': query_features,
            'selector_entropy': torch.distributions.Categorical(logits=selector_scores.float()).entropy().mean(),
        }
        self.active_fusion_mode = 'tacs_full_pool_shared_backbone_selector'
        return details

    def forward_selected_context(self, query_images, selected_context_images, return_features=False):
        context_images = selected_context_images.unsqueeze(1)
        return super().forward(query_images, context_images, context_scores=None, return_features=return_features)
'''
new_select = '''    def select_full_pool(
        self, query_images, candidate_feature_bank, query_text, query_text_valid,
        candidate_text_bank, candidate_text_valid, use_gumbel=True,
    ):
        visual_scores, query_features = self.score_full_candidate_pool(query_images, candidate_feature_bank)
        visual_scores = visual_scores.float()
        visual_z = (visual_scores - visual_scores.mean(dim=1, keepdim=True)) / visual_scores.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)

        query_text = F.normalize(query_text.float(), dim=1)
        candidate_text_bank = F.normalize(candidate_text_bank.float().to(query_text.device), dim=1)
        semantic_raw = query_text @ candidate_text_bank.T
        sem_valid = query_text_valid.bool().unsqueeze(1) & candidate_text_valid.bool().to(query_text.device).unsqueeze(0)
        sem_count = sem_valid.sum(dim=1, keepdim=True).clamp_min(1)
        sem_mean = (semantic_raw * sem_valid).sum(dim=1, keepdim=True) / sem_count
        sem_var = (((semantic_raw - sem_mean) * sem_valid) ** 2).sum(dim=1, keepdim=True) / sem_count
        semantic_z = ((semantic_raw - sem_mean) / sem_var.sqrt().clamp_min(1e-6)) * sem_valid

        semantic_mix = torch.sigmoid(self.selector.semantic_mix_logit)
        selector_scores = visual_z + semantic_mix * semantic_z
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
        details = {
            'selector_scores': selector_scores,
            'selected_indices': selected_indices,
            'selection_log_probs': selection_log_probs,
            'selector_entropy': entropy,
            'semantic_mix': semantic_mix.detach(),
            'query_features': query_features,
        }
        self.active_fusion_mode = 'dual_visual_semantic_topk_selector'
        return details

    def forward_selected_context(self, query_images, selected_context_images, return_features=False):
        context_images = selected_context_images if selected_context_images.ndim == 5 else selected_context_images.unsqueeze(1)
        return super().forward(query_images, context_images, context_scores=None, return_features=return_features)
'''
model_cell = replace_once(model_cell, old_select, new_select, "selector implementation")
model_cell = replace_once(
    model_cell,
    "candidate_image_bank, candidate_labels_tensor, candidate_fnames = build_candidate_image_bank(\n    candidate_df,\n    eval_transform,\n    device=torch.device(CANDIDATE_IMAGE_BANK_DEVICE),\n    dtype=CANDIDATE_IMAGE_BANK_DTYPE,\n)\n",
    "candidate_image_bank, candidate_labels_tensor, candidate_fnames = build_candidate_image_bank(\n    candidate_df,\n    eval_transform,\n    device=torch.device(CANDIDATE_IMAGE_BANK_DEVICE),\n    dtype=CANDIDATE_IMAGE_BANK_DTYPE,\n)\n\n# Caption vectors are aligned to candidate_df; invalid/leaking captions remain visual-only candidates.\ncandidate_text_entries = [text_embedding_lookup[str(image_id)] for image_id in candidate_df['image_id']]\ncandidate_text_bank = torch.stack([entry[0] for entry in candidate_text_entries]).to(DEVICE, dtype=torch.float32)\ncandidate_text_valid = torch.tensor([entry[1] for entry in candidate_text_entries], dtype=torch.bool, device=DEVICE)\nprint('Candidate semantic vectors:', tuple(candidate_text_bank.shape), '| valid:', int(candidate_text_valid.sum()))\n",
    "candidate bank + aligned candidate captions",
)
model_cell = replace_once(
    model_cell,
    "candidate_text_valid = torch.tensor([entry[1] for entry in candidate_text_entries], dtype=torch.bool, device=DEVICE)\n",
    "candidate_text_valid = torch.tensor([entry[1] for entry in candidate_text_entries], dtype=torch.bool, device=DEVICE)\n"
    "candidate_labels_device = candidate_labels_tensor.to(DEVICE, non_blocking=True)\n",
    "candidate labels on training device",
)
model_cell = model_cell.replace(
    "'selector_mode': 'full_candidate_pool_shared_backbone',",
    "'selector_mode': 'learned_visual_semantic_top3_without_replacement',",
)
set_source(cells[13], model_cell)

bank_helpers = cell_source(cells[14])
bank_helpers = replace_once(
    bank_helpers,
    "def gather_bank_images(indices):\n    positions = _candidate_positions(indices)\n    return candidate_image_bank.index_select(0, positions).to(\n        DEVICE, non_blocking=True\n    )\n",
    "def gather_bank_images(indices):\n    shape = tuple(indices.shape) if torch.is_tensor(indices) else tuple(torch.as_tensor(indices).shape)\n    positions = _candidate_positions(indices)\n    selected = candidate_image_bank.index_select(0, positions).to(DEVICE, non_blocking=True)\n    return selected.reshape(*shape, *candidate_image_bank.shape[1:])\n",
    "multi-context gather",
)
bank_helpers = replace_once(
    bank_helpers,
    "def gather_bank_labels(indices):\n    positions = _candidate_positions(indices).to(candidate_labels_tensor.device)\n    return candidate_labels_tensor.index_select(0, positions).to(\n        DEVICE, non_blocking=True\n    )\n",
    "def gather_bank_labels(indices):\n    shape = tuple(indices.shape) if torch.is_tensor(indices) else tuple(torch.as_tensor(indices).shape)\n    positions = _candidate_positions(indices).to(candidate_labels_tensor.device)\n    selected = candidate_labels_tensor.index_select(0, positions).to(DEVICE, non_blocking=True)\n    return selected.reshape(*shape)\n",
    "multi-context label gather",
)
set_source(cells[14], bank_helpers)

training = cell_source(cells[16])
training = replace_once(
    training,
    'VALID_MODES = {"no_context", "original_tacs", "text_only", "mtacs", "mtacs_shuffled"}',
    'VALID_MODES = {"no_context", "original_tacs", "text_only", "mtacs", "mtacs_shuffled", "mtacs_dual_top3"}',
    "valid experiment mode",
)
training = replace_once(
    training,
    'return mode in {"original_tacs", "mtacs", "mtacs_shuffled"}',
    'return mode in {"original_tacs", "mtacs", "mtacs_shuffled", "mtacs_dual_top3"}',
    "enable dual selector",
)
training = replace_once(
    training,
    "details = network.select_full_pool(query, bank_device, use_gumbel=True)\n                selected_images = select_bank_images_from_weights(details[\"selection_weights\"])\n",
    "details = network.select_full_pool(\n                    query, bank_device, text, text_valid, candidate_text_bank, candidate_text_valid, use_gumbel=True\n                )\n                selected_images = gather_bank_images(details[\"selected_indices\"])\n",
    "train selector call",
)
training = replace_once(
    training,
    """                selector_scores = details[\"selector_scores\"].float()
                policy_dist = torch.distributions.Categorical(logits=selector_scores)
                policy_actions = policy_dist.sample()
                log_probs = policy_dist.log_prob(policy_actions)
                entropy = policy_dist.entropy().mean()
                policy_context_images = gather_bank_images(policy_actions)
""",
    """                log_probs = details[\"selection_log_probs\"].sum(dim=1)
                entropy = details[\"selector_entropy\"]
                policy_context_images = selected_images
""",
    "top-k policy actions",
)
training = replace_once(
    training,
    "selected_same += (gather_bank_labels(details[\"selected_indices\"]) == labels).sum().item()",
    "selected_same += (gather_bank_labels(details[\"selected_indices\"]) == labels[:, None]).sum().item()",
    "train same-class rate",
)
same_rate = '"selected_same_class_rate": selected_same / denominator,'
if training.count(same_rate) != 2:
    raise RuntimeError(f"Expected two same-class metrics, found {training.count(same_rate)}")
training = training.replace(same_rate, '"selected_same_class_rate": selected_same / (denominator * CONTEXT_TOP_K),')
training = replace_once(
    training,
    'details = network.select_full_pool(query, bank_device, use_gumbel=False) if mode_uses_visual_selector(mode) else None\n',
    'details = network.select_full_pool(query, bank_device, text, text_valid, candidate_text_bank, candidate_text_valid, use_gumbel=False) if mode_uses_visual_selector(mode) else None\n',
    "eval selector call",
)
training = replace_once(
    training,
    """        for j in range(count):
            pos = int(details[\"selected_indices\"][j]) if details is not None else -1
            row = {
                \"image_id\": str(batch[\"image_id\"][j]), \"fname\": str(batch[\"fname\"][j]),
                \"ground_truth\": int(labels[j]), \"prediction\": int(preds[j]),
                \"correct\": bool(labels[j] == preds[j]),
                \"caption\": str(caption_text_by_id.get(str(batch[\"image_id\"][j]), \"\")),
                \"caption_valid\": bool(text_valid[j]), \"gate_value\": float(gate[j]) if gate is not None else np.nan,
                \"selected_context_position\": pos,
                \"selected_context_id\": str(candidate_df.iloc[pos][\"image_id\"]) if pos >= 0 else \"\",
                \"selected_context_fname\": str(candidate_df.iloc[pos][\"fname\"]) if pos >= 0 else \"\",
                \"selector_probability\": float(torch.softmax(details[\"selector_scores\"][j].float(), dim=0)[pos]) if pos >= 0 else np.nan,
            }
            rows.append(row)
            if details is not None:
                selected_same += int(candidate_labels_tensor[pos].item() == int(labels[j]))
""",
    """        for j in range(count):
            positions = details[\"selected_indices\"][j].tolist() if details is not None else []
            context_ids = [str(candidate_df.iloc[pos][\"image_id\"]) for pos in positions]
            context_fnames = [str(candidate_df.iloc[pos][\"fname\"]) for pos in positions]
            row = {
                \"image_id\": str(batch[\"image_id\"][j]), \"fname\": str(batch[\"fname\"][j]),
                \"ground_truth\": int(labels[j]), \"prediction\": int(preds[j]),
                \"correct\": bool(labels[j] == preds[j]),
                \"caption\": str(caption_text_by_id.get(str(batch[\"image_id\"][j]), \"\")),
                \"caption_valid\": bool(text_valid[j]), \"gate_value\": float(gate[j]) if gate is not None else np.nan,
                \"selected_context_positions\": positions,
                \"selected_context_ids\": \"|\".join(context_ids),
                \"selected_context_fnames\": \"|\".join(context_fnames),
                \"selected_context_probabilities\": [float(torch.softmax(details[\"selector_scores\"][j].float(), dim=0)[pos]) for pos in positions] if details is not None else [],
            }
            rows.append(row)
            if details is not None:
                same = candidate_labels_tensor[details[\"selected_indices\"][j]].to(labels.device) == labels[j]
                selected_same += int(same.sum().item())
""",
    "multi-context evaluation records",
)
training = replace_once(
    training,
    'same = candidate_labels_tensor[details["selected_indices"][j]].to(labels.device) == labels[j]',
    'same = candidate_labels_device[details["selected_indices"][j].to(candidate_labels_device.device)] == labels[j]',
    "evaluation candidate label device",
)
training = replace_once(
    training,
    '"selector_mode": "full_candidate_pool_shared_backbone"',
    '"selector_mode": "learned_visual_semantic_top3_without_replacement"',
    "checkpoint selector metadata",
)
training = replace_once(
    training,
    '"candidate_pool_size": len(candidate_df), "tacs_gumbel_tau": TACS_GUMBEL_TAU,',
    '"candidate_pool_size": len(candidate_df), "context_top_k": CONTEXT_TOP_K, "tacs_gumbel_tau": TACS_GUMBEL_TAU,',
    "checkpoint config metadata",
)
training = replace_once(
    training,
    '"fusion": "gated_additive_cls_representation",',
    '"fusion": "gated_additive_cls_representation", "selector_semantic_mix": float(torch.sigmoid(model.selector.semantic_mix_logit).detach().cpu()),',
    "saved semantic mix",
)
set_source(cells[16], training)

if len(cells) > 18 and cells[18].get("cell_type") == "code":
    comparison = cell_source(cells[18])
    comparison = replace_once(
        comparison,
        '("no_context", "original_tacs", "text_only", "mtacs", "mtacs_shuffled")',
        '("no_context", "original_tacs", "text_only", "mtacs", "mtacs_shuffled", "mtacs_dual_top3")',
        "metrics comparison modes",
    )
    set_source(cells[18], comparison)

# New experiment, fresh execution state, and independent artifact path.
for cell in cells:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
notebook["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {TARGET}")
