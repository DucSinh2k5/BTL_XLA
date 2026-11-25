"""Embedded notebook helpers for study06 (label-aware candidate text); runtime globals come from the notebook."""
from __future__ import annotations


def validate_label_context_output_directory(path):
    path = Path(path)
    protocol_path = path / "protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (previous.get("schema") != "dtd_prefix_label_context_v1"
                or previous.get("architecture_study", {}).get("stage") != ARCHITECTURE_STUDY["stage"]):
            raise ValueError("Output belongs to another experiment; study06 must not overwrite earlier studies")
    elif path.is_dir() and any(path.glob("runs/*/seed_*/*.pt")):
        raise ValueError("Output has checkpoints but no study06 protocol; choose a new output directory")


def candidate_context_texts(candidate_ids, candidate_class_ids, class_names, caption_view, template):
    """Training-pool label prompt, plus the candidate's own screened caption when it is valid."""
    if len(candidate_ids) != len(candidate_class_ids):
        raise ValueError("Candidate IDs and class IDs must align")
    lookup = {image_id: index for index, image_id in enumerate(caption_view["image_ids"])}
    texts = []
    for image_id, class_id in zip(candidate_ids, candidate_class_ids):
        label = str(class_names[int(class_id)]).replace("_", " ")
        text = template.format(label=label)
        index = lookup.get(image_id)
        if index is not None and bool(caption_view["valid"][index]) and caption_view["texts"][index]:
            text = f"{text}, {caption_view['texts'][index]}"
        texts.append(text)
    return texts


def make_label_context_banks(caption_banks, label_view, candidate_ids, device):
    """Keep the study01 bank layout; replace only candidate text with the label-aware view."""
    if list(label_view["image_ids"]) != list(candidate_ids):
        raise ValueError("Label-context view must follow the candidate pool order exactly")
    if not bool(torch.as_tensor(label_view["valid"]).all()):
        raise ValueError("Every training-pool candidate has a label, so every label-context text must be valid")

    def banks_for(spec, seed, override=None):
        banks = caption_banks(spec, seed, override)
        banks["c"] = {"embeddings": label_view["embeddings"].to(device),
                      "valid": torch.ones(len(candidate_ids), dtype=torch.bool, device=device),
                      "source_ids": list(candidate_ids), "texts": list(label_view["texts"])}
        return banks
    return banks_for


def make_label_context_model_class(base_class, visual_class):
    original_class = make_ablation_model_class(base_class, visual_class)

    class DTDLabelContextModel(original_class):
        """Exact study01 fusion math; only adds the fusion diagnostics the validation interventions read."""

        def forward_ablated(self, query_images, context_images, q_text, q_valid, c_text, c_valid, settings):
            if context_images is None:
                visual = self.query_norm(self._pool_backbone_features(query_images))
            else:
                _, visual, _ = visual_class.forward_selected_context(self, query_images, context_images, return_features=True)
            q_features = self.text_projection(q_text.float())
            c_features = self.context_text_projection(c_text.float())
            q_gate = torch.sigmoid(self.text_gate(torch.cat([visual.float(), q_features], 1))).squeeze(1)
            c_gate = torch.sigmoid(self.context_text_gate(torch.cat([visual.float(), c_features], 1))).squeeze(1)
            q_gate = q_gate * q_valid.to(q_gate.dtype) * bool(settings["query_fusion"])
            c_gate = c_gate * c_valid.to(c_gate.dtype) * bool(settings["candidate_fusion"])
            fused = self.mtacs_norm(visual.float() + q_gate[:, None] * q_features + c_gate[:, None] * c_features)
            result = {"logits": self.mtacs_classifier(fused), "features": F.normalize(fused, dim=1),
                      "query_gate": q_gate, "candidate_gate": c_gate}
            if getattr(self, "capture_fusion_diagnostics", False):
                visual_norm = visual.detach().float().norm(dim=1)
                text_norm = c_features.detach().float().norm(dim=1)
                weighted = (c_gate[:, None] * c_features).detach().float().norm(dim=1)
                result["fusion_diagnostics"] = {
                    "visual_norm": visual_norm, "candidate_raw_projection_norm": text_norm,
                    "candidate_fusion_norm": text_norm, "weighted_candidate_norm": weighted,
                    "weighted_text_visual_ratio": weighted / visual_norm.clamp_min(FUSION_NORM_EPS)}
            return result

    return DTDLabelContextModel
