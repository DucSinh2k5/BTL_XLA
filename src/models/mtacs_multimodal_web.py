"""SUN397 K=1 M-TACS: visual tokens, two gated text residuals and dual selector."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

import tacs_visual_web as visual
import sun397_caption_web as captions


SCHEMA = "sun397_mtacs_full_both_web_v1"


def masked_z(values, valid):
    safe = torch.where(valid, values.float(), 0.0)
    count = valid.sum(1, keepdim=True).clamp_min(1)
    mean = safe.sum(1, keepdim=True) / count
    centered = torch.where(valid, values.float() - mean, 0.0)
    variance = centered.square().sum(1, keepdim=True) / count
    return centered / variance.clamp_min(1e-12).sqrt()


class MultimodalTACS(visual.VisualTACS):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.head = nn.Identity()
        dim, text_dim = config["feature_dim"], config["text_dim"]
        self.text_projection = nn.Sequential(nn.Linear(text_dim, dim), nn.LayerNorm(dim), nn.Dropout(0.1))
        def gate():
            result = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
            nn.init.constant_(result[-1].bias, -1.0)
            return result
        self.query_gate = gate()
        self.candidate_gate = gate()
        self.fusion_norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(nn.Dropout(config["head_dropout"]), nn.Linear(dim, config["num_classes"]))
        self.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))

    def score(self, query_images, candidate_features, excluded_positions=None,
              query_text=None, query_valid=None, candidate_text=None, candidate_valid=None):
        query = self.encode_selector(query_images).float()
        with torch.autocast(device_type=query.device.type, enabled=False):
            raw = query @ candidate_features.to(query.device, dtype=torch.float32).T
            eligible = torch.isfinite(visual.mask_self(torch.zeros_like(raw), excluded_positions))
            visual_scores = masked_z(raw, eligible)
            if any(value is None for value in (query_text, query_valid, candidate_text, candidate_valid)):
                raise ValueError("full_both selector requires aligned query/candidate captions and validity masks")
            qvalid = query_valid.to(query.device, dtype=torch.bool).reshape(-1)
            cvalid = candidate_valid.to(query.device, dtype=torch.bool).reshape(-1)
            qtext = torch.where(qvalid[:, None], query_text.to(query.device).float(), 0.0)
            ctext = torch.where(cvalid[:, None], candidate_text.to(query.device).float(), 0.0)
            semantic_raw = F.normalize(qtext, dim=1) @ F.normalize(ctext, dim=1).T
            valid = qvalid[:, None] & cvalid[None, :] & eligible
            semantic_scores = masked_z(semantic_raw, valid)
            scores = visual_scores + self.semantic_mix_logit.sigmoid() * semantic_scores
        return scores.masked_fill(~eligible, -torch.inf), query

    def forward(self, query_images, context_images=None, query_text=None, query_valid=None,
                context_text=None, context_valid=None, return_details=False, return_attention=False):
        tokens, rope, grid = self._tokens(query_images)
        count = grid[0] * grid[1]
        paired = context_images is not None
        if paired:
            context_tokens, _, context_grid = self._tokens(context_images)
            if context_grid != grid:
                raise ValueError("Query/context patch grids differ")
            tokens = torch.cat((tokens, context_tokens[:, self.prefix_count:]), dim=1)
            rope = visual.concatenate_rope(rope)
        features, attention = self._encode(tokens, rope, return_attention, grid=grid, paired=paired)
        if query_text is None or query_valid is None:
            raise ValueError("full_both always receives query text plus a validity mask")
        valid_q = query_valid.to(features.device, dtype=torch.bool).reshape(-1, 1)
        projected_q = self.text_projection(torch.where(valid_q, query_text.float(), 0.0))
        qgate = self.query_gate(torch.cat((features.float(), projected_q), 1)).sigmoid() * valid_q
        fused = features.float() + qgate * projected_q
        cgate = torch.zeros_like(qgate)
        if paired:
            if context_text is None or context_valid is None:
                raise ValueError("A selected candidate image requires its corresponding caption and mask")
            valid_c = context_valid.to(features.device, dtype=torch.bool).reshape(-1, 1)
            projected_c = self.text_projection(torch.where(valid_c, context_text.float(), 0.0))
            cgate = self.candidate_gate(torch.cat((features.float(), projected_c), 1)).sigmoid() * valid_c
            fused = fused + cgate * projected_c
        logits = self.classifier(self.fusion_norm(fused))
        details = {"q_gate": qgate[:, 0], "c_gate": cgate[:, 0], "semantic_mix": self.semantic_mix_logit.sigmoid()}
        if return_attention:
            details["attention"] = {"query_classifier": attention[:, self.prefix_count:self.prefix_count + count].reshape(-1, *grid)}
            if paired:
                details["attention"]["context_classifier"] = attention[:, self.prefix_count + count:].reshape(-1, *grid)
        return (logits, details) if return_details or return_attention else logits


def build_model(repo_dir, config, weights_path=None):
    if config["schema"] != SCHEMA or config["text_dim"] != 512 or config["method"] != "M_TACS_full_both":
        raise ValueError("Expected SUN397 full_both M-TACS configuration")
    backbone = visual.build_model(repo_dir, config, weights_path).backbone
    return MultimodalTACS(backbone, config)


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = visual.read_json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "ready":
        raise ValueError("Incomplete or non-M-TACS bundle")
    required = {"model.pt", "candidate_features.pt", "candidate_text.pt", "text_embeddings.pt", "captions.csv",
                "config.json", "class_names.json", "candidate_pool.json", "caption_settings.json",
                "runtime/tacs_visual_web.py", "runtime/mtacs_multimodal_web.py", "runtime/sun397_caption_web.py",
                "vendor/dinov3/dinov3/hub/backbones.py", "models/florence/config.json", "models/clip/config.json"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("Bundle is missing required visual/text/captioner artifacts")
    for name, info in manifest["files"].items():
        path = visual.bundle_path(root, name)
        if not path.is_file() or path.stat().st_size != info["bytes"] or visual.file_sha256(path) != info["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class BundlePredictor:
    """Local image -> caption -> CLIP -> context selection -> full_both prediction."""
    def __init__(self, root, device="cpu", model_factory=build_model, caption_runner=None, cache_dir=None):
        self.root, self.device = Path(root).resolve(), torch.device(device)
        self.manifest = verify_bundle(self.root)
        self.config = visual.read_json(self.root / "config.json")
        self.classes = visual.read_json(self.root / "class_names.json")
        self.pool = visual.read_json(self.root / "candidate_pool.json")
        state = torch.load(self.root / "model.pt", map_location="cpu", weights_only=True)
        if (state["config"] != self.config or state["class_names"] != self.classes or len(self.classes) != self.config["num_classes"]
                or len(set(self.classes)) != len(self.classes)):
            raise ValueError("Model/config/class mapping mismatch")
        ids = [r["image_id"] for r in self.pool]
        if len(set(ids)) != len(ids) or len(ids) != self.config["candidate_count"]:
            raise ValueError("Invalid candidate IDs/count")
        for position, row in enumerate(self.pool):
            if (position != row["position"] or row["file"] not in self.manifest["files"]
                    or row["sha256"] != self.manifest["files"][row["file"]]["sha256"]
                    or self.classes[row["class_id"]] != row["class_name"]):
                raise ValueError("Candidate image/class/order mismatch")
        bank = torch.load(self.root / "candidate_features.pt", map_location="cpu", weights_only=True)
        texts = torch.load(self.root / "candidate_text.pt", map_location="cpu", weights_only=True)
        self.text_cache = torch.load(self.root / "text_embeddings.pt", map_location="cpu", weights_only=True)
        self.caption_rows = captions.load_csv(self.root / "captions.csv")
        model_sha = visual.file_sha256(self.root / "model.pt")
        for item in (bank, texts):
            if item["image_ids"] != ids or item["checkpoint_sha256"] != model_sha:
                raise ValueError("Candidate bank belongs to another model/order")
        expected_csv_sha = visual.file_sha256(self.root / "captions.csv")
        if self.text_cache["csv_sha256"] != expected_csv_sha or texts["csv_sha256"] != expected_csv_sha:
            raise ValueError("Text bank belongs to another caption CSV")
        if set(self.text_cache["image_ids"]) != set(self.caption_rows) or len(set(self.text_cache["image_ids"])) != len(self.caption_rows):
            raise ValueError("Text cache image IDs do not match CSV")
        self.text_index = {image_id: i for i, image_id in enumerate(self.text_cache["image_ids"])}
        self.hash_index = {}
        for image_id, row in self.caption_rows.items():
            self.hash_index.setdefault(row["image_sha256"], []).append(image_id)
        for name, item, count, dim in (("visual", bank, len(ids), self.config["feature_dim"]),
                                      ("candidate text", texts, len(ids), self.config["text_dim"]),
                                      ("all text", self.text_cache, len(self.caption_rows), self.config["text_dim"])):
            values = item["features"] if name == "visual" else item["embeddings"]
            if tuple(values.shape) != (count, dim) or not torch.isfinite(values).all():
                raise ValueError(f"Invalid {name} features")
            if name != "visual" and (torch.count_nonzero(values[~item["valid"].bool()]) != 0):
                raise ValueError("Masked captions must have exactly zero vectors")
            if name != "visual":
                norms = values[item["valid"].bool()].norm(dim=1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                    raise ValueError("Valid CLIP vectors are not unit normalized")
        positions = torch.tensor([self.text_index[image_id] for image_id in ids])
        if not torch.equal(texts["embeddings"], self.text_cache["embeddings"][positions]) or not torch.equal(texts["valid"], self.text_cache["valid"][positions]):
            raise ValueError("Candidate caption vectors are not aligned to selected image IDs")
        self.model = model_factory(self.root / "vendor/dinov3", self.config).to(self.device)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.eval()
        self.bank = bank["features"].float().to(self.device)
        self.text_bank = texts["embeddings"].float().to(self.device)
        self.text_valid = texts["valid"].bool().to(self.device)
        self.caption_settings = visual.read_json(self.root / "caption_settings.json")
        if (self.caption_settings["settings"]["policy"] != self.config["caption_policy"]
                or self.caption_settings["classes"] != self.classes):
            raise ValueError("Web/train caption policy mismatch")
        asset_ids = self.caption_settings["asset_identity"]
        if self.text_cache["clip_sha256"] != asset_ids["clip_sha256"] or self.text_cache["captioner_sha256"] != asset_ids["florence_sha256"]:
            raise ValueError("Web text model identity differs from training cache")
        for image_id, row in self.caption_rows.items():
            if (row["clip_assets_sha256"] != asset_ids["clip_sha256"] or row["captioner_assets_sha256"] != asset_ids["florence_sha256"]
                    or row["caption_policy"] != self.config["caption_policy"]
                    or captions.truth(row["caption_valid"]) != bool(self.text_cache["valid"][self.text_index[image_id]])):
                raise ValueError("Caption row/vector/policy binding mismatch")
        self.caption_runner = caption_runner
        self.cache_dir = Path(cache_dir or (Path(tempfile.gettempdir()) / ("mtacs_caption_" + model_sha[:12])))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_logits = self.last_query_embedding = self.last_query_valid = None

    def _caption(self, source, image, image_id, use_cache):
        digest = visual.file_sha256(source) if isinstance(source, (str, Path)) else hashlib.sha256(source).hexdigest() if isinstance(source, bytes) else None
        eligible = self.hash_index.get(digest, [])
        known = image_id if image_id in eligible else eligible[0] if len(eligible) == 1 else None
        if use_cache and known is not None:
            position = self.text_index[known]
            return self.caption_rows[known], self.text_cache["embeddings"][position:position + 1], self.text_cache["valid"][position:position + 1], "verified_cache", digest
        with tempfile.TemporaryDirectory(prefix="query_", dir=self.cache_dir) as temporary:
            path = Path(temporary) / "query.png"
            image.save(path)
            request = {**self.caption_settings, "mode": "predict", "image": str(path), "device": str(self.device),
                       "florence": str(self.root / "models/florence"), "clip": str(self.root / "models/clip")}
            if self.caption_runner is None:
                env = captions.caption_environment(self.cache_dir / "runtime", self.root / "offline_wheels")
                output = captions.invoke_worker(request, env, self.root / "runtime/sun397_caption_web.py", self.cache_dir)
            else:
                output = self.caption_runner(request)
        return output["rows"][0], output["embeddings"], output["valid"], "fresh_florence", digest

    def predict(self, source, top_k=5, explain=True, image_id=None, use_caption_cache=True):
        image = visual.read_rgb(source)
        row, text, valid, caption_source, digest = self._caption(source, image, image_id, use_caption_cache)
        if text.shape != (1, self.config["text_dim"]) or valid.shape != (1,) or not torch.isfinite(text).all():
            raise ValueError("Caption worker returned invalid query vectors")
        if bool(valid[0]) != captions.truth(row["caption_valid"]) or (not valid[0] and torch.count_nonzero(text)):
            raise ValueError("Query caption text/mask/vector disagree")
        query = visual.preprocess(image, self.config)[None].to(self.device)
        text, valid = text.to(self.device), valid.to(self.device)
        self.last_query_embedding, self.last_query_valid = text.cpu(), valid.cpu()
        with torch.no_grad():
            scores, _ = self.model.score(query, self.bank, query_text=text, query_valid=valid,
                                         candidate_text=self.text_bank, candidate_valid=self.text_valid)
            excluded = torch.tensor([r["image_id"] == image_id or (digest is not None and r["sha256"] == digest) for r in self.pool], device=self.device)
            # Exclude before z-normalization, not after: the score's moments must
            # also omit every identical uploaded candidate image.
            if excluded.any():
                features = self.model.encode_selector(query).float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    eligible = (~excluded)[None]
                    raw_v = features @ self.bank.T
                    qtext = torch.where(valid[:, None], text.float(), 0.0)
                    raw_s = F.normalize(qtext, dim=1) @ F.normalize(self.text_bank, dim=1).T
                    scores = (masked_z(raw_v, eligible) + self.model.semantic_mix_logit.sigmoid()
                              * masked_z(raw_s, eligible & valid[:, None] & self.text_valid[None]))
                    scores.masked_fill_(~eligible, -torch.inf)
            if not torch.isfinite(scores).any():
                raise ValueError("No candidate remains after self-exclusion")
            position = int(scores.argmax(1))
            candidate = self.pool[position]
            candidate_image = visual.read_rgb(visual.bundle_path(self.root, candidate["file"]))
            context = visual.preprocess(candidate_image, self.config)[None].to(self.device)
            logits, details = self.model(query, context, text, valid, self.text_bank[position:position + 1],
                                         self.text_valid[position:position + 1], return_details=True, return_attention=explain)
            self.last_logits = logits.float().cpu()
            probabilities, indices = logits.float().softmax(1)[0].topk(min(max(int(top_k), 1), len(self.classes)))
            selector_map = self.model.encode_selector(context, return_attention=True)[1][0] if explain else None
        caption = self.caption_rows[candidate["image_id"]]
        result = {"model": "mtacs", "dataset": "SUN397", "bundle_schema": SCHEMA,
            "prediction": {"class_id": int(indices[0]), "label": self.classes[int(indices[0])], "confidence": float(probabilities[0])},
            "top_k": [{"class_id": int(i), "label": self.classes[int(i)], "confidence": float(v)} for v, i in zip(probabilities, indices)],
            "evidence": {"query_caption": {"text": row["final_caption"], "raw_text": row["raw_caption"], "valid": bool(valid[0]),
                                          "status": "ready" if valid[0] else "masked", "source": caption_source, "policy": self.config["caption_policy"]},
                "candidate": {"image_id": candidate["image_id"], "position": position, "image_url": visual.png_data_url(candidate_image),
                              "caption": caption["final_caption"], "caption_valid": bool(self.text_valid[position]),
                              "selector_score": float(scores[0, position]), "status": "ready"},
                "fusion": {"q_gate": float(details["q_gate"][0]), "c_gate": float(details["c_gate"][0]),
                           "semantic_mix": float(details["semantic_mix"])}, "attention": {}}}
        if explain:
            for role, raw, original in (("query_classifier", details["attention"]["query_classifier"][0], image),
                                        ("candidate_selector", selector_map, candidate_image)):
                array = raw.float().cpu().numpy()
                overlay, informative = visual.attention_overlay(original, array)
                result["evidence"]["attention"][role] = {"image_url": visual.png_data_url(overlay),
                    "status": "ready" if informative else "flat_attention", "raw_grid": array.tolist(), "grid_shape": list(array.shape),
                    "method": visual.ATTENTION_METHOD, "class_conditional": False,
                    "note": "Visual backbone self-attention, not text attribution or causal/Grad-CAM explanation"}
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline SUN397 full_both M-TACS inference")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    parser.add_argument("--fresh-caption", action="store_true")
    args = parser.parse_args()
    predictor = BundlePredictor(args.bundle, args.device)
    visual.write_json(args.output, predictor.predict(args.image, use_caption_cache=not args.fresh_caption))
    print(args.output)


if __name__ == "__main__":
    main()
