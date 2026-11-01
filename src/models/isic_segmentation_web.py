"""Frozen DINOv3 + per-block gated context attention for ISIC deployment.

One frozen backbone is shared by selector feature extraction and segmentation.
Only utility projections, inserted attention, DPT and optional text fusion learn.
Masks and diagnostic labels are never inference or context-selection inputs.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import tacs_visual_web as io_utils


SCHEMA = "isic2017_segmentation_web_v1"
BACKBONES = {"dinov3_vitl16": (1024, 24, 16, "8aa4cbdd"),
             "dinov3_vits16": (384, 12, 6, "08c60483")}
ATTENTION_METHOD = "last_block_cls_to_patch_mean_heads"


def build_backbone(repo, config, weights=None):
    name = config["backbone"]
    if name not in BACKBONES or config["feature_dim"] != BACKBONES[name][0]:
        raise ValueError("Backbone/feature dimension mismatch; no silent model substitution")
    repo = Path(repo).resolve()
    if not (repo / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"Missing official DINOv3 source: {repo}")
    package = sys.modules.get("dinov3")
    if package is not None and not Path(package.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("A different DINOv3 repository is imported; restart the kernel")
    sys.path.insert(0, str(repo))
    module = importlib.import_module("dinov3.hub.backbones")
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("DINOv3 source did not resolve inside the requested repository")
    kwargs = {"pretrained": weights is not None}
    if weights is not None:
        weights = Path(weights).resolve()
        suffix = f"lvd1689m-{BACKBONES[name][3]}.pth"
        if not weights.is_file() or not weights.name.endswith(suffix):
            raise ValueError(f"Use the original official {name} LVD1689M checkpoint ({suffix})")
        kwargs["weights"] = str(weights)
    backbone = getattr(module, name)(**kwargs)
    if backbone.embed_dim != config["feature_dim"] or len(backbone.blocks) != BACKBONES[name][1]:
        raise ValueError("Loaded backbone has the wrong width or depth")
    return backbone


def build_model(repo, config, weights=None):
    return ISICSegmentor(build_backbone(repo, config, weights), config)


class _HardBank(torch.autograd.Function):
    """Exact ST derivative w.r.t. selection weights over a fixed CPU token bank."""
    @staticmethod
    def forward(ctx, weights, bank, chunk):
        if weights.ndim != 2 or weights.shape[1] != len(bank):
            raise ValueError("Selection/token bank shape mismatch")
        one_hot = F.one_hot(weights.argmax(1), len(bank)).to(weights.dtype)
        if not torch.equal(weights.detach(), one_hot):
            raise ValueError("Hard forward requires one-hot ST weights")
        ctx.save_for_backward(bank)
        ctx.chunk = int(chunk)
        ctx.weight_dtype = weights.dtype
        return bank.index_select(0, weights.argmax(1).to(bank.device)).to(weights.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, gradient):
        (bank,) = ctx.saved_tensors
        pieces = []
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            for offset in range(0, len(bank), ctx.chunk):
                block = bank[offset:offset + ctx.chunk].to(gradient.device, dtype=torch.float32)
                pieces.append(gradient.float().flatten(1) @ block.flatten(1).T)
        return torch.cat(pieces, 1).to(ctx.weight_dtype), None, None


def select_bank(weights, bank, chunk=8):
    return _HardBank.apply(weights, bank, chunk)


class UtilityHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Frozen CLS features alone cannot learn utility. These projections learn
        # from BOTH task gradients and policy gradients, sharing the DINO encoder.
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.candidate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.log_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, candidates):
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = F.normalize(self.query(query.float()), dim=-1)
            c = F.normalize(self.candidate(candidates.float()), dim=-1)
            return (q @ c.T) * self.log_scale.clamp(0.01, 100)


class GatedContextAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, query, context):
        q, c = self.query_norm(query), self.context_norm(context)
        update, _ = self.attention(q, c, c, need_weights=False)
        return query + torch.sigmoid(self.gate_logit) * update


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels), nn.GELU())

    def forward(self, x):
        return x + self.layers(x)


class DPTLite(nn.Module):
    """Four-depth reassembly at 4x/2x/1x/0.5x and top-down residual fusion."""
    def __init__(self, dim, channels):
        super().__init__()
        self.project = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, channels)) for _ in range(4)])
        self.resize = nn.ModuleList([nn.ConvTranspose2d(channels, channels, 4, stride=4),
            nn.ConvTranspose2d(channels, channels, 2, stride=2), nn.Identity(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)])
        self.refine = nn.ModuleList([ResidualConv(channels) for _ in range(4)])
        self.head = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(channels // 2, 1, 1))

    def forward(self, stages, grid, size):
        if len(stages) != 4:
            raise ValueError("DPT requires four distinct transformer depths")
        maps = [resize(proj(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, *grid))
                for tokens, proj, resize in zip(stages, self.project, self.resize)]
        x = self.refine[3](maps[3])
        for index in (2, 1, 0):
            x = F.interpolate(x, size=maps[index].shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine[index](x + maps[index])
        return F.interpolate(self.head(x), size=(size, size), mode="bilinear", align_corners=False)


class TextFusion(nn.Module):
    def __init__(self, text_dim, dim):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, dim), nn.GELU())
        self.gate = nn.Linear(dim * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, patches, text, valid):
        projected = self.projection(text.float()).to(patches.dtype)
        gate = torch.sigmoid(self.gate(torch.cat((patches.mean(1).float(), projected.float()), -1)))
        gate = gate * valid.float().reshape(-1, 1)
        return patches + gate[:, :, None] * projected[:, None, :], gate[:, 0]


class ISICSegmentor(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.config = dict(config)
        dim = int(backbone.embed_dim)
        if dim != int(config["feature_dim"]):
            raise ValueError("Backbone/config dimension mismatch")
        self.prefix = 1 + int(backbone.n_storage_tokens)
        self.grid = (config["image_size"] // 16,) * 2
        self.stage_indices = [round(len(backbone.blocks) * fraction) - 1 for fraction in (.25, .5, .75, 1)]
        if len(set(self.stage_indices)) != 4:
            raise ValueError("Backbone must expose four distinct depths")
        self.selector = UtilityHead(dim)
        self.context_blocks = nn.ModuleList([GatedContextAttention(dim, config["cross_heads"])
                                            for _ in backbone.blocks])
        self.decoder = DPTLite(dim, config["decoder_channels"])
        self.multimodal = config["mode"] == "mtacs"
        if self.multimodal:
            self.query_text = TextFusion(config["text_dim"], dim)
            self.candidate_text = TextFusion(config["text_dim"], dim)
            self.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))

    def train(self, mode=True):
        super().train(mode)
        # Frozen weights AND deterministic pretrained operations; allow input
        # gradients through them after trainable text/context blocks are added.
        self.backbone.eval()
        return self

    def rope(self):
        return self.backbone.rope_embed(H=self.grid[0], W=self.grid[1]) if self.backbone.rope_embed is not None else None

    @torch.no_grad()
    def prepare(self, images):
        tokens, grid = self.backbone.prepare_tokens_with_masks(images, None)
        if tuple(grid) != self.grid or tokens.shape[1] != self.prefix + math.prod(self.grid):
            raise ValueError("Unexpected DINOv3 grid/prefix count")
        return tokens.float()

    def _cls_map(self, block, tokens, rope):
        # Same qkv, key-bias mask and RoPE as the native DINOv3 attention.
        return io_utils.VisualTACS._cls_attention(block, tokens, rope)[:, self.prefix:].reshape(-1, *self.grid)

    def _norm_cls(self, tokens):
        norm = self.backbone.cls_norm if self.backbone.untie_cls_and_patch_norms else self.backbone.norm
        return norm(tokens[:, :self.prefix])[:, 0].float()

    @torch.no_grad()
    def encode_selector_tokens(self, initial, return_attention=False):
        tokens, rope, attention = initial, self.rope(), None
        for index, block in enumerate(self.backbone.blocks):
            if return_attention and index == len(self.backbone.blocks) - 1:
                attention = self._cls_map(block, tokens, rope)
            tokens = block(tokens, rope)
        return self._norm_cls(tokens), attention

    @torch.no_grad()
    def encode_selector(self, images, return_attention=False):
        return self.encode_selector_tokens(self.prepare(images), return_attention)

    def score(self, global_query, global_bank, allowed, q_text=None, q_valid=None, c_text=None, c_valid=None):
        scores = self.selector(global_query, global_bank.to(global_query.device))
        if self.multimodal:
            if any(value is None for value in (q_text, q_valid, c_text, c_valid)):
                raise ValueError("full_both requires both query and candidate text plus masks")
            semantic = F.normalize(q_text.float(), dim=1) @ F.normalize(c_text.float(), dim=1).T
            mask = q_valid.bool()[:, None] & c_valid.bool()[None, :]
            scores = scores + self.semantic_mix_logit.sigmoid() * semantic * mask
        if allowed.shape != scores.shape or not allowed.any(1).all():
            raise ValueError("No eligible candidate or invalid exclusion mask")
        return scores.masked_fill(~allowed.bool(), -torch.inf)

    def fuse(self, tokens, text, valid, candidate=False):
        if not self.multimodal:
            return tokens, tokens.new_zeros(len(tokens))
        layer = self.candidate_text if candidate else self.query_text
        patches, gate = layer(tokens[:, self.prefix:], text, valid)
        return torch.cat((tokens[:, :self.prefix], patches), 1), gate

    def decode(self, query_initial, context_initial=None, q_text=None, q_valid=None,
               c_text=None, c_valid=None, return_attention=False):
        if return_attention and self.training:
            raise ValueError("Attention is evaluation-only")
        query, q_gate = self.fuse(query_initial, q_text, q_valid)
        context, c_gate = (self.fuse(context_initial, c_text, c_valid, True) if context_initial is not None
                           else (None, query.new_zeros(len(query))))
        rope, stages, query_attention = self.rope(), [], None
        for index, (block, fusion) in enumerate(zip(self.backbone.blocks, self.context_blocks)):
            if return_attention and index == len(self.backbone.blocks) - 1:
                query_attention = self._cls_map(block, query, rope)
            # Bind modules in defaults: checkpoint's backward must NOT capture
            # the loop's final block via a late-bound closure.
            def step(q, c, block=block, fusion=fusion):
                q = block(q, rope)
                if c is None:
                    return q
                c = block(c, rope)
                patch = fusion(q[:, self.prefix:], c[:, self.prefix:])
                return torch.cat((q[:, :self.prefix], patch), 1), c
            if self.training and self.config.get("gradient_checkpointing", True) and torch.is_grad_enabled():
                result = checkpoint(step, query, context, use_reentrant=False, preserve_rng_state=True)
            else:
                result = step(query, context)
            if context is None:
                query = result
            else:
                query, context = result
            if index in self.stage_indices:
                stages.append(self.backbone.norm(query[:, self.prefix:]))
        logits = self.decoder(stages, self.grid, self.config["image_size"])
        details = {"q_gate": q_gate, "c_gate": c_gate}
        if return_attention:
            details["query_segmentor"] = query_attention
        return logits, details

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith("backbone.")}

    def restore_trainable(self, state):
        expected = {name for name in self.state_dict() if not name.startswith("backbone.")}
        if set(state) != expected:
            raise ValueError("Trainable checkpoint keys differ from the model architecture")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("backbone.") for key in result.missing_keys):
            raise ValueError("Incomplete trainable state load")


def segmentation_loss(logits, masks):
    bce = F.binary_cross_entropy_with_logits(logits.float(), masks.float(), reduction="none").mean((1, 2, 3))
    probabilities = logits.float().sigmoid()
    intersection = (probabilities * masks).sum((1, 2, 3))
    dice = (2 * intersection + 1e-6) / (probabilities.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1e-6)
    return bce + 1 - dice


def binary_metrics(logits, masks, threshold=.5):
    prediction, truth = logits.float().sigmoid() >= threshold, masks.bool()
    intersection = (prediction & truth).sum((1, 2, 3)).float()
    union = (prediction | truth).sum((1, 2, 3)).float()
    total = prediction.sum((1, 2, 3)).float() + truth.sum((1, 2, 3)).float()
    return (2 * intersection + 1e-6) / (total + 1e-6), (intersection + 1e-6) / (union + 1e-6)


def verify_bundle(root, *, allow_staged=False, allow_fixture=False):
    root = Path(root).resolve()
    manifest = io_utils.read_json(root / "manifest.json")
    statuses = {"ready"} | ({"staged"} if allow_staged else set()) | ({"fixture_unverified"} if allow_fixture else set())
    if manifest.get("status") == "fixture_unverified" and not allow_fixture:
        raise ValueError("Test fixture export cannot be used as a verified production model")
    if manifest.get("schema") != SCHEMA or manifest.get("status") not in statuses:
        raise ValueError("Wrong/incomplete ISIC deployment manifest")
    for name, digest in manifest["files"].items():
        if io_utils.file_sha256(io_utils.bundle_path(root, name)) != digest:
            raise ValueError(f"Bundle integrity failure: {name}")
    config = io_utils.read_json(root / "config.json")
    if manifest["mode"] != config["mode"] or config["context_top_k"] != 1:
        raise ValueError("Bundle mode/K does not match configuration")
    return manifest, config


class Predictor:
    def __init__(self, root, device="cpu", *, factory=None, caption_engine=None, _allow_staged=False):
        self.root = Path(root).resolve()
        self.manifest, self.config = verify_bundle(self.root, allow_staged=_allow_staged, allow_fixture=factory is not None)
        self.device = torch.device(device)
        repo = io_utils.bundle_path(self.root, "vendor/dinov3")
        weights = io_utils.bundle_path(self.root, self.manifest["backbone_file"])
        self.model = (factory(self.config) if factory else build_model(repo, self.config, weights)).to(self.device).eval()
        state = {}
        for name in ("models/selector.pt", "models/segmentor.pt"):
            piece = torch.load(self.root / name, map_location="cpu", weights_only=True)
            if set(state) & set(piece):
                raise ValueError("Duplicate state keys in model components")
            state.update(piece)
        self.model.restore_trainable(state)
        self.bank = torch.load(self.root / "candidate_bank.pt", map_location="cpu", weights_only=True)
        self.pool = io_utils.read_json(self.root / "candidate_pool.json")
        self.ground_truth_index = io_utils.read_json(self.root / "ground_truth_index.json")
        ids = [row["image_id"] for row in self.pool]
        if self.bank["image_ids"] != ids or len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("Candidate bank/manifest ordering differs")
        self.caption_engine = caption_engine
        self.text = None
        if self.config["mode"] == "mtacs":
            self.text = torch.load(self.root / "text_embeddings.pt", map_location="cpu", weights_only=True)
            self.text_position = {image_id: index for index, image_id in enumerate(self.text["image_ids"])}
            if any(image_id not in self.text_position for image_id in ids):
                raise ValueError("Missing candidate captions")
            positions = [self.text_position[image_id] for image_id in ids]
            self.c_text = self.text["embeddings"][positions].to(self.device)
            self.c_valid = self.text["valid"][positions].to(self.device)
            self.captions = io_utils.read_json(self.root / "caption_lookup.json")

    @torch.no_grad()
    def predict(self, source, *, return_attention=True):
        import hashlib
        original = io_utils.read_rgb(source)
        pixels = io_utils.preprocess(original, self.config).unsqueeze(0).to(self.device)
        initial = self.model.prepare(pixels)
        global_query, _ = self.model.encode_selector_tokens(initial)
        # Content matches may supply captions/ground truth for display, never a
        # filename-based guess. Inference itself accepts image pixels only.
        pixel_hash = hashlib.sha256(np.asarray(original).tobytes() + str(original.size).encode()).hexdigest()
        annotation = self.ground_truth_index.get(pixel_hash)
        group = annotation.get("group_id") if annotation else None
        allowed = torch.tensor([[row["pixel_sha256"] != pixel_hash and
                                not (group and group == row.get("group_id")) for row in self.pool]], device=self.device)
        q_text = q_valid = None
        query_caption = {"text": None, "valid": False, "status": "not_used"}
        if self.text is not None:
            by_pixels = self.manifest.get("caption_pixel_index", {}).get(pixel_hash)
            if by_pixels:
                index = self.text_position[by_pixels]
                q_text = self.text["embeddings"][index:index + 1].to(self.device)
                q_valid = self.text["valid"][index:index + 1].to(self.device)
                query_caption = {"text": self.captions[by_pixels]["final_caption"], "valid": bool(q_valid[0]),
                                 "status": "cached", "source": "verified_pixels"}
            else:
                if self.caption_engine is None:
                    raise RuntimeError("A new M-TACS image requires the bundled Florence/CLIP caption engine")
                row, vector, valid = self.caption_engine.caption(original)
                q_text, q_valid = vector.reshape(1, -1).to(self.device), valid.reshape(1).to(self.device)
                query_caption = {"text": row["final_caption"], "valid": bool(q_valid[0]),
                                 "status": "fresh", "source": "florence_prefix"}
        scores = self.model.score(global_query, self.bank["features"].to(self.device), allowed,
                                  q_text, q_valid, getattr(self, "c_text", None), getattr(self, "c_valid", None))
        position = int(scores.argmax(1))
        candidate = self.pool[position]
        context = self.bank["tokens"][position:position + 1].to(self.device, dtype=torch.float32)
        c_text = self.c_text[position:position + 1] if self.text is not None else None
        c_valid = self.c_valid[position:position + 1] if self.text is not None else None
        logits, details = self.model.decode(initial, context, q_text, q_valid, c_text, c_valid,
                                             return_attention=return_attention)
        probabilities = F.interpolate(logits.float().sigmoid(), size=(original.height, original.width),
                                      mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        mask = probabilities >= self.config["threshold"]
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        overlay = np.asarray(original, dtype=np.float32).copy()
        overlay[mask] = .6 * overlay[mask] + .4 * np.array([0, 220, 130])
        evidence = {"query_caption": query_caption, "candidate": {
            "image_id": candidate["image_id"], "position": position,
            "image_url": io_utils.png_data_url(io_utils.read_rgb(self.root / candidate["image"])),
            "caption": self.captions[candidate["image_id"]]["final_caption"] if self.text is not None else None,
            "caption_valid": bool(c_valid[0]) if c_valid is not None else False},
            "fusion": {"q_gate": float(details["q_gate"][0]), "c_gate": float(details["c_gate"][0])}}
        if return_attention:
            candidate_image = io_utils.read_rgb(self.root / candidate["image"])
            _, candidate_map = self.model.encode_selector(io_utils.preprocess(candidate_image, self.config)[None].to(self.device), True)
            attention = {}
            for name, image, values in [("query_segmentor", original, details["query_segmentor"][0]),
                                        ("candidate_selector", candidate_image, candidate_map[0])]:
                raw = values.float().cpu().numpy()
                heatmap, informative = io_utils.attention_overlay(image, raw)
                attention[name] = {"image_url": io_utils.png_data_url(heatmap), "grid": raw.tolist(),
                    "grid_shape": list(raw.shape), "method": ATTENTION_METHOD, "informative": informative}
            evidence["attention"] = attention
        ground_truth, comparison = {"status": "unavailable", "verified": False}, {"status": "unavailable"}
        if annotation:
            # Annotation is accessed only AFTER the prediction. It supplies no
            # input to the segmentor, utility head, captioner or candidate bank.
            with Image.open(io_utils.bundle_path(self.root, annotation["mask"])) as source_mask:
                truth = np.asarray(source_mask.convert("L")) > 0
            if truth.shape != mask.shape:
                raise ValueError("Verified image and exported ground-truth mask have different dimensions")
            ground_truth = {"status": "verified_pixels", "verified": True, "image_id": annotation["image_id"],
                "split": annotation["split"], "mask_url": io_utils.png_data_url(Image.fromarray(truth.astype(np.uint8) * 255))}
            overlap = float((truth & mask).sum())
            comparison = {"status": "available", "resolution": "original_image", "dice":
                (2 * overlap + 1e-6) / (float(truth.sum() + mask.sum()) + 1e-6), "iou":
                (overlap + 1e-6) / (float((truth | mask).sum()) + 1e-6)}
        return {"model": self.config["mode"], "task": "binary_segmentation", "logits": logits.cpu(),
            "probabilities": probabilities, "prediction": {"mask_url": io_utils.png_data_url(mask_image),
                "overlay_url": io_utils.png_data_url(Image.fromarray(overlay.clip(0, 255).astype(np.uint8))),
                "foreground_fraction": float(mask.mean()), "threshold": self.config["threshold"],
                "original_size": list(original.size)}, "evidence": evidence,
                "ground_truth": ground_truth, "comparison": comparison}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--captions", action="store_true")
    parser.add_argument("--verify-staged", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    captioner = None
    config = io_utils.read_json(Path(args.bundle) / "config.json")
    if args.captions and config["mode"] == "mtacs":
        from isic_caption_web import BundledCaptionEngine
        captioner = BundledCaptionEngine(args.bundle, args.device)
    result = Predictor(args.bundle, args.device, caption_engine=captioner, _allow_staged=args.verify_staged).predict(args.image)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.pop("logits"), output / "logits.pt")
    np.save(output / "probabilities.npy", result.pop("probabilities"))
    import base64
    for name, url in [("mask", result["prediction"]["mask_url"]), ("overlay", result["prediction"]["overlay_url"])]:
        (output / f"{name}.png").write_bytes(base64.b64decode(url.split(",", 1)[1]))
    io_utils.write_json(output / "prediction.json", result)


if __name__ == "__main__":
    main()
