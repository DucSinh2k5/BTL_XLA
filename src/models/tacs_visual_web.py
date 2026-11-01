"""Portable visual TACS model and inference contract for the SUN397 web bundle.

No captions, CLIP, label prior or external service is used. The same backbone
encodes selector images and the concatenated query/context patch sequence.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


SCHEMA = "sun397_tacs_visual_web_v1"
ATTENTION_METHOD = "last_block_cls_to_patch_attention_mean_heads"


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(handle)


def _hash_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bundle_path(root, relative):
    relative = str(relative)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "\\" in relative or ":" in relative or ".." in pure.parts:
        raise ValueError(f"Unsafe bundle-relative path: {relative}")
    root = Path(root).resolve()
    target = root.joinpath(*pure.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Bundle path escapes root: {relative}")
    return target


def read_rgb(source):
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess(image, config):
    """The exported evaluation transform, including EXIF and resize semantics."""
    image = read_rgb(image)
    size = int(config["image_size"])
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor(config["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(config["std"], dtype=torch.float32)[:, None, None]
    return (pixels - mean) / std


def concatenate_rope(rope):
    """Token order is [query patches, context patches], not interleaved."""
    if rope is None:
        return None
    if not isinstance(rope, (tuple, list)) or len(rope) != 2:
        raise TypeError("Expected DINOv3 RoPE (sin, cos); refusing a positional fallback")
    return tuple(torch.cat((part, part), dim=-2) for part in rope)


def mask_self(scores, excluded_positions=None):
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("TACS requires at least two candidates")
    if excluded_positions is not None:
        positions = torch.as_tensor(excluded_positions, dtype=torch.long, device=scores.device)
        if positions.shape != (scores.shape[0],) or (positions >= scores.shape[1]).any():
            raise ValueError("Invalid query-to-candidate mapping")
        rows = torch.arange(len(positions), device=scores.device)
        valid = positions >= 0
        scores = scores.clone()
        scores[rows[valid], positions[valid]] = -torch.inf
    if not torch.isfinite(scores).any(dim=1).all():
        raise ValueError("No eligible candidate for a query")
    return scores


class _StraightThroughImage(torch.autograd.Function):
    """Exact derivative of hard Gumbel weights @ fixed pixels, with CPU banking."""

    @staticmethod
    def forward(ctx, weights, images, chunk_size):
        if images.ndim != 4 or weights.shape[1] != len(images):
            raise ValueError("Image bank/selection shape mismatch")
        hard = F.one_hot(weights.argmax(1), len(images)).to(weights.dtype)
        if not torch.equal(weights.detach(), hard):
            raise ValueError("Straight-through image forward requires hard one-hot values")
        ctx.save_for_backward(images)
        ctx.chunk_size = int(chunk_size)
        ctx.weight_device, ctx.weight_dtype = weights.device, weights.dtype
        indices = weights.argmax(1).to(images.device)
        return images.index_select(0, indices).to(weights.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_images):
        (bank,) = ctx.saved_tensors
        grad = grad_images.float().flatten(1)
        chunks = []
        # Float32 avoids an fp16 bank matmul silently changing selector gradients.
        with torch.autocast(device_type=grad.device.type, enabled=False):
            for start in range(0, len(bank), ctx.chunk_size):
                block = bank[start:start + ctx.chunk_size].to(grad.device, dtype=torch.float32)
                chunks.append(grad @ block.flatten(1).T)
        return torch.cat(chunks, dim=1).to(ctx.weight_device, ctx.weight_dtype), None, None


def straight_through_images(weights, bank, chunk_size=32):
    return _StraightThroughImage.apply(weights, bank, chunk_size)


class VisualTACS(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = dict(config)
        dim = int(getattr(backbone, "embed_dim", config["feature_dim"]))
        if dim != int(config["feature_dim"]):
            raise ValueError("Backbone feature dimension does not match run configuration")
        self.head = nn.Sequential(nn.Dropout(float(config.get("head_dropout", 0.1))), nn.Linear(dim, int(config["num_classes"])))
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", True))

    @property
    def prefix_count(self):
        return 1 + int(getattr(self.backbone, "n_storage_tokens", 0))

    def _tokens(self, images):
        prepared = self.backbone.prepare_tokens_with_masks(images, None)
        if not isinstance(prepared, tuple) or len(prepared) != 2:
            raise TypeError("This runtime requires DINOv3 prepare_tokens_with_masks -> (tokens, grid)")
        tokens, grid = prepared
        height, width = map(int, grid)
        if tokens.shape[1] != self.prefix_count + height * width:
            raise ValueError("Unexpected prefix/patch token count")
        rope = self.backbone.rope_embed(H=height, W=width) if self.backbone.rope_embed is not None and not self.training else None
        return tokens, rope, (height, width)

    @staticmethod
    def _cls_attention(block, tokens, rope):
        attention = block.attn
        qkv = attention.qkv(block.norm1(tokens))
        batch, count, triple_dim = qkv.shape
        dim = triple_dim // 3
        heads = int(attention.num_heads)
        q, k, _ = qkv.reshape(batch, count, 3, heads, dim // heads).unbind(2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        if rope is not None:
            q, k = attention.apply_rope(q, k, rope)
        weights = ((q[:, :, :1].float() @ k.float().transpose(-2, -1)) * (dim // heads) ** -0.5).softmax(dim=-1)
        return weights.mean(dim=1)[:, 0]

    def _encode(self, tokens, rope, return_attention=False, grid=None, paired=False):
        if return_attention and self.training:
            raise RuntimeError("Attention export is evaluation-only")
        cls_attention = None
        for index, block in enumerate(self.backbone.blocks):
            block_rope = rope
            if self.training and self.backbone.rope_embed is not None:
                # Native DINOv3 resamples positional augmentation per block.
                # Pass that sample into checkpoint so backward reuses it exactly.
                block_rope = self.backbone.rope_embed(H=grid[0], W=grid[1])
                if paired:
                    block_rope = concatenate_rope(block_rope)
            if return_attention and index == len(self.backbone.blocks) - 1:
                cls_attention = self._cls_attention(block, tokens, block_rope)
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, block_rope, use_reentrant=False, preserve_rng_state=True)
            else:
                tokens = block(tokens, block_rope)
        if getattr(self.backbone, "untie_cls_and_patch_norms", False):
            cls = self.backbone.cls_norm(tokens[:, :self.prefix_count])[:, 0]
        else:
            cls = self.backbone.norm(tokens)[:, 0]
        return cls, cls_attention

    def encode_selector(self, images, return_attention=False):
        tokens, rope, grid = self._tokens(images)
        features, attention = self._encode(tokens, rope, return_attention, grid=grid)
        if return_attention:
            return features, attention[:, self.prefix_count:].reshape(-1, *grid)
        return features

    def score(self, query_images, candidate_features, excluded_positions=None):
        query = self.encode_selector(query_images).float()
        bank = candidate_features.to(query.device, dtype=torch.float32)
        # Eq. 1 utility: shared CLS dot product. No class labels enter scoring.
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = query @ bank.T
        return mask_self(scores, excluded_positions), query

    def forward(self, query_images, context_images=None, return_attention=False):
        tokens, rope, grid = self._tokens(query_images)
        patch_count = grid[0] * grid[1]
        if context_images is not None:
            context_tokens, _, context_grid = self._tokens(context_images)
            if grid != context_grid:
                raise ValueError("Query and context must have the same patch grid")
            tokens = torch.cat((tokens, context_tokens[:, self.prefix_count:]), dim=1)
            rope = concatenate_rope(rope)
        features, attention = self._encode(tokens, rope, return_attention, grid=grid, paired=context_images is not None)
        logits = self.head(features)
        if not return_attention:
            return logits
        result = {"query_classifier": attention[:, self.prefix_count:self.prefix_count + patch_count].reshape(-1, *grid)}
        if context_images is not None:
            result["context_classifier"] = attention[:, self.prefix_count + patch_count:].reshape(-1, *grid)
        return logits, result


def build_model(repo_dir, config, weights_path=None):
    if config["backbone"] != "dinov3_vitl16" or int(config["feature_dim"]) != 1024:
        raise ValueError("Production bundle requires DINOv3 ViT-L/16, 1024 dimensions; no S/B substitution")
    repo = Path(repo_dir).resolve()
    if not (repo / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"DINOv3 source/dinov3/hub/backbones.py missing: {repo}")
    kwargs = {"pretrained": weights_path is not None}
    if weights_path is not None:
        weights = Path(weights_path).resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if not weights.name.lower().endswith("lvd1689m-8aa4cbdd.pth"):
            raise ValueError("Use the official ViT-L/16 LVD1689M 8aa4cbdd .pth with its original filename; SAT/plus formats are not supported here")
        kwargs["weights"] = str(weights)
    # hubconf also imports unrelated detector/segmentor dependencies. Use the
    # exact official backbone factory without executing those entry points.
    package = sys.modules.get("dinov3")
    if package is not None and not Path(package.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Another DINOv3 source is already imported. Restart the kernel before loading this bundle.")
    sys.path.insert(0, str(repo))
    backbones = importlib.import_module("dinov3.hub.backbones")
    if not Path(backbones.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("DINOv3 backbone code did not load from the requested local repository")
    backbone = backbones.dinov3_vitl16(**kwargs)
    if len(backbone.blocks) != 24:
        raise ValueError("Expected 24 transformer blocks for ViT-L/16")
    return VisualTACS(backbone, config)


def normalize_map(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all() or raw.min() < 0:
        raise ValueError("Invalid attention map")
    span = float(raw.max() - raw.min())
    return (raw - raw.min()) / max(span, 1e-12), span > 1e-12


def attention_overlay(image, raw, alpha=0.45):
    image = read_rgb(image)
    normalized, informative = normalize_map(raw)
    heat = Image.fromarray(np.uint8(np.round(normalized * 255))).resize(image.size, Image.Resampling.BILINEAR)
    values = np.asarray(heat, dtype=np.float32) / 255.0
    colors = np.stack((values, 0.25 * (1 - values), 1 - values), axis=-1)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mixed = (1 - alpha) * pixels + alpha * colors
    # Flat attention is preserved and flagged, never replaced by invented structure.
    return Image.fromarray(np.uint8(np.clip(mixed * 255, 0, 255))), informative


def png_data_url(image):
    import base64
    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "ready":
        raise ValueError("Bundle is incomplete or uses another schema")
    files = manifest.get("files", {})
    required = {"model.pt", "config.json", "class_names.json", "candidate_pool.json", "candidate_features.pt",
                "runtime/tacs_visual_web.py", "vendor/dinov3/dinov3/hub/backbones.py"}
    if not required.issubset(files):
        raise ValueError("Manifest omits a required inference artifact")
    for name, info in files.items():
        path = bundle_path(root, name)
        if not path.is_file() or path.stat().st_size != int(info["bytes"]) or file_sha256(path) != info["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class BundlePredictor:
    """Load a trusted, exported bundle. No training Inputs or network are needed."""
    def __init__(self, root, device="cpu", model_factory=build_model):
        self.root = Path(root).resolve()
        self.manifest = verify_bundle(self.root)
        self.config = read_json(self.root / "config.json")
        self.class_names = read_json(self.root / "class_names.json")
        self.candidates = read_json(self.root / "candidate_pool.json")
        self.device = torch.device(device)
        if len(self.class_names) != self.config["num_classes"] or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Class order/count is invalid")
        if len(self.candidates) != self.config["candidate_count"]:
            raise ValueError("Candidate count mismatch")
        ids = [row["image_id"] for row in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate candidate IDs")
        for position, row in enumerate(self.candidates):
            if row["position"] != position or row["file"] not in self.manifest["files"]:
                raise ValueError("Candidate bank order or image manifest is invalid")
            if row["sha256"] != self.manifest["files"][row["file"]]["sha256"]:
                raise ValueError("Candidate content hash does not match its image")
            if self.class_names[row["class_id"]] != row["class_name"]:
                raise ValueError("Candidate class mapping mismatch")
            bundle_path(self.root, row["file"])
        checkpoint_state = torch.load(self.root / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint_state["config"] != self.config or checkpoint_state["class_names"] != self.class_names:
            raise ValueError("Model/config/class mapping mismatch")
        bank = torch.load(self.root / "candidate_features.pt", map_location="cpu", weights_only=True)
        if bank["image_ids"] != ids or bank["checkpoint_sha256"] != file_sha256(self.root / "model.pt"):
            raise ValueError("Candidate features belong to another checkpoint/order")
        if tuple(bank["features"].shape) != (len(ids), self.config["feature_dim"]) or not torch.isfinite(bank["features"]).all():
            raise ValueError("Candidate features are invalid")
        self.model = model_factory(self.root / "vendor/dinov3", self.config).to(self.device)
        self.model.load_state_dict(checkpoint_state["model_state_dict"], strict=True)
        self.model.eval()
        self.bank = bank["features"].float().to(self.device)
        self.last_logits = None

    def predict(self, source, top_k=5, explain=True, image_id=None):
        image = read_rgb(source)
        query = preprocess(image, self.config).unsqueeze(0).to(self.device)
        excluded = torch.zeros(len(self.candidates), dtype=torch.bool, device=self.device)
        if isinstance(source, (str, Path)):
            content_hash = file_sha256(source)
        elif isinstance(source, bytes):
            content_hash = hashlib.sha256(source).hexdigest()
        else:
            content_hash = None
        for i, row in enumerate(self.candidates):
            if row["image_id"] == image_id or (content_hash is not None and row["sha256"] == content_hash):
                excluded[i] = True
        with torch.no_grad():
            scores, _ = self.model.score(query, self.bank)
            scores = scores.masked_fill(excluded.unsqueeze(0), -torch.inf)
            if not torch.isfinite(scores).any():
                raise ValueError("No eligible candidate after self-exclusion")
            position = int(scores.argmax(1).item())
            row = self.candidates[position]
            candidate = read_rgb(bundle_path(self.root, row["file"]))
            context = preprocess(candidate, self.config).unsqueeze(0).to(self.device)
            if explain:
                logits, maps = self.model(query, context, return_attention=True)
                _, selector_map = self.model.encode_selector(context, return_attention=True)
            else:
                logits = self.model(query, context)
            self.last_logits = logits.detach().cpu()
            probabilities = logits.float().softmax(1)[0]
            values, indices = probabilities.topk(min(max(1, int(top_k)), len(self.class_names)))
        result = {
            "model": "tacs", "dataset": self.config["dataset"], "bundle_schema": SCHEMA,
            "prediction": {"class_id": int(indices[0]), "label": self.class_names[int(indices[0])], "confidence": float(values[0])},
            "top_k": [{"class_id": int(i), "label": self.class_names[int(i)], "confidence": float(v)} for v, i in zip(values, indices)],
            "evidence": {
                "query_caption": {"text": None, "status": "not_used"},
                "candidate": {"image_id": row["image_id"], "position": position, "image_url": png_data_url(candidate), "caption": None,
                              "selector_score": float(scores[0, position]), "status": "ready"},
                "attention": {},
            },
        }
        if explain:
            for name, raw, source_image in [("query_classifier", maps["query_classifier"][0], image),
                                            ("candidate_selector", selector_map[0], candidate)]:
                array = raw.float().cpu().numpy()
                overlay, informative = attention_overlay(source_image, array)
                result["evidence"]["attention"][name] = {
                    "image_url": png_data_url(overlay), "status": "ready" if informative else "flat_attention",
                    "method": ATTENTION_METHOD, "raw_grid": array.tolist(), "grid_shape": list(array.shape),
                    "normalization": "per_image_min_max_for_display_only", "class_conditional": False,
                }
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline visual TACS SUN397 bundle inference")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    parser.add_argument("--no-attention", action="store_true")
    args = parser.parse_args()
    predictor = BundlePredictor(args.bundle, args.device)
    write_json(args.output, predictor.predict(args.image, explain=not args.no_attention))
    print(f"Saved prediction and evidence to {args.output}")


if __name__ == "__main__":
    main()
