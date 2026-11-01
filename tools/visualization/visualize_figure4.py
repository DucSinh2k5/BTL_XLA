#!/usr/bin/env python
"""Create a Figure-4-style visualization for a trained TACS checkpoint.

Output panels:
    Query | TACS task attention | No-context attention | Selected context | Selector attention

REPRODUCTION ASSUMPTION:
The paper does not specify the exact attention extraction procedure used for
Figure 4. This script uses last-layer CLS-to-patch attention, averaged over
heads. If DINOv3 does not return attention weights, the script hooks the qkv
projection in the last transformer block and reconstructs QK attention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# CONFIG: edit these defaults, or override them from the command line.
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "dataset_root": "/kaggle/input/datasets/hiuphmnhtrung/sun397/SUN397/SUN397",
    "image_root": "",
    "dinov3_repo_dir": "/kaggle/input/datasets/nguyenducxu/dinov3-code/dinov3-main",
    "dinov3_model_name": "dinov3_vits16",
    "dinov3_weights": "/kaggle/input/datasets/nguyenducxu/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "tacs_checkpoint": "/kaggle/input/datasets/nguyenducxu/model-tacs/best_val_checkpoint (3).pt",
    "no_context_checkpoint": "",
    "selector_cache": "/kaggle/input/datasets/nguyenducxu/model-tacs/tacs_selector_cache.pt",
    "candidate_feature_bank": "/kaggle/input/datasets/nguyenducxu/model-tacs/trained_tacs_full_pool_candidate_feature_bank.pt",
    "candidate_pool_csv": "",
    "candidate_pool_indices": "",
    "candidate_pool_samples": "",
    "config_json": "",
    "output_dir": "outputs/figure4",
    "image_size": 256,
    "feature_dim": 384,
    "num_classes": 397,
    "seed": 42,
    "val_fraction": 0.10,
    "candidate_pool_fraction": 0.20,
    "train_on_full_train": False,
    "batch_size": 64,
}


# These globals mirror the notebook architecture. They are updated from the
# checkpoint config when possible before the model is constructed.
FEATURE_DIM = int(CONFIG["feature_dim"])
TOKEN_HEAD_DROPOUT = 0.30
QUERY_CONTEXT_GATE_DROPOUT = 0.20
QUERY_CONTEXT_GATE_BIAS = -0.70
TACS_SELECTOR_TEMPERATURE = 1.0
TACS_GUMBEL_TAU = 0.10
RETRIEVAL_SCORE_WEIGHT = 0.0
USE_BBOX_CROP = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def optional_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    value = str(value).strip()
    return Path(value) if value else None


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_state_prefix(state: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    out = dict(state)
    for prefix in prefixes:
        if out and all(key.startswith(prefix) for key in out.keys()):
            out = {key[len(prefix) :]: value for key, value in out.items()}
    return out


def get_state_dict(checkpoint: Any, preferred_keys: Sequence[str]) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(checkpoint, dict):
        for key in preferred_keys:
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return strip_state_prefix(value, ("module.", "model.", "network."))
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return strip_state_prefix(checkpoint, ("module.", "model.", "network."))
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_attr(obj: Any, dotted_name: str) -> Optional[Any]:
    cur = obj
    for part in dotted_name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SimpleImageDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, transform: transforms.Compose, crop_bbox: bool = False):
        self.annotations = annotations.reset_index(drop=True).copy()
        self.transform = transform
        self.crop_bbox = bool(crop_bbox)

    def __len__(self) -> int:
        return len(self.annotations)

    def _open_image(self, row: pd.Series) -> Image.Image:
        image = Image.open(row["image_path"]).convert("RGB")
        if self.crop_bbox and {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(row.index):
            left = max(0, int(row["bbox_x1"]) - 1)
            top = max(0, int(row["bbox_y1"]) - 1)
            right = min(image.width, max(left + 1, int(row["bbox_x2"])))
            bottom = min(image.height, max(top + 1, int(row["bbox_y2"])))
            image = image.crop((left, top, right, bottom))
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int]:
        row = self.annotations.iloc[index]
        image = self.transform(self._open_image(row))
        return image, int(index), str(row["fname"]), int(row["class_id"])


def load_class_names(dataset_root: Path) -> List[str]:
    candidate_paths = [
        dataset_root / "ClassName.txt",
        dataset_root.parent / "ClassName.txt",
        dataset_root / "classes.txt",
    ]
    for path in candidate_paths:
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                return [line.strip().lstrip("/") for line in f if line.strip()]
    raise FileNotFoundError(f"Cannot find SUN397 ClassName.txt under {dataset_root}")


def load_sun397_annotations(dataset_root: Path, image_root: Path, class_names: Sequence[str], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_to_id = {name.lstrip("/"): index for index, name in enumerate(class_names)}
    train_part_file: Optional[Path] = None
    test_part_file: Optional[Path] = None

    search_dirs = [
        dataset_root,
        dataset_root.parent / "sun397_other",
        dataset_root.parent / "sun397_other" / "Partitions",
        dataset_root.parent.parent / "sun397_other",
        Path("/kaggle/input/datasets/hiuphmnhtrung/sun397/SUN397/sun397_other"),
    ]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        t1 = directory / "Training_01.txt"
        te1 = directory / "Testing_01.txt"
        if t1.is_file() and te1.is_file():
            train_part_file, test_part_file = t1, te1
            break
        found = list(directory.glob("**/Training_01.txt"))
        if found:
            maybe_train = found[0]
            maybe_test = maybe_train.parent / "Testing_01.txt"
            if maybe_test.is_file():
                train_part_file, test_part_file = maybe_train, maybe_test
                break

    records: List[Dict[str, Any]] = []
    if train_part_file and test_part_file:
        for split_label, part_file in (("train", train_part_file), ("test", test_part_file)):
            with part_file.open("r", encoding="utf-8") as f:
                for line in f:
                    rel_path = line.strip().lstrip("/")
                    if not rel_path:
                        continue
                    class_folder = str(Path(rel_path).parent).replace("\\", "/").lstrip("/")
                    if class_folder not in class_to_id:
                        continue
                    image_path = image_root / rel_path
                    if not image_path.is_file():
                        image_path = dataset_root / rel_path
                    if not image_path.is_file():
                        continue
                    records.append(
                        {
                            "rel_path": rel_path,
                            "fname": Path(rel_path).name,
                            "image_path": str(image_path.resolve()),
                            "class_id": int(class_to_id[class_folder]),
                            "is_train": 1 if split_label == "train" else 0,
                        }
                    )
        df = pd.DataFrame(records)
    else:
        rng = np.random.default_rng(seed)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for class_name in tqdm(class_names, desc="scan SUN397 folders"):
            class_dir = image_root / class_name
            if not class_dir.is_dir():
                class_dir = dataset_root / class_name
            if not class_dir.is_dir():
                continue
            image_paths = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)
            if not image_paths:
                continue
            indices = np.arange(len(image_paths))
            rng.shuffle(indices)
            if len(image_paths) >= 100:
                train_indices = set(indices[:50])
                test_indices = set(indices[50:100])
            else:
                split = max(1, len(image_paths) // 2)
                train_indices = set(indices[:split])
                test_indices = set(indices[split:])
            for idx, path in enumerate(image_paths):
                if idx in train_indices:
                    is_train = 1
                elif idx in test_indices:
                    is_train = 0
                else:
                    continue
                records.append(
                    {
                        "rel_path": f"{class_name}/{path.name}",
                        "fname": path.name,
                        "image_path": str(path.resolve()),
                        "class_id": int(class_to_id[class_name]),
                        "is_train": is_train,
                    }
                )
        df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No SUN397 image records were loaded. Check dataset_root/image_root.")

    df["image_id"] = np.arange(len(df))
    df["bbox_x1"] = 0
    df["bbox_y1"] = 0
    df["bbox_x2"] = 99999
    df["bbox_y2"] = 99999

    train_df = df[df["is_train"] == 1].copy().reset_index(drop=True)
    train_df["split"] = "train"
    train_df["row_id"] = np.arange(len(train_df))
    test_df = df[df["is_train"] == 0].copy().reset_index(drop=True)
    test_df["split"] = "test"
    test_df["row_id"] = np.arange(len(test_df))
    return train_df, test_df


def split_train_val_by_class(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    for _, group in df.groupby("class_id", sort=True):
        group_indices = group.index.to_numpy()
        if len(group_indices) <= 1:
            continue
        n_val = max(1, int(round(len(group_indices) * val_fraction)))
        n_val = min(n_val, len(group_indices) - 1)
        val_indices.extend(rng.choice(group_indices, size=n_val, replace=False).tolist())
    val_indices_np = np.asarray(sorted(val_indices), dtype=int)
    return df.drop(index=val_indices_np).reset_index(drop=True), df.loc[val_indices_np].reset_index(drop=True)


def fix_image_paths(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    out = df.copy()
    fixed_paths: List[str] = []
    for _, row in out.iterrows():
        current = Path(str(row["image_path"]))
        if current.is_file():
            fixed_paths.append(str(current))
            continue
        class_name = str(row.get("class_name", "")).lstrip("/")
        rel_path = str(row.get("rel_path", "")).lstrip("/")
        fname = str(row["fname"])
        candidates = []
        if rel_path:
            candidates.append(image_root / rel_path)
        if class_name:
            candidates.append(image_root / class_name / fname)
        candidates.append(image_root / fname)
        found = next((p for p in candidates if p.is_file()), current)
        fixed_paths.append(str(found))
    out["image_path"] = fixed_paths
    return out


def load_candidate_pool(args: argparse.Namespace, train_df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    candidate_csv = optional_path(args.candidate_pool_csv)
    if candidate_csv and candidate_csv.is_file():
        candidate_df = pd.read_csv(candidate_csv)
        if "image_path" not in candidate_df.columns:
            raise ValueError(f"{candidate_csv} must contain image_path")
        return fix_image_paths(candidate_df, image_root).reset_index(drop=True)

    candidate_samples = optional_path(args.candidate_pool_samples)
    if candidate_samples and candidate_samples.is_file():
        data = load_json(candidate_samples)
        if isinstance(data, dict):
            data = data.get("samples") or data.get("candidate_pool") or data.get("items") or data
        if not isinstance(data, list):
            raise ValueError(f"Unsupported candidate samples JSON format: {candidate_samples}")
        candidate_df = pd.DataFrame(data)
        if "image_path" not in candidate_df.columns:
            raise ValueError(f"{candidate_samples} must contain image_path in each sample")
        return fix_image_paths(candidate_df, image_root).reset_index(drop=True)

    positions: Optional[np.ndarray] = None
    candidate_indices = optional_path(args.candidate_pool_indices)
    if candidate_indices and candidate_indices.is_file():
        positions = np.load(candidate_indices).astype(int)

    selector_cache = optional_path(args.selector_cache)
    if positions is None and selector_cache and selector_cache.is_file():
        cache = torch_load(selector_cache, map_location="cpu")
        if isinstance(cache, dict):
            for key in ("candidate_pool_positions", "candidate_pool_indices", "indices"):
                value = cache.get(key)
                if value is not None:
                    positions = torch.as_tensor(value).cpu().numpy().astype(int)
                    break

    if positions is None:
        rng = np.random.default_rng(int(args.seed))
        if float(args.candidate_pool_fraction) >= 1.0:
            positions = np.arange(len(train_df), dtype=int)
        else:
            size = max(1, int(round(len(train_df) * float(args.candidate_pool_fraction))))
            positions = np.sort(rng.choice(len(train_df), size=size, replace=False))
        print("WARNING: Rebuilt candidate pool from seed/fraction. Prefer passing the saved candidate_pool.csv or candidate_pool_indices.npy.")

    if positions.max(initial=-1) >= len(train_df):
        raise ValueError(
            f"Candidate index {int(positions.max())} exceeds train_df length {len(train_df)}. "
            "Use the exact candidate_pool.csv/samples from training."
        )
    return train_df.iloc[positions].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model definitions copied from the notebook architecture.
# ---------------------------------------------------------------------------


def load_dinov3_model(repo_dir: str, model_name: str, weights_path: str) -> nn.Module:
    repo_path = Path(repo_dir)
    weights = Path(weights_path)
    if not repo_path.is_dir():
        raise FileNotFoundError(f"DINOv3 repo dir not found: {repo_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    return torch.hub.load(str(repo_path), model_name, source="local", weights=str(weights))


def extract_cls_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = backbone(images)
    if isinstance(features, dict):
        if "x_norm_clstoken" in features:
            return features["x_norm_clstoken"]
        if "x_norm_cls_token" in features:
            return features["x_norm_cls_token"]
        if "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"].mean(dim=1)
        raise KeyError(f"Cannot find CLS token in DINO output keys: {list(features)}")
    if torch.is_tensor(features) and features.ndim == 3:
        return features[:, 0]
    return features


class DinoV3FrozenSimilarityContextClassifier(nn.Module):
    def __init__(self, repo_dir: str, model_name: str, weights_path: str, num_classes: int, feature_dim: int = 384):
        super().__init__()
        self.backbone = load_dinov3_model(repo_dir, model_name, weights_path)
        self.feature_dim = int(feature_dim)
        self.context_proj = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.context_attention = nn.Linear(feature_dim, 1)
        self.fusion_norm = nn.LayerNorm(feature_dim)
        self.token_head = nn.Sequential(nn.Dropout(p=TOKEN_HEAD_DROPOUT), nn.Linear(feature_dim, num_classes))
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(p=TOKEN_HEAD_DROPOUT),
            nn.Linear(feature_dim, num_classes),
        )

    def _prefix_token_count(self) -> int:
        if hasattr(self.backbone, "num_prefix_tokens"):
            return int(self.backbone.num_prefix_tokens)
        storage_tokens = getattr(self.backbone, "n_storage_tokens", None)
        if storage_tokens is None:
            storage_tokens = getattr(self.backbone, "num_register_tokens", 0)
        return 1 + int(storage_tokens or 0)

    def _patch_grid(self, images: torch.Tensor) -> Tuple[int, int]:
        patch_size = getattr(self.backbone, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size)
        return images.shape[-2] // patch_h, images.shape[-1] // patch_w

    def _unwrap_tokens(self, prepared: Any) -> torch.Tensor:
        if torch.is_tensor(prepared):
            return prepared
        if isinstance(prepared, (tuple, list)):
            for item in prepared:
                if torch.is_tensor(item):
                    return item
        if isinstance(prepared, dict):
            for key in ("x", "tokens", "last_hidden_state"):
                if key in prepared and torch.is_tensor(prepared[key]):
                    return prepared[key]
            for item in prepared.values():
                if torch.is_tensor(item):
                    return item
        raise TypeError(f"Cannot unwrap DINO token output of type {type(prepared)}")

    def _prepare_tokens_with_rope(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            prepared = self.backbone.prepare_tokens_with_masks(images, None)
        except TypeError:
            prepared = self.backbone.prepare_tokens_with_masks(images)
        tokens = self._unwrap_tokens(prepared)
        rope = None
        if hasattr(self.backbone, "rope_embed"):
            grid_h, grid_w = self._patch_grid(images)
            try:
                rope = self.backbone.rope_embed(H=grid_h, W=grid_w)
            except TypeError:
                rope = self.backbone.rope_embed(grid_h, grid_w)
        return tokens, rope

    def _repeat_rope_for_query_context(self, rope: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if rope is None or not torch.is_tensor(rope):
            return rope
        try:
            return rope.repeat_interleave(2, dim=-2)
        except RuntimeError:
            return rope

    def _run_blocks(self, tokens: torch.Tensor, rope: Optional[torch.Tensor]) -> torch.Tensor:
        for block in self.backbone.blocks:
            if rope is None:
                tokens = block(tokens)
                continue
            try:
                tokens = block(tokens, rope=rope)
            except TypeError:
                try:
                    tokens = block(tokens, rope)
                except TypeError:
                    tokens = block(tokens)
        if hasattr(self.backbone, "norm"):
            tokens = self.backbone.norm(tokens)
        return tokens

    def _forward_token_concat_features(self, query_images: torch.Tensor, context_images: torch.Tensor) -> torch.Tensor:
        query_tokens, query_rope = self._prepare_tokens_with_rope(query_images)
        context_tokens, _ = self._prepare_tokens_with_rope(context_images)
        prefix_tokens = self._prefix_token_count()
        pair_tokens = torch.cat(
            [query_tokens[:, :prefix_tokens], query_tokens[:, prefix_tokens:], context_tokens[:, prefix_tokens:]],
            dim=1,
        )
        pair_rope = self._repeat_rope_for_query_context(query_rope)
        pair_tokens = self._run_blocks(pair_tokens, pair_rope)
        return pair_tokens[:, 0]

    def _pool_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if "x_norm_cls_token" in features:
                return features["x_norm_cls_token"]
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"].mean(dim=1)
            for item in features.values():
                if torch.is_tensor(item):
                    return item[:, 0] if item.ndim == 3 else item
        if torch.is_tensor(features):
            return features[:, 0] if features.ndim == 3 else features
        raise TypeError(f"Cannot pool DINO features of type {type(features)}")

    def _forward_topk_token_attention(
        self,
        query_images: torch.Tensor,
        context_images: torch.Tensor,
        context_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context_images.ndim == 4:
            context_images = context_images.unsqueeze(1)
        batch_size, num_contexts, channels, height, width = context_images.shape
        query_repeated = query_images.unsqueeze(1).expand(-1, num_contexts, -1, -1, -1)
        query_repeated = query_repeated.reshape(batch_size * num_contexts, channels, height, width)
        context_flat = context_images.reshape(batch_size * num_contexts, channels, height, width)
        pair_features = self._forward_token_concat_features(query_repeated, context_flat)
        pair_features = pair_features.reshape(batch_size, num_contexts, self.feature_dim)
        query_summary = pair_features.mean(dim=1)
        attention_hidden = torch.tanh(self.context_proj(pair_features) + self.query_proj(query_summary).unsqueeze(1))
        attention_logits = self.context_attention(attention_hidden).squeeze(-1)
        if context_scores is not None:
            if context_scores.ndim == 1:
                context_scores = context_scores.unsqueeze(1)
            attention_logits = attention_logits + RETRIEVAL_SCORE_WEIGHT * context_scores.to(attention_logits)
        attention_weights = torch.softmax(attention_logits, dim=1)
        fused_features = torch.sum(pair_features * attention_weights.unsqueeze(-1), dim=1)
        return self.fusion_norm(fused_features), attention_weights

    def forward(
        self,
        query_images: torch.Tensor,
        context_images: torch.Tensor,
        context_scores: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Any:
        fused_features, attention_weights = self._forward_topk_token_attention(query_images, context_images, context_scores)
        logits = self.token_head(fused_features)
        if return_features:
            return logits, F.normalize(fused_features.float(), dim=1), attention_weights
        return logits


class DinoV3GatedTopKContextClassifier(DinoV3FrozenSimilarityContextClassifier):
    def __init__(self, repo_dir: str, model_name: str, weights_path: str, num_classes: int, feature_dim: int = 384):
        super().__init__(repo_dir, model_name, weights_path, num_classes, feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.final_norm = nn.LayerNorm(feature_dim)
        self.query_head = nn.Sequential(nn.Dropout(p=TOKEN_HEAD_DROPOUT), nn.Linear(feature_dim, num_classes))
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Dropout(p=QUERY_CONTEXT_GATE_DROPOUT),
            nn.Linear(feature_dim, 1),
        )
        nn.init.constant_(self.fusion_gate[-1].bias, QUERY_CONTEXT_GATE_BIAS)

    def forward(
        self,
        query_images: torch.Tensor,
        context_images: torch.Tensor,
        context_scores: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Any:
        context_features, attention_weights = self._forward_topk_token_attention(query_images, context_images, context_scores)
        query_features = self._pool_backbone_features(query_images)
        query_features = self.query_norm(query_features)
        context_features = self.context_norm(context_features)
        gate_input = torch.cat(
            [query_features, context_features, query_features * context_features, torch.abs(query_features - context_features)],
            dim=1,
        )
        gate = torch.sigmoid(self.fusion_gate(gate_input))
        query_logits = self.query_head(query_features)
        context_logits = self.token_head(context_features)
        logits = query_logits + gate * context_logits
        fused_features = self.final_norm(query_features + gate * context_features)
        if return_features:
            return logits, F.normalize(fused_features.float(), dim=1), attention_weights
        return logits


class TACSSelector(nn.Module):
    def __init__(self, temperature: float = 1.0, normalize: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def score(self, query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        query = query_features.float()
        candidates = candidate_features.float()
        if self.normalize:
            query = F.normalize(query, dim=-1)
            candidates = F.normalize(candidates, dim=-1)
        scores = query @ candidates.T
        return scores * self.logit_scale.clamp(0.01, 100.0) / max(self.temperature, 1e-6)


class DinoV3TACSClassifier(DinoV3GatedTopKContextClassifier):
    def __init__(self, repo_dir: str, model_name: str, weights_path: str, num_classes: int, feature_dim: int = 384):
        super().__init__(repo_dir, model_name, weights_path, num_classes, feature_dim)
        self.selector = TACSSelector(temperature=TACS_SELECTOR_TEMPERATURE, normalize=True)

    def selector_encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self._pool_backbone_features(images)

    def score_full_candidate_pool(
        self, query_images: torch.Tensor, candidate_feature_bank: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_features = self.selector_encode_images(query_images)
        selector_scores = self.selector.score(query_features, candidate_feature_bank.to(query_features.device))
        return selector_scores, query_features

    def select_full_pool(self, query_images: torch.Tensor, candidate_feature_bank: torch.Tensor, use_gumbel: bool = False) -> Dict[str, torch.Tensor]:
        selector_scores, query_features = self.score_full_candidate_pool(query_images, candidate_feature_bank)
        if self.training and use_gumbel:
            selection_weights = F.gumbel_softmax(selector_scores, tau=TACS_GUMBEL_TAU, hard=True, dim=1)
            selected_indices = selection_weights.argmax(dim=1)
        else:
            selected_indices = selector_scores.argmax(dim=1)
            selection_weights = F.one_hot(selected_indices, num_classes=selector_scores.size(1)).to(selector_scores.dtype)
        return {
            "selector_scores": selector_scores,
            "selection_weights": selection_weights,
            "selected_indices": selected_indices,
            "query_features": query_features,
            "selector_entropy": torch.distributions.Categorical(logits=selector_scores.float()).entropy().mean(),
        }

    def forward_selected_context(self, query_images: torch.Tensor, selected_context_images: torch.Tensor, return_features: bool = False) -> Any:
        return super().forward(query_images, selected_context_images.unsqueeze(1), context_scores=None, return_features=return_features)

    def forward_query_only(self, query_images: torch.Tensor) -> torch.Tensor:
        query_features = self._pool_backbone_features(query_images)
        query_features = self.query_norm(query_features)
        return self.query_head(query_features)


class DinoV3NoContextClassifier(nn.Module):
    def __init__(self, repo_dir: str, model_name: str, weights_path: str, num_classes: int, feature_dim: int = 384):
        super().__init__()
        self.backbone = load_dinov3_model(repo_dir, model_name, weights_path)
        self.head = nn.Linear(feature_dim, num_classes)

    def _pool_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images) if hasattr(self.backbone, "forward_features") else self.backbone(images)
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if "x_norm_cls_token" in features:
                return features["x_norm_cls_token"]
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"].mean(dim=1)
        if torch.is_tensor(features) and features.ndim == 3:
            return features[:, 0]
        return features

    def _prefix_token_count(self) -> int:
        if hasattr(self.backbone, "num_prefix_tokens"):
            return int(self.backbone.num_prefix_tokens)
        storage_tokens = getattr(self.backbone, "n_storage_tokens", None)
        if storage_tokens is None:
            storage_tokens = getattr(self.backbone, "num_register_tokens", 0)
        return 1 + int(storage_tokens or 0)

    def _patch_grid(self, images: torch.Tensor) -> Tuple[int, int]:
        patch_size = getattr(self.backbone, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size)
        return images.shape[-2] // patch_h, images.shape[-1] // patch_w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self._pool_backbone_features(images))


# ---------------------------------------------------------------------------
# Loading models and features
# ---------------------------------------------------------------------------


def apply_checkpoint_config(checkpoint: Any, args: argparse.Namespace) -> None:
    global FEATURE_DIM, TOKEN_HEAD_DROPOUT, QUERY_CONTEXT_GATE_DROPOUT
    global QUERY_CONTEXT_GATE_BIAS, TACS_SELECTOR_TEMPERATURE, USE_BBOX_CROP
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    FEATURE_DIM = int(config.get("feature_dim", args.feature_dim))
    TOKEN_HEAD_DROPOUT = float(config.get("token_head_dropout", TOKEN_HEAD_DROPOUT))
    QUERY_CONTEXT_GATE_DROPOUT = float(config.get("query_context_gate_dropout", QUERY_CONTEXT_GATE_DROPOUT))
    QUERY_CONTEXT_GATE_BIAS = float(config.get("query_context_gate_bias", QUERY_CONTEXT_GATE_BIAS))
    TACS_SELECTOR_TEMPERATURE = float(config.get("tacs_selector_temperature", TACS_SELECTOR_TEMPERATURE))
    USE_BBOX_CROP = bool(config.get("use_bbox_crop", USE_BBOX_CROP))


def load_tacs_model(args: argparse.Namespace, class_names: Sequence[str], device: torch.device) -> Tuple[DinoV3TACSClassifier, Dict[str, Any]]:
    checkpoint_path = optional_path(args.tacs_checkpoint)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"TACS checkpoint not found: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    apply_checkpoint_config(checkpoint, args)

    ckpt_class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    num_classes = len(ckpt_class_names) if isinstance(ckpt_class_names, list) else len(class_names)
    model = DinoV3TACSClassifier(
        repo_dir=args.dinov3_repo_dir,
        model_name=args.dinov3_model_name,
        weights_path=args.dinov3_weights,
        num_classes=num_classes,
        feature_dim=FEATURE_DIM,
    )

    full_state = get_state_dict(checkpoint, ("tacs_model_state_dict", "model_state_dict", "state_dict"))
    if full_state is not None:
        missing, unexpected = model.load_state_dict(full_state, strict=False)
    else:
        missing, unexpected = [], []
        task_state = get_state_dict(checkpoint, ("task_model_state_dict",))
        selector_state = get_state_dict(checkpoint, ("selector_state_dict",))
        if task_state is None and selector_state is None:
            raise KeyError(
                "Cannot find TACS state dict. Expected one of: "
                "tacs_model_state_dict, model_state_dict, state_dict, "
                "or task_model_state_dict + selector_state_dict."
            )
        if task_state is not None:
            missing, unexpected = model.load_state_dict(task_state, strict=False)
        if selector_state is not None:
            model.selector.load_state_dict(selector_state, strict=False)

    if missing:
        print(f"WARNING: TACS missing keys while loading: {list(missing)[:12]}")
    if unexpected:
        print(f"WARNING: TACS unexpected keys while loading: {list(unexpected)[:12]}")
    model.to(device).eval()
    return model, checkpoint


def load_no_context_model(args: argparse.Namespace, num_classes: int, device: torch.device) -> Optional[DinoV3NoContextClassifier]:
    checkpoint_path = optional_path(args.no_context_checkpoint)
    if checkpoint_path is None:
        return None
    if not checkpoint_path.is_file():
        warnings.warn(f"No-context checkpoint does not exist: {checkpoint_path}")
        return None
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    model = DinoV3NoContextClassifier(
        repo_dir=args.dinov3_repo_dir,
        model_name=args.dinov3_model_name,
        weights_path=args.dinov3_weights,
        num_classes=num_classes,
        feature_dim=FEATURE_DIM,
    )
    state = get_state_dict(checkpoint, ("no_context_model_state_dict", "model_state_dict", "state_dict"))
    if state is None:
        warnings.warn(f"Cannot find no-context state dict in {checkpoint_path}")
        return None
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: No-context missing keys while loading: {list(missing)[:12]}")
    if unexpected:
        print(f"WARNING: No-context unexpected keys while loading: {list(unexpected)[:12]}")
    model.to(device).eval()
    return model


def build_candidate_feature_bank(
    model: DinoV3TACSClassifier,
    candidate_df: pd.DataFrame,
    eval_transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    dataset = SimpleImageDataset(candidate_df, eval_transform, crop_bbox=USE_BBOX_CROP)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    features: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, *_ in tqdm(loader, desc="build candidate feature bank"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feats = model.selector_encode_images(images)
            features.append(feats.float().cpu())
    return torch.cat(features, dim=0)


def load_candidate_feature_bank(
    args: argparse.Namespace,
    model: DinoV3TACSClassifier,
    candidate_df: pd.DataFrame,
    eval_transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    bank_path = optional_path(args.candidate_feature_bank)
    if bank_path and bank_path.is_file():
        data = torch_load(bank_path, map_location="cpu")
        if torch.is_tensor(data):
            feature_bank = data
            bank_fnames = None
        elif isinstance(data, dict):
            feature_bank = data.get("candidate_feature_bank")
            if feature_bank is None:
                feature_bank = data.get("features")
            if feature_bank is None:
                feature_bank = data.get("feature_bank")
            bank_fnames = data.get("candidate_fnames") or data.get("fnames")
        else:
            raise ValueError(f"Unsupported candidate feature bank format: {bank_path}")
        if not torch.is_tensor(feature_bank):
            raise ValueError(f"Cannot find tensor candidate_feature_bank in {bank_path}")
        if len(feature_bank) == len(candidate_df):
            if bank_fnames is not None:
                expected = [str(x) for x in candidate_df["fname"].tolist()]
                actual = [str(x) for x in list(bank_fnames)]
                if expected[: min(20, len(expected))] != actual[: min(20, len(actual))]:
                    print("WARNING: candidate_fnames in feature bank do not match candidate_df order. Context images may be misaligned.")
            return feature_bank.float()
        print(
            f"WARNING: feature bank length {len(feature_bank)} != candidate pool length {len(candidate_df)}. "
            "Recomputing features from images."
        )
    return build_candidate_feature_bank(model, candidate_df, eval_transform, device, int(args.batch_size))


def load_models(
    args: argparse.Namespace,
    class_names: Sequence[str],
    device: torch.device,
) -> Tuple[DinoV3TACSClassifier, Optional[DinoV3NoContextClassifier], Dict[str, Any]]:
    tacs_model, checkpoint = load_tacs_model(args, class_names, device)
    no_context_model = load_no_context_model(args, len(class_names), device)
    return tacs_model, no_context_model, checkpoint


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------


class LastQKVAttentionCapture:
    def __init__(self, backbone: nn.Module):
        self.backbone = backbone
        self.attentions: List[torch.Tensor] = []
        self.handle: Optional[Any] = None
        self.num_heads: Optional[int] = None

    def __enter__(self) -> "LastQKVAttentionCapture":
        block = self._last_block()
        qkv_module, attn_module = self._find_qkv_module(block)
        self.num_heads = self._infer_num_heads(qkv_module, attn_module)

        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return
            attention = self._qkv_to_attention(out)
            self.attentions.append(attention.detach().float().cpu())

        self.handle = qkv_module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()

    def _last_block(self) -> nn.Module:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise AttributeError("Backbone has no transformer blocks attribute.")
        return blocks[-1]

    def _find_qkv_module(self, block: nn.Module) -> Tuple[nn.Module, Any]:
        for path in ("attn.qkv", "attn.qkv_linear", "attn.qkv_proj", "self_attn.qkv"):
            module = get_nested_attr(block, path)
            if isinstance(module, nn.Module):
                return module, get_nested_attr(block, path.rsplit(".", 1)[0])
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and module.out_features % 3 == 0:
                if "qkv" in name.lower() or module.out_features == 3 * module.in_features:
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    return module, get_nested_attr(block, parent_name) if parent_name else block
        raise AttributeError("Cannot locate a qkv Linear module in the last transformer block.")

    def _infer_num_heads(self, qkv_module: nn.Module, attn_module: Any) -> int:
        for attr_owner in (attn_module, qkv_module):
            for attr in ("num_heads", "n_heads", "num_attention_heads"):
                value = getattr(attr_owner, attr, None)
                if value is not None:
                    return int(value)
        embed_dim = int(getattr(qkv_module, "in_features", FEATURE_DIM))
        if embed_dim % 64 == 0:
            return max(1, embed_dim // 64)
        if embed_dim % 32 == 0:
            return max(1, embed_dim // 32)
        return 1

    def _qkv_to_attention(self, qkv_out: torch.Tensor) -> torch.Tensor:
        if qkv_out.ndim == 3:
            batch, tokens, three_dim = qkv_out.shape
            if three_dim % 3 != 0:
                raise ValueError(f"qkv output last dimension is not divisible by 3: {tuple(qkv_out.shape)}")
            embed_dim = three_dim // 3
            num_heads = int(self.num_heads or 1)
            if embed_dim % num_heads != 0:
                num_heads = 1
            head_dim = embed_dim // num_heads
            qkv = qkv_out.reshape(batch, tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
        elif qkv_out.ndim == 5:
            # Common layout: [3, B, H, N, D] or [B, N, 3, H, D]
            if qkv_out.shape[0] == 3:
                q, k = qkv_out[0], qkv_out[1]
            elif qkv_out.shape[2] == 3:
                qkv = qkv_out.permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
            else:
                raise ValueError(f"Unsupported 5D qkv layout: {tuple(qkv_out.shape)}")
        else:
            raise ValueError(f"Unsupported qkv output shape: {tuple(qkv_out.shape)}")
        attention = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return attention.softmax(dim=-1)


def get_cls_patch_attention(attention: torch.Tensor, patch_start: int, patch_count: int) -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError(f"Expected attention [B,H,N,N], got {tuple(attention.shape)}")
    if attention.shape[0] != 1:
        raise ValueError("Visualization expects batch size 1.")
    end = patch_start + patch_count
    if end > attention.shape[-1]:
        raise ValueError(f"Patch slice [{patch_start}:{end}] exceeds attention length {attention.shape[-1]}")
    cls_to_patch = attention[0, :, 0, patch_start:end].mean(dim=0)
    return cls_to_patch


def attention_to_heatmap(cls_patch_attention: torch.Tensor, grid_hw: Tuple[int, int], image_size: int) -> np.ndarray:
    grid_h, grid_w = grid_hw
    if cls_patch_attention.numel() != grid_h * grid_w:
        raise ValueError(f"Cannot reshape {cls_patch_attention.numel()} attention values into {grid_h}x{grid_w}")
    heat = cls_patch_attention.reshape(1, 1, grid_h, grid_w).float()
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-8)
    heat = F.interpolate(heat, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return heat[0, 0].cpu().numpy()


def overlay_attention(display_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    colored = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * display_image + alpha * colored, 0.0, 1.0)


@torch.no_grad()
def extract_task_attention(
    model: DinoV3TACSClassifier,
    query_tensor: torch.Tensor,
    context_tensor: torch.Tensor,
    image_size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    with LastQKVAttentionCapture(model.backbone) as capture:
        logits = model.forward_selected_context(query_tensor, context_tensor, return_features=False)
    if not capture.attentions:
        raise RuntimeError("No attention was captured from TACS task model.")
    # The gated TACS forward first runs pair tokens, then query-only features.
    pair_attention = capture.attentions[0]
    prefix_tokens = model._prefix_token_count()
    grid_hw = model._patch_grid(query_tensor)
    query_patch_count = grid_hw[0] * grid_hw[1]
    cls_patch = get_cls_patch_attention(pair_attention, prefix_tokens, query_patch_count)
    return attention_to_heatmap(cls_patch, grid_hw, image_size), logits


@torch.no_grad()
def extract_selector_attention(
    model: DinoV3TACSClassifier,
    selected_context_tensor: torch.Tensor,
    image_size: int,
) -> np.ndarray:
    with LastQKVAttentionCapture(model.backbone) as capture:
        _ = model.selector_encode_images(selected_context_tensor)
    if not capture.attentions:
        raise RuntimeError("No attention was captured from selector backbone.")
    attention = capture.attentions[-1]
    prefix_tokens = model._prefix_token_count()
    grid_hw = model._patch_grid(selected_context_tensor)
    cls_patch = get_cls_patch_attention(attention, prefix_tokens, grid_hw[0] * grid_hw[1])
    return attention_to_heatmap(cls_patch, grid_hw, image_size)


@torch.no_grad()
def extract_no_context_attention(
    model: DinoV3NoContextClassifier,
    query_tensor: torch.Tensor,
    image_size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    with LastQKVAttentionCapture(model.backbone) as capture:
        logits = model(query_tensor)
    if not capture.attentions:
        raise RuntimeError("No attention was captured from no-context model.")
    attention = capture.attentions[-1]
    prefix_tokens = model._prefix_token_count()
    grid_hw = model._patch_grid(query_tensor)
    cls_patch = get_cls_patch_attention(attention, prefix_tokens, grid_hw[0] * grid_hw[1])
    return attention_to_heatmap(cls_patch, grid_hw, image_size), logits


# ---------------------------------------------------------------------------
# Inference and visualization
# ---------------------------------------------------------------------------


def make_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image_for_model(path: Path, transform: transforms.Compose, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    pil = Image.open(path).convert("RGB")
    display = np.asarray(pil.resize((image_size, image_size), Image.BICUBIC)).astype(np.float32) / 255.0
    tensor = transform(pil).unsqueeze(0)
    return tensor, display


@torch.no_grad()
def select_context(
    model: DinoV3TACSClassifier,
    query_tensor: torch.Tensor,
    candidate_feature_bank: torch.Tensor,
    candidate_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, Any]:
    query_tensor = query_tensor.to(device)
    feature_bank = candidate_feature_bank.to(device)
    details = model.select_full_pool(query_tensor, feature_bank, use_gumbel=False)
    scores = details["selector_scores"].float()
    probs = torch.softmax(scores, dim=1)
    selected_pos = int(details["selected_indices"][0].item())
    row = candidate_df.iloc[selected_pos]
    return {
        "selected_pos": selected_pos,
        "row": row,
        "score": float(scores[0, selected_pos].detach().cpu()),
        "probability": float(probs[0, selected_pos].detach().cpu()),
        "scores": scores.detach().cpu(),
    }


def class_name(class_names: Sequence[str], class_id: int) -> str:
    if 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    return f"class_{class_id}"


def short_title(text: str, max_len: int = 44) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def visualize_sample(
    query_index: int,
    args: argparse.Namespace,
    class_names: Sequence[str],
    test_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    tacs_model: DinoV3TACSClassifier,
    no_context_model: Optional[DinoV3NoContextClassifier],
    candidate_feature_bank: torch.Tensor,
    eval_transform: transforms.Compose,
    device: torch.device,
    output_path: Path,
) -> None:
    if query_index < 0 or query_index >= len(test_df):
        raise IndexError(f"query_index {query_index} out of range [0, {len(test_df) - 1}]")

    query_row = test_df.iloc[int(query_index)]
    query_path = Path(str(query_row["image_path"]))
    query_tensor_cpu, query_display = load_image_for_model(query_path, eval_transform, int(args.image_size))
    query_tensor = query_tensor_cpu.to(device)

    context_selection = select_context(tacs_model, query_tensor, candidate_feature_bank, candidate_df, device)
    context_row = context_selection["row"]
    context_path = Path(str(context_row["image_path"]))
    context_tensor_cpu, context_display = load_image_for_model(context_path, eval_transform, int(args.image_size))
    context_tensor = context_tensor_cpu.to(device)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        tacs_heatmap, tacs_logits = extract_task_attention(tacs_model, query_tensor, context_tensor, int(args.image_size))
        selector_heatmap = extract_selector_attention(tacs_model, context_tensor, int(args.image_size))

    tacs_probs = torch.softmax(tacs_logits.float(), dim=1)[0].detach().cpu()
    tacs_pred = int(tacs_probs.argmax().item())
    tacs_conf = float(tacs_probs[tacs_pred].item())

    no_context_overlay: Optional[np.ndarray] = None
    no_context_pred: Optional[int] = None
    no_context_conf: Optional[float] = None
    no_context_error: Optional[str] = None
    if no_context_model is not None:
        try:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                no_context_heatmap, no_context_logits = extract_no_context_attention(
                    no_context_model, query_tensor, int(args.image_size)
                )
            no_context_overlay = overlay_attention(query_display, no_context_heatmap)
            no_context_probs = torch.softmax(no_context_logits.float(), dim=1)[0].detach().cpu()
            no_context_pred = int(no_context_probs.argmax().item())
            no_context_conf = float(no_context_probs[no_context_pred].item())
        except Exception as exc:
            no_context_error = f"{type(exc).__name__}: {exc}"

    tacs_overlay = overlay_attention(query_display, tacs_heatmap)
    selector_overlay = overlay_attention(context_display, selector_heatmap)

    true_id = int(query_row["class_id"])
    context_label = int(context_row["class_id"]) if "class_id" in context_row else -1
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(query_display)
    axes[0].set_title(f"Query\n{short_title(class_name(class_names, true_id))}", fontsize=10)

    axes[1].imshow(tacs_overlay)
    axes[1].set_title(
        "TACS Task Attention\n"
        f"pred={short_title(class_name(class_names, tacs_pred), 28)} p={tacs_conf:.2f}",
        fontsize=10,
    )

    if no_context_overlay is not None and no_context_pred is not None:
        axes[2].imshow(no_context_overlay)
        axes[2].set_title(
            "No-Context Attention\n"
            f"pred={short_title(class_name(class_names, no_context_pred), 28)} p={no_context_conf:.2f}",
            fontsize=10,
        )
    else:
        axes[2].imshow(np.ones_like(query_display))
        axes[2].text(
            0.5,
            0.5,
            "No-context checkpoint\nnot loaded",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[2].transAxes,
        )
        if no_context_error:
            axes[2].text(0.5, 0.18, short_title(no_context_error, 48), ha="center", va="center", fontsize=7, transform=axes[2].transAxes)
        axes[2].set_title("No-Context Attention", fontsize=10)

    axes[3].imshow(context_display)
    axes[3].set_title(
        "Selected Context\n"
        f"idx={context_selection['selected_pos']} p={context_selection['probability']:.3f}\n"
        f"{short_title(class_name(class_names, context_label), 34)}",
        fontsize=10,
    )

    axes[4].imshow(selector_overlay)
    axes[4].set_title("Selector Attention\non selected context", fontsize=10)

    fig.suptitle(
        f"Query index {query_index} | TACS score={context_selection['score']:.3f} | "
        f"same-class context={bool(context_label == true_id)}",
        fontsize=12,
    )
    plt.tight_layout()
    ensure_parent(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {output_path}")


def _argv_for_notebook_safe_argparse() -> Optional[List[str]]:
    """Ignore Jupyter/Colab kernel argv and use CONFIG defaults there."""
    executable_name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    running_kernel = (
        "ipykernel" in sys.modules
        or "google.colab" in sys.modules
        or executable_name in {"ipykernel_launcher.py", "colab_kernel_launcher.py"}
        or "-f" in sys.argv
    )
    return [] if running_kernel else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Figure-4-style TACS context selection and attention.")
    parser.add_argument("--dataset-root", default=CONFIG["dataset_root"])
    parser.add_argument("--image-root", default=CONFIG["image_root"])
    parser.add_argument("--dinov3-repo-dir", default=CONFIG["dinov3_repo_dir"])
    parser.add_argument("--dinov3-model-name", default=CONFIG["dinov3_model_name"])
    parser.add_argument("--dinov3-weights", default=CONFIG["dinov3_weights"])
    parser.add_argument("--tacs-checkpoint", default=CONFIG["tacs_checkpoint"])
    parser.add_argument("--no-context-checkpoint", default=CONFIG["no_context_checkpoint"])
    parser.add_argument("--selector-cache", default=CONFIG["selector_cache"])
    parser.add_argument("--candidate-feature-bank", default=CONFIG["candidate_feature_bank"])
    parser.add_argument("--candidate-pool-csv", default=CONFIG["candidate_pool_csv"])
    parser.add_argument("--candidate-pool-indices", default=CONFIG["candidate_pool_indices"])
    parser.add_argument("--candidate-pool-samples", default=CONFIG["candidate_pool_samples"])
    parser.add_argument("--config-json", default=CONFIG["config_json"])
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=int(CONFIG["image_size"]))
    parser.add_argument("--feature-dim", type=int, default=int(CONFIG["feature_dim"]))
    parser.add_argument("--num-classes", type=int, default=int(CONFIG["num_classes"]))
    parser.add_argument("--seed", type=int, default=int(CONFIG["seed"]))
    parser.add_argument("--val-fraction", type=float, default=float(CONFIG["val_fraction"]))
    parser.add_argument("--candidate-pool-fraction", type=float, default=float(CONFIG["candidate_pool_fraction"]))
    parser.add_argument("--train-on-full-train", action="store_true", default=bool(CONFIG["train_on_full_train"]))
    parser.add_argument("--batch-size", type=int, default=int(CONFIG["batch_size"]))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show", action="store_true")
    args, unknown_args = parser.parse_known_args(args=_argv_for_notebook_safe_argparse())
    if unknown_args:
        print(f"Ignored unknown Jupyter/Colab args: {unknown_args}")

    config_json = optional_path(args.config_json)
    if config_json and config_json.is_file():
        config = load_json(config_json)
        for key, value in config.items():
            arg_key = key.replace("-", "_")
            if hasattr(args, arg_key) and getattr(args, arg_key) in ("", None):
                setattr(args, arg_key, value)
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    dataset_root = Path(args.dataset_root)
    image_root = Path(args.image_root) if str(args.image_root).strip() else dataset_root
    eval_transform = make_eval_transform(int(args.image_size))

    class_names = load_class_names(dataset_root)
    full_train_df, test_df = load_sun397_annotations(dataset_root, image_root, class_names, int(args.seed))
    if args.train_on_full_train:
        train_df = full_train_df.reset_index(drop=True)
    else:
        train_df, _val_df = split_train_val_by_class(full_train_df, float(args.val_fraction), int(args.seed))

    tacs_model, no_context_model, checkpoint = load_models(args, class_names, device)
    ckpt_class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if isinstance(ckpt_class_names, list) and len(ckpt_class_names) == len(class_names):
        class_names = [str(x) for x in ckpt_class_names]

    candidate_df = load_candidate_pool(args, train_df, image_root)
    candidate_features = load_candidate_feature_bank(args, tacs_model, candidate_df, eval_transform, device)

    print(f"Device: {device}")
    print(f"Test images: {len(test_df):,}")
    print(f"Candidate pool: {len(candidate_df):,}")
    print(f"Candidate feature bank: {tuple(candidate_features.shape)}")
    print("Inference uses selector softmax + argmax. No Gumbel, no policy sampling, no reward.")

    output_dir = Path(args.output_dir)
    for offset in range(max(1, int(args.num_samples))):
        query_index = int(args.query_index) + offset
        if str(args.output).strip() and int(args.num_samples) == 1:
            output_path = Path(args.output)
        else:
            output_path = output_dir / f"sample_{query_index:04d}.png"
        visualize_sample(
            query_index=query_index,
            args=args,
            class_names=class_names,
            test_df=test_df,
            candidate_df=candidate_df,
            tacs_model=tacs_model,
            no_context_model=no_context_model,
            candidate_feature_bank=candidate_features,
            eval_transform=eval_transform,
            device=device,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
