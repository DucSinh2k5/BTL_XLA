import json
import random
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', DEVICE)

from pathlib import Path

# --- ĐƯỜNG DẪN OUTPUT (Ghi vào working của Kaggle) ---
PROJECT_DIR = Path('/kaggle/working')
OUT_PUT_DIR = PROJECT_DIR / 'outputs' / 'baseline_frozen_dino_similarity_ret_cars'
OUTPUT_DIR = OUT_PUT_DIR
OUT_PUT_DIR.mkdir(parents=True, exist_ok=True)

# --- ĐƯỜNG DẪN DATASET (Stanford Cars) ---
DATASET_ROOT = Path(
    '/kaggle/input/datasets/jutrera/stanford-car-dataset-by-classes-folder'
)
TRAIN_IMAGE_ROOT = DATASET_ROOT / 'car_data' / 'car_data' / 'train'
TEST_IMAGE_ROOT = DATASET_ROOT / 'car_data' / 'car_data' / 'test'

# --- ĐƯỜNG DẪN DINOv3 REPO & WEIGHTS (Chỉ trỏ tới thư mục chứa hubconf.py) ---
DINOV3_REPO_DIR = (
    '/kaggle/input/datasets/sinhnguync/dinov3-code/dinov3-main'  # thư mục chứa hubconf.py
)
DINOV3_MODEL_NAME = 'dinov3_vits16'
DINOV3_WEIGHTS = '/kaggle/input/datasets/sinhnguync/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'

BACKBONE_NAME = DINOV3_MODEL_NAME
FEATURE_DIM = 384

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  # legacy default kept for older cells; optimizer below uses param groups
BACKBONE_LR = 3e-6
HEAD_LR = 1e-4
BACKBONE_WEIGHT_DECAY = 0.05
HEAD_WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
TOKEN_HEAD_DROPOUT = 0.3
FREEZE_BACKBONE_EPOCHS = 5
WARMUP_EPOCHS = 5
PATIENCE = 8
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
NUM_CLASSES = 196
TOP_K = 5
CONTEXT_TOP_K = 4
RETRIEVAL_SCORE_WEIGHT = 0.25
SUPCON_WEIGHT = 0.05
SUPCON_TEMPERATURE = 0.10
USE_BBOX_CROP = True
CANDIDATE_POOL_FRACTION = 1.00
VAL_FRACTION = 0.10
RETRIEVAL_BATCH_SIZE = 128
RETRIEVAL_CHUNK_SIZE = 1024

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.70, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


def load_class_names(dataset_root):
    class_names = pd.read_csv(dataset_root / 'names.csv', header=None)[0].astype(str).tolist()
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f'Expected {NUM_CLASSES} classes, got {len(class_names)}')
    return class_names


def load_annotations(dataset_root, split):
    df = pd.read_csv(
        dataset_root / f'anno_{split}.csv',
        header=None,
        names=['fname', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id_1based'],
    )
    df['class_id'] = df['class_id_1based'].astype(int) - 1
    df['split'] = split
    df['row_id'] = np.arange(len(df))
    return df


def attach_image_paths(df, image_root):
    image_index = {path.name: path for path in sorted(Path(image_root).rglob('*.jpg'))}
    missing = [fname for fname in df['fname'] if fname not in image_index]
    if missing:
        raise FileNotFoundError(f'Thieu anh trong {image_root}, vi du: {missing[:5]}')
    out = df.copy()
    out['image_path'] = [str(image_index[fname]) for fname in out['fname']]
    return out


def split_train_val_by_class(df, val_fraction=0.10, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction phải nằm trong khoảng (0, 1)')
    rng = np.random.default_rng(seed)
    val_indices = []
    for _, group in df.groupby('class_id', sort=True):
        group_indices = group.index.to_numpy()
        if len(group_indices) <= 1:
            continue
        n_val = max(1, int(round(len(group_indices) * val_fraction)))
        n_val = min(n_val, len(group_indices) - 1)
        val_indices.extend(rng.choice(group_indices, size=n_val, replace=False).tolist())
    val_indices = np.array(sorted(val_indices), dtype=int)
    train_split = df.drop(index=val_indices).reset_index(drop=True)
    val_split = df.loc[val_indices].reset_index(drop=True)
    return train_split, val_split


class_names = load_class_names(DATASET_ROOT)
full_train_df = attach_image_paths(load_annotations(DATASET_ROOT, 'train'), TRAIN_IMAGE_ROOT).reset_index(drop=True)
test_df = attach_image_paths(load_annotations(DATASET_ROOT, 'test'), TEST_IMAGE_ROOT).reset_index(drop=True)
train_df, val_df = split_train_val_by_class(full_train_df, VAL_FRACTION, SEED)
NUM_CLASSES = len(class_names)

candidate_rng = np.random.default_rng(SEED)
if CANDIDATE_POOL_FRACTION >= 1.0:
    candidate_pool_positions = np.arange(len(train_df), dtype=int)
else:
    candidate_pool_size = max(1, int(round(len(train_df) * CANDIDATE_POOL_FRACTION)))
    candidate_pool_positions = np.sort(candidate_rng.choice(len(train_df), size=candidate_pool_size, replace=False))
candidate_df = train_df.iloc[candidate_pool_positions].reset_index(drop=True)
candidate_df.to_csv(OUT_PUT_DIR / 'candidate_pool.csv', index=False)
val_df.to_csv(OUT_PUT_DIR / 'validation_split.csv', index=False)

print('DATASET_ROOT:', DATASET_ROOT.resolve())
print('TRAIN_IMAGE_ROOT:', TRAIN_IMAGE_ROOT.resolve())
print('TEST_IMAGE_ROOT:', TEST_IMAGE_ROOT.resolve())
print('OUT_PUT_DIR:', OUT_PUT_DIR.resolve())
print(f'Standard train images: {len(full_train_df):,}')
print(f'Actual train images: {len(train_df):,} ({1 - VAL_FRACTION:.0%})')
print(f'Validation images: {len(val_df):,} ({VAL_FRACTION:.0%})')
print(f'Standard test images: {len(test_df):,}')
print(f'Candidate pool: {len(candidate_df):,}/{len(train_df):,} actual-train images ({CANDIDATE_POOL_FRACTION:.0%})')
print(f'Context top-k per query: {CONTEXT_TOP_K}')
print(f'BBox crop enabled: {USE_BBOX_CROP}')
print(f'Train label range: {int(train_df.class_id.min())}..{int(train_df.class_id.max())}')
print(f'Test label range: {int(test_df.class_id.min())}..{int(test_df.class_id.max())}')
display(train_df.head())
display(val_df.head())
display(candidate_df.head())
