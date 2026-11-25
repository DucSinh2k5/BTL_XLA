"""Stage new CSV + aligned CLIP vectors without editing the original ISIC web bundle."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "apps/sun397_demo"), str(ROOT / "tools/notebook_builders")]
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")

if __name__ == "__main__":
    import torch
    from build_isic_caption_firstvalid_v3_notebook import runtime_source
    from isic_caption_override import prepare_override
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = prepare_override(args.bundle, args.csv, args.output, runtime_source())
    print(json.dumps(result["report"], indent=2))
