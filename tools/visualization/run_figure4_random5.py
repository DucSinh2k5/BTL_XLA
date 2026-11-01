#!/usr/bin/env python
"""Generate five random Figure-4-style visualizations.

This is a small runner around visualize_figure4.py. It loads the TACS model
and candidate feature bank once, samples five random query images from the
SUN397 test split, then saves one PNG per query.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import visualize_figure4 as vf
except ModuleNotFoundError:
    # Notebook fallback: if you ran the content of visualize_figure4.py in the
    # previous cell instead of saving it as a real .py file, its functions live
    # in __main__. Reuse them from there.
    import __main__ as vf

    required_attrs = [
        "parse_args",
        "make_eval_transform",
        "load_class_names",
        "load_sun397_annotations",
        "split_train_val_by_class",
        "load_models",
        "load_candidate_pool",
        "load_candidate_feature_bank",
        "visualize_sample",
    ]
    missing_attrs = [name for name in required_attrs if not hasattr(vf, name)]
    if missing_attrs:
        raise ModuleNotFoundError(
            "Cannot import visualize_figure4, and the current notebook session "
            "does not contain the required functions. First run the full "
            "visualize_figure4.py cell, or save it as /kaggle/working/visualize_figure4.py. "
            f"Missing: {missing_attrs}"
        )


# Edit these two values when you want a different random draw.
NUM_RANDOM_SAMPLES = 5
RANDOM_SEED = 2026

# Keep this separate from the main script output so experiments are easy to find.
OUTPUT_DIR = Path("outputs/figure4_random5")


def main() -> None:
    # In Colab/Jupyter, visualize_figure4.parse_args() ignores kernel arguments
    # and uses CONFIG defaults. In CLI, it still accepts normal command-line args.
    args = vf.parse_args()
    args.output_dir = str(OUTPUT_DIR)
    args.num_samples = 1

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    dataset_root = Path(args.dataset_root)
    image_root = Path(args.image_root) if str(args.image_root).strip() else dataset_root
    eval_transform = vf.make_eval_transform(int(args.image_size))

    class_names = vf.load_class_names(dataset_root)
    full_train_df, test_df = vf.load_sun397_annotations(
        dataset_root,
        image_root,
        class_names,
        int(args.seed),
    )
    if args.train_on_full_train:
        train_df = full_train_df.reset_index(drop=True)
    else:
        train_df, _ = vf.split_train_val_by_class(
            full_train_df,
            float(args.val_fraction),
            int(args.seed),
        )

    tacs_model, no_context_model, checkpoint = vf.load_models(args, class_names, device)
    ckpt_class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if isinstance(ckpt_class_names, list) and len(ckpt_class_names) == len(class_names):
        class_names = [str(x) for x in ckpt_class_names]

    candidate_df = vf.load_candidate_pool(args, train_df, image_root)
    candidate_features = vf.load_candidate_feature_bank(
        args,
        tacs_model,
        candidate_df,
        eval_transform,
        device,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    num_samples = min(NUM_RANDOM_SAMPLES, len(test_df))
    query_indices = rng.choice(len(test_df), size=num_samples, replace=False).tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Random query indices:", query_indices)
    print(f"Saving figures to: {OUTPUT_DIR.resolve()}")

    for rank, query_index in enumerate(query_indices, start=1):
        output_path = OUTPUT_DIR / f"random_{rank:02d}_query_{int(query_index):04d}.png"
        vf.visualize_sample(
            query_index=int(query_index),
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
