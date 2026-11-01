# ISIC 2017 Florence-Prefix Full-Both Segmentation

Run [the standalone notebook](mtacs-isic2017-448-florence-prefix-full-both.ipynb) from top to bottom on Kaggle.
Only this `.ipynb` is needed. It generates/resumes **new ISIC captions.csv**, encodes CLIP text, trains segmentation,
evaluates test and exports comparisons. No Qwen model, precomputed DTD CSV or accompanying project `.py` is required.

## Why This Florence Method

The observed DTD split1/seed42 full_both classifier macro-F1 values were **0.810651 Prefix**, 0.809382 FirstValid,
0.790596 MultiGrounded and 0.779611 SingleGrounded. Prefix was the best of those tested Florence variants on DTD,
not a measured ISIC winner. This is an exploratory transfer motivated by those results, not a test-blind confirmation.
Sources: [caption-method evidence](../../../../../../reports/caption_ablation/dtd/caption_method_output_evidence.json).

The method uses frozen original `microsoft/Florence-2-base`, `<MORE_DETAILED_CAPTION>`, greedy decoding with one beam,
max48 new tokens, opening cleanup and at most eight retained words. It is not LoRA fine-tuning or multi-caption reranking.
On ISIC, the existing `isic_visible_attributes_v1` screening policy is retained: 5-8 words, clinical-term and token-limit
checks, zero vectors for rejected text. DTD's 47-class exclusions and the old CLIP 0.30 hard threshold are not applied.
The full generated response is saved and checked for clinical terms, including words beyond the retained prefix.

## Required Offline Inputs

- ISIC 2017 standard `train/images`, `train/masks`, `val/...`, `test/...`: **2,000/150/600** pairs.
- The same DINOv3 code and **ViT-S/16** weights as the visual/Qwen ISIC experiments, not DTD ViT-L/16 weights.
- The complete original legacy Florence-2-base folder. Default:
  `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`.
  Include `config.json`, safetensors weights, tokenizer/processor files and the three local Florence Python files.
  Native-converted, large, quantized, adapter-only or explicitly `-ft` checkpoints are rejected for this control.
- Full CLIP ViT-B/32 checkpoint and tokenizer, including text-projection weights.
- If the installed runtime is not compatible with the legacy checkpoint, attach `offline_wheels.zip`, `wheels.zip`
  or wheel directories containing `transformers==4.49.0`, `tokenizers>=0.21,<0.22`, `huggingface-hub>=0.26,<1`,
  and missing `einops`, `timm`, `accelerate`. Compiled wheels must match Kaggle's Python version/platform.

The existing offline runtime helper prepares **only a child-process target directory** using
`pip --no-index --only-binary=:all: --no-deps --target ...`. It never enables Internet or downgrades kernel packages.
Without compatible local wheels it stops with an actionable error; it does not download replacements.
CLIP encoding runs in a separate process using the parent environment. The parent must already support local CLIP.
`TOKENIZERS_PARALLELISM=false` is set before imports/workers.

Edit `ISIC_ROOT`, `FLORENCE_MODEL_REF`, `OFFLINE_ASSETS_ROOT`, `CLIP_MODEL_LOCAL_PATH`, `DINOV3_REPO_DIR` and
`DINOV3_WEIGHTS` if attached paths differ. No package installation is performed by the local notebook builder/tests.

## Same Segmentation Control

The model, split/pool creation, transforms, full_both fusion/selection, loss, training and evaluation functions are
reused from the Qwen notebook. Query and selected-candidate captions both enter fusion and semantic selection.
The captioner and its required generation runtime change; the caption content and natural validity masks may change.

Retained recipe: 448 resolution, frozen DINOv3 ViT-S/16, two final-token gated cross-attention blocks, DPT-lite,
BCE+Dice, fixed **400 training candidates**, **K=1**, batch8, seed42, max100 epochs, patience15. Exact self candidates
are excluded. Validation hard foreground Dice chooses the checkpoint; test is held out until training completes.
The notebook does not reuse visual/Qwen checkpoints, and does not claim exact reproduction of the paper's 256 protocol.

## Run and Resume

1. Attach inputs, enable GPU, edit paths if needed, then Run All.
2. Florence generates captions from RGB pixels and a constant task token, never masks/diagnoses/labels/filenames.
3. Export complete `caption_generation/captions.csv`; CLIP caches only valid caption embeddings.
4. Both workers exit before segmentation models initialize. Check `caption_coverage.csv` and the training gate logs.
5. Finish training/early stopping; test exports the best validation checkpoint's Dice/IoU and per-image predictions.

Optional smoke test: set `MAX_NEW_CAPTION_IMAGES=8`. Partial exports cannot train segmentation. Reset to 0 and rerun
to resume all 2,750 images. Invalid captions are not replaced with fabricated generic descriptions.

Default output root: `/kaggle/working/outputs/mtacs_isic2017_448_florence_prefix_full_both/`.
It is independent of both older runs. Keep the whole root to preserve atomic per-image caches, CSV, text cache,
asset/protocol hashes, split/pool manifests, best/last checkpoints, optimizer/scheduler/RNG state, history and test exports.
Changing assets, caption settings or the training protocol requires a new output root. Multiprocess augmentation
means resumed training is not guaranteed bitwise identical to uninterrupted training.

## Comparisons and Limits

The final `segmentation_comparison.csv` contains the new measured Florence result plus historical stdout references:

| Method | Test Dice | Test IoU | Source |
| --- | ---: | ---: | --- |
| Visual TACS variant 448 | 85.60% | 77.38% | Earlier displayed output, seed42 |
| Qwen3-VL-8B full_both 448 | 84.8689% | 76.5407% | Earlier displayed output, seed42 |
| Florence-Prefix full_both 448 | Not measured yet | Not measured yet | Run this notebook |

Both historical rows are rounded saved outputs, not newly rerun paired controls or multi-seed means.
The table exports percentage-point deltas against both references, without pre-filling Florence metrics.
For a report, compare caption coverage, gate/semantic-mix behavior and paired errors in addition to Dice/IoU;
more seeds are needed for an uncertainty estimate. This is not a clinical diagnostic system.

API reference: [Microsoft Florence-2-base model card](https://huggingface.co/microsoft/Florence-2-base).
