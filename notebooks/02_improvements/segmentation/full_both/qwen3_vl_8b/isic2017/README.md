# ISIC 2017 Qwen3-VL-8B Full-Both Segmentation

Run [the standalone notebook](mtacs-isic2017-448-qwen3-vl-8b-full-both.ipynb) on Kaggle from top to bottom.
Only the `.ipynb` is needed; all worker/model/training code is embedded. No project `.py` imports or Internet access
are required at runtime. No captions.csv must be prepared beforehand.

## Inputs

- ISIC 2017 original segmentation dataset, with `train/images`, `train/masks`, `val/...`, `test/...`.
- The same DINOv3 repo and **ViT-S/16** weights as the visual segmentation baseline, not the DTD ViT-L/16 weights.
- The complete `Qwen/Qwen3-VL-8B-Instruct` checkpoint: original floating-point safetensors shards/index, config,
  tokenizer, processor and chat template. The model size is checked from safetensors headers.
- Full CLIP ViT-B/32 model and tokenizer, including the text projection.
- A compatible installed Transformers/accelerate/PyTorch/torchvision stack. The preflight checks the actual native
  Qwen class. The Florence 4.49 offline environment cannot run Qwen3-VL; the notebook never installs packages.

Edit `ISIC_ROOT`, `QWEN_8B_MODEL_PATH`, `CLIP_MODEL_LOCAL_PATH`, `DINOV3_REPO_DIR` and `DINOV3_WEIGHTS` if automatic
discovery/defaults do not match attached inputs. `QWEN_RUNTIME_SITE_PACKAGES` optionally selects a prepared compatible
runtime in the child workers; it does not modify the parent kernel. With no override, the workers use the kernel's environment.

## Run Order

1. Validate standard 2,000/150/600 pairs and a fixed 400-image training candidate pool.
2. Generate/resume one pixel-only Qwen caption per image. No mask or diagnosis is passed to the VLM.
3. Export `caption_generation/captions.csv`, then encode valid captions using frozen CLIP in a separate process.
4. Initialize the frozen-DINO segmentation model, text heads and multimodal selector after both large-model workers exit.
5. Train query/candidate text fusion and text+visual selection, select by validation Dice, and evaluate test.
6. Export comparison against the previous rounded visual reference (Dice 85.60%, IoU 77.38%), curves and image/text/mask examples.

`MAX_NEW_CAPTION_IMAGES=0` means the full dataset. A positive smoke-test limit writes only a partial export;
training is blocked until it is reset to 0 and generation is resumed. Invalid captions are not retried with fabricated text.

## Design

- **Query fusion:** projected CLIP query text is gated into query patch tokens.
- **Candidate fusion:** the selected candidate caption is independently projected/gated into context tokens.
- **Selector:** original visual score plus a learned sigmoid coefficient times valid CLIP text cosine similarity.
- **Training:** straight-through selection for image tokens AND text; policy actions use the same self-masked scores.
- **Reward:** query image + query text, versus that same input plus candidate image + candidate text.
- **Policy:** `isic_visible_attributes_v1`, 5-8 cleaned English visual words, diagnosis/token-limit checks.
  DTD class-word exclusions and the old CLIP 0.30 hard rejection are NOT used for binary ISIC segmentation.
- **Masks:** rejected captions have zero text vectors and zero text gates; their images remain available.

The visual recipe retains 448 resolution, ViT-S/16, two final-token cross-attention blocks, DPT-lite,
BCE+Dice, batch 8, 100-epoch maximum, patience 15 and seed 42. Added text heads increase capacity; comparing this
full system to the historical run does not isolate only caption quality. This is not a clinical diagnostic system.

## Outputs and Resume

Default root: `/kaggle/working/outputs/mtacs_isic2017_448_qwen8b_full_both/`, separate from the visual experiment.
Keep the entire root to preserve per-image caption caches, CSV, text cache, locked protocols, split/pool manifests,
best/last checkpoints, RNG/optimizer/scheduler state, history, complete test predictions and metrics.
Changing model assets, captions, input pixels/masks or training settings requires a new output root.
Training resume with persistent augmentation workers is not guaranteed bitwise identical to an uninterrupted run.
The notebook does not load an old visual checkpoint into the new architecture.

No GPU generation/training has been measured by the builder. Local tests use synthetic data and tiny CPU models.

Sources: [official Qwen model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Transformers Qwen3-VL API](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).
