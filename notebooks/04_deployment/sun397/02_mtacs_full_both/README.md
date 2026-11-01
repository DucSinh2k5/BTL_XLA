# SUN397 M-TACS Full-Both ViT-L/16 Web Export

Notebook: `sun397-mtacs-full-both-vitl16-train-export-web.ipynb`.

## Model

Notebook 2 of the deployment series, not another DTD experiment. It uses SUN397,
DINOv3 **ViT-L/16**, 384px, **one** selected context. Query and candidate image
patch tokens go through the shared visual transformer. Frozen CLIP caption vectors
enter query/candidate gated residuals. A learned visual/semantic selector receives
both query and candidate captions. Invalid text has a zero vector and zero gate;
its image remains eligible. No candidate labels are supplied to selector/fusion.

Same train/val/test/pool setup and default 100 epochs as notebook1: partition01,
seed42, 17,865 train, 1,985 val, 19,850 test, 3,573 pool images from actual train.
Self-candidate masking, Gumbel straight-through and categorical task reward are
used. The reward baseline keeps query text but removes candidate image/text.
Use validation CE for checkpoint selection. Cached candidate features refresh at
epoch start and before validation. This is a declared L16/384px/single-seed/
cached-bank adaptation, not the paper's exact S16/256px protocol.

## Memory-Bounded Training

This notebook now uses the same memory repair as `TACS/tacs_web.ipynb`:

- Logical batch64 defines reward normalization; GPU compute batch8 is a separate
  execution chunk on an approximately96GB GPU. Smaller GPUs use smaller defaults.
- Gradient checkpointing remains enabled. Do not send64 paired examples through
  L/16 at once with checkpointing disabled. Lower only COMPUTE_BATCH_SIZE on OOM.
- A detached reward prepass and RNG-replayed selector gradients allow immediate
  backward per chunk while retaining the full logical-batch advantage statistics.
- Pixel bank is uploaded once to GPU FP32 when there is sufficient free VRAM,
  with explicit CPU fallback. Original fp16 pixel values and full straight-through
  derivatives are retained. Both selected image and selected caption use the same
  Gumbel weights, and the caption matmul is not detached.
- Query/candidate text fusion gates and the visual/semantic selector are unchanged.
  Policy rewards use the action's matching image, caption and validity mask.
- Initial batches print measured images/s and peak VRAM. History also records
  compute batch, bank device, feature-refresh time and validation time.

Restart the Kaggle kernel before Run All, especially after OOM or switching from
another notebook runtime. Exact GPU timing and maximum batch have not been
established by local CPU tests. This repair does not remove the text branches,
policy loss, Gumbel task gradients, or freeze the backbone to gain speed.

## Kaggle Inputs

1. SUN397 images, ClassName.txt, Training_01.txt and Testing_01.txt.
2. Official DINOv3 source including backbone modules, hubconf.py and LICENSE.
3. `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, original filename.
4. Complete original Microsoft Florence-2-base folder, including local Python
   model code, processor, tokenizer and all weight shards. The existing
   `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base` can be used.
5. Complete CLIP ViT-B/32 folder (512-dimensional projection).
6. Offline Florence dependencies from notebook-mt-moi when required: Transformers
   4.49.0, tokenizers0.21.x, Hugging Face Hub<1 and missing einops/timm/accelerate
   wheels compatible with the running Python/platform.

Edit section2 paths and Run All. Source is embedded; no project .py upload or CSV
input is needed for a fresh run. Captioning creates/resumes39,700 rows. When
RESUME_FROM_DIR contains a compatible registered cache, its CSV and frozen CLIP
vectors are copied byte-for-byte into the new output rather than regenerated.
Do not place an unregistered old/Different-policy CSV into shared/. Florence and
CLIP stay frozen; this notebook trains M-TACS, not a captioner fine-tune.

The training kernel never imports/downgrades Transformers. A legacy Florence
child uses an isolated overlay, optionally installed from attached wheels with
`pip --no-index`. No Internet is enabled. Tokenizer parallelism is disabled before
workers. A real Florence/CLIP probe runs before expensive training.

## Caption Policy

`sun397_lexical_clip_audit_v1` is explicitly a new SUN397 deployment policy:

- Strip boilerplate prefixes and retain the first eight readable words.
- Require 5-8 words; reject direct class terms (hierarchy qualifiers alone are
  not treated as class names).
- Record nearest-class CLIP cosine and a threshold0.30 similarity flag **for
  audit only**, not an automatic semantic veto or proof of leakage.
- Mask invalid vectors/gates; do not replace captions with invented text.

This is not `dtd_lexical_v2` and not the old hard semantic-threshold SUN schema.
Synonyms/implicit clues can remain. Captions are generated annotations, not
verified ground truth. Old metrics are not automatically preserved/comparable
across these policy/architecture differences. Never tune settings on test scores.

## Output and Deployment

New root: `/kaggle/working/outputs/sun397_mtacs_full_both_vitl16_web_v1_repaired_v2/`.

Download `exports/sun397_mtacs_full_both_vitl16_web_bundle.zip`, which includes:

```text
model.pt                         # Backbone, gates, projection, classifier, mix
config.json / class_names.json
candidate_pool.json / candidate_pool.csv
candidates/...                   # Every real selected-pool image
candidate_features.pt            # Rebuilt fp32 from best; bound to model SHA
candidate_text.pt                # Aligned caption vectors/masks and model SHA
captions.csv / text_embeddings.pt # All 39,700 registered caption records/vectors
caption_protocol.json / caption_settings.json
models/florence/...              # Full frozen weights/code/processor/tokenizer
models/clip/...                  # Full frozen weights/tokenizer/config
runtime/...                      # Image -> caption -> CLIP -> prediction CLI/API
vendor/dinov3/...                # Exact original backbone modules and license
offline_wheels/...               # Relevant supplied wheels, when available
attention_config.json
metrics.json / test_predictions.csv / history.csv
protocol.json / environment.json / provenance/...
manifest.json / smoke_test.json
examples/...                     # Real fresh-caption prediction and attention
```

The exporter refuses missing/changed best weights, CSV/cache/settings/source or
model assets. A new process loads only the bundle, checks cached-query logits
with/without attention, then forces fresh Florence/CLIP on a validation image.
Parent best and reloaded model must agree on candidate selection/logits for the
same returned vector before publishing. Every file is hashed; ZIP CRC is checked.

The trained backbone provides both attention maps, not a separate model: query
classifier attention to query patches, and selected candidate standalone selector
attention. These visual CLS maps are not text attribution, Grad-CAM or causal
explanations. q_gate/c_gate/semantic_mix are reported separately.

After extraction, no Kaggle Input mount is needed. Python, OS/CUDA-compatible
PyTorch and caption dependencies are still required. The bundle is not an OS
installer. Linux Kaggle wheels cannot install Windows tokenizers: prepare the
appropriate web-host environment. Keep original model/data license terms.

CLI: `python runtime/mtacs_multimodal_web.py --bundle . --image new.jpg`.
`--fresh-caption` forces regeneration for known images too. Unknown images always
use real Florence. `BundlePredictor.predict(...)` returns query caption, selected
candidate image/caption, top classes, fusion gates and attention data URLs/grids.
Serialize calls for one predictor. Wiring the existing FastAPI web remains a
separate step after obtaining the trained bundle.

## Resume and Verification

Keep the full run folder with `last.pt`, `best.pt`, `protocol.json`, shared/ and
embedded_runtime_v2/. Point RESUME_FROM_DIR at the previous full Output folder
(not a ZIP) if a new Kaggle session has empty working/. The default points to the
old run without the `_repaired_v2` suffix. The original folder is never rewritten.

Migration validates legacy source/config/classes/pool/order and all caption-cache
bindings. It copies compatible model/optimizer/scheduler/scaler/RNG/history/best
states into the repaired output. Legacy logical and effective batch sizes remain
fixed; a changed GPU chunk does not silently change reward normalization. An old
checkpoint without the complete bound shared cache is rejected, not trained with
newly regenerated captions. Partial registered caption runs without checkpoints
can resume their missing images. Existing repaired caption progress is not
replaced by an older partial CSV. Registered caption generation settings, including
its original caption batch, are retained on resume.

The new stochastic execution order is not a bitwise replay of the legacy loop.
Migration provenance is recorded in protocol.json and migration.json. If there is
no saved completed epoch, the notebook explicitly starts epoch1. The web ZIP alone
cannot resume optimizer-based training. Do not change epochs, seed, pool, learning
settings or CSV mid-run. Only COMPUTE_BATCH_SIZE can be changed freely in a
repaired run. Choose50 epochs before a new run if using that budget, consistently
for both deployment models.
Use the same compatible PyTorch/GPU precision mode when resuming optimizer/scaler
state; exact numerical replay across different hardware is not guaranteed.

Original notebook backup: `tmp/notebook_backups/sun397_mtacs_web.before_memory_repair.ipynb`.
Builder: `tools/notebook_builders/build_sun397_mtacs_web_notebook.py` (now generates
the repaired version); repair builder: `tools/notebook_builders/repair_sun397_mtacs_web_notebook.py`.
Training patch: `src/training/sun397_mtacs_web_training_patch.py`.
Tests: `tests/test_sun397_mtacs_web_export.py` and `tests/test_sun397_mtacs_web_repair.py`.
CPU checks use an actual tiny
official DINOv3, plus explicitly injected caption fixtures for service contracts;
they do not establish real Florence quality or full L16 GPU/SUN397 metrics. The
actual notebook enforces real model preflight and a fresh real-caption cold check.
