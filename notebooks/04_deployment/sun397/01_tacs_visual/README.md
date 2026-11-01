# SUN397 Visual TACS ViT L16 Training and Web Export

Notebook: `sun397-tacs-visual-vitl16-train-export-web.ipynb`.

This is notebook 1 for deployment. It trains a new visual-only TACS on SUN397,
using DINOv3 **ViT-L/16**, 384px, one context, shared backbone, patch concatenation,
Gumbel straight-through task gradients and categorical policy loss. No existing
S/16 weights are reused as a trained classifier. L/16, image size, single seed,
microbatch/gradient accumulation, held-out validation and detached epoch-refreshed
candidate feature caching are declared adaptations, not an exact paper protocol.

## Kaggle Inputs

- SUN397 images plus `ClassName.txt`, `Training_01.txt`, `Testing_01.txt`.
- Official DINOv3 source including `hubconf.py`, `dinov3/`, license and notices.
- Official `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` with its original
  filename, not S/16, B/16, SAT or L/16-plus.
- A compatible GPU PyTorch environment with NumPy, Pillow and tqdm.

No caption CSV, CLIP, Florence, Transformers, external API or project source upload
is required. Runtime/training Python sources are embedded. Internet stays off;
the notebook does not install packages. Change the Input paths in section 2.
The runtime calls the official local backbone factory directly, avoiding unrelated
detection/segmentation imports from `hubconf.py`.

## Run and Resume

Run All. Defaults: partition 01, 17,865 train / 1,985 val / 19,850 test,
3,573 training-only candidates, K=1, seed42, 100 epochs, minimum validation CE.
The final test is separate from training and uses the best checkpoint in fp32.
Set a different epoch budget **before** starting a new run, never to bypass resume
identity. Keep the same output folder, `last.pt` and `best.pt` to resume.
Restarting the kernel resumes only if that working folder still exists. A new
Kaggle session may start with empty `/kaggle/working`: attach the prior saved
Output and restore the full run folder to `OUTPUT_DIR` first, including
`protocol.json` and the embedded runtime. The deployment ZIP alone cannot resume
training because it intentionally excludes optimizer/scheduler/RNG state.

CPU candidate pixels are fp16 to reduce RAM; the custom straight-through operation
keeps the exact derivative of a hard-weighted fixed image bank in fp32 chunks.
The bank does not silently become an index-only gradient path. With workers=0,
RNG/loader states are restored at completed epoch boundaries. A partial interrupted
epoch is replayed. Finite loss, correct self-exclusion and raw-image ID alignment
are checked rather than skipped.

## Output

Main output root:
`/kaggle/working/outputs/sun397_tacs_visual_vitl16_web_v1/`.

Download:
`exports/sun397_tacs_visual_vitl16_web_bundle.zip`.
Rerunning export reuses a complete verified ZIP belonging to the same best
checkpoint. A mismatched or partial export is preserved and rejected.

The ZIP contains:

```text
sun397_tacs_visual_vitl16_web_bundle/
  model.pt                       # Full best model, backbone and task head
  config.json                    # Backbone, resize, mean/std, K and protocol
  class_names.json               # Exact 397-class index order
  candidate_pool.json            # Ordered IDs, class map, hashes, portable paths
  candidate_pool.csv
  candidate_features.pt          # Rebuilt from best, bound to model SHA256
  candidates/...                 # All real pool images
  runtime/tacs_visual_web.py     # Model, predictor, real attention and CLI
  runtime/verify_export.py       # Cold offline reload and explanation parity check
  vendor/dinov3/...              # Original backbone modules and license notices
  attention_config.json
  manifest.json                  # File sizes and SHA256
  smoke_test.json
  environment.json
  requirements.txt
  metrics.json
  test_predictions.csv
  history.csv
  protocol.json
  provenance/training_source.py
  provenance/train_split.csv
  provenance/validation_split.csv
  provenance/test_split.csv
  examples/query.jpg             # Extension follows the actual example image
  examples/prediction.json       # Actual prediction, candidate and attention grids
  examples/reload_logits.pt
```

The original Kaggle dataset mount and pretrained `.pth` are **not** required after
extraction: the full trained backbone state and model source are in the ZIP.
Vendored modules are copied verbatim: models, layers, utils and the backbone hub
entry points. Unused detection, segmentation and training config trees are omitted
to keep the portable bundle focused and avoid excessive Windows path lengths.
Python/PyTorch and OS/CUDA dependencies must still be installed; this is not an
operating-system/runtime installer or an offline wheel archive.

## Attention and Web Integration

The trained shared backbone produces both maps; no extra attention model is
trained. Query classifier map: last-block CLS attention restricted to query
patches in the query+context sequence. Candidate selector map: standalone encoder
CLS attention to the selected candidate's patches. Heads are averaged. Raw grids,
map method, role, display normalization and PNG data URLs are returned.

These are real attention maps, not Grad-CAM, not textual names of image features
and not class-conditioned causal explanations. Flat maps are flagged, not replaced
by artificial highlights. The full five-column Figure 4 comparison would also need
a separately trained no-context baseline, which is not included here.

Cold reload must use vendored source, strict model loading and an exact candidate
bank, and must preserve selection/logits when maps are enabled. Failed checks
prevent publishing the ZIP. The runtime provides `BundlePredictor.predict(...)`
with the existing web's `prediction` and `evidence` fields, but modifying/wiring the
current FastAPI application is a separate deployment step. Serialize calls with
a lock for one GPU predictor. Check original model/dataset redistribution terms.

## Implementation and Checks

Builder: `tools/notebook_builders/build_sun397_tacs_web_notebook.py`.
Shared model: `src/models/tacs_visual_web.py`.
Training/export: `src/training/train_sun397_tacs_web.py`.
Tests: `tests/test_sun397_tacs_web_export.py`.

Tests use a small official DINOv3 configuration on CPU to check actual API,
RoPE concatenation, attention, straight-through gradients, resume and packaging,
including inference in a new process from the extracted ZIP alone.
They do not provide SUN397 metrics or validate full L/16 GPU memory/performance.
The actual notebook enforces the 1024-dimension, 24-block L/16 backbone and performs
its own full-model preflight and cold bundle reload after training.
