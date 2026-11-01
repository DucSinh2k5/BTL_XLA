# 02: Candidate-Only With Norm-Calibrated Fusion

[Open notebook](mtacs-dtd-vitl16-prefix-candidate-only-norm-fusion.ipynb)

This is the next controlled training experiment after [study01](../../01_candidate_only_prefix/dtd/README.md).
The baseline saved test Macro-F1 is 0.853032. The user's completed study02 run now reports val/test F1
0.835824/0.849329 (50 epochs, best38); the template itself remains clean and unexecuted.
See the [saved-output review](../../../../../reports/multimodal_architecture_ablation/dtd/STUDY_02_03_REVIEW_AND_STUDY_04_PLAN.md).

## What Changes

```text
Study01: c = text_projection(CLIP_caption)
Study02: c = L2_normalize(text_projection(CLIP_caption), epsilon=1e-6)

gate = sigmoid(candidate_gate(concat(visual, c))) * caption_valid
fused = LayerNorm(visual + gate * c)
```

The contextual visual vector already uses L2 normalization. Study02 normalizes projected candidate text
before both gate computation and fusion, without additional parameters, forced gates or losses.
The query text branch remains disabled. The no-context reward branch is unchanged and has no active candidate text.
This tests feature scale, not learned semantic alignment; normalization need not improve performance.

The selector still uses images only, K=1; the caption is gathered from the selected candidate index.
No query caption, m-LTM, Partial OT, semantic selector, token resampler or ASWP is enabled here.

## Required Kaggle Inputs

Use the same inputs as study01:

- DTD `dtd/images/` and official `dtd/labels/train1.txt`, `val1.txt`, `test1.txt`.
- Complete DINOv3 local code repository with `hubconf.py`.
- ViT-L/16 checkpoint `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
- Complete CLIP ViT-B/32 Hugging Face model/tokenizer folder, 512-dimensional text projection.
- Exact [captions_prefix_control.csv](../../../../../data/captions/dtd/grounded_controls/captions_prefix_control.csv).

No study01 `best.pt`, Florence/Qwen weights, references CSV, legacy wheels or project Python files are needed.
Do not warm-start study02 with study01 weights. Training starts fresh, or resumes study02's own unfinished checkpoint.

Expected CSV SHA256: `b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011`.
Caption generation is disabled; the file is staged byte-for-byte and audited with fixed `dtd_lexical_v2`.
Invalid captions remain masked; only training-pool captions are text-encoded. Synthetic captions are not human ground truth.

## Run On Kaggle

Upload this notebook and attach the inputs above. Enable GPU, keep Internet off and Run All from the first cell.
Check these paths if your attachments differ:

- Configuration: `CAPTIONS_INPUT_PATH` can stay blank for exact filename + hash discovery, or contain your exact uploaded CSV path.
- Configuration: `CLIP_MODEL_LOCAL_PATH`, default `/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit`.
- Settings: `DATASET_ROOT`, default `/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd`.
- Settings: `DINOV3_REPO_DIR` must contain `hubconf.py`; `DINOV3_WEIGHTS` must name the ViT-L checkpoint.

Default recipe remains split1, pool/training seed42, 376 training candidates, K1, 384x384,
ViT-L/16, 50 epochs/50-epoch LR horizon, no early stopping, minimum validation-loss checkpoint.
The same GPU-memory batch policy gives batch32 on the baseline's approximately 95-GiB GPU.
Output is independent:

```text
/kaggle/working/outputs/mtacs_dtd_split1_architecture_02_prefix_candidate_only_norm_fusion_vitl16/
```

The early output guard rejects study01/foreign protocols before staging this experiment's artifacts.
Keep a complete copy of this output directory for resume and reporting.

## Validation Diagnostics

After all declared epochs, the notebook reloads **study02's own best checkpoint** and evaluates validation only:

| Intervention | Caption input | Context images |
| --- | --- | --- |
| `aligned` | Correct candidate caption | Selected once per query batch |
| `text_off` | Candidate text validity forced false | Identical to aligned |
| `text_shuffled` | Eligible training-pool captions deranged; validity mask preserved | Identical to aligned |

Query text stays off in every mode. No labels enter the shuffle. Invalid candidate positions remain unchanged.
If only one eligible caption exists, the export marks the shuffle untestable rather than claiming a useful permutation.
Repeated captions can remain text-identical despite changed source IDs; the assignment file records both changes.

Exports include validation Accuracy/F1/loss, prediction-change rate, logit/probability changes, paired F1 deltas,
and per-image gate/norm statistics. Norm summaries report all samples, available-caption samples and active-caption samples.
The ratio `||gate * projected_normalized_text|| / ||visual||` is more informative than gate alone;
neither quantity proves text is semantically correct or measures the benefit of training with text.
Bootstrap intervals resample validation queries, not training seeds.

Diagnostics do not change weights or select a different checkpoint; they never evaluate test interventions.
They require a completed run and cache outputs only when checkpoint/protocol/settings match.
Set `RUN_VALIDATION_DIAGNOSTICS=False` to skip them. Set `RUN_FINAL_TEST=False` for validation-only development;
the normal locked final test otherwise runs after diagnostics without changing the model.

## DataLoader Notes

`TOKENIZERS_PARALLELISM=false` is set before tokenizer or worker use. Diagnostic loading uses `num_workers=0`.
Training worker defaults remain the same as study01 to preserve the recipe.
If notebook multiprocessing shutdown errors recur, set the following in the configuration cell before settings:

```python
os.environ["DTD_PREFIX_NORM_FUSION_WORKERS"] = "0"
```

Worker count is part of the recorded training protocol. Changing it can alter augmentation RNG; report this deviation
and use the same setting for a matched baseline rerun when attributing small gains to normalization.
Do not change it mid-resume or point study02 at study01's output directory.

## Main Outputs

- `architecture_02_prefix_candidate_only_norm_fusion_results.csv`, `..._report.md` and `..._training.png`.
- Root `validation_interventions.csv`: aggregate intervention summary across the declared training seed(s).
- `runs/candidate_only/seed_42/validation_norm_diagnostics/`: per-image predictions, norm distributions,
  shuffle assignments, intervention summary and cache metadata.
- Standard `best.pt`, `history.csv`, val/test predictions and metrics, protocol, environment, caption method and pool/cache manifests.

Keep split, CSV, pool, batch/worker policy and seed fixed for comparisons. Do not select settings by repeatedly comparing test F1.
