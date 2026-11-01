# 03: Candidate-Only, Norm Fusion And Auxiliary Alignment

[Open notebook](mtacs-dtd-vitl16-prefix-candidate-only-mltm-alignment.ipynb)

This is an independent experiment against [study02 norm-calibrated fusion](../../02_candidate_only_norm_fusion/dtd/README.md).
It keeps the same inference path, image-only selector and masked query text; it adds an auxiliary
image-caption alignment loss and a learned positive diagonal metric. It does not train a captioner or create a new CSV.

## What The Two New Variants Test

| Variant | Change | Question |
| --- | --- | --- |
| Study02 | L2-normalize projected candidate text before gate/fusion | Does controlling feature magnitude help? |
| Study03 | Keep study02, add auxiliary matching loss and a positive diagonal metric | Does explicitly learning image-caption correspondence help beyond scale control? |

Study02 already learns its text projection and gate through the original task objective; it is not a non-learning model.
Study03 adds a separate alignment signal to those same projection parameters. Compare 03 to 02,
not only to study01, to distinguish the effect of alignment from normalization.
Each trains from the same pretrained setup and fresh heads; study03 does NOT continue from study02's best checkpoint.

## Auxiliary Objective

For each training step, sample up to 32 eligible training-pool image IDs without replacement,
independently of which contexts the visual selector chose for the classification queries.
For each sampled image, use **its own caption**. The image anchor is its detached DINO CLS feature
from the epoch-start candidate bank; captions use the actual fusion text projection, bypassing dropout
only for the auxiliary objective, followed by the same L2 normalization.

```text
L_total = L_study02 + weight(epoch) * KL(target_matching || Sinkhorn_plan)
weight(epoch) = 0.05 * min(1, epoch / 5)
```

The transport cost is a Mahalanobis distance with a learned positive diagonal, normalized to mean 1.
It starts as the identity and adds 1,024 trainable parameters on ViT-L/16, without perturbing common weight initialization.
The diagonal is training-only; it does not replace the visual selector or change inference scoring.
Gradients reach the shared caption projection and metric, not the detached image teacher or frozen CLIP.
The DINO backbone continues learning from the original classification/context objective.

The method is inspired by [mini-batch Learning-to-Match](https://arxiv.org/abs/2405.10084),
which fits paired correspondence using entropic transport and a Mahalanobis ground cost.
This notebook is a DTD adaptation, **not an exact reproduction of the audio-text paper**:
the image side is detached, the metric is diagonal with fixed mean, and duplicate captions use balanced positive blocks.
Partial OT is not implemented here. The embedded PyTorch log-Sinkhorn solver needs no POT installation.

### Duplicate And Noisy Captions

Two candidate images with the same canonical caption are not forced to compete as unique text identities.
The target spreads each row's mass uniformly over columns sharing that caption; row/column marginals remain uniform.
Otherwise, the paired target is diagonal. No class labels or validation/test captions define this target.

Captions remain synthetic weak correspondences, not independently verified ground truth.
Different captions may still describe similar textures, and image-caption mismatches may remain.
Balanced transport can therefore be harmful; this experiment measures that possibility rather than assuming alignment helps.
Normalizing vectors or minimizing a matching loss does not guarantee a perfectly shared semantic space.

## Defaults And Inputs

Unchanged: Florence-Prefix CSV and `dtd_lexical_v2`, official DTD split1, pool/training seed42,
376 training candidates, K1/self-exclusion, DINOv3 ViT-L16, 384x384, the GPU-aware batch policy,
50 epochs/50-epoch LR horizon, no early stopping, and best validation-classification-loss checkpoint.
Self-exclusion applies to retrieval; own-image/own-caption positives for auxiliary alignment are intentional.

New settings in the configuration cell:

| Setting | Default |
| --- | ---: |
| `ALIGNMENT_WEIGHT` | 0.05 |
| `ALIGNMENT_WARMUP_EPOCHS` | 5 |
| `ALIGNMENT_BATCH_SIZE` | 32 |
| `ALIGNMENT_OT_EPSILON` | 0.1 |
| `ALIGNMENT_SINKHORN_ITERS` | 100 |
| `ALIGNMENT_MAX_MARGINAL_ERROR` | 0.005, relative to the uniform marginal |
| `ALIGNMENT_SAMPLE_SEED` | 2026 |

These are declared starting hyperparameters, not values selected using test results.
Weight 0 is an explicit zero-loss control; sampling and extra loss computation are skipped.
Training history records raw alignment KL, effective weight, positive transport mass, pair cosine and marginal error.

Attach the same five inputs as study02:

- DTD images and official split1 label files.
- Complete local DINOv3 code with `hubconf.py`.
- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
- Complete CLIP ViT-B/32 model/tokenizer directory.
- Exact [captions_prefix_control.csv](../../../../../data/captions/dtd/grounded_controls/captions_prefix_control.csv).

CSV SHA256: `b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011`.
No Florence/Qwen weights, references CSV, study01/02 checkpoint, extra transport package or project Python files are needed.
The notebook is self-contained, offline and does not install packages or regenerate captions.

## Run And Resume

Upload the notebook, attach those inputs, enable GPU and Run All from the top.
Check `CAPTIONS_INPUT_PATH` (blank allows hash-checked discovery), `CLIP_MODEL_LOCAL_PATH`,
`DATASET_ROOT`, `DINOV3_REPO_DIR` and `DINOV3_WEIGHTS` if your attachment paths differ.

Default output:

```text
/kaggle/working/outputs/mtacs_dtd_split1_architecture_03_prefix_candidate_only_mltm_alignment_vitl16/
```

Do not reuse study01/02 output paths or checkpoints. Alignment settings are locked into the run protocol.
The independent sampler derives indices from training seed, epoch and step, supporting epoch-boundary resume
without changing the classifier's Torch/NumPy random streams. Fresh uniform sampling does not require extra image inference.

Training workers retain prior defaults; diagnostic loading uses zero workers and tokenizer parallelism is disabled early.
If multiprocessing errors recur, set `os.environ["DTD_PREFIX_MLTM_ALIGNMENT_WORKERS"] = "0"`
in configuration before settings. Record the worker/RNG deviation and match it across controls for small-effect comparisons.

## Diagnostics And Results

The existing validation aligned/off/shuffled interventions retain the same context image per query,
the same masks and the same checkpoint. They do not update weights or optimize the test set.

An additional table, `training_pool_alignment_summary.csv`, records fit on eligible training candidates:
transport KL/positive mass, pair cosine, exact-source and caption-group top1, and metric condition.
**This is a training-pool fit diagnostic, not held-out retrieval performance.** Exact-source top1 is ambiguous
for duplicate captions; inspect group top1 too. It cannot establish caption truthfulness or downstream benefit.

Main exports:

- `architecture_03_prefix_candidate_only_mltm_alignment_results.csv` and `..._report.md`.
- Classification learning curves and `..._alignment_training.png`.
- `runs/candidate_only/seed_42/alignment_pair_manifest.csv` and `alignment_config.json`.
- Standard protocol, checkpoints, history, val/test metrics and predictions, CSV/hash and candidate manifests.
- Validation interventions, per-image norm/gate diagnostics and training-pool alignment fit in `validation_norm_diagnostics/`.

Best checkpoint remains selected by validation classification loss, not alignment loss or training-pool retrieval.
Set `RUN_FINAL_TEST=False` while developing on validation. The template remains clean and unexecuted.
The user's completed study03 run now reports val/test F1 0.834947/0.848722 (50 epochs, best38),
without a classification gain over study02. See the [saved-output review and study04 plan](../../../../../reports/multimodal_architecture_ablation/dtd/STUDY_02_03_REVIEW_AND_STUDY_04_PLAN.md).
