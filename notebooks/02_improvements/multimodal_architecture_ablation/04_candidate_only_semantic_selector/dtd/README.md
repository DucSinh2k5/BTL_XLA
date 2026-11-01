# 04: Candidate-Only With An Aligned Image-Caption Selector

[Open notebook](mtacs-dtd-vitl16-prefix-candidate-only-semantic-selector.ipynb).
[Review of actual study02/03 outputs](../../../../../reports/multimodal_architecture_ablation/dtd/STUDY_02_03_REVIEW_AND_STUDY_04_PLAN.md).

Study02/03 completed at test macro-F1 0.849329/0.848722, below the historical study01 0.853032.
Their fixed-image validation text-off/shuffle interventions did not reduce F1. Study03 learned training-pool
image-caption correspondence but did not improve classification. No study04 GPU results are available.
Keep study03 as the structural control, not as a validation/test winner.

## Controlled Change

Retain study03's normalized candidate-caption fusion, same own-image/caption alignment loss and positive diagonal metric.
Use that aligned space to score query-image CLS versus candidate-caption embeddings in the selector.
Query captions remain zero/masked; this is not the older text-query/text-candidate selector.

```text
visual_z = original image-only selector score normalization
semantic = -distance(diagonal_metric, normalize(query_DINO_CLS), aligned_candidate_caption)
semantic_z = clip(masked_z(semantic), -3, 3)
selector_score = visual_z + alpha * semantic_z
alpha = 0.25 * sigmoid(existing_semantic_mix_logit) * epoch_ramp
```

Train image-only selection for epochs1..5; ramp semantic selection in epochs6..10, then retain its learned value.
The checkpoint stores the ramp epoch. These are declared conservative starting settings, not test-tuned optima.
No extra trainable parameters over study03: reuse its already-existing semantic_mix_logit.
Query image receives selector gradients; frozen CLIP captions do not. The same text projection/metric
receives the original auxiliary loss and now selector policy gradients as well.

Invalid captions contribute zero semantic residual but their images remain candidates.
Semantic mean/variance exclude self and invalid captions. Constant/insufficient caption scores fall back to visual selection.
Exact self image is never selected. Statistics do not couple different query images; no inference Sinkhorn/OT is applied.
This is a project adaptation using [m-LTM-inspired alignment](https://arxiv.org/abs/2405.10084), not an exact paper reproduction.

Fusion is unchanged: query image + one selected candidate image + that candidate's original valid caption.
No query caption, captioner fine-tuning, new CSV, token fusion, Partial OT or ASWP is introduced.

## Inputs And Run

Attach the same five input groups:

- DTD with images/ and labels/train1.txt, val1.txt, test1.txt.
- Complete local DINOv3 code with hubconf.py.
- dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth.
- Complete local CLIP ViT-B/32 model/tokenizer folder.
- Exact [captions_prefix_control.csv](../../../../../data/captions/dtd/grounded_controls/captions_prefix_control.csv).

CSV SHA256: b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011.
Policy remains dtd_lexical_v2; semantic leakage scores are diagnostic-only. Captions are synthetic weak inputs, not ground truth.
Only training-pool captions are encoded. No Florence/Qwen weights, references, old best.pt, project code or pip/network are needed.

Upload this notebook and Run All from the top. Use CAPTIONS_INPUT_PATH for an explicit uploaded CSV, or leave blank for hash-checked discovery.
Use CLIP_MODEL_LOCAL_PATH, DTD_ROOT, DINOV3_REPO_DIR and DINOV3_WEIGHTS for nondefault attachment paths.
The DINO defaults follow the executed study03 attachment paths; always check the printed assets.

Defaults unchanged: split1, pool/training seed42, 376 candidates, K1, ViT-L16, 384, same GPU-aware batch policy,
50 epochs/50-epoch LR horizon, no early stopping, best minimum validation classification CE.
Train independently from the same pretrained setup. Do not initialize from or reuse output/checkpoints of study02/03.

Output:

```text
/kaggle/working/outputs/mtacs_dtd_split1_architecture_04_prefix_candidate_only_semantic_selector_vitl16/
```

Own environment overrides: DTD_PREFIX_SEMANTIC_SELECTOR_SEED, OUTPUT, EPOCHS and WORKERS
(each prefixed DTD_PREFIX_SEMANTIC_SELECTOR_). Keep workers/batch/runtime matched for small-effect comparisons.
TOKENIZERS_PARALLELISM=false is set before tokenizer/worker usage; diagnostic loaders always use zero workers.
No automatic package downgrading/installing. The saved study02/03 runtimes differed in Torch/NumPy;
their tiny score difference alone is not a clean causal effect.

## Diagnostics And Exports

Both diagnostic suites are validation-only and use the completed own best checkpoint.

| Suite | Interventions | Context images | Fusion captions |
| --- | --- | --- | --- |
| Fusion | aligned / text_off / text_shuffled | Fixed to aligned selections | Changed only as stated |
| Selector | aligned / visual_selector_only / selector_text_shuffled | Reselected | Always original caption of selected image |

Invalid-caption availability masks are preserved for shuffles, using only eligible training candidates.
Export per-image selected image ID, original fusion caption ID, and separate selector-caption source ID.
This prevents a selector shuffle from accidentally corrupting fusion too.
Paired confidence intervals resample validation queries; they do not measure variation across training seeds.
Neither intervention is an independently retrained ablation.

Root exports include architecture_04_prefix_candidate_only_semantic_selector_results.csv, report.md,
learning/alignment curves, validation_interventions.csv, validation_selector_interventions.csv and training_pool_alignment_summary.csv.
Per-run history, best.pt, val/test predictions, protocol/assignment manifests are retained as in study03.
Detailed validation artifacts are in runs/candidate_only/seed_42/validation_norm_diagnostics/ and validation_selector_diagnostics/.
History adds train_selector_semantic_weight. This selector alpha is different from c_gate in fusion.

RUN_FINAL_TEST=True runs the locked test after completion. Set False during validation-only development.
For a retrained visual-selector control, set SELECTOR_SEMANTIC_MAX_WEIGHT=0.0 before Run All in a NEW output directory.
The zero-weight path preserves study03's sampling, common initialization and classification/alignment losses.
Do not choose alpha, checkpoints or later architecture variants by repeated test-score tuning.

## Implementation

Builder: tools/notebook_builders/build_dtd_prefix_semantic_selector_notebook.py.
Embedded helpers: src/training/mtacs_dtd_semantic_selector_cells.py.
CPU tests cover score masks/fallback, no query-text dependency, gradient routing, exact zero control,
epoch-boundary resume, both diagnostic suites and final result/report generation.
These tests do not substitute for a full GPU training run.
