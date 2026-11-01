# Organization Worklog

## 2026-10-26

### Study02/03 Saved-Output Audit And Study04 Aligned Semantic Selector

- Read the user's executed stydu2_da_train.ipynb and study3_datrain.ipynb plus both extracted result packages without executing notebook cells or loading checkpoint pickle. Preserved all supplied files and the canonical Prefix CSV.
- Recomputed val/test accuracy and macro-F1 from the 1,880 per-split predictions; verified 50 history epochs, best38/min-val-CE, same pool/split IDs, caption hash, hyperparameters and aligned candidate caption identities. Study02 val/test F1=0.835824/0.849329; study03=0.834947/0.848722. Fixed-context validation off/shuffle did not reduce F1; alignment learned training-pool matching without a classification gain.
- Exported a reproducible read-only audit, evidence JSON, comparison/intervention CSVs and review/next-step report. Noted different Torch/NumPy versions, changed DINO attachment sources and missing full pretrained-weight hashes; tiny differences are not attributed solely to alignment.
- Added standalone study04 under 04_candidate_only_semantic_selector/dtd/, retaining study03 fusion/alignment as a structural control. The new selector residual compares query DINO images with candidate CLIP captions using shared alignment projection/metric; query captions remain masked. Bounded learned alpha, 5-epoch visual warmup plus 5-epoch ramp, masked/clipped semantic z and invalid/constant-caption fallback. No new trainable parameters over study03; ramp epoch is checkpointed.
- Added separate validation interventions: fixed-image fusion text off/shuffle, and reselected-context selector off/shuffle with original per-image fusion captions. Retained fixed Prefix policy, K1, official split1/seed42, 50 epochs and own-output/checkpoint guards. No old checkpoint warm start or new caption generation.
- Verification: 75 scoped CPU tests passed, including no query-text dependency, gradient routing, self/missing/constant masks, query-batch independence, rejecting test queries before diagnostic inference, exact zero-weight equivalence to study03, active-selector epoch-boundary resume and full training/diagnostics/test/report smoke workflows. No full study04 GPU training or performance improvement is claimed.

### DTD Architecture Study 03: Auxiliary m-LTM-Inspired Alignment

- Added a standalone study03 notebook based on study02, with the same normalized candidate-only inference and image-only selection, fixed Prefix CSV, 50 epochs and independent output/protocol identity.
- Added a training-only auxiliary KL objective through a differentiable balanced log-Sinkhorn plan; the cost uses an identity-initialized positive mean-one diagonal Mahalanobis metric (+1024 parameters on ViT-L16).
- Alignment samples eligible training-pool images and their own captions, not query/retrieved-context pseudo-pairs; it reuses detached epoch-start image anchors and the actual fusion text projection without dropout. Duplicate canonical captions receive balanced positive-block targets.
- Retained original task/selector losses and checkpoint rule. The independent seed/epoch/step sampler consumes no global classifier RNG; zero alignment weight is an explicit normalized-fusion control. Added alignment logs, pair/config manifests and training-pool fit diagnostics, explicitly not held-out retrieval.
- Documented the adaptation's differences from the audio-text m-LTM paper: detached image side, diagonal metric parameterization, duplicate-aware targets and no Partial OT. No captioner fine-tuning, CSV regeneration, previous-checkpoint warm start or GPU score is claimed.
- Verification: 64 scoped CPU tests passed, including a closed-form transport reference, differentiable Sinkhorn gradients, balanced duplicate targets, shared-projection gradients with detached teachers, unchanged inference/initialization, zero-weight equivalence to study02, epoch-boundary optimizer/metric/sampler resume and full train/diagnostic/test/report workflows. Parsed all 26 clean cells and verified 100 local documentation links; executed baseline and Prefix CSV hashes are unchanged. No GPU training was performed locally.

### DTD Architecture Study 02: Norm-Calibrated Candidate Fusion

- Added a standalone Prefix candidate_only notebook under `02_candidate_only_norm_fusion/dtd/` with independent output/protocol identity and an early foreign-output guard.
- Retained model parameters, initialization, image-only selector, losses/optimizer, augmentations, K1, split1/seed42, ViT-L16/384 and the default 50-epoch recipe. The only model change is L2 normalization of projected candidate text before the learned gate and fusion; query text stays disabled.
- Embedded validation-only aligned/text-off/text-shuffled diagnostics using the new model's best checkpoint, fixed selected images, label-free eligible-caption derangement, unchanged masks and zero-worker diagnostic loading.
- Added paired validation deltas, probability/logit changes, per-image norm/gate ratios, stratified norm summaries and shuffle/source manifests with strict completed-run/identity/cache guards.
- Kept training worker defaults for comparability, with a documented optional zero-worker override and its augmentation-RNG/protocol caveat. No caption regeneration, baseline warm start or GPU result was inserted.
- Preserved the user's executed study01 notebook, baseline template and canonical Prefix CSV.
- Verification: 54 focused CPU tests passed, including unchanged parameter initialization/selector, unit-norm text, masked-branch equivalence, projection-scale invariance, eligible-only shuffle, fixed-context validation, output/cache guards and train/resume/final-test/report workflows. All 26 cells parse, 97 local documentation links resolve, and the executed baseline/Prefix CSV hashes are unchanged. No GPU training was performed locally.

## 2026-10-25

### DTD Architecture Baseline 01 Saved-Output Audit

- Read `dtd-architecture-study-01.ipynb` without executing cells or modifying the file; SHA256 `3254ec626634f7922472963e4ccb64c5016cdc2f43730ba6c3fb411e739e5e07`.
- Confirmed Prefix CSV hash against local bytes, unchanged model/training definitions versus the baseline template, 324/376 eligible pool captions and best epoch 38.
- Recorded saved val/test Accuracy 0.841489/0.854255, Macro-F1 0.840162/0.853032 and completed 50-epoch status, with a missing epoch-48 stdout line and worker exceptions explicitly retained as caveats.
- Noted that visual features are L2-normalized but projected text uses LayerNorm; the mean candidate gate 0.001117 is not an interpretable percentage contribution by itself.
- Added a structured evidence file and report; recommended validation text off/shuffle and norm-ratio diagnostics, then an isolated normalized-fusion control before learned alignment. No checkpoint loaded, notebook relocated or GPU training rerun.

### DTD Florence-Prefix Candidate-Only Architecture Baseline

- Added standalone baseline 01 under `notebooks/02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/`.
- Bound the exact Prefix CSV by SHA256 and generation metadata; retained `dtd_lexical_v2` without caption regeneration or policy changes.
- Kept the existing candidate_only model, optimizer/losses, augmentations, train/evaluation/resume functions and pool selection.
- Disabled query text and text-aware selection; encode only the 376 training-pool captions, gathering caption/image at the same selected position.
- Retained official DTD split1, ViT-L/16, 384, K1, seed42, default 50 epochs and the validation-loss checkpoint rule.
- Isolated output paths and protocol identity; exported provenance, caption coverage, metrics and learning curves without prefilled historical scores.
- Added Kaggle input/path instructions. Original executed notebooks, caption files and prior manifests are unchanged; no GPU training was performed locally.
- Verification: 54 focused CPU tests passed, including real Prefix CSV binding/audit/cache, masked query inputs, candidate-text gradients, visual-only selection, train/resume/final-test/report workflows and existing screening/budget guards. All 24 cells parse, 96 documentation links resolve, and all 5,640 CSV image IDs match the local official DTD split1 files.

### ISIC Florence-Prefix Full-Both Caption Control

- Added an independent standalone notebook under `notebooks/02_improvements/segmentation/full_both/florence_prefix/isic2017/`.
- Chose the frozen Prefix procedure from observed DTD Florence-method results (macro-F1 0.810651), explicitly not a proven ISIC winner.
- Reused Qwen full_both model, data/pool, loss and training/evaluation definitions; changed only caption backend, provenance and historical comparison labels.
- Embedded pixel-only greedy Florence generation, per-image resume caches, the same ISIC clinical/validity masks and frozen CLIP encoding.
- Reused the offline 4.49 child-runtime helper for the user's `notebook-mt-moi/Florence-2-base` attachment; kernel Transformers is not downgraded.
- Kept 448/S16, 400 training candidates, K=1, batch8, max100 epochs, patience15 and seed42; added both historical visual and Qwen references without inventing Florence scores.
- Preserved both executed ISIC notebooks and the existing Qwen template. No GPU caption generation or training was run locally.
- Verification: 50 scoped CPU tests passed for both caption backends, unchanged segmentation/training contracts, offline runtime isolation and saved-output audits. Parsed all 19 new cells, verified README links and confirmed both executed notebook hashes are unchanged.

### ISIC Qwen8B Saved-Output Comparison

- Audited saved outputs from `isic-segmen-captions.ipynb` against the original visual 448 run without executing, editing or relocating either notebook.
- Recorded Qwen8B full_both test Dice 0.848689, IoU 0.765407, best epoch 26 and early stopping at 41/100; historical visual remains 0.8560/0.7738, best epoch 49.
- Added the ISIC comparison report and JSON evidence; updated the Table 3 extension row and readiness/status links.
- Verified 2,750/2,750 policy coverage, nonzero validation text gates, common visual source/metric definitions and the reviewed embedded worker. Kept single-seed, rounded-baseline, missing independent CSV/checkpoint and changed-capacity/retrieval limitations explicit.
- Verification: 33 focused comparison/segmentation/parser/model tests passed. Both executed notebook SHA-256 hashes remain unchanged; no GPU run or notebook-cell execution was performed by this audit.

## 2026-10-24

- Read saved validation/test tables, 50 epoch logs, coverage and metadata from six DTD caption-method runs.
- Added Qwen3-VL-8B-Instruct/32B-Instruct classifier results to the caption-method report and JSON evidence.
- Renamed and relocated four Florence and two Qwen executed notebooks into `experiments/completed/caption_ablation/dtd/`, one folder per method.
- Verified full-file SHA-256, byte length and saved-output SHA-256 for all six before/after moves; no notebook content changed or cells executed.
- Preserved historical migration inventories and registered these later imports with a separate relocation manifest and results index.
- Updated navigation and report-readiness links; runnable templates and caption CSV inputs remain in their existing locations.
- Verification: 14 focused evidence/parser/preservation/link tests passed; all six report rows match dedicated saved metric tables. No GPU training was performed locally.

### ISIC 2017 Segmentation Evidence

- Read `tacs_isic_segement_448.ipynb` without changing its location, bytes or execution state.
- Added Table 3 project entries for TACS-variant x ISIC 2017: displayed test Dice 85.60%, IoU 77.38%; best epoch 49, early stopping at 64/100.
- Verified the paper's Table 3 and implementation details from the local PDF; marked the 448/final-token fusion/DPT-lite and single-seed differences explicitly.
- Kept missing project baselines/datasets unfilled; no published number substituted for an unmeasured project run.
- Added structured segmentation evidence and extended the generic output audit to recognize test Dice/IoU and `forbidden_indices` self-mask terminology.
- Verification: 20 focused segmentation/caption evidence and parser tests passed, including original notebook byte hash, test-vs-paper metric separation, epoch consistency and report links.

### ISIC Qwen8B Full-Both Notebook

- Created a separate unexecuted notebook under `notebooks/02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/`.
- Embedded offline per-image resumable Qwen3-VL-8B generation and an isolated CLIP encoder before segmentation initialization.
- Added query/candidate text token fusion, visual+text selection and differentiable selected-text gathering, preserving exact-self exclusions in both task/policy paths.
- Retained the 448/S16 visual recipe, BCE+Dice, 20% pool, K=1, batch 8, 100-epoch maximum and validation-Dice checkpoint rule; output root is independent.
- Added ISIC-specific visible-attribute policy rather than the DTD class filter; mask/diagnosis data are excluded from caption payloads.
- Added run/asset/data locks, checkpoint resume, complete test exports, historical comparison and caption/mask examples. No GPU result has been claimed or inserted into Table 3.
- Verification: 114 scoped CPU tests passed across ISIC model/caption/workflow contracts, existing Qwen DTD exporters/classifiers and result audits. The ISIC test runs the actual new train/evaluate functions with synthetic image-mask batches and a tiny frozen backbone; no Qwen/DINO weights or external data are downloaded.

## 2026-10-22

Organized the TACS subset of XLA into a dedicated `TACS/` project.

- Applied 154 planned move entries, inventorying 211 files and 78 notebooks.
- Categorized notebooks by paper table/task, method and dataset, or by improvement study/method/dataset.
- Preserved all notebook source cells, outputs, CSV inputs, model artifacts, plots and paper bytes.
- Separated eight executed DTD text-context runs from their runnable standalone templates.
- Separated the four caption-method classifier notebooks from their CSV inputs.
- Preserved historical versions and annotation evidence; no duplicate checkpoint bundle was deleted.
- Updated builder/test source locators and the caption-method notebook manifest.
- Added empty dataset/method slots, progress/run/artifact folders, documentation and local path examples.
- Left unrelated coursework, raw datasets, Python environments and shared dependencies in XLA.

### Verification

- Full regression suite: **238 passed**, one existing Starlette TestClient deprecation warning.
- Evidence: `manifests/regression_tests.xml` relative to the project root.
- Final preservation audit: **211 files checked**, **78 notebooks unchanged**, **0 errors**.
- 171 inventoried files are byte-identical; the 40 locator/manifest updates are explicitly listed in the audit JSON.
- Checked 105 main navigation links; all resolve.
- Parsed 38 Python source/tool files successfully.
- Caption CLI help and uvicorn CLI help work in the checking interpreter.
- Figure 4 CLI needs `torchvision`, absent from that interpreter. No dependency was installed automatically.
- No GPU training, new predictions, caption generation, checkpoint conversion or metric changes were performed.

An intermediate Windows checkpoint-replacement test encountered a temporary file lock;
its isolated retry and the final full regression run both passed without changing the notebook's training code.
