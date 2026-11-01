# DTD Multimodal Architecture Study

Start with [01: candidate_only + Florence-Prefix](01_candidate_only_prefix/dtd/README.md).
This is an independent baseline, not a new measured improvement and not a paper reproduction claim.

## Controlled Protocol

- DTD official split 1: 1,880 training, 1,880 validation and 1,880 test images.
- DINOv3 ViT-L/16; 384x384; K=1; 376 training-pool images; exact-self exclusion.
- Pool seed 42, training seed 42, 50 epochs and 50-epoch LR schedule by default; no early stopping.
- Select the checkpoint by minimum validation cross-entropy. Match GPU/batch policy across runs.
- Frozen `captions_prefix_control.csv`, policy `dtd_lexical_v2`; CLIP semantic scores are diagnostics only.
- These captions are synthetic model inputs, not human-verified ground truth.

Expected CSV SHA256: `b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011`.
The notebook verifies the full file, method metadata and official split membership before using training-pool captions.
It never uses candidate class labels as selector/fusion inputs, nor validation/test captions as candidate context.

## Experiment Map

| Stage | Question | Status |
| --- | --- | --- |
| 01: Prefix candidate_only | What is the candidate-caption baseline with this exact CSV? | Saved split1/seed42 result: test F1 0.853032; [audit and next steps](../../../reports/multimodal_architecture_ablation/dtd/BASELINE_01_FLORENCE_PREFIX_CANDIDATE_ONLY.md) |
| [02: Norm-calibrated fusion](02_candidate_only_norm_fusion/dtd/README.md) | Does normalizing projected candidate text help beyond the baseline's feature scale? | Saved split1/seed42 val/test F1 0.835824/0.849329; 50 epochs, best epoch38 |
| [03: Auxiliary m-LTM-inspired alignment](03_candidate_only_mltm_alignment/dtd/README.md) | Does explicit matching help beyond study02 normalization? | Saved split1/seed42 val/test F1 0.834947/0.848722; training-pool matching improves, classification not improved |
| [04: Aligned image-caption selector](04_candidate_only_semantic_selector/dtd/README.md) | Does query-image / candidate-text selection help after alignment? | Standalone 50-epoch template, no query text; separate selector/fusion validation interventions; no GPU metrics yet |
| Token-fusion branch | Does retaining visual/text tokens improve on vector addition? | Planned |
| Pooling branch | Does adaptive pooling help after token fusion? | Planned |
| Robustness | Does the validation-chosen variant also work with Qwen8B captions? | Planned |

Test these directions separately before combining them. Keep the Prefix CSV fixed during architecture comparisons.
Use validation for method selection; report independent final tests and multiple seeds when available.
Prior candidate_only and full_both scores are historical references, not results of this new baseline.

Read the [study02/03 saved-output review and study04 plan](../../../reports/multimodal_architecture_ablation/dtd/STUDY_02_03_REVIEW_AND_STUDY_04_PLAN.md).
Both completed runs' fixed-image validation text-off/shuffle F1 did not drop. Study03 is study04's structural control,
not a proven winner. Runtime Torch/NumPy versions and attachment paths differed between those saved runs.
