# Improvement Experiments

All groups below are further divided by dataset; original notebook names are preserved.

| Group | Purpose |
| --- | --- |
| `query_text` | Query caption fusion; visual candidate image |
| `query_candidate_text` | Query and selected context caption fusion, SUN397 K=1 |
| `multimodal_top3` | Separate K=3 experiments; selector-text and context-text-fusion variants |
| `adaptive_context` | Adaptive/diverse visual context and backbone variants |
| `visual_top3` | Three visual contexts, no caption improvement claim |
| `zero_shot_siglip` | SigLIP zero-shot fusion experiment |
| `text_context_ablation` | Eight standalone DTD controls, 50 epochs |
| `caption_generation` | Decoder LoRA, grounded reranking, or paired frozen Qwen3-VL 8B/32B caption pipelines |
| `caption_ablation` | Caption CSV comparisons, including Florence controls and Qwen8B/32B, with the same full_both classifier |
| [Multimodal architecture ablation](multimodal_architecture_ablation/README.md) | Fixed-CSV DTD architecture study; baseline 01 is candidate_only with Florence-Prefix, 50 epochs |
| [Segmentation full_both](segmentation/full_both/qwen3_vl_8b/isic2017/README.md) | ISIC 2017, 448: integrated Qwen3-VL-8B caption generation, CLIP text encoding, query/candidate fusion and multimodal selector |
| [Segmentation Florence-Prefix](segmentation/full_both/florence_prefix/isic2017/README.md) | Same ISIC full_both control with frozen Florence-Prefix captions; isolated offline legacy runtime; no new metrics yet |

The caption input files live in `data/captions/dtd/`; they are not mixed with notebooks.
The caption-method notebooks take precomputed CSV inputs and do not regenerate captions.
DTD notebooks 02-08 in text-context ablation can create/resume captions with local offline assets.
The visual-only notebook does not require captions. Do not change masks/thresholds using test performance.

See [all notebooks](../README.md) and [experiment protocol](../../docs/protocols/EXPERIMENT_PROTOCOL.md).
