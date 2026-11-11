# Notebook Catalog

Paths are relative to the TACS project root. Initial migration filenames and notebook bytes are preserved.
Six later caption-method result imports were renamed without changing their bytes;
see the [executed-results index](../experiments/completed/caption_ablation/dtd/README.md)
for original/new names, metrics and preservation hashes.
`research_variant` means an implementation exists, NOT that it reproduces the published protocol/results.
`cells_with_outputs` records saved output presence, not a claim that training completed successfully.
Read configuration evidence in [notebook_catalog.json](../manifests/notebook_catalog.json).
Empty method/dataset folders contain only `.gitkeep`; they are planned slots, not experiments.


Later notebook imports and duplicate handling are recorded in [Imported Notebook Reorganization](IMPORTS_2026-11-04.md).

## Quick Start

- ISIC segmentation **ViT-L/16, 448**: [TACS visual và M-TACS full_both train/export cho web](04_deployment/isic2017/README.md), bản M-TACS tự tạo/resume caption Florence-Prefix.
- Train SUN397 visual TACS **ViT-L/16** and export real web inference/attention assets: [deployment notebook 1](04_deployment/sun397/01_tacs_visual/README.md).
- Train SUN397 **M-TACS full_both ViT-L/16**, generate captions and export Florence/CLIP plus web inference/attention assets: [deployment notebook 2](04_deployment/sun397/02_mtacs_full_both/README.md).
- New DTD text-context experiment: choose one standalone notebook under `02_improvements/text_context_ablation/`.
- Compare caption CSVs (four Florence methods and two Qwen methods): choose the matching template under `02_improvements/caption_ablation/`.
- Start the fixed-CSV architecture study: [candidate_only + Florence-Prefix baseline](02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/README.md).
- Next architecture control: [candidate_only + norm-calibrated fusion](02_improvements/multimodal_architecture_ablation/02_candidate_only_norm_fusion/dtd/README.md), with validation text off/shuffle and gate/norm diagnostics.
- Auxiliary matching experiment: [candidate_only + norm fusion + m-LTM-inspired alignment](02_improvements/multimodal_architecture_ablation/03_candidate_only_mltm_alignment/dtd/README.md), training-only own-image/caption pairs; no query text or semantic selector.
- Context-selection experiment: [study04 aligned query-image/candidate-caption selector](02_improvements/multimodal_architecture_ablation/04_candidate_only_semantic_selector/dtd/README.md), same Prefix CSV/50 epochs; no query caption and separate selector/fusion diagnostics.
- Read the six executed caption-method runs: [completed caption ablation](../experiments/completed/caption_ablation/dtd/README.md).
- Caption generation/training: `02_improvements/caption_generation/`.
- ISIC full_both segmentation with integrated Qwen8B captions: [standalone notebook and inputs](02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/README.md).
- Alternate ISIC caption backend, keeping the same full_both model: [Florence-Prefix notebook and offline inputs](02_improvements/segmentation/full_both/florence_prefix/isic2017/README.md).
- Read actual eight-run outputs: `../experiments/completed/text_context_ablation/dtd/`.
- Archived combined DTD suites remain builder dependencies; do not delete them.

## All Notebooks

| Dataset | Method | Status | Output cells | Notebook |
| --- | --- | --- | ---: | --- |
| dtd | full_both_with_decoder_lora_captions | saved_execution | 7 | [caption-cai-tien.ipynb](<../experiments/completed/caption_decoder_lora/dtd/caption-cai-tien.ipynb>) |
| dtd | both_fusion | saved_execution | 7 | [both-fusion.ipynb](<../experiments/completed/text_context_ablation/dtd/both-fusion.ipynb>) |
| dtd | candidate_only | saved_execution | 7 | [candidate-only.ipynb](<../experiments/completed/text_context_ablation/dtd/candidate-only.ipynb>) |
| dtd | full_both | saved_execution | 7 | [full-both.ipynb](<../experiments/completed/text_context_ablation/dtd/full-both.ipynb>) |
| dtd | full_redacted | saved_execution | 7 | [full-redacted.ipynb](<../experiments/completed/text_context_ablation/dtd/full-redacted.ipynb>) |
| dtd | full_shuffle_both | saved_execution | 7 | [full-shuffle-both.ipynb](<../experiments/completed/text_context_ablation/dtd/full-shuffle-both.ipynb>) |
| dtd | full_unscreened | saved_execution | 7 | [full-unscreened.ipynb](<../experiments/completed/text_context_ablation/dtd/full-unscreened.ipynb>) |
| dtd | query_only | saved_execution | 7 | [query-only.ipynb](<../experiments/completed/text_context_ablation/dtd/query-only.ipynb>) |
| dtd | visual_only | saved_execution | 6 | [visual-only.ipynb](<../experiments/completed/text_context_ablation/dtd/visual-only.ipynb>) |
| stanford_cars | feature_averaged_context | research_variant | 0 | [baseline_feature_averaged_context_cars.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/feature_averaged_context/stanford_cars/baseline_feature_averaged_context_cars.ipynb>) |
| stanford_cars | frozen_dino_similarity | research_variant | 7 | [ban-cuoi-frozen-dino-car.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/frozen_dino_similarity/stanford_cars/ban-cuoi-frozen-dino-car.ipynb>) |
| stanford_cars | frozen_dino_similarity | research_variant | 0 | [baseline_frozen_dino_similarity_ret_cars.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/frozen_dino_similarity/stanford_cars/baseline_frozen_dino_similarity_ret_cars.ipynb>) |
| stanford_cars | frozen_dino_similarity | research_variant | 0 | [notebook7dcd2fd8f0.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/frozen_dino_similarity/stanford_cars/notebook7dcd2fd8f0.ipynb>) |
| cub200 | no_context | research_variant | 14 | [baseline_no_tacs_image_classification (seed42).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification (seed42).ipynb>) |
| cub200 | no_context | research_variant | 14 | [baseline_no_tacs_image_classification (seed43).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification (seed43).ipynb>) |
| cub200 | no_context | research_variant | 14 | [baseline_no_tacs_image_classification.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification.ipynb>) |
| cub200 | no_context | research_variant | 15 | [baseline_no_tacs_image_classification_(seed_44).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification_(seed_44).ipynb>) |
| stanford_cars | no_context | research_variant | 13 | [seed44-car.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/stanford_cars/seed44-car.ipynb>) |
| sun397 | no_context | research_variant | 0 | [baseline_no_tacs_cars.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/sun397/baseline_no_tacs_cars.ipynb>) |
| stanford_cars | random_context | research_variant | 8 | [ban-cuoi-Random Context Retrieval baseline.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/random_context/stanford_cars/ban-cuoi-Random Context Retrieval baseline.ipynb>) |
| stanford_cars | random_context | research_variant | 0 | [ban-cuoi-random-context-ret-car.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/random_context/stanford_cars/ban-cuoi-random-context-ret-car.ipynb>) |
| stanford_cars | random_context | research_variant | 10 | [baseline_random_context_ret_cars.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/random_context/stanford_cars/baseline_random_context_ret_cars.ipynb>) |
| stanford_cars | tacs | research_variant | 8 | [ban-cuoi-frozen-dino-car.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/ban-cuoi-frozen-dino-car.ipynb>) |
| stanford_cars | tacs | research_variant | 0 | [ban-cuoi-tacs-car.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/ban-cuoi-tacs-car.ipynb>) |
| stanford_cars | tacs | research_variant | 0 | [baseline_tacs_cars.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/baseline_tacs_cars.ipynb>) |
| stanford_cars | tacs | research_variant | 0 | [notebook05aceb61b4.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/notebook05aceb61b4.ipynb>) |
| stanford_cars | tacs | research_variant | 0 | [tacs-chuanpaper-car-b16.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/tacs-chuanpaper-car-b16.ipynb>) |
| stanford_cars | tacs | research_variant | 0 | [tacs-chuanpaper-car-l16.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/tacs-chuanpaper-car-l16.ipynb>) |
| stanford_cars | tacs | research_variant | 7 | [tacs-chuanpaper-sun-b16 (2).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/tacs-chuanpaper-sun-b16 (2).ipynb>) |
| sun397 | tacs | research_variant | 0 | [ban-cuoi-tacs-sun.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/ban-cuoi-tacs-sun.ipynb>) |
| sun397 | tacs | research_variant | 7 | [tacs-chuanpaper-sun-b16 (1).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-chuanpaper-sun-b16 (1).ipynb>) |
| sun397 | tacs | research_variant | 8 | [tacs-sun-da-sua (1).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-sun-da-sua (1).ipynb>) |
| sun397 | tacs | research_variant | 8 | [tacs-sun-da-sua (2).ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-sun-da-sua (2).ipynb>) |
| sun397 | tacs | research_variant | 0 | [tacs-sun-true-val-split.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-sun-true-val-split.ipynb>) |
| sun397 | tacs | research_variant | 0 | [tacs-vitb16-sun-true-val-split.ipynb](<../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-vitb16-sun-true-val-split.ipynb>) |
| sun397 | blank_image | research_variant | 0 | [table4_blank_image_sun.ipynb](<../notebooks/01_paper_reproduction/classification/table_04_context_pairs/blank_image/sun397/table4_blank_image_sun.ipynb>) |
| sun397 | duplicate_query | research_variant | 0 | [table4_duplicate_query_sun.ipynb](<../notebooks/01_paper_reproduction/classification/table_04_context_pairs/duplicate_query/sun397/table4_duplicate_query_sun.ipynb>) |
| sun397 | noisy_query | research_variant | 0 | [table4_noisy_query_sun.ipynb](<../notebooks/01_paper_reproduction/classification/table_04_context_pairs/noisy_query/sun397/table4_noisy_query_sun.ipynb>) |
| sun397 | tacs_attention_checkpoint_export | research_variant | 0 | [tacs_train_with_checkpoints.ipynb](<../notebooks/01_paper_reproduction/figure_04_attention/tacs/sun397/tacs_train_with_checkpoints.ipynb>) |
| isic2017 | tacs_segmentation | research_variant | 0 | [tacs_isic2017_segmentation.ipynb](<../notebooks/01_paper_reproduction/segmentation/table_03_main_results/tacs/isic2017/tacs_isic2017_segmentation.ipynb>) |
| sun397 | adaptive_context | research_variant | 0 | [tacs-adaptive-sun-dinov3-vitb-l.ipynb](<../notebooks/02_improvements/adaptive_context/sun397/tacs-adaptive-sun-dinov3-vitb-l.ipynb>) |
| sun397 | adaptive_context | research_variant | 8 | [tacs-adaptive-sun.ipynb](<../notebooks/02_improvements/adaptive_context/sun397/tacs-adaptive-sun.ipynb>) |
| sun397 | adaptive_context | research_variant | 0 | [tacs_adaptive_diverse_context_sun.ipynb](<../notebooks/02_improvements/adaptive_context/sun397/tacs_adaptive_diverse_context_sun.ipynb>) |
| dtd | florence_prefix | standalone_template | 0 | [mtacs-train-vitl16-dtd-caption-ablation-01-florence-prefix.ipynb](<../notebooks/02_improvements/caption_ablation/01_florence_prefix/dtd/mtacs-train-vitl16-dtd-caption-ablation-01-florence-prefix.ipynb>) |
| dtd | florence_firstvalid | standalone_template | 0 | [mtacs-train-vitl16-dtd-caption-ablation-02-florence-firstvalid.ipynb](<../notebooks/02_improvements/caption_ablation/02_florence_firstvalid/dtd/mtacs-train-vitl16-dtd-caption-ablation-02-florence-firstvalid.ipynb>) |
| dtd | florence_singlegrounded | standalone_template | 0 | [mtacs-train-vitl16-dtd-caption-ablation-03-florence-singlegrounded.ipynb](<../notebooks/02_improvements/caption_ablation/03_florence_singlegrounded/dtd/mtacs-train-vitl16-dtd-caption-ablation-03-florence-singlegrounded.ipynb>) |
| dtd | florence_multigrounded | standalone_template | 0 | [mtacs-train-vitl16-dtd-caption-ablation-04-florence-multigrounded.ipynb](<../notebooks/02_improvements/caption_ablation/04_florence_multigrounded/dtd/mtacs-train-vitl16-dtd-caption-ablation-04-florence-multigrounded.ipynb>) |
| dtd | caption_generation/decoder_lora | research_variant | 0 | [florence2-dtd-caption-finetune.ipynb](<../notebooks/02_improvements/caption_generation/decoder_lora/dtd/florence2-dtd-caption-finetune.ipynb>) |
| dtd | caption_generation/grounded_rerank | research_variant | 0 | [florence2-dtd-caption-grounded-rerank.ipynb](<../notebooks/02_improvements/caption_generation/grounded_rerank/dtd/florence2-dtd-caption-grounded-rerank.ipynb>) |
| sun397 | multimodal_top3/context_text_fusion | research_variant | 0 | [mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb](<../notebooks/02_improvements/multimodal_top3/context_text_fusion/sun397/mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb>) |
| sun397 | multimodal_top3/selector_text_only | research_variant | 0 | [mtacs-train-captions-vitl16-multimodal-top3.ipynb](<../notebooks/02_improvements/multimodal_top3/selector_text_only/sun397/mtacs-train-captions-vitl16-multimodal-top3.ipynb>) |
| sun397 | query_candidate_text | research_variant | 0 | [mtacs-train-captions-vitl16-context-text.ipynb](<../notebooks/02_improvements/query_candidate_text/sun397/mtacs-train-captions-vitl16-context-text.ipynb>) |
| dtd | query_text | research_variant | 0 | [mtacs-train-captions-vitl16-dtd.ipynb](<../notebooks/02_improvements/query_text/dtd/mtacs-train-captions-vitl16-dtd.ipynb>) |
| sun397 | query_text | research_variant | 0 | [mtacs-train-captions-vitl16.ipynb](<../notebooks/02_improvements/query_text/sun397/mtacs-train-captions-vitl16.ipynb>) |
| sun397 | query_text | research_variant | 9 | [mtacs-train-captions.ipynb](<../notebooks/02_improvements/query_text/sun397/mtacs-train-captions.ipynb>) |
| sun397 | query_text | research_variant | 0 | [mtacs_sun397_visual_text_context.ipynb](<../notebooks/02_improvements/query_text/sun397/mtacs_sun397_visual_text_context.ipynb>) |
| sun397 | query_text | research_variant | 0 | [mtacs_sun397_visual_text_context_offline.ipynb](<../notebooks/02_improvements/query_text/sun397/mtacs_sun397_visual_text_context_offline.ipynb>) |
| dtd | visual_only | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-01-visual-only.ipynb](<../notebooks/02_improvements/text_context_ablation/01_visual_only/dtd/mtacs-train-vitl16-dtd-ablation-01-visual-only.ipynb>) |
| dtd | query_only | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-02-query-only.ipynb](<../notebooks/02_improvements/text_context_ablation/02_query_only/dtd/mtacs-train-vitl16-dtd-ablation-02-query-only.ipynb>) |
| dtd | candidate_only | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-03-candidate-only.ipynb](<../notebooks/02_improvements/text_context_ablation/03_candidate_only/dtd/mtacs-train-vitl16-dtd-ablation-03-candidate-only.ipynb>) |
| dtd | both_fusion | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-04-both-fusion.ipynb](<../notebooks/02_improvements/text_context_ablation/04_both_fusion/dtd/mtacs-train-vitl16-dtd-ablation-04-both-fusion.ipynb>) |
| dtd | full_both | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb](<../notebooks/02_improvements/text_context_ablation/05_full_both/dtd/mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb>) |
| dtd | full_shuffle_both | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-06-full-shuffle-both.ipynb](<../notebooks/02_improvements/text_context_ablation/06_full_shuffle_both/dtd/mtacs-train-vitl16-dtd-ablation-06-full-shuffle-both.ipynb>) |
| dtd | full_unscreened | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-07-full-unscreened.ipynb](<../notebooks/02_improvements/text_context_ablation/07_full_unscreened/dtd/mtacs-train-vitl16-dtd-ablation-07-full-unscreened.ipynb>) |
| dtd | full_redacted | standalone_template | 0 | [mtacs-train-vitl16-dtd-ablation-08-full-redacted.ipynb](<../notebooks/02_improvements/text_context_ablation/08_full_redacted/dtd/mtacs-train-vitl16-dtd-ablation-08-full-redacted.ipynb>) |
| sun397 | visual_top3 | research_variant | 0 | [tacs_top3_context_sun.ipynb](<../notebooks/02_improvements/visual_top3/sun397/tacs_top3_context_sun.ipynb>) |
| sun397 | zero_shot_siglip | research_variant | 0 | [siglip2_zero_shot_tacs_fusion_sun397.ipynb](<../notebooks/02_improvements/zero_shot_siglip/sun397/siglip2_zero_shot_tacs_fusion_sun397.ipynb>) |
| dtd | combined_ablation_suite | builder_dependency | 0 | [mtacs-train-captions-vitl16-dtd-text-ablation.ipynb](<../notebooks/90_archive/dtd/combined_ablation_suite/mtacs-train-captions-vitl16-dtd-text-ablation.ipynb>) |
| dtd | combined_ablation_suite | builder_dependency | 3 | [notebookb7a9dba884.ipynb](<../notebooks/90_archive/dtd/combined_ablation_suite/notebookb7a9dba884.ipynb>) |
| dtd | combined_ablation_100_epochs | archived | 11 | [notebookb7a9dba884.before-50-epochs.ipynb](<../notebooks/90_archive/dtd/historical_versions/notebookb7a9dba884.before-50-epochs.ipynb>) |
| isic2017 | historical | archived | 0 | [tacs_isic2017_segmentation_before_paper_faithful.ipynb](<../notebooks/90_archive/isic2017/historical_versions/tacs_isic2017_segmentation_before_paper_faithful.ipynb>) |
| stanford_cars | historical | archived | 9 | [notebook05aceb61b4_before_tacs_fix.ipynb](<../notebooks/90_archive/stanford_cars/historical_versions/notebook05aceb61b4_before_tacs_fix.ipynb>) |
| stanford_cars | historical | archived | 0 | [notebook7dcd2fd8f0_before_context_prior_fulltrain_epoch100.ipynb](<../notebooks/90_archive/stanford_cars/historical_versions/notebook7dcd2fd8f0_before_context_prior_fulltrain_epoch100.ipynb>) |
| stanford_cars | historical | archived | 0 | [notebook7dcd2fd8f0_before_overfit_regularization.ipynb](<../notebooks/90_archive/stanford_cars/historical_versions/notebook7dcd2fd8f0_before_overfit_regularization.ipynb>) |
| stanford_cars | historical | archived | 0 | [notebook7dcd2fd8f0_before_topk_attention_optimization.ipynb](<../notebooks/90_archive/stanford_cars/historical_versions/notebook7dcd2fd8f0_before_topk_attention_optimization.ipynb>) |
| stanford_cars | historical | archived | 8 | [notebook7dcd2fd8f0_before_val_split.ipynb](<../notebooks/90_archive/stanford_cars/historical_versions/notebook7dcd2fd8f0_before_val_split.ipynb>) |
| sun397 | historical | archived | 0 | [ban-cuoi-tacs-sun_before_full_candidate_pool.ipynb](<../notebooks/90_archive/sun397/historical_versions/ban-cuoi-tacs-sun_before_full_candidate_pool.ipynb>) |
| sun397 | historical | archived | 6 | [ban-cuoi-tacs-sun_before_selector_paper_faithful.ipynb](<../notebooks/90_archive/sun397/historical_versions/ban-cuoi-tacs-sun_before_selector_paper_faithful.ipynb>) |

## Misleading Original Names

- `SUN/table2/baseline_no_tacs_cars.ipynb`: **sun397 / no_context**. Paper-related implementation; not a certification of matching the paper protocol. Original filename/folder is misleading; dataset follows notebook configuration.
- `SUN/table2/ban-cuoi-Random Context Retrieval baseline.ipynb`: **stanford_cars / random_context**. Paper-related implementation; not a certification of matching the paper protocol. Original filename/folder is misleading; dataset follows notebook configuration.
- `SUN/table2/ban-cuoi-frozen-dino-car.ipynb`: **stanford_cars / tacs**. Paper-related implementation; not a certification of matching the paper protocol. Original filename/folder is misleading; dataset follows notebook configuration. Learned TACS selector, not the frozen retriever suggested by its filename.
- `tacs-chuanpaper-sun-b16 (2).ipynb`: **stanford_cars / tacs**. Paper-related implementation; not a certification of matching the paper protocol. Original filename/folder is misleading; dataset follows notebook configuration.
