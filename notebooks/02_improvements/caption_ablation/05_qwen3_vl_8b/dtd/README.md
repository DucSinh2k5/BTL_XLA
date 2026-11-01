# Qwen3-VL-8B Captions + DTD Full-Both

This notebook copies the original DTD notebook 05 full_both classifier/training and changes only the caption
input binding, provenance checks and artifact names. It does not generate captions or load Qwen/Florence weights.

Attach the complete audited Qwen8B `captions.csv` (or `captions_qwen3vl_8b_instruct.csv`), DTD, local DINOv3
code/ViT-L/16 weights and the same CLIP ViT-B/32 folder. Upload the notebook directly to Kaggle; no project imports
are required. Run all cells offline. Model/optimizer/train/evaluation code is unchanged from notebook 05.

Set `CAPTIONS_INPUT_PATH` in the configuration cell to the exact attached CSV path. Blank means discover a unique
matching Qwen8B CSV by its generation metadata. The file must contain all 5,640 official split-1 image IDs, audit
scores, natural masks and generation fingerprints from the Qwen caption notebook. A raw/partial, Florence or 32B
CSV is rejected. Multiple different Qwen8B inputs require an explicit choice. No expected file hash is invented;
the actual SHA256 is computed, saved and locked when you run the notebook.

Defaults: DTD split 1, pool/training seed 42, 50 epochs, K=1, 376 train-only candidates, both text fusion gates and
visual+semantic selection. `dtd_lexical_v2` stays fixed; CLIP class scores and aliases are diagnostic-only. Invalid
captions remain zero-masked, not fabricated. Compare natural caption coverage along with downstream metrics.

Output root is `outputs/mtacs_dtd_split1_caption_05_qwen3vl_8b_vitl16/`. Resume only this method's own unfinished
run. Existing Florence outputs/caches are never overwritten or adopted. Select best checkpoints by validation loss;
test opens after the declared epoch budget. GPU/batch policy remains the original ViT-L/16 policy for comparability.

Important artifacts: `caption_method.json`, `protocol.json`, `shared/caption_model_inputs.csv`,
`shared/caption_role_coverage.csv`, `runs/full_both/seed_42/best.pt`, per-run metrics/predictions/history,
`caption_05_qwen3vl_8b_results.csv`, the report and training plot. No Qwen8B F1/accuracy has been measured locally.
