# Qwen3-VL-32B Captions + DTD Full-Both

This is another copy of the original notebook 05 full_both classifier, paired with the Qwen8B caption experiment.
Only the accepted caption-method metadata, input/output variables, hash/provenance binding and report names differ.
The folder's `06` denotes the sixth caption-method experiment, NOT a different text-role architecture: full_both
is still the original notebook-05 architecture, not full_shuffle_both.

Attach the complete audited Qwen32B `captions.csv` (or `captions_qwen3vl_32b_instruct.csv`), DTD, the local DINOv3
code/ViT-L/16 weights and the same CLIP ViT-B/32 folder. No Qwen32B/8B or Florence weights are needed to train this
classifier. Upload this notebook directly to Kaggle; all helpers are embedded and all asset loading is offline.

Set `CAPTIONS_INPUT_PATH` in the configuration cell to the exact attached 32B CSV path. Blank means discover a
unique complete Qwen32B export by metadata, including files simply named `captions.csv`. 8B, Florence, partial,
mixed or malformed exports are rejected. The actual uploaded CSV's SHA256 is computed and locked on first use;
no unknown input hash or result is prefilled. No caption regeneration or package installation is performed.

Defaults match the 8B experiment: official DTD split 1, pool/training seed 42, 50 epochs, K=1, 376 train-only
candidates, both text fusion gates and visual+semantic selection. Keep GPU/batch, optimizer and preprocessing
matched when comparing results. Policy stays `dtd_lexical_v2`; class-CLIP and alias scores are diagnostic-only.
Invalid text is zero-masked while its image remains available. Caption quality/leakage claims need separate evidence.

Outputs are isolated in `outputs/mtacs_dtd_split1_caption_06_qwen3vl_32b_vitl16/`.
The run cannot silently adopt 8B checkpoints, CLIP caches or caption files. Start fresh, then resume only its own
unfinished run with the same CSV/protocol. Best checkpoint selection uses validation loss; test opens after 50 epochs.

Artifacts include `caption_method.json`, `protocol.json`, `shared/caption_model_inputs.csv`, caption-role coverage,
`runs/full_both/seed_42/best.pt`, metrics/predictions/history, `caption_06_qwen3vl_32b_results.csv`, report and plot.
This notebook does not contain measured 32B-caption classification accuracy/F1 until training is actually run.

Builder: `tools/notebook_builders/build_dtd_caption_qwen32b_full_both_notebook.py`. It derives the standalone
32B notebook from the existing 8B builder without changing or overwriting the 8B notebook or its source helpers.
