# DTD Qwen3-VL Captioner Comparison

`qwen3-vl-dtd-captions-8b-32b.ipynb` creates two complete image-derived caption datasets using frozen
Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct in one offline notebook. It does not fine-tune either model.

Attach DTD, both complete original floating-point Instruct checkpoints and the existing local CLIP ViT-B/32.
Set the exact checkpoint paths in the configuration cell. A native Qwen3-VL-compatible Transformers runtime
(>=4.57, with its dependencies) is required; the legacy Florence 4.49 offline wheels cannot run these models.
No network requests or package installations are made. Checkpoints are loaded sequentially in child processes;
caption/audit caches are atomic and resumable, and tokenizer parallelism is disabled before tokenizer imports.

Completed named exports under the notebook's `OUTPUT_ROOT / exports` are:

- `captions_qwen3vl_8b_instruct.csv`
- `captions_qwen3vl_32b_instruct.csv`

Each has all 5,640 DTD image IDs for the official split, including invalid-caption rows with explicit masks.
The original full responses, cleaned 5-8-word inputs, validity, lexical checks and diagnostic CLIP scores remain
auditable. Global class names never enter the VLM prompt. Policy is `dtd_lexical_v2`; class-CLIP/alias diagnostics
do not silently change the availability mask. These generated descriptions are not human ground truth.

Run separate copies of DTD text-context ablation notebook 05 `full_both` with each CSV and fresh output/cache
directories. Instructions are embedded at the end of the generation notebook. Keep 50 epochs, K=1, fixed splits,
pool, seeds and classifier hyperparameters. Do not substitute these CSVs into the four Florence-method notebooks
that are intentionally bound to different hashes. No downstream F1/accuracy is available until classifiers run.

Builder: `tools/notebook_builders/build_qwen3vl_dtd_caption_notebook.py`.
Worker/helpers: `src/captioning/dtd_qwen_caption.py`.
The generated notebook embeds all project helpers and needs no repository imports on Kaggle.
