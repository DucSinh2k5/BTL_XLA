# 01: Candidate-Only With Florence-Prefix

[Open notebook](mtacs-dtd-vitl16-prefix-candidate-only-baseline.ipynb)

Saved run received 2026-10-25: validation Macro-F1 0.840162, test Macro-F1 0.853032, best epoch 38.
See the [saved-output audit and next-step plan](../../../../../reports/multimodal_architecture_ablation/dtd/BASELINE_01_FLORENCE_PREFIX_CANDIDATE_ONLY.md).
This is one split/seed, not a significance claim; the template below remains unexecuted and historical artifacts are unchanged.

Input flow: query image -> image-only selector -> one candidate image -> gather that candidate's
CLIP caption vector -> visual/context-text gated fusion -> 47-class prediction.
Query captions and semantic selection are disabled. The learned projection/gate from the old candidate_only
model is retained; no new alignment, Partial OT, token resampler or ASWP is enabled.

## Kaggle Inputs

| Input | Required contents |
| --- | --- |
| DTD | `dtd/images/` and official `dtd/labels/train1.txt`, `val1.txt`, `test1.txt` |
| DINOv3 code | Complete local repository, including `hubconf.py` and the `dinov3` package |
| DINOv3 ViT-L/16 | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`; not ViT-B or ViT-S weights |
| CLIP ViT-B/32 | Complete local Hugging Face model/tokenizer folder with text projection; 512-dimensional text embeddings |
| Florence-Prefix CSV | Exact [captions_prefix_control.csv](../../../../../data/captions/dtd/grounded_controls/captions_prefix_control.csv), 5,640 rows |

No Florence/Qwen model, references CSV or legacy Transformers wheels are needed. Caption generation is disabled.
The notebook uses local files only and does not install/download packages. Its first cell checks required packages.
Upload the notebook plus the inputs above, enable GPU, keep Internet off and run cells from top to bottom.

## Paths To Check

- Configuration cell: `CAPTIONS_INPUT_PATH` can remain blank if the attached file is named `captions_prefix_control.csv`.
  Discovery also verifies its SHA256. For a renamed copy, set this variable to the exact uploaded CSV path.
- Configuration cell: `CLIP_MODEL_LOCAL_PATH` defaults to `/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit`.
- Data/settings cell: `DATASET_ROOT` defaults to `/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd`.
- Data/settings cell: set `DINOV3_REPO_DIR` to the folder containing `hubconf.py`, and `DINOV3_WEIGHTS` to the exact `.pth` file.
  Check these when attaching a differently named DINOv3 dataset.

Expected CSV SHA256: `b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011`.
Use the original file bytes; opening and re-exporting the CSV can change its hash.
Wrong-method CSVs or conflicting cached CSVs are rejected instead of silently substituted.

## Training And Outputs

Default: official split 1, pool/training seed 42, pool 376/1,880 training images, K=1,
ViT-L/16 at 384x384, 50 epochs, no early stopping, best validation-loss checkpoint.
Training queries include pool members, but the identical image is excluded from its own context selection.
Batch size follows the existing GPU-memory policy: 32/16/8/4 for >=80/>=40/>=20/<20 GiB.
Match the GPU/batch configuration in later experiments. `q_gate=0` is expected; `c_gate` is learned
for selected captions whose validity mask is true. Invalid captions stay masked; their images remain candidates.

Output root:
`/kaggle/working/outputs/mtacs_dtd_split1_architecture_01_prefix_candidate_only_vitl16/`

- `architecture_01_prefix_candidate_only_results.csv`: validation/test metrics with method flags and CSV hash.
- `architecture_01_prefix_candidate_only_report.md` and `architecture_01_prefix_candidate_only_training.png`.
- `caption_method.json`, `architecture_study.json`, `protocol.json`, `environment.json` and experiment/test plans.
- `shared/captions.csv`: byte-identical copy; caption audits, actual model text/masks and cached pool embeddings.
- `runs/candidate_only/seed_42/`: best checkpoint, history, status, predictions, metrics and caption assignments.

An unfinished run retains `last.pt` for resume; after completion the default cleanup keeps `best.pt`.
Preserve the complete output folder and resume only its own compatible checkpoint, not an old full_both run.
The final test waits for the declared epoch budget; set `RUN_FINAL_TEST=False` during validation-only development.

The older candidate_only F1 0.850321 and Prefix full_both F1 0.810651 are not results of this baseline.
No GPU metrics have been inserted. Export the executed notebook and full outputs after training to establish this row.
