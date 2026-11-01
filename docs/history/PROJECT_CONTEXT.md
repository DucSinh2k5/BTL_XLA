# M-TACS Project Context

Updated: 2026-10-18, Asia/Saigon. Scope: the SUN397/DTD M-TACS work inside
`F:/Documents/CODE/Python/cv_project/XLA`, not the unrelated coursework and datasets here.
Evidence: the current conversation, source/builders, generated notebooks, tests and local artifacts.
This is a handover record, not proof that Kaggle training has completed.

## Current Stopping Point

- USER DECISION: run the eight DTD ablations with the existing captions first;
  inspect results before implementing caption-quality improvements.
- VERIFIED: all eight standalone notebooks and their builders exist and match;
  their saved code parses, default budgets are 50 epochs, and saved outputs are empty.
- USER-REPORTED: experiments were run externally on Kaggle; notebook 08 has now been supplied.
- VERIFIED: eight executed notebooks are supplied in `output_ablation/`, with completed
  50-epoch status and saved validation/test metrics. Original generated notebooks remain unexecuted.
- UNKNOWN: multiple-seed variance and exact per-run CSV/weight/pool hashes.
  Only notebooks, not the complete exported run directories, have been supplied so far.
- Do not regenerate captions, alter policies/seeds/budgets, overwrite checkpoints or
  start a new captioner experiment during this comparison without a new request.
- Eight-run assessment: `output_ablation/analysis/DANH_GIA_ABLATION_DTD_01_07.md` (original name
  retained for link continuity). Next: verify original artifacts, especially CSV identity for 07/08,
  then decide on seed confirmation and later caption improvements using validation.

## Architecture and Scope

- Classification research extends visual task-aligned context selection with captions.
  These are project variants, not an exact reproduction of the original paper.
- Florence-2 generates captions from image pixels. The current captioner has NOT been
  fine-tuned on curated DTD captions; generated text is pseudo-caption data, not human ground truth.
- Frozen CLIP ViT-B/32 embeds captions. DINOv3 ViT-L/16 supplies image features in the DTD runs.
- A fixed pool contains 20% of training images. Pool images can also be training queries;
  selection excludes the query's own image ID. Held-out images/captions never enter the pool.
- One context image is selected per query. When candidate text is enabled, the same selected
  image index gathers its caption. Query/candidate labels are not captioner or classifier inputs;
  training query labels supervise the classification loss.
- Fusion uses visual features plus independently masked, learned query/candidate text gates.
  Disabled/unavailable text has a zero contribution; its image remains usable.
- The multimodal selector adds standardized caption similarity to standardized visual scores:
  `visual_z + sigmoid(semantic_mix_logit) * semantic_z`. Initial coefficient: 0.5.
  Invalid text pairs contribute zero semantic evidence. Train uses policy sampling; eval uses argmax.

## Eight Standalone DTD Experiments

Notebook filename prefix: `mtacs-train-vitl16-dtd-ablation-`.

| Suffix | Configuration | Query fusion | Candidate fusion | Semantic selector | Caption view |
| --- | --- | --- | --- | --- | --- |
| `01-visual-only.ipynb` | visual_only | Off | Off | Off | Disabled; no caption generation required |
| `02-query-only.ipynb` | query_only | On | Off | Off | Screened |
| `03-candidate-only.ipynb` | candidate_only | Off | On | Off | Screened |
| `04-both-fusion.ipynb` | both_fusion | On | On | Off | Screened |
| `05-full-both.ipynb` | full_both | On | On | On | Screened |
| `06-full-shuffle-both.ipynb` | full_shuffle_both | On | On | On | Screened, both roles shuffled with masks preserved |
| `07-full-unscreened.ipynb` | full_unscreened | On | On | On | Aligned pre-leakage-filter captions |
| `08-full-redacted.ipynb` | full_redacted | On | On | On | Aligned captions with class names/fixed aliases removed |

- Defaults: official DTD split 1, seed 42, 1,880 images per train/val/test partition,
  376 training candidates, 50-epoch LR horizon, no early stopping, best checkpoint by validation loss.
- `DTD_SPLIT=1..10` selects official partition files, not training seeds.
- Each standalone runs only its configuration and has a separate output/checkpoint directory:
  `outputs/mtacs_dtd_split1_ablation_0N_<configuration>_vitl16/`.
- Final test is guarded until the declared training budget completes. Resume uses the run's
  own identity/protocol. Do not substitute visual_only checkpoints for a text configuration.
- Keep split/pool, seed, original CSV, preprocessing, GPU-dependent batch configuration,
  epoch budget and checkpoint rule aligned across comparisons. One seed is exploratory.

## Caption Policy and Offline Generation

- Text notebooks default to `dtd_lexical_v2`. It removes generic openings, prefers `raw_caption`,
  falls back to `final_caption`, and retains at most eight normalized words (eligible length: 5-8).
- For screened experiments, direct class terms are rejected. Legacy semantic-only rejects with
  finite scores may be recovered; unexplained invalid captions are not automatically restored.
  CLIP leakage scores are diagnostic-only under this policy, not a calibrated leakage detector.
- Legacy CSV text/flags/scores are preserved. Old scores may refer to text before cleanup.
  Actual inputs/masks are exported to `shared/caption_model_inputs.csv` and `caption_audit.csv`.
- 07 uses availability only, ignoring leakage flags/scores/class terms. It does not feed long raw
  paragraphs: text construction still matches 05. 08 starts from the same pre-filter text and
  removes every class name and fixed `DTD_EXTRA_TERMS` alias, using the same vocabulary for all
  images, never the image's true label. It repeats deletion if a compound alias is exposed.
- 08 accepts 1-4 surviving words after redaction; empty text is masked, never refilled.
  Vocabulary, before/after text and coverage changes are exported for inspection.
- Compare 07/08 coverage as well as scores. Neither redaction nor score differences prove
  absence/presence of illicit label leakage; DTD class words are also real texture descriptions.
- Original CSV creation/resume is embedded in 02-08. A complete compatible CSV skips Florence.
  Use the original audited `captions.csv`, not `caption_model_inputs.csv`, as reusable input.
- Kaggle paths previously established:
  - DTD: `/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd`
  - Legacy Florence: `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`
  - Offline assets: `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi`
  - CLIP: `/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit`
- Legacy Florence requires compatible Transformers 4.49.x dependencies. Embedded generation
  uses a separate subprocess and local wheels if necessary, not Internet or a kernel downgrade.
  Wheels must match Python/platform. Completed CSV batches are flushed for resuming.
- `TOKENIZERS_PARALLELISM=false` is set before tokenizer use; caption workers default to zero.
- Overrides include `DTD_CAPTIONS_INPUT_PATH`, `FLORENCE_MODEL_REF`, `DTD_OFFLINE_ASSETS_ROOT`,
  `CLIP_MODEL_LOCAL_PATH`, `DTD_CAPTION_INPUT_POLICY` and per-configuration epoch/output variables.

## Important Implementation Files

- `build_mtacs_dtd_{visual_only,query_only,candidate_only,both_fusion,full_both,full_shuffle_both,full_unscreened,full_redacted}_notebook.py`:
  build-time notebook generators. Exported notebooks do not import these local builder modules.
- `mtacs_dtd_ablation_cells.py`: shared model/training/evaluation helpers embedded by builders.
- `dtd_caption_screening.py`: cleanup and explicit input-policy helpers.
- `dtd_caption_generation_cells.py`: offline model/wheel discovery, generation worker and CSV resume.
- `notebookb7a9dba884.ipynb`: source for the older combined ablation suite. Prefer the standalone
  notebooks for the current eight runs; their later policy/view overrides are authoritative.
- `mtacs-train-captions-vitl16.ipynb`: historical SUN397 query-caption configuration, K=1.
- `mtacs-train-captions-vitl16-context-text.ipynb`: SUN397 query/context-caption extension, K=1.
  Top-3 prototypes are separate experiments and must not be silently substituted for K=1.
- `TONG_HOP_CAI_TIEN_MTACS_SO_VOI_TACS.md`: earlier research/history notes; predates the completed
  eight-notebook implementation. Historical claims are not current DTD run results.
- `tests/test_dtd_*.py`: static, mocked offline generation and synthetic CPU integration tests.

## Web Demo

- `sun397_demo/`: FastAPI backend with HTML/CSS/JavaScript frontend; classification only.
  `/` shows visual TACS; `/mtacs` shows query/candidate caption M-TACS.
- VERIFIED: no checkpoint/pool inference adapter is connected. Valid classification requests
  deliberately return HTTP 503 and null prediction/evidence. Captions and attention overlays
  remain empty, not fabricated. Existing unrelated `.pt` files are not automatically compatible.
- Still pending: connect the exact matching checkpoint, class mapping, transforms, Florence/CLIP,
  train-only candidate image/text bank and validated attention extraction. Segmentation is deferred.
- Local commands, after selecting the intended Python environment:
  `python -m pip install -r sun397_demo/requirements.txt`
  `python -m uvicorn sun397_demo.app:app --host 127.0.0.1 --port 8000`
  The handover did not install dependencies or start the server.

## Environment, Version Control and Validation

- Active root is a Git repository on `main`, tracking `origin/main`; one visible commit:
  `dc801c0` (2026-09-11, Initial commit). Remote name: origin; URL intentionally not copied.
- At handover start: three tracked files, 144 untracked status entries, no tracked/staged diffs.
  Most research files are untracked. A clean tracked diff DOES NOT mean the research is backed up.
  Preserve all files; no reset/clean, branch switch, commit or push without authorization.
- No applicable AGENTS.md was found in the workspace inventory or ancestor directories inspected.
- Current `python`: 3.10.11, torch 2.13.0 CPU-only, pytest 8.4.2; no CUDA, Transformers or globally
  installed scikit-learn. `tmp/dtd-ablation-test-deps` supplies existing local test dependencies.
  `.venv` exists but is a different environment; do not assume package parity.
- All eight saved notebooks were checked for valid Python syntax and equality with their builders,
  without rewriting or executing notebooks. Saved training outputs are empty.
- Handover baseline: 143 tests passed (133 DTD, 10 demo API) in 109.03 seconds.
  One Starlette/httpx TestClient deprecation warning; no test failures. No packages were changed.
- Caption-adaptation addition: 174 tests passed (31 new caption fine-tuning tests + the 143 baseline
  tests), using synthetic CPU models/data. Validated empty-target guard, train-only sampling/review,
  strict decoder LoRA scope, token-weighted accumulation, exact epoch-boundary resume in the CPU
  harness, checkpoint/CSV identities, offline subprocess setup and preservation of old notebooks.
  The nbformat package is unavailable locally; saved notebook JSON/Python syntax and deterministic
  builder output were checked instead. This is not a real Florence/PEFT GPU training validation.
- Equivalent targeted rerun from the project root (PowerShell):

```powershell
@'
import sys
from pathlib import Path
sys.path.insert(0, 'tmp/dtd-ablation-test-deps')
import pytest
tests = sorted(str(path) for path in Path('tests').glob('test_dtd_*.py'))
raise SystemExit(pytest.main(tests + ['sun397_demo/tests', '-q']))
'@ | python -X utf8 -
```

Tests use temporary synthetic artifacts. `pytest-of-Dell/` and test-created reports/checkpoints
are NOT Kaggle experiment results. No GPU training or model-quality improvement was verified locally.

## Results, Remaining Work and Next Action

- Historical user-reported SUN397 macro-F1: query-text ViT-L/16 0.8237, context-text experiment
  0.8108. Exact second configuration/logs are uncertain; do not infer a controlled candidate-text
  effect or declare either the best current architecture from these two numbers.
- DTD saved-output test macro-F1, split 1 / seed 42 / 50 epochs (verified 2026-10-18):
  visual_only 0.848168; query_only 0.823067; candidate_only 0.850321; both_fusion 0.810661;
  full_both 0.809124; full_shuffle_both 0.833392; full_unscreened 0.789774; full_redacted 0.811105.
  Candidate-only exceeds visual baseline by only four correct test images and has worse validation;
  do not claim a reliable improvement. Full variants do not beat baseline in this run.
  08 improves on 07 by 2.1331 F1 percentage points, but remains 3.7063 points below visual-only.
  Both use 100% caption availability. Redaction changes 580 train, 573 val, 570 test and 105 pool
  captions, empties none and leaves zero matches from the fixed class/alias vocabulary.
  Saved source matches builders except DINOv3 input paths; exact asset/CSV identity is not verified.
  Several runs, including 08, regenerated captions. Query-only stream shows epochs 1-41 while the completed
  table/plot report 50; request history.csv to verify the missing log lines.
- Collect each run's root `protocol.json`, `<configuration>_results.csv`,
  `runs/<configuration>/seed_42/history.csv`, available `val_metrics.json`/`test_metrics.json`,
  and `shared/caption_role_coverage.csv`. Add original CSV/hash and 07/08 comparison reports.
  Check actual filenames in the supplied output. Checkpoints are not needed for an initial table;
  add prediction CSVs for paired error analysis. Do not tune variants using repeated test feedback.
- New review-first caption adaptation pipeline (2026-10-18): user initially had NO approved
  targets and asked to create a review template first. `florence2-dtd-caption-finetune.ipynb`,
  built by `build_florence_dtd_caption_finetune_notebook.py` from `dtd_caption_finetune.py`,
  creates 188 train-only review rows with EMPTY targets and optional unverified Florence drafts.
  It blocks training until human-approved references/provenance exist (minimum 50 is a guard,
  not a data-sufficiency claim). Instructions: `HUONG_DAN_FINETUNE_CAPTION_DTD.md`.
- Subsequent user-authorized AI annotation: `caption_references.csv` now has 188 individually
  image-inspected teacher captions from `archive (23)/dtd/images`, all verified official train1.
  Original IDs/drafts/order/header preserved; 188 unique captions, all 7-8 words, no fixed class/root
  vocabulary matches. Existing structural guard passes and reference split is 141/47.
  IMPORTANT: these are AI-only teacher references, not independently human-reviewed gold targets.
  `reviewer_id=ducsinh` denotes the requesting/authorizing owner, not a claim of personal inspection;
  `approved` denotes completed AI annotation as requested. Exact backend model ID is unavailable,
  recorded honestly as `openai-codex:model-id-unavailable`. See `caption_references.annotation_notes.md`
  and `.annotation_report.json`; backup/image hashes/contact sheets live in
  `tmp/annotate_dtd_references/`. Do not report `human_reviewed_train_only` for these AI-only references
  unless a real human review is subsequently performed; current training summary's fixed origin label
  needs adjusting before an AI-only training run. No notebook/training code or real weights were changed
  during annotation, and no fine-tuning was run.
- Implemented but NOT trained on real assets: decoder-only Florence LoRA, frozen vision/base,
  5 captioner epochs, train-only reference fit/holdout, best holdout token NLL, epoch-boundary
  resume and new fingerprinted CSV export using dtd_lexical_v2. Transformer 4.49.x + PEFT 0.14.0
  run in an isolated offline worker. An offline PEFT wheel may need attaching; it was not found
  locally. Kernel dependencies and old eight-run artifacts were not modified. No GPU quality
  improvement has been verified. Teacher API generation and continual learning remain unimplemented.
- Text-only LLM rewriting cannot verify unseen image details. Synthetic captions stay synthetic
  unless independently checked; DTD classification labels are not caption ground truth.
  Captioner references must use official train only; choose using reference holdout, then freeze
  and regenerate captions/CLIP caches in a new M-TACS directory. New exports update BOTH raw_caption
  and final_caption because dtd_lexical_v2 prefers raw_caption. Keep the old comparison unchanged.

## Caption Experiment Update (2026-10-21)

- User supplied `caption-cai-tien.ipynb`, an executed downstream full_both classifier run using
  the decoder-LoRA CSV, not the Florence training notebook. Official split 1 / seed 42 / batch 32 /
  K=1 / 50 epochs; best validation-loss checkpoint epoch 39. Test macro-F1 0.806707, accuracy
  0.807447, loss 1.311028; validation macro-F1 0.796779. Old full_both test macro-F1 0.809124,
  accuracy 0.810106. The LoRA-caption run is lower by 0.2417 F1 percentage points and five correct
  test images. No reliable caption/downstream improvement is established. Earlier notes saying
  no real assets were trained describe local implementation-time verification, not this later
  user-supplied downstream result. Florence training history itself was not provided.
- New alternative requested: `florence2-dtd-caption-grounded-rerank.ipynb`, built by
  `build_florence_dtd_grounded_caption_notebook.py` from `dtd_grounded_caption.py`. Frozen original
  Florence + full image/text CLIP; no LoRA, training references or API. One greedy plus three beam
  hypotheses, extract within-sentence 5-8-word eligible spans, rank by image-text cosine to the
  actual image. Global class prompts provide audit scores only, not selection. Three beam texts
  are NOT three context images; matched downstream comparison keeps K=1.
- Policy remains dtd_lexical_v2, semantic class scores diagnostic-only; partial alias vocabulary
  diagnostic-only. Selected short text is written to BOTH raw_caption/final_caption; original
  long model output and span provenance remain in extra columns and per-image audit cache.
  No eligible span yields an explicit prefix fallback, never a fabricated replacement.
- New output root `outputs/florence2_dtd_split1_grounded_caption_v1/`. Main CSV `exports/captions.csv`;
  fresh prefix, first-valid-span and single-greedy-grounded controls under `controls/`. Only a
  complete 5,640-image export is placed in normal output; smoke exports use `preview_incomplete/`.
  Atomic per-image resume locks code/assets/images/settings/runtime hashes; changed protocol
  requires a new directory. Offline legacy 4.49 worker is isolated from the kernel, no PEFT required.
- Optional old/LoRA CSV comparison validates exact split IDs, preserves source files and compares
  actual effective inputs. Common-intersection-mask CSVs prevent semantic-only reject recovery
  from resurrecting intentionally masked text. This controls availability, not semantic selector
  context choices. Fresh prefix controls are not asserted identical to historical CSVs.
- Instructions: `HUONG_DAN_CAPTION_DTD_GROUNDED.md`. CPU mocked/static tests in
  `tests/test_dtd_grounded_caption.py`; real GPU execution and downstream F1 remain unverified.
  Higher optimized CLIP scores are not independent evidence of caption quality or reduced leakage.
- Final local rerun: 198 tests passed (24 new grounded-caption tests plus the 174 earlier tests),
  one existing Starlette TestClient deprecation warning. Notebook has 15 cells, cleared outputs,
  valid Python syntax and deterministic builder output; no local project imports required on Kaggle.

## Uploaded Caption Controls and Method 01 Training Notebook

- User uploaded four CSVs into `all_4_captions/`, requesting individual full_both classifier notebooks,
  starting ONLY with method 1 (Florence-Prefix). All files have 5,640 distinct IDs in the same order,
  train/val/test 1,880 each, matching official DTD split 1 locally. Shared captioner fingerprint:
  `c768be69004fc41e31214eeef4c6a7a4d8176912e21cd72f77a3ff2d23fbdb01`.
- Metadata mappings: `captions_prefix_control.csv` -> Florence-Prefix, stored-valid 4,749;
  `captions_first_valid_control.csv` -> Florence-FirstValid, 5,640;
  `captions_single_grounded_control.csv` -> Florence-SingleGrounded, 5,640;
  `captions_multibeam_grounded.csv` -> Florence-MultiGrounded, 5,640.
  All use dtd_lexical_v2; full availability is NOT evidence of visual truth or no label shortcuts.
- Created `all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-01-florence-prefix.ipynb`,
  builder `build_dtd_caption_method01_prefix_notebook.py`, 24 cells copied from original notebook 05.
  Architecture/model/loss/train/eval cells unchanged. Default 50 epochs, split 1, seed 42, pool seed 42,
  K=1, same VRAM-dependent batch policy. Precomputed Prefix CSV only, no caption regeneration.
- Method-specific scoped env vars: DTD_PREFIX_CAPTIONS_INPUT_PATH, DTD_CAPTION_01_PREFIX_OUTPUT,
  DTD_CAPTION_01_PREFIX_EPOCHS, DTD_CAPTION_01_PREFIX_SEED. Blank input discovers exact Prefix filename
  and SHA in Inputs; explicit paths are also verified for hash, method, fingerprint and official IDs/splits.
  Wrong/stale CSV caches are rejected without overwrite. CSVs were read, not modified.
- Isolated output `outputs/mtacs_dtd_split1_caption_01_florence_prefix_vitl16/`; classifier still named
  full_both within runs. Protocol and result CSVs include caption method identity. First run is fresh;
  resume only its own unfinished run, never original notebook 05 or another caption method's checkpoint.
- Folder contains `README_CAPTION_ABLATIONS.md` and `caption_methods_manifest.json` with all four mappings.
  Method 2-4 classifier notebooks have NOT been created yet. Tests for method 01 cover self-contained
  builder output, unchanged training/model cells, real uploaded CSV audit/bank alignment, rejection of
  wrong methods/splits/hashes/cache contamination, and disabled regeneration. No GPU training executed.

- Subsequent request: created method 02 `all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-02-florence-firstvalid.ipynb`
  with `build_dtd_caption_method02_firstvalid_notebook.py`, reusing method 01's in-memory builder without
  rewriting its notebook. Bound to `captions_first_valid_control.csv`, SHA256
  `37645afc643aaf632cf7226f5cfab25de75a8250156fb366eb42aaf3e8655187`, stored/effective screened coverage 5,640/5,640.
  This chooses the first eligible span from greedy captions, NO image-CLIP caption ranking; the unchanged
  full_both classifier still uses CLIP text vectors and a visual+semantic selector.
  Same 50 epochs, official split 1, seed/pool seed 42, K=1, adaptive batch and core model/training/test cells.
  Own output `outputs/mtacs_dtd_split1_caption_02_florence_firstvalid_vitl16/` and report/plot/result stems.
  Env vars: DTD_FIRST_VALID_CAPTIONS_INPUT_PATH, DTD_CAPTION_02_FIRST_VALID_OUTPUT/EPOCHS/SEED.
  Input hash/method/fingerprint/split guards and no-regeneration policy retained. Notebook 01 and CSVs unchanged;
  manifest/README now list both notebooks. Method 03-04 training notebooks still pending. No GPU training run.

- Subsequent request: created method 03 `all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-03-florence-singlegrounded.ipynb`
  with `build_dtd_caption_method03_singlegrounded_notebook.py`, reusing the earlier in-memory builders.
  Bound to `captions_single_grounded_control.csv`, SHA256
  `6dc67d548014d1a8bb4f0af9c124b26025e20382262c3e2fc9d4d00f682fbcbd`, coverage 5,640/5,640.
  Same greedy descriptions and eligible span pool as FirstValid, but image-text CLIP ranking chooses
  the short text. No additional beam hypotheses. Higher optimized CLIP scores do not establish quality.
  Same full_both classifier, 50 epochs, split 1, seed/pool seed 42, K=1, adaptive batch and core train/test.
  Isolated output `outputs/mtacs_dtd_split1_caption_03_florence_singlegrounded_vitl16/`; scoped env vars
  DTD_SINGLE_GROUNDED_CAPTIONS_INPUT_PATH and DTD_CAPTION_03_SINGLE_GROUNDED_OUTPUT/EPOCHS/SEED.
  Existing CSV/hash/method/fingerprint/split guards and disabled generation retained. Method 01-02 notebooks
  and CSVs not modified. README/manifest list all three. Only method 04 classifier notebook remains pending.
  No real GPU training or new classification metrics were produced locally.

- Subsequent request: created method 04 `all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-04-florence-multigrounded.ipynb`
  with `build_dtd_caption_method04_multigrounded_notebook.py`. Bound to `captions_multibeam_grounded.csv`, SHA256
  `82bfd187c1c69a3727d79bbfd46834fa335077a379e79da7f9c8ef06593bd78f`, coverage 5,640/5,640.
  Greedy + three beam caption hypotheses, same span extraction and image-CLIP ranking as method 03;
  this does NOT change K (still one selected context image). Same full_both architecture, fusion,
  semantic selector, losses, preprocessing, 50 epochs, official split 1, seed/pool seed 42 and adaptive batch.
  Isolated output `outputs/mtacs_dtd_split1_caption_04_florence_multigrounded_vitl16/`; scoped vars
  DTD_MULTI_GROUNDED_CAPTIONS_INPUT_PATH and DTD_CAPTION_04_MULTI_GROUNDED_OUTPUT/EPOCHS/SEED.
  Hash/method/fingerprint/split/cache guards and disabled generation retained. Four classifier notebooks
  are now complete; manifest and README list all four. Previous notebooks and CSVs remain unchanged.
  Tests cover uploaded multi-hypothesis provenance, same masks vs SingleGrounded, unchanged classifier
  logic and ID-aligned banks. No GPU training or new F1 measurements were produced locally.
