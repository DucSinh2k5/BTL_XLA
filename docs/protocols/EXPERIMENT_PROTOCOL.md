# Experiment and Evidence Protocol

## Separation of Responsibilities

The research model/selector definitions are still embedded in notebooks. Existing reusable caption code lives
in `src/captioning`; the large ablation-cell source lives in `src/training`.
The empty `src/models`, `src/selection` and `src/evaluation` folders are future extraction slots, not implemented APIs.
Reorganizing files does not change model architecture or scientific claims.

## Reproducible Run Identity

For each new run record dataset/split, pool seed, training seed, ordered candidate IDs, caption CSV SHA-256,
caption policy/masks, checkpoint architecture, backbone/pretraining, K, batch/accumulation, epochs, optimizer,
selection losses, checkpoint criterion and metric definition. Keep the generated notebook's own protocol guards.
Use separate run directories when any of these changes; do not rename incompatible checkpoints to `best.pt`.

Suggested storage for new imports:

```text
experiments/completed/<study>/<dataset>/<method>/seed_<seed>/
  protocol.json
  metrics.json
  training_history.csv
  test_predictions.csv
  candidate_pool.csv
  run.ipynb
```

These are expected artifact names, not files created by the organizer. Checkpoints/large banks may instead
be stored in `artifacts/` with explicit links and hashes in the run manifest.

## Train / Validation / Test

- Candidate pools contain training images only; exclude the query itself when it is a pool member.
- DTD has ten official split definitions; preserve each run's split number. Do not combine outputs from different splits.
- Select thresholds, caption policies, checkpoints and hyperparameters using train/validation only.
- Keep test labels out of caption generation, semantic ranking and model/threshold selection.
- A caption class-name/CLIP audit is heuristic. High test performance alone does not prove the absence of leakage.
- Raw/unscreened captions are an explicit shortcut control, not evidence of safe text.
- `full_shuffle_both` tests semantic alignment while keeping masks; it is not a new caption-quality method.

## Reporting

Keep paper metrics distinct from local ones (for DTD the paper Table 2 reports accuracy, not macro F1).
Report mean/std over matched seeds/splits where available. A single saved run does not establish statistical significance.
Do not promote a method to "best" based solely on the user's earlier working assumption.
The historical output-analysis report remains the source for the eight completed DTD runs.
The four grounded-control CSVs are inputs; coverage and optimized CLIP scores are not downstream results.

Teacher annotations and historical source hashes must retain their provenance. New human verification should be
recorded separately rather than retroactively relabeling teacher-only references as human ground truth.
