# 06: Candidate-Only With Label-Aware Candidate Text

[Open notebook](mtacs-dtd-vitl16-prefix-candidate-only-label-context.ipynb) · built by
`tools/notebook_builders/build_dtd_prefix_label_context_notebook.py` from `src/training/mtacs_dtd_label_context_cells.py`.

Study01 (best so far, test Macro-F1 85.30) is kept unchanged: image-only selector, K=1, gated residual fusion of
the selected candidate's text. Only the candidate text changes.

## What Changes

```text
Study01: t_a = CLIP(Florence caption of candidate a)                      # derived from a's pixels
Study06: t_a = CLIP("a photo of a <class of a> texture, <caption of a>")  # caption appended only if valid
```

- The candidate pool holds training images only, so the candidate's label is legitimately known at inference,
  like a kNN / retrieval memory. The query's label is never an input; query text stays off; self-exclusion stays.
- Every candidate now has valid text (the label always exists).
- This is a **label-aware retrieval-augmented variant**, not the original TACS protocol. Report it as such.

## Kaggle Inputs

DTD, DINOv3 code + ViT-L/16 weights, CLIP ViT-B/32. The frozen Prefix CSV is embedded in the notebook.

## How to Read the Result

- Compare test Macro-F1 with study01 (85.30) and visual_only A01 (84.82).
- `text_off` and `text_shuffled` (wrong label for the same image) should both lower validation F1 if the
  model relies on the candidate label. `candidate_gate` shows how much text enters the fusion.
- Check how often the selected context has the same class as the query (`selected_context_id` in the
  per-image predictions vs. `shared/candidate_label_context_texts.csv`).
- One seed: repeat with `DTD_PREFIX_LABEL_CONTEXT_SEED=43` and `44` before claiming a gain under ~0.5 point.

No GPU run exists yet; this template has no outputs.
