# 05: Candidate-Only With an In-ViT Text Token

[Open notebook](mtacs-dtd-vitl16-prefix-candidate-only-text-token.ipynb) · built by
`tools/notebook_builders/build_dtd_prefix_text_token_notebook.py` from `src/training/mtacs_dtd_text_token_cells.py`.

Studies 01–04 add the candidate caption as one CLIP vector to the pooled feature after the ViT, so no patch
can attend to it (study01 gate ≈ 0.001). Study05 moves the caption inside the paired ViT instead.

## What Changes

```text
Study01: fused = LayerNorm(visual + gate * text_projection(CLIP_caption))     # after the ViT
Study05: token = scale * LayerNorm(Linear(CLIP_caption))   (null token if the caption is invalid)
         ViT input = [CLS, registers, token, query patches, context patches]  # token gets no RoPE
         fused = LayerNorm(visual)                                            # late gate unused
```

- `scale` is a learned scalar starting at 0 and the null token starts at 0, so training starts near the visual path.
- New parameters: Linear(512→1024) + LayerNorm, one scalar, one 1024-d null token (head LR).
- Query captions stay off; the selector is image-only; K=1; frozen Prefix CSV; same 50-epoch split1 protocol.

## Kaggle Inputs

DTD, DINOv3 code + ViT-L/16 weights, and CLIP ViT-B/32. The frozen `captions_prefix_control.csv` is embedded:
if `CAPTIONS_INPUT_PATH` is blank or not a file, a cell writes it to the working directory and checks its SHA-256.

## How to Read the Result

- Compare test Macro-F1 with study01 (85.30%) and visual_only A01 (84.82%).
- The validation interventions must show a drop for `text_off` (null token) or `text_shuffled`;
  otherwise any gain is not from the caption.
- `candidate_gate` reports |scale|; `weighted_text_visual_ratio` = token norm / mean query patch-token norm.
- One seed is not enough for gains under ~0.3 points; repeat with seeds 43 and 44 via `DTD_PREFIX_TEXT_TOKEN_SEED`.

No GPU run exists yet; this template has no outputs.
