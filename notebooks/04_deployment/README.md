# Training and Deployment Bundles

- `sun397/01_tacs_visual/`: standalone SUN397 visual TACS ViT-L/16 training,
  evaluation, real attention and complete portable web bundle export.
- `sun397/02_mtacs_full_both/`: SUN397 M-TACS ViT-L/16, query/candidate captions
  in fusion and selector, integrated caption generation, and a full web bundle
  including frozen Florence/CLIP assets.
- [ISIC2017 segmentation](isic2017/README.md): separate TACS visual and M-TACS
  full_both notebooks, ViT-L/16 at 448, official splits, integrated Florence-Prefix
  captions for M-TACS, and complete segmentation/attention web bundles.

This task-oriented folder is separate from historical paper result notebooks.
Training output, not the unexecuted template, supplies deployment weights and
metrics. See each notebook's README for exact Inputs, protocol and output files.
