# Existing Research Source

| Location | Existing responsibility |
| --- | --- |
| `captioning/dtd_caption_screening.py` | Caption cleanup and lexical policy |
| `captioning/dtd_caption_generation_cells.py` | Standalone offline caption worker/cells embedded by builders |
| `captioning/dtd_caption_finetune.py` | Reference validation, decoder LoRA training and export helpers |
| `captioning/dtd_grounded_caption.py` | Frozen caption hypotheses, span selection and grounded ranking |
| `training/mtacs_dtd_ablation_cells.py` | Shared source for ablation models/train/evaluation cells |

The existing flat import names are preserved. Builders bootstrap their import paths and pytest uses
the project configuration. Generated Kaggle notebooks continue to embed their runtime code.

`models/`, `selection/` and `evaluation/` are empty planned boundaries. TACS/DINO models and selectors
currently remain in the notebooks; no production model package was invented by the organizer.
