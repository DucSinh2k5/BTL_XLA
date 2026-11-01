# Paper Reproduction Map

The table numbering below was checked against the local CVPR 2026 paper in `docs/paper/`.
These are organizational categories, not assertions that local runs match all paper settings.

| Paper item | Folder | Meaning |
| --- | --- | --- |
| Table 1 | `../../configs/paths.local.example.json` | Dataset inventory; no training method |
| Table 2 | `classification/table_02_main_results/` | Classification: five methods, fifteen datasets |
| Table 3 | `segmentation/table_03_main_results/` | Segmentation: five methods, three datasets |
| Table 4 | `classification/table_04_context_pairs/` | No/blank/duplicate/noisy/similar/TACS context |
| Table 5 | `classification/table_05_optimization/` | Frozen, differentiable-only, policy-only, dual optimization |
| Figure 4 | `figure_04_attention/` | Query/context attention and checkpoint export |

Each table follows `method/dataset/notebook.ipynb`. A missing implementation stays empty.
Table 4 no-context/similar/TACS baselines and Table 5 frozen/full-TACS can reuse the corresponding Table 2 notebooks;
no extra notebook copies have been created. Use the [catalog](../README.md) to locate those canonical files.
Table 5 differentiable-only/policy-only slots do NOT yet contain dedicated notebooks.

Paper dataset aliases: CUB = CUB-200-2011, Cars = Stanford Cars, SUN = SUN397,
Col. = colorectal, ISIC classification = ISIC 2019; ISIC segmentation = ISIC 2017.

Before reporting a reproduction, verify backbone/pretraining, split, candidate pool, K, losses,
checkpoint selection, metric definition and seeds. Macro F1 is not the paper's DTD accuracy column.
Do not put the new multimodal DTD `visual_only` control in the paper's no-context row:
it still uses a selected visual context image.
