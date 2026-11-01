"""Create layout placeholders and navigable indexes from the verified inventory."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["fgvc_aircraft", "caltech101", "caltech256", "cifar10", "cifar100", "cub200", "dtd",
            "food101", "oxford_pets", "stanford_cars", "sun397", "aptos2019", "colorectal", "ddsm", "isic2019"]
METHODS = ["no_context", "random_context", "frozen_dino_similarity", "feature_averaged_context", "tacs"]


def placeholder(relative):
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    if not any(path.iterdir()):
        (path / ".gitkeep").touch()


def write(relative, text):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    catalog = json.loads((ROOT / "manifests/notebook_catalog.json").read_text(encoding="utf-8"))
    for table, methods in (("table_02_main_results", METHODS),
                           ("table_04_context_pairs", ["no_context", "blank_image", "duplicate_query", "noisy_query", "frozen_dino_similarity", "tacs"]),
                           ("table_05_optimization", ["frozen_retriever", "differentiable_only", "policy_only", "dual_path_tacs"])):
        for method in methods:
            for dataset in DATASETS:
                placeholder(f"notebooks/01_paper_reproduction/classification/{table}/{method}/{dataset}")
    for method in METHODS:
        for dataset in ("isic2017", "kvasir_seg", "drive"):
            placeholder(f"notebooks/01_paper_reproduction/segmentation/table_03_main_results/{method}/{dataset}")
    for dataset in DATASETS + ["isic2017", "kvasir_seg", "drive"]:
        placeholder(f"data/raw/{dataset}")
        placeholder(f"data/processed/{dataset}")
        placeholder(f"notebooks/00_data_preparation/{dataset}")
    for relative in ("configs/training", "configs/inference", "src/models", "src/selection", "src/evaluation",
                     "artifacts/feature_banks", "artifacts/text_embeddings", "artifacts/captioners",
                     "experiments/planned", "experiments/running", "experiments/completed/paper_reproduction",
                     "experiments/completed/caption_ablation/dtd", "reports/paper_tables", "reports/figures",
                     "reports/presentation", "docs/progress", "notebooks/03_analysis_and_evaluation/classification/dtd",
                     "notebooks/03_analysis_and_evaluation/classification/sun397",
                     "notebooks/03_analysis_and_evaluation/segmentation/isic2017", "apps/segmentation"):
        placeholder(relative)
    registry = {
        "path_base": "project_root", "note": "Raw datasets stay outside TACS; paths are optional existing local assets, not bundled data.",
        "datasets": {
            "dtd": {"local_path": "../archive (23)/dtd", "kaggle_path": "/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd", "official_splits": "labels/{train,val,test}{1..10}.txt"},
            "stanford_cars": {"local_paths": ["../archive (10)", "../archive (13)"]},
            "cub200": {"local_paths": ["../CUB_200_2011/CUB_200_2011", "../archive (4)/CUB_200_2011"]},
            "cifar100": {"local_path": "../archive (12)/cifar100"},
            "sun397": {"local_path": None, "kaggle_path": "/kaggle/input/datasets/hiuphmnhtrung/sun397/SUN397/SUN397"},
        },
        "models": {"dinov3_repo": None, "dinov3_weights": None, "clip_model": None, "florence_model": None},
        "note_models": "Do not activate imported checkpoints until architecture/pool/class map compatibility is verified."
    }
    write("configs/paths.local.example.json", json.dumps(registry, ensure_ascii=False, indent=2))
    header = """# Notebook Catalog

Paths are relative to the TACS project root. Original filenames and notebook bytes are preserved.
`research_variant` means an implementation exists, NOT that it reproduces the published protocol/results.
`cells_with_outputs` records saved output presence, not a claim that training completed successfully.
Read configuration evidence in [notebook_catalog.json](../manifests/notebook_catalog.json).
Empty method/dataset folders contain only `.gitkeep`; they are planned slots, not experiments.

## Quick Start

- New DTD text-context experiment: choose one standalone notebook under `02_improvements/text_context_ablation/`.
- Compare four caption CSVs: choose one notebook under `02_improvements/caption_ablation/`.
- Caption generation/training: `02_improvements/caption_generation/`.
- Read actual eight-run outputs: `../experiments/completed/text_context_ablation/dtd/`.
- Archived combined DTD suites remain builder dependencies; do not delete them.

## All Notebooks

| Dataset | Method | Status | Output cells | Notebook |
| --- | --- | --- | ---: | --- |
"""
    for row in catalog:
        header += f"| {row['dataset']} | {row['method']} | {row['status']} | {row['cells_with_outputs']} | [{Path(row['path']).name}](<../{row['path']}>) |\n"
    header += "\n## Misleading Original Names\n\n"
    for row in catalog:
        if "misleading" in row.get("note", ""):
            header += f"- `{row['old_path']}`: **{row['dataset']} / {row['method']}**. {row['note']}\n"
    write("notebooks/README.md", header)
    table_index = """# Paper Reproduction Map

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
"""
    write("notebooks/01_paper_reproduction/README.md", table_index)
    write("notebooks/02_improvements/README.md", """# Improvement Experiments

All groups below are further divided by dataset; original notebook names are preserved.

| Group | Purpose |
| --- | --- |
| `query_text` | Query caption fusion; visual candidate image |
| `query_candidate_text` | Query and selected context caption fusion, SUN397 K=1 |
| `multimodal_top3` | Separate K=3 experiments; selector-text and context-text-fusion variants |
| `adaptive_context` | Adaptive/diverse visual context and backbone variants |
| `visual_top3` | Three visual contexts, no caption improvement claim |
| `zero_shot_siglip` | SigLIP zero-shot fusion experiment |
| `text_context_ablation` | Eight standalone DTD controls, 50 epochs |
| `caption_generation` | Decoder LoRA or grounded reranking caption pipelines |
| `caption_ablation` | Four input CSV methods with the same full_both classifier |

The caption input files live in `data/captions/dtd/`; they are not mixed with notebooks.
The four caption-method notebooks already take CSV inputs and do not regenerate captions.
DTD notebooks 02-08 in text-context ablation can create/resume captions with local offline assets.
The visual-only notebook does not require captions. Do not change masks/thresholds using test performance.

See [all notebooks](../README.md) and [experiment protocol](../../docs/protocols/EXPERIMENT_PROTOCOL.md).
""")
    write("docs/progress/STATUS.md", f"""# Project Status After Organization

- Cataloged notebooks: {len(catalog)}; preserved original outputs and source cells.
- Eight DTD text-context execution notebooks and analysis are under `experiments/completed/text_context_ablation/dtd/`.
- A saved decoder-LoRA-caption classifier run is under `experiments/completed/caption_decoder_lora/dtd/`.
- Four new caption-method training templates are present; no new training was performed by this migration.
- Paper-related notebooks exist for CUB, Stanford Cars, SUN397 and ISIC 2017; published equivalence remains to be audited.
- Web classification frontend/backend exists; model adapter is not connected. Segmentation demo is an empty future slot.
- Checkpoints were imported without unpickling/activating; architecture/class-map/pool compatibility is unverified.
- `experiments/running/` is empty: local files do not establish whether an external Kaggle session is still running.

## Next Evidence To Collect

1. Export each new caption-method run with protocol, input CSV hash, training history and best checkpoint identity.
2. Match split/backbone/batch/seed/budget across comparisons; report multiple seeds when available.
3. Verify a compatible SUN397 checkpoint and candidate pool before connecting the demo adapter.
4. Fill missing paper-table methods only after implementations and real outputs exist.
""")
    counts = Counter(row["dataset"] for row in catalog)
    write("manifests/layout_summary.json", json.dumps({"notebooks": len(catalog), "by_dataset": dict(counts),
        "paper_tables": [2, 3, 4, 5], "notebook_content_changed": False}, indent=2))
    print(f"Created indexes, dataset registry and placeholders; {len(catalog)} notebooks indexed.")


if __name__ == "__main__":
    main()
