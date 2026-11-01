"""Inventory and plan the one-time XLA -> TACS migration (no moves)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
PAPER = "notebooks/01_paper_reproduction"
IMPROVED = "notebooks/02_improvements"
ARCHIVE = "notebooks/90_archive"
MOVES = []
CATALOG = []


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def add(source, destination, reason):
    src = WORKSPACE / source
    if not src.exists():
        raise FileNotFoundError(src)
    dst = PROJECT / destination
    if dst.exists():
        raise FileExistsError(dst)
    MOVES.append({"old": source, "new": destination, "reason": reason,
                  "kind": "directory" if src.is_dir() else "file"})


def notebook(source, folder, dataset, method, status="research_variant", note=""):
    destination = f"{folder}/{Path(source).name}"
    add(source, destination, "notebook classification from source/configuration")
    original = json.loads((WORKSPACE / source).read_text(encoding="utf-8"))
    text = "\n".join("".join(c.get("source", [])) for c in original["cells"])
    evidence = [line.strip() for line in text.splitlines() if re.match(
        r"\s*(NUM_CLASSES|DINOV3_MODEL_NAME|SEED|EXPERIMENT_MODE|CONTEXT_TOP_K|TRAIN_ON_FULL_TRAIN)\s*=", line)]
    CATALOG.append({"path": destination, "old_path": source, "dataset": dataset,
                    "method": method, "status": status, "note": note,
                    "cells_with_outputs": sum(bool(c.get("outputs")) for c in original["cells"]),
                    "configuration_evidence": evidence, "sha256": digest(WORKSPACE / source)})


def main():
    base = f"{PAPER}/classification/table_02_main_results"
    groups = {
        ("no_context", "cub200"): ["baseline_no_tacs_image_classification.ipynb",
            "CUB/baseline_no_tacs_image_classification (seed42).ipynb",
            "CUB/baseline_no_tacs_image_classification (seed43).ipynb",
            "CUB/baseline_no_tacs_image_classification_(seed_44).ipynb"],
        ("no_context", "stanford_cars"): ["CARS/seed44-car.ipynb"],
        ("no_context", "sun397"): ["SUN/table2/baseline_no_tacs_cars.ipynb"],
        ("random_context", "stanford_cars"): ["baseline_random_context_ret_cars.ipynb",
            "ban-cuoi-random-context-ret-car.ipynb", "SUN/table2/ban-cuoi-Random Context Retrieval baseline.ipynb"],
        ("frozen_dino_similarity", "stanford_cars"): ["baseline_frozen_dino_similarity_ret_cars.ipynb",
            "ban-cuoi-frozen-dino-car.ipynb", "notebook7dcd2fd8f0.ipynb"],
        ("feature_averaged_context", "stanford_cars"): ["baseline_feature_averaged_context_cars.ipynb"],
        ("tacs", "stanford_cars"): ["baseline_tacs_cars.ipynb", "ban-cuoi-tacs-car.ipynb",
            "notebook05aceb61b4.ipynb", "tacs-chuanpaper-car-b16.ipynb", "tacs-chuanpaper-car-l16.ipynb",
            "tacs-chuanpaper-sun-b16 (2).ipynb", "SUN/table2/ban-cuoi-frozen-dino-car.ipynb"],
        ("tacs", "sun397"): ["ban-cuoi-tacs-sun.ipynb", "tacs-sun-da-sua (1).ipynb",
            "tacs-sun-da-sua (2).ipynb", "tacs-sun-true-val-split.ipynb",
            "tacs-vitb16-sun-true-val-split.ipynb", "tacs-chuanpaper-sun-b16 (1).ipynb"],
    }
    for (method, dataset), paths in groups.items():
        for path in paths:
            note = "Paper-related implementation; not a certification of matching the paper protocol."
            if path.startswith("SUN/") or "sun-b16 (2)" in path:
                note += " Original filename/folder is misleading; dataset follows notebook configuration."
            if "SUN/table2/ban-cuoi-frozen" in path:
                note += " Learned TACS selector, not the frozen retriever suggested by its filename."
            notebook(path, f"{base}/{method}/{dataset}", dataset, method, note=note)

    for method in ("blank_image", "duplicate_query", "noisy_query"):
        notebook(f"table4_{method}_sun.ipynb",
                 f"{PAPER}/classification/table_04_context_pairs/{method}/sun397", "sun397", method)
    notebook("tacs_train_with_checkpoints.ipynb", f"{PAPER}/figure_04_attention/tacs/sun397",
             "sun397", "tacs_attention_checkpoint_export")
    notebook("tacs_isic2017_segmentation.ipynb",
             f"{PAPER}/segmentation/table_03_main_results/tacs/isic2017", "isic2017", "tacs_segmentation")

    groups = {
        ("query_text", "sun397"): ["mtacs-train-captions-vitl16.ipynb", "mtacs-train-captions.ipynb",
            "mtacs_sun397_visual_text_context.ipynb", "mtacs_sun397_visual_text_context_offline.ipynb"],
        ("query_text", "dtd"): ["mtacs-train-captions-vitl16-dtd.ipynb"],
        ("query_candidate_text", "sun397"): ["mtacs-train-captions-vitl16-context-text.ipynb"],
        ("multimodal_top3/selector_text_only", "sun397"): ["mtacs-train-captions-vitl16-multimodal-top3.ipynb"],
        ("multimodal_top3/context_text_fusion", "sun397"): ["mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb"],
        ("adaptive_context", "sun397"): ["tacs-adaptive-sun.ipynb", "tacs_adaptive_diverse_context_sun.ipynb",
            "tacs-adaptive-sun-dinov3-vitb-l.ipynb"],
        ("visual_top3", "sun397"): ["tacs_top3_context_sun.ipynb"],
        ("zero_shot_siglip", "sun397"): ["siglip2_zero_shot_tacs_fusion_sun397.ipynb"],
        ("caption_generation/decoder_lora", "dtd"): ["florence2-dtd-caption-finetune.ipynb"],
        ("caption_generation/grounded_rerank", "dtd"): ["florence2-dtd-caption-grounded-rerank.ipynb"],
    }
    for (method, dataset), paths in groups.items():
        for path in paths:
            notebook(path, f"{IMPROVED}/{method}/{dataset}", dataset, method)
    methods = ["visual-only", "query-only", "candidate-only", "both-fusion", "full-both",
               "full-shuffle-both", "full-unscreened", "full-redacted"]
    for index, method in enumerate(methods, 1):
        notebook(f"mtacs-train-vitl16-dtd-ablation-{index:02d}-{method}.ipynb",
                 f"{IMPROVED}/text_context_ablation/{index:02d}_{method.replace('-', '_')}/dtd",
                 "dtd", method.replace("-", "_"), "standalone_template")
    for index, method in enumerate(("prefix", "firstvalid", "singlegrounded", "multigrounded"), 1):
        notebook(f"all_4_captions/mtacs-train-vitl16-dtd-caption-ablation-{index:02d}-florence-{method}.ipynb",
                 f"{IMPROVED}/caption_ablation/{index:02d}_florence_{method}/dtd",
                 "dtd", f"florence_{method}", "standalone_template")
    for path in WORKSPACE.glob("*.ipynb"):
        if "_before_" not in path.name:
            continue
        dataset = "isic2017" if "isic" in path.name else "stanford_cars" if "notebook05" in path.name or "notebook7d" in path.name else "sun397"
        notebook(path.name, f"{ARCHIVE}/{dataset}/historical_versions", dataset, "historical", "archived")
    for name in ("mtacs-train-captions-vitl16-dtd-text-ablation.ipynb", "notebookb7a9dba884.ipynb"):
        notebook(name, f"{ARCHIVE}/dtd/combined_ablation_suite", "dtd", "combined_ablation_suite", "builder_dependency",
                 "Preserved as a builder source; use standalone notebooks for new runs.")
    notebook("tmp/notebookb7a9dba884.before-50-epochs.ipynb", f"{ARCHIVE}/dtd/historical_versions",
             "dtd", "combined_ablation_100_epochs", "archived")
    notebook("caption-cai-tien.ipynb", "experiments/completed/caption_decoder_lora/dtd", "dtd",
             "full_both_with_decoder_lora_captions", "saved_execution")

    for path in WORKSPACE.glob("build_*.py"):
        add(path.name, f"tools/notebook_builders/{path.name}", "notebook builder")
    for name in ("dtd_caption_finetune.py", "dtd_caption_generation_cells.py", "dtd_caption_screening.py", "dtd_grounded_caption.py"):
        add(name, f"src/captioning/{name}", "caption-stage implementation")
    add("mtacs_dtd_ablation_cells.py", "src/training/mtacs_dtd_ablation_cells.py", "embedded training-cell source")
    add("generate_sun397_captions.py", "tools/captioning/generate_sun397_captions.py", "caption generation CLI")
    for name in ("run_figure4_random5.py", "visualize_figure4.py"):
        add(name, f"tools/visualization/{name}", "Figure 4 visualization tool")
    for path in WORKSPACE.glob("_inspect_cell_*.py"):
        add(path.name, f"tools/archive/extracted_cells/{path.name}", "historical inspection fragment")
    add("test_transformer.py", "tools/archive/captioning_prototype/test_transformer.py", "early captioning prototype; not a test")
    add("tests", "tests", "project regression tests")
    add("sun397_demo", "apps/sun397_demo", "backend/frontend package, kept together")
    add("output_ablation", "experiments/completed/text_context_ablation/dtd", "eight saved notebooks and immutable evidence")
    add("model_TACS", "artifacts/checkpoints/imported_unverified/model_TACS", "imported assets; compatibility not verified")
    add("mtacs_sun397_assets", "configs/mtacs_sun397_assets", "existing SUN397 assets")
    for name in ("best_val_checkpoint (3).pt", "final_tacs_full_pool_dinov3_vits16_sun397.pt", "tacs_selector_cache.pt", "trained_tacs_full_pool_candidate_feature_bank.pt"):
        add(name, f"artifacts/checkpoints/imported_unverified/root/{name}", "preserve original artifact and provenance, no deduplication")
    for path in (WORKSPACE / "all_4_captions").glob("*.csv"):
        add(f"all_4_captions/{path.name}", f"data/captions/dtd/grounded_controls/{path.name}", "caption experiment input, content unchanged")
    for name in ("caption_methods_manifest.json", "README_CAPTION_ABLATIONS.md"):
        add(f"all_4_captions/{name}", f"data/captions/dtd/grounded_controls/{name}", "caption input documentation")
    add("all_4_captions.zip", "data/archives/all_4_captions.zip", "original caption bundle snapshot")
    add("captions (5).csv", "data/captions/dtd/decoder_lora/captions (5).csv", "decoder-LoRA caption output")
    for name in ("caption_references.csv", "caption_references.annotation_notes.md", "caption_references.annotation_report.json"):
        add(name, f"data/annotations/dtd/{name}", "teacher reference annotations and provenance")
    # The review folder has an external node_modules junction: leave dependencies
    # outside the project and move only the actual annotation evidence files.
    for path in (WORKSPACE / "tmp/annotate_dtd_references").iterdir():
        if path.is_file():
            add(f"tmp/annotate_dtd_references/{path.name}", f"reports/annotation_review/dtd/{path.name}",
                "annotation evidence, original backup, contact sheets")
    add("tmp/caption_cai_tien_inspection", "reports/caption_decoder_lora/dtd", "saved learning curve inspection")
    for name in ("HUONG_DAN_FINETUNE_CAPTION_DTD.md", "HUONG_DAN_CAPTION_DTD_GROUNDED.md"):
        add(name, f"docs/guides/{name}", "caption experiment guide")
    add("TONG_HOP_CAI_TIEN_MTACS_SO_VOI_TACS.md", "reports/TONG_HOP_CAI_TIEN_MTACS_SO_VOI_TACS.md", "historical research summary")
    add("PROJECT_CONTEXT.md", "docs/history/PROJECT_CONTEXT.md", "historical handover, preserve old path statements")
    for path in WORKSPACE.glob("18.*.pdf"):
        add(path.name, f"docs/paper/{path.name}", "paper reference")

    old_names = [row["old"] for row in MOVES]
    new_names = [row["new"] for row in MOVES]
    if len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names):
        raise ValueError("Duplicate migration source or destination")
    for name in old_names:
        if any(name != parent and name.startswith(parent + "/") for parent in old_names):
            raise ValueError(f"Nested migration source: {name}")
    files = []
    for move in MOVES:
        source = WORKSPACE / move["old"]
        for path in source.rglob("*") if source.is_dir() else (source,):
            if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            suffix = path.relative_to(source).as_posix() if source.is_dir() else ""
            destination = f"{move['new']}/{suffix}" if suffix else move["new"]
            files.append({"old": path.relative_to(WORKSPACE).as_posix(), "new": destination,
                          "bytes": path.stat().st_size, "sha256_before": digest(path)})
            if path.suffix == ".ipynb" and destination not in {row["path"] for row in CATALOG}:
                original = json.loads(path.read_text(encoding="utf-8"))
                CATALOG.append({"path": destination, "old_path": path.relative_to(WORKSPACE).as_posix(),
                    "dataset": "dtd", "method": path.stem.replace("-", "_"), "status": "saved_execution",
                    "note": "Saved outputs; never rerun as part of migration.",
                    "cells_with_outputs": sum(bool(c.get("outputs")) for c in original["cells"]),
                    "configuration_evidence": [], "sha256": digest(path)})
    manifest = PROJECT / "manifests"
    manifest.mkdir(parents=True, exist_ok=True)
    for name, value in (("migration_plan.json", MOVES), ("migration_files.json", files),
                        ("notebook_catalog.json", sorted(CATALOG, key=lambda row: row["path"]))):
        target = manifest / name
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Planned {len(MOVES)} moves, {len(files)} files, {len(CATALOG)} notebooks. No files moved.")


if __name__ == "__main__":
    main()
