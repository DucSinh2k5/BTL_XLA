"""Create the single-context M-TACS variant that fuses query and context captions."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import json
from pathlib import Path


ROOT = PROJECT_ROOT
SOURCE = project_path('mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb')
TARGET = project_path('mtacs-train-captions-vitl16-context-text.ipynb')


def source_of(cell):
    return "".join(cell.get("source", []))


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

# Keep the query caption and add the selected candidate's caption, with K=1.
for cell in cells:
    if cell.get("cell_type") not in {"code", "markdown"}:
        continue
    source = source_of(cell)
    source = source.replace("mtacs_dual_top3_context_text", "mtacs_dual_context_text")
    source = source.replace("dual_top3_context_text", "dual_context_text")
    source = source.replace("dual_visual_semantic_top3_with_context_text", "dual_visual_semantic_single_context_with_context_text")
    source = source.replace("learned_visual_semantic_top3_without_replacement", "learned_visual_semantic_single_context_selector")
    source = source.replace("mtacs_sun397_visual_text_context_vitl16_dual_top3_context_text",
                            "mtacs_sun397_visual_text_context_vitl16_context_text")
    source = source.replace("CONTEXT_TOP_K = 3", "CONTEXT_TOP_K = 1")
    source = source.replace("visual + query/context semantic top-3 context", "visual + query/context semantic single context")
    source = source.replace("visual and CLIP-caption similarity; selected context images", "visual and CLIP-caption similarity; the single selected context image")
    source = source.replace("selected visual contexts and their captions change together", "the selected visual context and its caption change together")
    source = source.replace("top-3 context", "single context")
    set_source(cell, source)

intro = source_of(cells[0])
intro = intro.replace(
    "`EXPERIMENT_MODE` supports `no_context`, `original_tacs`, `text_only`, `mtacs`, and `mtacs_shuffled`.",
    "`EXPERIMENT_MODE` defaults to `mtacs_dual_context_text`: one selected candidate image, the query caption, and the selected candidate caption are fused.",
)
set_source(cells[0], intro)

model = source_of(cells[13])
duplicate_device_labels = (
    "candidate_labels_device = candidate_labels_tensor.to(DEVICE, non_blocking=True)\n"
    "candidate_image_ids = candidate_df['image_id'].astype(str).tolist()\n"
    "candidate_labels_device = candidate_labels_tensor.to(DEVICE, non_blocking=True)"
)
model = model.replace(
    duplicate_device_labels,
    "candidate_labels_device = candidate_labels_tensor.to(DEVICE, non_blocking=True)\n"
    "candidate_image_ids = candidate_df['image_id'].astype(str).tolist()",
)
set_source(cells[13], model)

# The notebook already has a device-resident label bank. Keep evaluation read-only
# with respect to that global tensor to avoid accidentally shadowing it locally.
training = source_of(cells[16])
if "candidate_labels_tensor = candidate_labels_tensor.to(DEVICE)" in training:
    training = training.replace(
        "candidate_labels_tensor = candidate_labels_tensor.to(DEVICE)",
        "candidate_labels_device = candidate_labels_tensor.to(DEVICE, non_blocking=True)",
    )
    training = training.replace(
        'candidate_labels_tensor[details["selected_indices"][j]]',
        'candidate_labels_device[details["selected_indices"][j].to(candidate_labels_device.device)]',
    )
set_source(cells[16], training)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {TARGET}")
