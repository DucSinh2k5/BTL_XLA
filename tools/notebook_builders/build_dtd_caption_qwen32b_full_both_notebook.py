"""Derive a separate full_both run for Qwen32B captions without changing the 8B run."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT

import ast
import hashlib
import json

import build_dtd_caption_qwen8b_full_both_notebook as qwen8b


SOURCE = qwen8b.SOURCE
TARGET = PROJECT_ROOT / "notebooks/02_improvements/caption_ablation/06_qwen3_vl_32b/dtd/mtacs-train-vitl16-dtd-caption-ablation-06-qwen3-vl-32b.ipynb"


def source_of(cell):
    return qwen8b.source_of(cell)


def to_qwen32b(source):
    # Rejection messages must name the OTHER model after changing the accepted identity.
    source = source.replace("Florence/32B/", "Florence/8B/")
    for before, after in (
        ("caption_05_qwen3vl_8b", "caption_06_qwen3vl_32b"),
        ("Qwen3-VL-8B", "Qwen3-VL-32B"),
        ("qwen3_vl_8b", "qwen3_vl_32b"),
        ("qwen3vl8b", "qwen3vl32b"),
        ("Qwen8B", "Qwen32B"),
        ("QWEN8B", "QWEN32B"),
        ("qwen8b", "qwen32b"),
        ("qwen3vl_8b", "qwen3vl_32b"),
    ):
        source = source.replace(before, after)
    return source


def build_notebook():
    notebook = qwen8b.build_notebook()
    for index, cell in enumerate(notebook["cells"]):
        source = to_qwen32b(source_of(cell))
        cell["source"] = source.splitlines(keepends=True)
        cell["id"] = f"dtd-caption-qwen32b-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source)
            cell.update(outputs=[], execution_count=None)
    helpers = "".join(source_of(notebook["cells"][index]) for index in (8, 9, 11, 12, 14, 15))
    utilities = source_of(notebook["cells"][6]).split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    utilities += f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(SOURCE.read_bytes()).hexdigest()!r}\n"
    utilities += f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n"
    ast.parse(utilities)
    notebook["cells"][6]["source"] = utilities.splitlines(keepends=True)
    notebook["metadata"]["dtd_ablation"].update(caption_method="qwen3_vl_32b_instruct",
        caption_display_name="Qwen3-VL-32B-Instruct", caption_generation=False, captioner_finetuning=False,
        caption_csv_sha256="computed_and_locked_at_runtime", default_epochs=50,
        default_caption_input_policy="dtd_lexical_v2")
    return notebook


def main():
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
