"""One-time mechanical path updates; never edits notebook cells or input CSVs."""
from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def replace_nodes(source, edits):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    encoded = source.encode("utf-8")
    spans = [(offsets[node.lineno - 1] + node.col_offset,
              offsets[node.end_lineno - 1] + node.end_col_offset, value.encode("utf-8")) for node, value in edits]
    for start, end, value in sorted(spans, reverse=True):
        encoded = encoded[:start] + value + encoded[end:]
    return encoded.decode("utf-8")


def transform(path, builder=False):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            left = node.left
            if isinstance(left, ast.Name) and left.id == "ROOT":
                edits.append((node, f"project_path({node.right.value!r})"))
            elif isinstance(left, ast.Name) and left.id == "FOLDER" and node.right.value.endswith(".ipynb"):
                edits.append((node, f"project_path({'all_4_captions/' + node.right.value!r})"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "ROOT" and node.func.attr == "glob":
            edits.append((node, "project_glob(" + ", ".join(ast.get_source_segment(source, arg) for arg in node.args) + ")"))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROOT" for t in node.targets):
            if "__file__" in ast.get_source_segment(source, node.value):
                edits.append((node.value, "PROJECT_ROOT"))
    if not edits and builder:
        # Caption method builders inherit ROOT through another builder, but need the resolver too.
        pass
    source = replace_nodes(source, edits)
    import_line = ("from _project_bootstrap import PROJECT_ROOT, project_path\n" if builder
                   else "from project_paths import PROJECT_ROOT, project_path, project_glob\n")
    if import_line not in source:
        tree = ast.parse(source)
        future = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
        index = future[-1].end_lineno if future else next((node.lineno - 1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))), 0)
        lines = source.splitlines(keepends=True)
        lines.insert(index, "\n" + import_line)
        source = "".join(lines)
    # Manifest paths now point from the project root to notebooks, not to adjacent CSVs.
    if path.name.startswith("build_dtd_caption_method"):
        source = source.replace("TARGET.name", "TARGET.relative_to(prefix.ROOT).as_posix()" if "import build_dtd_caption_method01" in source else "TARGET.relative_to(ROOT).as_posix()")
        source = source.replace("prefix.TARGET.relative_to(prefix.ROOT).as_posix()", "prefix.TARGET.relative_to(prefix.ROOT).as_posix()")
        source = source.replace("target.name", "target.relative_to(prefix.ROOT).as_posix()")
    if path.name.startswith("test_dtd_caption_method"):
        for prefix in ("prefix", "firstvalid", "builder"):
            source = source.replace(f"{prefix}.TARGET.name", f"{prefix}.TARGET.relative_to(PROJECT_ROOT).as_posix()")
        source = source.replace("target.name", "target.relative_to(PROJECT_ROOT).as_posix()")
    source = source.replace('builder.ROOT.joinpath("mtacs_dtd_ablation_cells.py")', 'project_path("mtacs_dtd_ablation_cells.py")')
    source = source.replace('Path(__file__).resolve().parents[1] / "notebookb7a9dba884.ipynb"', 'project_path("notebookb7a9dba884.ipynb")')
    ast.parse(source)
    path.write_text(source, encoding="utf-8", newline="\n")


def main():
    marker = PROJECT / "manifests/path_updates_complete.json"
    if marker.exists():
        raise SystemExit("Locator migration is already complete; do not run this one-time tool again.")
    for path in (PROJECT / "tools/notebook_builders").glob("build_*.py"):
        transform(path, builder=True)
    transform(PROJECT / "src/captioning/dtd_caption_generation_cells.py")
    for path in (PROJECT / "tests").glob("test_dtd_*.py"):
        transform(path)
    path = PROJECT / "data/captions/dtd/grounded_controls/caption_methods_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    moves = json.loads((PROJECT / "manifests/migration_plan.json").read_text(encoding="utf-8"))
    mapping = {row["old"]: row["new"] for row in moves}
    for method, name in manifest["created_training_notebooks"].items():
        manifest["created_training_notebooks"][method] = mapping[f"all_4_captions/{name}"]
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"complete": True}) + "\n", encoding="utf-8")
    print("Updated builder/test file locators and caption notebook manifest. Notebook/CSV bytes untouched.")


if __name__ == "__main__":
    main()
