# Development Tools

- `notebook_builders/`: original builders, now writing to method/dataset notebook locations via `project_paths.py`.
- `captioning/`: standalone SUN397 caption generator CLI.
- `visualization/`: Figure-4-style attention tools; provide verified checkpoints/pools and dataset paths explicitly.
- `project_management/validate_layout.py`: content preservation, inventory and path checks.
- `project_management/audit_report_readiness.py`: read saved notebook source/output evidence without executing cells; generate the report-readiness inventory.
- `project_management/create_project_index.py`: regenerate navigation/status placeholders from the recorded catalog.
- `project_management/prepare_layout.py`, `relocate_code_paths.py`: one-time migration utilities, not routine commands.
- `archive/`: old extracted cells and early caption prototype, not current tested entry points.

Some original builders execute at import time, whereas newer builders expose `build_notebook()`.
Do not bulk-import all builders into a live session just to enumerate them: that could rewrite notebooks.
Regression tests protect the newer standalone DTD generators and scientific runtime behavior.

The standalone paired Qwen3-VL caption notebook is built with
`python tools/notebook_builders/build_qwen3vl_dtd_caption_notebook.py` from the TACS root.
It embeds `src/captioning/dtd_qwen_caption.py` and exports 8B/32B DTD captions for separate classifier runs.
