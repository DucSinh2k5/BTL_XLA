# Running the Reorganized Project

Run commands from `F:\Documents\CODE\Python\cv_project\XLA\TACS`.
The original Python environment remains in `..\.venv`; no environment was moved or modified.
Activate the desired environment first. Installing requirements is a user operation, not part of the migration.

## Web Classification Demo

**Cập nhật 01/11/2026:** hai bundle SUN397 thật nằm ở `artifacts/deployment/sun397/` và được web tự nạp khi khởi động. Dùng `./start_web.ps1`, hoặc lệnh Uvicorn bên dưới. Cấu hình worker/dependency và hướng dẫn hiện hành ở [README demo](../../apps/sun397_demo/README.md).

```powershell
python -m pip install -r requirements/web.txt
python -m uvicorn --app-dir apps sun397_demo.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> for paper TACS, or <http://127.0.0.1:8000/mtacs> for multimodal TACS.
If port 8000 is occupied, use another port. `--app-dir apps` is required after the package move.
Workers preload at server startup; HTTP 503 during loading or failure is intentional. `/readyz` returns 200 when both models are loaded. Predictions, captions and attention are returned by the actual bundle runtimes after loading succeeds.
Do not feed an unverified `.pt` checkpoint to `torch.load` just to bypass that state.

## Notebooks on Kaggle

1. Choose the notebook using `notebooks/README.md`, not only its historical filename.
2. Upload that notebook and attach the same dataset/model inputs used previously.
3. Set the notebook's input paths if the Kaggle attachment paths differ, then run in order.
4. For a new experiment, use its own output directory. For a true resume, keep the original protocol and matching assets.

Notebook cells, metadata and outputs were not rewritten during organization.
Official DTD split configuration, 50-epoch settings, caption policies and cached-input identities are unchanged.
The four caption-method classifier notebooks require their matching CSV in
`data/captions/dtd/grounded_controls/`; uploading a different method CSV does not bypass the hash guards.
Caption-generation notebooks still need complete local Florence/CLIP checkpoints and matching offline wheels.
Do not install the legacy Florence requirements into a working Transformers 5 environment without planning an isolated runtime.

## Local Data

`configs/paths.local.example.json` is a registry/example, not a loader automatically injected into every notebook.
Copy/edit your own `configs/paths.local.json` if useful, then set the actual notebook/CLI path variables explicitly.
Examples relative to the project root:

- DTD: `../archive (23)/dtd`
- Stanford Cars: `../archive (10)` or `../archive (13)` (different layouts)
- CUB: `../CUB_200_2011/CUB_200_2011`
- CIFAR100: `../archive (12)/cifar100`

Raw datasets and pretrained DINOv3/CLIP/Florence weights are not bundled into this project.

## Regression Tests and Layout Audit

```powershell
python -m pip install -r requirements/dev.txt
python -m pytest -q
python tools/project_management/validate_layout.py
```

`pyproject.toml` supplies Python paths for the existing flat research modules, builders, tests and app.
Tests use small synthetic CPU examples. They do not train ViT-L/16 or certify Kaggle GPU/Transformers compatibility.
The Python interpreter used for the organization checks does not have `torchvision` installed;
the Figure 4 CLI needs the full base dependencies before it can run locally. No automatic installation was attempted.
The layout audit hashes all migrated scientific artifacts, checks every notebook against its original SHA-256,
and verifies catalog/CSV notebook links. Locator-only code updates are listed separately.

For this existing machine only, a pre-existing offline test dependency folder remains at
`../tmp/dtd-ablation-test-deps`. It is not a portable project dependency or a new installation.

## Notebook Builders

Builders are optional development tools. Running one may overwrite its generated notebook;
do not run it over a downloaded executed notebook whose outputs you want to keep.

```powershell
python tools/notebook_builders/build_mtacs_dtd_full_both_notebook.py
python tools/notebook_builders/build_dtd_caption_method04_multigrounded_notebook.py
python tools/captioning/generate_sun397_captions.py --help
python tools/visualization/visualize_figure4.py --help
```

The path resolver `project_paths.py` maps old workspace-relative source names to the cataloged locations.
It does not copy notebooks back into the project root. Archived builder dependencies must remain available.
Do not rerun the one-time `prepare_layout.py` or `relocate_code_paths.py` migration utilities.
