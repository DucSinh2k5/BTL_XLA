# Workspace Migration

## Scope

The TACS-related subset of `XLA` was moved into `XLA/TACS`. This is an organization change,
not a new training run, a modification of the paper protocol, or an automatic checkpoint conversion.

- 154 move entries covering 211 inventoried files and 78 notebooks.
- Every notebook, CSV input, `.pt` artifact, paper PDF and saved plot retains its original bytes.
- Builders and test locators were updated for the new paths; the caption notebook manifest now contains project-relative notebook paths.
- Original code and the original caption manifest are backed up in `manifests/original_code_before_path_updates.zip`.
- Existing code-generated cache files are not treated as scientific artifacts in the hash inventory.

## Lookup

- [Move plan: old path to new path](../manifests/migration_plan.json)
- [Per-file original hashes](../manifests/migration_files.json)
- [Notebook catalog and configuration evidence](../manifests/notebook_catalog.json)
- [Latest content-preservation check](../manifests/migration_validation.json)

All `old` paths are relative to `XLA`; all `new` paths are relative to `TACS`.
For example, the base SUN397 notebook moved from `mtacs-train-captions-vitl16.ipynb`
to `notebooks/02_improvements/query_text/sun397/mtacs-train-captions-vitl16.ipynb`.
The old DTD combined notebook is intentionally in `notebooks/90_archive/dtd/combined_ablation_suite/`
because standalone builders still use it as a source.

## Left Outside

Raw dataset archives, `.venv`, Git/editor metadata, shared caches, coursework, UML and unrelated documents remain in `XLA`.
The original broad `XLA/requirements.txt` was not moved or repurposed.
The annotation review's external `node_modules` junction remains under `XLA/tmp`; only actual evidence files were moved.
The historical review scripts and extracted inspection fragments are evidence, not current runnable entry points.
Their old path literals are preserved; use the current caption notebooks for new annotation work.

## Future Moves

Update notebook catalogs, builder paths and CSV manifests together, and run tests plus the content audit.
Do not delete archived notebook sources just because no new experiment should run from them.
Do not replace historical output paths inside saved metrics with fabricated modern paths.
For a rollback, consult the move map and code backup; no automatic destructive rollback command is provided.
