# Data and Model Inputs

`raw/` and `processed/` contain dataset placeholders only. Existing multi-gigabyte datasets stay outside TACS;
see `../configs/paths.local.example.json`. Missing datasets/models have not been downloaded.

- `captions/dtd/decoder_lora/captions (5).csv`: complete exported captions from the earlier decoder-LoRA pipeline.
- `captions/dtd/grounded_controls/`: four method-specific CSVs, unmodified input bytes and hashes.
- `annotations/dtd/`: 188 reference captions and their teacher/provenance reports.
- `archives/all_4_captions.zip`: original bundle snapshot; do not use it to overwrite newer inputs automatically.

Caption/model input files are not a training/test metrics table. For outcomes use the completed experiment evidence.
Do not rename different caption files to the same name and bypass identity guards.
