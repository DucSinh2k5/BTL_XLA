# Experiment Lifecycle

- `planned/`: empty until a concrete run configuration is registered.
- `running/`: empty; this organizer cannot establish whether an external Kaggle process is currently active.
- `completed/text_context_ablation/dtd/`: the eight original executed notebooks plus saved plots/analysis.
- `completed/caption_decoder_lora/dtd/`: saved `caption-cai-tien.ipynb`, a classifier run using LoRA-generated captions.
- `completed/caption_ablation/dtd/`: six executed caption-method classifier runs, organized by Florence/Qwen method. See its [index and preservation manifest](completed/caption_ablation/dtd/README.md).
- `completed/paper_reproduction/`: empty import area for future standardized paper run exports.

Other notebooks may also retain historical outputs; see the catalog's `cells_with_outputs` field.
The folder `completed` stores available evidence, not a certification that every protocol is correct.
Do not clear historical outputs, fabricate absent checkpoints or copy metrics from another method.
