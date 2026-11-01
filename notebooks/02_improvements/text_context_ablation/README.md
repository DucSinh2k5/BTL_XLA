# DTD Text-Context Ablation

Each configuration has its own folder and one DTD notebook. Defaults remain 50 epochs and official split 1.
Notebook content and outputs were not changed by the directory migration.

| Folder | Query text in fusion | Candidate text in fusion | Text in selector | Input control |
| --- | --- | --- | --- | --- |
| `01_visual_only/dtd` | No | No | No | Text masked; selected visual context is still used |
| `02_query_only/dtd` | Yes | No | No | Screened query captions |
| `03_candidate_only/dtd` | No | Yes | No | Screened candidate captions |
| `04_both_fusion/dtd` | Yes | Yes | No | Screened captions, visual selector |
| `05_full_both/dtd` | Yes | Yes | Yes | Screened captions, multimodal selector |
| `06_full_shuffle_both/dtd` | Yes | Yes | Yes | Shuffle both roles while preserving availability masks |
| `07_full_unscreened/dtd` | Yes | Yes | Yes | Captions before screening, no shuffle |
| `08_full_redacted/dtd` | Yes | Yes | Yes | Redact class names/aliases |

`visual_only` is NOT the paper's no-context baseline: its model still receives a context image.
The text notebooks include their existing offline create/resume caption stage when a CSV is needed.
The caption policy is `dtd_lexical_v2`; class-prompt CLIP similarity is an audit, not proof of absence of leakage.
Unscreened/redacted/shuffle runs intentionally define different text-view controls, not eight new captioners.

Use `../../01_paper_reproduction/` only for paper baselines, not for these new multimodal controls.
Executed copies and real results are under `experiments/completed/text_context_ablation/dtd/` relative to TACS.
The old combined suite is retained under `notebooks/90_archive/dtd/combined_ablation_suite/` as a builder dependency.
