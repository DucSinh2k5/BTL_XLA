# DTD Caption-Method Comparison

These notebooks fix the full_both classifier and change the input CSV method.
They are separate from the eight text-context architecture/control ablations.

| Folder | Method | Caption CSV |
| --- | --- | --- |
| `01_florence_prefix/dtd` | Florence-Prefix | `captions_prefix_control.csv` |
| `02_florence_firstvalid/dtd` | Florence-FirstValid | `captions_first_valid_control.csv` |
| `03_florence_singlegrounded/dtd` | Florence-SingleGrounded | `captions_single_grounded_control.csv` |
| `04_florence_multigrounded/dtd` | Florence-MultiGrounded | `captions_multibeam_grounded.csv` |
| `05_qwen3_vl_8b/dtd` | Qwen3-VL-8B-Instruct | Attached complete Qwen8B `captions.csv` or `captions_qwen3vl_8b_instruct.csv` |
| `06_qwen3_vl_32b/dtd` | Qwen3-VL-32B-Instruct | Attached complete Qwen32B `captions.csv` or `captions_qwen3vl_32b_instruct.csv` |

The first four inputs are in `data/captions/dtd/grounded_controls/` relative to TACS; the Qwen8B/32B CSVs are attached separately.
All retain query/candidate caption fusion, visual+semantic selection, K=1 and 50 epochs by default.
The multibeam caption method does not mean three retrieved context images.
The first four notebooks have pre-bound Florence input identities/hashes; do not substitute other CSV methods.
The Qwen8B/32B notebooks verify their respective method metadata and lock each supplied CSV's actual hash at runtime.

Prefix changes valid-caption availability relative to the other controls. FirstValid vs SingleGrounded
isolates ranking more closely; MultiGrounded adds generated hypotheses. Optimized image-text CLIP scores
are not independent measures of caption quality. Report coverage/mask differences along with classifier results.

The earlier decoder-LoRA CSV and executed classifier run remain separate under
`data/captions/dtd/decoder_lora/` and `experiments/completed/caption_decoder_lora/dtd/`.
No metrics were generated or filled in during organization.

The new Qwen8B notebook also keeps the original notebook-05 classifier and 50-epoch/K=1 protocol, but binds an
attached complete Qwen8B CSV at runtime rather than assuming a locally supplied SHA256. Its own output directory,
metadata checks and hash lock prevent accidental reuse of Florence/32B captions or stale checkpoints. It has no
caption-generation fallback and does not require Qwen/Florence weights. Upload the generated notebook directly.

The paired Qwen32B notebook uses the same classifier/training and separate output/cache/checkpoint names.
Its folder number `06` is the caption-method index, not architecture ablation 06 full_shuffle_both.
It still trains notebook-05 full_both with aligned query/candidate text and a visual+semantic selector.
