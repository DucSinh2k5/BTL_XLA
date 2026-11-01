# Completed DTD Caption-Method Runs

Imported and audited on 2026-10-24. These are executed **classifier** notebooks,
not caption-generation notebooks or blank training templates. All six use
`full_both`, DTD split 1, seed 42, 50 epochs, ViT-L/16 and one selected visual
context (`K=1`). Method identity was checked against notebook metadata and saved
CSV hashes, not inferred from the original filenames alone.

| Original filename in `notebooks/` | Method folder | Executed notebook | Test Macro-F1 |
| --- | --- | --- | ---: |
| `caption-cai-tien-prefix.ipynb` | `01_florence_prefix` | [Florence-Prefix](01_florence_prefix/mtacs-dtd-full-both-florence-prefix-split1-seed42-50ep.ipynb) | 0.810651 |
| `caption-cai-tien-first-valid.ipynb` | `02_florence_firstvalid` | [Florence-FirstValid](02_florence_firstvalid/mtacs-dtd-full-both-florence-firstvalid-split1-seed42-50ep.ipynb) | 0.809382 |
| `caption-cai-tien-single.ipynb` | `03_florence_singlegrounded` | [Florence-SingleGrounded](03_florence_singlegrounded/mtacs-dtd-full-both-florence-singlegrounded-split1-seed42-50ep.ipynb) | 0.779611 |
| `caption-cai-tien-first-multi.ipynb` | `04_florence_multigrounded` | [Florence-MultiGrounded](04_florence_multigrounded/mtacs-dtd-full-both-florence-multigrounded-split1-seed42-50ep.ipynb) | 0.790596 |
| `qwen_8b.ipynb` | `05_qwen3_vl_8b` | [Qwen3-VL-8B-Instruct](05_qwen3_vl_8b/mtacs-dtd-full-both-qwen3-vl-8b-split1-seed42-50ep.ipynb) | 0.821445 |
| `qwen_32b.ipynb` | `06_qwen3_vl_32b` | [Qwen3-VL-32B-Instruct](06_qwen3_vl_32b/mtacs-dtd-full-both-qwen3-vl-32b-split1-seed42-50ep.ipynb) | 0.820788 |

`05` and `06` number caption methods, not architecture-ablation settings.
Both Qwen runs use query/candidate text in fusion and the visual-semantic selector.

- [Full table, coverage and Vietnamese review](../../../../reports/caption_ablation/dtd/KET_QUA_SO_SANH_PHUONG_PHAP_CAPTION_DTD.md).
- [Machine-readable metric evidence](../../../../reports/caption_ablation/dtd/caption_method_output_evidence.json).
- [Original/new filenames, byte hashes and output hashes](relocation_manifest.json).
- [Runnable templates](../../../../notebooks/02_improvements/caption_ablation/).

No notebook bytes were changed during relocation. Saved outputs, plots, metadata,
widgets and execution counts remain intact. The initial organization catalog in
`manifests/notebook_catalog.json` is a historical snapshot; this README and the
relocation manifest register these six later result imports separately.

The full report also retains two historical Florence references. The older
visual-only architecture control is discussed separately, not counted as a
caption method. Single-seed/split rankings do not establish statistical significance
or eliminate label-leakage concerns.
