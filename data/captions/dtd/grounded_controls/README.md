# Four Caption Inputs

The CSVs are unchanged; training notebooks now live separately by method and dataset.

| CSV | Method | Training notebook |
| --- | --- | --- |
| `captions_prefix_control.csv` | Florence-Prefix | [01 Prefix](<../../../../notebooks/02_improvements/caption_ablation/01_florence_prefix/dtd/mtacs-train-vitl16-dtd-caption-ablation-01-florence-prefix.ipynb>) |
| `captions_first_valid_control.csv` | Florence-FirstValid | [02 FirstValid](<../../../../notebooks/02_improvements/caption_ablation/02_florence_firstvalid/dtd/mtacs-train-vitl16-dtd-caption-ablation-02-florence-firstvalid.ipynb>) |
| `captions_single_grounded_control.csv` | Florence-SingleGrounded | [03 SingleGrounded](<../../../../notebooks/02_improvements/caption_ablation/03_florence_singlegrounded/dtd/mtacs-train-vitl16-dtd-caption-ablation-03-florence-singlegrounded.ipynb>) |
| `captions_multibeam_grounded.csv` | Florence-MultiGrounded | [04 MultiGrounded](<../../../../notebooks/02_improvements/caption_ablation/04_florence_multigrounded/dtd/mtacs-train-vitl16-dtd-caption-ablation-04-florence-multigrounded.ipynb>) |

All four use the same classifier architecture (`full_both`), DTD split 1 and 50-epoch default budget.
Each CSV has 5,640 image IDs. Caption masks/coverage can differ; check the experiment reports when comparing outcomes.
The historical `README_CAPTION_ABLATIONS.md` is preserved verbatim; notebook names in it are historical basenames,
not claims that notebooks still sit next to these CSVs. The JSON manifest uses current project-relative paths.
