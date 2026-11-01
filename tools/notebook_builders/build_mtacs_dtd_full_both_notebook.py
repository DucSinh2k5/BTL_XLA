"""Build DTD ablation 05: both caption-fusion roles and multimodal selection."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from pathlib import Path

import build_mtacs_dtd_both_fusion_notebook as both
import build_mtacs_dtd_visual_only_notebook as visual


ROOT = PROJECT_ROOT
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb')

INTRO = """# DTD ablation 05: full_both, ViT-L/16

Only `full_both` is trained. Inputs to fusion are query image + query caption + one selected
context image + that context image's caption. Both text gates are enabled. Unlike notebook 04,
the selector also combines DINOv3 visual scores with CLIP query-caption/candidate-caption similarity.

Selector score = visual_z + sigmoid(semantic_mix_logit) * semantic_z. The two similarity sources
are standardized per query; semantic statistics use valid caption pairs only. The mixing coefficient
is trainable, initialized to 0.5, and optimized by the existing policy/entropy objective. Invalid
pairs contribute zero semantic evidence. A query without valid text uses image-only selection;
an invalid candidate caption does not remove its image from the pool. Selection samples one context
during training and uses argmax during evaluation. The chosen image index gathers its own caption.
Candidate labels are never classifier/selector inputs. The CLIP caption encoder is frozen.

This is the paired comparison to notebook 04: the model class, parameter count, fusion, data split,
pool, initialization, losses, augmentation and optimizer are unchanged; semantic_selector is now
enabled. Default: official DTD split 1, seed 42, 50 epochs with a 50-epoch LR schedule and no early
stopping. The best checkpoint is selected by validation loss. Train/val/test each have 1,880 images;
the fixed pool contains 376 training images. A query cannot select itself. Validation/test images
and their captions never enter the candidate pool. Their own image-derived query captions are
available for classification using the same image-only caption pipeline as training.

CAPTION_INPUT_POLICY defaults to `dtd_lexical_v2`, matching updated notebooks 02/03/04. Remove
generic openings before retaining 5-8 words from raw_caption; screen direct DTD class terms.
CLIP leakage-audit scores remain diagnostic-only, separate from semantic retrieval scores.
Legacy audit scores are marked when their original text differs from the cleaned model input.
Neither lexical checks nor CLIP prove absence of synonyms or implicit class information. Use the
same policy across ablations; legacy_strict is a separate experiment, not an interchangeable mask.

Attach DTD, local DINOv3 code/ViT-L weights and CLIP ViT-B/32. The notebook reuses a complete
audited DTD captions.csv from notebook 04/02/03, or creates/resumes it with local Florence-2 when
missing. Completed batches are flushed. Caption generation receives pixels only, never per-image
labels or filenames. For generation, attach `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi`
with Florence-2-base and compatible offline wheels. Legacy Florence uses an isolated subprocess;
all model/dependency loading is offline and the kernel is not downgraded. Configure
CAPTIONS_INPUT_PATH, FLORENCE_MODEL_REF and OFFLINE_ASSETS_ROOT if attachment paths differ.

The original CSV is preserved; shared/caption_model_inputs.csv records the actual input text/masks.
CLIP vectors are cached for those cleaned texts and candidate vectors follow training-pool image IDs.
Text gates are independently masked. All-invalid query/pool branches stop before training rather
than silently changing the ablation. Coverage is exported to shared/caption_role_coverage.csv.
selector_semantic_mix is logged with validation/test metrics and each epoch's validation history.

Output is isolated in outputs/mtacs_dtd_split1_ablation_05_full_both_vitl16. Resume uses this
notebook's own last.pt. Previous notebooks and checkpoints are not imported or overwritten.
Run all cells in order with the same GPU/batch settings as notebook 04. Final test opens only after
this configuration completes its epoch budget. Select settings on validation, not test scores.
"""


def build_notebook():
    notebook = copy.deepcopy(both.build_notebook())
    code = {index: visual.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: INTRO}
    changes[1] = code[1].replace("both-fusion", "full-both")
    changes[2] = code[2].replace("both_fusion", "full_both").replace("DTD_BOTH_FUSION_SEED", "DTD_FULL_BOTH_SEED")
    changes[2] = visual.replace_once(changes[2], '"semantic_selector": False', '"semantic_selector": True')
    changes[4] = code[4].replace("DTD_BOTH_FUSION_OUTPUT", "DTD_FULL_BOTH_OUTPUT").replace(
        "ablation_04_both_fusion_vitl16", "ablation_05_full_both_vitl16").replace("DTD_BOTH_FUSION_EPOCHS", "DTD_FULL_BOTH_EPOCHS")
    stage = code[9].replace("both_fusion", "full_both")
    stage = visual.replace_once(stage,
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n',
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_04_both_fusion_vitl16" / "shared" / "captions.csv",\n')
    stage = visual.replace_once(stage,
        'or not settings["candidate_fusion"] or settings["semantic_selector"]',
        'or not settings["candidate_fusion"] or not settings["semantic_selector"]')
    stage = stage.replace("both text gates on, semantic selector off", "both text gates and semantic selector on")
    stage = stage.replace("both_text_embeddings.pt", "full_text_embeddings.pt")
    changes[9] = stage
    training = code[14].replace("both_text_manifest.json", "full_text_manifest.json").replace(
        "both_text_assignments.csv", "full_text_assignments.csv")
    training = visual.replace_once(training,
        '    return metrics, frame, torch.cat(probabilities)',
        '    metrics["selector_semantic_mix"] = float(torch.sigmoid(network.selector.semantic_mix_logit).detach().cpu())\n'
        '    return metrics, frame, torch.cat(probabilities)')
    training = visual.replace_once(training,
        "f'c_gate={validation[\"mean_candidate_gate\"]:.3f}')",
        "f'c_gate={validation[\"mean_candidate_gate\"]:.3f} '\n"
        "                  f'sem_mix={validation[\"selector_semantic_mix\"]:.3f}')")
    changes[14] = training
    changes[15] = visual.replace_once(code[15],
        '"candidate_caption_rate", "mean_query_gate", "mean_candidate_gate"]',
        '"candidate_caption_rate", "mean_query_gate", "mean_candidate_gate", "selector_semantic_mix"]')
    protocol = code[17].replace("both_fusion", "full_both").replace("both-fusion comparison", "full-both comparison")
    protocol = visual.replace_once(protocol, '"selector_sampling": "categorical_policy_on_visual_z",',
        '"selector_sampling": "categorical_policy_on_visual_semantic_z",\n'
        '    "selector_score": "visual_z + sigmoid(semantic_mix_logit) * semantic_z",\n'
        '    "selector_semantic_mix_initial": 0.5, "invalid_semantic_pairs": "zero_contribution",')
    changes[17] = protocol
    report = code[23].replace("both_fusion", "full_both")
    report = visual.replace_once(report,
        "Both text gates are enabled; selection is image-only.",
        "Both text gates are enabled; selection combines image and caption similarities.")
    report += '\nselector_formula = "Selector: visual_z + sigmoid(semantic_mix_logit) * semantic_z; initial mix 0.5; invalid text pairs contribute zero."\nprint(selector_formula)\nwith (OUT_PUT_DIR / "full_both_report.md").open("a", encoding="utf-8") as handle:\n    handle.write("\\n## Multimodal selector\\n\\n" + selector_formula + "\\n\\nselector_semantic_mix in the metrics is the learned coefficient, not a leakage probability.\\n")\n'
    changes[23] = report
    changes[10] = "## Same classifier and fusion: visual + semantic selector enabled\n"
    changes[18] = "## Train full_both only: 50 epochs by default\n"
    changes[20] = "## Final test after full_both finishes\n"
    changes[22] = "## Metrics, learned semantic mix, learning curves and caption coverage\n"
    helpers = code[8] + changes[9] + code[11] + code[12] + changes[14] + changes[15]
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = (utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\n"
                  f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n")
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = cell["id"].replace("dtd-both-fusion-", "dtd-full-both-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    notebook["metadata"]["dtd_ablation"].update(experiment="full_both", default_caption_input_policy="dtd_lexical_v2")
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
