"""Build DTD ablation 06 with mask-preserving query/candidate caption shuffles."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
from pathlib import Path

import build_mtacs_dtd_full_both_notebook as full
import build_mtacs_dtd_visual_only_notebook as visual


ROOT = PROJECT_ROOT
TARGET = project_path('mtacs-train-vitl16-dtd-ablation-06-full-shuffle-both.ipynb')

INTRO = """# DTD ablation 06: full_shuffle_both, ViT-L/16

Train only `full_shuffle_both`. Keep notebook 05's model class, parameter count, two text-fusion
gates and visual+semantic selector. Change only caption assignments: shuffle both query-role and
candidate-role captions, preserving each receiving image's original caption-valid mask exactly.
Invalid caption inputs remain zero and their gates stay masked. Images are never shuffled.

Shuffle valid query captions separately within each official split AND training-pool membership
group. A train query never borrows a validation/test caption; pool and non-pool captions never
cross groups. Candidate-role caption donors are restricted to the training pool. No ground-truth
labels or class matching enter the permutations. Query/candidate maps use independent local RNGs
with offsets 1009/2017 from the training seed, and stay fixed across epochs and evaluation/resume.
Valid groups with at least two images are deranged (no original donor ID); a singleton stays aligned
to preserve its mask and is explicitly recorded. Different donor IDs can carry identical caption
strings, so reports distinguish changed donor IDs from changed text. Independent maps can coincide
by chance, especially in two-image groups; they are not forced to have different donors across roles.

Fusion receives query image + its assigned query-role caption + one selected context image + that
image's assigned candidate-role caption. The selector uses the same assigned caption vectors as
fusion, not the original aligned text. Score = visual_z + sigmoid(semantic_mix_logit) * semantic_z.
The mixing coefficient starts at 0.5 and is learned with the same policy/entropy objective as 05.
Invalid text pairs contribute zero semantic evidence; unavailable query text falls back to visual
selection. A context's image index gathers its assigned text. Candidate labels are never inputs.

Default: official DTD split 1, seed 42, 50 epochs, a 50-epoch LR schedule, no early stopping, best
checkpoint selected by validation loss. All data, pool, model initialization, optimizer, losses and
augmentation settings match notebook 05. Train/val/test each have 1,880 images; the fixed pool holds
376 training images. A query cannot select its own image; this excludes image identity, not the
caption donor ID introduced by the negative-control permutation. Test opens after the epoch budget.
Held-out query-caption shuffles are deliberate negative controls using other images' generated
captions within that held-out split, never their labels. This is not a deployable independent-image
inference pipeline, nor a fixed-context intervention: shuffled text may also change context selection.

Use the same `dtd_lexical_v2` policy as notebook 05 BEFORE shuffling: remove generic openings,
retain 5-8 words, screen direct DTD class names. CLIP leakage scores are diagnostic-only and do not
prove absence of synonyms/implicit class information. Masks are not recomputed after shuffling.
Do not compare this run with a baseline using different captions, masks, seeds, split or LR horizon.

Attach DTD, local DINOv3 code/ViT-L weights and CLIP ViT-B/32. Reuse a complete DTD captions.csv
from notebook 05/04/02/03 or create/resume it using local Florence-2. Generation receives pixels
only and flushes finished batches. For offline generation, attach
`/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi` with Florence-2-base and compatible wheels.
Legacy Florence uses an isolated subprocess; there are no network downloads or kernel downgrades.
Configure CAPTIONS_INPUT_PATH, FLORENCE_MODEL_REF and OFFLINE_ASSETS_ROOT for other local paths.

The original CSV is never shuffled or overwritten. shared/caption_source_inputs.csv contains the
cleaned PRE-SHUFFLE text/masks; the frozen CLIP cache is also pre-shuffle. Each run exports
shuffle_text_assignments.csv (receiver/donor IDs), shuffle_model_inputs.csv (actual assigned text,
both masks, split/pool provenance) and shuffle_text_manifest.json (including singleton IDs).
full_shuffle_both_shuffle_summary.csv reports mask preservation and valid-caption shuffle rates.
selector_semantic_mix remains in logs, history and metrics. Output/checkpoints are isolated in
outputs/mtacs_dtd_split1_ablation_06_full_shuffle_both_vitl16; only this run's last.pt can resume it.
Previous notebooks and outputs are unchanged. Run all cells in order with the same GPU/batch setup.
"""

BANK_FUNCTION = """def make_ablation_banks(spec, seed, override=None):
    settings = {**spec, **(override or {})}
    if (settings["name"] != "full_shuffle_both" or settings["view"] != "screened"
            or not settings["query_fusion"] or not settings["candidate_fusion"] or not settings["semantic_selector"]
            or settings["q_transform"] != "shuffle" or settings["c_transform"] != "shuffle"):
        raise ValueError("This notebook supports full_shuffle_both only: both text roles shuffled, fusion/semantic selector on.")
    view = TEXT_CACHE["views"]["screened"]
    ids = view["image_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in the caption cache")
    lookup = {image_id: index for index, image_id in enumerate(ids)}
    split_by_id = {str(image_id): split for split, frame in SPLIT_FRAMES.items() for image_id in frame.image_id}
    if set(ids) != set(split_by_id):
        raise ValueError("Caption cache image IDs do not match query splits")
    training_ids = set(SPLIT_FRAMES["train"].image_id.astype(str))
    candidate_ids = set(CANDIDATE_IDS)
    if candidate_ids - training_ids:
        raise ValueError("Candidate text bank must contain training images only")
    positions = torch.tensor([lookup[image_id] for image_id in CANDIDATE_IDS], dtype=torch.long)
    valid = torch.as_tensor(view["valid"], dtype=torch.bool).cpu()
    banks = {"settings": settings, "q_index": lookup, "singletons": {}}
    for role, offset in (("q", 1009), ("c", 2017)):
        sources, singletons = deranged_sources(ids, valid, split_by_id, CANDIDATE_IDS, int(seed) + offset)
        donors = torch.as_tensor(sources, dtype=torch.long)
        matrix = view["embeddings"].index_select(0, donors).clone()
        role_valid = valid.clone()
        matrix[~role_valid] = 0
        if role == "c":
            matrix = matrix.index_select(0, positions).to(DEVICE)
            role_valid = role_valid.index_select(0, positions).to(DEVICE)
            donors = donors.index_select(0, positions)
            singletons = [image_id for image_id in singletons if image_id in candidate_ids]
        banks["singletons"][role] = singletons
        banks[role] = {"embeddings": matrix, "valid": role_valid,
                       "source_ids": [ids[int(index)] for index in donors],
                       "texts": [view["texts"][int(index)] for index in donors]}
    return banks


def shuffle_role_inputs(banks):
    view = TEXT_CACHE["views"]["screened"]
    lookup = {image_id: index for index, image_id in enumerate(view["image_ids"])}
    split_by_id = {str(image_id): split for split, frame in SPLIT_FRAMES.items() for image_id in frame.image_id}
    pool = set(CANDIDATE_IDS)
    rows = []
    for role, ids in (("q", view["image_ids"]), ("c", CANDIDATE_IDS)):
        role_valid = banks[role]["valid"].cpu().tolist()
        singletons = set(banks["singletons"][role])
        for index, image_id in enumerate(ids):
            donor = banks[role]["source_ids"][index]
            original_valid = bool(view["valid"][lookup[image_id]])
            donor_valid = bool(view["valid"][lookup[donor]])
            if role_valid[index] != original_valid or role_valid[index] != donor_valid:
                raise ValueError("Caption shuffle changed a recipient mask or borrowed an invalid donor")
            if split_by_id[image_id] != split_by_id[donor] or (image_id in pool) != (donor in pool):
                raise ValueError("Caption shuffle crossed split or training-pool boundaries")
            text = banks[role]["texts"][index]
            rows.append({"role": role, "image_id": image_id, "caption_source_id": donor,
                "split": split_by_id[image_id], "caption_source_split": split_by_id[donor],
                "image_in_candidate_pool": image_id in pool, "source_in_candidate_pool": donor in pool,
                "caption_available": role_valid[index], "original_caption_available": original_valid,
                "donor_caption_available": donor_valid, "model_caption": text,
                "original_caption": view["texts"][lookup[image_id]],
                "source_id_changed": image_id != donor,
                "text_changed": text != view["texts"][lookup[image_id]],
                "singleton_not_deranged": image_id in singletons})
    return pd.DataFrame(rows)
"""

SHUFFLE_REPORT = '''
shuffle_tables = []
for training_seed in TRAIN_SEEDS:
    path = RUNS_DIR / "full_shuffle_both" / f"seed_{training_seed}" / "shuffle_model_inputs.csv"
    if not path.is_file():
        continue
    assignments = pd.read_csv(path).fillna("")
    for key in ("caption_available", "original_caption_available", "source_id_changed", "text_changed", "singleton_not_deranged"):
        assignments[key] = assignments[key].map(bool_value)
    for (role, split, in_pool), group in assignments.groupby(["role", "split", "image_in_candidate_pool"]):
        available = group.loc[group.caption_available]
        shuffle_tables.append({"seed": training_seed, "role": role, "split": split,
            "in_candidate_pool": in_pool, "images": len(group), "valid_captions": len(available),
            "mask_preserved_rate": float(group.caption_available.eq(group.original_caption_available).mean()),
            "changed_source_id_rate_valid": float(available.source_id_changed.mean()),
            "changed_text_rate_valid": float(available.text_changed.mean()),
            "singletons_not_deranged": int(available.singleton_not_deranged.sum())})
shuffle_summary = pd.DataFrame(shuffle_tables, columns=["seed", "role", "split", "in_candidate_pool", "images",
    "valid_captions", "mask_preserved_rate", "changed_source_id_rate_valid", "changed_text_rate_valid", "singletons_not_deranged"])
shuffle_summary.to_csv(OUT_PUT_DIR / "full_shuffle_both_shuffle_summary.csv", index=False)
display(shuffle_summary)
with (OUT_PUT_DIR / "full_shuffle_both_report.md").open("a", encoding="utf-8") as handle:
    handle.write("\\n## Mask-preserving caption shuffle\\n\\n" + markdown_table(shuffle_summary) +
        "\\n\\nRates of changed source/text are measured among valid captions only. Singleton groups remain aligned. "
        "Different donor IDs can carry identical text. Held-out query shuffles are negative controls, not independent-image deployment.\\n")
'''


def build_notebook():
    notebook = copy.deepcopy(full.build_notebook())
    code = {index: visual.source_of(cell) for index, cell in enumerate(notebook["cells"])}
    changes = {0: INTRO}
    changes[1] = code[1].replace("full-both", "full-shuffle-both")
    config = code[2].replace("full_both", "full_shuffle_both").replace("DTD_FULL_BOTH_SEED", "DTD_FULL_SHUFFLE_BOTH_SEED")
    changes[2] = visual.replace_once(config,
        '"q_transform": "aligned", "c_transform": "aligned"',
        '"q_transform": "shuffle", "c_transform": "shuffle"')
    changes[4] = code[4].replace("DTD_FULL_BOTH_OUTPUT", "DTD_FULL_SHUFFLE_BOTH_OUTPUT").replace(
        "ablation_05_full_both_vitl16", "ablation_06_full_shuffle_both_vitl16").replace("DTD_FULL_BOTH_EPOCHS", "DTD_FULL_SHUFFLE_BOTH_EPOCHS")
    stage = code[9].split("def make_ablation_banks", 1)[0].replace("full_both", "full_shuffle_both")
    stage = visual.replace_once(stage,
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n',
        'caption_source = ensure_dtd_caption_csv(SPLIT_FRAMES, CAPTIONS_INPUT_PATH, CAPTION_CSV, [\n'
        '    PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_ablation_05_full_both_vitl16" / "shared" / "captions.csv",\n')
    stage = stage.replace("caption_model_inputs.csv", "caption_source_inputs.csv").replace(
        "full_text_embeddings.pt", "shuffle_source_text_embeddings.pt")
    derangement = visual.definitions((project_path('mtacs_dtd_ablation_cells.py')).read_text(encoding="utf-8"), ["deranged_sources"])
    changes[9] = (stage + "\n\n" + derangement + "\n\n" + BANK_FUNCTION + "\n\n"
                  + visual.definitions(code[9], ["batch_caption_inputs"]))
    training = code[14].replace("full_text_manifest.json", "shuffle_text_manifest.json").replace(
        "full_text_assignments.csv", "shuffle_text_assignments.csv")
    changes[14] = visual.replace_once(training,
        '    pd.DataFrame(assignments).to_csv(run_dir / "shuffle_text_assignments.csv", index=False)',
        '    pd.DataFrame(assignments).to_csv(run_dir / "shuffle_text_assignments.csv", index=False)\n'
        '    shuffle_role_inputs(banks).to_csv(run_dir / "shuffle_model_inputs.csv", index=False)')
    protocol = code[17].replace("full_both", "full_shuffle_both").replace("full-both comparison", "full-shuffle-both comparison")
    changes[17] = visual.replace_once(protocol, '"caption_input_policy": CAPTION_INPUT_POLICY,',
        '"caption_input_policy": CAPTION_INPUT_POLICY,\n'
        '    "caption_assignment": "valid_only_derangement_within_split_and_pool",\n'
        '    "shuffle_seed_offsets": {"q": 1009, "c": 2017}, "recipient_masks": "preserved_exactly",\n'
        '    "heldout_query_caption_shuffle": "within_split_negative_control_not_deployment",')
    report = code[23].replace("full_both", "full_shuffle_both")
    report = visual.replace_once(report,
        "Query image + query caption + one context image + its caption. Both text gates are enabled; selection combines image and caption similarities.",
        "Query image + assigned query caption + one context image + its assigned candidate-role caption. Fusion and multimodal selector use these shuffled vectors.")
    changes[23] = report + "\n" + SHUFFLE_REPORT
    changes[7] = "## Create/resume source captions.csv; clean and cache before mask-preserving shuffles\n"
    changes[10] = "## Same fusion and multimodal selector as 05; both caption roles shuffled\n"
    changes[18] = "## Train full_shuffle_both only: 50 epochs by default\n"
    changes[20] = "## Final negative-control test after full_shuffle_both finishes\n"
    changes[22] = "## Metrics, shuffle/mask audit, learned semantic mix and learning curves\n"
    helpers = code[8] + changes[9] + code[11] + code[12] + changes[14] + code[15]
    utilities = code[6].split("SOURCE_NOTEBOOK_SHA256 =", 1)[0]
    changes[6] = (utilities + f"\nSOURCE_NOTEBOOK_SHA256 = {hashlib.sha256(visual.SOURCE.read_bytes()).hexdigest()!r}\n"
                  f"HELPER_SOURCE_SHA256 = {hashlib.sha256(helpers.encode()).hexdigest()!r}\n")
    for index, cell in enumerate(notebook["cells"]):
        if index in changes:
            cell["source"] = (changes[index].strip() + "\n").splitlines(keepends=True)
        cell["id"] = cell["id"].replace("dtd-full-both-", "dtd-full-shuffle-both-")
        if cell["cell_type"] == "code":
            ast.parse(visual.source_of(cell))
    notebook["metadata"]["dtd_ablation"].update(experiment="full_shuffle_both",
        default_caption_input_policy="dtd_lexical_v2", mask_preserving_shuffle=True)
    return notebook


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
