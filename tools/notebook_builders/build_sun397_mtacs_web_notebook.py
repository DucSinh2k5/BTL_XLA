"""Create notebook 2: standalone SUN397 full_both M-TACS and complete web bundle."""
from __future__ import annotations

import ast
import json

from _project_bootstrap import PROJECT_ROOT
from build_sun397_tacs_web_notebook import cell, build_notebook as visual_notebook


TARGET = PROJECT_ROOT / "notebooks/04_deployment/sun397/02_mtacs_full_both/sun397-mtacs-full-both-vitl16-train-export-web.ipynb"
SOURCES = {name: PROJECT_ROOT / path for name, path in {
    "tacs_visual_web.py": "src/models/tacs_visual_web.py",
    "train_sun397_tacs_web.py": "src/training/train_sun397_tacs_web.py",
    "sun397_caption_web.py": "src/captioning/sun397_caption_web.py",
    "mtacs_multimodal_web.py": "src/models/mtacs_multimodal_web.py",
    "train_sun397_mtacs_web.py": "src/training/train_sun397_mtacs_web.py",
}.items()}


def build_legacy_notebook():
    payload = {name: path.read_text(encoding="utf-8") for name, path in SOURCES.items()}
    for source in payload.values():
        ast.parse(source)
    original = visual_notebook()
    paths = "".join(original["cells"][4]["source"]).replace("sun397_tacs_visual_vitl16_web_v1", "sun397_mtacs_full_both_vitl16_web_v1")
    paths = paths.replace("SUN397_TACS_WEB_", "SUN397_MTACS_WEB_")
    paths += '''
FLORENCE_MODEL_DIR = Path(os.getenv("FLORENCE_MODEL_REF", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base"))
CLIP_MODEL_DIR = Path(os.getenv("CLIP_MODEL_LOCAL_PATH", "/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit"))
OFFLINE_ASSETS_ROOT = Path(os.getenv("OFFLINE_ASSETS_ROOT", "/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi"))
CAPTION_BATCH_SIZE = int(os.getenv("SUN397_CAPTION_BATCH", "8" if GPU_GIB >= 40 else "4"))
CAPTION_MAX_NEW_TOKENS = 48
'''
    cells = [cell("markdown", '''# SUN397 M-TACS full_both ViT-L/16: captions, training and complete web export

Notebook 2 of the deployment series. **SUN397, DINOv3 ViT-L/16 (1024 dimensions), 384px, K=1**. Query image and selected candidate image are fused at patch-token level. Their Florence-generated captions are encoded by frozen CLIP ViT-B/32 and enter two gated residuals. The selector uses eligible visual z-scores plus a learned sigmoid-weighted caption-cosine z-score. An invalid caption has an exactly zero vector and zero effective gate; its image is still eligible. This is `full_both`, not query_only/candidate_only/top3.

Same partition/validation/pool policy and default 100-epoch budget as notebook 1: 17,865 train, 1,985 val, 19,850 test and a fixed 3,573-image training-only pool, seed42. Pool queries cannot select themselves. Best checkpoint uses validation CE; no test-label prompts, test-based hyperparameter selection or online training. This L/16,384px,cached-bank,single-seed deployment adaptation is not an exact reproduction of the paper's S/16,256px benchmark or a byte-compatible replacement for older M-TACS checkpoints.

This notebook **generates/resumes a new full 39,700-row captions.csv**, caches aligned frozen CLIP vectors, trains M-TACS and exports all visual and text inference artifacts. Do not paste an unregistered historical DTD/SUN CSV into shared/. Old captions may have another prompt/policy/checkpoint and are not silently reused. Neither Florence nor CLIP is fine-tuned here.

**Caption policy `sun397_lexical_clip_audit_v1`**: remove boilerplate, keep up to eight readable words, require 5-8, and reject direct SUN397 class terms. CLIP class-prompt cosine is recorded for audit only; threshold0.30 is NOT treated as proof of leakage or an automatic semantic veto. This deliberately avoids interpreting semantic association as certain leakage. Synonyms and implicit class clues may remain: the checks do not establish ground truth or guarantee leak-free text. It is a declared new SUN policy, not the DTD-only policy or a promise to preserve old metrics. Inspect the caption validity coverage before training.

Attach SUN397 + official partitions, local DINOv3 source/license and L/16 LVD1689M .pth, complete original Microsoft Florence-2-base, complete CLIP ViT-B/32, and offline wheels from notebook-mt-moi if the kernel is not compatible with legacy Florence. The source is embedded; no project .py upload is required. Internet stays off. Legacy Florence runs in a child with Transformers4.49.x; offline pip --no-index may prepare an isolated target, never downgrade the notebook kernel. No native/community/Qwen checkpoint substitution.

The ZIP includes full best M-TACS, the frozen Florence and CLIP weights/tokenizers/processors/model code, every pool image/caption, ordered visual/text banks, all caption records, exact preprocessing, inference adapter/CLI, real attention, metrics, source/license notices, hashes and a fresh-caption smoke check. It is not an OS/Python/CUDA installer. Kaggle Linux wheels are not Windows wheels; prepare compatible dependencies on the web host.

**Real attention, no extra attention model:** query classifier CLS-to-query-patch attention and selected candidate standalone selector-encoder CLS-to-patch attention, last block averaged over heads. This is visual backbone attention, not text attribution, causal proof, Grad-CAM or a complete recreation of every Figure4 panel. Text gates and semantic mix are shown separately. The existing FastAPI application is not modified by this notebook; connect the returned bundle adapter after training.

References: [TACS](https://arxiv.org/abs/2512.00489), [DINOv3](https://github.com/facebookresearch/dinov3), [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base), [CLIP API 4.49](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/clip).'''),
        cell("markdown", "## 1 Offline kernel preflight\nNo Transformers import in the training kernel. Caption workers are isolated; tokenizer parallelism is disabled before imports."),
        original["cells"][2],
        cell("markdown", '''## 2 Inputs, output and budget

Correct the paths below before Run All. Use the official `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`. Florence requires config, weights, local Python model code, processor and tokenizer; CLIP requires its full model/tokenizer folder. OFFLINE_ASSETS_ROOT contains offline_wheels.zip/wheels.zip or compatible wheel files. No CSV input is required: this notebook creates its own registered CSV.

EPOCHS=100 matches notebook1's default. Set50 before a new run if intentionally choosing a shorter budget for both models; do not change the horizon on resume. Keep train batch>=2 and effective batch64. CPU pool pixels need about3.2GB; leave RAM/VRAM for L/16 and optimizer. Caption models are released when the caption child ends, before classifier training. The full export/ZIP additionally needs several GB of free disk. No silent size/backbone fallback.'''),
        cell("code", paths),
        cell("markdown", "## 3 Embedded runtime and fixed model configuration"),
        cell("code", '''import json
RUNTIME_DIR = OUTPUT_DIR / "embedded_runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_SOURCES = ''' + repr(payload) + '''
for name, source in EMBEDDED_SOURCES.items():
    path = RUNTIME_DIR / name
    if path.is_file() and path.read_text(encoding="utf-8") != source:
        raise ValueError("Embedded source changed under this output directory; preserve it and start a fresh run")
    path.write_text(source, encoding="utf-8")
sys.path.insert(0, str(RUNTIME_DIR.resolve()))
import tacs_visual_web as visual
import train_sun397_tacs_web as common
import sun397_caption_web as caption_service
import mtacs_multimodal_web as mtacs
import train_sun397_mtacs_web as workflow
for module in (visual, common, caption_service, mtacs, workflow):
    if Path(module.__file__).resolve() != (RUNTIME_DIR / Path(module.__file__).name).resolve():
        raise RuntimeError("A different runtime is already imported. Restart the kernel and Run All")
CONFIG = workflow.default_config()
CONFIG.update(epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH_SIZE,
              workers=NUM_WORKERS, seed=SEED, image_size=IMAGE_SIZE,
              feature_batch_size=16 if GPU_GIB >= 40 else 8, eval_batch_size=8 if GPU_GIB >= 40 else 4)
workflow.validate_config(CONFIG)
common.seed_all(SEED)
SHARED_DIR = OUTPUT_DIR / "shared"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_SETTINGS = {"policy": caption_service.POLICY, "task": caption_service.PROMPT,
                    "max_new_tokens": CAPTION_MAX_NEW_TOKENS, "batch_size": CAPTION_BATCH_SIZE,
                    "precision": "float32", "semantic_audit_threshold": 0.30, "aliases": []}
display(CONFIG)
display(CAPTION_SETTINGS)'''),
        cell("markdown", "## 4 Validate SUN397 and frozen caption assets\nMissing images, wrong class count, different checkpoint formats and incomplete shards stop before training. Test labels are not passed to Florence."),
        cell("code", '''for path in (DATASET_ROOT, TRAIN_PARTITION, TEST_PARTITION, DINOV3_REPO_DIR):
    if not path.exists():
        raise FileNotFoundError(f"Required Input missing: {path}; fix section 2")
WEIGHTS_FILE = common.resolve_weights(DINOV3_WEIGHTS)
MODEL_ASSETS = caption_service.validate_assets(FLORENCE_MODEL_DIR, CLIP_MODEL_DIR)
class_names, train_rows, val_rows, test_rows, candidate_rows = common.load_sun397(DATASET_ROOT, TRAIN_PARTITION, TEST_PARTITION, CONFIG)
print("Train/val/test/pool:", len(train_rows), len(val_rows), len(test_rows), len(candidate_rows))
ASSETS = {"pretrained_weights_sha256": visual.file_sha256(WEIGHTS_FILE),
          "dino_source_sha256": common.source_tree_identity(DINOV3_REPO_DIR),
          "train_partition_sha256": visual.file_sha256(TRAIN_PARTITION), "test_partition_sha256": visual.file_sha256(TEST_PARTITION),
          "caption_assets": MODEL_ASSETS}
common.csv_write(OUTPUT_DIR / "test_split.csv", test_rows)
CAPTION_ENV = caption_service.caption_environment(SHARED_DIR / "caption_runtime", OFFLINE_ASSETS_ROOT)
probe = caption_service.invoke_worker({"mode": "probe", "image": val_rows[0]["image_path"], "classes": class_names,
    "settings": CAPTION_SETTINGS, "florence": MODEL_ASSETS["florence"], "clip": MODEL_ASSETS["clip"], "device": str(DEVICE)},
    CAPTION_ENV, work_dir=SHARED_DIR)
assert tuple(probe["embeddings"].shape) == (1, 512)
display(probe["rows"][0])
print("Real Florence/CLIP child preflight passed; kernel Transformers was not changed")'''),
        cell("markdown", '''## 5 Generate/resume captions.csv and frozen CLIP embeddings

Input to the captioner is image pixels plus the fixed caption task, never a row label/path hint. The class list is used afterward for lexical and similarity audit only. Each completed batch is fsynced. Registered image hashes, source, prompt, settings and frozen assets must match on resume. A new Kaggle session may have empty working/: restore the previous full run folder, including shared/, embedded_runtime, best.pt, last.pt and protocol.json before Run All. A deployment ZIP intentionally does not contain optimizer state.

Policy note: semantic_similarity_flag is only an audit signal; caption_valid is determined by readable length and direct lexical leakage. A masked caption stays visible in the demo but has zero text vector/gate. No silent threshold adjustment based on test results.'''),
        cell("code", '''caption_records = []
for record in common.tqdm(train_rows + val_rows + test_rows, desc="Bind captions to image bytes"):
    caption_records.append({"image_id": record["image_id"], "image_path": record["image_path"], "split": record["split"],
                            "image_sha256": visual.file_sha256(record["image_path"])})
TEXT_CACHE, CAPTION_ENV = workflow.prepare_captions(caption_records, class_names, MODEL_ASSETS, CAPTION_SETTINGS,
                                                  SHARED_DIR, DEVICE, OFFLINE_ASSETS_ROOT, CAPTION_ENV)
TEXT_TABLE = workflow.TextTable(TEXT_CACHE, SHARED_DIR / "captions.csv", train_rows + val_rows + test_rows, CONFIG)
coverage = []
for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows), ("candidate_pool", candidate_rows)):
    _, valid = TEXT_TABLE.gather([r["image_id"] for r in rows], "cpu")
    coverage.append({"split": split, "images": len(rows), "valid_captions": int(valid.sum()), "valid_fraction": float(valid.float().mean())})
display(coverage)
visual.write_json(SHARED_DIR / "caption_coverage.json", coverage)
if coverage[0]["valid_captions"] == 0 or coverage[3]["valid_captions"] == 0:
    raise RuntimeError("Query or candidate text branch has no valid training captions. Inspect the audit; do not run a mislabeled full_both experiment")
ASSETS.update(caption_csv_sha256=visual.file_sha256(SHARED_DIR / "captions.csv"),
              text_cache_sha256=visual.file_sha256(SHARED_DIR / "text_embeddings.pt"),
              caption_settings_sha256=visual.file_sha256(SHARED_DIR / "caption_settings.json"))
pool_ids = {c["image_id"] for c in candidate_rows}
pool_hashes = {r["image_sha256"] for r in caption_records if r["image_id"] in pool_ids}
EXPORT_EXAMPLE = next(r for r in val_rows if visual.file_sha256(r["image_path"]) not in pool_hashes)
print("Complete caption CSV:", SHARED_DIR / "captions.csv")'''),
        cell("markdown", "## 6 Construct full_both L16 and check real attention\nSame visual token/RoPE path as notebook1. Caption vectors affect both fusion and selector. Attention capture must not change predictions."),
        cell("code", '''model = mtacs.build_model(DINOV3_REPO_DIR, CONFIG, WEIGHTS_FILE).to(DEVICE).eval()
probe_images = torch.stack([visual.preprocess(r["image_path"], CONFIG) for r in candidate_rows[:2]]).to(DEVICE)
probe_text, probe_valid = TEXT_TABLE.gather([r["image_id"] for r in candidate_rows[:2]], DEVICE)
with torch.no_grad():
    native = model.backbone.forward_features(probe_images)["x_norm_clstoken"]
    assert torch.allclose(native, model.encode_selector(probe_images), atol=1e-4, rtol=1e-4)
    plain = model(probe_images[:1], probe_images[1:], probe_text[:1], probe_valid[:1], probe_text[1:], probe_valid[1:])
    explained, details = model(probe_images[:1], probe_images[1:], probe_text[:1], probe_valid[:1],
                               probe_text[1:], probe_valid[1:], return_attention=True)
    assert torch.allclose(plain, explained, atol=1e-5, rtol=1e-5)
    assert details["attention"]["query_classifier"].shape[-2:] == (IMAGE_SIZE // 16, IMAGE_SIZE // 16)
    scores, _ = model.score(probe_images, native, [0, 1], probe_text, probe_valid, probe_text, probe_valid)
    assert scores.argmax(1).tolist() == [1, 0]
    assert torch.isfinite(plain).all()
print("L16/384/K1 full_both preflight passed", details["q_gate"].tolist(), details["c_gate"].tolist())
del probe_images, probe_text, probe_valid, native, plain, explained, details, scores
torch.cuda.empty_cache()
common.seed_all(SEED)'''),
        cell("markdown", '''## 7 Train/resume M-TACS

Hard Gumbel task gradients flow through selected image pixels and selected text vectors. The independent categorical policy action uses that action's matching candidate image AND caption in its reward. The reward baseline retains the same query caption, but removes candidate image/text. Pool labels are never selector/fusion inputs. Log q_gate/c_gate/semantic_mix; zero gates can be correct when captions are masked.

Best uses validation CE. Keep last.pt, best.pt, protocol and shared text assets to resume; no horizon/seed/pool/CSV change under the same run. Microbatch advantage normalization, cached candidate keys and this new caption policy are declared differences from the original paper/older notebooks.'''),
        cell("code", '''if RUN_TRAINING:
    BEST, HISTORY = workflow.train_run(model, class_names, train_rows, val_rows, candidate_rows, TEXT_TABLE,
                                       CONFIG, ASSETS, OUTPUT_DIR, DEVICE)
else:
    complete = visual.read_json(OUTPUT_DIR / "training_complete.json")
    BEST = torch.load(OUTPUT_DIR / "best.pt", map_location="cpu", weights_only=True)
    expected = workflow.run_identity(CONFIG, class_names, {"train": train_rows, "val": val_rows, "pool": candidate_rows}, ASSETS)
    if BEST["identity"] != expected or complete["identity"] != expected:
        raise ValueError("Completed checkpoint does not match this configuration/caption cache")
    model.load_state_dict(BEST["model_state_dict"], strict=True)
    model.eval()
display({"best_epoch": BEST["epoch"], "validation": BEST["val_metrics"]})'''),
        cell("markdown", "## 8 Final test from best and fixed caption cache\nRebuild final visual bank in fp32. Test captions are generated from their own image pixels; test labels are used only for metrics. No candidate pool expansion or test-time training."),
        cell("code", '''if not RUN_FINAL_TEST:
    raise RuntimeError("RUN_FINAL_TEST must be True before publishing the evaluated bundle")
complete = visual.read_json(OUTPUT_DIR / "training_complete.json")
if complete["identity"] != BEST["identity"] or complete["epochs_completed"] != CONFIG["epochs"]:
    raise RuntimeError("Finish training before final evaluation/export")
model.load_state_dict(BEST["model_state_dict"], strict=True)
model.eval()
FINAL_BANK = common.refresh_bank(model, candidate_rows, CONFIG, DEVICE, amp=False)
TEST_METRICS, TEST_PREDICTIONS = workflow.evaluate(model, common.make_loader(test_rows, CONFIG), FINAL_BANK,
    candidate_rows, TEXT_TABLE, CONFIG, DEVICE, label="SUN397 M-TACS final test / fp32", amp=False)
visual.write_json(OUTPUT_DIR / "test_metrics.json", dict(TEST_METRICS, best_epoch=BEST["epoch"], precision="float32"))
common.csv_write(OUTPUT_DIR / "test_predictions.csv", TEST_PREDICTIONS)
display(TEST_METRICS)
del FINAL_BANK
torch.cuda.empty_cache()'''),
        cell("markdown", '''## 9 Export every model and verify a new offline process

Include frozen Florence and CLIP assets, not only their names or old Kaggle paths. Copy the exact trained DINO backbone, gates, projection, classifier and semantic mix. Save pool images, checkpoint-bound visual features, image-ID-aligned candidate captions/vectors and the all-image cache. The cold process loads only bundled source/assets, checks cached-query attention parity, then forces a real fresh Florence caption + CLIP encoding. Parent best and reloaded model must choose the same candidate and agree on logits for that exact fresh vector. No stub/fallback prediction is permitted.

Attention is recomputed from the trained backbone. The predictor displays masked captions honestly and still uses visual evidence. A complete verified ZIP can be reused on re-export; a mismatched/partial one is preserved and rejected. Keep last.pt separately if training resume is needed.'''),
        cell("code", '''if not RUN_EXPORT:
    raise RuntimeError("Set RUN_EXPORT=True to produce the complete web bundle")
EXPORT_ZIP = workflow.export_bundle(model, BEST, candidate_rows, class_names, TEXT_TABLE, CONFIG, DINOV3_REPO_DIR,
    OUTPUT_DIR, EXPORT_EXAMPLE, TEST_METRICS, TEST_PREDICTIONS, MODEL_ASSETS, OFFLINE_ASSETS_ROOT)
print("Download this ZIP:", EXPORT_ZIP)'''),
        cell("markdown", "## 10 Inspect real captions, selected context and attention\nThese outputs come from the reloaded best model and a fresh caption worker, not placeholders. Existing web adapter wiring is performed after obtaining this bundle."),
        cell("code", '''import base64
from IPython.display import FileLink, Image as DisplayImage, display
bundle = EXPORT_ZIP.with_suffix("")
result = visual.read_json(bundle / "examples/prediction.json")
display(visual.read_json(bundle / "smoke_test.json"))
display(result["prediction"])
display(result["evidence"]["query_caption"])
print("Selected candidate:", result["evidence"]["candidate"]["image_id"])
print("Candidate caption:", result["evidence"]["candidate"]["caption"])
display(DisplayImage(data=base64.b64decode(result["evidence"]["candidate"]["image_url"].split(",", 1)[1])))
display(result["evidence"]["fusion"])
for role, item in result["evidence"]["attention"].items():
    print(role, item["status"], item["method"])
    display(DisplayImage(data=base64.b64decode(item["image_url"].split(",", 1)[1])))
display(FileLink(str(EXPORT_ZIP)))
print(EXPORT_ZIP)
print("Download the full run directory separately when optimizer-based resume is required.")'''),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}, "tacs": {"dataset": "SUN397", "backbone": "dinov3_vitl16",
        "variant": "mtacs_full_both", "purpose": "train_export_web", "caption_policy": "sun397_lexical_clip_audit_v1"}},
        "nbformat": 4, "nbformat_minor": 5}


def build_notebook():
    from repair_sun397_mtacs_web_notebook import BACKUP, build_notebook as repaired_notebook
    return repaired_notebook() if BACKUP.is_file() else repaired_notebook(build_legacy_notebook())


if __name__ == "__main__":
    notebook = build_notebook()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET}")
