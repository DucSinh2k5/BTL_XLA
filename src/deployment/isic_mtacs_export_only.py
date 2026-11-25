"""Export an interrupted full_both ISIC run using its immutable captions/cache.

No training, caption regeneration or embedding replacement is performed. The
only new caption is a forced fresh-image deployment check in an offline child.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

import isic_export_only as common


def caption_directory(run, reference, assets):
    candidates = [Path(reference).expanduser()] if str(reference).strip() else [Path(run) / "shared/caption_generation"]
    if not str(reference).strip():
        for current, dirs, files in os.walk(Path(run) / "shared"):
            dirs[:] = [n for n in dirs if n not in {"generated", "caption_runtime", "__pycache__"}]
            if "web_caption_settings.json" in files:
                candidates.append(Path(current))
    matches = []
    required = {"captions.csv": "caption_csv_sha256", "text_cache/clip_text.pt": "text_cache_sha256",
                "web_caption_settings.json": "caption_settings_sha256"}
    for path in sorted(set(p.resolve() for p in candidates)):
        if all((path / name).is_file() and common.sha(path / name) == assets.get(key) for name, key in required.items()):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError("Set CAPTION_OUTPUT_DIR to the ORIGINAL shared/caption_generation folder. "
            "Its captions.csv, text_cache/clip_text.pt and web_caption_settings.json must match run_binding.json. "
            f"Matching folders: {matches}. Do not regenerate captions for this checkpoint.")
    return matches[0]


def registered_asset(reference, expected, kind, core, roots):
    """Find exactly the model files registered at training, never a substitute."""
    if not expected or any(Path(n).name != n or "\\" in n for n in expected):
        raise ValueError(f"Invalid registered {kind} asset names")
    visited, candidates = set(), []
    explicit = Path(str(reference)).expanduser() if str(reference).strip() else None
    if explicit is not None and explicit.is_dir():
        candidates.append(explicit)
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [n for n in dirs if n not in {".git", ".cache", "__pycache__", "legacy_packages", "hf_modules",
                        "images", "masks", "generated", "vendor", "candidate_images", "ground_truth_masks"}]
            if len(Path(current).relative_to(root).parts) >= 10:
                dirs[:] = []
            if "config.json" in files:
                candidates.append(Path(current))
    failures = []
    for path in candidates:
        path = path.resolve()
        if path in visited:
            continue
        visited.add(path)
        try:
            cfg = common.read_json(path / "config.json")
        except (OSError, ValueError):
            continue
        if cfg.get("model_type") != ("florence2" if kind == "florence" else "clip"):
            continue
        if any(not (path / name).is_file() for name in expected):
            continue
        if core.checkpoint_asset_identity(path) == expected:
            return path
        failures.append(str(path))
    raise FileNotFoundError(f"Attach the SAME complete local {kind} model used to create the saved captions. "
        f"Registered files/hashes do not match any checkpoint. Set {kind.upper()}_MODEL_REF explicitly. Checked: {failures}")


def validate_captions(ctx, reference=""):
    core = importlib.import_module("isic_caption_core")
    folder = caption_directory(ctx["run"], reference, ctx["binding"]["assets"])
    required = ("status.json", "generation_protocol/protocol.json", "text_cache/protocol.json")
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing original caption provenance: {folder / name}")
    registered = common.read_json(folder / "web_caption_settings.json")
    generation = common.read_json(folder / "generation_protocol/protocol.json")
    text_protocol = common.read_json(folder / "text_cache/protocol.json")
    records = [{key: row[key] for key in ("image_id", "image_path", "split", "image_sha256")}
               for role in ("train", "val", "test") for row in ctx["frames"][role]]
    rows, status = core.validated_export({"output": str(folder), "records": records})
    images = [{key: row[key] for key in ("image_id", "split", "image_sha256")} for row in records]
    code_sha = common.sha(ctx["runtime"] / "isic_caption_core.py")
    settings = registered.get("settings", {})
    if (registered.get("policy") != core.POLICY or registered.get("method") != core.METHOD
            or registered.get("model_id") != core.MODEL_ID or settings.get("precision") != "float32"
            or generation.get("dtype") != "float32" or generation.get("settings") != settings
            or generation.get("method") != core.METHOD or generation.get("model_id") != core.MODEL_ID
            or generation.get("policy") != core.POLICY or generation.get("images") != images
            or generation.get("code_sha256") != code_sha or common.object_hash(generation) != status["identity"]
            or generation.get("model_assets") != registered.get("florence_assets")):
        raise ValueError("Original caption generation/settings/code/image binding is inconsistent; do not replace it")
    if (text_protocol.get("caption_csv_sha256") != status["csv_sha256"]
            or text_protocol.get("clip_assets") != registered.get("clip_assets")
            or text_protocol.get("policy") != core.POLICY or text_protocol.get("code_sha256") != code_sha):
        raise ValueError("CLIP text protocol does not match this original caption CSV/code/checkpoint")
    cache_path = folder / "text_cache/clip_text.pt"
    table = ctx["work"].TextTable(cache_path)
    core.validate_text_cache(table.data, rows, status["csv_sha256"], common.object_hash(text_protocol))
    if table.data["embeddings"].dtype != torch.float32 or table.data["embeddings"].shape != (len(rows), ctx["config"]["text_dim"]):
        raise ValueError("Expected the original FP32 512-dimensional CLIP text table")
    if ctx["config"]["text_dim"] != 512:
        raise ValueError("This full_both deployment uses CLIP ViT-B/32 text_dim=512")
    coverage = {}
    for role, frame in [*ctx["frames"].items(), ("candidate_pool", ctx["pool"])]:
        _, valid = table.gather(frame, "cpu")
        coverage[role] = {"images": len(frame), "valid": int(valid.sum())}
    if not coverage["train"]["valid"] or not coverage["candidate_pool"]["valid"]:
        raise ValueError("The saved full_both run has no usable query/candidate training text")
    ctx.update(caption_root=folder, caption_registered=registered, caption_rows=rows, text_table=table,
               caption_core=core, caption_coverage=coverage)
    # Prefer a usable validation caption for the cached path parity test.
    ctx["example_record"] = next((r for r in ctx["frames"]["val"] if bool(table.data["valid"][table.positions[r["image_id"]]])), ctx["frames"]["val"][0])
    return ctx


def prepare(source_run, dataset_root, dino_repo, dino_weights, output, fallback_sources,
            *, caption_output="", florence_ref="", clip_ref="", offline_assets="", device="cuda",
            feature_batch_size=4, run_full_test=False, factory=None):
    ctx = common.prepare(source_run, dataset_root, dino_repo, dino_weights, output, fallback_sources,
        mode="mtacs", device=device, feature_batch_size=feature_batch_size, run_full_test=run_full_test, factory=factory)
    for name in ("isic_caption_core", "isic_caption_web"):
        loaded = sys.modules.get(name)
        if loaded and Path(loaded.__file__).resolve() != ctx["runtime"] / (name + ".py"):
            raise RuntimeError("Another caption runtime is imported. Restart the kernel and Run All.")
    validate_captions(ctx, caption_output)
    roots = [Path(offline_assets), ctx["run"], Path("/kaggle/input")]
    core = ctx["caption_core"]
    asset_paths = {key: registered_asset(ref, ctx["caption_registered"][key + "_assets"], key, core, roots)
                   for key, ref in (("florence", florence_ref), ("clip", clip_ref))}
    if factory is None:
        core.validate_florence_prefix_folder(asset_paths["florence"])
        importlib.import_module("isic_caption_web").validate_clip_assets(asset_paths["clip"])
    ctx.update(caption_asset_paths=asset_paths, offline_assets=offline_assets, stage_extra=stage_captions,
               cold_source=COLD_SOURCE)
    common.write_json(ctx["output"] / "caption_coverage.json", ctx["caption_coverage"])
    print("Original captions and text cache verified:", ctx["caption_root"])
    print("Caption coverage:", ctx["caption_coverage"])
    return ctx


def stage_wheels(ctx):
    core = ctx["caption_core"]
    destination = ctx["stage"] / "offline_wheels"
    roots = [ctx["caption_root"] / "caption_runtime", Path(ctx["offline_assets"]), Path("/kaggle/input")]
    dirs = core.offline_caption_wheels(roots, ctx["output"] / "collected_wheels")
    prefixes = ("transformers-", "tokenizers-", "huggingface_hub-", "einops-", "timm-", "accelerate-")
    wheel_map = {}
    for folder in dirs:
        for path in sorted(Path(folder).glob("*.whl")):
            if path.name.startswith(prefixes):
                common.copy_verified(path, destination / path.name)
                wheel_map[path.name] = common.sha(path)
    if ctx["factory"] is None and any(not any(n.startswith(prefix) for n in wheel_map)
            for prefix in ("transformers-4.49.", "tokenizers-0.21.", "huggingface_hub-0.")):
        raise FileNotFoundError("Attach offline wheels for transformers 4.49.x, tokenizers 0.21.x and huggingface-hub<1 "
            "using OFFLINE_ASSETS_ROOT. The old caption_runtime/wheels can also be reused. "
            "Installed legacy_packages are platform-specific and are not copied into the web bundle.")
    common.write_json(ctx["stage"] / "offline_wheels_manifest.json", wheel_map)


def stage_captions(ctx):
    stage, folder = ctx["stage"], ctx["caption_root"]
    for key, path in ctx["caption_asset_paths"].items():
        shutil.copytree(path, stage / "caption_models" / key, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__", "*.pyc"))
        if ctx["caption_core"].checkpoint_asset_identity(stage / "caption_models" / key) != ctx["caption_registered"][key + "_assets"]:
            raise ValueError("Copied caption model files differ from their trained asset binding")
    for original, target in (("captions.csv", "captions.csv"), ("web_caption_settings.json", "caption_settings.json"),
                             ("text_cache/clip_text.pt", "text_embeddings.pt")):
        common.copy_verified(folder / original, stage / target)
    for name in ("generation_protocol/protocol.json", "text_cache/protocol.json", "text_cache/status.json", "status.json"):
        if (folder / name).is_file():
            common.copy_verified(folder / name, stage / "caption_provenance" / name)
    common.write_json(stage / "caption_lookup.json", {r["image_id"]: r for r in ctx["caption_rows"]})
    common.write_json(stage / "caption_coverage.json", ctx["caption_coverage"])
    pixels, index, ambiguous = {}, {}, []
    for rows in ctx["frames"].values():
        for row in rows:
            pixels.setdefault(row["pixel_sha256"], []).append(row["image_id"])
    lookup = {r["image_id"]: r for r in ctx["caption_rows"]}
    table = ctx["text_table"]
    for digest, ids in pixels.items():
        first = table.positions[ids[0]]
        equal = all(lookup[identity]["final_caption"] == lookup[ids[0]]["final_caption"]
                    and torch.equal(table.data["embeddings"][table.positions[identity]], table.data["embeddings"][first])
                    and bool(table.data["valid"][table.positions[identity]]) == bool(table.data["valid"][first]) for identity in ids)
        if equal:
            index[digest] = ids[0]
        else:
            ambiguous.append({"pixel_sha256": digest, "image_ids": ids, "action": "generate_fresh_caption_for_uploaded_pixels"})
    ctx["caption_pixel_index"] = index
    common.write_json(stage / "caption_pixel_ambiguities.json", ambiguous)
    stage_wheels(ctx)
    (stage / "requirements.txt").write_text("torch>=2.5\nnumpy\nPillow\ntqdm\n"
        "# Caption child (4.49.x), isolated from the segmentor kernel:\ntransformers==4.49.0\ntokenizers>=0.21,<0.22\n"
        "huggingface-hub>=0.26,<1\naccelerate\neinops\ntimm\n", encoding="utf-8")


FRESH_SOURCE = r'''import hashlib, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
root, device=Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0,str(root/'runtime'))
import isic_caption_core as core
from isic_caption_web import _LocalCaptionEngine
from tacs_visual_web import read_json, write_json, read_rgb
reference=read_json(root/'example/reference.json')
images=list((root/'example').glob('query.*'))
image=next(p for p in images if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
registered=read_json(root/'caption_settings.json')
torch.manual_seed(registered['settings']['seed'])
engine=_LocalCaptionEngine(root,device)
row,vector,valid=engine.caption(image)  # Forced generation: bypass the pixel/CSV cache.
if vector.shape!=(512,) or vector.dtype!=torch.float32 or not torch.isfinite(vector).all():
    raise ValueError('Fresh caption embedding has invalid shape/values')
if (not bool(valid) and vector.count_nonzero()) or (bool(valid) and abs(float(vector.norm())-1)>1e-4):
    raise ValueError('Fresh caption mask/normalization is invalid')
final,approved,clinical,words=core.caption_policy(row['raw_caption'])
# Token-limit rejection can make a structurally valid caption invalid; it must never become valid otherwise.
if row['final_caption']!=final or (bool(valid) and not approved):
    raise ValueError('Fresh caption does not obey the saved policy')
cache=torch.load(root/'text_embeddings.pt',map_location='cpu',weights_only=True)
lookup=read_json(root/'caption_lookup.json')
i=next(i for i,v in enumerate(cache['valid']) if bool(v))
text=lookup[cache['image_ids'][i]]['final_caption']
tokens=engine.tokenizer([text],padding=True,truncation=True,
    max_length=engine.clip.config.max_position_embeddings,return_tensors='pt')
with torch.inference_mode():
    encoded=torch.nn.functional.normalize(engine.clip(**{k:v.to(engine.device) for k,v in tokens.items()}).text_embeds.float(),dim=1)[0].cpu()
if not torch.allclose(encoded,cache['embeddings'][i],atol=5e-4,rtol=5e-4):
    raise ValueError('Reloaded CLIP does not reproduce the original trained text embedding')
rgb=read_rgb(image)
pixel_hash=hashlib.sha256(np.asarray(rgb).tobytes()+str(rgb.size).encode()).hexdigest()
torch.save({'row':row,'vector':vector.clone(),'valid':valid.clone(),
    'pixel_sha256':pixel_hash},root/'example/fresh_caption.pt')
write_json(root/'example/fresh_caption_check.json',{'forced_fresh_generation':True,'florence_clip_reload':True,
    'original_clip_embedding_parity':True,'device':device,'fresh_caption':row,'valid':bool(valid),
    'embedding_shape':list(vector.shape),'benchmark_metric':False})
print('Actual exported Florence + CLIP fresh-image check passed',flush=True)
'''


# The shared script verifies exact checkpoint logits/context and both real attention maps.
# Before it serializes results, exercise the fresh-text path using the actual
# Florence/CLIP output produced by a preceding independent process.
_FRESH_SEGMENTATION_CHECK = r'''
fresh_report=net.io_utils.read_json(root/'example/fresh_caption_check.json')
if not all(fresh_report.get(k) for k in ('forced_fresh_generation','florence_clip_reload','original_clip_embedding_parity')):
    raise ValueError('Fresh caption engine has not been independently verified')
fresh=torch.load(root/'example/fresh_caption.pt',map_location='cpu',weights_only=True)
class VerifiedFreshCaption:
    def caption(self,source):
        rgb=net.io_utils.read_rgb(source)
        digest=__import__('hashlib').sha256(np.asarray(rgb).tobytes()+str(rgb.size).encode()).hexdigest()
        if digest!=fresh['pixel_sha256']:
            raise ValueError('Fresh caption verification image changed')
        return fresh['row'],fresh['vector'],fresh['valid']
cached_index=predictor.manifest.get('caption_pixel_index',{})
predictor.manifest['caption_pixel_index']={}
predictor.caption_engine=VerifiedFreshCaption()
new_result=predictor.predict(image,return_attention=False)
predictor.manifest['caption_pixel_index']=cached_index
if new_result['evidence']['query_caption']['status']!='fresh' or not torch.isfinite(new_result['logits']).all():
    raise ValueError('Fresh full_both inference path failed')
net.io_utils.write_json(root/'example/fresh_segmentation_check.json',{
    'fresh_caption_used':True,'selected_context_id':new_result['evidence']['candidate']['image_id'],
    'query_caption':new_result['evidence']['query_caption'],'fusion':new_result['evidence']['fusion']})
'''

COLD_SOURCE = common.COLD_SOURCE.replace('torch.save(explained.pop("logits")',
    _FRESH_SEGMENTATION_CHECK + '\ntorch.save(explained.pop("logits")').replace(
    '"device":device, "benchmark_metric":False',
    '"device":device, "benchmark_metric":False, "fresh_caption_engine":True, "fresh_full_both_inference":True, "original_clip_embedding_parity":True')


def verify_and_publish(ctx, *, device=None, caption_device=None):
    if ctx["factory"] is not None:
        # Explicit tiny model fixtures cannot become ready/deployable.
        return common.verify_and_publish(ctx, device=device)
    core, output, stage = ctx["caption_core"], ctx["output"], ctx["stage"]
    script = output / "verify_fresh_caption.py"
    script.write_text(FRESH_SOURCE, encoding="utf-8")
    # Only copied wheels can repair this isolated child. Never change kernel packages.
    old_path = os.environ.pop("PYTHONPATH", None)
    try:
        env = core.caption_runtime_environment("legacy", stage / "offline_wheels", output / "caption_verify_runtime")
    finally:
        if old_path is not None:
            os.environ["PYTHONPATH"] = old_path
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    log_path = output / "fresh_caption_reload.log"
    print(f"Forced fresh-caption/CLIP check on {caption_device or device or ctx['device']}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(script), str(stage), str(caption_device or device or ctx["device"])],
            cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError("Fresh caption verification failed; nothing is marked ready.\n" + log_path.read_text("utf-8")[-6000:])
    print(log_path.read_text("utf-8")[-1600:])
    # The next child checks every bound file, including the actual fresh-caption
    # output and its CLIP parity report. Rebind hashes while status stays staged.
    common.stage_manifest(ctx)
    return common.verify_and_publish(ctx, device=device)


def package_downloads(ctx, *, part_mib=256):
    manifest = common.read_json(ctx["stage"] / "manifest.json")
    if not all(manifest.get("verification", {}).get(k) for k in
            ("fresh_caption_engine", "fresh_full_both_inference", "original_clip_embedding_parity")):
        raise ValueError("Complete M-TACS web export requires verified real Florence/CLIP fresh-image inference")
    return common.package_downloads(ctx, part_mib=part_mib)
