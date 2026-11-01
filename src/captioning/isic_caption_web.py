"""ISIC Florence-Prefix generation/resume and a portable fresh-image adapter.

isic_caption_core.py is embedded by the notebook builder from the existing,
tested ISIC caption pipeline. Transformers never imports in the training kernel.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

import isic_caption_core as core
import tacs_visual_web as io


def validate_clip_assets(folder):
    folder = Path(folder)
    config = io.read_json(folder / "config.json")
    if (config.get("model_type") != "clip" or config.get("projection_dim") != 512
            or config.get("vision_config", {}).get("patch_size") != 32):
        raise ValueError("Use a full local CLIP ViT-B/32, not a different text embedding space")
    if not (folder / "tokenizer_config.json").is_file() or not (
            (folder / "tokenizer.json").is_file() or all((folder / name).is_file() for name in ("vocab.json", "merges.txt"))):
        raise FileNotFoundError("CLIP tokenizer files are incomplete; fix Inputs before generating captions")
    weights = list(folder.glob("*.safetensors")) + list(folder.glob("pytorch_model*.bin"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise FileNotFoundError("CLIP weights are missing/empty")
    for index in folder.glob("*.index.json"):
        names = set(io.read_json(index).get("weight_map", {}).values())
        if not names or any(not io.bundle_path(folder, name).is_file() for name in names):
            raise FileNotFoundError("CLIP weight index has missing shards")
    return folder


def prepare_captions(frames, florence_ref, clip_ref, offline_assets, output, settings, runtime_dir):
    if settings.get("precision") != "float32":
        raise ValueError("This deployment protocol fixes Florence/CLIP precision to float32")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = [Path(offline_assets), Path("/kaggle/input")]
    florence, backend = core.florence_checkpoint(florence_ref, roots)
    if backend != "legacy":
        raise ValueError("This method fixes Microsoft Florence-2-base legacy + isolated Transformers 4.49")
    core.validate_florence_prefix_folder(florence)
    clip = core.resolve_clip_model(clip_ref, roots)
    validate_clip_assets(clip)
    records = [{key: row[key] for key in ("image_id", "image_path", "split", "image_sha256")}
               for role in ("train", "val", "test") for row in frames[role]]
    # Explicit whitelist prevents masks, diagnoses and clinical metadata entering
    # either the caption prompt or its generation worker payload.
    core.validate_caption_records(records)
    worker = Path(runtime_dir) / "isic_caption_core.py"
    payload = {"records": records, "output": str(output), "settings": dict(settings),
        "model_path": str(florence), "clip_path": str(clip), "max_new_images": 0,
        "code_sha256": io.file_sha256(worker)}
    request = output / "request.json"
    io.write_json(request, payload)
    env = core.caption_runtime_environment("legacy", offline_assets, output / "caption_runtime")
    for phase in ("generate", "encode"):
        core.run_worker(worker, request, phase, env, output / f"{phase}.log")
    rows, status = core.validated_export(payload)
    cache_file = output / "text_cache/clip_text.pt"
    cache = torch.load(cache_file, map_location="cpu", weights_only=True)
    core.validate_text_cache(cache, rows, status["csv_sha256"], cache["identity"])
    if cache["embeddings"].shape[1] != 512:
        raise ValueError("Unexpected CLIP text dimension")
    valid_by_split = {role: sum(row["split"] == role and row["caption_valid"].lower() == "true" for row in rows)
                      for role in ("train", "val", "test")}
    if valid_by_split["train"] == 0:
        raise ValueError("No valid training captions; inspect the CSV before calling this full_both")
    io.write_json(output / "web_caption_settings.json", {"settings": settings, "policy": core.POLICY,
        "method": core.METHOD, "model_id": core.MODEL_ID, "valid_by_split": valid_by_split,
        "florence_assets": core.checkpoint_asset_identity(florence), "clip_assets": core.checkpoint_asset_identity(clip),
        "source": "pixel_only_automatic_caption_not_ground_truth"})
    return cache_file, {"florence": str(florence), "clip": str(clip), "offline_assets_root": str(offline_assets)}


class _LocalCaptionEngine:
    def __init__(self, root, device):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.root = Path(root)
        registered = io.read_json(self.root / "caption_settings.json")
        if registered["policy"] != core.POLICY or registered["method"] != core.METHOD:
            raise ValueError("Caption policy/method differs from the training export")
        if registered["settings"].get("precision") != "float32":
            raise ValueError("Caption precision differs from the FP32 training export")
        if not importlib.metadata.version("transformers").startswith("4.49."):
            raise RuntimeError("The local caption worker requires Transformers 4.49.x")
        settings = {**registered["settings"], "device": str(device)}
        self.captioner = core.IsicFlorencePrefixCaptioner(self.root / "caption_models/florence", settings)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.root / "caption_models/clip"), local_files_only=True)
        self.clip, info = CLIPTextModelWithProjection.from_pretrained(str(self.root / "caption_models/clip"),
            local_files_only=True, output_loading_info=True)
        missing = [key for key in info.get("missing_keys", []) if not key.endswith("position_ids")]
        if missing or info.get("mismatched_keys"):
            raise ValueError("The exported CLIP text tower did not load completely")
        self.clip = self.clip.to(self.device).eval().requires_grad_(False)

    def caption(self, image):
        raw, limited = self.captioner.generate([io.read_rgb(image)])[0]
        final, valid, clinical, words = core.caption_policy(raw, limited)
        vector = torch.zeros(512, dtype=torch.float32)
        if valid:
            tokens = self.tokenizer([final], padding=True, truncation=True,
                max_length=self.clip.config.max_position_embeddings, return_tensors="pt")
            with torch.inference_mode():
                values = self.clip(**{key: value.to(self.device) for key, value in tokens.items()}).text_embeds
                values = torch.nn.functional.normalize(values.float(), dim=1)[0].cpu()
            # Copy into a normal tensor: no in-place mutation of an inference
            # tensor outside InferenceMode (the previous web caption failure).
            vector = values.clone()
        row = {"raw_caption": raw, "final_caption": final, "caption_valid": valid,
            "clinical_term_detected": clinical, "word_count": words, "caption_policy": core.POLICY}
        return row, vector, torch.tensor(valid, dtype=torch.bool)


class BundledCaptionEngine:
    """Persistent local models on 4.49; isolated offline child on another version.

For a production web deployment, keep this object in a dedicated compatible
worker so the local models are reused rather than loaded for every request.
"""
    def __init__(self, bundle, device="cpu", offline_assets=None, runtime=None):
        self.root, self.device = Path(bundle).resolve(), str(device)
        self.local = None
        try:
            version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            version = ""
        if version.startswith("4.49."):
            self.local = _LocalCaptionEngine(self.root, device)
        else:
            assets = offline_assets or os.getenv("ISIC_CAPTION_OFFLINE_ASSETS") or self.root / "offline_wheels"
            runtime = runtime or Path(tempfile.gettempdir()) / "isic_caption_runtime" / io.file_sha256(self.root / "caption_settings.json")[:16]
            self.environment = core.caption_runtime_environment("legacy", assets, runtime)

    def caption(self, image):
        if self.local is not None:
            return self.local.caption(image)
        with tempfile.TemporaryDirectory(prefix="isic_caption_") as directory:
            directory = Path(directory)
            image_file, result_file = directory / "query.png", directory / "result.pt"
            io.read_rgb(image).save(image_file)
            command = [sys.executable, str(self.root / "runtime/isic_caption_web.py"), "--bundle", str(self.root),
                "--image", str(image_file), "--result", str(result_file), "--device", self.device]
            completed = subprocess.run(command, env=self.environment, text=True, capture_output=True, timeout=900)
            if completed.returncode:
                raise RuntimeError("Bundled caption worker failed: " + completed.stderr[-4000:])
            result = torch.load(result_file, map_location="cpu", weights_only=True)
            return result["row"], result["vector"], result["valid"]


def probe_export(bundle, device, offline_assets, runtime):
    # Force generation even for a known image: the segmentor's pixel cache must
    # not hide missing Florence/CLIP weights in a supposedly complete export.
    engine = BundledCaptionEngine(bundle, device, offline_assets, runtime)
    row, vector, valid = engine.caption(Path(bundle) / "example/query.png")
    if vector.shape != (512,) or not torch.isfinite(vector).all() or (not bool(valid) and vector.count_nonzero()):
        raise ValueError("Fresh exported caption engine has invalid vector/mask output")
    return {"fresh_caption": row, "valid": bool(valid), "embedding_shape": list(vector.shape)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    for name in ("bundle", "image", "result"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    engine = _LocalCaptionEngine(arguments.bundle, arguments.device)
    row, vector, valid = engine.caption(io.read_rgb(arguments.image))
    torch.save({"row": row, "vector": vector, "valid": valid}, arguments.result)
