"""Persistent ISIC worker speaking the existing isolated inference protocol."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType
from unittest.mock import patch

from inference import EVENT_PREFIX
from worker import watch_parent, readonly_checkpoint


def emit(event, **values):
    print(EVENT_PREFIX + json.dumps({"event": event, **values}, ensure_ascii=False), file=sys.__stdout__, flush=True)


def mapped_predictor(module, root, device, caption_engine=None):
    """Restore verified, frozen CPU parameters using read-only file mappings."""
    import torch
    from accelerate import init_empty_weights
    manifest, config = module.verify_bundle(root)
    files = [root / manifest["backbone_file"], root / "models/selector.pt", root / "models/segmentor.pt",
             root / "candidate_bank.pt"]
    if config["mode"] == "mtacs":
        files.append(root / "text_embeddings.pt")
    states = {p.resolve(): readonly_checkpoint(p) for p in files}
    trainable = {}
    for name in ("selector.pt", "segmentor.pt"):
        piece = states[(root / "models" / name).resolve()]
        if set(trainable) & set(piece):
            raise ValueError("Duplicate learned ISIC parameters")
        trainable.update(piece)
    state = {"backbone." + name: value for name, value in states[(root / manifest["backbone_file"]).resolve()].items()}
    if set(state) & set(trainable):
        raise ValueError("Backbone/learned checkpoint keys overlap")
    state.update(trainable)
    def factory(cfg):
        with init_empty_weights(include_buffers=False):
            model = module.build_model(root / "vendor/dinov3", cfg)
        torch.nn.Module.load_state_dict(model, state, strict=True, assign=True)
        if any(p.is_meta for p in model.parameters()):
            raise ValueError("ISIC checkpoint did not restore every parameter")
        model.requires_grad_(False)
        def restore(self, values):
            expected = {k for k in self.state_dict() if not k.startswith("backbone.")}
            if set(values) != expected:
                raise ValueError("Learned ISIC parameter keys differ from the saved architecture")
            result = torch.nn.Module.load_state_dict(self, values, strict=False, assign=True)
            if result.unexpected_keys or any(not k.startswith("backbone.") for k in result.missing_keys):
                raise ValueError("Incomplete ISIC learned-state restoration")
            self.requires_grad_(False)
        model.restore_trainable = MethodType(restore, model)
        return model
    original_load = torch.load
    def load(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve() in states:
            return states[Path(file).resolve()]
        return original_load(file, *args, **kwargs)
    with patch.object(torch, "load", load):
        return module.Predictor(root, device, factory=factory, caption_engine=caption_engine)


def load_caption_engine(root, runtime, device):
    from isic_caption_web import BundledCaptionEngine
    if device != "cpu":
        return BundledCaptionEngine(root, device)
    from transformers import CLIPTextModelWithProjection
    from caption_memory import caption_weight_loading
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    digest = manifest["files"]["caption_models/florence/model.safetensors"]
    configured = os.getenv("SUN397_CAPTION_WEIGHT_CACHE", "")
    cache = Path(configured) if configured else runtime / "caption_fp32"
    if not (cache / (digest + ".fp32")).is_file() or not (cache / (digest + ".json")).is_file():
        cache = runtime / "caption_fp32"
    payload = {"florence": str(root / "caption_models/florence"), "clip": str(root / "caption_models/clip")}
    with caption_weight_loading(payload, cache, digest, readonly_checkpoint,
            progress=lambda message: emit("progress", message=message), clip_classes=(CLIPTextModelWithProjection,)):
        return BundledCaptionEngine(root, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tacs", "mtacs"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--caption-override", type=Path)
    args = parser.parse_args()
    if args.parent_pid:
        watch_parent(args.parent_pid)
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1")
    os.environ["HF_MODULES_CACHE"] = str(args.runtime / "hf_modules")
    try:
        import torch
        import numpy as np
        from PIL import Image
        torch.set_num_threads(args.threads)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
        root = args.bundle.resolve()
        sys.path.insert(0, str(root / "runtime"))
        emit("progress", message="Đang xác minh và nạp backbone, selector, segmentor ISIC…")
        with redirect_stdout(sys.stderr):
            module = importlib.import_module("isic_segmentation_web")
            captioner = None
            if args.model == "mtacs":
                emit("progress", message="Đang nạp Florence-2 và CLIP cho ISIC…")
                if hasattr(module, "build_model"):
                    module.verify_bundle(root)
                    captioner = load_caption_engine(root, args.runtime, device)
                else:
                    captioner = importlib.import_module("isic_caption_web").BundledCaptionEngine(root, device)
            predictor = (mapped_predictor(module, root, device, captioner) if device == "cpu" and hasattr(module, "build_model")
                         else module.Predictor(root, device, caption_engine=captioner))
            if predictor.config["mode"] != args.model:
                raise ValueError("ISIC bundle mode does not match selected model")
            predictor.model.requires_grad_(False)
            caption_experiment = None
            if args.caption_override:
                if args.model != "mtacs":
                    raise ValueError("Caption intervention is only available for M-TACS")
                from isic_caption_override import apply_override
                caption_experiment = apply_override(predictor, captioner, args.caption_override)
        emit("ready", details={"device": device, "backbone": predictor.config["backbone"],
            "image_size": predictor.config["image_size"], "candidate_count": len(predictor.pool)})
    except Exception:
        traceback.print_exc(file=sys.stderr)
        emit("error", message="Không nạp được bộ model ISIC. Xem log worker.")
        return 1
    requests = (args.runtime / "requests").resolve()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["command"] == "stop":
                break
            image, output = Path(request["image"]).resolve(), Path(request["output"]).resolve()
            if (request["command"] != "predict" or request.get("model") != args.model
                    or not image.is_relative_to(requests) or not output.is_relative_to(requests)):
                raise ValueError("Invalid request paths/model")
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                result = predictor.predict(image)
            if caption_experiment:
                result["caption_experiment"] = caption_experiment
                if result["evidence"]["query_caption"]["status"] == "fresh":
                    result["evidence"]["query_caption"]["source"] = caption_experiment["method"]
            result.pop("logits", None)
            probability = np.clip(result.pop("probabilities"), 0, 1)
            colors = np.stack((45 + 210 * probability, 82 + 48 * probability, 145 - 80 * probability), axis=-1)
            result["prediction"]["probability_url"] = module.io_utils.png_data_url(Image.fromarray(colors.astype(np.uint8)))
            result["timing"] = {"inference_seconds": round(time.perf_counter() - started, 3)}
            output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            emit("result")
        except Exception:
            traceback.print_exc(file=sys.stderr)
            emit("prediction_error", message="Suy luận ISIC thất bại. Xem log worker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
