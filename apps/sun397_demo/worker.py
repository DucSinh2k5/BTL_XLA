"""One model per process; all visual/text models stay resident after startup."""
from __future__ import annotations

import argparse
import faulthandler
from contextlib import redirect_stdout
import importlib
import json
import mmap
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import warnings
import zipfile
from types import MethodType
from unittest.mock import patch

_CHECKPOINT_MAPPINGS = []


def watch_parent(parent_pid):
    """Release this worker's models even if the API is stopped abruptly."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, parent_pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot monitor API parent process")
    def monitor():
        result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        kernel.CloseHandle(handle)
        if result == 0:
            os._exit(0)
    threading.Thread(target=monitor, daemon=True).start()


def emit(event, **values):
    print("SUN397_EVENT " + json.dumps({"event": event, **values}, ensure_ascii=False),
          file=sys.__stdout__ or sys.stdout, flush=True)


def readonly_checkpoint(path, load=None):
    """CPU inference tensors backed by a read-only mapping, without COW commit.

    Windows' default torch mmap reserves paging space for a private writable
    mapping. All parameters here are frozen and loaded by assignment, so a
    read-only buffer is sufficient. Never use these tensors for training or
    perform in-place writes to them.
    """
    import torch
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        order_names = [name for name in archive.namelist() if name.endswith("/byteorder")]
        if order_names and archive.read(order_names[0]).decode() != sys.byteorder:
            raise ValueError("Read-only loading cannot perform in-place byte-order conversion")
    with path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable.*")
        storage = torch.frombuffer(mapped, dtype=torch.uint8).untyped_storage()
    original = torch.UntypedStorage.from_file
    def from_file(filename, shared=False, size=0):
        if Path(filename).resolve() == path:
            if size != len(mapped):
                raise ValueError("Checkpoint mapping size mismatch")
            return storage
        return original(filename, shared, size)
    with patch.object(torch.UntypedStorage, "from_file", staticmethod(from_file)):
        state = (load or torch.load)(path, map_location="cpu", weights_only=True, mmap=True)
    _CHECKPOINT_MAPPINGS.append(mapped)
    return state


def mapped_factory(module, state, shared_repo=None):
    """Keep CPU weights file-backed; preserve real, nonpersistent RoPE buffers."""
    import torch
    from accelerate import init_empty_weights
    def build(repo, config):
        with init_empty_weights(include_buffers=False):
            model = module.build_model(shared_repo or repo, config)
        model.load_state_dict(state["model_state_dict"], strict=True, assign=True)
        if any(parameter.is_meta for parameter in model.parameters()):
            raise ValueError("Checkpoint did not materialize every model parameter")
        def strict_load(self, values, strict=True, assign=False):
            # BundlePredictor loads again after .to(device). CPU can rebind the
            # same mapped tensors; CUDA must copy into its existing GPU weights.
            cpu = next(self.parameters()).device.type == "cpu"
            return torch.nn.Module.load_state_dict(self, values, strict=strict, assign=cpu)
        model.load_state_dict = MethodType(strict_load, model)
        return model
    return build


def load_predictor(model, bundle, runtime, device, threads, shared_bundle=None):
    import torch
    torch.set_num_threads(max(1, threads))
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng worker Python chưa có CUDA khả dụng.")
    sys.path.insert(0, str(bundle / "runtime"))
    if shared_bundle is not None:
        own_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        base_manifest = json.loads((shared_bundle / "manifest.json").read_text(encoding="utf-8"))
        def shared_sources(manifest):
            return {name: info for name,info in manifest["files"].items()
                    if name.startswith("vendor/dinov3/") or name == "runtime/tacs_visual_web.py"}
        if shared_sources(own_manifest) != shared_sources(base_manifest):
            raise ValueError("Shared worker requires identical verified DINOv3 and visual runtime sources")
    emit("progress", message="Đang kiểm tra bundle và nạp DINOv3 ViT-L/16…")
    module = importlib.import_module("tacs_visual_web" if model == "tacs" else "mtacs_multimodal_web")
    # Verification inside the original predictor remains intact. Mapping avoids
    # simultaneous anonymous copies of >1 GiB weights during constructor/load.
    weights_file = (bundle / "model.pt").resolve()
    state = readonly_checkpoint(weights_file) if device == "cpu" else torch.load(
        weights_file, map_location="cpu", weights_only=True, mmap=True)
    original_load = torch.load
    def load(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve() == weights_file:
            return state
        return original_load(file, *args, **kwargs)
    with redirect_stdout(sys.stderr):
        with patch.object(torch, "load", load):
            factory = mapped_factory(module, state, shared_bundle / "vendor/dinov3" if shared_bundle else None)
            if model == "tacs":
                predictor = module.BundlePredictor(bundle, device=device, model_factory=factory)
            else:
                predictor = module.BundlePredictor(bundle, device=device, model_factory=factory,
                                                   cache_dir=runtime / "caption_cache")
    if model == "mtacs":
        emit("progress", message="Đang nạp sẵn Florence-2 và CLIP cho caption ảnh mới…")
        captions = importlib.import_module("sun397_caption_web")
        payload = {**predictor.caption_settings, "device": device,
                   "florence": str(bundle / "models/florence"), "clip": str(bundle / "models/clip")}
        with redirect_stdout(sys.stderr):
            if device == "cpu":
                from caption_memory import caption_weight_loading
                source_sha = predictor.manifest["files"]["models/florence/model.safetensors"]["sha256"]
                cache_directory = Path(os.getenv("SUN397_CAPTION_WEIGHT_CACHE", str(runtime / "caption_fp32")))
                with caption_weight_loading(payload, cache_directory, source_sha, readonly_checkpoint,
                        progress=lambda message: emit("progress", message=message)):
                    engine = captions.CaptionEngine(payload)
            else:
                engine = captions.CaptionEngine(payload)
        def caption_runner(request):
            if request["settings"] != payload["settings"] or request["classes"] != payload["classes"]:
                raise ValueError("Caption request không khớp policy/lớp của model đã nạp.")
            image = captions.visual.read_rgb(request["image"])
            with redirect_stdout(sys.stderr):
                rows, vectors, valid = engine.audit(engine.generate([image]))
            return {"rows": rows, "embeddings": vectors, "valid": valid}
        predictor.caption_runner = caption_runner
    predictor.model.requires_grad_(False)
    return predictor, {"device": device, "backbone": predictor.config["backbone"],
                       "image_size": predictor.config["image_size"],
                       "candidate_count": predictor.config["candidate_count"],
                       "caption_models_loaded": model == "mtacs"}


def main():
    faulthandler.enable(file=sys.stderr, all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tacs", "mtacs", "both"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tacs-bundle", type=Path)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid:
        watch_parent(args.parent_pid)
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
    args.runtime.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(args.runtime / "hf_modules")
    try:
        if args.model == "both":
            if args.tacs_bundle is None:
                raise ValueError("Shared worker needs both bundle paths")
            mtacs, mtacs_details = load_predictor("mtacs", args.bundle.resolve(), args.runtime.resolve(),
                                                args.device, args.threads)
            tacs, tacs_details = load_predictor("tacs", args.tacs_bundle.resolve(), args.runtime.resolve(),
                                               args.device, args.threads, shared_bundle=args.bundle.resolve())
            predictors = {"tacs": tacs, "mtacs": mtacs}
            details = {"shared_process": True, "models": {"tacs": tacs_details, "mtacs": mtacs_details}}
        else:
            predictor, details = load_predictor(args.model, args.bundle.resolve(), args.runtime.resolve(),
                                                 args.device, args.threads)
            predictors = {args.model: predictor}
        emit("ready", details=details)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1
    requests = (args.runtime / "requests").resolve()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["command"] == "stop":
                break
            if request["command"] != "predict":
                raise ValueError("Unknown worker command")
            selected_model = request.get("model", args.model)
            predictor = predictors[selected_model]
            image, output = Path(request["image"]).resolve(), Path(request["output"]).resolve()
            if not image.is_relative_to(requests) or not output.is_relative_to(requests):
                raise ValueError("Inference paths must stay in the request directory")
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                result = predictor.predict(str(image), explain=True)
            if selected_model == "mtacs":
                evidence = result.get("evidence", {})
                candidate = evidence.get("candidate", {})
                caption_rows = getattr(predictor, "caption_rows", {})
                candidate_caption = caption_rows.get(candidate.get("image_id"), {})
                if isinstance(candidate_caption, dict):
                    candidate["raw_text"] = candidate_caption.get("raw_caption", "")
            result["timing"] = {"inference_seconds": round(time.perf_counter() - started, 3)}
            output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            emit("result")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("prediction_error", message=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
