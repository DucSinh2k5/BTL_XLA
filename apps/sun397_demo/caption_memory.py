"""Exact FP32, read-only CPU weight loading for the frozen caption models."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import tempfile
import warnings
from unittest.mock import patch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fp32_cache(source, directory, expected_sha, progress=None):
    """Stream source F16/F32 tensors into a derived FP32 file using bounded RAM."""
    import numpy as np
    source, directory = Path(source), Path(directory)
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
    raw, index = directory / (expected_sha + ".fp32"), directory / (expected_sha + ".json")
    if raw.is_file() and index.is_file():
        info = json.loads(index.read_text(encoding="utf-8"))
        if (info["source_sha256"] != expected_sha or info["bytes"] != raw.stat().st_size
                or file_hash(raw) != info["sha256"]):
            raise ValueError("Derived caption cache checksum mismatch; preserve it for inspection")
        return raw, info
    if file_hash(source) != expected_sha:
        raise ValueError("Caption weights differ from their verified source identity")
    with source.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    expected_bytes = sum(math.prod(item["shape"]) * 4 for item in tensors.values())
    if shutil.disk_usage(directory).free < expected_bytes + 64 * 1024**2:
        fallback = Path(tempfile.gettempdir()) / "sun397_caption_fp32"
        if directory.resolve() != fallback.resolve():
            if progress:
                progress("Ổ chứa project thiếu chỗ trống; dùng thư mục cache tạm của Windows…")
            return fp32_cache(source, fallback, expected_sha, progress)
        raise OSError("Not enough disk space for the reusable FP32 Florence cache")
    temporary = raw.with_suffix(".tmp")
    info = {"source_sha256": expected_sha, "metadata": header.get("__metadata__"), "tensors": {}}
    digest, written, last_progress = hashlib.sha256(), 0, -1
    with source.open("rb") as incoming, temporary.open("wb") as output:
        for name, item in tensors.items():
            if item["dtype"] not in {"F16", "F32"}:
                raise ValueError("This CPU caption cache supports source F16/F32 weights only")
            count = math.prod(item["shape"])
            start, end = item["data_offsets"]
            element_bytes = 2 if item["dtype"] == "F16" else 4
            if end - start != count * element_bytes or end + header_size + 8 > source.stat().st_size:
                raise ValueError("Invalid safetensors tensor bounds")
            incoming.seek(8 + header_size + start)
            info["tensors"][name] = {"shape": item["shape"], "offset": written, "count": count}
            remaining = end - start
            while remaining:
                chunk = incoming.read(min(remaining, 512 * 1024))
                if not chunk:
                    raise ValueError("Truncated caption weight file")
                values = np.frombuffer(chunk, dtype="<f2" if element_bytes == 2 else "<f4")
                if not np.isfinite(values).all():
                    raise ValueError("Caption weights contain nonfinite values")
                data = values.astype("<f4", copy=False).tobytes()
                output.write(data)
                digest.update(data)
                written += len(data)
                remaining -= len(chunk)
            percent = int(10 * written / max(expected_bytes, 1))
            if progress and percent != last_progress:
                progress(f"Đang chuẩn bị cache Florence trên CPU: {min(percent * 10, 100)}%")
                last_progress = percent
        output.flush()
        os.fsync(output.fileno())
    if written != expected_bytes:
        raise ValueError("Converted caption cache size mismatch")
    info.update(bytes=written, sha256=digest.hexdigest())
    temporary.replace(raw)
    index.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return raw, info


class ReadonlySafeWeights:
    def __init__(self, path, info):
        with Path(path).open("rb") as handle:
            self.mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.info = info

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Returned tensors keep the mapping alive; do not invalidate their storage.
        return False

    def metadata(self):
        return self.info["metadata"]

    def keys(self):
        return list(self.info["tensors"])

    def get_tensor(self, name):
        import torch
        item = self.info["tensors"][name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable.*")
            return torch.frombuffer(self.mapping, dtype=torch.float32, count=item["count"],
                                    offset=item["offset"]).reshape(item["shape"])


@contextmanager
def caption_weight_loading(payload, directory, expected_sha, checkpoint_loader, progress=None):
    """Retain original HF constructors/checks, replacing only CPU storage loading."""
    import torch
    from transformers import CLIPModel
    from transformers import modeling_utils
    florence_file = (Path(payload["florence"]) / "model.safetensors").resolve()
    clip_file = (Path(payload["clip"]) / "pytorch_model.bin").resolve()
    original_safe, original_safe_load = modeling_utils.safe_open, modeling_utils.safe_load_file
    original_torch_load, original_clip = torch.load, CLIPModel.from_pretrained
    safe_weights, clip_weights = None, None
    def weights():
        nonlocal safe_weights
        if safe_weights is None:
            raw, info = fp32_cache(florence_file, directory, expected_sha, progress)
            safe_weights = ReadonlySafeWeights(raw, info)
        return safe_weights
    def safe_open(filename, *args, **kwargs):
        if Path(filename).resolve() == florence_file:
            return weights()
        return original_safe(filename, *args, **kwargs)
    def safe_load(filename, *args, **kwargs):
        if Path(filename).resolve() == florence_file:
            reader = weights()
            return {name: reader.get_tensor(name) for name in reader.keys()}
        return original_safe_load(filename, *args, **kwargs)
    def torch_load(filename, *args, **kwargs):
        nonlocal clip_weights
        if isinstance(filename, (str, Path)) and Path(filename).resolve() == clip_file:
            if clip_weights is None:
                clip_weights = checkpoint_loader(clip_file, load=original_torch_load)
            return clip_weights
        return original_torch_load(filename, *args, **kwargs)
    def clip_from_pretrained(cls, *args, **kwargs):
        kwargs["low_cpu_mem_usage"] = True
        return original_clip(*args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(modeling_utils, "safe_open", safe_open))
        stack.enter_context(patch.object(modeling_utils, "safe_load_file", safe_load))
        stack.enter_context(patch.object(torch, "load", torch_load))
        stack.enter_context(patch.object(CLIPModel, "from_pretrained", classmethod(clip_from_pretrained)))
        yield
