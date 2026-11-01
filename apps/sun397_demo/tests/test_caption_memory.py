import hashlib
from pathlib import Path

import pytest


def test_frozen_hf_models_load_exact_fp32_weights_from_readonly_cache(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    from sun397_demo.caption_memory import caption_weight_loading, file_hash, fp32_cache
    from sun397_demo.worker import readonly_checkpoint
    text = transformers.CLIPTextConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
        bos_token_id=0, eos_token_id=2, pad_token_id=1)
    vision = transformers.CLIPVisionConfig(hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8)
    config = transformers.CLIPConfig.from_text_vision_configs(text, vision, projection_dim=4)
    source = transformers.CLIPModel(config).half().eval()
    florence, clip = tmp_path / "florence", tmp_path / "clip"
    source.save_pretrained(florence, safe_serialization=True)
    source.float().save_pretrained(clip, safe_serialization=False)
    source_file = florence / "model.safetensors"
    sha = file_hash(source_file)
    original_bytes = source_file.read_bytes()
    expected = transformers.CLIPModel.from_pretrained(florence, torch_dtype=torch.float32,
                                                      local_files_only=True).eval()
    payload = {"florence": str(florence), "clip": str(clip)}
    with caption_weight_loading(payload, tmp_path / "cache", sha, readonly_checkpoint):
        restored = transformers.CLIPModel.from_pretrained(florence, torch_dtype=torch.float32,
                                                          local_files_only=True).eval()
        restored_bin = transformers.CLIPModel.from_pretrained(clip, torch_dtype=torch.float32,
                                                              local_files_only=True).eval()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(restored_bin.state_dict()[key], value, rtol=0, atol=0)
    inputs = {"input_ids": torch.tensor([[0,3,2,1]]), "pixel_values": torch.rand(1,3,16,16)}
    with torch.inference_mode():
        expected_logits = expected(**inputs).logits_per_image
        torch.testing.assert_close(restored(**inputs).logits_per_image, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(restored_bin(**inputs).logits_per_image, expected_logits, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in restored.parameters())
    assert source_file.read_bytes() == original_bytes
    raw, info = fp32_cache(source_file, tmp_path / "cache", sha)
    assert info["source_sha256"] == sha and raw.stat().st_size == info["bytes"]
    assert file_hash(raw) == info["sha256"]


def test_corrupt_derived_cache_is_rejected(tmp_path):
    pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    import torch
    from sun397_demo.caption_memory import fp32_cache, file_hash
    source = tmp_path / "source.safetensors"
    safetensors.save_file({"values":torch.tensor([1.0,-0.5,3.0], dtype=torch.float16)}, source,
                          metadata={"format":"pt"})
    sha = file_hash(source)
    raw, _ = fp32_cache(source, tmp_path / "cache", sha)
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fp32_cache(source, tmp_path / "cache", sha)


def test_cache_uses_windows_temp_when_project_volume_is_full(tmp_path, monkeypatch):
    import json
    import struct
    from types import SimpleNamespace
    from sun397_demo import caption_memory
    header = json.dumps({"__metadata__":{"format":"pt"},
                         "values":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}).encode()
    source = tmp_path / "source.safetensors"
    source.write_bytes(struct.pack("<Q",len(header)) + header + struct.pack("<ee",1.0,-0.25))
    project_cache = tmp_path / "full_volume"
    temporary = tmp_path / "temp_volume"
    monkeypatch.setattr(caption_memory.tempfile,"gettempdir",lambda:str(temporary))
    monkeypatch.setattr(caption_memory.shutil,"disk_usage",lambda path:SimpleNamespace(
        free=0 if Path(path)==project_cache else 1024**3))
    raw, info = caption_memory.fp32_cache(source,project_cache,caption_memory.file_hash(source))
    assert raw.parent == temporary / "sun397_caption_fp32"
    assert struct.unpack("<ff",raw.read_bytes()) == (1.0,-0.25)
    assert info["bytes"] == 8
