"""Frozen Florence-Prefix backend for the standalone ISIC full-both notebook.

The builder embeds this backend beside the existing ISIC cache/policy/CLIP
pipeline. Worker callbacks keep that pipeline identical to the Qwen control.
"""
from __future__ import annotations

import json
from pathlib import Path

from dtd_caption_finetune import checkpoint_asset_identity
from dtd_qwen_caption import lock_protocol, runtime_versions, safetensors_parameter_count, weight_files
from isic_qwen_caption import validate_caption_records


METHOD = "florence_prefix_control"
MODEL_ID = "microsoft/Florence-2-base"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def validate_florence_prefix_folder(path):
    path = Path(path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (config.get("model_type") != "florence2"
            or "AutoModelForCausalLM" not in config.get("auto_map", {})):
        raise ValueError("Florence-Prefix requires the original legacy Microsoft Florence-2-base checkpoint")
    required = ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py",
                "preprocessor_config.json", "tokenizer_config.json")
    if any(not (path / name).is_file() for name in required):
        raise FileNotFoundError("Incomplete local Florence model code/processor/tokenizer files")
    if not ((path / "tokenizer.json").is_file() or all((path / name).is_file() for name in ("vocab.json", "merges.txt"))):
        raise FileNotFoundError("Missing Florence tokenizer vocabulary")
    if (path / "adapter_config.json").exists() or "-ft" in path.name.lower():
        raise ValueError("Use the original frozen base model, not a LoRA or -ft captioner")
    count = safetensors_parameter_count(weight_files(path))
    if not 180_000_000 <= count <= 300_000_000:
        raise ValueError(f"Expected Florence-2-base scale, not large/quantized/adapter weights: {count:,}")
    return path, count


class IsicFlorencePrefixCaptioner:
    def __init__(self, path, settings):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not transformers.__version__.startswith("4.49."):
            raise RuntimeError("Run the legacy Florence worker with the isolated offline Transformers 4.49 environment")
        self.device = torch.device("cuda:0")
        self.settings = settings
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        model, info = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=self.dtype, trust_remote_code=True, local_files_only=True,
            attn_implementation="eager", low_cpu_mem_usage=True, output_loading_info=True,
        )
        missing = [key for key in info.get("missing_keys", []) if not key.endswith("position_ids")]
        if missing or info.get("mismatched_keys"):
            raise ValueError(f"Incomplete Florence weights: {missing}")
        self.model = model.to(self.device).eval().requires_grad_(False)

    def generate(self, images):
        import torch

        inputs = self.processor(text=[TASK_PROMPT] * len(images), images=images,
                                return_tensors="pt", padding=True)
        moved = {key: value.to(device=self.device, dtype=self.dtype if key == "pixel_values" else value.dtype)
                 for key, value in inputs.items() if torch.is_tensor(value)}
        with torch.inference_mode():
            output = self.model.generate(**moved, max_new_tokens=self.settings["max_new_tokens"],
                                         num_beams=1, do_sample=False, use_cache=True)
        texts = self.processor.batch_decode(output, skip_special_tokens=False)
        if len(texts) != len(images):
            raise ValueError("Florence response count mismatch")
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        responses = []
        for text, image, sequence in zip(texts, images, output.cpu().tolist()):
            parsed = self.processor.post_process_generation(text, task=TASK_PROMPT,
                                                           image_size=(image.width, image.height))
            if not isinstance(parsed, dict) or not isinstance(parsed.get(TASK_PROMPT), str):
                raise ValueError("Florence did not return the requested detailed-caption field")
            # Florence is encoder-decoder: its first decoder token can itself be EOS.
            new_tokens = sequence[1:]
            limited = (len(new_tokens) >= self.settings["max_new_tokens"]
                       and not any(token in eos for token in new_tokens))
            responses.append((parsed[TASK_PROMPT].strip(), limited))
        return responses


def florence_worker_main(request_path, phase, *, generate_records, encode_texts):
    import torch

    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    validate_caption_records(payload["records"])
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU")
    settings = payload["settings"]
    if type(payload.get("max_new_images", 0)) is not int or payload.get("max_new_images", 0) < 0:
        raise ValueError("max_new_images must be a nonnegative integer")
    for key in ("batch_size", "max_new_tokens"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32:
        raise ValueError("Caption seed must be an integer in [0, 2**32)")
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    if phase == "generate":
        from importlib.metadata import version
        if not version("transformers").startswith("4.49."):
            raise RuntimeError("Legacy Florence generation must use the isolated Transformers 4.49 runtime")
        path, count = validate_florence_prefix_folder(payload["model_path"])
        protocol = {"schema": 1, "method": METHOD, "model_id": MODEL_ID,
                    "model_assets": checkpoint_asset_identity(path), "parameters": count,
                    "prompt": TASK_PROMPT, "decoding": "greedy_beam1_prefix_cleanup_first8",
                    "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
                    "policy": "isic_visible_attributes_v1",
                    "images": [{key: row[key] for key in ("image_id", "split", "image_sha256")}
                               for row in payload["records"]],
                    "settings": settings, "runtime": runtime_versions(), "code_sha256": payload["code_sha256"]}
        output = Path(payload["output"])
        identity = lock_protocol(output / "generation_protocol" / "protocol.json", protocol)
        if not (output / "status.json").exists() and (output / "captions.csv").exists():
            raise ValueError("Unregistered captions.csv; use a fresh output root")
        generate_records(payload, identity)
    elif phase == "encode":
        encode_texts(payload)
    else:
        raise ValueError("Phase must be generate or encode")
