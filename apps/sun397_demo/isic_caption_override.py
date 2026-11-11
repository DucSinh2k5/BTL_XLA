"""A separate, verified caption intervention over an unchanged ISIC checkpoint.

Importing this module does not import torch. The original training/export bundle
is never edited; caption text, masks and embeddings are replaced together.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

SCHEMA = "isic_caption_inference_override_v1"
METHOD = "florence_attributes_firstvalid_v3"
POLICY = "isic_visible_attributes_v2"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf-8")
    temporary.replace(path)


def load_core(path):
    name = "isic_caption_override_core_" + sha(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.METHOD != METHOD or module.POLICY != POLICY:
        raise ValueError("Wrong caption intervention runtime")
    return module


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing/duplicate caption header")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Empty or malformed captions")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in caption intervention")
    return rows


def boolean(value):
    if str(value).lower() not in {"true", "false"}:
        raise ValueError("Caption validity/clinical flags must be true or false")
    return str(value).lower() == "true"


def validate_rows(rows, old_rows, core):
    if [(r["image_id"], r["split"]) for r in rows] != [(r["image_id"], r["split"]) for r in old_rows]:
        raise ValueError("Caption override must preserve the complete original image/split order")
    fingerprints = {r["captioner_fingerprint"] for r in rows}
    if len(fingerprints) != 1 or not core.re.fullmatch(r"[a-f0-9]{64}", next(iter(fingerprints))):
        raise ValueError("CSV does not have one consistent generation fingerprint")
    for row, old in zip(rows, old_rows):
        if row["caption_input_policy"] != POLICY or row["vlm_model"] != core.MODEL_ID:
            raise ValueError("Unexpected caption policy/model")
        clean, valid, clinical, words = core.caption_policy(row["source_long_caption"], boolean(row["generation_token_limit_reached"]))
        if (row["final_caption"] != clean or boolean(row["caption_valid"]) != valid
                or int(row["word_count"]) != words or boolean(row["clinical_term_detected"]) != clinical
                or boolean(row["leakage_detected"]) != clinical):
            raise ValueError("Caption fields violate the unchanged attributes-v2 policy: " + row["image_id"])
        baseline, baseline_valid, _, _ = core.caption_policy(old["source_long_caption"], boolean(old["generation_token_limit_reached"]))
        if row["baseline_v2_caption"] != baseline or boolean(row["baseline_v2_valid"]) != baseline_valid:
            raise ValueError("Original per-image caption baseline does not match the imported CSV")
        if baseline_valid and (not valid or clean != baseline):
            raise ValueError("A valid baseline-v2 caption was changed")
    # FirstValid may start from the original v1 CSV or an already extracted v2 CSV.
    # Source validity must match ONE of those complete sources, never a per-row mixture.
    source_valid = [boolean(r["source_caption_valid"]) for r in rows]
    if source_valid not in ([boolean(r["caption_valid"]) for r in old_rows],
                            [boolean(r["baseline_v2_valid"]) for r in rows]):
        raise ValueError("Source validity does not describe the original v1 or v2 source")


def load_clip_text(bundle):
    """Read-only CLIP text weights, sharing the OS file cache with web workers."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection
    from worker import readonly_checkpoint
    root = Path(bundle) / "caption_models/clip"
    config = read_json(root / "config.json")
    text_config = CLIPTextConfig.from_dict(config["text_config"])
    if config.get("projection_dim") != 512 or text_config.projection_dim != 512:
        raise ValueError("Use the unchanged CLIP ViT-B/32 embedding space")
    with init_empty_weights(include_buffers=False):
        model = CLIPTextModelWithProjection(text_config)
    state = readonly_checkpoint(root / "pytorch_model.bin")
    expected = set(model.state_dict())
    if expected - set(state):
        raise ValueError("Incomplete CLIP text weights")
    torch.nn.Module.load_state_dict(model, {k: state[k] for k in expected}, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("CLIP text parameters were not restored")
    model.eval().requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(root, local_files_only=True)


def encode_rows(rows, model, tokenizer, batch_size=64):
    import torch
    valid = torch.tensor([boolean(row["caption_valid"]) for row in rows], dtype=torch.bool)
    vectors = torch.zeros((len(rows), 512), dtype=torch.float32)
    positions = valid.nonzero().flatten().tolist()
    for start in range(0, len(positions), batch_size):
        group = positions[start:start + batch_size]
        tokens = tokenizer([rows[i]["final_caption"] for i in group], padding=True, truncation=True,
                           max_length=model.config.max_position_embeddings, return_tensors="pt")
        with torch.inference_mode():
            values = torch.nn.functional.normalize(model(**tokens).text_embeds.float(), dim=1)
        vectors[group] = values.cpu()
        print(f"CLIP text: {min(start + batch_size, len(positions))}/{len(positions)}", flush=True)
    return {"image_ids": [r["image_id"] for r in rows], "embeddings": vectors, "valid": valid}


def prepare_override(bundle, source_csv, output, core_source, encoder=None):
    import collections
    import torch
    bundle, source_csv, output = Path(bundle).resolve(), Path(source_csv).resolve(), Path(output).resolve()
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Keep caption interventions outside the verified training bundle")
    original_manifest = read_json(bundle / "manifest.json")
    if original_manifest.get("mode") != "mtacs" or original_manifest.get("status") != "ready":
        raise ValueError("A ready, original ISIC M-TACS bundle is required")
    if (bundle / "captions.csv").resolve() == source_csv:
        raise ValueError("Choose the new FirstValid CSV")
    # Verify the original caption data and the CLIP assets used to encode new text.
    for name in ["captions.csv", "caption_settings.json", "caption_lookup.json", "caption_models/clip/config.json",
                 "caption_models/clip/pytorch_model.bin", "caption_models/clip/tokenizer.json",
                 "caption_models/clip/tokenizer_config.json"]:
        if sha(bundle / name) != original_manifest["files"][name]:
            raise ValueError("Original bundle integrity failure: " + name)
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "caption_core.py"
    runtime.write_text(core_source, encoding="utf-8")
    core = load_core(runtime)
    rows, old_rows = read_rows(source_csv), read_rows(bundle / "captions.csv")
    validate_rows(rows, old_rows, core)
    settings = {**read_json(bundle / "caption_settings.json")["settings"], "batch_size": 1,
                "retry_greedy_tokens": 128, "retry_beam_tokens": 192, "retry_beams": 4, "retry_returns": 3}
    core.validate_retry_settings(settings)
    bindings = {"base_manifest_sha256": sha(bundle / "manifest.json"), "source_csv_sha256": sha(source_csv),
                "caption_runtime_sha256": sha(runtime)}
    prior_manifest = output / "manifest.json"
    if prior_manifest.is_file():
        previous = read_json(prior_manifest)
        if any(previous.get(k) != value for k, value in bindings.items()):
            raise ValueError("Another caption intervention exists here; use a new output directory")
        verify_override(output, bundle)
        print("Reusing complete caption intervention:", output, flush=True)
        return previous
    shutil.copyfile(source_csv, output / "captions.csv")
    if encoder is None:
        model, tokenizer = load_clip_text(bundle)
        text = encode_rows(rows, model, tokenizer)
    else:
        text = encoder(rows)
    text.update(caption_csv_sha256=sha(output / "captions.csv"), identity=core.object_hash(bindings))
    torch.save(text, output / "text_embeddings.pt")
    lookup = {r["image_id"]: dict(r, caption_valid=boolean(r["caption_valid"])) for r in rows}
    write_json(output / "caption_lookup.json", lookup)
    write_json(output / "settings.json", {"method": METHOD, "policy": POLICY, "settings": settings})
    report = {"images": len(rows), "old_masked": sum(not boolean(r["caption_valid"]) for r in old_rows),
              "baseline_v2_masked": sum(not boolean(r["baseline_v2_valid"]) for r in rows),
              "new_masked": sum(not boolean(r["caption_valid"]) for r in rows),
              "selection_origins": dict(collections.Counter(r["selection_origin"] for r in rows)),
              "valid_by_split": {role: sum(r["split"] == role and boolean(r["caption_valid"]) for r in rows)
                                 for role in ("train", "val", "test")}}
    write_json(output / "report.json", report)
    files = ["captions.csv", "text_embeddings.pt", "caption_lookup.json", "settings.json", "caption_core.py", "report.json"]
    manifest = {"schema": SCHEMA, "method": METHOD, "policy": POLICY, **bindings,
                "caption_csv_sha256": sha(output / "captions.csv"), "generation_fingerprint": rows[0]["captioner_fingerprint"],
                "experiment": "caption_inference_intervention", "segmentation_checkpoint_retrained": False,
                "generation_audit": "CSV imported; full Kaggle retry audit not supplied",
                "initial_fresh_generation_max_new_tokens": settings["max_new_tokens"],
                "files": {name: sha(output / name) for name in files}, "report": report}
    write_json(output / "manifest.json", manifest)
    verify_override(output, bundle)
    return manifest


def verify_override(output, bundle):
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    manifest = read_json(output / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("method") != METHOD
            or manifest.get("policy") != POLICY or manifest.get("segmentation_checkpoint_retrained") is not False
            or manifest.get("base_manifest_sha256") != sha(bundle / "manifest.json")):
        raise ValueError("Caption intervention does not belong to this original checkpoint")
    for name, digest in manifest["files"].items():
        path = (output / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(output) or sha(path) != digest:
            raise ValueError("Caption intervention integrity failure: " + name)
    return manifest


class FirstValidCaptionEngine:
    def __init__(self, original_engine, core, settings):
        self.local = original_engine.local
        if self.local is None:
            raise RuntimeError("The caption intervention requires the existing compatible Transformers 4.49 worker")
        self.core, self.settings = core, settings
        self.captioner = core.IsicFlorenceFirstValidCaptioner.__new__(core.IsicFlorenceFirstValidCaptioner)
        self.captioner.__dict__.update(self.local.captioner.__dict__)
        self.captioner.settings = {**settings, "device": str(self.local.device)}

    def caption(self, image):
        import torch
        import torch.nn.functional as F
        from PIL import Image, ImageOps
        core, settings = self.core, self.settings
        if isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        else:
            with Image.open(image) as opened:
                rgb = ImageOps.exif_transpose(opened).convert("RGB")
        selected, valid, clinical, count, raw = "", False, False, 0, ""
        schedule = [("initial_v2", settings["max_new_tokens"], 1, 1),
                    ("greedy_retry", settings["retry_greedy_tokens"], 1, 1),
                    ("beam_retry", settings["retry_beam_tokens"], settings["retry_beams"], settings["retry_returns"])]
        origin = "still_masked_after_retries"
        try:
            for phase, tokens, beams, returns in schedule:
                candidates = self.captioner.generate_candidates([rgb], max_new_tokens=tokens,
                    num_beams=beams, num_return_sequences=returns)[0]
                for candidate in candidates:
                    final, accepted, banned, words = core.caption_policy(candidate["response"], candidate["token_limit"])
                    if phase == "initial_v2":
                        selected, valid, clinical, count, raw = final, accepted, banned, words, candidate["response"]
                    if accepted:
                        selected, valid, clinical, count, raw = final, True, banned, words, candidate["response"]
                        origin = phase
                        break
                if valid:
                    break
        finally:
            rgb.close()
        vector = torch.zeros(512, dtype=torch.float32)
        if valid:
            tokens = self.local.tokenizer([selected], padding=True, truncation=True,
                max_length=self.local.clip.config.max_position_embeddings, return_tensors="pt")
            with torch.inference_mode():
                encoded = self.local.clip(**{k: v.to(self.local.device) for k, v in tokens.items()}).text_embeds
                values = F.normalize(encoded.float(), dim=1)[0].cpu()
            vector = values.clone()
        row = {"final_caption": selected, "caption_valid": valid, "clinical_term_detected": clinical,
               "word_count": count, "source_long_caption": raw, "caption_policy": POLICY,
               "caption_method": METHOD, "selection_origin": origin}
        return row, vector, torch.tensor(valid, dtype=torch.bool)


def apply_override(predictor, original_engine, output):
    import torch
    manifest = verify_override(output, predictor.root)
    output = Path(output)
    core = load_core(output / "caption_core.py")
    rows = read_rows(output / "captions.csv")
    validate_rows(rows, read_rows(predictor.root / "captions.csv"), core)
    text = torch.load(output / "text_embeddings.pt", map_location="cpu", weights_only=True)
    expected_ids = [r["image_id"] for r in rows]
    valid = torch.tensor([boolean(r["caption_valid"]) for r in rows], dtype=torch.bool)
    embeddings = text["embeddings"]
    if (text["image_ids"] != expected_ids or text["caption_csv_sha256"] != manifest["caption_csv_sha256"]
            or not torch.equal(text["valid"], valid) or embeddings.shape != (len(rows), 512)
            or not torch.isfinite(embeddings).all() or torch.count_nonzero(embeddings[~valid]).item()
            or not torch.allclose(embeddings[valid].norm(dim=1), torch.ones(int(valid.sum())), atol=2e-5)):
        raise ValueError("Caption text/embedding/mask alignment failed")
    positions = {image_id: i for i, image_id in enumerate(expected_ids)}
    # Identical pixel content can occur under multiple registered image IDs;
    # the original pixel index contains one representative, not all CSV rows.
    if not set(predictor.manifest["caption_pixel_index"].values()).issubset(positions):
        raise ValueError("New captions do not match the verified dataset pixel identities")
    indices = [positions[r["image_id"]] for r in predictor.pool]
    predictor.text, predictor.text_position = text, positions
    predictor.c_text = embeddings[indices].to(predictor.device)
    predictor.c_valid = valid[indices].to(predictor.device)
    predictor.captions = read_json(output / "caption_lookup.json")
    predictor.caption_engine = FirstValidCaptionEngine(original_engine, core, read_json(output / "settings.json")["settings"])
    return {"method": METHOD, "policy": POLICY, "experiment": "caption_inference_intervention",
            "segmentation_checkpoint_retrained": False, "csv_sha256": manifest["caption_csv_sha256"],
            "valid_captions": int(valid.sum()), "masked_captions": int((~valid).sum())}
