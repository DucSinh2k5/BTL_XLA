"""Read-only checkpoint-to-web export. Never trains or invents completion markers."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import torch


VERSION = "sun397_snapshot_export_v1"
INFERENCE_FIXES = (
    ("vectors[~mask] = 0", "vectors = vectors.masked_fill((~mask).unsqueeze(1), 0.0)"),
    ("vectors[~masks] = 0", "vectors = vectors.masked_fill((~masks).unsqueeze(1), 0.0)"),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_id"] = int(row["class_id"])
        pure = PurePosixPath(row["image_id"])
        if pure.is_absolute() or ".." in pure.parts or "\\" in row["image_id"] or ":" in row["image_id"]:
            raise ValueError("Unsafe image ID: " + row["image_id"])
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs in " + str(path))
    return rows


def fixed_caption_source(source):
    for old, new in INFERENCE_FIXES:
        source = source.replace(old, new)
    compile(source, "caption_worker", "exec")
    return source


def source_variants(source, caption=False):
    variants = [source]
    if caption:
        variants.append(fixed_caption_source(source))
        previous = source
        for old, new in INFERENCE_FIXES:
            previous = previous.replace(new, old)
        variants.append(previous)
    return variants


def original_identity(config, classes, frames, assets, source_hashes, variant):
    ids = {key: [(r["image_id"], int(r["class_id"])) for r in rows] for key, rows in frames.items()}
    if variant == "tacs":
        model_sha, helper_sha = source_hashes["runtime"], source_hashes["training"]
    else:
        model_sha, helper_sha = source_hashes["tacs_visual_web"], source_hashes["train_sun397_tacs_web"]
        assets = dict(assets, mtacs_sources=source_hashes)
    payload = {"config": config, "classes": classes, "splits_and_pool": ids, "assets": assets,
               "model_source_sha256": model_sha, "training_source_sha256": helper_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resolve_image(record, root):
    root = Path(root).expanduser().resolve()
    pure = PurePosixPath(record["image_id"])
    for prefix in (root, root / "SUN397", root / "images"):
        path = prefix.joinpath(*pure.parts).resolve()
        if path.is_relative_to(root) and path.is_file():
            return dict(record, image_path=str(path))
    raise FileNotFoundError(f"Cannot resolve {record['image_id']} below DATASET_ROOT={root}")


def _runtime_sources(run, output, variant, fallback, recorded):
    required = ["tacs_visual_web.py", "train_sun397_tacs_web.py"]
    if variant == "mtacs":
        required += ["sun397_caption_web.py", "mtacs_multimodal_web.py", "train_sun397_mtacs_web.py"]
    critical = {"tacs_visual_web.py": recorded["runtime"] if variant == "tacs" else recorded["tacs_visual_web"]}
    if variant == "mtacs":
        critical.update({"sun397_caption_web.py": recorded["sun397_caption_web"],
                         "mtacs_multimodal_web.py": recorded["mtacs_multimodal_web"]})
    helper_hashes = {"train_sun397_tacs_web.py": recorded["training"] if variant == "tacs" else recorded["train_sun397_tacs_web"]}
    if variant == "mtacs":
        helper_hashes["train_sun397_mtacs_web.py"] = recorded["training"]
    directory = output / "export_runtime"
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in required:
        paths = [run / folder / name for folder in ("embedded_runtime_v2", "embedded_runtime")]
        original = next((p for p in paths if p.is_file()), None)
        content = original.read_text(encoding="utf-8") if original else fallback.get(name)
        if content is None:
            raise FileNotFoundError("Original runtime missing and no known fallback: " + name)
        provided_sha = hashlib.sha256(content.encode()).hexdigest()
        if original is not None and name in helper_hashes and hashlib.sha256(content.encode()).hexdigest() != helper_hashes[name]:
            raise ValueError("Original helper source changed after training: " + name)
        if name in critical:
            matched = next((value for value in source_variants(content, name == "sun397_caption_web.py")
                            if hashlib.sha256(value.encode()).hexdigest() == critical[name]), None)
            if matched is None:
                raise ValueError("Inference source differs from checkpoint protocol: " + name + ". Attach its original embedded_runtime folder.")
            content = matched
        input_sha = hashlib.sha256(content.encode()).hexdigest()
        if name == "sun397_caption_web.py":
            content = fixed_caption_source(content)
        compile(content, name, "exec")
        path = directory / name
        if path.is_file() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Export runtime already belongs to another source; select a new OUTPUT_DIR")
        if not path.is_file():
            path.write_text(content, encoding="utf-8", newline="\n")
        loaded = sys.modules.get(Path(name).stem)
        if loaded is not None and Path(loaded.__file__).resolve() != path.resolve():
            raise RuntimeError("Another TACS runtime is imported; restart the kernel before running this export-only notebook")
        provenance[name] = {"origin": str(original) if original else "embedded_known_fallback",
                            "provided_source_sha256": provided_sha,
                            "input_sha256": input_sha, "export_sha256": sha(path),
                            "caption_inference_tensor_fix": input_sha != sha(path)}
    sys.path.insert(0, str(directory.resolve()))
    modules = {Path(name).stem: importlib.import_module(Path(name).stem) for name in required}
    return modules, provenance


def prepare(source_run, dataset_root, dino_repo, output, variant, fallback_sources,
            device="cuda", feature_batch_size=8, workers=0, example_image=None,
            florence=None, clip=None, offline_assets_root=None,
            run_full_test=False, model_factory=None, caption_runner=None):
    if variant not in {"tacs", "mtacs"}:
        raise ValueError("variant must be tacs or mtacs")
    source = Path(source_run).expanduser().resolve()
    best_path = source if source.is_file() else source / "best.pt"
    run = best_path.parent
    output = Path(output).expanduser().resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("Use a separate writable OUTPUT_DIR outside the original run")
    required = ("protocol.json", "class_names.json", "candidate_pool.csv", "train_split.csv", "validation_split.csv")
    for path in [best_path] + [run / name for name in required]:
        if not path.is_file():
            raise FileNotFoundError(f"Required saved artifact missing: {path}. Attach the original run folder, not best.pt alone.")
    if feature_batch_size < 1 or workers < 0:
        raise ValueError("Invalid inference batch/workers")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; enable a GPU or select cpu")
    protocol = json_read(run / "protocol.json")
    best_sha = sha(best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if not all(key in checkpoint for key in ("model_state_dict", "config", "class_names", "identity", "epoch")):
        raise ValueError("Unsupported checkpoint schema; do not guess architecture or rename state_dict keys")
    config = checkpoint["config"]
    classes = checkpoint["class_names"]
    expected_method = "TACS_visual" if variant == "tacs" else "M_TACS_full_both"
    expected_schema = "sun397_tacs_visual_web_v1" if variant == "tacs" else "sun397_mtacs_full_both_web_v1"
    if (config.get("method") != expected_method or config.get("schema") != expected_schema
            or config.get("dataset") != "SUN397" or config.get("context_top_k") != 1
            or bool(config.get("caption_used")) != (variant == "mtacs")):
        raise ValueError("Checkpoint is not the requested SUN397 web architecture; original ablation checkpoints are not interchangeable")
    if model_factory is None and (config.get("backbone") != "dinov3_vitl16" or config.get("feature_dim") != 1024
                                  or config.get("image_size") != 384 or config.get("num_classes") != 397):
        raise ValueError("Production export requires L/16, 1024 dimensions, 384px and 397 classes")
    if config != protocol["config"] or classes != json_read(run / "class_names.json") or checkpoint["identity"] != protocol["identity"]:
        raise ValueError("best.pt, protocol and class order do not belong to the same experiment")
    if len(classes) != config["num_classes"] or len(set(classes)) != len(classes):
        raise ValueError("Invalid authoritative class order")
    frames = {key: csv_read(run / name) for key, name in
              (("train", "train_split.csv"), ("val", "validation_split.csv"), ("pool", "candidate_pool.csv"))}
    for rows in frames.values():
        for row in rows:
            if not 0 <= row["class_id"] < len(classes) or str(PurePosixPath(row["image_id"]).parent) != classes[row["class_id"]]:
                raise ValueError("Saved image/class mapping does not match checkpoint class order")
    pool_ids = {r["image_id"] for r in frames["pool"]}
    if not pool_ids or not pool_ids <= {r["image_id"] for r in frames["train"]} or pool_ids & {r["image_id"] for r in frames["val"]}:
        raise ValueError("Candidate pool must be the saved training-only pool")
    if original_identity(config, classes, frames, protocol["assets"], protocol["source_sha256"], variant) != checkpoint["identity"]:
        raise ValueError("Saved pool/order/splits/assets or source identity changed; do not regenerate a random pool")
    for index, row in enumerate(frames["pool"]):
        if "position" in row and int(row["position"]) != index:
            raise ValueError("Candidate positions are not in their trained order")
    output.mkdir(parents=True, exist_ok=True)
    modules, source_provenance = _runtime_sources(run, output, variant, fallback_sources, protocol["source_sha256"])
    visual, common = modules["tacs_visual_web"], modules["train_sun397_tacs_web"]
    dino_repo = Path(dino_repo).expanduser().resolve()
    if common.source_tree_identity(dino_repo) != protocol["assets"].get("dino_source_sha256"):
        raise ValueError("DINOv3 source differs from the recorded training source; attach the original code folder")
    candidates = [resolve_image(row, dataset_root) for row in frames["pool"]]
    candidate_hashes = {row["image_id"]: sha(row["image_path"]) for row in candidates}
    if sha(best_path) != best_sha:
        raise ValueError("best.pt changed while being read; stop concurrent training and export a stable snapshot")
    complete_path = run / "training_complete.json"
    completion = "not_recorded; export of checkpoint snapshot only"
    if complete_path.is_file():
        marker = json_read(complete_path)
        if (marker["identity"] != checkpoint["identity"] or marker["best_sha256"] != best_sha
                or marker["best_epoch"] != checkpoint["epoch"] or marker.get("epochs_completed") != config["epochs"]):
            raise ValueError("Existing completion marker does not match best.pt")
        completion = "original completion marker verified"
    context = {"variant": variant, "run": run, "best_path": best_path, "best_sha": best_sha,
               "checkpoint": checkpoint, "config": config, "classes": classes, "protocol": protocol,
               "frames": frames, "candidates": candidates, "candidate_hashes": candidate_hashes, "dataset_root": Path(dataset_root),
               "dino_repo": dino_repo, "output": output, "device": device, "modules": modules,
               "source_provenance": source_provenance, "feature_batch_size": feature_batch_size, "workers": workers,
               "example_image": Path(example_image).resolve() if example_image else None,
               "run_full_test": bool(run_full_test), "model_factory": model_factory, "caption_runner": caption_runner,
               "completion": completion, "offline_assets_root": Path(offline_assets_root) if offline_assets_root else None}
    if variant == "mtacs":
        service, workflow = modules["sun397_caption_web"], modules["train_sun397_mtacs_web"]
        shared = run / "shared"
        for name in ("captions.csv", "text_embeddings.pt", "caption_protocol.json", "caption_settings.json"):
            if not (shared / name).is_file():
                raise FileNotFoundError("M-TACS export requires the ORIGINAL shared cache: " + str(shared / name))
        for key, name in (("caption_csv_sha256", "captions.csv"), ("text_cache_sha256", "text_embeddings.pt"),
                          ("caption_settings_sha256", "caption_settings.json")):
            if sha(shared / name) != protocol["assets"].get(key):
                raise ValueError("Original caption cache binding changed: " + name)
        cache = torch.load(shared / "text_embeddings.pt", map_location="cpu", weights_only=True)
        records = frames["train"] + frames["val"]
        test_path = run / "test_split.csv"
        if not test_path.is_file():
            raise FileNotFoundError("M-TACS full caption coverage also requires test_split.csv")
        test = csv_read(test_path)
        records += test
        table = workflow.TextTable(cache, shared / "captions.csv", records, config)
        settings = json_read(shared / "caption_settings.json")
        if settings["classes"] != classes or settings["settings"]["policy"] != config["caption_policy"]:
            raise ValueError("Caption settings/class list differs from trained text table")
        if model_factory is None:
            if not florence or not clip:
                raise ValueError("Attach complete frozen Florence and CLIP folders for novel-image inference")
            assets = service.validate_assets(florence, clip)
        else:
            # Tests inject a tiny visual model and explicit caption service only.
            assets = {"florence": str(florence), "clip": str(clip),
                      "florence_sha256": service.tree_identity(florence), "clip_sha256": service.tree_identity(clip)}
        for key in ("florence_sha256", "clip_sha256"):
            if assets[key] != settings["asset_identity"][key] or assets[key] != protocol["assets"]["caption_assets"][key]:
                raise ValueError("Frozen text model weights differ from the caption cache: " + key)
        for row in candidates:
            if candidate_hashes[row["image_id"]] != table.rows[row["image_id"]]["image_sha256"]:
                raise ValueError("Candidate image bytes differ from its registered caption: " + row["image_id"])
        context.update(table=table, model_assets=assets, caption_settings=settings, shared=shared)
        if caption_runner is None:
            service.caption_environment(output / "caption_preflight", context["offline_assets_root"])
    history_path = run / "history.csv"
    last_saved = None
    if history_path.is_file():
        with history_path.open(encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        last_saved = history[-1].get("epoch") if history else None
    context["last_saved_epoch_from_history"] = last_saved
    print(f"Snapshot: {variant}, best epoch={checkpoint['epoch']}, history last epoch={last_saved}, pool={len(candidates)}")
    print("Completion:", completion, "| Full test:", "enabled" if run_full_test else "not run; demo snapshot export")
    print("No optimizer, pretrained DINO checkpoint, training loop or full caption regeneration is needed")
    return context


def _copy_file(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if sha(source) != sha(target):
            raise ValueError("Partial export file differs from source: " + str(target))
        return
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _copy_assets(source, target, service):
    source, target = Path(source), Path(target)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_file() and not {".cache", ".git", "__pycache__"}.intersection(relative.parts) and path.suffix != ".pyc":
            _copy_file(path, target / relative)
    if service.tree_identity(source) != service.tree_identity(target):
        raise ValueError("Frozen model copy is incomplete or has unexpected files")


def _progress(context, name, create):
    """Reuse completed expensive stages only after checking their recorded hashes."""
    stage = context["stage"]
    progress_path = stage / "export_progress.json"
    progress = json_read(progress_path) if progress_path.is_file() else {"files": {}}
    path = stage / name
    if name in progress["files"]:
        if not path.is_file() or sha(path) != progress["files"][name]:
            raise ValueError("Staged export artifact changed: " + name)
        print("Reusing verified export artifact:", name)
        return path
    if path.exists():
        raise FileExistsError("Unregistered partial artifact; preserve this output and use a new OUTPUT_DIR: " + name)
    create(path)
    progress["files"][name] = sha(path)
    json_write(progress_path, progress)
    return path


def _manifest(context):
    stage = context["stage"]
    files = {path.relative_to(stage).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "manifest.json"
             and "__pycache__" not in path.parts and not path.name.endswith(".tmp")}
    json_write(stage / "manifest.json", {"schema": context["runtime"].SCHEMA, "status": "ready", "files": files,
                                        "run_identity": context["checkpoint"]["identity"],
                                        "training_checkpoint_sha256": context["best_sha"],
                                        "model_sha256": sha(stage / "model.pt"),
                                        "operation": "export_only_snapshot", "export_request": context["request_identity"]})


def build(context):
    ctx = context
    variant, output, modules = ctx["variant"], ctx["output"], ctx["modules"]
    visual, common = modules["tacs_visual_web"], modules["train_sun397_tacs_web"]
    runtime = visual if variant == "tacs" else modules["mtacs_multimodal_web"]
    ctx["runtime"] = runtime
    request = {"version": VERSION, "operation": "export_only_snapshot", "variant": variant,
               "best_sha256": ctx["best_sha"], "run_identity": ctx["checkpoint"]["identity"],
               "source_provenance": ctx["source_provenance"], "full_test": ctx["run_full_test"],
               "candidate_image_sha256": ctx["candidate_hashes"],
               "example_image_sha256": sha(ctx["example_image"]) if ctx["example_image"] else None}
    ctx["request_identity"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    export_root = output / "exports"
    export_root.mkdir(exist_ok=True)
    name = "sun397_tacs_visual_vitl16_web_bundle" if variant == "tacs" else "sun397_mtacs_full_both_vitl16_web_bundle"
    target = export_root / name
    ctx.update(target=target, archive=target.with_suffix(".zip"))
    if target.is_dir():
        manifest = runtime.verify_bundle(target)
        if manifest.get("export_request") != ctx["request_identity"]:
            raise ValueError("Existing bundle belongs to a different export request")
        ctx.update(stage=target, already_verified=True)
        return ctx
    if ctx["archive"].exists():
        raise FileExistsError("Archive exists without its verified directory; preserve and inspect this output")
    stage = export_root / (".stage_" + ctx["request_identity"][:16])
    stage.mkdir(exist_ok=True)
    ctx["stage"] = stage
    if (stage / "export_request.json").is_file() and json_read(stage / "export_request.json") != request:
        raise ValueError("Staging request changed")
    json_write(stage / "export_request.json", request)
    payload_bytes = sum(value.numel() * value.element_size() for value in ctx["checkpoint"]["model_state_dict"].values())
    payload_bytes += sum(Path(row["image_path"]).stat().st_size for row in ctx["candidates"])
    if variant == "mtacs":
        payload_bytes += sum(path.stat().st_size for key in ("florence", "clip")
                             for path in Path(ctx["model_assets"][key]).rglob("*") if path.is_file())
        payload_bytes += sum((ctx["shared"] / name).stat().st_size for name in ("captions.csv", "text_embeddings.pt"))
    already_staged = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
    if shutil.disk_usage(export_root).free < max(0, 2 * payload_bytes - already_staged) + 1024**3:
        raise OSError("Not enough free disk for the remaining bundle plus ZIP. Original run is unchanged; preserve it and free other space.")
    portable = dict(ctx["config"], candidate_count=len(ctx["candidates"]), inference_precision="float32")
    json_write(stage / "config.json", portable)
    json_write(stage / "class_names.json", ctx["classes"])
    model_path = _progress(ctx, "model.pt", lambda path: common.atomic_torch_save(path, {
        "schema": runtime.SCHEMA, "config": portable, "class_names": ctx["classes"],
        "model_state_dict": ctx["checkpoint"]["model_state_dict"], "epoch": ctx["checkpoint"]["epoch"],
        "run_identity": ctx["checkpoint"]["identity"], "source_best_sha256": ctx["best_sha"]}))
    factory = ctx["model_factory"] or runtime.build_model
    model = factory(ctx["dino_repo"], ctx["config"])
    model.load_state_dict(ctx["checkpoint"]["model_state_dict"], strict=True)
    model.to(ctx["device"]).eval().requires_grad_(False)
    ctx["model"] = model
    pool = []
    for position, record in enumerate(common.tqdm(ctx["candidates"], desc="Copy exact saved candidate pool")):
        name = "candidates/" + record["image_id"]
        _copy_file(record["image_path"], visual.bundle_path(stage, name))
        digest = sha(stage / name)
        if digest != ctx["candidate_hashes"][record["image_id"]]:
            raise ValueError("Candidate image changed after snapshot validation: " + record["image_id"])
        pool.append({"position": position, "image_id": record["image_id"], "file": name,
                     "class_id": record["class_id"], "class_name": ctx["classes"][record["class_id"]],
                     "sha256": digest})
    json_write(stage / "candidate_pool.json", pool)
    common.csv_write(stage / "candidate_pool.csv", pool)
    cfg = {**ctx["config"], "feature_batch_size": ctx["feature_batch_size"], "eval_batch_size": ctx["feature_batch_size"], "workers": ctx["workers"]}
    ctx["eval_config"] = cfg

    def feature_bank(path):
        values = common.refresh_bank(model, ctx["candidates"], cfg, ctx["device"], amp=False)
        common.atomic_torch_save(path, {"features": values, "image_ids": [r["image_id"] for r in pool],
                                       "checkpoint_sha256": sha(model_path), "precision": "float32"})

    bank_path = _progress(ctx, "candidate_features.pt", feature_bank)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    if (bank["image_ids"] != [r["image_id"] for r in pool] or bank["checkpoint_sha256"] != sha(model_path)
            or tuple(bank["features"].shape) != (len(pool), ctx["config"]["feature_dim"]) or not torch.isfinite(bank["features"]).all()):
        raise ValueError("Staged candidate bank is not aligned to this snapshot")
    ctx["bank"] = bank["features"]
    ctx["pool"] = pool
    vendor = stage / "vendor/dinov3"
    if not vendor.exists():
        common.copy_vendor(ctx["dino_repo"], vendor)
    for key in ("tacs_visual_web",) if variant == "tacs" else ("tacs_visual_web", "mtacs_multimodal_web", "sun397_caption_web"):
        _copy_file(modules[key].__file__, stage / "runtime" / (key + ".py"))
    for name in ("protocol.json", "class_names.json", "train_split.csv", "validation_split.csv", "test_split.csv", "history.csv", "training_complete.json"):
        if (ctx["run"] / name).is_file():
            _copy_file(ctx["run"] / name, stage / "provenance/original_run" / name)
    for name in ("test_metrics.json", "test_predictions.csv"):
        if (ctx["run"] / name).is_file():
            _copy_file(ctx["run"] / name, stage / "provenance/unverified_original_metrics" / name)
    _copy_file(__file__, stage / "provenance/export_only_source.py")
    json_write(stage / "environment.json", common.environment())
    json_write(stage / "export_provenance.json", {**request, "best_epoch": ctx["checkpoint"]["epoch"],
        "training_completion": ctx["completion"], "last_saved_epoch_from_history": ctx["last_saved_epoch_from_history"],
        "original_run": str(ctx["run"]), "candidate_order": "saved CSV; not resampled",
        "training_called": False, "full_caption_regeneration": False,
        "original_metrics": "reference only; not asserted to be verified by this export"})
    if variant == "mtacs":
        service, workflow = modules["sun397_caption_web"], modules["train_sun397_mtacs_web"]
        texts, valid = ctx["table"].gather([r["image_id"] for r in pool], "cpu")
        _progress(ctx, "candidate_text.pt", lambda path: common.atomic_torch_save(path, {
            "embeddings": texts, "valid": valid, "image_ids": [r["image_id"] for r in pool],
            "checkpoint_sha256": sha(model_path), "csv_sha256": sha(ctx["shared"] / "captions.csv")}))
        for name in ("captions.csv", "text_embeddings.pt", "caption_protocol.json", "caption_settings.json"):
            _copy_file(ctx["shared"] / name, stage / name)
        for key in ("florence", "clip"):
            _copy_assets(ctx["model_assets"][key], stage / "models" / key, service)
            if service.tree_identity(stage / "models" / key) != ctx["model_assets"][key + "_sha256"]:
                raise ValueError("Frozen text model changed during export: " + key)
        (stage / "offline_wheels").mkdir(exist_ok=True)
        if ctx["offline_assets_root"]:
            folders = service.wheel_directories(ctx["offline_assets_root"], output / "caption_preflight/wheels")
            for folder in folders:
                for path in Path(folder).glob("*.whl"):
                    if path.name.startswith(("transformers-", "tokenizers-", "huggingface_hub-", "accelerate-", "einops-", "timm-", "safetensors-")):
                        _copy_file(path, stage / "offline_wheels" / path.name)
        cold_source = workflow.COLD_VERIFY_SOURCE
        requirements = "torch\nnumpy\nPillow\ntqdm\ntransformers==4.49.0\naccelerate\neinops\ntimm\n"
    else:
        cold_source = common.COLD_VERIFY_SOURCE
        requirements = "torch\nnumpy\nPillow\ntqdm\n"
    (stage / "runtime/verify_export.py").write_text(cold_source, encoding="utf-8")
    (stage / "requirements.txt").write_text(requirements, encoding="utf-8")
    json_write(stage / "attention_config.json", {"method": visual.ATTENTION_METHOD,
        "query_classifier": "paired classifier CLS attention to query patches",
        "candidate_selector": "selected candidate standalone selector CLS attention",
        "extra_attention_model": False, "class_conditional": False, "causal_explanation": False})
    pool_hashes = {r["sha256"] for r in pool}
    if ctx["example_image"]:
        example = {"image_path": str(ctx["example_image"]), "image_id": "external_export_example"}
    else:
        example = None
        for record in ctx["frames"]["val"]:
            resolved = resolve_image(record, ctx["dataset_root"])
            if sha(resolved["image_path"]) not in pool_hashes:
                example = resolved
                break
        if example is None:
            raise ValueError("No nonduplicate held-out example; supply EXAMPLE_IMAGE explicitly")
    example_path = stage / "examples" / ("query" + Path(example["image_path"]).suffix.lower())
    _copy_file(example["image_path"], example_path)
    ctx.update(example=example, example_path=example_path)
    metrics, predictions = None, None
    if ctx["run_full_test"]:
        test = [resolve_image(row, ctx["dataset_root"]) for row in csv_read(ctx["run"] / "test_split.csv")]
        if variant == "tacs":
            metrics, predictions = common.evaluate(model, common.make_loader(test, cfg), ctx["bank"], None, ctx["candidates"], cfg, ctx["device"], label="Optional final test / snapshot", amp=False)
        else:
            metrics, predictions = workflow.evaluate(model, common.make_loader(test, cfg), ctx["bank"], ctx["candidates"], ctx["table"], cfg, ctx["device"], label="Optional final test / snapshot", amp=False)
        common.csv_write(stage / "test_predictions.csv", predictions)
    json_write(stage / "metrics.json", {"test": metrics, "test_status": "evaluated_by_export" if metrics is not None else "not_run_export_only",
                                        "validation_from_checkpoint": ctx["checkpoint"].get("val_metrics"),
                                        "best_epoch": ctx["checkpoint"]["epoch"], "source_best_sha256": ctx["best_sha"]})
    (stage / "README.md").write_text(
        "# SUN397 checkpoint snapshot web bundle\n\n"
        "Exported from existing best.pt without training or an invented training_complete marker.\n"
        "See export_provenance.json and metrics.json: a smoke test is not a full test-set score.\n"
        "Full weights, exact saved candidate pool, aligned bank, runtime, DINOv3 source and real attention are included.\n"
        + ("Frozen Florence/CLIP assets and original caption CSV/vectors are included. Unknown images generate a fresh caption.\n" if variant == "mtacs" else "No captioner/CLIP/text branch is used.\n")
        + "Use an OS/CUDA-compatible Python environment; Kaggle Linux wheels are not Windows wheels.\n"
        "No original Kaggle model/dataset mount is required after extraction. Keep original dataset/model license terms.\n"
        "Attention is last-block mean-head CLS-to-patch, not causal attribution or a separately trained model.\n\n"
        + ("python runtime/mtacs_multimodal_web.py" if variant == "mtacs" else "python runtime/tacs_visual_web.py")
        + " --bundle . --image YOUR_IMAGE.jpg --output prediction.json\n", encoding="utf-8")
    _manifest(ctx)
    return ctx


def verify_and_publish(context):
    ctx = context
    stage, runtime = ctx["stage"], ctx["runtime"]
    if not ctx.get("already_verified"):
        production = ctx["model_factory"] is None and ctx["caption_runner"] is None
        if production:
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            env.pop("PYTHONPATH", None)
            process = subprocess.run([sys.executable, str(stage / "runtime/verify_export.py"), "--bundle", str(stage),
                                      "--image", str(ctx["example_path"]), "--image-id", ctx["example"]["image_id"], "--device", str(ctx["device"])],
                                     cwd=stage, env=env, text=True, capture_output=True)
            if process.returncode:
                raise RuntimeError("Offline reload failed; completed export artifacts are preserved for retry:\n" + process.stdout[-3000:] + process.stderr[-6000:])
            prediction = json_read(stage / "examples/prediction.json")
            data = torch.load(stage / "examples/reload.pt", map_location="cpu", weights_only=True) if ctx["variant"] == "mtacs" else {
                "logits": torch.load(stage / "examples/reload_logits.pt", map_location="cpu", weights_only=True)}
        else:
            if ctx["variant"] == "tacs":
                predictor = runtime.BundlePredictor(stage, str(ctx["device"]), ctx["model_factory"])
                prediction = predictor.predict(ctx["example_path"], image_id=ctx["example"]["image_id"], explain=True)
                data = {"logits": predictor.last_logits}
            else:
                predictor = runtime.BundlePredictor(stage, str(ctx["device"]), ctx["model_factory"], ctx["caption_runner"])
                prediction = predictor.predict(ctx["example_path"], image_id=ctx["example"]["image_id"], explain=True, use_caption_cache=False)
                data = {"logits": predictor.last_logits, "text": predictor.last_query_embedding, "valid": predictor.last_query_valid}
            del predictor
        visual = ctx["modules"]["tacs_visual_web"]
        with torch.no_grad():
            query = visual.preprocess(ctx["example_path"], ctx["config"])[None].to(ctx["device"])
            excluded = [i for i, row in enumerate(ctx["pool"]) if row["image_id"] == ctx["example"]["image_id"] or row["sha256"] == sha(ctx["example_path"])]
            if len(excluded) > 1:
                raise ValueError("Example duplicates multiple pool images; choose another held-out example")
            if ctx["variant"] == "tacs":
                scores, _ = ctx["model"].score(query, ctx["bank"].to(ctx["device"]), [excluded[0] if excluded else -1])
            else:
                text, valid = data["text"].to(ctx["device"]), data["valid"].to(ctx["device"])
                texts, masks = ctx["table"].gather([r["image_id"] for r in ctx["pool"]], ctx["device"])
                scores, _ = ctx["model"].score(query, ctx["bank"].to(ctx["device"]), [excluded[0] if excluded else -1], text, valid, texts, masks)
            position = int(scores.argmax(1))
            if prediction["evidence"]["candidate"]["position"] != position:
                raise ValueError("Reload context selection differs from best.pt")
            image = visual.preprocess(ctx["candidates"][position]["image_path"], ctx["config"])[None].to(ctx["device"])
            logits = ctx["model"](query, image) if ctx["variant"] == "tacs" else ctx["model"](query, image, text, valid, texts[position:position+1], masks[position:position+1])
            reference = logits.float().cpu()
        if not torch.allclose(reference, data["logits"], atol=1e-4, rtol=1e-4):
            raise ValueError("Reload logits differ from best.pt")
        if set(prediction["evidence"]["attention"]) != {"query_classifier", "candidate_selector"}:
            raise ValueError("Both real attention readouts are required")
        json_write(stage / "examples/prediction.json", prediction)
        json_write(stage / "smoke_test.json", {"passed": True, "cold_subprocess_reload": production,
            "fresh_florence_clip_integration": production and ctx["variant"] == "mtacs", "test_injected_services": not production,
            "source_is_bundle_only": production,
            "attention_roles": list(prediction["evidence"]["attention"]),
            "max_abs_logit_error": float((reference-data["logits"]).abs().max()), "training_called": False})
        _manifest(ctx)
        runtime.verify_bundle(stage)
        stage.rename(ctx["target"])
        ctx["stage"] = ctx["target"]
    archive = ctx["archive"]
    if archive.is_file():
        record = json_read(archive.parent / "download_manifest.json")
        if record["sha256"] != sha(archive) or record["export_request"] != ctx["request_identity"]:
            raise ValueError("Existing ZIP binding mismatch")
        print("Reusing verified export ZIP:", archive)
        return archive
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
        for path in sorted(ctx["target"].rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".tmp"):
                handle.write(path, ctx["target"].name + "/" + path.relative_to(ctx["target"]).as_posix())
    with zipfile.ZipFile(temporary) as handle:
        if handle.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
    temporary.replace(archive)
    json_write(archive.parent / "download_manifest.json", {"zip": archive.name, "bytes": archive.stat().st_size,
               "sha256": sha(archive), "export_request": ctx["request_identity"], "source_best_sha256": ctx["best_sha"]})
    print("Complete export-only ZIP:", archive)
    return archive
