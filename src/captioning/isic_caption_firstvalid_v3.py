"""Frozen Florence retries for masked attributes-v2 captions.

Embedded after the unchanged v2 extraction/policy helpers by the builder.
No ground-truth masks, diagnoses or class labels are accepted as generation inputs.
"""
from __future__ import annotations


def validate_retry_settings(settings):
    for key in ("batch_size", "max_new_tokens", "retry_greedy_tokens", "retry_beam_tokens", "retry_beams", "retry_returns"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if settings["retry_beams"] < 2 or settings["retry_returns"] > settings["retry_beams"]:
        raise ValueError("Require 2 <= retry_beams and retry_returns <= retry_beams")
    if (settings.get("precision") != "float32" or type(settings.get("seed")) is not int
            or not 0 <= settings["seed"] < 2**32):
        raise ValueError("Use the frozen FP32 Florence protocol with an integer seed")


class IsicFlorenceFirstValidCaptioner(IsicFlorencePrefixCaptioner):
    def generate_candidates(self, images, *, max_new_tokens, num_beams, num_return_sequences):
        import torch
        if num_return_sequences > num_beams or (num_beams == 1 and num_return_sequences != 1):
            raise ValueError("Invalid deterministic beam/return configuration")
        inputs = self.processor(text=[TASK_PROMPT] * len(images), images=images, return_tensors="pt", padding=True)
        moved = {k:v.to(device=self.device, dtype=self.dtype if k == "pixel_values" else v.dtype)
                 for k,v in inputs.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            output = self.model.generate(**moved, max_new_tokens=max_new_tokens, num_beams=num_beams,
                num_return_sequences=num_return_sequences, do_sample=False, use_cache=True, early_stopping=False)
        decoded = self.processor.batch_decode(output, skip_special_tokens=False)
        if len(decoded) != len(images) * num_return_sequences:
            raise ValueError("Incomplete grouped Florence beam outputs")
        sequences = output.cpu().tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list,tuple)) else [eos])
        pad = self.model.generation_config.pad_token_id
        groups = []
        for image_index, image in enumerate(images):
            candidates = []
            for rank in range(num_return_sequences):
                index = image_index * num_return_sequences + rank
                parsed = self.processor.post_process_generation(decoded[index], task=TASK_PROMPT,
                    image_size=(image.width,image.height))
                if not isinstance(parsed,dict) or not isinstance(parsed.get(TASK_PROMPT),str):
                    raise ValueError("Florence returned another task/field")
                # Exclude the initial decoder token (which may itself be EOS).
                # Strip padding from shorter rows in a multi-image batch. EOS
                # forced at the length cap does not certify a complete response.
                new_tokens = list(sequences[index][1:])
                if pad not in eos:
                    while new_tokens and new_tokens[-1] == pad:
                        new_tokens.pop()
                candidates.append({"response":parsed[TASK_PROMPT].strip(),
                    "token_limit":len(new_tokens) >= max_new_tokens, "new_tokens":len(new_tokens), "rank":rank})
            groups.append(candidates)
        return groups


def firstvalid_audit_path(output, image_id):
    return Path(output) / "retry_audit" / (object_hash(image_id) + ".json")


def firstvalid_write(output, record, identity, trace):
    atomic_json(firstvalid_audit_path(output,record["image_id"]), trace)
    if trace["complete"]:
        chosen = trace["selected_row"]
        row = dict(chosen, baseline_v2_caption=trace["baseline_row"]["final_caption"],
            baseline_v2_valid=bool(trace["baseline_row"]["caption_valid"]),
            source_caption_valid=trace["source_caption_valid"],
            selection_origin=trace["selection_origin"],
            retry_count=sum(len(r["candidates"]) for r in trace["rounds"]),
            retry_rounds=len(trace["rounds"]))
        atomic_json(cache_path(Path(output)/"generated",record["image_id"]), {"identity":identity,"row":row})


def generate_firstvalid_records(payload, identity, factory=IsicFlorenceFirstValidCaptioner):
    import gc
    import torch
    from PIL import Image, ImageOps
    from tqdm.auto import tqdm
    settings, records, output = payload["settings"], payload["records"], Path(payload["output"])
    validate_retry_settings(settings)
    registration, source = prior_caption_registration(payload)
    if registration is None or len(source) != len(records):
        raise ValueError("Attach complete registered v1/v2 long captions for every image. This experiment retries only v2 rejects.")
    traces = {}
    for record in records:
        path = firstvalid_audit_path(output, record["image_id"])
        if path.is_file():
            trace = json.loads(path.read_text("utf-8"))
            if trace.get("identity") != identity or trace.get("image_id") != record["image_id"]:
                raise ValueError("Retry audit belongs to another image/protocol")
            if trace["complete"]:
                firstvalid_write(output,record,identity,trace)  # Repair an interrupted atomic commit.
        else:
            if cache_path(output/"generated",record["image_id"]).is_file():
                raise ValueError("A generated row exists without its retry provenance; use a fresh output")
            original = source[record["image_id"]]
            baseline = make_generation_row(record, original["source_long_caption"],
                str(original["generation_token_limit_reached"]).lower() == "true", identity)
            trace = {"identity":identity,"image_id":record["image_id"],"baseline_row":baseline,
                "source_caption_valid":str(original["caption_valid"]).lower() == "true",
                "rounds":[],"selected_row":baseline,"selection_origin":"baseline_v2",
                "complete":bool(baseline["caption_valid"])}
            firstvalid_write(output,record,identity,trace)
        traces[record["image_id"]] = trace
    baseline_masked = sum(not t["baseline_row"]["caption_valid"] for t in traces.values())
    unresolved = [r for r in records if not traces[r["image_id"]]["complete"]]
    if len(unresolved) > baseline_masked:
        raise ValueError("Unexpected retry population")
    print(f"Images={len(records)}; baseline-v2 masked={baseline_masked}; remaining retry images={len(unresolved)}",flush=True)
    captioner = None
    schedule = [("greedy_retry",settings["retry_greedy_tokens"],1,1),
                ("beam_retry",settings["retry_beam_tokens"],settings["retry_beams"],settings["retry_returns"])]
    try:
        for phase, tokens, beams, returns in schedule:
            todo = [r for r in records if not traces[r["image_id"]]["complete"]
                    and not any(x["phase"] == phase for x in traces[r["image_id"]]["rounds"])]
            if not todo:
                continue
            if captioner is None:
                captioner = factory(payload["model_path"],settings)
            offset, batch_size = 0, settings["batch_size"]
            progress = tqdm(total=len(todo),desc="ISIC " + phase,unit="image")
            try:
                while offset < len(todo):
                    batch = todo[offset:offset + batch_size]
                    images = []
                    try:
                        for record in batch:
                            if file_sha256(record["image_path"]) != record["image_sha256"]:
                                raise ValueError("Image changed before retry: " + record["image_id"])
                            with Image.open(record["image_path"]) as im:
                                images.append(ImageOps.exif_transpose(im).convert("RGB"))
                        generated = captioner.generate_candidates(images,max_new_tokens=tokens,
                            num_beams=beams,num_return_sequences=returns)
                        if len(generated) != len(batch) or any(len(g) != returns for g in generated):
                            raise ValueError("Incomplete retry response group")
                    except torch.cuda.OutOfMemoryError:
                        if batch_size == 1:
                            raise RuntimeError("Florence retry OOM at batch 1; committed audit is resumable. Free GPU memory and rerun") from None
                        batch_size = max(1,batch_size//2)
                        gc.collect(); torch.cuda.empty_cache()
                        continue
                    finally:
                        for im in images:
                            im.close()
                    for record, candidates in zip(batch,generated):
                        trace = traces[record["image_id"]]
                        attempts = []
                        chosen = None
                        for candidate in candidates:
                            row = make_generation_row(record,candidate["response"],candidate["token_limit"],identity)
                            attempts.append({"rank":candidate["rank"],"new_tokens":candidate["new_tokens"],"row":row})
                            if chosen is None and row["caption_valid"]:
                                chosen = row
                        trace["rounds"].append({"phase":phase,"max_new_tokens":tokens,"num_beams":beams,
                            "num_return_sequences":returns,"candidates":attempts})
                        if chosen is not None:
                            trace.update(selected_row=chosen,selection_origin=phase,complete=True)
                        elif phase == "beam_retry":
                            trace.update(complete=True,selection_origin="still_masked_after_retries")
                        firstvalid_write(output,record,identity,trace)
                    offset += len(batch); progress.update(len(batch))
            finally:
                progress.close()
    finally:
        del captioner
        gc.collect()
        torch.cuda.empty_cache()
    if not all(t["complete"] for t in traces.values()):
        raise RuntimeError("Retries are incomplete; rerun to resume")
    rows = [cached_row(output/"generated",r,identity) for r in records]
    if any(r is None for r in rows):
        raise ValueError("Missing committed final caption")
    valid = sum(bool(r["caption_valid"]) for r in rows)
    recovered = sum(not t["baseline_row"]["caption_valid"] and t["selected_row"]["caption_valid"] for t in traces.values())
    if len(records)-valid != baseline_masked-recovered:
        raise ValueError("A valid baseline caption was lost or mask counts are inconsistent")
    columns = COLUMNS + ["baseline_v2_caption","baseline_v2_valid","source_caption_valid","selection_origin","retry_rounds"]
    atomic_csv(output/"captions.csv",rows,columns)
    status = {"complete":True,"rows":len(rows),"expected":len(records),"identity":identity,
        "csv_sha256":file_sha256(output/"captions.csv"),"csv_path":str((output/"captions.csv").resolve()),
        "baseline_v2_masked":baseline_masked,"recovered_from_v2":recovered,"final_masked":len(records)-valid,
        "valid_by_split":{role:sum(bool(r["caption_valid"]) for r in rows if r["split"]==role) for role in ("train","val","test")}}
    atomic_json(output/"status.json",status)
    print(json.dumps(status,indent=2),flush=True)
    return status


def firstvalid_worker_main(request_path, phase):
    import torch
    from importlib.metadata import version
    payload = json.loads(Path(request_path).read_text("utf-8"))
    validate_caption_records(payload["records"])
    validate_retry_settings(payload["settings"])
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU")
    if not version("transformers").startswith("4.49."):
        raise RuntimeError("Use the isolated offline Transformers 4.49 worker")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(payload["settings"]["seed"])
    torch.cuda.manual_seed_all(payload["settings"]["seed"])
    if phase == "generate":
        path,count = validate_florence_prefix_folder(payload["model_path"])
        registration,_ = prior_caption_registration(payload)
        if registration is None:
            raise ValueError("A registered old caption source is required")
        protocol = {"schema":1,"method":METHOD,"model_id":MODEL_ID,"policy":POLICY,"prompt":TASK_PROMPT,
            "model_assets":checkpoint_asset_identity(path),"parameters":count,"dtype":"float32",
            "decoding":"keep_valid_v2_then_greedy_retry_then_first_valid_beam",
            "long_response_source":registration,"settings":payload["settings"],"runtime":runtime_versions(),
            "code_sha256":payload["code_sha256"],
            "images":[{k:r[k] for k in ("image_id","split","image_sha256")} for r in payload["records"]]}
        identity = lock_protocol(Path(payload["output"])/"generation_protocol/protocol.json",protocol)
        return generate_firstvalid_records(payload,identity)
    if phase == "encode":
        return encode_isic_texts(payload)
    raise ValueError("Unknown caption-only worker phase")


def resolve_prior_source(reference, roots):
    """Find one original Florence run (duplicate copies are grouped by CSV hash)."""
    explicit = str(reference).strip()
    paths = [Path(explicit).expanduser()] if explicit else []
    searches = paths or [Path(r) for r in roots]
    if explicit and not paths[0].exists():
        raise FileNotFoundError("PRIOR_CAPTIONS_SOURCE does not exist: " + explicit)
    discovered = []
    for root in searches:
        if root.is_file():root=root.parent
        if not root.is_dir():continue
        for current,dirs,files in os.walk(root):
            folder=Path(current)
            dirs[:] = [n for n in dirs if n not in {"images","masks","candidate_images","ground_truth_masks","caption_models","generated","caption_runtime","vendor","legacy_packages",".cache","__pycache__"}]
            if len(folder.relative_to(root).parts)>=10:dirs[:]=[]
            if "captions.csv" not in files:continue
            for proto,status in [(folder/"generation_protocol/protocol.json",folder/"status.json"),
                    (folder/"caption_provenance/generation_protocol/protocol.json",folder/"caption_provenance/status.json")]:
                if not proto.is_file() or not status.is_file():continue
                p=json.loads(proto.read_text("utf-8"));s=json.loads(status.read_text("utf-8"))
                if p.get("method") not in {"florence_prefix_control","florence_extractive_attributes_v2"}:continue
                if not s.get("complete") or file_sha256(folder/"captions.csv")!=s.get("csv_sha256") or object_hash(p)!=s.get("identity"):
                    raise ValueError("Source caption CSV/protocol/status changed: " + str(folder))
                discovered.append((folder.resolve(),p,s))
    by_hash={}
    for item in discovered:by_hash.setdefault(item[2]["csv_sha256"],[]).append(item)
    if len(by_hash)!=1:
        raise FileNotFoundError("Attach the registered v1 or v2 ISIC Florence output/bundle and set PRIOR_CAPTIONS_SOURCE exactly. "
            f"Found {len(by_hash)} different caption sources: {[str(x[0]) for x in discovered]}")
    folder,protocol,status=min(next(iter(by_hash.values())),key=lambda x:(len(x[0].parts),str(x[0])))
    return folder,protocol,status


def recover_image_records(protocol, dataset_root, roots):
    """Read RGB image files only. Segmentation masks never enter this pipeline."""
    from tqdm.auto import tqdm
    records=protocol.get("images",[])
    if not records or len({r["image_id"] for r in records})!=len(records):
        raise ValueError("Source image binding is empty or duplicated")
    wanted={r["image_id"] for r in records}
    for r in records:
        if (set(r)!={"image_id","split","image_sha256"} or not re.fullmatch(r"ISIC_\d+",r["image_id"])
                or r["split"] not in {"train","val","test"} or not re.fullmatch(r"[a-f0-9]{64}",r["image_sha256"])):
            raise ValueError("Malformed original pixel-only image registration")
    searches=[Path(dataset_root).expanduser()] if str(dataset_root).strip() else [Path(r) for r in roots]
    if str(dataset_root).strip() and not searches[0].is_dir():raise FileNotFoundError("ISIC_ROOT not found")
    index={}
    for root in searches:
        if not root.is_dir():continue
        for current,dirs,files in os.walk(root):
            dirs[:]=[n for n in dirs if n not in {".git",".cache","__pycache__","models","caption_models","candidate_images","ground_truth_masks","masks","generated","vendor","legacy_packages"}
                     and "groundtruth" not in n.lower() and "ground_truth" not in n.lower()]
            for name in files:
                match=re.fullmatch(r"(ISIC_\d+)\.(?:jpg|jpeg|png|bmp|webp)",name,re.I)
                if match and match[1].upper() in wanted:
                    index.setdefault(match[1].upper(),[]).append(Path(current)/name)
    result=[]
    digests={}
    for row in tqdm(records,desc="Verify original RGB image hashes"):
        path=None
        for candidate in sorted(set(p.resolve() for p in index.get(row["image_id"],[]))):
            if candidate not in digests:
                digests[candidate]=file_sha256(candidate)
            digest=digests[candidate]
            if digest==row["image_sha256"]:path=candidate;break
        if path is None:raise FileNotFoundError("No original RGB image with matching hash for " + row["image_id"] + "; attach ISIC and set ISIC_ROOT")
        result.append(dict(row,image_path=str(path)))
    validate_caption_records(result)
    return result


def write_firstvalid_reports(output):
    import csv
    import collections
    output=Path(output)
    with (output/"captions.csv").open(encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle))
    true=lambda x:str(x).lower()=="true"
    summaries=[]
    for role in ("all","train","val","test"):
        subset=rows if role=="all" else [r for r in rows if r["split"]==role]
        if not subset:continue
        before=sum(not true(r["baseline_v2_valid"]) for r in subset)
        after=sum(not true(r["caption_valid"]) for r in subset)
        recovered=sum(not true(r["baseline_v2_valid"]) and true(r["caption_valid"]) for r in subset)
        if after!=before-recovered:raise ValueError("Mask comparison is inconsistent")
        if any(true(r["baseline_v2_valid"]) and (not true(r["caption_valid"]) or r["baseline_v2_caption"]!=r["final_caption"]) for r in subset):
            raise ValueError("A valid v2 caption was changed")
        summaries.append({"split":role,"images":len(subset),"source_csv_masked":sum(not true(r["source_caption_valid"]) for r in subset),
            "baseline_v2_masked":before,"firstvalid_masked":after,"recovered_from_v2":recovered,
            "baseline_mask_percent":100*before/len(subset),"firstvalid_mask_percent":100*after/len(subset)})
    atomic_csv(output/"mask_comparison.csv",summaries,list(summaries[0]))
    counts=collections.Counter(r["final_caption"] for r in rows if true(r["caption_valid"]))
    review=[]
    for row in rows:
        trace=json.loads(firstvalid_audit_path(output,row["image_id"]).read_text("utf-8"))
        review.append({"image_id":row["image_id"],"split":row["split"],"baseline_caption":row["baseline_v2_caption"],
            "baseline_valid":row["baseline_v2_valid"],"final_caption":row["final_caption"],"final_valid":row["caption_valid"],
            "selection_origin":row["selection_origin"],"baseline_long_caption":trace["baseline_row"]["source_long_caption"],
            "selected_long_caption":row["source_long_caption"],"retry_count":row["retry_count"]})
    atomic_csv(output/"caption_retries_review.csv",review,list(review[0]))
    report={"policy":POLICY,"method":METHOD,"mask_comparison":summaries,"unique_valid_captions":len(counts),
        "top_valid_captions":counts.most_common(10),"selection_origins":dict(collections.Counter(r["selection_origin"] for r in rows)),
        "filters_relaxed":False,"no_gt_masks_or_diagnoses_in_generation_inputs":True,
        "segmentation_training_performed":False,"not_caption_factual_accuracy_or_segmentation_metrics":True,
        "manual_image_review_required":True}
    atomic_json(output/"caption_quality_report.json",report)
    return report
