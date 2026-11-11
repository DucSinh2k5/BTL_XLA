"""Full-both snapshot recovery; fixtures are explicitly not deployable models."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in ("src/deployment", "src/models", "src/training", "tools/notebook_builders", "tests"):
    sys.path.insert(0, str(ROOT / folder))
import isic_export_only as common
import isic_mtacs_export_only as export
import build_isic_mtacs_export_only_notebook as builder
import train_isic_segmentation_web as work
from test_isic_export_only import make_fixture

torch.set_num_threads(1)


@pytest.fixture
def runtime_isolation(monkeypatch):
    names = ("isic_segmentation_web", "train_isic_segmentation_web", "tacs_visual_web", "isic_caption_core", "isic_caption_web")
    paths = list(sys.path)
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in names:
        sys.modules.pop(name, None)
    sys.path[:] = paths


def make_multimodal(root):
    run, dataset, repo, weights, factory = make_fixture(root)
    config = common.read_json(run / "protocol.json")["config"]
    config.update(mode="mtacs", text_dim=512)
    fallback = builder.embedded_sources("mtacs")
    for name, source in fallback.items():
        (run / "embedded_runtime" / name).write_text(source, encoding="utf8")
    core = types.ModuleType("fixture_caption_core")
    exec(compile(fallback["isic_caption_core.py"], "fixture_caption_core", "exec"), core.__dict__)
    assets = {}
    fingerprints = {}
    for name in ("florence", "clip"):
        path = root / name; path.mkdir()
        common.write_json(path / "config.json", {"model_type": "florence2" if name == "florence" else "clip"})
        torch.save({"test_fixture_only": torch.tensor(1)}, path / "pytorch_model.bin")
        assets[name] = path
        fingerprints[name] = core.checkpoint_asset_identity(path)
    frames = {role: common.read_rows(run / (role + "_split.csv")) for role in ("train", "val", "test")}
    records = [{k: row[k] for k in ("image_id", "image_path", "split", "image_sha256")} for rows in frames.values() for row in rows]
    folder = run / "shared/caption_generation"
    settings = {"batch_size": 8, "max_new_tokens": 64, "seed": 42, "precision": "float32"}
    generation = {"schema": 1, "method": core.METHOD, "model_id": core.MODEL_ID, "model_assets": fingerprints["florence"],
        "parameters": 1, "prompt": core.TASK_PROMPT, "decoding": "greedy_beam1_prefix_cleanup_first8", "dtype": "float32",
        "policy": core.POLICY, "images": [{k:r[k] for k in ("image_id","split","image_sha256")} for r in records],
        "settings": settings, "runtime": {"fixture": True}, "code_sha256": common.sha(run / "embedded_runtime/isic_caption_core.py")}
    generation_identity = common.object_hash(generation)
    rows = [core.make_generation_row(record,
        "brown uneven surface with pale scattered marks" if i % 3 else "melanoma is visible here on brown skin",
        False, generation_identity) for i, record in enumerate(records)]
    folder.mkdir(parents=True)
    with (folder / "captions.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=core.COLUMNS); writer.writeheader(); writer.writerows(rows)
    csv_sha = common.sha(folder / "captions.csv")
    common.write_json(folder / "generation_protocol/protocol.json", generation)
    common.write_json(folder / "status.json", {"complete": True, "csv_sha256": csv_sha, "identity": generation_identity})
    text_protocol = {"schema": 1, "caption_csv_sha256": csv_sha, "clip_assets": fingerprints["clip"],
        "policy": core.POLICY, "runtime": {"fixture": True}, "code_sha256": generation["code_sha256"]}
    common.write_json(folder / "text_cache/protocol.json", text_protocol)
    valid = torch.tensor([str(row["caption_valid"]).lower() == "true" for row in rows])
    vectors = torch.nn.functional.normalize(torch.randn(len(rows), 512, generator=torch.Generator().manual_seed(24)), dim=1)
    vectors[~valid] = 0
    table = {"identity": common.object_hash(text_protocol), "image_ids": [r["image_id"] for r in rows],
        "embeddings": vectors, "valid": valid, "caption_csv_sha256": csv_sha}
    work.atomic_torch(folder / "text_cache/clip_text.pt", table)
    common.write_json(folder / "web_caption_settings.json", {"settings": settings, "policy": core.POLICY,
        "method": core.METHOD, "model_id": core.MODEL_ID, "florence_assets": fingerprints["florence"], "clip_assets": fingerprints["clip"]})
    binding = common.read_json(run / "run_binding.json")
    binding.update(config=config, runtime=common.source_identity(run / "embedded_runtime"))
    binding["assets"].update(caption_csv_sha256=csv_sha, text_cache_sha256=common.sha(folder / "text_cache/clip_text.pt"),
        caption_settings_sha256=common.sha(folder / "web_caption_settings.json"))
    identity = common.object_hash(binding)
    common.write_json(run / "run_binding.json", binding)
    common.write_json(run / "protocol.json", {"identity": identity, "config": config})
    model = factory(config)
    with torch.no_grad():
        model.query_text.gate.bias.fill_(-1.)
        model.candidate_text.gate.bias.fill_(-1.)
    work.atomic_torch(run / "best.pt", {"identity": identity, "config": config, "epoch": 49,
        "validation": {"dice": .75, "iou": .6, "loss": .4, "q_gate": .2, "c_gate": .2}, "state": model.trainable_state()})
    return run, dataset, repo, weights, factory, fallback, assets


def prepare_fixture(root):
    run, dataset, repo, weights, factory, fallback, assets = make_multimodal(root)
    ctx = export.prepare(run, dataset, repo, weights, root / "export", fallback,
        florence_ref=assets["florence"], clip_ref=assets["clip"], offline_assets=root / "no_wheels", device="cpu", factory=factory)
    return ctx


def test_notebook_is_self_contained_without_training_or_caption_regeneration():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text("utf8")) == notebook
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    sections = {c["metadata"]["section"]: ''.join(c["source"]) for c in notebook["cells"]}
    for c in notebook["cells"]:
        if c["cell_type"] != "code": continue
        assert not c["outputs"] and c["execution_count"] is None
        tree = ast.parse(''.join(c["source"]))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                assert name not in {"run_training", "train_epoch", "AdamW", "backward", "fit", "prepare_captions", "generate_isic_records"}
    assert 'RUN_FULL_TEST = False' in sections["configuration"]
    assert 'DOWNLOAD_PART_MIB = 256' in sections["configuration"]
    assert 'TOKENIZERS_PARALLELISM' in sections["preflight"]
    ast.parse(export.FRESH_SOURCE); ast.parse(export.COLD_SOURCE)
    assert 'fresh_full_both_inference' in export.COLD_SOURCE


def test_full_both_parts_cached_text_and_real_transformer_parity(tmp_path, runtime_isolation):
    ctx = prepare_fixture(tmp_path)
    before = {str(p.relative_to(ctx["run"])):common.sha(p) for p in ctx["run"].rglob('*') if p.is_file()}
    assert common.discover_run(ctx["run"].parent, mode="mtacs") == ctx["run"]
    assert ctx["snapshot"]["completed_epochs"] == 94
    common.build(ctx)
    stage = ctx["stage"]
    for name in ("captions.csv", "text_embeddings.pt", "caption_settings.json", "caption_lookup.json",
                 "caption_models/florence/pytorch_model.bin", "caption_models/clip/pytorch_model.bin",
                 "runtime/isic_caption_web.py", "runtime/isic_caption_core.py"):
        assert (stage / name).is_file()
    for src, dest in (("captions.csv", "captions.csv"), ("text_cache/clip_text.pt", "text_embeddings.pt")):
        assert common.sha(ctx["caption_root"] / src) == common.sha(stage / dest)
    selector = torch.load(stage / 'models/selector.pt', weights_only=True)
    segmentor = torch.load(stage / 'models/segmentor.pt', weights_only=True)
    original = torch.load(ctx["run"] / 'best.pt', weights_only=True)['state']
    assert 'semantic_mix_logit' in selector
    assert any(k.startswith('query_text.') for k in segmentor)
    assert any(k.startswith('candidate_text.') for k in segmentor)
    assert not set(selector) & set(segmentor)
    assert all(torch.equal(v, original[k]) for k,v in {**selector,**segmentor}.items())
    common.verify_and_publish(ctx)
    assert common.read_json(stage / 'manifest.json')['status'] == 'fixture_unverified'
    result = common.read_json(stage / 'example/prediction.json')
    assert result['evidence']['query_caption']['status'] == 'cached'
    assert result['evidence']['fusion']['q_gate'] > 0
    # Invalid candidate captions stay zero and cannot contribute through c_gate.
    candidate = result['evidence']['candidate']
    assert bool(result['evidence']['fusion']['c_gate'] > 0) == candidate['caption_valid']
    assert common.read_json(stage / 'caption_pixel_ambiguities.json')
    with pytest.raises(ValueError, match='Florence/CLIP'):
        export.package_downloads(ctx)
    assert before == {str(p.relative_to(ctx["run"])):common.sha(p) for p in ctx["run"].rglob('*') if p.is_file()}


def test_modified_caption_cache_is_rejected(tmp_path, runtime_isolation):
    run, dataset, repo, weights, factory, fallback, assets = make_multimodal(tmp_path)
    path = run / 'shared/caption_generation/text_cache/clip_text.pt'
    altered = torch.load(path, weights_only=True)
    altered['embeddings'][1,0] += .1
    torch.save(altered, path)
    with pytest.raises(FileNotFoundError, match='ORIGINAL'):
        export.prepare(run, dataset, repo, weights, tmp_path / 'export', fallback,
            florence_ref=assets['florence'], clip_ref=assets['clip'], device='cpu', factory=factory)


def test_independent_fresh_and_cold_scripts_with_explicit_fixture_injection(tmp_path, runtime_isolation):
    ctx = prepare_fixture(tmp_path)
    common.build(ctx)
    stage = ctx['stage']
    # Exercise the actual subprocess scripts, substituting only the pretrained
    # teacher/CLIP in this small offline fixture. It may never become ready.
    prefix = f'''import sys, torch
from pathlib import Path
from types import SimpleNamespace
root=Path({str(stage)!r})
sys.path.insert(0,str(root/'runtime'))
import isic_caption_core as core
import isic_caption_web as caption
cache=torch.load(root/'text_embeddings.pt',map_location='cpu',weights_only=True)
i=next(i for i,v in enumerate(cache['valid']) if bool(v))
vec=cache['embeddings'][i]
class FixtureEngine:
    def __init__(self,*args):
        self.device=torch.device('cpu')
        self.tokenizer=lambda *a,**kw:{{'input_ids':torch.ones(1,2,dtype=torch.long)}}
        class Clip:
            config=SimpleNamespace(max_position_embeddings=77)
            def __call__(self,**kw):return SimpleNamespace(text_embeds=vec[None])
        self.clip=Clip()
    def caption(self,image):
        raw='brown irregular surface with pale small marks'
        text,valid,clinical,count=core.caption_policy(raw)
        return {{'raw_caption':raw,'final_caption':text,'caption_valid':valid,'word_count':count}},vec,torch.tensor(valid)
caption._LocalCaptionEngine=FixtureEngine
sys.argv=['fixture',str(root),'cpu']
'''
    fresh_script = tmp_path / 'fresh_fixture.py'
    fresh_script.write_text(prefix + export.FRESH_SOURCE, encoding='utf8')
    result = subprocess.run([sys.executable, str(fresh_script)], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert common.read_json(stage/'example/fresh_caption_check.json')['original_clip_embedding_parity'] is True
    common.stage_manifest(ctx)
    dino_repo = ROOT / 'artifacts/deployment/sun397/tacs_visual/vendor/dinov3'
    prefix = f'''import sys, torch
from pathlib import Path
root=Path({str(stage)!r})
sys.path.insert(0,{str(dino_repo)!r})
sys.path.insert(0,str(root/'runtime'))
from dinov3.models.vision_transformer import DinoVisionTransformer
import isic_segmentation_web as module
def tiny_factory(cfg):
    backbone=DinoVisionTransformer(img_size=32,patch_size=16,embed_dim=32,depth=4,num_heads=4,n_storage_tokens=2,ffn_ratio=2,drop_path_rate=0.)
    backbone.load_state_dict(torch.load(root/'models/frozen_fixture.pth',map_location='cpu',weights_only=True))
    return module.ISICSegmentor(backbone,cfg)
real_predictor=module.Predictor
module.Predictor=lambda *a,**kw:real_predictor(*a,factory=tiny_factory,**kw)
sys.argv=['fixture',str(root),{str(ctx['example_path'])!r},'cpu']
'''
    cold_script = tmp_path / 'cold_fixture.py'
    cold_script.write_text(prefix + export.COLD_SOURCE, encoding='utf8')
    result = subprocess.run([sys.executable, str(cold_script)], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = common.read_json(stage/'example/fresh_segmentation_check.json')
    assert report['fresh_caption_used'] is True and report['query_caption']['status'] == 'fresh'
    assert report['fusion']['q_gate'] > 0
    # Direct script execution does not publish a ready manifest.
    assert common.read_json(stage/'manifest.json')['status'] == 'staged'
