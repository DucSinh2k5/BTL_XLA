"""Snapshot recovery checks; tiny fixture outputs can never be served as real models."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in ("src/deployment", "src/models", "src/training", "tools/notebook_builders"):
    sys.path.insert(0, str(ROOT / folder))
import isic_export_only as export
import isic_segmentation_web as original_net
import train_isic_segmentation_web as original_work
import build_isic_export_only_notebook as builder

torch.set_num_threads(1)


@pytest.fixture
def runtime_isolation(monkeypatch):
    paths = list(sys.path)
    for name in ("isic_segmentation_web", "train_isic_segmentation_web", "tacs_visual_web"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in ("isic_segmentation_web", "train_isic_segmentation_web", "tacs_visual_web"):
        sys.modules.pop(name, None)
    sys.path[:] = paths


def test_downloaded_real_metadata_and_original_runtime_match():
    config, binding, frames, pool, snapshot = export.validate_metadata(builder.SAVED_RUN)
    assert config["image_size"] == 448 and config["feature_dim"] == 1024
    assert snapshot["best_epoch"] == 49 and snapshot["completed_epochs"] == 94
    assert snapshot["training_complete"] is False
    assert {key:len(rows) for key, rows in frames.items()} == {"train":2000, "val":149, "test":597}
    assert len(pool) == 400
    assert len(snapshot['repeated_pixels_within_split']) == 2
    assert export.source_identity(builder.SAVED_RUN / "embedded_runtime") == binding["runtime"]
    class ShapeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim, self.n_storage_tokens = 1024, 4
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])
    with torch.device('meta'):
        shape_model = original_net.ISICSegmentor(ShapeBackbone(), config)
    actual = torch.load(builder.SAVED_RUN/'best.pt',map_location='meta',weights_only=True)
    shape_model.restore_trainable(actual['state'])
    assert len(shape_model.context_blocks)==24


def test_notebook_is_current_self_contained_and_has_no_executable_training():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text("utf-8")) == notebook
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    sections = {c["metadata"]["section"]: ''.join(c["source"]) for c in notebook["cells"]}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse(''.join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    assert name not in {"run_training", "train_epoch", "AdamW", "backward", "fit"}
    assert 'RUN_FULL_TEST = False' in sections["configuration"]
    assert 'DOWNLOAD_PART_MIB = 256' in sections["configuration"]
    assert 'COLD_RELOAD_DEVICE = DEVICE' in sections["configuration"]
    assert 'TOKENIZERS_PARALLELISM' in sections["preflight"]
    ast.parse(export.COLD_SOURCE)
    ast.parse(export.JOIN_SOURCE)


def test_archive_stream_parts_roundtrip_and_detect_corruption(tmp_path):
    sink = export._SplitArchive(tmp_path, 'bundle.zip', 1024)
    contents = {"bundle/models/selector.pt":os.urandom(5000), "bundle/config.json":b'{"mode":"tacs"}'}
    with zipfile.ZipFile(sink, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    sink.close()
    combined = b''.join(p.read_bytes() for p in sink.files)
    assert all(p.stat().st_size <= 1024 for p in sink.files)
    with zipfile.ZipFile(io.BytesIO(combined)) as archive:
        assert archive.testzip() is None
        assert {name:archive.read(name) for name in archive.namelist()} == contents
    export.write_json(tmp_path / 'download_manifest.json', {'archive_name':'bundle.zip', 'archive_sha256':hashlib.sha256(combined).hexdigest(),
        'parts':[{'name':p.name,'bytes':p.stat().st_size,'sha256':export.sha(p)} for p in sink.files]})
    joiner = tmp_path / 'join_downloads.py'
    joiner.write_text(export.JOIN_SOURCE, encoding='utf8')
    result = subprocess.run([sys.executable, str(joiner)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'bundle.zip').read_bytes() == combined
    first = sink.files[0]
    first.write_bytes(b'X' + first.read_bytes()[1:])
    failed = subprocess.run([sys.executable, str(joiner), str(tmp_path / 'corrupted.zip')], text=True, capture_output=True)
    assert failed.returncode != 0 and 'Corrupt part' in failed.stderr
    assert not (tmp_path / 'corrupted.zip').exists()


def make_fixture(root):
    repo = ROOT / 'artifacts/deployment/sun397/tacs_visual/vendor/dinov3'
    if not (repo / 'dinov3/models/vision_transformer.py').is_file():
        pytest.skip('The official DINOv3 source is required for tiny real-transformer checks')
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    config = original_work.default_config('tacs')
    config.update(feature_dim=32, cross_heads=4, decoder_channels=8, image_size=32, epochs=100,
                  batch_size=1, effective_batch_size=4, eval_batch_size=1, feature_batch_size=2,
                  strict_official_counts=False)
    def factory(cfg):
        with torch.random.fork_rng():
            torch.manual_seed(13)
            backbone = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=32, depth=4, num_heads=4,
                n_storage_tokens=2, ffn_ratio=2, drop_path_rate=0.)
            backbone.init_weights()
            return original_net.ISICSegmentor(backbone, cfg)
    dataset = root / 'dataset'
    for role, count, offset in [('train',10,0),('val',2,100),('test',2,200)]:
        for index in range(count):
            identity = f'ISIC_{index+offset:07d}'
            pixels = np.random.default_rng(index+offset).integers(0,256,(42,38,3),dtype=np.uint8)
            mask = np.zeros((42,38),np.uint8); mask[6:35,8:30] = 255
            for folder, array, suffix in [('images',pixels,'.png'),('masks',mask,'_segmentation.png')]:
                path = dataset / role / folder / (identity+suffix)
                path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(array).save(path)
    # Match the real saved run's within-test pixel aliases with different masks.
    first = dataset/'test/images/ISIC_0000200.png'
    (dataset/'test/images/ISIC_0000201.png').write_bytes(first.read_bytes())
    conflict = np.zeros((42,38),np.uint8); conflict[12:20,12:20] = 255
    Image.fromarray(conflict).save(dataset/'test/masks/ISIC_0000201_segmentation.png')
    frames, pool = original_work.load_isic(dataset, config)
    run = root / 'run'; (run / 'embedded_runtime').mkdir(parents=True)
    for name in export.RUNTIME_FILES:
        (run / 'embedded_runtime' / name).write_bytes((builder.SAVED_RUN / 'embedded_runtime' / name).read_bytes())
    # The vendor fixture contains registered Python in data/ to catch accidentally dropping source folders.
    vendor = root / 'vendor'; (vendor / 'dinov3/data').mkdir(parents=True)
    (vendor / 'dinov3/data/keep.py').write_text('VALUE = 1\n')
    (vendor / 'LICENSE').write_text('Test-only vendor fixture; no production model license claim.')
    weights = root / 'frozen_fixture.pth'; torch.save(factory(config).backbone.state_dict(), weights)
    binding = {'config':config, 'assets':{'backbone_sha256':export.sha(weights),'dino_source':export.source_identity(vendor)},
        'runtime':export.source_identity(run / 'embedded_runtime'),
        'splits':{role:[{k:r[k] for k in ('image_id','image_sha256','mask_sha256','group_id')} for r in rows] for role,rows in frames.items()},
        'pool_ids':[r['image_id'] for r in pool]}
    identity = export.object_hash(binding)
    export.write_json(run / 'protocol.json',{'identity':identity,'config':config})
    export.write_json(run / 'run_binding.json',binding)
    for role, rows in frames.items(): original_work.csv_write(run / f'{role}_split.csv', rows)
    original_work.csv_write(run / 'candidate_pool.csv',pool)
    model = factory(config)
    with torch.no_grad(): model.decoder.head[-1].bias.add_(.3)
    validation = {'dice':.75, 'iou':.6, 'loss':.4, 'q_gate':0.,'c_gate':0.}
    original_work.atomic_torch(run / 'best.pt',{'identity':identity,'config':config,'state':model.trainable_state(),'epoch':49,'validation':validation})
    original_work.csv_write(run / 'history.csv',[{'epoch':i, 'val_dice':.75 if i==49 else .7, 'val_iou':.6} for i in range(1,95)])
    original_work.build_bank(model,pool,config,run / 'shared/candidate_bank.pt',binding['assets']['backbone_sha256'])
    return run,dataset,vendor,weights,factory


def test_interrupted_best_only_snapshot_exports_exact_parts_and_attention(tmp_path, runtime_isolation):
    run,dataset,repo,weights,factory = make_fixture(tmp_path)
    fallback = {p.name:p.read_text('utf8') for p in (run/'embedded_runtime').glob('*.py')}
    before = {str(p.relative_to(run)):export.sha(p) for p in run.rglob('*') if p.is_file()}
    ctx = export.prepare(run,dataset,repo,weights,tmp_path/'export',fallback,device='cpu',factory=factory)
    assert not (run/'last.pt').exists()
    assert ctx['snapshot']['training_complete'] is False
    assert ctx['snapshot']['completed_epochs'] == 94
    export.build(ctx)
    for name in ['models/selector.pt','models/segmentor.pt','candidate_bank.pt','ground_truth_index.json','vendor/dinov3/dinov3/data/keep.py']:
        assert (ctx['stage']/name).is_file()
    selector = torch.load(ctx['stage']/'models/selector.pt',weights_only=True)
    segmentor = torch.load(ctx['stage']/'models/segmentor.pt',weights_only=True)
    original = torch.load(run/'best.pt',weights_only=True)['state']
    assert set(selector).isdisjoint(segmentor)
    assert set(selector)|set(segmentor)==set(original)
    assert all(torch.equal(value,original[name]) for name,value in {**selector,**segmentor}.items())
    assert export.read_json(ctx['stage']/'metrics.json')['test_status']=='not_run_export_only'
    conflicts = export.read_json(ctx['stage']/'ground_truth_ambiguities.json')
    assert conflicts[0]['automatic_ground_truth']=='omitted_conflicting_masks'
    assert conflicts[0]['pixel_sha256'] not in export.read_json(ctx['stage']/'ground_truth_index.json')
    export.verify_and_publish(ctx)
    manifest = export.read_json(ctx['stage']/'manifest.json')
    assert manifest['status']=='fixture_unverified'
    assert manifest['training_snapshot']['training_complete'] is False
    result = export.read_json(ctx['stage']/'example/prediction.json')
    assert result['evidence']['attention']['query_segmentor']['grid_shape']==[2,2]
    assert result['evidence']['attention']['candidate_selector']['grid_shape']==[2,2]
    assert result['ground_truth']['verified'] is True
    # Execute the exact production verification/serialization code in a NEW
    # process with the explicitly injected tiny backbone. No ready model is forged.
    dino_repo = ROOT/'artifacts/deployment/sun397/tacs_visual/vendor/dinov3'
    wrapper = tmp_path/'cold_fixture.py'
    prefix = f'''import sys, torch
from pathlib import Path
root=Path({str(ctx['stage'])!r})
sys.path.insert(0,{str(dino_repo)!r})
sys.path.insert(0,str(root/'runtime'))
from dinov3.models.vision_transformer import DinoVisionTransformer
import isic_segmentation_web as module
def tiny_factory(cfg):
    backbone=DinoVisionTransformer(img_size=32,patch_size=16,embed_dim=32,depth=4,num_heads=4,
        n_storage_tokens=2,ffn_ratio=2,drop_path_rate=0.)
    backbone.load_state_dict(torch.load(root/'models/frozen_fixture.pth',map_location='cpu',weights_only=True))
    return module.ISICSegmentor(backbone,cfg)
real_predictor=module.Predictor
module.Predictor=lambda *a,**kw:real_predictor(*a,factory=tiny_factory,**kw)
sys.argv=['fixture',str(root),{str(ctx['example_path'])!r},'cpu']
'''
    wrapper.write_text(prefix+export.COLD_SOURCE,encoding='utf8')
    cold = subprocess.run([sys.executable,str(wrapper)],text=True,capture_output=True,timeout=60)
    assert cold.returncode==0,cold.stderr
    assert export.read_json(ctx['stage']/'smoke_test.json')['cold_offline_reload'] is True
    for name in ['mask.png','overlay.png','query_segmentor.png','candidate_selector.png','reloaded_logits.pt']:
        assert (ctx['stage']/'example'/name).is_file()
    with pytest.raises(ValueError,match='production'):
        ctx['net'].Predictor(ctx['stage'])
    with pytest.raises(ValueError,match='verified production'):
        export.package_downloads(ctx)
    assert before == {str(p.relative_to(run)):export.sha(p) for p in run.rglob('*') if p.is_file()}


def test_wrong_pool_order_is_rejected_before_any_model_load(tmp_path):
    # Real saved run metadata can be copied without copying the large best tensor storage.
    run = tmp_path/'run'; run.mkdir()
    for name in ['protocol.json','run_binding.json','candidate_pool.csv','train_split.csv','val_split.csv','test_split.csv']:
        (run/name).write_bytes((builder.SAVED_RUN/name).read_bytes())
    meta = torch.load(builder.SAVED_RUN/'best.pt',map_location='meta',weights_only=True)
    torch.save(meta,run/'best.pt')
    rows=export.read_rows(run/'candidate_pool.csv')
    original_work.csv_write(run/'candidate_pool.csv',list(reversed(rows)))
    with pytest.raises(ValueError,match='Candidate order'):
        export.validate_metadata(run)
