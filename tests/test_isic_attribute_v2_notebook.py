"""Caption extraction/provenance/phase guards; no VLM accuracy is asserted."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/notebook_builders'), str(ROOT / 'src/captioning')]
import build_isic_attribute_v2_notebook as builder
from isic_caption_attributes_v2 import attribute_caption, attribute_caption_details


def module_from(source, name='test_attribute_core'):
    module = types.ModuleType(name)
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def test_real_generic_pair_recovers_distinct_attribute_descriptions():
    path = ROOT / 'artifacts/deployment/isic2017/mtacs_full_both/captions.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = {r['image_id']:r for r in csv.DictReader(handle)}
    q, c = rows['ISIC_0000098'], rows['ISIC_0000413']
    assert q['final_caption'] == c['final_caption']
    q_new, c_new = [attribute_caption(r['source_long_caption']) for r in (q,c)]
    assert q_new != c_new and q_new and c_new
    assert 'brown' in q_new and 'circle' in q_new
    assert 'brown' in c_new and 'area' in c_new
    assert "person's skin" not in q_new + c_new


def test_no_padding_from_boilerplate_and_no_cross_sentence_color_borrowing():
    assert attribute_caption("The image is a close-up of a person's skin. The skin appears to be.") == ''
    assert attribute_caption('White background. A small circular patch.') == ''
    result = attribute_caption_details('A small circular patch beside a large white ruler.')
    assert result['caption'] not in {'small white circular patch', 'large white circular patch'}
    core = module_from(builder.caption_core())
    text, valid, clinical, count = core.caption_policy('A small dark brown circular area. This could be melanoma.')
    assert text and not valid and clinical  # Clinical filter still checks the WHOLE original response.


def test_core_and_notebook_are_current_and_caption_only_is_real():
    notebook = builder.build_notebook()
    assert json.loads(builder.TARGET.read_text('utf8')) == notebook
    sections = {c['metadata']['section']:''.join(c['source']) for c in notebook['cells']}
    assert len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']))
            assert c['outputs'] == [] and c['execution_count'] is None
    assert 'CAPTIONS_ONLY = True' in sections['configuration']
    assert 'isic2017_mtacs_attributes_v2' in sections['configuration']
    for name in ('model','bank','training','evaluation','export'):
        tree = ast.parse(sections[name])
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
        assert ast.unparse(tree.body[0].test) == 'not CAPTIONS_ONLY'
    sources = builder.sources()
    core = module_from(sources['isic_caption_core.py'])
    assert core.POLICY == builder.POLICY and core.METHOD == builder.METHOD
    assert 'long_response_source' in sources['isic_caption_core.py']
    assert 'prior_captions_source' in sources['isic_caption_web.py']
    assert 'CORE' not in sections['configuration']  # No side-effectful inference/training shortcut.


def test_reuse_long_responses_creates_new_rows_without_captioner_or_old_mutation(tmp_path):
    core = module_from(builder.caption_core())
    old_core = module_from(builder.baseline.caption_core(), 'test_old_core')
    model = tmp_path / 'florence'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'explicit-test-fixture-no-model-loading')
    prior = tmp_path / 'old_bundle'; prior.mkdir()
    records = []
    for i, role in enumerate(('train','val')):
        p = tmp_path / f'ISIC_{i:07d}.png'
        Image.new('RGB',(20,20),(50+i,30,20)).save(p)
        records.append({'image_id':p.stem,'split':role,'image_path':str(p),'image_sha256':core.file_sha256(p)})
    protocol = {'model_assets':core.checkpoint_asset_identity(model),
        'images':[{k:r[k] for k in ('image_id','split','image_sha256')} for r in records], 'policy':old_core.POLICY}
    old_identity = core.object_hash(protocol)
    responses = ["The image is a close-up of a person's skin. In the center of the skin there is a small dark brown circle.",
                 "The image shows skin. A large circular area is covered in a light brown substance."]
    rows = [old_core.make_generation_row(r,s,False,old_identity) for r,s in zip(records,responses)]
    with (prior / 'captions.csv').open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=core.COLUMNS);writer.writeheader();writer.writerows(rows)
    core.atomic_json(prior / 'caption_provenance/generation_protocol/protocol.json',protocol)
    core.atomic_json(prior / 'caption_provenance/status.json',{'identity':old_identity,'complete':True,'csv_sha256':core.file_sha256(prior/'captions.csv')})
    payload = {'records':records,'model_path':str(model),'prior_captions_source':str(prior),
        'output':str(tmp_path/'new_captions'),'settings':{'batch_size':8},'max_new_images':0}
    before = {str(p.relative_to(prior)):core.file_sha256(p) for p in prior.rglob('*') if p.is_file()}
    identity, reused = core.prior_caption_registration(payload)
    assert identity['reusable_images'] == 2 and len(reused) == 2
    def forbidden_factory(*args):
        raise AssertionError('No VLM inference should run when registered long responses are available')
    new_identity = core.object_hash({'policy':core.POLICY,'long_response_source':identity})
    core.validate_caption_records(records)
    core.generate_attribute_records(payload,new_identity,forbidden_factory)
    generated,status = core.validated_export(payload)
    assert len(generated) == 2 and status['complete']
    assert generated[0]['final_caption'] != rows[0]['final_caption']
    assert all(r['source_long_caption'] == old['source_long_caption'] for r,old in zip(generated,rows))
    assert all(r['prompt_version'] == builder.POLICY and r['captioner_fingerprint'] == new_identity for r in generated)
    assert before == {str(p.relative_to(prior)):core.file_sha256(p) for p in prior.rglob('*') if p.is_file()}
    changed = dict(payload,records=[dict(records[0],image_sha256='0'*64), records[1]])
    with pytest.raises(ValueError,match='another image/split'):
        core.prior_caption_registration(changed)


def test_saved_split_recovery_preserves_exact_order_and_original_bundle_root(tmp_path, monkeypatch):
    snapshot = module_from(builder.frame_helper_source(), 'test_snapshot_helpers')
    source = tmp_path / 'bundle/provenance/original_run'; source.mkdir(parents=True)
    data = tmp_path / 'dataset'
    config = {'mode':'mtacs','schema':'isic2017_segmentation_web_v1','backbone':'dinov3_vitl16',
        'feature_dim':1024,'image_size':448,'context_top_k':1,'pool_fraction':.2,'seed':42}
    frames = {}
    for role, count, offset in [('train',10,0),('val',1,100),('test',1,200)]:
        frame=[]
        for i in range(count):
            identity=f'ISIC_{i+offset:07d}'
            image=data/role/'images'/(identity+'.png'); image.parent.mkdir(parents=True,exist_ok=True)
            mask=data/role/'masks'/(identity+'_segmentation.png'); mask.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(20,20),(i+offset,30,20)).save(image);Image.new('L',(20,20),255).save(mask)
            pixels=np.asarray(Image.open(image).convert('RGB'))
            row={'image_id':identity,'split':role,'image_sha256':snapshot.sha(image),'mask_sha256':snapshot.sha(mask),
                'pixel_sha256':hashlib.sha256(pixels.tobytes()+str((20,20)).encode()).hexdigest(),'group_id':'',
                'image_path':str(image),'mask_path':str(mask)}
            frame.append(row)
        frames[role]=frame
    pool=[frames['train'][4],frames['train'][1]]
    binding={'config':config,'splits':{role:[{k:r[k] for k in ('image_id','image_sha256','mask_sha256','group_id')} for r in rows]
        for role,rows in frames.items()},'pool_ids':[r['image_id'] for r in pool]}
    hash_object=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
    (source/'run_binding.json').write_text(json.dumps(binding))
    (source/'protocol.json').write_text(json.dumps({'identity':hash_object(binding)}))
    for name,rows in [(f'{role}_split.csv', rows) for role,rows in frames.items()]+[('candidate_pool.csv',pool)]:
        with (source/name).open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    workflow=types.SimpleNamespace(resolve_dataset_root=lambda path:Path(path),object_hash=hash_object,
        tqdm=lambda values,**kwargs:values,
        pixel_hash=lambda im:hashlib.sha256(np.asarray(im).tobytes()+str(im.size).encode()).hexdigest())
    def read_rgb(path):
        with Image.open(path) as im:return im.convert('RGB')
    scope={'SAVED_SPLITS_SOURCE':str(tmp_path/'bundle'),'CONFIG':config,'ISIC_ROOT':str(data),'LONG_CAPTIONS_SOURCE':'',
        'workflow':workflow,'snapshot_io':snapshot,'model_code':types.SimpleNamespace(io_utils=types.SimpleNamespace(read_rgb=read_rgb))}
    monkeypatch.setenv('ISIC_ATTRIBUTE_LONG_CAPTIONS_SOURCE','')
    exec('import os, json\nfrom pathlib import Path\n'+builder.DATASET_RECOVERY,scope)
    assert [r['image_id'] for r in scope['CANDIDATE_POOL']] == binding['pool_ids']
    assert scope['LONG_CAPTIONS_SOURCE'] == str(tmp_path/'bundle')
    assert len(scope['FRAMES']['train']) == 10 and len(scope['FRAMES']['test']) == 1
