"""Caption interventions preserve the original bundle and align text/masks/vectors."""
import csv
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'apps/sun397_demo'), str(ROOT / 'tools/notebook_builders')]
import isic_caption_override as override
from build_isic_caption_firstvalid_v3_notebook import runtime_source


def test_actual_csv_passes_policy_and_isic_identity_checks():
    core = types.ModuleType('test_override_core'); exec(runtime_source(), core.__dict__)
    new = override.read_rows(ROOT / 'captions (6).csv')
    old = override.read_rows(ROOT / 'artifacts/deployment/isic2017/mtacs_full_both/captions.csv')
    override.validate_rows(new, old, core)
    assert len(new) == 2750
    assert sum(not override.boolean(r['caption_valid']) for r in new) == 26
    changed = [dict(r) for r in new]
    changed[0]['final_caption'] = 'invented caption unrelated to its source'
    with pytest.raises(ValueError, match='violate'):
        override.validate_rows(changed, old, core)
    with pytest.raises(ValueError, match='image/split order'):
        override.validate_rows(new[::-1], old, core)


def test_new_embedding_mask_uses_zero_for_rejected_text():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {'input_ids': torch.ones((len(texts), 3), dtype=torch.long)}
    class Clip:
        config = types.SimpleNamespace(max_position_embeddings=77)
        def __call__(self, input_ids):
            return types.SimpleNamespace(text_embeds=torch.ones((len(input_ids), 512)))
    rows = [{'image_id':'a', 'caption_valid':'True', 'final_caption':'small dark brown circular skin area'},
            {'image_id':'b', 'caption_valid':'False', 'final_caption':'rejected'}]
    encoded = override.encode_rows(rows, Clip(), Tokenizer())
    assert encoded['valid'].tolist() == [True, False]
    assert torch.count_nonzero(encoded['embeddings'][1]) == 0
    assert encoded['embeddings'][0].norm().item() == pytest.approx(1.)


def test_fresh_image_retries_and_uses_first_valid_caption():
    from PIL import Image
    core = types.ModuleType('test_override_core'); exec(runtime_source(), core.__dict__)
    calls = []
    class Captioner:
        def generate_candidates(self, images, **kwargs):
            calls.append(kwargs['num_beams'])
            texts = ['The image shows skin.'] if kwargs['num_beams'] == 1 else [
                'A small dark brown circular area. This is melanoma.', 'A large light brown circular area.']
            return [[{'response':t, 'token_limit':False} for t in texts]]
    engine = override.FirstValidCaptionEngine.__new__(override.FirstValidCaptionEngine)
    engine.core = core; engine.captioner = Captioner()
    engine.settings = {'max_new_tokens':64, 'retry_greedy_tokens':128, 'retry_beam_tokens':192, 'retry_beams':4, 'retry_returns':3}
    engine.local = types.SimpleNamespace(device='cpu',
        tokenizer=lambda texts, **kw:{'input_ids':torch.ones((1,2),dtype=torch.long)},
        clip=types.SimpleNamespace())
    class Clip:
        config = types.SimpleNamespace(max_position_embeddings=77)
        def __call__(self, **kw): return types.SimpleNamespace(text_embeds=torch.ones((1,512)))
    engine.local.clip = Clip()
    row, vector, valid = engine.caption(Image.new('RGB',(20,20)))
    assert calls == [1,1,4] and valid.item()
    assert row['final_caption'] == 'large light brown circular area'
    assert row['selection_origin'] == 'beam_retry'
    assert not vector.is_inference()
