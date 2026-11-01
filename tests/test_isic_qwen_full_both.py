"""CPU contracts for the integrated ISIC notebook; never load real model weights."""
import ast
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import build_isic_qwen8b_full_both_notebook as builder
import isic_qwen_caption as captions
from mtacs_isic_full_both import make_isic_full_both_class, text_for_indices


TEXT = "dark brown center with pale irregular border"


def test_standalone_notebook_has_complete_ordered_workflow_and_empty_outputs():
    nb = builder.build_notebook()
    labels = [c["metadata"]["section"] for c in nb["cells"]]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert len(set(c["id"] for c in nb["cells"])) == len(nb["cells"])
    assert labels.index("generate_and_encode") < labels.index("multimodal_model") < labels.index("train_and_resume") < labels.index("test_and_comparison")
    config = "".join(next(c["source"] for c in nb["cells"] if c["metadata"]["section"] == "configuration"))
    assert 'IMAGE_SIZE = 448' in config and 'EPOCHS = 100' in config and 'PATIENCE = 15' in config
    assert 'BATCH_SIZE = 8' in config and 'CANDIDATE_POOL_FRACTION = 0.20' in config
    assert 'mtacs_isic2017_448_qwen8b_full_both' in config
    assert nb["metadata"]["isic_full_both"]["baseline_metrics_are_rounded_stdout"] == {"dice": .856, "iou": .7738}
    assert nb["metadata"]["isic_full_both"]["context_k"] == 1


def test_embedded_caption_module_executes_without_repo_imports_or_gpu():
    core = builder.embed_caption_core()
    ns = {"__name__": "test_embedded"}
    exec(core, ns)
    assert ns["CAPTION_PROMPT"] == captions.CAPTION_PROMPT
    assert ns["caption_policy"](TEXT) == captions.caption_policy(TEXT)
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("dtd_", "isic_")) for n in ast.parse(core).body)
    assert "mask" not in ns["validate_caption_records"].__annotations__


def test_generated_notebook_is_current_and_baseline_was_not_modified():
    assert json.loads(builder.TARGET.read_text(encoding="utf-8")) == builder.build_notebook()
    assert hashlib.sha256(builder.SOURCE.read_bytes()).hexdigest() == "052b4edf0110f2caf6d1b2e9a1d8619dfdfb57d56d97c8768553b9a566215599"


@pytest.mark.parametrize("text, limited, valid, clinical", [
    (TEXT, False, True, False),
    ("The image shows " + TEXT, False, True, False),
    (TEXT, True, False, False),
    ("dark skin", False, False, False),
    ("benign brown patch with pale uneven edges", False, False, True),
    (TEXT + " consistent with melanoma", False, False, True),
    ("striped brown region with dotted pale border", False, True, False),
])
def test_isic_policy_is_not_the_dtd_class_filter(text, limited, valid, clinical):
    clean, actual, found, count = captions.caption_policy(text, limited)
    assert actual == valid and found == clinical
    if actual:
        assert 5 <= count <= 8 and not clean.lower().startswith(('the ', 'an ', 'a '))


@pytest.fixture
def payload(tmp_path):
    records = []
    for split in ("train", "val", "test"):
        for j in range(2):
            path = tmp_path / f"{split}_{j}.png"
            Image.new('RGB', (16, 16), (100 + j, 60, 40)).save(path)
            records.append({'image_id': f'{split}_{j}', 'image_path': str(path), 'split': split,
                            'image_sha256': captions.file_sha256(path)})
    return {'records': records, 'settings': {'batch_size': 2}, 'output': str(tmp_path / 'captions'),
            'model_path': 'mock', 'max_new_images': 0}


class FakeCaptioner:
    seen = []

    def __init__(self, path, settings):
        pass

    def generate(self, images):
        self.seen.append(len(images))
        return [(TEXT, False) for image in images]


def test_smoke_test_resume_exports_only_complete_csv(payload):
    from pathlib import Path
    FakeCaptioner.seen = []
    captions.validate_caption_records(payload['records'])
    payload['max_new_images'] = 2
    status = captions.generate_isic_records(payload, 'identity', FakeCaptioner)
    assert not status['complete'] and not (Path(payload['output']) / 'captions.csv').exists()
    with pytest.raises(RuntimeError, match='partial'):
        captions.validated_export(payload)
    payload['max_new_images'] = 0
    status = captions.generate_isic_records(payload, 'identity', FakeCaptioner)
    rows, exported = captions.validated_export(payload)
    assert status['complete'] and len(rows) == 6 and sum(FakeCaptioner.seen) == 6
    captions.generate_isic_records(payload, 'identity', FakeCaptioner)
    assert sum(FakeCaptioner.seen) == 6
    assert all(row['source_long_caption'] == TEXT and row['caption_valid'] == 'True' for row in rows)


def test_masks_diagnoses_and_duplicate_ids_cannot_enter_caption_worker(payload):
    record = dict(payload['records'][0], mask_path='must not enter the VLM')
    with pytest.raises(ValueError, match='never masks'):
        captions.validate_caption_records([record])
    with pytest.raises(ValueError, match='unique'):
        captions.validate_caption_records(payload['records'] + [payload['records'][0]])


def test_caption_generation_oom_reduces_batch_without_dropping_rows(payload):
    class SmallGPU(FakeCaptioner):
        def generate(self, images):
            if len(images) > 1:
                raise torch.cuda.OutOfMemoryError('synthetic')
            return super().generate(images)
    FakeCaptioner.seen = []
    status = captions.generate_isic_records(payload, 'identity', SmallGPU)
    assert status['rows'] == 6 and sum(FakeCaptioner.seen) == 6


def test_text_cache_alignment_zero_masks_and_unit_norm():
    rows = [{'image_id': 'a', 'caption_valid': 'true'}, {'image_id': 'b', 'caption_valid': 'false'}]
    cache = {'identity': 'id', 'image_ids': ['a', 'b'], 'caption_csv_sha256': 'sha',
             'valid': torch.tensor([True, False]), 'embeddings': torch.tensor([[1., 0.], [0., 0.]])}
    captions.validate_text_cache(cache, rows, 'sha', 'id')
    cache['valid'] = torch.tensor([True, True])
    with pytest.raises(ValueError, match='masks'):
        captions.validate_text_cache(cache, rows, 'sha', 'id')
    cache['valid'] = torch.tensor([True, False])
    cache['embeddings'][1, 0] = 1
    with pytest.raises(ValueError, match='zero'):
        captions.validate_text_cache(cache, rows, 'sha', 'id')


def test_straight_through_selected_caption_has_gradient_to_selector_scores():
    torch.manual_seed(42)
    scores = torch.randn(2, 4, requires_grad=True)
    weights = F.gumbel_softmax(scores, tau=.5, hard=True, dim=1)
    text, valid = text_for_indices(weights, torch.randn(4, 6), torch.ones(4, dtype=torch.bool))
    assert torch.equal(valid, torch.ones(2))
    text.square().sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    ns = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'DEVICE': torch.device('cpu'), 'IMAGE_SIZE': 16,
          'TACS_SELECTOR_DROPOUT': 0., 'TACS_SELECTOR_TEMPERATURE': 1., 'NUM_CROSS_ATTENTION_LAYERS': 2,
          'CROSS_ATTENTION_HEADS': 2, 'DECODER_DROPOUT': 0.}
    baseline = json.loads(builder.SOURCE.read_text(encoding='utf-8'))
    source = ''.join(baseline['cells'][12]['source'])
    for name in ('TACSSelector', 'GatedCrossAttentionBlock', 'DPTLiteHead', 'TACSSegmentationModel'):
        exec(builder.definition(source, name), ns)

    class FrozenStub(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.proj = nn.Linear(3, 8).requires_grad_(False)

        def forward_features(self, images):
            with torch.no_grad():
                values = F.adaptive_avg_pool2d(images, 2).flatten(2).transpose(1, 2)
                tokens = self.proj(values)
                return tokens, tokens.mean(1)

    ns['FrozenDinoV3FeatureExtractor'] = FrozenStub
    return make_isic_full_both_class(ns['TACSSegmentationModel'])('mock', 'mock', 'mock', feature_dim=8, text_dim=6)


def test_real_visual_decoder_with_text_has_correct_shape_and_trains_both_text_heads(tiny_model):
    torch.manual_seed(4)
    model = tiny_model
    tokens, global_features = model.encode_images(torch.randn(2, 3, 16, 16))
    candidate_tokens = torch.randn(3, 4, 8)
    q_text, c_bank = torch.randn(2, 6), torch.randn(3, 6)
    valid = torch.ones(3, dtype=torch.bool)
    details = model.select_full_both(global_features, torch.randn(3, 8), q_text,
        torch.ones(2, dtype=torch.bool), c_bank, valid, forbidden_indices=torch.tensor([0, -1]), sample=True)
    assert details['selection_weights'][0, 0] == 0
    weights = details['selection_weights']
    context = (weights @ candidate_tokens.flatten(1)).reshape(2, 4, 8)
    c_text, c_valid = text_for_indices(weights, c_bank, valid)
    logits, q_gate, c_gate = model.decode_full_both(tokens, context, q_text, torch.ones(2), c_text, c_valid)
    assert logits.shape == (2, 1, 16, 16)
    assert (q_gate > 0).all() and (c_gate > 0).all()
    F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert model.query_text_projection[1].weight.grad.abs().sum() > 0
    assert model.candidate_text_projection[1].weight.grad.abs().sum() > 0
    assert model.semantic_mix_logit.grad is not None and torch.isfinite(model.semantic_mix_logit.grad)
    assert all(p.grad is None for p in model.feature_extractor.parameters())


def test_invalid_caption_masks_disable_text_fusion_and_semantic_score(tiny_model):
    tokens, text = torch.randn(2, 4, 8), torch.randn(2, 6)
    fused, gates = tiny_model.fuse_text(tokens, text, torch.zeros(2))
    assert torch.equal(fused, tokens) and torch.equal(gates, torch.zeros(2))
    tiny_model.eval()
    query, candidates = torch.randn(2, 8), torch.randn(3, 8)
    actual = tiny_model.select_full_both(query, candidates, text, torch.zeros(2, dtype=torch.bool),
                                       torch.randn(3, 6), torch.ones(3, dtype=torch.bool))
    assert torch.equal(actual['selector_scores'], tiny_model.selector.score(query, candidates))


def test_notebook_uses_st_text_in_task_path_and_hard_action_text_in_policy_and_eval():
    nb = builder.build_notebook()
    src = {c['metadata']['section']: ''.join(c['source']) for c in nb['cells']}
    training = src['train_and_resume']
    assert "text_for_indices(details['selection_weights']" in training
    assert 'query_only_with_text' in training
    assert 'candidate_text_bank[actions]' in training
    assert "val_metrics['dice'] > best_val_dice" in training
    assert 'test_loader' not in training
    assert 'candidate_text_bank[indices]' in src['locked_protocol_and_evaluation']
    assert "best_checkpoint_path" in src['test_and_comparison']
    assert 'TOKENIZERS_PARALLELISM' in src['offline_preflight']
    assert 'subprocess.Popen' in src['embedded_caption_worker']


def test_actual_train_and_evaluation_functions_run_on_cpu_with_tiny_data(tiny_model):
    import pandas as pd
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from mtacs_isic_notebook_cells import PROTOCOL_CELL, TRAIN_CELL
    model = tiny_model
    ids = ['q0', 'q1', 'q2', 'q3']
    q_bank, c_bank = torch.randn(4, 6), torch.randn(3, 6)
    q_valid = torch.tensor([True, False, True, True])
    c_valid = torch.tensor([True, True, False])
    c_bank[~c_valid] = 0
    patch_bank = torch.randn(3, 4, 8)
    dataset = [{'image': torch.rand(3, 16, 16), 'mask': (torch.rand(1, 16, 16) > .5).float(),
                'fname': image_id + '.jpg', 'image_id': image_id, 'row_id': i} for i, image_id in enumerate(ids)]
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ns = {'torch': torch, 'nn': nn, 'F': F, 'model': model, 'DEVICE': torch.device('cpu'), 'tqdm': tqdm,
          'pd': pd, 'train_loader': loader, 'optimizer': optimizer,
          'scaler': torch.amp.GradScaler('cuda', enabled=False),
          'TACS_POLICY_WEIGHT': 1., 'TACS_ENTROPY_WEIGHT': .01, 'TACS_GUMBEL_TAU': .1,
          'GRAD_CLIP_NORM': 1., 'THRESHOLD': .5, 'BCE_LOSS_WEIGHT': 1., 'DICE_LOSS_WEIGHT': 1.,
          'candidate_global_bank': torch.randn(3, 8), 'candidate_text_bank': c_bank, 'candidate_text_valid': c_valid,
          'candidate_df': pd.DataFrame({'image_id': ['c0', 'c1', 'c2']}),
          'caption_by_id': {image_id: TEXT for image_id in ids + ['c0', 'c1', 'c2']},
          'text_for_indices': text_for_indices}
    ns['query_text_inputs'] = lambda image_ids: (q_bank[[ids.index(x) for x in image_ids]], q_valid[[ids.index(x) for x in image_ids]])
    ns['forbidden_pool_indices_from_row_ids'] = lambda row_ids: torch.tensor([0 if int(i) == 0 else -1 for i in row_ids])
    ns['select_candidate_tokens_from_weights'] = lambda weights: (weights.float() @ patch_bank.flatten(1)).reshape(-1, 4, 8)
    ns['gather_candidate_tokens'] = lambda indices: patch_bank[indices]
    baseline = json.loads(builder.SOURCE.read_text(encoding='utf-8'))
    losses = ''.join(baseline['cells'][16]['source'])
    for name in ('segmentation_loss_per_sample', 'dice_iou_from_logits', 'soft_dice_iou_from_logits'):
        exec(builder.definition(losses, name), ns)
    for name in ('move_batch_to_device', 'full_both_selection', 'evaluate_full_both'):
        exec(builder.definition(PROTOCOL_CELL, name), ns)
    ns['evaluate_full_both'] = torch.no_grad()(ns['evaluate_full_both'])
    exec(builder.definition(TRAIN_CELL, 'train_full_both_epoch'), ns)
    train = ns['train_full_both_epoch'](1)
    metrics, rows = ns['evaluate_full_both'](loader, return_rows=True)
    assert all(math.isfinite(x) for x in train.values())
    assert 0 <= metrics['dice'] <= 1 and 0 <= metrics['iou'] <= 1
    assert len(rows) == len(dataset) and set(rows.image_id) == set(ids)
    assert metrics['query_caption_rate'] == .75
    assert {'query_caption', 'candidate_caption', 'context_image_id', 'q_gate', 'c_gate'}.issubset(rows.columns)
