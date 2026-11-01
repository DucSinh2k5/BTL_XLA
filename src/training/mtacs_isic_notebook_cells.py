"""Standalone notebook cells for the ISIC Qwen full_both experiment."""

TEXT_BANK_CELL = r'''
rows, caption_status = validated_export(caption_payload)
CAPTION_CSV = CAPTION_OUTPUT / 'captions.csv'
text_path = CAPTION_OUTPUT / 'text_cache' / 'clip_text.pt'
text_status = json.loads((CAPTION_OUTPUT / 'text_cache' / 'status.json').read_text(encoding='utf-8'))
if file_sha256(text_path) != text_status['sha256']:
    raise ValueError('CLIP text artifact hash mismatch')
text_cache = torch.load(text_path, map_location='cpu', weights_only=True)
if text_cache['image_ids'] != [r['image_id'] for r in caption_records] or text_cache['caption_csv_sha256'] != caption_status['csv_sha256']:
    raise ValueError('CLIP vectors do not align with the current image IDs / captions.csv')
if text_cache['identity'] != text_status['identity']:
    raise ValueError('CLIP cache protocol mismatch')
validate_text_cache(text_cache, rows, caption_status['csv_sha256'], text_status['identity'])
text_embeddings = text_cache['embeddings'].float().to(DEVICE)
text_valid = text_cache['valid'].bool().to(DEVICE)
if not torch.isfinite(text_embeddings).all() or torch.count_nonzero(text_embeddings[~text_valid]):
    raise ValueError('Invalid/nonfinite text embeddings; rejected captions must have zero vectors')
TEXT_DIM = text_embeddings.shape[1]
TEXT_POSITION = {image_id: pos for pos, image_id in enumerate(text_cache['image_ids'])}
caption_by_id = {row['image_id']: row['final_caption'] for row in rows}
candidate_text_positions = torch.tensor([TEXT_POSITION[str(x)] for x in candidate_df.image_id], device=DEVICE)
candidate_text_bank = text_embeddings[candidate_text_positions]
candidate_text_valid = text_valid[candidate_text_positions]

def query_text_inputs(image_ids):
    positions = torch.tensor([TEXT_POSITION[str(image_id)] for image_id in image_ids], device=DEVICE)
    return text_embeddings[positions], text_valid[positions]

coverage = []
for role, frame in (('train', train_df), ('val', val_df), ('test', test_df), ('candidate_pool', candidate_df)):
    positions = torch.tensor([TEXT_POSITION[str(x)] for x in frame.image_id], device=DEVICE)
    coverage.append({'role': role, 'images': len(frame), 'usable_captions': int(text_valid[positions].sum()),
                     'usable_rate': float(text_valid[positions].float().mean())})
pd.DataFrame(coverage).to_csv(OUT_PUT_DIR / 'caption_coverage.csv', index=False)
display(pd.DataFrame(coverage))
print('Caption CSV SHA256:', caption_status['csv_sha256'])
print('CLIP text vectors:', tuple(text_embeddings.shape), '| policy:', POLICY)
del text_cache
'''


PROTOCOL_CELL = r'''
import importlib.metadata
import gc

split_identity = []
for role, frame in (('train', train_df), ('val', val_df), ('test', test_df)):
    for row in frame.itertuples(index=False):
        split_identity.append({'image_id': str(row.image_id), 'split': role,
                               'image_sha256': file_sha256(row.image_path), 'mask_sha256': file_sha256(row.mask_path)})
protocol = {
    'schema': 1, 'experiment': 'isic2017_qwen8b_full_both_448', 'seed': SEED,
    'split_identity': split_identity, 'candidate_ids': candidate_df.image_id.astype(str).tolist(),
    'caption_csv_sha256': caption_status['csv_sha256'], 'text_cache_identity': text_status['identity'],
    'dino_weights_sha256': file_sha256(resolve_dinov3_weights()), 'code_sha256': TRAINING_CODE_SHA256,
    'configuration': {key: globals()[key] for key in (
        'IMAGE_SIZE', 'BATCH_SIZE', 'NUM_WORKERS', 'EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY',
        'WARMUP_EPOCHS', 'PATIENCE', 'GRAD_CLIP_NORM', 'CANDIDATE_POOL_FRACTION', 'TACS_GUMBEL_TAU',
        'TACS_POLICY_WEIGHT', 'TACS_ENTROPY_WEIGHT', 'TACS_SELECTOR_DROPOUT', 'TACS_SELECTOR_TEMPERATURE',
        'NUM_CROSS_ATTENTION_LAYERS', 'CROSS_ATTENTION_HEADS', 'DECODER_DROPOUT', 'DICE_LOSS_WEIGHT',
        'BCE_LOSS_WEIGHT', 'THRESHOLD')},
    'runtime': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')},
    'fusion': 'query_text_to_query_tokens_and_candidate_text_to_context_tokens_v1',
    'selector': 'original_visual_score_plus_sigmoid_mix_times_valid_clip_text_cosine',
    'checkpoint_selection': 'maximum_validation_hard_dice',
}
protocol_path = OUT_PUT_DIR / 'training_protocol' / 'protocol.json'
if not protocol_path.exists() and any(p.exists() for p in (best_checkpoint_path, last_checkpoint_path)):
    raise ValueError('Unregistered checkpoint in output directory; use a new output root')
PROTOCOL_ID = lock_protocol(protocol_path, protocol)
for role, frame in (('train', train_df), ('val', val_df), ('test', test_df), ('candidate_pool', candidate_df)):
    atomic_csv(OUT_PUT_DIR / f'{role}_split.csv', frame.to_dict('records'), list(frame.columns))
print('Locked training protocol:', PROTOCOL_ID)

def atomic_checkpoint(path, value):
    temporary = Path(path).with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def move_batch_to_device(batch):
    return {'images': batch['image'].to(DEVICE, non_blocking=True),
            'masks': batch['mask'].to(DEVICE, non_blocking=True),
            'fname': batch['fname'], 'image_id': batch['image_id'],
            'row_id': batch['row_id'].to(DEVICE, non_blocking=True)}

def full_both_selection(query_global, q_text, q_valid, forbidden=None, training=False):
    return model.select_full_both(query_global, candidate_global_bank, q_text, q_valid,
        candidate_text_bank, candidate_text_valid, forbidden_indices=forbidden,
        sample=training, tau=TACS_GUMBEL_TAU)

@torch.no_grad()
def evaluate_full_both(loader, desc='val', return_rows=False):
    model.eval()
    totals = {key: 0.0 for key in ('loss', 'dice', 'iou', 'soft_dice', 'soft_iou', 'q_gate', 'c_gate',
                                   'query_caption_rate', 'candidate_caption_rate')}
    count, predictions = 0, []
    for batch in tqdm(loader, desc=desc, leave=False):
        moved = move_batch_to_device(batch)
        q_text, q_valid = query_text_inputs(moved['image_id'])
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            q_tokens, q_global = model.encode_images(moved['images'])
            details = full_both_selection(q_global, q_text, q_valid)
            indices = details['selected_indices']
            context = gather_candidate_tokens(indices)
            logits, q_gate, c_gate = model.decode_full_both(q_tokens, context, q_text, q_valid,
                candidate_text_bank[indices], candidate_text_valid[indices])
            losses = segmentation_loss_per_sample(logits, moved['masks'])
        dice, iou = dice_iou_from_logits(logits.float(), moved['masks'], threshold=THRESHOLD)
        soft_dice, soft_iou = soft_dice_iou_from_logits(logits.float(), moved['masks'])
        values = {'loss': losses, 'dice': dice, 'iou': iou, 'soft_dice': soft_dice, 'soft_iou': soft_iou,
                  'q_gate': q_gate, 'c_gate': c_gate, 'query_caption_rate': q_valid.float(),
                  'candidate_caption_rate': candidate_text_valid[indices].float()}
        for key, value in values.items():
            totals[key] += value.float().sum().item()
        count += len(dice)
        if return_rows:
            for j, image_id in enumerate(moved['image_id']):
                pos = int(indices[j])
                context_id = str(candidate_df.iloc[pos].image_id)
                predictions.append({'image_id': str(image_id), 'fname': moved['fname'][j],
                    'context_pool_position': pos, 'context_image_id': context_id,
                    'query_caption': caption_by_id[str(image_id)], 'query_caption_valid': bool(q_valid[j]),
                    'candidate_caption': caption_by_id[context_id], 'candidate_caption_valid': bool(candidate_text_valid[pos]),
                    **{key: float(value[j]) for key, value in values.items()}})
    if count != len(loader.dataset):
        raise ValueError('Evaluation did not cover every requested image')
    return {key: value / count for key, value in totals.items()}, pd.DataFrame(predictions)
'''


TRAIN_CELL = r'''
def train_full_both_epoch(epoch):
    model.train()
    model.feature_extractor.eval()
    totals = {key: 0.0 for key in ('loss', 'dice', 'iou', 'reward', 'policy_loss', 'q_gate', 'c_gate')}
    count = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch:03d} [train]', leave=False):
        moved = move_batch_to_device(batch)
        q_text, q_valid = query_text_inputs(moved['image_id'])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            q_tokens, q_global = model.encode_images(moved['images'])
            forbidden = forbidden_pool_indices_from_row_ids(moved['row_id'])
            details = full_both_selection(q_global, q_text, q_valid, forbidden, training=True)
            context = select_candidate_tokens_from_weights(details['selection_weights'])
            c_text, c_valid = text_for_indices(details['selection_weights'], candidate_text_bank, candidate_text_valid)
            logits, q_gate, c_gate = model.decode_full_both(q_tokens, context, q_text, q_valid, c_text, c_valid)
            grad_loss = segmentation_loss_per_sample(logits, moved['masks']).mean()
            policy = torch.distributions.Categorical(logits=details['selector_scores'].float())
            actions = policy.sample()
            with torch.no_grad():
                # Keep the query caption in BOTH reward terms; only the retrieved image/text is added.
                no_context = model.query_only_with_text(q_tokens, q_text, q_valid)
                policy_logits, _, _ = model.decode_full_both(q_tokens, gather_candidate_tokens(actions),
                    q_text, q_valid, candidate_text_bank[actions], candidate_text_valid[actions])
                rewards = (segmentation_loss_per_sample(no_context.float(), moved['masks']) -
                           segmentation_loss_per_sample(policy_logits.float(), moved['masks'])).detach()
                advantages = ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)).detach()
            policy_loss = -(policy.log_prob(actions) * advantages).mean()
            loss = grad_loss + TACS_POLICY_WEIGHT * policy_loss - TACS_ENTROPY_WEIGHT * policy.entropy().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        dice, iou = dice_iou_from_logits(logits.detach().float(), moved['masks'], threshold=THRESHOLD)
        size = len(dice)
        for key, value in {'loss': loss, 'policy_loss': policy_loss, 'reward': rewards.mean(),
                           'dice': dice.mean(), 'iou': iou.mean(), 'q_gate': q_gate.mean(), 'c_gate': c_gate.mean()}.items():
            totals[key] += float(value.detach()) * size
        count += size
    return {key: value / count for key, value in totals.items()}

def checkpoint_state(epoch, history, best_dice, best_epoch, bad_epochs):
    return {'protocol_id': PROTOCOL_ID, 'epoch': epoch, 'history': history, 'best_dice': best_dice,
            'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
            'rng_loader': train_generator.get_state()}

history, best_val_dice, best_epoch, bad_epochs, start_epoch = [], -1.0, 0, 0, 1
if last_checkpoint_path.is_file():
    if not RESUME_TRAINING:
        raise ValueError('Existing run found; enable RESUME_TRAINING or use a new output root')
    state = torch.load(last_checkpoint_path, map_location='cpu', weights_only=False)
    if state['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Checkpoint protocol does not match this run')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    history, best_val_dice, best_epoch, bad_epochs = state['history'], state['best_dice'], state['best_epoch'], state['bad_epochs']
    start_epoch = state['epoch'] + 1
    torch.set_rng_state(state['rng_torch'])
    torch.cuda.set_rng_state_all(state['rng_cuda'])
    np.random.set_state(state['rng_numpy'])
    random.setstate(state['rng_python'])
    train_generator.set_state(state['rng_loader'])
    del state
    print('Resuming at epoch', start_epoch, '| best epoch', best_epoch)
completion_path = OUT_PUT_DIR / 'completed.json'
if not completion_path.is_file() and bad_epochs < PATIENCE:
    for epoch in range(start_epoch, EPOCHS + 1):
        train_metrics = train_full_both_epoch(epoch)
        val_metrics, _ = evaluate_full_both(val_loader, desc=f'Epoch {epoch:03d} [val]')
        scheduler.step()
        history.append({'epoch': epoch, **{'train_' + k: v for k, v in train_metrics.items()},
                        **{'val_' + k: v for k, v in val_metrics.items()}, 'lr': optimizer.param_groups[0]['lr']})
        improved = val_metrics['dice'] > best_val_dice
        if improved:
            best_val_dice, best_epoch, bad_epochs = val_metrics['dice'], epoch, 0
        else:
            bad_epochs += 1
        state = checkpoint_state(epoch, history, best_val_dice, best_epoch, bad_epochs)
        if improved:
            atomic_checkpoint(best_checkpoint_path, {'protocol_id': PROTOCOL_ID, 'epoch': epoch,
                'model_state_dict': model.state_dict(), 'val_metrics': val_metrics})
        atomic_checkpoint(last_checkpoint_path, state)
        del state
        pd.DataFrame(history).to_csv(OUT_PUT_DIR / 'training_history.csv', index=False)
        print(f"full_both epoch={epoch:03d} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
              f"q_gate={val_metrics['q_gate']:.4f} c_gate={val_metrics['c_gate']:.4f} "
              f"semantic_mix={float(torch.sigmoid(model.semantic_mix_logit).detach()):.4f} best={best_epoch:03d} bad={bad_epochs}/{PATIENCE}", flush=True)
        if bad_epochs >= PATIENCE:
            print('Early stopping at epoch', epoch)
            break
if not history or (history[-1]['epoch'] < EPOCHS and bad_epochs < PATIENCE):
    raise RuntimeError('Training is incomplete; resume before final test')
atomic_json(completion_path, {'protocol_id': PROTOCOL_ID, 'epochs_observed': int(history[-1]['epoch']),
    'epoch_limit': EPOCHS, 'best_epoch': best_epoch, 'best_val_dice': best_val_dice,
    'stopping': 'early_stopping' if bad_epochs >= PATIENCE else 'epoch_limit'})
display(pd.DataFrame(history).tail())
'''


TEST_CELL = r'''
completed = json.loads((OUT_PUT_DIR / 'completed.json').read_text(encoding='utf-8'))
if completed['protocol_id'] != PROTOCOL_ID:
    raise ValueError('Wrong completed run')
metrics_path = OUT_PUT_DIR / 'metrics.json'
if metrics_path.is_file():
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics['protocol_id'] != PROTOCOL_ID or metrics['best_epoch'] != completed['best_epoch']:
        raise ValueError('Stale test result; use a separate experiment root')
    print('Reusing locked test result:', metrics_path)
else:
    state = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
    if state['protocol_id'] != PROTOCOL_ID or state['epoch'] != completed['best_epoch']:
        raise ValueError('Wrong best checkpoint')
    model.load_state_dict(state['model_state_dict'])
    del state
    # No test masks or scores were used by caption generation, retrieval, training or checkpoint choice.
    test_metrics, test_rows = evaluate_full_both(test_loader, desc='test full_both', return_rows=True)
    test_rows.to_csv(OUT_PUT_DIR / 'test_predictions.csv', index=False)
    metrics = {'protocol_id': PROTOCOL_ID, 'method': 'ISIC Qwen3-VL-8B full_both 448',
        'caption_csv_sha256': caption_status['csv_sha256'], 'seed': SEED, 'image_size': IMAGE_SIZE,
        'train_size': len(train_df), 'val_size': len(val_df), 'test_size': len(test_df),
        **completed, **{'test_' + key: value for key, value in test_metrics.items()},
        'selector_semantic_mix': float(torch.sigmoid(model.semantic_mix_logit).detach())}
    atomic_json(metrics_path, metrics)
print(f"test_dice: {metrics['test_dice']:.6f} ({metrics['test_dice'] * 100:.2f}%)")
print(f"test_iou: {metrics['test_iou']:.6f} ({metrics['test_iou'] * 100:.2f}%)")
comparison = pd.DataFrame([
    {'method': 'Visual TACS variant 448, historical seed42', 'test_dice': BASELINE_DICE,
     'test_iou': BASELINE_IOU, 'new_run': False},
    {'method': 'Qwen3-VL-8B full_both 448', 'test_dice': metrics['test_dice'],
     'test_iou': metrics['test_iou'], 'new_run': True},
])
comparison['dice_delta_percentage_points'] = (comparison.test_dice - BASELINE_DICE) * 100
comparison['iou_delta_percentage_points'] = (comparison.test_iou - BASELINE_IOU) * 100
comparison.to_csv(OUT_PUT_DIR / 'segmentation_comparison.csv', index=False)
display(comparison)
print('Historical reference is rounded stdout, not a newly rerun baseline or a three-seed mean.')

history_df = pd.read_csv(OUT_PUT_DIR / 'training_history.csv')
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for key, ax in zip(('dice', 'iou', 'loss'), axes):
    ax.plot(history_df.epoch, history_df['train_' + key], label='train')
    ax.plot(history_df.epoch, history_df['val_' + key], label='val')
    ax.axvline(completed['best_epoch'], color='gray', linestyle='--')
    ax.set_title(key)
    ax.set_xlabel('Epoch')
    ax.legend()
fig.tight_layout()
fig.savefig(OUT_PUT_DIR / 'learning_curves.png', dpi=160)
plt.show()
'''
