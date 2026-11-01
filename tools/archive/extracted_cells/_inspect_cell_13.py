from sklearn.metrics import f1_score


def macro_f1_from_predictions(labels, predictions):
    return float(f1_score(labels, predictions, average='macro', zero_division=0))


def supervised_contrastive_loss(features, labels, temperature=0.10):
    features = F.normalize(features.float(), dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    positives_mask = mask * logits_mask
    positive_counts = positives_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


def set_backbone_trainable(network, trainable):
    for name, param in network.named_parameters():
        if name.startswith('backbone.'):
            param.requires_grad_(trainable)
    if hasattr(network, 'backbone'):
        network.backbone.train(mode=trainable)


def current_lrs(opt):
    return {group.get('name', f'group_{index}'): group['lr'] for index, group in enumerate(opt.param_groups)}


def move_batch_to_device(batch):
    return {'query_images': batch['query_image'].to(DEVICE, non_blocking=True), 'context_images': batch['context_images'].to(DEVICE, non_blocking=True), 'context_scores': batch['context_scores'].to(DEVICE, non_blocking=True), 'labels': batch['label'].to(DEVICE, non_blocking=True)}


def train_one_epoch(network, loader, epoch, backbone_trainable=True):
    network.train(); set_backbone_trainable(network, backbone_trainable)
    running_loss = running_ce_loss = running_supcon_loss = 0.0
    running_correct = running_total = 0
    all_predictions, all_labels = [], []
    progress = tqdm(loader, desc=f'Epoch {epoch:03d} [train]', leave=False)
    for batch in progress:
        moved = move_batch_to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
            logits, features, _attention = network(moved['query_images'], moved['context_images'], context_scores=moved['context_scores'], return_features=True)
            ce_loss = criterion(logits, moved['labels'])
            supcon_loss = supervised_contrastive_loss(features, moved['labels'], temperature=SUPCON_TEMPERATURE)
            loss = ce_loss + SUPCON_WEIGHT * supcon_loss
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer); scaler.update()
        predictions = logits.argmax(dim=1); batch_size = moved['labels'].size(0)
        running_loss += loss.item() * batch_size; running_ce_loss += ce_loss.item() * batch_size; running_supcon_loss += supcon_loss.item() * batch_size
        running_correct += (predictions == moved['labels']).sum().item(); running_total += batch_size
        all_predictions.extend(predictions.detach().cpu().tolist()); all_labels.extend(moved['labels'].detach().cpu().tolist())
        progress.set_postfix(loss=running_loss / max(running_total, 1), acc=running_correct / max(running_total, 1))
    return {'loss': running_loss / max(running_total, 1), 'ce_loss': running_ce_loss / max(running_total, 1), 'supcon_loss': running_supcon_loss / max(running_total, 1), 'accuracy': running_correct / max(running_total, 1), 'macro_f1': macro_f1_from_predictions(all_labels, all_predictions)}


@torch.no_grad()
def evaluate_one_epoch(network, loader, desc='val'):
    network.eval(); running_loss = running_correct = running_total = 0; all_predictions, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        moved = move_batch_to_device(batch)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
            logits = network(moved['query_images'], moved['context_images'], context_scores=moved['context_scores']); loss = criterion(logits, moved['labels'])
        predictions = logits.argmax(dim=1); batch_size = moved['labels'].size(0)
        running_loss += loss.item() * batch_size; running_correct += (predictions == moved['labels']).sum().item(); running_total += batch_size
        all_predictions.extend(predictions.detach().cpu().tolist()); all_labels.extend(moved['labels'].detach().cpu().tolist())
    return {'loss': running_loss / max(running_total, 1), 'accuracy': running_correct / max(running_total, 1), 'macro_f1': macro_f1_from_predictions(all_labels, all_predictions)}


def build_checkpoint_payload(epoch, train_metrics, val_metrics, best_val_loss, best_epoch, history):
    return {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'class_names': class_names, 'metrics': {'train_loss': train_metrics['loss'], 'train_ce_loss': train_metrics.get('ce_loss'), 'train_supcon_loss': train_metrics.get('supcon_loss'), 'train_accuracy': train_metrics['accuracy'], 'train_macro_f1': train_metrics['macro_f1'], 'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'], 'val_macro_f1': val_metrics['macro_f1'], 'best_val_loss': best_val_loss, 'best_epoch': best_epoch}, 'history': history, 'config': {'dataset_root': str(DATASET_ROOT), 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'backbone_lr': BACKBONE_LR, 'head_lr': HEAD_LR, 'backbone_weight_decay': BACKBONE_WEIGHT_DECAY, 'head_weight_decay': HEAD_WEIGHT_DECAY, 'label_smoothing': LABEL_SMOOTHING, 'token_head_dropout': TOKEN_HEAD_DROPOUT, 'freeze_backbone_epochs': FREEZE_BACKBONE_EPOCHS, 'warmup_epochs': WARMUP_EPOCHS, 'patience': PATIENCE, 'grad_clip_norm': GRAD_CLIP_NORM, 'candidate_pool_fraction': CANDIDATE_POOL_FRACTION, 'context_top_k': CONTEXT_TOP_K, 'retrieval_score_weight': RETRIEVAL_SCORE_WEIGHT, 'supcon_weight': SUPCON_WEIGHT, 'supcon_temperature': SUPCON_TEMPERATURE, 'val_fraction': VAL_FRACTION, 'use_bbox_crop': USE_BBOX_CROP, 'image_size': IMAGE_SIZE, 'dino_model': DINOV3_MODEL_NAME, 'context_method': 'full_pool_topk_frozen_dino_similarity_retrieval', 'fusion_mode': getattr(model, 'active_fusion_mode', 'unknown'), 'same_class_context_rate_train_any_topk': train_same_class_context_rate, 'same_class_context_rate_val_any_topk': val_same_class_context_rate, 'same_class_context_rate_train_top1': train_top1_same_class_context_rate, 'same_class_context_rate_val_top1': val_top1_same_class_context_rate}}


RESUME_FROM_LAST_CHECKPOINT = False
history = []; start_epoch = 1; best_val_loss = float('inf'); best_epoch = 0; bad_epochs = 0
if RESUME_FROM_LAST_CHECKPOINT and last_checkpoint_path.is_file():
    checkpoint = torch.load(last_checkpoint_path, map_location=DEVICE); model.load_state_dict(checkpoint['model_state_dict']); optimizer.load_state_dict(checkpoint['optimizer_state_dict']); scheduler.load_state_dict(checkpoint['scheduler_state_dict']); scaler.load_state_dict(checkpoint['scaler_state_dict']); start_epoch = checkpoint['epoch'] + 1; history = checkpoint.get('history', []); metrics = checkpoint.get('metrics', {}); best_val_loss = metrics.get('best_val_loss', min((row.get('val_loss', float('inf')) for row in history), default=float('inf'))); best_epoch = metrics.get('best_epoch', 0); print(f'Resuming from epoch {start_epoch}')
for epoch in range(start_epoch, EPOCHS + 1):
    backbone_trainable = epoch > FREEZE_BACKBONE_EPOCHS
    train_metrics = train_one_epoch(model, train_loader, epoch, backbone_trainable=backbone_trainable)
    val_metrics = evaluate_one_epoch(model, val_loader, desc=f'Epoch {epoch:03d} [val]')
    scheduler.step(); is_best = val_metrics['loss'] < best_val_loss - 1e-6
    if is_best: best_val_loss, best_epoch, bad_epochs = val_metrics['loss'], epoch, 0
    else: bad_epochs += 1
    lr_values = current_lrs(optimizer)
    record = {'epoch': epoch, 'train_loss': train_metrics['loss'], 'train_ce_loss': train_metrics.get('ce_loss'), 'train_supcon_loss': train_metrics.get('supcon_loss'), 'train_accuracy': train_metrics['accuracy'], 'train_macro_f1': train_metrics['macro_f1'], 'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'], 'val_macro_f1': val_metrics['macro_f1'], 'best_val_loss': best_val_loss, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'backbone_trainable': backbone_trainable, 'backbone_lr': lr_values.get('backbone'), 'head_lr': lr_values.get('head'), 'fusion_mode': getattr(model, 'active_fusion_mode', 'unknown')}
    history.append(record); payload = build_checkpoint_payload(epoch, train_metrics, val_metrics, best_val_loss, best_epoch, history); torch.save(payload, last_checkpoint_path)
    if is_best: torch.save(payload, best_checkpoint_path)
    print(f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_metrics['loss']:.4f} ce={train_metrics.get('ce_loss', 0):.4f} supcon={train_metrics.get('supcon_loss', 0):.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['macro_f1']:.4f} | val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f} | best_val={best_val_loss:.4f}@{best_epoch:03d} bad={bad_epochs}/{PATIENCE} | bb_train={backbone_trainable} bb_lr={lr_values.get('backbone'):.2e} head_lr={lr_values.get('head'):.2e} | fusion={getattr(model, 'active_fusion_mode', 'unknown')}")
    if bad_epochs >= PATIENCE:
        print(f'Early stopping at epoch {epoch}; best validation loss was {best_val_loss:.4f} at epoch {best_epoch}.'); break
final_train_metrics = train_metrics; final_val_metrics = val_metrics; final_payload = build_checkpoint_payload(epoch, final_train_metrics, final_val_metrics, best_val_loss, best_epoch, history); torch.save(final_payload, checkpoint_path)
print('Saved final checkpoint to:', checkpoint_path.resolve())
if best_checkpoint_path.is_file(): print('Saved best validation checkpoint to:', best_checkpoint_path.resolve())
history_df = pd.DataFrame(history); display(history_df.tail())
