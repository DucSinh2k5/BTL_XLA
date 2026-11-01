from pathlib import Path
import gc
import torch


def resolve_dinov3_weights():
    path = Path(DINOV3_WEIGHTS)
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy file weight: {path}')
    return str(path)


def load_dinov3_model(repo_dir, model_name, weights_path):
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục mã nguồn: {repo_dir}')
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy file weight: {weights_path}')
    return torch.hub.load(str(repo_dir), model_name, source='local', weights=str(weights_path))


def load_dinov3_backbone():
    return load_dinov3_model(DINOV3_REPO_DIR, DINOV3_MODEL_NAME, resolve_dinov3_weights())


def ensure_dinov3_assets():
    resolve_dinov3_weights()
    repo_dir = Path(DINOV3_REPO_DIR)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục mã nguồn: {repo_dir}')
    print('DINOv3 assets đã sẵn sàng từ Kaggle Input.')


def extract_cls_features(backbone, images):
    features = backbone(images)
    if isinstance(features, dict):
        if 'x_norm_clstoken' in features:
            return features['x_norm_clstoken']
        if 'x_norm_cls_token' in features:
            return features['x_norm_cls_token']
        if 'x_norm_patchtokens' in features:
            return features['x_norm_patchtokens'].mean(dim=1)
        raise KeyError(f'Khong tim thay CLS token trong output keys: {list(features)}')
    return features[:, 0] if torch.is_tensor(features) and features.ndim == 3 else features


def extract_frozen_dino_embeddings(annotations, backbone, desc):
    dataset = SimpleImageDataset(annotations, eval_transform, crop_bbox=USE_BBOX_CROP)
    loader = DataLoader(dataset, batch_size=RETRIEVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    embeddings = []
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        for images, *_ in tqdm(loader, desc=desc, leave=False):
            images = images.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                features = extract_cls_features(backbone, images)
            features = F.normalize(features.float(), dim=1)
            embeddings.append(features.cpu())
    return torch.cat(embeddings, dim=0)


def retrieve_topk_contexts(query_embeddings, candidate_embeddings, query_df, candidate_df, top_k=4, exclude_self=False, chunk_size=1024):
    candidate_embeddings = candidate_embeddings.float()
    top_k = int(max(1, min(top_k, len(candidate_df))))
    selected_positions = []
    selected_scores = []
    candidate_row_id_to_position = {int(row_id): pos for pos, row_id in enumerate(candidate_df['row_id'].tolist())}
    for start in tqdm(range(0, len(query_embeddings), chunk_size), desc=f'retrieve top-{top_k} contexts', leave=False):
        end = min(start + chunk_size, len(query_embeddings))
        similarities = query_embeddings[start:end].float() @ candidate_embeddings.T
        if exclude_self:
            for local_index, query_index in enumerate(range(start, end)):
                query_row_id = int(query_df.iloc[query_index]['row_id'])
                context_pos = candidate_row_id_to_position.get(query_row_id)
                if context_pos is not None and len(candidate_df) > 1:
                    similarities[local_index, context_pos] = -float('inf')
        scores, positions = torch.topk(similarities, k=top_k, dim=1)
        selected_positions.append(positions.cpu())
        selected_scores.append(scores.cpu())
    return torch.cat(selected_positions, dim=0).numpy(), torch.cat(selected_scores, dim=0).numpy()


def build_similarity_pairs_df(query_df, candidate_df, context_positions, context_scores):
    context_positions = np.asarray(context_positions)
    context_scores = np.asarray(context_scores)
    if context_positions.ndim == 1:
        context_positions = context_positions[:, None]
        context_scores = context_scores[:, None]
    rows = []
    for query_pos, (positions_row, scores_row) in enumerate(zip(context_positions, context_scores)):
        query = query_df.iloc[query_pos]
        contexts = [candidate_df.iloc[int(pos)] for pos in positions_row]
        context_class_ids = [int(context['class_id']) for context in contexts]
        query_class_id = int(query['class_id'])
        rows.append({
            'query_fname': query['fname'], 'query_split': query['split'], 'query_class_id': query_class_id, 'query_car_name': class_names[query_class_id],
            'top1_context_fname': contexts[0]['fname'], 'top1_context_class_id': context_class_ids[0], 'top1_context_car_name': class_names[context_class_ids[0]],
            'top1_context_pool_position': int(positions_row[0]), 'top1_context_train_row_id': int(contexts[0]['row_id']), 'top1_dino_similarity': float(scores_row[0]),
            'context_fnames': '|'.join(str(context['fname']) for context in contexts), 'context_class_ids': '|'.join(str(class_id) for class_id in context_class_ids),
            'context_pool_positions': '|'.join(str(int(pos)) for pos in positions_row), 'dino_similarities': '|'.join(f'{float(score):.6f}' for score in scores_row),
            'same_class': bool(query_class_id == context_class_ids[0]), 'same_class_top1': bool(query_class_id == context_class_ids[0]), 'same_class_any_topk': bool(query_class_id in context_class_ids),
        })
    return pd.DataFrame(rows)


class DinoV3FrozenSimilarityContextClassifier(nn.Module):
    def __init__(self, repo_dir, model_name, weights_path, num_classes, feature_dim=384, allow_feature_fallback=False):
        super().__init__()
        self.backbone = load_dinov3_model(repo_dir, model_name, weights_path)
        self.feature_dim = feature_dim
        self.allow_feature_fallback = allow_feature_fallback
        self.active_fusion_mode = 'topk_token_attention'
        self.context_proj = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.context_attention = nn.Linear(feature_dim, 1)
        self.fusion_norm = nn.LayerNorm(feature_dim)
        self.token_head = nn.Sequential(nn.Dropout(p=TOKEN_HEAD_DROPOUT), nn.Linear(feature_dim, num_classes))
        self.feature_fusion = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.GELU(), nn.Dropout(p=TOKEN_HEAD_DROPOUT), nn.Linear(feature_dim, num_classes))
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def _prefix_token_count(self):
        if hasattr(self.backbone, 'num_prefix_tokens'):
            return int(self.backbone.num_prefix_tokens)
        storage_tokens = getattr(self.backbone, 'n_storage_tokens', None)
        if storage_tokens is None:
            storage_tokens = getattr(self.backbone, 'num_register_tokens', 0)
        return 1 + int(storage_tokens or 0)

    def _patch_grid(self, images):
        patch_size = getattr(self.backbone, 'patch_size', 16)
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size)
        return images.shape[-2] // patch_h, images.shape[-1] // patch_w

    def _unwrap_tokens(self, prepared):
        if torch.is_tensor(prepared):
            return prepared
        if isinstance(prepared, (tuple, list)):
            for item in prepared:
                if torch.is_tensor(item):
                    return item
        if isinstance(prepared, dict):
            for key in ('x', 'tokens', 'last_hidden_state'):
                if key in prepared and torch.is_tensor(prepared[key]):
                    return prepared[key]
            for item in prepared.values():
                if torch.is_tensor(item):
                    return item
        raise TypeError(f'Cannot unwrap DINO token output of type {type(prepared)}')

    def _prepare_tokens_with_rope(self, images):
        try:
            prepared = self.backbone.prepare_tokens_with_masks(images, None)
        except TypeError:
            prepared = self.backbone.prepare_tokens_with_masks(images)
        tokens = self._unwrap_tokens(prepared)
        rope = None
        if hasattr(self.backbone, 'rope_embed'):
            grid_h, grid_w = self._patch_grid(images)
            try:
                rope = self.backbone.rope_embed(H=grid_h, W=grid_w)
            except TypeError:
                rope = self.backbone.rope_embed(grid_h, grid_w)
        return tokens, rope

    def _repeat_rope_for_query_context(self, rope):
        if rope is None or not torch.is_tensor(rope):
            return rope
        try:
            return rope.repeat_interleave(2, dim=-2)
        except RuntimeError:
            return rope

    def _run_blocks(self, tokens, rope):
        for block in self.backbone.blocks:
            if rope is None:
                tokens = block(tokens)
                continue
            try:
                tokens = block(tokens, rope=rope)
            except TypeError:
                try:
                    tokens = block(tokens, rope)
                except TypeError:
                    tokens = block(tokens)
        if hasattr(self.backbone, 'norm'):
            tokens = self.backbone.norm(tokens)
        return tokens

    def _forward_token_concat_features(self, query_images, context_images):
        query_tokens, query_rope = self._prepare_tokens_with_rope(query_images)
        context_tokens, _ = self._prepare_tokens_with_rope(context_images)
        prefix_tokens = self._prefix_token_count()
        pair_tokens = torch.cat([query_tokens[:, :prefix_tokens], query_tokens[:, prefix_tokens:], context_tokens[:, prefix_tokens:]], dim=1)
        pair_rope = self._repeat_rope_for_query_context(query_rope)
        pair_tokens = self._run_blocks(pair_tokens, pair_rope)
        return pair_tokens[:, 0]

    def _pool_backbone_features(self, images):
        features = self.backbone.forward_features(images)
        if isinstance(features, dict):
            if 'x_norm_clstoken' in features:
                return features['x_norm_clstoken']
            if 'x_norm_patchtokens' in features:
                return features['x_norm_patchtokens'].mean(dim=1)
            for item in features.values():
                if torch.is_tensor(item):
                    return item[:, 0] if item.ndim == 3 else item
        if torch.is_tensor(features):
            return features[:, 0] if features.ndim == 3 else features
        raise TypeError(f'Cannot pool DINO features of type {type(features)}')

    def _forward_feature_fusion(self, query_images, context_images):
        if context_images.ndim == 5:
            context_images = context_images[:, 0]
        query_features = self._pool_backbone_features(query_images)
        context_features = self._pool_backbone_features(context_images)
        self.active_fusion_mode = 'feature_fusion'
        return self.feature_fusion(torch.cat([query_features, context_features], dim=1))

    def _forward_topk_token_attention(self, query_images, context_images, context_scores=None):
        if context_images.ndim == 4:
            context_images = context_images.unsqueeze(1)
        batch_size, num_contexts, channels, height, width = context_images.shape
        query_repeated = query_images.unsqueeze(1).expand(-1, num_contexts, -1, -1, -1).reshape(batch_size * num_contexts, channels, height, width)
        context_flat = context_images.reshape(batch_size * num_contexts, channels, height, width)
        pair_features = self._forward_token_concat_features(query_repeated, context_flat).reshape(batch_size, num_contexts, self.feature_dim)
        query_summary = pair_features.mean(dim=1)
        attention_hidden = torch.tanh(self.context_proj(pair_features) + self.query_proj(query_summary).unsqueeze(1))
        attention_logits = self.context_attention(attention_hidden).squeeze(-1)
        if context_scores is not None:
            if context_scores.ndim == 1:
                context_scores = context_scores.unsqueeze(1)
            attention_logits = attention_logits + RETRIEVAL_SCORE_WEIGHT * context_scores.to(attention_logits.device, dtype=attention_logits.dtype)
        attention_weights = torch.softmax(attention_logits, dim=1)
        fused_features = torch.sum(pair_features * attention_weights.unsqueeze(-1), dim=1)
        fused_features = self.fusion_norm(fused_features)
        self.active_fusion_mode = 'topk_token_attention'
        return fused_features, attention_weights

    def forward(self, query_images, context_images, context_scores=None, return_features=False):
        try:
            fused_features, attention_weights = self._forward_topk_token_attention(query_images, context_images, context_scores=context_scores)
            logits = self.token_head(fused_features)
            if return_features:
                return logits, F.normalize(fused_features.float(), dim=1), attention_weights
            return logits
        except Exception as exc:
            if self.allow_feature_fallback:
                print(f'Top-k token attention failed, fallback to feature fusion: {type(exc).__name__}: {exc}')
                logits = self._forward_feature_fusion(query_images, context_images)
                if return_features:
                    return logits, F.normalize(logits.float(), dim=1), None
                return logits
            raise RuntimeError('Top-k token attention failed. Keep allow_feature_fallback=False for debugging.') from exc


ensure_dinov3_assets()
retriever_backbone = load_dinov3_backbone().to(DEVICE).eval()
candidate_embeddings = extract_frozen_dino_embeddings(candidate_df, retriever_backbone, 'candidate embeddings')
train_query_embeddings = extract_frozen_dino_embeddings(train_df, retriever_backbone, 'train query embeddings')
val_query_embeddings = extract_frozen_dino_embeddings(val_df, retriever_backbone, 'val query embeddings')
test_query_embeddings = extract_frozen_dino_embeddings(test_df, retriever_backbone, 'test query embeddings')
del retriever_backbone
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
train_context_positions, train_context_scores = retrieve_topk_contexts(train_query_embeddings, candidate_embeddings, train_df, candidate_df, top_k=CONTEXT_TOP_K, exclude_self=True, chunk_size=RETRIEVAL_CHUNK_SIZE)
val_context_positions, val_context_scores = retrieve_topk_contexts(val_query_embeddings, candidate_embeddings, val_df, candidate_df, top_k=CONTEXT_TOP_K, exclude_self=False, chunk_size=RETRIEVAL_CHUNK_SIZE)
test_context_positions, test_context_scores = retrieve_topk_contexts(test_query_embeddings, candidate_embeddings, test_df, candidate_df, top_k=CONTEXT_TOP_K, exclude_self=False, chunk_size=RETRIEVAL_CHUNK_SIZE)
train_similarity_pairs_df = build_similarity_pairs_df(train_df, candidate_df, train_context_positions, train_context_scores)
val_similarity_pairs_df = build_similarity_pairs_df(val_df, candidate_df, val_context_positions, val_context_scores)
test_similarity_pairs_df = build_similarity_pairs_df(test_df, candidate_df, test_context_positions, test_context_scores)
train_similarity_pairs_df.to_csv(OUT_PUT_DIR / 'train_frozen_dino_topk_similarity_pairs.csv', index=False)
val_similarity_pairs_df.to_csv(OUT_PUT_DIR / 'val_frozen_dino_topk_similarity_pairs.csv', index=False)
test_similarity_pairs_df.to_csv(OUT_PUT_DIR / 'test_frozen_dino_topk_similarity_pairs.csv', index=False)
train_top1_same_class_context_rate = float(train_similarity_pairs_df['same_class_top1'].mean())
val_top1_same_class_context_rate = float(val_similarity_pairs_df['same_class_top1'].mean())
test_top1_same_class_context_rate = float(test_similarity_pairs_df['same_class_top1'].mean())
train_same_class_context_rate = float(train_similarity_pairs_df['same_class_any_topk'].mean())
val_same_class_context_rate = float(val_similarity_pairs_df['same_class_any_topk'].mean())
test_same_class_context_rate = float(test_similarity_pairs_df['same_class_any_topk'].mean())
torch.save({'candidate_embeddings': candidate_embeddings, 'train_query_embeddings': train_query_embeddings, 'val_query_embeddings': val_query_embeddings, 'test_query_embeddings': test_query_embeddings, 'candidate_pool_positions': torch.as_tensor(candidate_pool_positions, dtype=torch.long), 'train_context_positions': torch.as_tensor(train_context_positions, dtype=torch.long), 'train_context_scores': torch.as_tensor(train_context_scores, dtype=torch.float32), 'val_context_positions': torch.as_tensor(val_context_positions, dtype=torch.long), 'val_context_scores': torch.as_tensor(val_context_scores, dtype=torch.float32), 'test_context_positions': torch.as_tensor(test_context_positions, dtype=torch.long), 'test_context_scores': torch.as_tensor(test_context_scores, dtype=torch.float32), 'seed': SEED, 'candidate_pool_fraction': CANDIDATE_POOL_FRACTION, 'context_top_k': CONTEXT_TOP_K, 'use_bbox_crop': USE_BBOX_CROP}, OUT_PUT_DIR / 'frozen_dino_topk_retrieval_cache.pt')
train_dataset = StanfordCarsSimilarityContextDataset(train_df, candidate_df, train_context_positions, train_context_scores, train_transform, train_transform, crop_bbox=USE_BBOX_CROP)
val_dataset = StanfordCarsSimilarityContextDataset(val_df, candidate_df, val_context_positions, val_context_scores, eval_transform, eval_transform, crop_bbox=USE_BBOX_CROP)
test_dataset = StanfordCarsSimilarityContextDataset(test_df, candidate_df, test_context_positions, test_context_scores, eval_transform, eval_transform, crop_bbox=USE_BBOX_CROP)
loader_kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': torch.cuda.is_available()}
if NUM_WORKERS > 0:
    loader_kwargs['persistent_workers'] = True
    loader_kwargs['prefetch_factor'] = 2
train_generator = torch.Generator()
train_generator.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator, **loader_kwargs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
model = DinoV3FrozenSimilarityContextClassifier(DINOV3_REPO_DIR, DINOV3_MODEL_NAME, resolve_dinov3_weights(), NUM_CLASSES, FEATURE_DIM, allow_feature_fallback=False).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
backbone_params = [p for name, p in model.named_parameters() if name.startswith('backbone.')]
head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': BACKBONE_LR, 'weight_decay': BACKBONE_WEIGHT_DECAY, 'name': 'backbone'}, {'params': head_params, 'lr': HEAD_LR, 'weight_decay': HEAD_WEIGHT_DECAY, 'name': 'head'}])
warmup_epochs = max(1, min(WARMUP_EPOCHS, EPOCHS))
main_epochs = max(1, EPOCHS - warmup_epochs)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-7)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
checkpoint_path = OUT_PUT_DIR / 'final_topk_context_attention_dinov3_vits16_cars.pt'
last_checkpoint_path = OUT_PUT_DIR / 'checkpoint_last.pt'
best_checkpoint_path = OUT_PUT_DIR / 'best_val_checkpoint.pt'
sample = train_dataset[0]
print('Saved retrieval cache:', (OUT_PUT_DIR / 'frozen_dino_topk_retrieval_cache.pt').resolve())
print('Saved train top-k similarity pairs:', (OUT_PUT_DIR / 'train_frozen_dino_topk_similarity_pairs.csv').resolve())
print('Saved val top-k similarity pairs:', (OUT_PUT_DIR / 'val_frozen_dino_topk_similarity_pairs.csv').resolve())
print('Saved test top-k similarity pairs:', (OUT_PUT_DIR / 'test_frozen_dino_topk_similarity_pairs.csv').resolve())
print('Sample tensors:', sample['query_image'].shape, sample['context_images'].shape)
print('Query contexts:', sample['fname'], '->', sample['context_fnames'])
print('Context scores:', sample['context_scores'].tolist())
print(f'Train top1/any-top{CONTEXT_TOP_K} same-class context rate: {train_top1_same_class_context_rate:.4f} / {train_same_class_context_rate:.4f}')
print(f'Val top1/any-top{CONTEXT_TOP_K} same-class context rate: {val_top1_same_class_context_rate:.4f} / {val_same_class_context_rate:.4f}')
print(f'Test top1/any-top{CONTEXT_TOP_K} same-class context rate: {test_top1_same_class_context_rate:.4f} / {test_same_class_context_rate:.4f}')
print('Initial fusion mode:', model.active_fusion_mode)
display(test_similarity_pairs_df.head())
print(model)
