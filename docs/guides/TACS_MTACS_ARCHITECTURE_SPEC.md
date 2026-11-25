# Đặc tả kiến trúc TACS + M-TACS (để dựng notebook phân loại mới)

> **Dành cho agent:** file này mô tả đầy đủ tư tưởng, kiến trúc, cấu hình và vòng huấn luyện của TACS và M-TACS trong dự án, đủ để viết **một notebook độc lập** cho một bộ dữ liệu phân loại ảnh khác.
> Các giá trị cấu hình lấy từ notebook DTD đã chạy (`notebooks/02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/`) và bản web SUN397 (`src/models/`, `src/training/`).
> Mục 10 là danh sách cell gợi ý; mục 11 là những gì phải đổi khi sang dataset mới.

---

## 1. Tư tưởng thuật toán

- **Bài toán:** phân loại ảnh truy vấn `x_q`, được phép xem thêm **đúng một ảnh ngữ cảnh** `x_c` lấy từ một kho ứng viên trích từ tập train.
- **Ý tưởng TACS:** không chọn ảnh *giống nhất* (kNN), mà học một **bộ chọn** được thưởng khi ảnh nó chọn **làm giảm loss** so với không dùng ngữ cảnh. Bộ chọn và mạng phân loại được train cùng nhau.
- **Khó khăn:** phép chọn là argmax rời rạc, không lấy đạo hàm được. Có hai cách cho bộ chọn học:
  - **REINFORCE:** lấy mẫu chỉ số theo phân phối Categorical, thưởng bằng mức giảm loss. Đây là cách **notebook DTD đang dùng**.
  - **Gumbel-Softmax straight-through:** chọn one-hot cứng ở lượt xuôi, dùng softmax mềm ở lượt ngược. Bài báo gốc và bản web SUN397 dùng **cả hai**. Xem mục 5.3.
- **Ý tưởng M-TACS:** gắn thêm **caption của ảnh ứng viên** (sinh offline bằng VLM, mã hóa bằng CLIP) làm ngữ cảnh văn bản, trộn vào đặc trưng qua một **cổng học được**.
- **Kết quả ablation đã biết (DTD):** chỉ gắn caption cho **ảnh ứng viên** (`candidate_only`) mới không gây hại; caption của ảnh truy vấn làm giảm khoảng 2.5 điểm. Thiết kế mặc định cho notebook mới là `candidate_only` (S01).

---

## 2. Sơ đồ tổng thể

```
                     ┌──────────────── Kho ứng viên P (20% train, cố định) ────────────────┐
                     │  ảnh (fp16, giữ trên CPU/GPU)   đặc trưng CLS (làm mới mỗi epoch)   │
                     │  caption CLIP 512-d + mặt nạ valid (tính sẵn offline)               │
                     └─────────────────────────────────────────────────────────────────────┘
x_q ──► DINOv3 CLS ──► Selector: điểm s_j = scale · cos(CLS_q, CLS_j)   (−∞ cho chính x_q)
                          │  train: a ~ Categorical(softmax(s))   eval: a = argmax(s)
                          ▼
             x_c = P.image[a],  t_c = P.caption_emb[a],  m_c = P.valid[a]
                          │
  ┌───────────────────────┴───────────────────────────────────────────────┐
  │ Visual TACS                                                           │
  │  pair  = ViT([CLS, registers, patch(x_q), patch(x_c)]) → CLS_pair     │
  │  query = ViT(x_q) → CLS_q                                             │
  │  gate_v = σ(MLP([q, p, q·p, |q−p|]));  v = LN(q + gate_v · p)         │
  └───────────────────────┬───────────────────────────────────────────────┘
                          ▼  (M-TACS, candidate_only)
          c = Proj(t_c);  g = σ(MLP([v, c])) · m_c;  z = LN(v + g · c) ──► Linear ──► logits
```

---

## 3. Dữ liệu và giao thức

| Mục | Quy tắc |
|---|---|
| Split | Dùng split chính thức nếu có (train / val / test). Nếu không có val, tách 10% từ train theo seed cố định. Không bao giờ lấy val/test vào kho. |
| Kho ứng viên | **20% tập train** (sau khi tách val), lấy ngẫu nhiên theo `POOL_SEED`, **cố định** suốt train/val/test. Lưu danh sách `image_id` ra `candidate_pool.csv`. |
| Ảnh trong kho | Vẫn được dùng làm ảnh truy vấn khi train. |
| Self-exclusion | Với mỗi truy vấn, `excluded_position = vị trí của chính nó trong kho` (hoặc −1); gán điểm −∞ trước softmax/argmax. Kiểm tra mỗi truy vấn còn ít nhất 1 ứng viên. |
| K | 1 ảnh ngữ cảnh mỗi truy vấn. |
| Chọn checkpoint | Theo **val loss nhỏ nhất**. Test chỉ chạy **một lần** ở cuối, sau khi khóa mọi lựa chọn. |
| Thước đo | Accuracy và Macro-F1 (báo cả hai). |
| Seed | `SEED = 42` (train), `POOL_SEED = 42`; khi so sánh chỉ đổi một yếu tố. Kết luận mức tăng < 0.5 điểm cần 3 seed. |

---

## 4. Thành phần mô hình

### 4.1 Backbone: DINOv3
- Nạp offline: `torch.hub.load(DINOV3_REPO_DIR, "dinov3_vitl16", source="local", weights=WEIGHTS_PATH)`.
- Cỡ: `dinov3_vits16` (384-d), `dinov3_vitb16` (768-d), **`dinov3_vitl16` (1024-d, mặc định)**. Patch 16.
- Chuẩn hóa ảnh ImageNet: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).
- Chuỗi token = `[CLS, register × R, patch × H·W]`. Số token đầu `prefix = 1 + R` (DINOv3 có R = 4). Lấy số này từ backbone, đừng gán cứng.
- **RoPE** chỉ xoay các patch token ở **cuối chuỗi**: trong `apply_rope`, `prefix = N − sin.shape[-2]`, mọi token trước đó không bị xoay.

### 4.2 Bộ mã hóa cặp ảnh (phân loại)
```python
q_tok, rope = backbone.prepare_tokens_with_masks(x_q)          # [B, prefix + HW, D]
c_tok, _    = backbone.prepare_tokens_with_masks(x_c)
pair = cat([q_tok[:, :prefix], q_tok[:, prefix:], c_tok[:, prefix:]], dim=1)   # [B, prefix + 2HW, D]
pair_rope = (cat([sin, sin], dim=-2), cat([cos, cos], dim=-2))  # cùng lưới 2D cho cả hai ảnh
for blk in backbone.blocks: pair = blk(pair, rope=pair_rope)
pair_cls = backbone.norm(pair)[:, 0]
```
- **Lưu ý khi viết mới:** code DTD hiện mở rộng RoPE bằng `repeat_interleave(2)` và bỏ qua nếu RoPE là tuple. Notebook mới nên nối `sin`/`cos` như trên để mỗi ảnh nhận đúng vị trí 2D, và assert `pair.shape[1] − prefix == 2·H·W`.
- Fine-tune toàn bộ backbone (LR nhỏ, mục 6).

### 4.3 Nhánh truy vấn và trộn ảnh (Visual TACS)
```python
q = LN_q(CLS(backbone.forward_features(x_q)))     # đặc trưng ảnh truy vấn đứng một mình
p = LN_c(pair_cls)                                 # đặc trưng của cặp
gate_v = sigmoid(MLP_4D→D→1([q, p, q*p, |q-p|]))   # bias lớp cuối khởi tạo −0.7
logits_visual = Head_q(q) + gate_v * Head_p(p)
v = normalize(LN(q + gate_v * p))                  # đặc trưng đưa vào M-TACS
```
- Không ngữ cảnh (dùng cho phần thưởng): `v0 = LN_q(CLS(x_q))`, đi qua cùng đầu phân loại M-TACS.

### 4.4 Bộ chọn
```python
s = logit_scale.clamp(0.01, 100) * normalize(CLS_q) @ normalize(bank).T / TEMPERATURE   # TEMPERATURE = 1.0
s = z_score(s, dim=1)                        # chuẩn hóa theo hàng (DTD làm vậy trước khi lấy mẫu)
s[i, excluded[i]] = -inf
train: a ~ Categorical(logits=s)   →  log_prob(a), entropy
eval : a = argmax(s)
```
- `bank` = CLS của mọi ứng viên, tính bằng **chính backbone đang train**, `torch.no_grad()`, làm mới **đầu mỗi epoch và trước khi đánh giá val**.

### 4.5 Kênh văn bản M-TACS (offline)
1. **Sinh caption** cho mọi ảnh (train/val/test) bằng VLM chỉ từ pixel:
   - Florence-2-base (`<MORE_DETAILED_CAPTION>`), cần `transformers==4.49.0`: chạy trong **tiến trình con riêng** rồi giải phóng VRAM.
   - Hoặc Qwen3-VL-8B-Instruct (chất lượng tốt hơn khoảng 1 điểm F1 trên DTD).
   - Rút gọn còn 5–8 từ mô tả thuộc tính nhìn thấy được.
2. **Lọc lộ nhãn:** chặn tên lớp, từ đồng nghĩa và dạng biến thể (danh sách cấm sinh từ `class_names`). Điểm CLIP so với tên lớp chỉ dùng để **kiểm tra**, không loại cứng.
3. **Không xóa ảnh khi caption lỗi:** `valid = False`, embedding = 0.
4. **Mã hóa** bằng CLIP ViT-B/32 text encoder đóng băng → 512-d, lưu `text_embeddings.pt` kèm `valid`.
5. Ghi `captions.csv` (cột: `image_id, split, raw_caption, final_caption, caption_valid, leakage_detected, vlm_model, prompt_version`) và **SHA-256** của file; ghi tạm `.tmp` rồi `os.replace` để resume an toàn.
6. **Kiểm tra rò rỉ:** train logistic regression chỉ trên embedding caption của train, đo trên val. Caption đã lọc nên gần mức ngẫu nhiên (DTD: 2.13% với 47 lớp).

### 4.6 Đầu trộn M-TACS (S01, `candidate_only`, mặc định)
```python
c = Dropout(0.1)(LN(Linear(512, D)(t_c)))
g = sigmoid(MLP_2D→D→1([v, c])) * m_c              # bias lớp cuối khởi tạo −1.0; m_c = valid của ứng viên
z = LN(v + g * c)
logits = Linear(D, num_classes)(Dropout(0.3)(z))
```
- Caption lấy theo **đúng chỉ số `a`** mà bộ chọn chọn.
- Kênh caption của ảnh truy vấn **tắt** (cổng nhân 0).

### 4.7 Biến thể tùy chọn (đã thử, không vượt S01)
| Mã | Thay đổi | Test F1 DTD |
|---|---|---|
| S02 | `c = normalize(c)` trước cổng | 84.93 |
| S03 | + loss phụ căn chỉnh ảnh–caption (log-Sinkhorn OT, trọng số 0.05) | 84.87 |
| S04 | Bộ chọn: `s = z_visual + α · clip(z_semantic, −3, 3)`, α = 0.25 · σ(w), khởi động 5 epoch chỉ dùng ảnh rồi tăng dần 5 epoch | 85.07 |
| S05 | Caption thành 1 token chèn sau CLS/register (không RoPE), hệ số khởi tạo 0, null token cho caption lỗi | 85.05 |

---

## 5. Vòng huấn luyện

### 5.1 Một bước train (đúng như notebook DTD)
```python
a, log_prob, entropy = selector.sample(x_q, bank, excluded)
x_c, t_c, m_c = P.image[a], P.text[a], P.valid[a]
out = model(x_q, x_c, t_c, m_c)                       # logits, features
ce = cross_entropy(out.logits, y, label_smoothing=0.1)
supcon = supervised_contrastive(out.features, y, temperature=0.1)

model.eval(); with no_grad():                          # phần thưởng tính không dropout
    l0 = cross_entropy(model(x_q, context=None).logits, y, reduction="none")
    l1 = cross_entropy(model(x_q, x_c, t_c, m_c).logits, y, reduction="none")
    r = l0 - l1
    adv = (r - r.mean()) / r.std(unbiased=False).clamp_min(1e-6)
model.train()

policy = -(log_prob * adv.detach()).mean()
loss = ce + 0.05 * supcon + 1.0 * policy - 0.01 * entropy
AMP + GradScaler; clip_grad_norm_(1.0); optimizer.step()
```

### 5.2 Mỗi epoch
1. Làm mới `bank` (no_grad).
2. Train một epoch như 5.1.
3. Làm mới `bank`, đánh giá val (argmax, không lấy mẫu).
4. Lưu `last.pt` (đầy đủ optimizer, scheduler, scaler, RNG) và `best.pt` nếu val loss giảm. Ghi `history.csv`.
5. Không early stopping trong chuỗi so sánh (giữ đủ số epoch); có thể bật patience cho lượt khám phá.

### 5.3 Nhánh Gumbel-Softmax ST (tùy chọn, giống bài báo)
```python
y_soft = softmax((s + Gumbel()) / tau)                 # tau = 0.1
y_hard = one_hot(y_soft.argmax(1))
y = y_hard - y_soft.detach() + y_soft
x_c = einsum("bn,nchw->bchw", y, P.image_fp32_chunk)   # nhân theo khối (chunk 128) để vừa bộ nhớ
L = L_grad(qua y) + λ · L_policy                        # λ = 1
```

---

## 6. Cấu hình mặc định

| Tham số | Giá trị |
|---|---|
| Backbone | DINOv3 ViT-L/16, fine-tune toàn bộ |
| Ảnh | 384 × 384 |
| Epoch / lịch LR | 50; warmup 5 epoch (hệ số 0.2 → 1.0), sau đó cosine, sàn 0.001 |
| Optimizer | AdamW, 3 nhóm: backbone LR **3e-6** (WD 0.05); selector LR 1e-4 (WD 1e-4); các đầu còn lại LR **1e-4** (WD 1e-4) |
| Batch (theo VRAM) | ≥ 80 GB: 32 · ≥ 40 GB: 16 · ≥ 20 GB: 8 · thấp hơn: 4 |
| Loss | CE (label smoothing 0.1) + 0.05 · SupCon (τ 0.1) + 1.0 · policy − 0.01 · entropy |
| Selector | temperature 1.0, `logit_scale` học được (khởi tạo 1.0) |
| Gumbel (nếu dùng) | τ = 0.1 |
| Dropout | Đầu phân loại 0.3; chiếu văn bản 0.1 |
| Cổng | Visual gate bias −0.7; text gate bias −1.0 |
| Augmentation train | Resize (S + 48), RandomResizedCrop S (scale 0.60–1.0, ratio 0.80–1.25), HFlip 0.5, ColorJitter (0.25, 0.25, 0.20, 0.04) p = 0.75 |
| Eval | Resize S × S, không augmentation |
| Grad clip | 1.0 |
| AMP | Bật trên CUDA |

---

## 7. Đánh giá và chẩn đoán

- Báo `accuracy`, `macro_f1`, `loss`, best epoch, tỷ lệ caption hợp lệ, cổng trung bình, tỷ lệ chọn **ảnh cùng lớp** (để biết bộ chọn có chọn ảnh tương phản không).
- Lưu dự đoán từng ảnh: `image_id, ground_truth, prediction, confidence, selected_context_id, candidate_caption`.
- **Phép thử can thiệp trên val** (cùng checkpoint, **giữ nguyên ảnh ngữ cảnh**):
  - `aligned` (caption đúng), `text_off` (mặt nạ = 0), `text_shuffled` (xáo caption giữa các ứng viên hợp lệ).
  - Nếu `text_off` hoặc `text_shuffled` không làm giảm F1 thì mức tăng không đến từ văn bản.
  - Báo khoảng tin cậy bootstrap của chênh lệch F1.
- **Đối chứng nên chạy kèm:** visual_only (cùng mọi thứ, tắt văn bản) và caption bị xáo trộn.

---

## 8. Kỹ thuật bộ nhớ và độ bền

- **Kho pixel ứng viên:** giữ ở fp16 (CPU hoặc GPU nếu đủ VRAM); chỉ copy ảnh được chọn lên GPU.
- **Nếu ViT-L tràn bộ nhớ:** tách `LOGICAL_BATCH_SIZE` (ví dụ 64) và `COMPUTE_BATCH_SIZE` (ví dụ 8):
  1. Lượt 1 `no_grad` theo từng phần nhỏ, lưu trạng thái RNG mỗi phần, tính phần thưởng cho cả batch logic để chuẩn hóa đúng.
  2. Lượt 2 khôi phục RNG để lấy lại **đúng** lựa chọn, bật gradient checkpointing, cộng dồn đạo hàm rồi mới `optimizer.step()`.
  - Không được thay bằng giảm batch đơn thuần, vì chuẩn hóa phần thưởng phụ thuộc kích thước batch.
- **Kaggle 12 giờ/phiên:** mỗi giai đoạn (sinh caption → CLIP → train → test) là một cell có cache; checkpoint `last.pt` đủ để resume. Ghi file bằng `.tmp` + `os.replace`. Kiểm tra protocol/fingerprint trước khi resume để không trộn lẫn lượt chạy.
- **Sinh caption:** tự giảm batch một nửa khi gặp CUDA OOM.

---

## 9. Bài học bắt buộc tuân thủ

1. Mặc định `candidate_only`; **không** đưa caption của ảnh truy vấn vào đặc trưng.
2. Luôn lọc lộ nhãn và kiểm tra rò rỉ trước khi train.
3. Caption lỗi: che bằng mặt nạ, **không xóa ảnh** khỏi bất kỳ tập nào.
4. Khóa file caption bằng SHA-256 khi so sánh các kiến trúc.
5. Mọi quyết định (ngưỡng, biến thể, checkpoint) dựa trên **val**; test chạy một lần.
6. Mỗi lần so sánh chỉ đổi một yếu tố; luôn chạy kèm visual_only.
7. Kết quả âm cũng được lưu và báo cáo.

---

## 10. Cấu trúc notebook gợi ý

| Cell | Nội dung |
|---|---|
| 0 | Markdown: mục tiêu, giao thức, input Kaggle cần attach |
| 1 | Import, `TOKENIZERS_PARALLELISM=false`, phát hiện GPU/VRAM, seed |
| 2 | Cấu hình (mục 6), đường dẫn dataset/DINOv3/CLIP/VLM, `OUTPUT_DIR`, kiểm tra output không trùng thí nghiệm khác |
| 3 | Đọc dataset → `SPLIT_FRAMES` (train/val/test, `image_id`, `class_id`, `image_path`), kiểm tra không trùng giữa các split |
| 4 | Tạo kho ứng viên 20% train, `CANDIDATE_POSITION`, lưu `candidate_pool.csv` |
| 5 | Sinh caption (tiến trình con, resume) → lọc → `captions.csv` + SHA-256 |
| 6 | Kiểm tra rò rỉ (logistic regression trên caption) |
| 7 | Mã hóa CLIP → `text_embeddings.pt`, mặt nạ `valid` |
| 8 | Transform, Dataset, DataLoader; kho pixel ứng viên fp16 |
| 9 | Nạp DINOv3; lớp Visual TACS (4.2–4.4) |
| 10 | Lớp M-TACS (4.6), hàm `select_context`, `run_forward` |
| 11 | Loss SupCon, `train_epoch` (5.1), `evaluate`, `refresh_bank`, optimizer/scheduler |
| 12 | Vòng train có resume, lưu `history.csv`, `best.pt`, `last.pt` |
| 13 | Phép thử can thiệp trên val (mục 7) |
| 14 | Test một lần với `best.pt`; xuất `results.csv`, dự đoán từng ảnh |
| 15 | Báo cáo: bảng metric, đường học, tỷ lệ chọn cùng lớp, ví dụ ảnh truy vấn + ngữ cảnh + caption |

---

## 11. Khi chuyển sang bộ dữ liệu phân loại khác

| Phải đổi | Ghi chú |
|---|---|
| `class_names`, cách đọc split | Ưu tiên split chính thức; ghi rõ số ảnh mỗi split |
| Danh sách từ cấm khi lọc caption | Sinh từ tên lớp + từ đồng nghĩa của dataset mới |
| Prompt VLM | Mô tả thuộc tính phù hợp miền (ví dụ: chim → màu lông, hình dạng mỏ; xe → kiểu thân xe) nhưng **không** gọi tên lớp |
| `IMAGE_SIZE` | 384 cho ViT-L; dataset ảnh nhỏ có thể dùng 256 để tiết kiệm |
| Kích thước kho | Vẫn 20% train; kho rất lớn thì cân nhắc RAM cho kho pixel |
| `BATCH_SIZE` | Theo bảng VRAM; nếu tràn bộ nhớ, áp dụng mục 8 |
| Số epoch | 50 là mặc định DTD; dataset lớn có thể cần nhiều hơn |
| Mốc so sánh | Luôn chạy visual_only (TACS chỉ dùng ảnh) với cùng cấu hình |
