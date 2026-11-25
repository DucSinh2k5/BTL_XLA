# Kiến thức ôn tập: Baseline TACS và M-TACS

> Mục tiêu: hiểu đủ sâu để trả lời trơn tru khi được hỏi về baseline và M-TACS.
> Đọc theo thứ tự: mục 1 (1 phút) → mục 2 (nền tảng) → mục 3–6 (dự án) → mục 7 (câu hỏi khó).
> Mọi con số trong file đều là số thật từ notebook đã chạy.

---

## 1. Dự án trong 1 phút (học thuộc)

"Bài toán: khi phân loại hoặc phân đoạn một ảnh, mô hình được phép xem thêm **một ảnh tham khảo** lấy từ tập train. Cách cũ lấy ảnh **giống nhất**, nhưng ảnh giống nhất chưa chắc giúp đoán đúng.

**TACS** (CVPR 2026) học một **bộ chọn** được thưởng khi ảnh nó chọn **làm giảm loss** của bài toán. Vì phép chọn là rời rạc, TACS kết hợp **Gumbel-Softmax** (cho đạo hàm) và **REINFORCE** (cho phần thưởng).

Em **tái lập TACS** sát bài báo (CUB-200: 82.42 so với 82.1), sau đó làm hai cải tiến:
- **Cải tiến 1:** nâng backbone lên DINOv3 ViT-L/16 và tăng kích thước ảnh, điểm tăng mạnh (DTD 77.7 → 84.95).
- **Cải tiến 2, M-TACS:** thêm caption (mô tả văn bản) cho ảnh tham khảo. Qua 20 cấu hình ablation, thiết kế tốt nhất là **chỉ gắn caption cho ảnh ứng viên**, đạt 85.30 F1 so với 84.82 của TACS ViT-L."

---

## 2. Kiến thức nền tảng

### 2.1 Vision Transformer (ViT)
- Ảnh được cắt thành các **patch** 16×16 pixel. Mỗi patch được chiếu thành một vector, gọi là **token**.
- Ảnh 384×384 với patch 16 → 24×24 = **576 patch token**.
- Ngoài patch, chuỗi có thêm **token đặc biệt** ở đầu: **CLS** (tóm tắt toàn ảnh, dùng để phân loại) và các **register token** (DINOv3 có 4 register, dùng để "chứa" thông tin toàn cục, giúp bản đồ chú ý sạch hơn).
- Các khối Transformer dùng **self-attention**: mỗi token nhìn tất cả token khác.
- **RoPE** (rotary position embedding): cách mã hóa vị trí bằng phép xoay vector q/k theo tọa độ 2D của patch. Trong DINOv3, RoPE chỉ áp cho patch token, không áp cho CLS/register.

### 2.2 DINOv3
- Backbone ViT của Meta, **tự giám sát** (học không cần nhãn) trên dữ liệu rất lớn, cho đặc trưng thị giác mạnh.
- Ba cỡ dùng trong dự án: **ViT-S/16** (384 chiều, ~21M tham số), **ViT-B/16** (768 chiều), **ViT-L/16** (1024 chiều, ~300M tham số).
- **Fine-tune** = cập nhật trọng số backbone (dùng cho phân loại). **Đóng băng (frozen)** = giữ nguyên backbone, chỉ train phần thêm (dùng cho phân đoạn).

### 2.3 CLIP và VLM
- **CLIP** gồm bộ mã hóa ảnh và bộ mã hóa văn bản, học sao cho ảnh và câu mô tả đúng của nó có vector gần nhau. Dự án dùng **bộ mã hóa văn bản CLIP ViT-B/32** (đóng băng) để biến caption thành vector **512 chiều**.
- **VLM** (vision-language model) nhận ảnh và sinh ra văn bản:
  - **Florence-2-base** (0.23B tham số, nhỏ, nhanh)
  - **Qwen3-VL-8B / 32B** (lớn, mô tả tốt hơn)

### 2.4 Hai kỹ thuật học phép chọn rời rạc
- **Vấn đề:** chọn ảnh = argmax, không có đạo hàm, nên không lan truyền ngược được.
- **Gumbel-Softmax straight-through:** cộng nhiễu Gumbel vào điểm số rồi lấy softmax với nhiệt độ τ thấp (τ = 0.1). **Lượt xuôi** dùng one-hot cứng (đúng 1 ảnh); **lượt ngược** giả như đã dùng softmax mềm để lấy đạo hàm. Ưu: phương sai thấp. Nhược: đạo hàm bị lệch (biased).
- **REINFORCE (policy gradient):** coi bộ chọn là một **chính sách** π. Lấy mẫu hành động a, nhận phần thưởng r, cập nhật theo `−r · log π(a)`. Ưu: không lệch. Nhược: phương sai cao. Giảm phương sai bằng cách **chuẩn hóa phần thưởng theo batch** (trừ trung bình, chia độ lệch chuẩn).
- TACS **dùng cả hai** để bù nhược điểm của nhau.

### 2.5 Thước đo
- **Accuracy**: tỉ lệ đoán đúng.
- **Macro-F1**: tính F1 cho từng lớp rồi lấy trung bình, coi mọi lớp quan trọng như nhau. DTD cân bằng (40 ảnh test mỗi lớp) nên Accuracy ≈ Macro-F1.
- **Dice** = 2|A∩B| / (|A| + |B|); **IoU** = |A∩B| / |A∪B|, với A là mặt nạ dự đoán và B là mặt nạ đúng. Dice luôn ≥ IoU.
- Bài báo báo cáo **Top-1 Accuracy**; chuỗi DTD của dự án dùng **Macro-F1** làm thước đo chính. Khi so với bài báo luôn so accuracy với accuracy.

### 2.6 Quy tắc thực nghiệm
- **Train / val / test:** chọn mô hình, ngưỡng, checkpoint bằng **val**; **test chỉ chạy một lần** cuối cùng.
- **Seed:** hạt giống ngẫu nhiên. Cùng cấu hình nhưng khác seed vẫn cho điểm hơi khác; đó là lý do cần nhiều seed để kết luận mức tăng nhỏ.
- **Ablation:** thay đổi **một yếu tố** mỗi lần, giữ nguyên mọi thứ khác, để biết yếu tố đó có tác dụng gì.
- **Rò rỉ dữ liệu (leakage):** mô hình "nhìn thấy" đáp án qua đường không hợp lệ (ví dụ caption chứa tên lớp, hoặc ảnh tự chọn chính nó).

---

## 3. TACS chi tiết (baseline)

### 3.1 Các thành phần
| Thành phần | Chi tiết |
|---|---|
| Kho ứng viên | **20% tập train**, cố định suốt train/val/test. Không bao giờ lấy từ val/test. DTD: 376 ảnh; SUN397: 3,573 ảnh; ISIC: 400 ảnh |
| Self-exclusion | Ảnh truy vấn nằm trong kho thì điểm của chính nó bị gán −∞, không được tự chọn |
| Số ảnh ngữ cảnh | **K = 1** |
| Bộ chọn | Lấy CLS của DINOv3 cho ảnh truy vấn và mọi ứng viên, chuẩn hóa, tính cosine (nhân hệ số học được), ra một điểm cho mỗi ứng viên |
| Kho đặc trưng | Đặc trưng của ứng viên được tính lại mỗi epoch **không lấy đạo hàm**, để chấm điểm nhanh trên toàn kho |
| Mạng phân loại | Ghép `[CLS, register, patch truy vấn, patch ngữ cảnh]` thành một chuỗi, đi chung qua ViT; fine-tune toàn bộ |
| Mạng phân đoạn | Backbone đóng băng; patch truy vấn nhìn sang patch ngữ cảnh qua **Gated Cross-Attention** (2 khối cuối); đầu **DPT-lite** giải mã mặt nạ; loss BCE + Dice |

### 3.2 Hàm mất mát
```
L_TACS = L_grad + λ · L_policy        (λ = 1)

r(x_q, a) = L_task(x_q, không ngữ cảnh) − L_task(x_q, x_c^a)     (tách đạo hàm, chuẩn hóa theo batch)
L_policy  = − r̂ · log π(a | x_q)
```
- r > 0 nghĩa là ảnh ngữ cảnh giúp giảm loss, nên xác suất chọn nó được tăng.
- Mỗi bước train phải chạy **hai lượt**: có ngữ cảnh và không ngữ cảnh. Đây là lý do TACS tốn bộ nhớ.
- Trong code DTD, L_task là cross-entropy với **label smoothing 0.1**, cộng thêm **supervised contrastive loss** (trọng số 0.05) trên đặc trưng.

### 3.3 Điều bài báo phát hiện
- TACS chọn ảnh **khác lớp** nhiều hơn 40–70% so với Frozen DINO, và ảnh được chọn đa dạng hơn (LPIPS cao hơn).
- Diễn giải: mô hình học cách chọn **ví dụ tương phản** để làm rõ ranh giới, giống bác sĩ so sánh với ca khác để phân biệt.

### 3.4 Kết quả tái lập
| Dataset | Cấu hình | Bài báo | Của em |
|---|---|---|---|
| CUB-200 | No-Context, ViT-S, 256, 3 seed | 82.1 | **82.42** (82.46 / 82.27 / 82.53) |
| Stanford Cars | No-Context, ViT-S, 256 | 92.4 | 91.59 |
| SUN397 | TACS, ViT-S, 256 | 71.8 | 74.01 |
| ISIC 2017 | TACS, ViT-S, ảnh 448 | 85.0 / 76.5 | 85.60 / 77.38 |

Lưu ý: trên CUB chỉ có No-Context. Bảng CUB chứng minh **pipeline dữ liệu và huấn luyện đúng**; TACS có ngữ cảnh được kiểm chứng trên SUN397 và Cars.

---

## 4. Cải tiến 1: mở rộng quy mô

### 4.1 Ý tưởng và kết quả
- ViT-S (22M) → ViT-B → **ViT-L (300M)**; ảnh 256 → **384** (phân loại), **448** (phân đoạn).
- Lý do: backbone lớn có đặc trưng phong phú hơn; ảnh lớn giữ chi tiết nhỏ (hoa văn, vật thể nhỏ, ranh giới tổn thương).

| Dataset | Bài báo | Bản nâng cấp |
|---|---|---|
| DTD | 77.7 | 84.95 (ViT-L, 384) |
| SUN397 | 71.8 | 74.01 → 79.42 (ViT-B, 384) → 81.11 (ViT-L, 384, val) |
| Cars | 94.1 | 95.15 (ViT-B, 384) |
| ISIC (Dice) | 85.0 | 85.60 (ViT-S, 448) → 88.62 (ViT-L, 448, val) |

### 4.2 Vấn đề bộ nhớ và cách giải (hay được hỏi)
- **Vấn đề:** chuỗi ghép 2 ảnh 384 có khoảng 1,150 token; mỗi bước phải chạy cả lượt có và không có ngữ cảnh; batch 64. Kết quả là tràn cả GPU 95 GB trên SUN397.
- **Không thể chỉ giảm batch xuống 8**, vì REINFORCE chuẩn hóa phần thưởng theo batch: batch nhỏ làm thống kê nhiễu, thay đổi thuật toán.
- **Giải pháp:** tách **batch logic 64** và **batch tính toán 8**.
  1. *Lượt 1 (không lấy đạo hàm):* chạy từng phần 8 ảnh, lưu trạng thái sinh số ngẫu nhiên (RNG) của mỗi phần, tính đủ 64 phần thưởng để chuẩn hóa đúng trên 64.
  2. *Lượt 2:* khôi phục đúng RNG để bộ chọn lặp lại **chính xác** lựa chọn của lượt 1, bật **gradient checkpointing**, cộng dồn đạo hàm qua 8 phần rồi mới cập nhật một lần.
- Kết quả: giữ đúng thuật toán, vừa bộ nhớ.

---

## 5. Cải tiến 2: M-TACS

### 5.1 Pipeline caption
```
Ảnh ──► VLM (Florence-2 / Qwen3-VL) ──► caption 5–8 từ ──► bộ lọc ──► CLIP text (512-d) ──► lưu sẵn text_embeddings.pt
```
- Caption chỉ được sinh **từ pixel của ảnh**; VLM không nhận nhãn.
- **Bộ lọc `dtd_lexical_v2` (DTD):** bỏ tiền tố chung chung, chặn tên lớp và từ đồng nghĩa; điểm CLIP so với tên lớp chỉ dùng để kiểm tra, không loại cứng (bộ lọc cũ ngưỡng 0.30 loại nhầm gần như toàn bộ caption).
- **Chính sách ISIC `isic_visible_attributes_v1`:** chỉ mô tả màu sắc, đường viền, cấu trúc; cấm thuật ngữ chẩn đoán; phủ 100% trên 2,750 ảnh.
- **Caption lỗi không xóa ảnh:** vector 0 + mặt nạ `valid = False`, cổng tự bằng 0. Nhờ vậy tập dữ liệu và kho ứng viên giống hệt baseline, so sánh công bằng.
- Kỹ thuật: Florence-2 cần transformers 4.49 (code cũ), nên chạy trong **tiến trình con riêng**, ghi CSV, giải phóng VRAM rồi mới nạp DINOv3.

### 5.2 Kiến trúc trộn (S01, thiết kế tốt nhất)
```
z  = đặc trưng ảnh từ ViT (cặp truy vấn + ngữ cảnh)
t  = vector CLIP của caption ảnh ứng viên được chọn
c  = W·t                          (chiếu 512 → 1024)
g  = sigmoid(MLP([z, c])) · m      (m = 1 nếu caption hợp lệ; bias khởi tạo −1 nên cổng ban đầu nhỏ)
z' = LayerNorm(z + g · c)  ──► bộ phân loại
```
- Caption lấy theo **đúng chỉ số** ảnh ứng viên mà bộ chọn chọn.
- Cổng `g` cho mô hình tự quyết định dùng văn bản nhiều hay ít; nếu caption vô ích, cổng có thể về gần 0 mà không phá đặc trưng ảnh.

### 5.3 Ba vị trí có thể đặt văn bản (cốt lõi của ablation)
| Vị trí | Ý nghĩa |
|---|---|
| Query text | Caption của ảnh truy vấn cộng vào đặc trưng |
| Candidate text | Caption của ảnh ứng viên được chọn cộng vào đặc trưng |
| Selector text | Caption tham gia tính điểm chọn ứng viên |

Tên cấu hình: `visual_only` (không văn bản nhưng **vẫn có ảnh ngữ cảnh**, khác `no_context` của bài báo), `query_only`, `candidate_only`, `both_fusion` (query + candidate), `full_both` (cả ba vị trí).

### 5.4 Các phiên bản kiến trúc (chuỗi S)
| Mã | Thay đổi | Test F1 | Ý chính |
|---|---|---|---|
| S01 | Candidate-only, caption Prefix cố định | **85.30** | Tốt nhất; phát hiện vector văn bản có độ lớn gấp ~32 lần vector ảnh nên cổng rất nhỏ (~0.001) |
| S02 | Chuẩn hóa L2 vector văn bản | 84.93 | Cổng mở lớn hơn ~55 lần (0.062) nhưng điểm không tăng |
| S03 | + căn chỉnh m-LTM (Sinkhorn Optimal Transport, metric Mahalanobis chéo) | 84.87 | Căn chỉnh tốt trên kho nhưng không giúp phân loại |
| S04 | + đưa độ tương hợp ảnh truy vấn–caption ứng viên vào **bộ chọn** | 85.07 | `điểm = z_visual + α·z_semantic`, α học được ≈ 0.131 (tối đa 0.25), khởi động 5 epoch chỉ dùng ảnh; tắt caption làm đổi 5–11% lựa chọn |
| S05 | Caption thành **một token trong ViT** (không RoPE, hệ số khởi tạo 0) | 85.05 | Token học được rất nhỏ (~0.7% độ lớn patch) |

### 5.5 Kết quả ablation cần nhớ
- Vị trí: **candidate_only 85.03** > visual_only 84.82 > shuffle 83.34 > query_only 82.31 > full_both 80.91.
- Lọc caption: đã lọc 80.91 vs chưa lọc 78.98 (**+1.93**); che tên lớp 81.11.
- Nguồn caption (full_both): Florence Prefix 81.07, LoRA 80.67, **Qwen-8B 82.14**, Qwen-32B 82.08; CLIP rerank (Grounded) giảm điểm vì ưu tiên câu chung chung.
- ISIC (ViT-L, 448, val): TACS 88.62 → M-TACS **88.83** Dice.

### 5.6 Vì sao caption ở ảnh truy vấn lại hại? (hiểu để giải thích)
- Caption của ảnh truy vấn được sinh từ **chính ảnh đó**, không có thông tin mới, chỉ thêm nhiễu vào đặc trưng mà ViT-L đã biểu diễn rất tốt.
- Caption của ảnh ứng viên mô tả **ảnh tham khảo**, đi kèm đúng ảnh mà bộ chọn chọn, nên đóng vai trò ngữ cảnh bổ trợ thay vì ghi đè đặc trưng của ảnh cần đoán.

---

## 6. Hạn chế (biết rõ để chủ động nói)
- Mức tăng của M-TACS nhỏ (+0.2 đến +0.5) và chỉ có **1 seed**.
- **Kênh văn bản chưa được tận dụng tối đa:** ở khâu trộn, cổng văn bản luôn nhỏ; khi tắt hoặc xáo caption, điểm trên val gần như không đổi. Nơi văn bản có tác dụng đo được là **khâu chọn** (S04).
- Lý do sâu: caption sinh từ ảnh là thông tin **trùng lặp** với ảnh; 5–8 từ quá thô so với 576 patch token.
- Bản web M-TACS là **full_both**, train trước khi ablation chốt thiết kế.
- **Hướng phát triển:** trộn ở mức token bằng cross-attention, dùng văn bản mang thông tin mới (mô tả lớp của ảnh ngữ cảnh), backbone có không gian chung ảnh–văn bản (SigLIP 2, dino.txt), chạy 3 seed, thử K = 3.

---

## 7. Câu hỏi khó và cách trả lời

### Về baseline TACS
| Câu hỏi | Trả lời |
|---|---|
| Sao không dùng luôn ảnh giống nhất? | Ảnh giống nhất thường cùng lớp, gần như bản sao, không thêm thông tin; bài báo cho thấy Frozen DINO còn kém hơn không dùng ngữ cảnh trên CUB và DTD. TACS chọn theo **lợi ích thực tế** (giảm loss). |
| Sao phải dùng cả Gumbel lẫn REINFORCE? | Gumbel cho đạo hàm phương sai thấp nhưng lệch; REINFORCE không lệch nhưng phương sai cao. Kết hợp để học ổn định và đúng hướng. |
| Phần thưởng tính thế nào? | Loss khi không có ngữ cảnh trừ loss khi có ảnh được chọn; tách khỏi đồ thị đạo hàm và chuẩn hóa theo batch. |
| Sao cần self-exclusion? | Nếu ảnh truy vấn tự chọn chính nó, mô hình chỉ việc "chép" đặc trưng của mình, phần thưởng giả và học sai. |
| Sao kho ứng viên chỉ 20% train và cố định? | Giữ đúng giao thức bài báo, giảm chi phí chấm điểm, và lúc test chỉ dùng ảnh train (không lộ val/test). |
| Phân loại và phân đoạn khác nhau thế nào? | Phân loại ghép token hai ảnh và fine-tune toàn bộ ViT; phân đoạn giữ backbone cố định, dùng cross-attention có cổng và đầu DPT. |
| Sao baseline của em cao hơn bài báo? | Chênh lệch nhỏ trên CUB (+0.3) nằm trong dao động hợp lý; SUN397 cao hơn có thể do cách tách val và chi tiết huấn luyện. Em không tuyên bố vượt bài báo ở cùng cấu hình; mục đích là chứng minh pipeline đúng. |

### Về Cải tiến 1
| Câu hỏi | Trả lời |
|---|---|
| Chỉ là dùng mô hình to hơn? | Đúng là mở rộng quy mô, nhưng với TACS không đơn giản: mỗi bước chạy hai lượt trên chuỗi khoảng 1,150 token, tràn cả GPU 95 GB. Em phải thiết kế lại vòng train (tách batch logic/tính toán, chạy lại RNG, gradient checkpointing) mà vẫn giữ đúng thuật toán. |
| Sao không giảm batch cho đơn giản? | Vì chuẩn hóa phần thưởng REINFORCE phụ thuộc kích thước batch; giảm batch sẽ thay đổi thuật toán và làm thống kê nhiễu. |

### Về M-TACS
| Câu hỏi | Trả lời |
|---|---|
| Caption có lộ nhãn không? | Có bộ lọc chặn tên lớp và từ đồng nghĩa; caption chưa lọc cho điểm thấp hơn 1.93, cho thấy bộ lọc có tác dụng. VLM không nhận nhãn khi sinh caption. |
| Caption lỗi thì sao? | Không xóa ảnh; vector 0 + mặt nạ, cổng bằng 0, nên dữ liệu giống hệt baseline. |
| Sao chọn candidate-only? | Ablation 8 cấu hình cho thấy chỉ candidate-only vượt TACS; caption ở ảnh truy vấn giảm 2.5 điểm vì trùng thông tin với chính ảnh đó. |
| Sao phiên bản đơn giản S01 lại tốt nhất? | Các phiên bản phức tạp hơn (chuẩn hóa, Optimal Transport, token) thêm tham số và mục tiêu phụ nhưng không thêm thông tin mới; với 1,880 ảnh train, đơn giản thường ổn định hơn. |
| Có bằng chứng mô hình dùng văn bản không? | Ở bộ chọn (S04): tắt caption làm đổi 5–11% ảnh ngữ cảnh. Ở khâu trộn, cổng còn nhỏ; đây là hạn chế em đã nêu và là hướng cải tiến. |
| Mức tăng 0.48 có ý nghĩa không? | Còn nhỏ và mới 1 seed; cần chạy 3 seed. Điều chắc chắn là ablation chỉ ra thiết kế đúng và sai một cách rõ ràng. |
| Qwen hay Florence tốt hơn? | Qwen3-VL-8B tốt hơn Florence khoảng 1 điểm F1; 32B không hơn 8B nên 8B hiệu quả hơn về chi phí. |
| Sao làm ablation trên DTD mà demo trên SUN397? | DTD nhỏ (5,640 ảnh) nên chạy được hơn 20 cấu hình cùng giao thức; SUN397 ảnh rõ, nhãn khung cảnh dễ kiểm chứng bằng mắt nên hợp để demo. |
| `visual_only` khác `no_context` thế nào? | `visual_only` vẫn có 1 ảnh ngữ cảnh, chỉ tắt văn bản (chính là TACS); `no_context` không có ảnh ngữ cảnh nào. |

---

## 8. Bảng thuật ngữ nhanh

| Thuật ngữ | Giải thích ngắn |
|---|---|
| Query / truy vấn | Ảnh cần dự đoán |
| Candidate / ứng viên | Ảnh trong kho có thể được chọn làm ngữ cảnh |
| Context / ngữ cảnh | Ảnh (và caption) được chọn để hỗ trợ dự đoán |
| Selector / bộ chọn | Mạng chấm điểm và chọn ứng viên |
| Gate / cổng | Hệ số 0–1 học được, quyết định dùng bao nhiêu văn bản |
| Self-exclusion | Cấm ảnh tự chọn chính nó |
| Ablation | Thí nghiệm bật/tắt từng thành phần |
| Leakage / lộ nhãn | Thông tin đáp án lọt vào đầu vào |
| Gradient checkpointing | Không lưu toàn bộ kích hoạt, tính lại khi lan truyền ngược để tiết kiệm bộ nhớ |
| Optimal Transport (Sinkhorn) | Bài toán ghép phân phối ảnh và văn bản với chi phí nhỏ nhất, dùng ở S03 |
| Bundle | Gói mô hình tự chứa dùng cho web |

---

## 9. Ôn nhanh 30 phút trước khi trình bày
1. Đọc to mục 1 hai lần.
2. Tự vẽ lại sơ đồ TACS (kho 20% → bộ chọn → cặp ảnh → mạng tác vụ) và viết công thức phần thưởng.
3. Nhớ 5 con số: **82.42** (CUB), **84.82** (TACS ViT-L), **85.30** (M-TACS tốt nhất), **−2.5** (caption ở truy vấn), **+1.93** (bộ lọc).
4. Nói được 3 câu về hạn chế và 3 hướng phát triển.
5. Đọc lại bảng câu hỏi khó ở mục 7 và tự trả lời bộ câu hỏi luyện tập ở mục 10.

---

## 10. Bộ câu hỏi luyện tập theo mức độ (dễ → cực khó)
Bổ sung cho mục 7, không trùng câu. Phần **in đậm** là ý chính phải nói ra.

### 10.1 Mức 1: Dễ (nắm khái niệm)

**1. Ngữ cảnh (context) trong TACS là gì?**
Ngữ cảnh là một ảnh khác lấy từ tập train. Ảnh này được ghép cùng ảnh cần phân loại rồi đưa chung vào ViT. Mô hình nhìn **cả hai ảnh cùng lúc**, nên có thêm một "ví dụ tham khảo" để quyết định.

**2. K = 1 nghĩa là gì?**
Mỗi ảnh truy vấn chỉ được ghép với **1 ảnh ngữ cảnh**. Bài báo dùng K = 1 làm cấu hình chính. Em giữ nguyên để so sánh được với bài báo, và vì mỗi ảnh ngữ cảnh thêm vào sẽ làm số token tăng lên tương ứng.

**3. DTD có gì đặc biệt?**
DTD gồm 47 lớp kết cấu bề mặt (sọc, chấm, vân gỗ...), tổng 5,640 ảnh, chia đều 1880 train / 1880 val / 1880 test. Bộ này nhỏ và cân bằng lớp, nên phù hợp để chạy nhiều ablation với cùng một giao thức.

**4. Vì sao kho ứng viên trên DTD là 376 ảnh?**
376 là 20% của 1880 ảnh train, đúng tỷ lệ bài báo dùng. Kho này **cố định trong suốt quá trình huấn luyện**, và lúc test cũng chỉ chọn ngữ cảnh từ kho này.

**5. Caption là gì và lấy từ đâu?**
Caption là một câu mô tả ảnh do mô hình VLM sinh ra (Florence-2 hoặc Qwen3-VL). Câu đó được mã hóa bằng CLIP thành vector 512 chiều. CLIP được **đóng băng**, không huấn luyện thêm.

### 10.2 Mức 2: Trung bình (hiểu cách vận hành)

**6. Lúc huấn luyện và lúc test, bộ chọn hoạt động khác nhau thế nào?**
Lúc **huấn luyện**, bộ chọn **lấy mẫu** theo xác suất (phân phối Categorical) để còn khám phá các ứng viên khác. Lúc **test**, nó lấy **argmax**, tức chọn ứng viên có điểm cao nhất, nên kết quả tất định và lặp lại được.

**7. Vì sao cần thêm số hạng entropy (−0.01·entropy) vào loss?**
Để chống **sụp đổ lựa chọn**. Nếu không có nó, bộ chọn sớm dồn hết xác suất vào vài ảnh quen thuộc và ngừng thử các ứng viên khác. Entropy thưởng cho việc giữ phân phối còn "mở" lúc đầu.

**8. Đặc trưng của kho ứng viên được cập nhật khi nào?**
Đặc trưng được tính lại (không lưu đạo hàm) ở **đầu mỗi epoch** và **trước mỗi lần đánh giá val**. Trong một epoch, đặc trưng hơi "cũ" so với trọng số hiện tại. Đó là đánh đổi để không phải chạy lại 376 ảnh ở mỗi bước, giống ý tưởng memory bank.

**9. Vì sao learning rate của backbone (3e-6) nhỏ hơn của các head (1e-4) khoảng 30 lần?**
Backbone DINOv3 đã được huấn luyện trước rất tốt. LR lớn sẽ phá vỡ đặc trưng có sẵn, nhất là trên tập nhỏ 1880 ảnh. Các head thì khởi tạo mới nên cần học nhanh. Lịch học là warmup 5 epoch rồi giảm theo cosine.

**10. Vì sao chọn checkpoint theo val loss nhỏ nhất, không theo val F1 cao nhất?**
Val loss **mượt và ổn định hơn**. F1 trên 1880 ảnh có thể nhảy vài ảnh giữa các epoch. Quy tắc này được cố định **trước** khi xem kết quả test để không "chọn ngẫu nhiên may mắn". Ví dụ: ở bundle SUN397 W01, epoch 43 có val acc cao nhất, nhưng checkpoint được đóng gói là epoch 29 vì đó là epoch có loss nhỏ nhất.

### 10.3 Mức 3: Khó (bảo vệ thiết kế và kết quả)

**11. A06 (caption bị xáo trộn, 83.34) lại cao hơn A05 (caption đúng, 80.91). Giải thích thế nào?**
Khi bị xáo, caption không còn liên quan tới ảnh, nên mô hình **học cách bỏ qua văn bản** và kết quả tiến gần bản chỉ dùng ảnh (84.82). Còn caption đúng đặt ở ảnh truy vấn thì "trông có vẻ hữu ích", nên mô hình dựa vào nó và bị overfit trên tập train nhỏ. Đây là bằng chứng rằng **văn bản ở nhánh truy vấn gây hại**, không phải nhờ nó có nội dung đúng.

**12. Gate của S01 chỉ khoảng 0.001. Như vậy có phải văn bản vô dụng?**
Cần nói kỹ hai ý.
- **Không thể đọc gate một mình.** Vector văn bản sau LayerNorm có độ lớn khoảng 32, trong khi đặc trưng ảnh có độ lớn khoảng 1. Gate nhỏ một phần là để bù độ lệch này: đóng góp thực của văn bản xấp xỉ 0.001 × 32 ≈ 3% so với đặc trưng ảnh, nhỏ nhưng khác 0.
- **Nhưng em không né sự thật.** Thí nghiệm tắt hoặc xáo văn bản trên val gần như không làm đổi kết quả. Vì vậy em kết luận ở **khâu trộn**, văn bản đóng góp rất ít. Tác dụng rõ ràng duy nhất đo được nằm ở **khâu chọn** (S04).

**13. 85.30 so với 84.82, tức chênh 0.48 điểm, tương đương bao nhiêu ảnh?**
Khoảng **9 ảnh** trên 1880 ảnh test. Mức này nằm trong dao động thường gặp giữa các seed. Muốn khẳng định cần chạy 3 seed và làm kiểm định cặp (McNemar trên cùng tập test). Em trình bày con số này là **"một tín hiệu cùng chiều với ablation"**, không phải "cải tiến đã được chứng minh".

**14. Làm sao biết bộ lọc caption thật sự chặn được việc lộ nhãn?**
Em chạy một **probe**: dùng chính caption đã lọc để đoán lớp. Kết quả là **2.13%**, bằng đúng mức đoán ngẫu nhiên 1/47. Ngoài ra, bản dùng caption chưa lọc (A07) thấp hơn, cho thấy bộ lọc có tác dụng thật.

**15. Study06 dùng nhãn của ảnh ứng viên. Đó có phải gian lận không?**
**Không**, vì ảnh ứng viên đều lấy từ tập train, nên nhãn của nó đã biết sẵn lúc suy luận, giống hệt cách kNN hoặc RAC (CVPR 2022) hoạt động. **Nhãn của ảnh truy vấn không bao giờ là đầu vào.** Điều kiện bắt buộc là giữ **self-exclusion**: nếu ảnh train được phép chọn chính nó thì mô hình đọc được nhãn của chính mình, thành lộ nhãn thật. Tuy nhiên đây là biến thể khác giao thức TACS gốc, phải báo cáo riêng và so với baseline kNN.

### 10.4 Mức 4: Rất khó (hiểu sâu toán và lý thuyết)

**16. Vì sao không lan truyền đạo hàm qua argmax được, và Gumbel-ST giải quyết thế nào?**
Argmax là hàm bậc thang, đạo hàm bằng 0 ở hầu hết mọi điểm, nên bộ chọn không nhận được tín hiệu học. Gumbel-ST làm hai việc:
- **Lượt tiến** dùng one-hot cứng, nên mô hình thật sự chỉ thấy 1 ảnh.
- **Lượt lùi** giả vờ đã dùng softmax mềm với nhiệt độ τ = 0.1, nên đạo hàm có thể chảy về bộ chọn.

Cái giá là đạo hàm bị **lệch**, vì lượt lùi không đúng với hàm thật. τ càng nhỏ thì càng sát one-hot nhưng phương sai càng lớn.

**17. Phần thưởng phụ thuộc vào chính mô hình đang học. Gradient của REINFORCE có còn đúng không?**
Với trọng số tại mỗi bước, REINFORCE ước lượng đúng (không lệch) đạo hàm của "phần thưởng kỳ vọng" theo bộ chọn: ∇E[r] = E[r·∇log π]. Phần thưởng được tính trong eval mode, no_grad và tách khỏi đồ thị, nên nó chỉ là **một con số đánh giá**, không có đạo hàm đi qua nó. Vì mô hình thay đổi liên tục nên bài toán **không dừng**: cùng một ứng viên có thể được thưởng khác nhau theo thời gian. Đó là lý do phải chuẩn hóa phần thưởng theo batch và cập nhật kho ứng viên mỗi epoch.

**18. Vì sao phải chuẩn hóa phần thưởng theo batch (trừ trung bình, chia độ lệch chuẩn)?**
Có hai lý do.
- **Trừ trung bình** đóng vai trò baseline: giảm phương sai mà không làm lệch gradient. Nhờ vậy chỉ những lựa chọn **tốt hơn mức trung bình** mới được khuyến khích. Nếu không làm vậy, khi mọi phần thưởng đều dương, mọi lựa chọn đều bị đẩy lên.
- **Chia độ lệch chuẩn** giữ độ lớn của gradient ổn định khi loss giảm dần trong quá trình train.

Vì thống kê này phụ thuộc kích thước batch, nên em giữ batch logic 64 dù mỗi lần chỉ tính được 8 ảnh.

**19. Ghép 2 ảnh vào ViT tốn thêm bao nhiêu chi phí?**
Ảnh 384 với patch 16 cho 24 × 24 = 576 token. Ghép 2 ảnh thành khoảng 1152 token, chưa tính CLS và register. Self-attention tốn chi phí theo **bình phương số token**, nên chi phí attention tăng khoảng **4 lần**. Ngoài ra, mỗi bước train còn chạy thêm một lượt không ngữ cảnh để tính phần thưởng, và phải chấm điểm toàn bộ kho ứng viên. Đây là lý do SUN397 bị tràn cả GPU 95 GiB.

### 10.5 Mức 5: Cực khó (câu "bẫy" và câu phản biện)

**20. Nếu cô hỏi: "Có thể TACS tốt hơn chỉ vì mô hình xử lý gấp đôi token (thêm tính toán), chứ không phải vì chọn ngữ cảnh?" thì trả lời sao?**
Cần trả lời bằng **các phép đối chứng**:
- **Ngữ cảnh ngẫu nhiên**, cùng số token nhưng không chọn: bài báo đã so sánh và kho mã của em có bản chạy trên Stanford Cars.
- **Frozen DINO**, chọn theo độ giống nhưng không học: bài báo đạt 75.4 so với 77.7 của TACS trên DTD.
- **No-context**: không có ảnh ngữ cảnh.

Cả ba đều dùng cùng backbone. Bản chọn bằng cách học tốt hơn bản ngẫu nhiên và bản chọn theo độ giống, nên lợi ích không chỉ đến từ việc có thêm token. Em nên nói thêm một ý trung thực: em chưa tự chạy đủ bộ đối chứng này trên DTD với ViT-L, nên phần này em dựa vào kết quả của bài báo.

**Câu bonus (phản biện mạnh nhất): "Vậy M-TACS của em đóng góp gì, khi văn bản gần như không được dùng?"**
Đóng góp của em là **một nghiên cứu phân tích có kiểm soát**, không phải tuyên bố "em làm tăng điểm":
1. Em chỉ ra **văn bản nên đặt ở đâu**: chỉ ở ảnh ứng viên. Đặt ở ảnh truy vấn làm giảm 2.5 điểm.
2. Em đưa ra **bằng chứng cơ chế**: văn bản tác động ở **khâu chọn** (S04 đổi 5–11% lựa chọn), còn ở khâu trộn thì gần như không.
3. Em xây **giao thức chống lộ nhãn**: bộ lọc, probe 2.13%, mặt nạ cho caption lỗi.

Câu chốt: **"Kết quả âm cũng là kết quả, miễn là được đo đúng cách và giải thích được."**
