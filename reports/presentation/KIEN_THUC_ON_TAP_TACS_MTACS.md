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
5. Đọc lại bảng câu hỏi khó ở mục 7.
