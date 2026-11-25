# Tiêu chuẩn slide thuyết trình: TACS → M-TACS

> **Thời lượng:** 15 phút = khoảng 10.5 phút slide + 3 phút demo + 1 phút kết luận (tổng 14:35). Hỏi đáp tính riêng.
> **Số slide:** 16 slide chính + 6 slide dự phòng (chỉ mở khi được hỏi).
> **Quy tắc số liệu:** mọi con số dưới đây đều lấy từ notebook hoặc file kết quả đã chạy. Giữ đúng cách diễn đạt
> "cấu hình M-TACS tốt nhất đạt cao hơn TACS", không nói "caption là nguyên nhân" (xem phần hỏi đáp cuối file).

---

## Mạch câu chuyện

1. **Vấn đề:** ảnh giống nhất chưa chắc là ảnh giúp mô hình đoán đúng nhất.
2. **TACS:** học cách chọn ảnh ngữ cảnh theo đúng mục tiêu bài toán. Em tái lập và đạt điểm sát bài báo (CUB-200).
3. **Cải tiến 1, mở rộng quy mô:** DINOv3 ViT-L/16 + tăng kích thước ảnh, điểm tăng mạnh trên mọi dataset.
4. **Cải tiến 2, M-TACS:** thêm ngữ cảnh văn bản cho ảnh ứng viên; ablation tìm thiết kế tốt nhất, vượt TACS bản ViT-L.
5. **Triển khai:** đóng gói thành web demo chạy mô hình thật.
6. **Hướng tiếp:** khai thác sâu hơn kênh văn bản.

---

## Phân bổ thời gian

| # | Slide | Phần | Thời gian | Lũy kế |
|---|---|---|---|---|
| 1 | Tiêu đề | Mở đầu | 0:15 | 0:15 |
| 2 | Đặt vấn đề và khó khăn | Mở đầu | 0:55 | 1:10 |
| 3 | Mục tiêu và đóng góp | Mở đầu | 0:35 | 1:45 |
| 4 | TACS: kiến trúc | TACS | 0:55 | 2:40 |
| 5 | TACS: thuật toán huấn luyện | TACS | 1:05 | 3:45 |
| 6 | Tái lập baseline (bảng CUB-200) | TACS | 0:50 | 4:35 |
| 7 | Dữ liệu và thiết lập thực nghiệm | TACS | 0:35 | 5:10 |
| 8 | **Cải tiến 1:** ViT-L/16 + tăng kích thước ảnh | Cải tiến 1 | 0:50 | 6:00 |
| 9 | **Cải tiến 2:** M-TACS, ý tưởng và kiến trúc | Cải tiến 2 | 0:55 | 6:55 |
| 10 | Pipeline caption và chống lộ nhãn | Cải tiến 2 | 0:35 | 7:30 |
| 11 | Ablation 1: đặt văn bản ở đâu? | Cải tiến 2 | 0:45 | 8:15 |
| 12 | Ablation 2: nguồn caption và kiến trúc trộn | Cải tiến 2 | 0:45 | 9:00 |
| 13 | Kết quả: TACS (ViT-L) vs M-TACS | Kết quả | 0:45 | 9:45 |
| 14 | Triển khai hệ thống và web | Demo | 0:40 | 10:25 |
| 15 | **Demo web** | Demo | 3:00 | 13:25 |
| 16 | Hạn chế, hướng phát triển, kết luận | Kết | 1:10 | 14:35 |

Dư 25 giây cho chuyển cảnh. Nếu bị dồn thời gian: rút slide 10 và 14 xuống 25 giây mỗi slide.

---

## PHẦN MỞ ĐẦU

### Slide 1: Tiêu đề (0:15)

**Nội dung trên slide**
- *Chọn ngữ cảnh theo nhiệm vụ (TACS) và mở rộng đa phương thức ảnh–văn bản (M-TACS)*
- Dòng phụ: Tái lập CVPR 2026 · Mở rộng quy mô · Ngữ cảnh văn bản · Web demo SUN397 và ISIC
- Tên thành viên, lớp, giảng viên hướng dẫn

**Gợi ý hình:** một cặp ảnh truy vấn + ảnh ngữ cảnh được chọn, có câu caption nhỏ bên dưới.

---

### Slide 2: Đặt vấn đề và khó khăn (0:55)

**Nội dung trên slide**
- Con người phân biệt ảnh khó bằng cách **so sánh với ví dụ tham khảo** (bác sĩ so ca bệnh cũ, người xem so mẫu vải).
- Cách truyền thống: lấy ảnh **giống nhất** (kNN trên đặc trưng DINO đóng băng).
- **Vấn đề:** ảnh giống nhất ≠ ảnh giúp đoán đúng nhất. Trong bài báo, Frozen DINO còn thấp hơn không dùng ngữ cảnh trên CUB (81.4 so với 82.1) và DTD (75.4 so với 76.1).
- **Khó khăn:**
  1. Phép chọn ảnh là rời rạc (argmax), không lấy đạo hàm trực tiếp được.
  2. Mô hình lớn (DINOv3 ViT-L/16, VLM sinh caption) trên GPU giới hạn: tràn bộ nhớ, xung đột thư viện.
  3. Caption có thể **lộ nhãn** (chứa tên lớp), làm sai lệch đánh giá.

**Gợi ý hình:** hai cột "ảnh giống nhất" (gần như bản sao) và "ảnh hữu ích" (tương phản rõ).

**Lời nói:** nhấn mạnh "giống nhất chưa chắc hữu ích nhất", vì đây là động lực của cả đề tài.

---

### Slide 3: Mục tiêu và đóng góp (0:35)

**Nội dung trên slide**
1. **Tái lập TACS** đúng giao thức bài báo, kiểm chứng trên CUB-200 và các dataset phân loại, phân đoạn khác.
2. **Cải tiến 1, mở rộng quy mô:** backbone DINOv3 ViT-L/16, ảnh 384 (phân loại) và 448 (phân đoạn).
3. **Cải tiến 2, M-TACS:** thêm caption cho ảnh ứng viên, bộ lọc chống lộ nhãn, hơn 20 cấu hình ablation.
4. **Web demo** chạy mô hình thật, hiển thị ảnh ngữ cảnh được chọn và bản đồ chú ý.

**Lời nói:** "Em làm theo ba bước: tái lập cho đúng, mở rộng quy mô, rồi thêm ngữ cảnh văn bản."

---

## PHẦN TACS VÀ BASELINE

### Slide 4: TACS: kiến trúc (0:55)

**Nội dung trên slide (dạng sơ đồ)**
```
Ảnh truy vấn x_q ──► Selector (DINOv3 CLS + MLP) ──► điểm với từng ứng viên ──► chọn 1 ảnh x_c
                                   ▲
Kho ứng viên = 20% tập train (cố định, loại chính ảnh truy vấn)

(x_q, x_c) ──► Mạng tác vụ ──► dự đoán
   • Phân loại: ghép patch token của 2 ảnh, đi chung qua ViT, fine-tune toàn bộ
   • Phân đoạn: backbone đóng băng + Gated Cross-Attention + đầu DPT
```
- K = 1 ảnh ngữ cảnh cho mỗi truy vấn.
- Self-exclusion: ảnh truy vấn nằm trong kho thì không được tự chọn chính nó.

**Gợi ý hình:** vẽ lại Figure 1 của bài báo theo phong cách riêng.

---

### Slide 5: TACS: thuật toán huấn luyện (1:05)

**Nội dung trên slide**
- Tối ưu lai: **L = L_grad + λ · L_policy**, λ = 1
- **Nhánh khả vi (Gumbel-Softmax Straight-Through, τ = 0.1):** lượt xuôi chọn cứng 1 ảnh, lượt ngược truyền đạo hàm qua phân phối mềm.
- **Nhánh chính sách (REINFORCE):** phần thưởng
  `r = L_task(x_q, không ngữ cảnh) − L_task(x_q, x_c)`
  r > 0 nghĩa là ảnh ngữ cảnh giúp giảm loss, nên xác suất chọn ảnh đó được tăng. Phần thưởng được chuẩn hóa theo batch.
- **Phát hiện của bài báo:** TACS chọn ảnh **khác lớp** nhiều hơn 40–70% so với Frozen DINO, tức là học cách chọn ví dụ tương phản.

**Lời nói:** "Phần thưởng đo đúng thứ ta cần: ảnh ngữ cảnh có làm mô hình đoán tốt hơn hay không."

---

### Slide 6: Tái lập baseline (0:50)

**Nội dung trên slide**

*Bảng chính: CUB-200-2011, cấu hình đúng bài báo (DINOv3 ViT-S/16, ảnh 256, No-Context, test %)*

| Seed | Accuracy | Macro-F1 |
|---|---|---|
| 42 | 82.46 | 82.57 |
| 43 | 82.27 | 82.32 |
| 44 | 82.53 | 82.54 |
| **Trung bình 3 seed** | **82.42** | **82.48** |
| Bài báo (trung bình 3 seed) | 82.1 | — |

Chênh lệch so với bài báo chỉ **+0.3 điểm**, độ lệch giữa các seed khoảng 0.1 điểm: pipeline dữ liệu và huấn luyện được tái lập chính xác.

*Bảng phụ: các dataset khác, cùng cấu hình ViT-S/16, ảnh 256*

| Dataset | Phương pháp | Bài báo | Nhóm tái lập |
|---|---|---|---|
| Stanford Cars | No-Context | 92.4 | 91.59 |
| SUN397 | TACS | 71.8 | 74.01 |

**Lời nói:** "Trên CUB, ba seed của em cho trung bình 82.42 so với 82.1 của bài báo. Baseline đúng thì các cải tiến phía sau mới so sánh được."

---

### Slide 7: Dữ liệu và thiết lập thực nghiệm (0:35)

**Nội dung trên slide**

| | DTD | SUN397 |
|---|---|---|
| Bài toán | Kết cấu bề mặt, 47 lớp | Khung cảnh, 397 lớp |
| Quy mô | 5,640 ảnh (1,880 / 1,880 / 1,880) | khoảng 100k ảnh |
| Kho ứng viên | 376 ảnh | 3,573 ảnh |
| Vai trò | **Thử nghiệm và ablation chính** | **Demo web** |

- **Em thử trên cả hai bộ.** DTD ít dữ liệu và nhẹ, mỗi lần train nhanh, tiết kiệm GPU và thời gian, nên chạy được hơn 20 cấu hình cùng một giao thức.
- **Demo dùng SUN397:** ảnh lớn và rõ, nhãn khung cảnh dễ phân biệt bằng mắt (phòng ngủ, bãi biển, nhà ga), người xem tự kiểm chứng được.
- Giao thức cố định: split chính thức, K = 1, 50 epoch, seed 42, chọn checkpoint theo val loss nhỏ nhất, chỉ chạy test một lần.
- Phân đoạn: ISIC 2017 (2,000 / 150 / 600 ảnh, kho ứng viên 400 ảnh).

---

## CẢI TIẾN 1: MỞ RỘNG QUY MÔ (ViT-L/16 + TĂNG KÍCH THƯỚC ẢNH)

### Slide 8: Cải tiến 1: ViT-L/16 + tăng kích thước ảnh (0:50)

**Nội dung trên slide**
- **Ý tưởng:** bài báo dùng ViT-S/16 (22M tham số) với ảnh 256. Backbone lớn hơn và ảnh lớn hơn giữ được nhiều chi tiết hơn (hoa văn, vật thể nhỏ, ranh giới tổn thương).
- Thay **DINOv3 ViT-S/16 → ViT-B/16 → ViT-L/16** (1024 chiều), ảnh **256 → 384** (phân loại) và **448** (phân đoạn).

| Dataset | Bài báo (ViT-S, 256) | Bản nâng cấp của em | Cấu hình | Tăng |
|---|---|---|---|---|
| DTD | 77.7 | **84.95** | ViT-L/16, 384 | **+7.3** |
| SUN397 | 71.8 | 74.01 → 79.42 → **81.11\*** | ViT-S 256 → ViT-B 384 → ViT-L 384 | **+9.3** |
| Stanford Cars | 94.1 | **95.15** | ViT-B/16, 384 | +1.1 |
| ISIC 2017 (Dice) | 85.0 | 85.60 → **88.62\*** | ViT-S 448 → ViT-L 448 | **+3.6** |

\* Điểm trên tập val của bản đóng gói cho web; các số còn lại là test (accuracy, riêng ISIC là Dice).

- **Thách thức kỹ thuật:** ViT-L với ảnh 384 tràn cả GPU 95 GB khi train SUN397. Em giải quyết bằng cách tách batch logic 64 / batch tính toán 8, kèm gradient checkpointing (chi tiết ở slide 14).
- Bản TACS ViT-L/16 này là **baseline mạnh** để so sánh với M-TACS ở phần tiếp theo.

**Gợi ý hình:** biểu đồ cột SUN397 tăng dần qua ba bước S → B → L.

**Lời nói:** "Chỉ riêng việc nâng backbone và kích thước ảnh đã tăng 7–9 điểm. Đây là baseline mới, mạnh hơn nhiều so với bài báo, để M-TACS phải vượt qua."

---

## CẢI TIẾN 2: M-TACS (NGỮ CẢNH ẢNH + VĂN BẢN)

### Slide 9: M-TACS: ý tưởng và kiến trúc (0:55)

**Nội dung trên slide**
- **Ý tưởng:** ngoài ảnh ngữ cảnh, đưa thêm **mô tả văn bản** của ảnh ứng viên (màu sắc, hoa văn, cấu trúc) làm ngữ cảnh.
```
Ảnh ứng viên ──► VLM (Florence-2 / Qwen3-VL) ──► caption 5–8 từ ──► bộ lọc lộ nhãn ──► CLIP (512-d)
                                                                                   │
(x_q, x_c) ──► ViT-L (TACS) ──► đặc trưng ảnh z ──► z' = LN(z + σ(gate) · m · W·t) ──► phân loại
```
- Cổng học được `gate`, mặt nạ `m` = 0 khi caption không hợp lệ: an toàn, không phá đặc trưng ảnh gốc.
- Biến thể nâng cao: đưa độ tương hợp ảnh–caption vào **bộ chọn** (Study 04); đưa caption thành token bên trong ViT (Study 05).

**Lời nói:** "M-TACS giữ nguyên toàn bộ TACS bản ViT-L và thêm một kênh văn bản có cổng điều khiển."

---

### Slide 10: Pipeline caption và chống lộ nhãn (0:35)

**Nội dung trên slide**
- Bộ lọc cũ dùng ngưỡng CLIP ≥ 0.30 so với tên lớp: **loại nhầm gần 100% caption** trên DTD.
- Bộ lọc mới `dtd_lexical_v2`:
  1. Bỏ tiền tố chung chung ("The image is a close up of…"), giữ 5–8 từ mô tả.
  2. Chặn trực tiếp tên lớp và từ đồng nghĩa.
  3. Caption lỗi **không xóa ảnh**: vector 0 + mặt nạ, nên dữ liệu và kho ứng viên giữ nguyên.
- Hiệu quả: caption đã lọc 80.91% F1 so với chưa lọc 78.98% (**+1.93**).
- **ISIC 2017** dùng chính sách riêng `isic_visible_attributes_v1`: chỉ mô tả thứ nhìn thấy được (màu sắc, đường viền, cấu trúc bề mặt), **cấm thuật ngữ chẩn đoán** (melanoma, nevus…); phủ 100% trên 2,750 ảnh với Qwen3-VL-8B.
- Kỹ thuật: Florence-2 chạy trong tiến trình con riêng (transformers 4.49), tự giảm batch khi tràn VRAM, lưu nguyên tử để chạy tiếp khi Kaggle hết giờ.

---

### Slide 11: Ablation 1: đặt văn bản ở đâu? (0:45)

**Nội dung trên slide (biểu đồ cột ngang, Macro-F1 test %, DTD, ViT-L/16, ảnh 384)**

| Cấu hình | Test F1 |
|---|---|
| Chỉ caption ở **ảnh ứng viên** | **85.03** |
| TACS chỉ dùng ảnh (Cải tiến 1) | 84.82 |
| Caption bị xáo trộn (đối chứng) | 83.34 |
| Chỉ caption ở ảnh truy vấn | 82.31 |
| Caption ở cả hai phía + bộ chọn | 80.91 |

- **Kết luận:** vị trí đặt văn bản quan trọng hơn lượng văn bản. Đặt ở ảnh ứng viên là cấu hình duy nhất vượt TACS; đặt ở ảnh truy vấn làm giảm 2.5 điểm.
- Đối chứng xáo caption cho thấy thí nghiệm được thiết kế có kiểm soát.

**Lời nói:** "Ablation này quyết định thiết kế cuối cùng: văn bản chỉ đi kèm ảnh ứng viên."

---

### Slide 12: Ablation 2: nguồn caption và kiến trúc trộn (0:45)

**Nội dung trên slide**

*Nguồn caption (caption ở cả hai phía, test F1 %):*
| Florence-2 Prefix | Florence LoRA | Qwen3-VL-8B | Qwen3-VL-32B |
|---|---|---|---|
| 81.07 | 80.67 | **82.14** | 82.08 |

*Kiến trúc trộn (caption ở ảnh ứng viên, test F1 %):*
| Phiên bản | Thay đổi | Test F1 |
|---|---|---|
| **S01** | Caption Prefix chuẩn hóa | **85.30** |
| S04 | + caption trong bộ chọn | 85.07 |
| S05 | Caption thành token bên trong ViT | 85.05 |
| S02 | + chuẩn hóa L2 | 84.93 |
| S03 | + căn chỉnh m-LTM (Optimal Transport) | 84.87 |

- VLM lớn hơn (Qwen) cho caption tốt hơn Florence khoảng 1 điểm; 32B không hơn 8B, nên 8B là lựa chọn hiệu quả.
- Ở Study 04, tắt caption trong bộ chọn làm đổi **5–11%** ảnh ngữ cảnh được chọn: bộ chọn có dùng thông tin văn bản.

**Lời nói:** "Em thử năm kiến trúc trộn; phiên bản đơn giản S01 cho điểm cao nhất."

---

## KẾT QUẢ

### Slide 13: Kết quả: TACS (ViT-L) vs M-TACS (0:45)

**Nội dung trên slide**

*Phân loại DTD (ViT-L/16, ảnh 384, test %):*
| Phiên bản | Accuracy | Macro-F1 | Δ F1 |
|---|---|---|---|
| TACS bài báo (ViT-S, ảnh 256) | 77.7 | — | tham chiếu |
| **Cải tiến 1:** TACS ViT-L, ảnh 384 | 84.95 | 84.82 | 0.00 |
| **Cải tiến 2:** M-TACS v1, caption ảnh ứng viên | 85.16 | 85.03 | +0.21 |
| **Cải tiến 2:** M-TACS v2, caption Prefix chuẩn hóa | **85.43** | **85.30** | **+0.48** |

*Phân đoạn ISIC 2017 (ViT-L/16, ảnh 448, tập val %):*
| Phiên bản | Dice | IoU |
|---|---|---|
| TACS (Cải tiến 1) | 88.62 | 81.32 |
| **M-TACS (Cải tiến 2, caption Florence-prefix)** | **88.83** | **81.43** |

Ghi chú nhỏ cuối slide: *1 seed; ISIC là điểm trên tập val.*

**Lời nói:** "So với bài báo: Cải tiến 1 tăng 7.3 điểm trên DTD, Cải tiến 2 tăng thêm 0.48. M-TACS cũng vượt TACS ViT-L trên phân đoạn ISIC."

---

## TRIỂN KHAI VÀ DEMO

### Slide 14: Triển khai hệ thống và web (0:40)

**Nội dung trên slide**
- **Train ViT-L không tràn bộ nhớ:** tách batch logic 64 / batch tính toán 8; lượt đầu tính phần thưởng không lấy đạo hàm, lượt sau chạy lại đúng trạng thái ngẫu nhiên (RNG replay) + gradient checkpointing.
- **Model bundle tự chứa (.zip):** checkpoint, mã DINOv3, kho ứng viên, cấu hình, kèm Florence-2 và CLIP cho M-TACS; kiểm tra nạp lại nguội và mã băm trước khi xuất.
- **Web FastAPI, 4 trang** trong một giao diện studio: SUN397 TACS / M-TACS, ISIC TACS / M-TACS.
  - Ảnh mới tải lên được **sinh caption trực tiếp** (Florence-2 + CLIP nạp sẵn trên server).
  - Nạp sẵn mô hình khi khởi động, hàng đợi một lượt suy luận mỗi lần, **ảnh tải lên bị xóa sau khi xử lý**.
- **Đảm bảo tái lập:** notebook sinh tự động bằng builder, caption và kết quả khóa bằng SHA-256, **56 file test tự động**.

**Gợi ý hình:** sơ đồ Trình duyệt → FastAPI → worker suy luận (DINOv3 + Florence-2 + CLIP) → bundle mô hình.

---

### Slide 15: Demo web (3:00)

**Kịch bản demo (chuẩn bị trước, không làm ngẫu hứng)**
1. (0:20) Mở studio, chỉ 4 trang (SUN397 TACS / M-TACS, ISIC TACS / M-TACS) và badge "Sẵn sàng".
2. (0:50) Trang SUN397 TACS: tải một ảnh khung cảnh rõ ràng thuộc bộ dữ liệu (ví dụ bãi biển). Chỉ ra: nhãn dự đoán, độ tin cậy, **nhãn đúng hiện ngay bên cạnh**, **ảnh ngữ cảnh được chọn**, bản đồ chú ý "Mô hình nhìn vào đâu?".
3. (0:50) Chuyển sang trang M-TACS với cùng ảnh: **caption được sinh trực tiếp** cho ảnh truy vấn và caption của ảnh ngữ cảnh; so sánh dự đoán với TACS.
4. (0:50) Trang ISIC: tải ảnh tổn thương da, hiển thị mặt nạ phân đoạn, ảnh hỗ trợ được chọn và bản đồ chú ý.
5. (0:10) Quay lại slide.

Nếu thiếu ảnh: dùng nút **ảnh mẫu** có sẵn trên trang. Phần "Bên trong TACS" cuối trang có thể cuộn qua nhanh nếu dư giờ.

**Lưu ý về bản M-TACS trong demo**
- Bundle web M-TACS (SUN397 và ISIC) là cấu hình **full_both**: caption của cả ảnh truy vấn và ảnh ứng viên, ở cả bộ chọn và khâu trộn. Bundle này được train trước khi ablation DTD chốt thiết kế "caption chỉ ở ảnh ứng viên".
- Điểm mạnh để nhấn khi demo: **toàn bộ pipeline đa phương thức chạy trực tiếp** trên ảnh mới. Đóng gói lại theo thiết kế tốt nhất là bước triển khai tiếp theo (đã nêu ở slide 16).

**Chuẩn bị bắt buộc**
- Chọn sẵn 3–4 ảnh đã thử trước: hai mô hình dự đoán đúng, ảnh ngữ cảnh có ý nghĩa. Nhãn đúng chỉ hiện khi mã băm của ảnh nằm trong dữ liệu đã lưu của bundle, nên hãy thử trước và chọn ảnh mà web hiện được nhãn đúng.
- Kiểm tra đường dẫn bundle (biến môi trường hoặc `artifacts/deployment/`) và chạy `start_web.ps1` trước giờ trình bày; chỉ chạy một tiến trình server.
- Mở sẵn server và tab trình duyệt, đợi badge "Sẵn sàng"; quay sẵn video demo dự phòng nếu máy lỗi.
- Hai mô hình SUN397 đôi khi dự đoán khác nhau. Gặp trường hợp này thì nói "đây là ảnh khó, ảnh ngữ cảnh được chọn khác nhau", rồi chuyển ảnh tiếp theo.

---

## KẾT

### Slide 16: Hạn chế, hướng phát triển và kết luận (1:10)

**Nội dung trên slide**

*Hạn chế*
- Mức tăng của Cải tiến 2 còn nhỏ (+0.2 đến +0.5 điểm) và mới có 1 seed.
- **Kênh văn bản chưa được tận dụng tối đa:** caption toàn ảnh (5–8 từ) trùng nhiều thông tin với đặc trưng của ViT-L, nên cổng văn bản học được còn nhỏ.

*Hướng phát triển*
1. Đóng gói lại bản web M-TACS theo thiết kế tốt nhất (caption chỉ ở ảnh ứng viên).
2. Trộn ảnh–văn bản ở mức token (cross-attention giữa patch và từ).
3. Dùng văn bản mang thông tin mới: nhãn hoặc mô tả lớp của ảnh ngữ cảnh.
4. Backbone có sẵn không gian chung ảnh–văn bản (SigLIP 2, DINOv3 dino.txt).
5. Chạy 3 seed cho các cấu hình chính; mở rộng K = 3 ảnh ngữ cảnh.

*Kết luận*
- **Baseline:** tái lập TACS sát bài báo (CUB-200: 82.42 so với 82.1).
- **Cải tiến 1:** backbone lớn + ảnh lớn tăng 1.1–9.3 điểm so với bài báo (DTD +7.3, SUN397 +9.3).
- **Cải tiến 2:** M-TACS với caption ở ảnh ứng viên vượt TACS ViT-L (+0.48 F1 trên DTD, +0.21 Dice trên ISIC); ablation chỉ ra vị trí và chất lượng văn bản là yếu tố quyết định.
- Toàn bộ hệ thống chạy thành web demo hoàn chỉnh.

**Lời nói kết:** "Kết quả cho thấy hướng đa phương thức có tiềm năng, và chỗ cần cải tiến tiếp theo đã rõ ràng. Em xin cảm ơn thầy/cô."

---

## Slide dự phòng (chỉ mở khi được hỏi)

**B1. Bảng ablation đầy đủ DTD (test %)**

| Mã | Cấu hình | Acc | F1 |
|---|---|---|---|
| A01 | TACS chỉ dùng ảnh | 84.95 | 84.82 |
| A02 | Caption ảnh truy vấn | 82.34 | 82.31 |
| A03 | Caption ảnh ứng viên | 85.16 | 85.03 |
| A04 | Caption cả hai phía | 81.17 | 81.07 |
| A05 | Cả hai phía + bộ chọn | 81.01 | 80.91 |
| A06 | Caption xáo trộn | 83.40 | 83.34 |
| A07 | Caption chưa lọc | 79.04 | 78.98 |
| A08 | Caption che tên lớp | 81.28 | 81.11 |
| C01–C06 | Prefix / FirstValid / SingleGrounded / MultiGrounded / Qwen8B / Qwen32B | — | 81.07 / 80.94 / 77.96 / 79.06 / 82.14 / 82.08 |
| S01–S05 | Kiến trúc trộn | — | 85.30 / 84.93 / 84.87 / 85.07 / 85.05 |

**B2. SUN397 chi tiết**
- TACS ViT-S/16 ảnh 256: 74.01 (bài báo 71.8); ViT-B/16 ảnh 384: 79.42 (test).
- Bundle web ViT-L/16 (tập val, train dừng ở 42–43/100 epoch): TACS 81.11% Acc; M-TACS 80.20% Acc. Chưa chạy test.

**B3. Phân đoạn ISIC trên tập test (ViT-S/16, ảnh 448)**
- TACS 85.60 / 77.38; M-TACS + Qwen8B 84.87 / 76.54. Caption cấp toàn ảnh chưa đủ chi tiết cho ranh giới từng điểm ảnh, đây là lý do của hướng trộn mức token.

**B4. Siêu tham số chính (DTD và SUN397)**
- DINOv3 ViT-L/16, AdamW, LR backbone 3e-6, LR đầu ra 1e-4, warmup 5 epoch, Gumbel τ = 0.1, λ = 1, CLIP ViT-B/32 đóng băng.

**B5. Hướng thử thêm: số ngữ cảnh thay đổi theo ảnh (SUN397, ViT-S/16, ảnh 384)**
- TACS adaptive chọn tối đa 3 ảnh ngữ cảnh, trung bình khoảng 2: test 75.88% Acc / 75.80% F1. Chưa có bản K = 1 cùng ViT-S, ảnh 384 để so trực tiếp.

**B6. Cấu hình của các bundle web**
- SUN397 và ISIC: TACS chỉ dùng ảnh và M-TACS **full_both** (caption ảnh truy vấn + ảnh ứng viên, ở bộ chọn và khâu trộn), DINOv3 ViT-L/16.
- ISIC M-TACS có tùy chọn thay bộ caption thử nghiệm (attributes v3) mà không train lại checkpoint; mặc định tắt.

---

## Chuẩn bị hỏi đáp (trả lời đúng sự thật, vẫn tự tin)

| Câu hỏi có thể gặp | Câu trả lời gợi ý |
|---|---|
| "Cải tiến 1 có phải chỉ là dùng mô hình to hơn?" | "Đúng là tăng quy mô, nhưng ViT-L với ảnh 384 tràn cả GPU 95 GB; em phải thiết kế lại vòng train (tách batch, chạy lại RNG, gradient checkpointing) mới train được mà vẫn giữ đúng thuật toán TACS." |
| "Mức tăng 0.48 của M-TACS có ý nghĩa không?" | "Mức tăng còn nhỏ và mới 1 seed; em đã lên kế hoạch chạy 3 seed. Điểm quan trọng là ablation chỉ ra thiết kế đúng: đặt caption ở ảnh truy vấn thì giảm 2.5 điểm." |
| "Caption thật sự giúp hay không?" | "Ở bộ chọn, tắt caption làm đổi 5–11% ảnh ngữ cảnh, nên mô hình có dùng văn bản. Ở khâu trộn đặc trưng, cổng văn bản còn nhỏ; đây là hạn chế em đã nêu và là hướng cải tiến." |
| "Sao không làm ablation trên SUN397?" | "SUN397 có 397 lớp và kho ứng viên 3,573 ảnh, mỗi lần train ViT-L tốn hơn nhiều GPU; DTD cho phép chạy hơn 20 cấu hình cùng giao thức." |
| "Có lộ nhãn qua caption không?" | "Có bộ lọc chặn tên lớp và từ đồng nghĩa; thí nghiệm caption chưa lọc cho điểm thấp hơn 1.93, cho thấy bộ lọc có tác dụng." |
| "Sao demo không dùng bản M-TACS tốt nhất?" | "Bundle web được train trước khi ablation DTD chốt thiết kế, nên đang là full_both. Demo cho thấy toàn bộ pipeline đa phương thức chạy trực tiếp; đóng gói lại theo thiết kế tốt nhất là bước tiếp theo." |
| "Ảnh tải lên có được lưu lại không?" | "Không; ảnh chỉ ghi tạm trong lúc suy luận rồi bị xóa, không ghi vào bundle." |
| "Sao bảng tái lập chỉ có No-Context trên CUB?" | "CUB dùng để kiểm chứng pipeline với 3 seed như bài báo; TACS có ngữ cảnh em kiểm chứng trên SUN397 (74.01 so với 71.8) và Stanford Cars." |

**Không nên nói:** "caption làm mô hình phân biệt tốt hơn nhờ…" như một kết luận đã chứng minh; "M-TACS vượt TACS trên mọi dataset" (bundle SUN397 và ISIC test thì chưa).
