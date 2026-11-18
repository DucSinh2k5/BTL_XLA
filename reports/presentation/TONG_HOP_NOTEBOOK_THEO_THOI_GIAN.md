# Tổng hợp các lượt chạy theo thời gian: TACS → M-TACS

> 46 lượt chạy có output, sắp theo thời gian (11/09 – 13/11/2026). Ngày lấy từ dấu thời gian trong output hoặc metadata notebook; lượt train bản web lấy theo ngày xuất bundle.
> Kết quả là điểm **test** (Accuracy / Macro-F1, %), trừ các ô ghi *val*. Viết tắt: NC = No-Context, ep = epoch.

## Giai đoạn 0: Tái lập baseline TACS (11/09 – 27/09)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 1 | 11/09 | CUB-200 NC, seed 42 | ViT-S/16, ảnh 256, 100 ep, không ngữ cảnh | 82.46 / 82.57 | Tái lập bảng 2 bài báo (82.1) |
| 2 | 11/09 | CUB-200 NC, seed 43 | Như trên | 82.27 / 82.32 | Có thêm 1 lượt chạy lặp lại cho kết quả y hệt |
| 3 | 11/09 | CUB-200 NC, seed 44 | Như trên | 82.53 / 82.54 | Trung bình 3 seed: **82.42** |
| 4 | 12/09 | Cars NC, seed 44 | ViT-S/16, 256, train toàn bộ tập train | 91.59 / 91.43 | Bài báo 92.4 |
| 5 | 12/09 | Cars Random Context (bản đầu) | ViT-S/16, 256, pool 20% | 1.79 / 0.61 | **Lỗi**, bỏ không dùng |
| 6 | 13/09 | Cars TACS, trước khi sửa bộ chọn | ViT-S/16, 256, seed 43 | 86.30 / 86.17 | Bản lỗi, lưu trong archive |
| 7 | 14/09 | Cars TACS, trước khi tách val | ViT-S/16, 256 | 77.73 / 77.55 | Bản lỗi (rơi về trộn đặc trưng), lưu trong archive |
| 8 | 24/09 | Cars Frozen DINO | ViT-S/16, ảnh 448, K = 4, pool 100% | 93.60 / 93.45 | Khác chuẩn bài báo (K, pool) |
| 9 | 24/09 | Cars Random Context | ViT-S/16, ảnh 448, K = 1 | 94.38 / 94.29 | Bản đã sửa lỗi của #5 |
| 10 | 24/09 | Cars TACS shortlist-4 | ViT-S/16, ảnh 448, K = 1 | 94.98 / 94.90 | Bài báo 94.1 |
| 11 | 25/09 | SUN397 TACS, trước khi sửa bộ chọn theo bài báo | ViT-S/16, ảnh 448 | 76.97 / 76.84 | Bộ chọn chưa đúng bài báo, lưu trong archive |
| 12 | 26/09 | SUN397 TACS (bản 1) | ViT-S/16, 256, val 10% | 75.31 / 75.09 | Bản đầu, chưa dùng trong báo cáo |
| 13 | 27/09 | **SUN397 TACS (bản 2)** | ViT-S/16, 256, tách val chuẩn, dừng sớm ở ep 23 | **74.01 / 73.85** | Bản dùng trong báo cáo (bài báo 71.8) |
| 14 | 27/09 | **ISIC TACS (I01)** | ViT-S/16 đóng băng, ảnh 448, 2 khối cross-attention + DPT | **Dice 85.60 / IoU 77.38** | Dừng sớm ở ep 64, best ep 49 (bài báo 85.0 / 76.5) |

## Giai đoạn 1: Cải tiến 1, backbone lớn + ảnh lớn (26/09 – 08/10)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 15 | 26/09 – 08/10 | Cars TACS | ViT-B/16, ảnh 384 | **95.15 / 95.03** | +1.1 so với bài báo |
| 16 | 26/09 – 08/10 | SUN397 TACS | ViT-B/16, ảnh 384 | **79.42 / 79.27** | +5.4 so với bản ViT-S |

## Giai đoạn 2: Thử M-TACS đầu tiên và ngữ cảnh thích ứng trên SUN397 (06/10 – 07/10)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 17 | 06/10 | SUN397 M-TACS caption ảnh truy vấn | ViT-S/16, ảnh 384, caption Florence (60.55% hợp lệ) | 76.12 / 76.02 | Bản M-TACS đầu tiên |
| 18 | 07/10 | SUN397 TACS adaptive | ViT-S/16, ảnh 384, tối đa 3 ngữ cảnh (trung bình ~2) | 75.88 / 75.80 | Thử số ngữ cảnh thay đổi theo ảnh |

## Giai đoạn 3: Chuyển sang DTD, ablation vị trí văn bản (15/10 – 17/10)

Từ đây, mọi dòng DTD dùng chung: split 1, ViT-L/16, ảnh 384, K = 1, 50 ep, seed 42.

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 19 | 15/10 | Bộ ablation gộp (bản trước 50 ep) | Kiểm tra rò rỉ caption | Bộ phân loại chỉ dùng caption: đã lọc **2.13%** (≈ đoán ngẫu nhiên 1/47), chưa lọc 14.52% | Bộ lọc xóa thông tin nhãn; sau đó tách thành 8 notebook riêng |
| 20 | 16/10 | A01 visual_only | Chỉ dùng ảnh, có ảnh ngữ cảnh | 84.95 / **84.82** | **Baseline TACS ViT-L** |
| 21 | 16/10 | A02 query_only | Caption ảnh truy vấn | 82.34 / 82.31 | −2.51 |
| 22 | 17/10 | A04 both_fusion | Caption cả hai phía | 81.17 / 81.07 | −3.75 |
| 23 | 17/10 | A03 candidate_only | Caption ảnh ứng viên | 85.16 / **85.03** | **+0.21** |
| 24 | 17/10 | A05 full_both | Cả hai phía + bộ chọn | 81.01 / 80.91 | −3.91 |
| 25 | 17/10 | A06 full_shuffle_both | Như A05, caption bị xáo trộn | 83.40 / 83.34 | Đối chứng |
| 26 | 17/10 | A07 full_unscreened | Như A05, caption chưa lọc | 79.04 / 78.98 | −5.84 |
| 27 | 17/10 | A08 full_redacted | Như A05, caption che tên lớp | 81.28 / 81.11 | −3.71 |

## Giai đoạn 4: Nguồn caption và mở rộng sang ISIC (18/10 – 24/10)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 28 | 18/10 | LoRA | Fine-tune decoder Florence trên 188 caption mẫu, full_both | 80.74 / 80.67 | |
| 29 | 21/10 | C01 Florence-Prefix | full_both, caption chuẩn hóa | 81.12 / 81.07 | |
| 30 | 21/10 | C02 Florence-FirstValid | Phủ 100% | 81.06 / 80.94 | |
| 31 | 21/10 | C03 SingleGrounded | CLIP chọn lại câu | 78.19 / 77.96 | |
| 32 | 22/10 | C04 MultiGrounded | CLIP + 3 beam | 79.31 / 79.06 | |
| 33 | 23/10 | Sinh caption Qwen3-VL 8B và 32B | Chạy riêng cho DTD | 74.66% / 72.89% caption hợp lệ | Giai đoạn chuẩn bị dữ liệu |
| 34 | 23/10 | C05 Qwen3-VL-8B | full_both | 82.29 / **82.14** | Tốt nhất trong full_both |
| 35 | 23/10 | C06 Qwen3-VL-32B | full_both | 82.18 / 82.08 | |
| 36 | 24/10 | **ISIC M-TACS (I02)** | ViT-S/16, ảnh 448, full_both, caption Qwen-8B (phủ 100%) | Dice 84.87 / IoU 76.54 | Dừng sớm ở ep 41; thấp hơn I01 |

## Giai đoạn 5: Kiến trúc trộn trên DTD (25/10 – 26/10)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 37 | 25/10 | **S01** | candidate_only, caption Prefix khóa SHA-256 | 85.43 / **85.30** | **Tốt nhất toàn dự án** |
| 38 | 26/10 | S02 | + chuẩn hóa L2 | 85.05 / 84.93 | |
| 39 | 26/10 | S03 | + căn chỉnh m-LTM (Optimal Transport) | 84.95 / 84.87 | |
| 40 | 26/10 | S04 | + caption vào bộ chọn (α ≈ 0.131) | 85.16 / 85.07 | Tắt caption làm đổi 5–11% lựa chọn |

## Giai đoạn 6: Train và đóng gói bản web ViT-L (31/10 – 02/11)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 41 | 31/10 | SUN397 TACS web | ViT-L/16, 384, batch 64/8 + RNG replay | *val* 81.11 / 80.82 | Dừng ở ep 43/100 do hết giờ |
| 42 | 31/10 | SUN397 M-TACS full_both web (+ xuất bundle) | ViT-L/16, 384 | *val* 80.20 / 79.84 | Dừng ở ep 42/100 |
| 43 | 01/11 | ISIC M-TACS full_both web | ViT-L/16, 448, caption Florence-prefix | *val* Dice 88.83 / IoU 81.43 | Dừng ở ep 93/100 |
| 44 | 02/11 | ISIC TACS web | ViT-L/16, 448 | *val* Dice 88.62 / IoU 81.32 | Dừng ở ep 94/100 |
| 45 | 02/11 | Xuất bundle ISIC M-TACS | Chỉ đóng gói, không train | Smoke test đạt | |

## Giai đoạn 7: Thử nghiệm mới nhất (13/11)

| # | Ngày | Lượt chạy | Cấu hình | Kết quả | Ghi chú |
|---|---|---|---|---|---|
| 46 | 13/11 | S05 text token | Caption thành token trong ViT, DTD | 85.16 / 85.05 | Token học được rất nhỏ |

## Ghi chú

- **SUN397 ViT-S/256 có hai lượt:** bản 1 đạt 75.31 (26/09), bản 2 đạt 74.01 (27/09). Báo cáo dùng bản 2 vì có tách val chuẩn và dừng sớm theo val.
- **Lượt #19:** bộ phân loại chỉ dùng caption đã lọc đạt 2.13%, bằng mức đoán ngẫu nhiên với 47 lớp. Đây là bằng chứng caption không lộ nhãn, đồng thời giải thích vì sao caption khó làm tăng điểm phân loại.
- **Khối lượng:** 14 lượt tái lập baseline (5 bản lỗi hoặc bản thử), 2 lượt Cải tiến 1 với ViT-B (cộng 3 lượt ViT-L ở giai đoạn web), 2 lượt thử trên SUN397, 21 cấu hình DTD (20 ablation + 1 kiểm tra rò rỉ), 1 lượt sinh caption Qwen, 2 lượt ISIC ViT-S, 5 lượt train hoặc xuất bundle web.
- Mọi lượt chỉ có 1 seed (trừ CUB). Ngày của lượt train web là ngày xuất bundle, không phải ngày bắt đầu train.
