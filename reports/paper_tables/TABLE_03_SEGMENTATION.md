# Table 3: Medical Segmentation

Cập nhật 25/10/2026 từ output của [visual TACS 448](../../tacs_isic_segement_448.ipynb) và [Qwen8B full_both segmentation](../../isic-segmen-captions.ipynb). Không sửa hoặc thực thi lại notebook. Bằng chứng visual: [table_03_segmentation_evidence.json](table_03_segmentation_evidence.json). So sánh caption: [bảng và nhận xét chi tiết](SO_SANH_ISIC_VISUAL_VA_QWEN8B.md), [JSON evidence](isic_visual_vs_qwen8b_comparison_evidence.json).

## 1. Điền vào hàng/cột nào?

Run này tương ứng **nhóm hàng TACS, dataset ISIC 2017, hai cột Dice và IoU**. Nó không thuộc classification Table 2, cũng không phải thí nghiệm caption/M-TACS DTD.

Tên báo cáo chính xác: **TACS tái triển khai điều chỉnh (448, final-token cross-attention + DPT-lite)**. Không gắn nhãn là chạy lại nguyên bản code/protocol của tác giả.

## 2. Table 3 của project: chỉ điền số đã đo

Dice/IoU theo phần trăm; dấu ↑ nghĩa là cao hơn tốt hơn. “Chưa có” là chưa có kết quả đã xác minh trong phạm vi bảng này, không phải điểm bằng 0.

| Phương pháp | DRIVE Dice ↑ | DRIVE IoU ↑ | ISIC 2017 Dice ↑ | ISIC 2017 IoU ↑ | Kvasir-SEG Dice ↑ | Kvasir-SEG IoU ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No-Context | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |
| Random Context Retrieval | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |
| Frozen DINO Similarity Retrieval | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |
| Feature-Averaged Context | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |
| TACS tái triển khai điều chỉnh, 448* | Chưa có | Chưa có | **85.60** | **77.38** | Chưa có | Chưa có |
| M-TACS full_both + Qwen3-VL-8B, 448 (mở rộng) | Chưa có | Chưa có | **84.87** | **76.54** | Chưa có | Chưa có |

*Hàng visual TACS: single-seed 42, ảnh 448×448, chỉ dùng ảnh context, không dùng caption. Hàng M-TACS là phần mở rộng dùng text query/candidate trong fusion và selector, không phải TACS nguyên bản của paper. Chữ đậm đánh dấu ô có kết quả, không phải khẳng định thắng các baseline chưa chạy. Khi chạy baseline, phải dùng cùng split, resolution, decoder, augmentation, budget/checkpoint rule và metric với run TACS nếu muốn quy khác biệt cho selector.*

Nếu chỉ báo cáo một dataset segmentation, có thể rút bảng xuống hai cột ISIC 2017; không bắt buộc train DRIVE/Kvasir chỉ để lấp đầy bảng.

## 3. So với số công bố trong paper

Đã đối chiếu Table 3 trên trang PDF 6 và implementation details trên trang PDF 5 của [paper local](<../../docs/paper/18. Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.pdf>).

| Nguồn | Cấu hình | ISIC 2017 Dice (%) | ISIC 2017 IoU (%) |
| --- | --- | ---: | ---: |
| TACS trong paper, reference | ViT-S/16, 256×256, trung bình ba seed | 85.00 | 76.50 |
| TACS của project, notebook đã chạy | ViT-S/16, 448×448, final-token cross-attention + DPT-lite, seed 42 | 85.60 | 77.38 |
| Chênh lệch số hiển thị, điểm phần trăm | Khác protocol, không phải đối chứng cùng điều kiện | +0.60 | +0.88 |

Số hiện có cao hơn reference về mặt số học, **chưa đủ kết luận vượt paper**. Resolution, vị trí fusion/head, số seed và thời điểm dừng train khác nhau; chưa xác minh độc lập hash split/patient IDs và artifact đánh giá của hai bên.

## 4. Kết quả và protocol đã đọc

| Hạng mục | Bằng chứng trong notebook |
| --- | --- |
| Dataset | ISIC 2017 segmentation; runtime train/val/test = 2.000/150/600 cặp image-mask |
| Backbone | `dinov3_vits16`, frozen; feature dimension 384 |
| Input thực tế | Tensor mẫu `(3, 448, 448)`; candidate patch bank `(400, 784, 384)` |
| Candidate pool | 400/2.000 ảnh train, 20%, cố định; chọn một context mỗi query |
| Cơ chế | Learned selector; straight-through Gumbel-Softmax, tau 0.1; policy reward dựa vào giảm BCE+Dice loss |
| Fusion/head | Hai gated cross-attention blocks sau final patch tokens; decoder `DPTLiteHead` |
| Train | Seed 42 khai báo trong source; batch 8; AdamW LR 1e-4; warmup 5 + cosine; epoch limit 100 |
| Checkpoint | Chọn validation hard Dice cao nhất; patience 15; dừng ở epoch 64; best epoch 49 |
| Validation tại epoch 49 | Dice 0.8731; IoU 0.7959; loss 0.3688, đều là số log làm tròn |
| Test checkpoint thực sự được load | `/kaggle/working/outputs/tacs_isic2017_segmentation/best_val_checkpoint.pt` |
| Test đã in | `test_dice: 0.8560 (85.60%)`; `test_iou: 0.7738 (77.38%)` |
| Metric | Hard mask từ sigmoid > 0.5; tính Dice/IoU foreground cho từng ảnh rồi trung bình qua test; mask/prediction ở resolution 448 |

Không lấy validation Dice 87.31% thay cho test Dice 85.60%. Không lấy năm ảnh trong `test_rows.head()` để tính trung bình toàn bộ 600 ảnh. Không suy IoU trung bình từ Dice trung bình bằng công thức một ảnh. Test loss và soft Dice/IoU không được in đầy đủ trong output, nên không tự điền thêm số.

## 5. Giới hạn và phần cần bổ sung

- Paper dùng gated cross-attention ở mỗi transformer block và lightweight DPT; notebook này áp dụng sau final tokens với DPT-lite. Đây là khác biệt source đã ghi rõ, không chỉ khác tên file.
- Notebook đã có 64 dòng log epoch liên tiếp, test load best checkpoint và không có saved error output; không gọi đây là hoàn thành đủ 100 epoch vì đã early-stop.
- Source có self-exclusion cho query nằm trong pool; chưa xác minh loại các ảnh liên quan cùng bệnh nhân như yêu cầu paper. Số lượng split khớp mô tả dataset không thay thế kiểm tra danh sách IDs/hash.
- Chưa có `metrics.json`, `training_history.csv`, `test_predictions.csv`, split/pool CSV và checkpoint độc lập ở local trong audit này. Các dòng “Saved ...” là đường dẫn export trên Kaggle, không chứng minh artifact đã được chuyển vào project.
- Năm prediction mẫu đều chọn cùng một context; chỉ là quan sát trên năm ảnh, chưa chứng minh selector chọn cùng ảnh cho toàn bộ test. Cần full prediction CSV để kiểm tra phân bố retrieval.
- Với báo cáo segmentation, ưu tiên thêm **No-Context và Frozen DINO Similarity** cùng protocol 448 để có đối chứng nội bộ; thêm seed 43/44 nếu muốn báo cáo mean/std. Random/Feature-Averaged bổ sung khi budget cho phép. Không điền số tác giả vào các baseline project đang thiếu.

## 6. Đoạn nhận xét cho báo cáo

Đã nhận [output Qwen3-VL-8B full_both](../../isic-segmen-captions.ipynb): test Dice 84.8689%, IoU 76.5407%, thấp hơn visual khoảng 0.73/0.84 điểm phần trăm. Best validation Dice 87.55% tại epoch 26; early-stop ở epoch 41/100. Xem [so sánh đầy đủ](SO_SANH_ISIC_VISUAL_VA_QWEN8B.md). [Template tạo/resume caption và train](../../notebooks/02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/README.md) vẫn là file chưa có output, không ghi đè bằng notebook đã chạy.

> Trên ISIC 2017, bản tái triển khai TACS dùng frozen DINOv3 ViT-S/16, gated cross-attention trên final patch tokens và DPT-lite đạt Dice 85.60% và IoU 77.38% trên 600 ảnh test. Mô hình được chọn theo validation Dice, với checkpoint tốt nhất ở epoch 49 và early stopping tại epoch 64. Kết quả cho thấy pipeline segmentation đã tạo được dự đoán và metric test, nhưng chưa đủ để quy lợi ích cho task-aligned retrieval do thiếu baseline cùng protocol. Mặc dù số hiển thị cao hơn reference TACS trong paper 0.60 điểm Dice và 0.88 điểm IoU, khác biệt resolution, kiến trúc fusion/head và số seed khiến đây chưa phải bằng chứng vượt phương pháp gốc.
