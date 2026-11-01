# ISIC 2017: Visual TACS So Với Qwen8B Full-Both

Ngày đọc output: 25/10/2026. Nguồn: [visual 448](../../tacs_isic_segement_448.ipynb) và [segmentation dùng caption](../../isic-segmen-captions.ipynb). Không sửa, đổi tên, di chuyển hoặc thực thi lại hai notebook. Bằng chứng máy đọc được: [isic_visual_vs_qwen8b_comparison_evidence.json](isic_visual_vs_qwen8b_comparison_evidence.json).

## 1. Kết quả chính

| Cấu hình | Test Dice (%) | Test IoU (%) | Best val Dice (%) | Best epoch | Epoch đã chạy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual TACS variant, không caption | **85.60** | **77.38** | 87.31 | 49 | 64 |
| M-TACS Qwen3-VL-8B `full_both` | 84.8689 | 76.5407 | **87.55** | 26 | 41 |
| Chênh lệch caption so với visual, điểm phần trăm, xấp xỉ | **-0.7311** | **-0.8393** | +0.24 | Không áp dụng | Không áp dụng |

**Kết luận trực tiếp: bản có caption chưa cải thiện segmentation; trong cặp run này, test Dice và IoU đều thấp hơn visual.** Mức giảm khoảng 0.73/0.84 điểm phần trăm, không phải giảm 0.73/0.84 trên thang 0-1.

Visual stdout in `test_dice: 0.8560`, `test_iou: 0.7738`; caption stdout in `0.848689`, `0.765407`. Bảng comparison trong notebook mới dùng giá trị nội bộ trước khi làm tròn và in delta -0.731123/-0.839338 điểm. Sai khác ở các chữ số cuối so với phép trừ stdout ở trên là do độ chính xác hiển thị, không phải hai kết quả test khác nhau. Chưa có hai file `metrics.json` gốc để đối chiếu full precision độc lập.

## 2. Có phải train lỗi hoặc nhánh text chưa hoạt động?

Không thấy saved error output. Log mới có đủ 41 epoch liên tiếp, best validation Dice tại epoch 26 và 15 epoch không cải thiện sau đó. Dừng tại 41/100 là **early stopping theo đúng patience**, không phải mặc nhiên run chưa hoàn tất. Bản visual cũng early-stop tại 64/100, với best epoch 49. Source test của bản mới kiểm tra completed/protocol và nạp best checkpoint; chưa kiểm tra trực tiếp byte checkpoint bên ngoài notebook.

Coverage đọc từ output:

| Vai trò | Số ảnh | Caption qua policy | Tỷ lệ |
| --- | ---: | ---: | ---: |
| Query train | 2.000 | 2.000 | 100% |
| Query validation | 150 | 150 | 100% |
| Query test | 600 | 600 | 100% |
| Candidate pool, là tập con train | 400 | 400 | 100% |

Output worker báo đã sinh đủ 2.750 caption từ trạng thái cached=0 và mã hóa CLIP tensor `(2750, 512)`. Caption CSV SHA-256 được in trong runtime: `d57ac29301254cd2bc2ee369a6f23ff0318a56065f884ebe83e529c7c052ced2`.

Tại **validation epoch tốt nhất 26**, log cho thấy:

| Đại lượng | Giá trị hiển thị |
| --- | ---: |
| Validation Dice | 0.8755 |
| Validation IoU | 0.7974 |
| Query text gate | 0.1315 |
| Candidate text gate | 0.0048 |
| Selector semantic mix | 0.5026 |

Vì vậy không có dấu hiệu toàn bộ caption bị loại khiến text tự biến thành visual-only. Cơ chế gate có giá trị khác 0 và source dùng text ở cả hai nhánh fusion/selector. **Không** diễn giải q_gate=0.1315 là text đóng góp chính xác 13.15% thông tin, hoặc c_gate nhỏ là candidate caption hoàn toàn vô dụng. Đây cũng không phải các gate test đã được in từ checkpoint tốt nhất.

Candidate caption còn tác động qua semantic selector, nên gate fusion nhỏ không đo được toàn bộ ảnh hưởng của nó lên context được chọn.

Coverage 100% chỉ có nghĩa vượt qua policy `isic_visible_attributes_v1`; không có nghĩa caption đúng 100%, được người kiểm duyệt hoặc đã chứng minh không có shortcut/leakage. Chưa được cung cấp CSV này riêng để đọc nội dung từng caption.

## 3. So sánh có cùng protocol không?

Các thiết lập chính đã khớp trong source/runtime:

- ISIC 2017, train/val/test 2.000/150/600; candidate pool 400 ảnh train (20%), K=1.
- Frozen DINOv3 ViT-S/16, input 448×448, hai final-token gated cross-attention blocks và DPT-lite.
- Batch 8, NUM_WORKERS=4, AdamW LR 1e-4, weight decay 1e-4, warmup 5 + cosine.
- Epoch limit 100, patience 15, chọn checkpoint theo validation hard Dice; seed 42 theo source/metadata.
- Loss BCE+Dice, threshold sigmoid >0.5; Dice/IoU foreground được tính từng ảnh rồi trung bình, không lấy soft metric.
- Gumbel tau 0.1, policy weight 1.0 và entropy weight 0.01.

Đối chiếu AST: 22/22 định nghĩa function/class chung giữa visual và caption giống nhau, gồm visual model/decoder/loss/metric helpers. 63/63 định nghĩa trong notebook caption giống template đã tạo; embedded Qwen worker cũng khớp nguyên chuỗi. Không thấy việc âm thầm thay công thức Dice/IoU hoặc tắt full_both trong source.

Tuy nhiên đây chưa phải một phép kiểm tra chỉ thay nội dung caption và giữ mọi thứ khác cố định:

- Full_both thêm hai projection/gate text, semantic selector và thay context được chọn cũng như quá trình tối ưu. Số trainable parameters tăng từ 6.306.754 lên 6.704.325.
- Runtime dùng đường dẫn bản đính kèm DINOv3 repo/weights khác chủ upload, dù cùng model/filename. Chưa có hash weight và split/pool manifest của cả hai run để xác minh byte-identical độc lập.
- Mỗi phương pháp mới có một seed. Hai file prediction CSV đầy đủ và checkpoint chưa được cung cấp để paired bootstrap hoặc phân tích per-image/per-class. Không thể lập CI hoặc khẳng định giảm có ý nghĩa thống kê từ hai số trung bình.
- Baseline trong notebook mới là reference nhập từ output cũ, không phải visual model được train lại cùng phiên. Audit này đã đọc lại notebook visual gốc để xác nhận số baseline, không chỉ tin bảng chép sẵn.

## 4. Diễn giải kết quả

**Validation nhỉnh hơn nhưng test thấp hơn:** full_both đạt best val Dice 87.55%, so với visual 87.31%; lợi ích nhỏ này không chuyển thành test gain. Có thể liên quan biến thiên training/validation hoặc khả năng khái quát hóa của fusion/selector mới; chưa đủ dữ liệu để chốt nguyên nhân là overfitting.

**Thêm caption không tự đảm bảo tốt hơn cho dense prediction:** giả thuyết có thể là mô tả ngắn/global chưa bổ sung thông tin không gian cần để vẽ chính xác biên tổn thương, hoặc semantic retrieval chọn context khác nhưng không hữu ích cho segmentation. Đây là giả thuyết cần ablation, không phải kết luận đã chứng minh hay khẳng định Qwen mô tả sai.

**Không kết luận Qwen/text luôn có hại:** phát biểu đúng hiện tại là cấu hình full_both cụ thể này chưa vượt visual TACS trên ISIC trong single-seed experiment đang có. Không nên đổi checkpoint, prompt hoặc threshold dựa trên test để làm bảng đẹp hơn; nếu phát triển tiếp, công bố đây là vòng thăm dò và chốt quyết định bằng train/validation.

## 5. Thí nghiệm tiếp theo để tìm nguyên nhân

Ưu tiên giữ nguyên CSV, split/pool, budget và metric rồi tách ba thành phần: `query_only`, `candidate_only`, `both_fusion` (visual selector), so với `full_both` hiện tại. Cách này phân biệt text fusion có vấn đề hay semantic retrieval đang làm giảm hiệu quả. Không cần sinh lại caption chỉ để chạy các cấu hình này.

Trước khi tốn thêm compute, xuất các artifact đã có trên Kaggle: `metrics.json`, `training_history.csv`, `test_predictions.csv`, `training_protocol/protocol.json`, split/pool CSV và `caption_generation/captions.csv`. Chúng cho phép kiểm tra lỗi theo ảnh, phân bố candidate được chọn và nội dung caption. Thêm seed 43/44 cho cả visual/full_both nếu cần đánh giá độ ổn định; không chỉ chạy thêm seed cho phương pháp muốn chứng minh tốt hơn.

## 6. Đoạn văn dùng cho báo cáo

> Trên ISIC 2017 với frozen DINOv3 ViT-S/16 và độ phân giải 448×448, mở rộng full_both sử dụng caption Qwen3-VL-8B cho query, candidate và selector đạt test Dice 84.87% và IoU 76.54%, thấp hơn visual TACS lần lượt khoảng 0.73 và 0.84 điểm phần trăm. Mặc dù best validation Dice tăng nhẹ từ 87.31% lên 87.55%, kết quả test chưa cho thấy lợi ích từ cơ chế bổ sung văn bản. Tất cả caption đều qua policy và gate text khác 0 trên validation, nên sự suy giảm không thể giải thích đơn giản bằng việc toàn bộ text bị mask. Kết quả hiện tại phản ánh một cấu hình và một seed; cần ablation riêng fusion/semantic retrieval và đánh giá nhiều seed trước khi kết luận về nguyên nhân hoặc tác động tổng quát của caption đối với segmentation.
