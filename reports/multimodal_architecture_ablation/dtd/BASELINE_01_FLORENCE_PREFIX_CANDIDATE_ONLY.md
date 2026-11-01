# DTD Architecture Study 01: Candidate-Only + Florence-Prefix

Ngày đọc: 25/10/2026. Nguồn là output đã lưu trong [dtd-architecture-study-01.ipynb](../../../dtd-architecture-study-01.ipynb); không thực thi cell, load checkpoint hay chạy lại huấn luyện.

SHA256 notebook: `3254ec626634f7922472963e4ccb64c5016cdc2f43730ba6c3fb411e739e5e07`.
Bằng chứng có cấu trúc: [baseline_01_output_evidence.json](baseline_01_output_evidence.json).

## 1. Xác Nhận Cấu Hình

- DTD official split 1: train/validation/test đều 1.880 ảnh; 47 lớp.
- DINOv3 ViT-L/16, 384x384; candidate pool 376 ảnh training, K=1; pool/training seed mặc định 42.
- Runtime báo CUDA, GPU memory khoảng 95 GiB, training batch 32, retrieval batch 128, feature-refresh batch 64.
- `candidate_only`: không dùng query text, selector chỉ dùng ảnh, caption của candidate được chọn vào fusion.
- CSV đúng Florence-Prefix, policy `dtd_lexical_v2`; SHA256 trong stdout khớp file cục bộ:
  `b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011`.
- Trong pool có 324/376 caption đủ điều kiện, tương đương 86,1702%; 52 caption bị mask, ảnh vẫn nằm trong pool.
- Bộ diagnostic ghi 376/376 có semantic flag; policy v2 không dùng flag/threshold 0.30 này để loại cứng.
- Bảng trạng thái ghi `completed`, 50 epoch, LR horizon 50, best epoch 38. Checkpoint chọn theo minimum validation loss, không theo maximum validation F1.

So với [template baseline](../../../notebooks/02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/mtacs-dtd-vitl16-prefix-candidate-only-baseline.ipynb), source chỉ thay default đường dẫn CSV. Các định nghĩa hàm/lớp có AST giống nhau; đây không phải một model/fusion mới được thay vào khi train.

## 2. Kết Quả Đã Lưu

Số bên dưới lấy từ bảng metrics riêng ở cell 19 và 21, không lấy bảng cuối đã làm tròn bốn chữ số.

| Chỉ số | Validation, best epoch 38 | Test, cùng best checkpoint |
| --- | ---: | ---: |
| Accuracy | 0.841489 / 84,1489% | 0.854255 / 85,4255% |
| Macro-F1 | 0.840162 / 84,0162% | 0.853032 / 85,3032% |
| Loss | 1.178962 | 1.160942 |
| Query-caption rate | 0 | 0 |
| Selected-candidate-caption valid rate | 0.820745 | 0.822340 |
| Mean query gate | 0 | 0 |
| Mean candidate gate | 0.001115 | 0.001117 |

Accuracy hiển thị tương ứng khoảng 1.606/1.880 ảnh test đúng. Candidate-caption rate tương ứng khoảng 1.546/1.880 lần chọn context test có caption dùng được; đây là số suy ra từ tỷ lệ hiển thị, chưa đối chiếu prediction CSV độc lập.

Trong các dòng epoch được lưu, validation F1 lớn nhất là khoảng 0.8430 ở epoch 32; loss ở đó là 1.1854. Epoch 38 có loss thấp hơn, nên chọn epoch 38 là đúng với tiêu chí đã khai báo. Không đổi tiêu chí sau khi đã xem test.

## 3. Đối Chiếu Với Run Lịch Sử

Các hàng là kết quả đã đọc trực tiếp từ notebook tương ứng, không phải các seed lặp của cùng một run. Chưa có đầy đủ artifact để chứng nhận mọi điều kiện runtime/weights giữa các lần chạy đều giống nhau.

| Cấu hình | Val Macro-F1 (%) | Test Macro-F1 (%) | Chênh lệch test của baseline mới |
| --- | ---: | ---: | ---: |
| Visual-only lịch sử, vẫn có context ảnh | 84,6578 | 84,8168 | +0,4864 điểm phần trăm |
| Candidate-only lịch sử, CSV riêng của lần chạy cũ | 83,9829 | 85,0321 | +0,2711 điểm phần trăm |
| Full-both + Florence-Prefix lịch sử | 80,2258 | 81,0651 | +4,2381 điểm phần trăm |
| **Candidate-only + Florence-Prefix mới** | **84,0162** | **85,3032** | Mốc tham chiếu mới |

Nhận xét:

- Có thể lấy 0.853032 làm kết quả test của baseline 01 trong nhóm architecture study này.
- Validation F1 của baseline mới thấp hơn visual-only lịch sử 0,6416 điểm phần trăm; test nhỉnh hơn chưa chứng minh cải thiện ổn định.
- Chênh lệch test với visual-only tương ứng khoảng 9 ảnh đúng; với candidate-only cũ khoảng 5 ảnh đúng theo Accuracy. Chưa có prediction CSV để biết các ảnh đúng/sai giao nhau như thế nào.
- Full-both thay đồng thời query text và semantic selector so với candidate-only. Không quy toàn bộ mức chênh 4,2381 điểm cho riêng query caption, selector hay chất lượng CSV.
- Chỉ một seed và một DTD split; không có cơ sở báo std, p-value hoặc kết luận vượt phương pháp gốc.

## 4. Điểm Cần Chẩn Đoán Trước Khi Nâng Cấp

### 4.1. Gate Nhỏ Nhưng Không Phải Phần Trăm Đóng Góp

Trong `forward_ablated` ở cell 12, fusion có dạng:

```text
fused = LayerNorm(v + g_c * P_c(t_c))
```

Nhánh query bị mask nên không xuất hiện trong công thức cho cấu hình này. Gate candidate giảm từ khoảng 0.089 ở epoch 1 xuống khoảng 0.001 ở các epoch cuối; không phải tất cả caption đều bị loại vì hơn 82% context test vẫn có text hợp lệ.

Tuy nhiên, cell 11 trả `v` đã L2-normalize, còn `P_c` là Linear -> LayerNorm -> Dropout, không L2-normalize đầu ra. Hai nhánh có thể có độ lớn khác nhau. Vì vậy **không diễn giải `g_c=0.001117` thành “text đóng góp 0,1117%” hoặc kết luận text đã bị bỏ hoàn toàn**.

LayerNorm có tham số học được; không suy ra chính xác norm text sau training chỉ từ kích thước 1.024. Cần đo trên checkpoint thật:

```text
r = ||g_c * P_c(t_c)||_2 / (||v||_2 + epsilon)
```

Xuất mean/median/percentile của gate và r, cả trên toàn bộ mẫu lẫn riêng mẫu caption hợp lệ. Đây là chẩn đoán thang đo, chưa phải bằng chứng căn chỉnh ngữ nghĩa.

### 4.2. Đối Chứng Validation Không Cần Train Lại

Dùng cùng best checkpoint epoch 38 và cùng context ảnh đã chọn cho từng validation query. Không thay selector, ảnh, class mapping hoặc evaluation preprocessing.

| Can thiệp | Thay đổi duy nhất | Câu hỏi |
| --- | --- | --- |
| Aligned | Dùng đúng caption như baseline | Mốc checkpoint hiện tại |
| Text off | Mask toàn bộ candidate text | Text có tác dụng trực tiếp ở inference không? |
| Text shuffled | Đảo vector/text giữa candidate có caption hợp lệ, giữ mask và context ảnh | Tác dụng có phụ thuộc caption khớp đúng ảnh không? |

Thu thập validation Accuracy, Macro-F1, CE, tỷ lệ đổi dự đoán và độ thay đổi logits/xác suất. Nếu cần khoảng tin cậy, dùng dự đoán theo cùng image ID để bootstrap có ghép cặp, không so hai bảng aggregate rồi tự suy ý nghĩa thống kê.

Không dùng nhãn để tạo ánh xạ shuffle; không bật lại caption bị policy mask. Tắt text ở checkpoint đã train không tương đương train một visual-only model từ đầu, nên đối chứng này không thay thế hoàn toàn matched visual-only retraining.

## 5. Hướng Làm Tiếp Theo

### Bước 1: Lưu Baseline Và Chạy Diagnostic

Giữ nguyên CSV, notebook kết quả và toàn bộ output. Chạy ba validation intervention ở trên trước khi tiêu tốn thêm 50 epoch; không sửa gate thủ công hoặc tăng threshold dựa trên test.

### Bước 2: Notebook 02 - Candidate-Only Với Norm-Calibrated Fusion

Đề xuất một đối chứng đơn giản: L2-normalize text sau projection tại điểm fusion để có thang norm so sánh được với vector visual hiện tại. Vẫn dùng gate học được và validity mask, không ép gate khác 0, không đổi semantic selector hay thêm query caption.

Đây là thay đổi một yếu tố để kiểm nghiệm ảnh hưởng của thang đo; không tuyên bố L2 normalization làm hai modality tự động cùng ngữ nghĩa hoặc chắc chắn tăng điểm. Giữ CSV/hash, split, pool, K=1, backbone/resolution, batch, losses, 50 epoch và checkpoint criterion như baseline.

### Bước 3: Thêm Alignment Thành Ablation Riêng

Sau khi có đối chứng scale, kiểm nghiệm learned alignment lấy cảm hứng từ m-LTM/Mahalanobis; Partial OT là một biến thể sau đó để kiểm tra robustness với cặp mô tả không khớp. m-LTM gốc được đánh giá cho audio-text, nên chuyển sang ảnh-caption TACS là đề xuất cần kiểm nghiệm, không phải lợi ích đã được chứng minh cho DTD. [Nguồn m-LTM](https://arxiv.org/abs/2405.10084).

Alignment cần dùng cặp **ảnh training/candidate và caption của chính ảnh đó**, không coi mọi query và caption context được retrieve là cặp positive đúng tuyệt đối. Loss phải ràng buộc biểu diễn thực sự đi vào fusion, không chỉ một projection phụ không tác động đường dự đoán; dùng training data và caption hợp lệ, chọn hệ số loss bằng validation.

Khi so sánh thêm alignment, cần so trực tiếp với control có cùng normalization. Không thay normalization, alignment loss, query text và semantic selector đồng thời rồi quy mọi chênh lệch cho alignment.

### Bước 4: Các Nhánh Sau

- Semantic selector mới: chỉ thêm sau khi đánh giá alignment; giữ query caption tắt nếu câu hỏi là query-image / candidate-text selection. DINO-image và CLIP-text không được dot-product trực tiếp khi chưa có ánh xạ phù hợp.
- Token fusion/resampler: nhánh riêng so với baseline vector fusion, có control về capacity và K.
- ASWP: thêm sau token representation để kiểm tra pooling, không gộp ngay vào notebook 02. Resampling và ASWP được đề xuất cho retrieval trong OmniRet; việc thích nghi cho classification TACS cần ablation riêng. [Nguồn OmniRet](https://arxiv.org/abs/2603.02098).
- Sau khi chốt cấu hình bằng validation, đánh giá final test theo kế hoạch và thêm seeds 43/44 nếu đủ tài nguyên. Khi đổi caption sang Qwen8B, chạy cả baseline và biến thể được chọn để có đối chứng cùng CSV.

## 6. Cảnh Báo Và Artifact Cần Giữ

- Không có Jupyter `output_type=error`, nhưng stderr có lỗi/cảnh báo `QueueFeederThread`, `Bad file descriptor`, semaphore và `_MultiProcessingDataLoaderIter.__del__`. Sau đó vẫn có log đến epoch 50 và bảng completed/test; không gọi đây là một lần chạy hoàn toàn sạch cảnh báo.
- Stdout lưu 49 dòng epoch, thiếu epoch 48. Bảng trạng thái ghi 50 epoch; cần `history.csv`/training state để kiểm tra độc lập dòng còn thiếu, không tự bù log.
- CLIP load report có key `vision_model`/`visual_projection` không dùng khi nạp text-only model và buffer position IDs. Saved report không liệt kê missing text/projection parameters; không coi các key vision dư này tự động là lỗi nạp nhánh text.
- Thông báo “Downloading: file:///kaggle/input/...” ở DINO là sao chép checkpoint local sang Torch cache, không phải bằng chứng notebook đã tải weights qua Internet.
- Có thể thử `NUM_WORKERS=0` trong phiên chẩn đoán để giảm vấn đề worker; ghi lại thay đổi môi trường, không chỉnh notebook kết quả lịch sử.

Artifact nên export từ output root:

```text
protocol.json
caption_method.json
environment.json
candidate_pool.csv
shared/captions.csv
shared/candidate_text_embeddings.pt
runs/candidate_only/seed_42/best.pt
runs/candidate_only/seed_42/history.csv
runs/candidate_only/seed_42/val_predictions.csv
runs/candidate_only/seed_42/test_predictions.csv
```

Đối chiếu tên file thực tế trong output; các path `val_predictions.csv`/`test_predictions.csv` tuân theo implementation lưu kết quả của notebook. Chưa có các artifact này được audit độc lập trong yêu cầu hiện tại; không load `.pt` không rõ nguồn trong quá trình chỉ đọc kết quả.

## Kết Luận Ngắn

Baseline candidate_only + Florence-Prefix đạt test Macro-F1 85,3032% và có thể làm mốc cho nhóm architecture study. Tuy nhiên, validation chưa vượt visual-only lịch sử và giá trị gate chưa đủ để định lượng đóng góp text do khác thang norm. Bước hợp lý tiếp theo là kiểm tra text off/shuffle và norm ratio trên validation, sau đó thử norm-calibrated fusion như một ablation riêng trước khi thêm alignment và các thành phần phức tạp hơn.
