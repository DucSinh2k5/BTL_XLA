# Bốn CSV Và Notebook Train Theo Method

Đã đọc toàn bộ bốn CSV, kiểm tra cột `generation_method`, fingerprint, image ID và split. Mỗi file có 5.640 ảnh, chia 1.880 train / 1.880 val / 1.880 test; cùng thứ tự và cùng fingerprint sinh caption. Image ID/split khớp official DTD split 1 trong dataset local.

| Method | File CSV | Metadata `generation_method` | Caption hợp lệ lưu trong CSV |
| --- | --- | --- | ---: |
| 1. Florence-Prefix | `captions_prefix_control.csv` | `prefix_control` | 4.749 / 5.640 |
| 2. Florence-FirstValid | `captions_first_valid_control.csv` | `first_valid_control` | 5.640 / 5.640 |
| 3. Florence-SingleGrounded | `captions_single_grounded_control.csv` | `single_grounded_control` | 5.640 / 5.640 |
| 4. Florence-MultiGrounded | `captions_multibeam_grounded.csv` | `multibeam_grounded` | 5.640 / 5.640 |

Đây là độ phủ theo policy, không phải kết quả phân loại hoặc chứng minh caption đúng/không leak. Ba file đạt 100% vẫn có thể chứa từ đồng nghĩa/alias hoặc mô tả chưa chuẩn.

## Notebook Đã Tạo: Method 1

`mtacs-train-vitl16-dtd-caption-ablation-01-florence-prefix.ipynb`

Notebook dựa trực tiếp trên bản 05 `full_both`. Mô hình vẫn dùng query image + query caption + một context image + caption của context; selector vẫn kết hợp ảnh và text. Không đổi cấu trúc, loss, augmentation, optimizer hay quy tắc chọn checkpoint.

- Official DTD split 1, pool seed 42, training seed 42 mặc định.
- 50 epoch, K=1, candidate pool 376 ảnh train, loại chính query khỏi lựa chọn context.
- Batch vẫn tự chọn theo VRAM như notebook 05: batch 32 trên GPU từ 80 GiB. Giữ cùng GPU/batch ở các method sau.
- Policy `dtd_lexical_v2`, điểm CLIP theo tên lớp chỉ để audit.
- CSV đã có sẵn, không sinh lại caption và không cần attach Florence/offline wheels cho bước train này.
- Checkpoint, cache text và output riêng; không dùng checkpoint từ notebook 05 hay method khác.

## Chạy Trên Kaggle

1. Upload notebook method 1 và attach bốn CSV này làm Input, giữ nguyên tên file.
2. Attach DTD, DINOv3 code/ViT-L weights và CLIP giống lần train notebook 05. Bật GPU.
3. Chạy từ trên xuống. Mặc định `CAPTIONS_INPUT_PATH` để trống: notebook tìm đúng `captions_prefix_control.csv` trong Inputs và kiểm tra hash, không chọn một file `captions.csv` khác.
4. Nếu muốn chỉ rõ đường dẫn, sửa tại cell cấu hình:

```python
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_DATASET/captions_prefix_control.csv"
```

Giữ `GENERATE_CAPTIONS_IF_MISSING = False`. CSV khác method, thiếu ảnh, đổi split hoặc khác hash sẽ bị chặn. Đừng bật generation để vượt lỗi; kiểm tra đúng Input/đường dẫn. Bản này dùng CSV tự nhiên, không tự thay bằng CSV matched-mask.

Thư mục output mặc định:

```text
/kaggle/working/outputs/mtacs_dtd_split1_caption_01_florence_prefix_vitl16/
```

Các file chính sau khi train:

- `caption_01_florence_prefix_results.csv`: metrics kèm tên method và hash CSV.
- `caption_01_florence_prefix_report.md`: báo cáo, coverage và provenance.
- `caption_01_florence_prefix_training.png`: learning curves.
- `runs/full_both/seed_42/best.pt`: checkpoint tốt nhất theo validation loss của riêng method 1.
- `shared/caption_model_inputs.csv`: nội dung và mask thực tế đi vào mô hình.
- `caption_method.json` và `protocol.json`: danh tính method, CSV và cấu hình train.

Tên `full_both` trong đường dẫn checkpoint chỉ kiến trúc train, không có nghĩa dùng lại checkpoint cũ. Prefix là đối chứng mới; không điền sẵn điểm Florence lịch sử vào bảng kết quả của nó.

## Notebook Đã Tạo: Method 2

`mtacs-train-vitl16-dtd-caption-ablation-02-florence-firstvalid.ipynb`

Notebook này nhận đúng `captions_first_valid_control.csv`, metadata `first_valid_control`. Caption được chọn là đoạn 5–8 từ hợp lệ đầu tiên từ mô tả greedy, không xếp hạng caption bằng CLIP và không thêm beam cho method này. Nhánh text và semantic selector của M-TACS vẫn dùng CLIP giống bản Prefix; không thay cấu trúc mô hình phân loại.

Giữ cùng split 1, seed/pool seed 42, K=1, 50 epoch, VRAM-dependent batch, loss, optimizer, augmentation và chọn checkpoint theo validation loss. Khi so với Prefix, lưu ý coverage tăng từ 4.749 lên 5.640 caption hợp lệ; không quy toàn bộ thay đổi kết quả cho chất lượng nội dung.

Trên Kaggle, attach CSV thứ 2 và các input DTD/DINOv3/CLIP như method 1, chạy từ trên xuống. Có thể để trống `CAPTIONS_INPUT_PATH` để tự tìm đúng tên/hash file, hoặc đặt:

```python
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_DATASET/captions_first_valid_control.csv"
```

Output riêng:

```text
/kaggle/working/outputs/mtacs_dtd_split1_caption_02_florence_firstvalid_vitl16/
```

Các file kết quả mang tiền tố `caption_02_florence_firstvalid`: `results.csv`, `report.md`, `training.png`. Checkpoint của riêng method 2 nằm trong `runs/full_both/seed_42/`. Không nạp checkpoint hoặc cache của method 1.

Các biến môi trường riêng nếu cần: `DTD_FIRST_VALID_CAPTIONS_INPUT_PATH`, `DTD_CAPTION_02_FIRST_VALID_OUTPUT`, `DTD_CAPTION_02_FIRST_VALID_EPOCHS`, `DTD_CAPTION_02_FIRST_VALID_SEED`. Mặc định vẫn 50 epoch, seed 42; không kế thừa đường dẫn/seed dành riêng cho Prefix.

## Notebook Đã Tạo: Method 3

`mtacs-train-vitl16-dtd-caption-ablation-03-florence-singlegrounded.ipynb`

Notebook nhận đúng `captions_single_grounded_control.csv`, metadata `single_grounded_control`. CSV này xếp hạng các đoạn hợp lệ từ một mô tả greedy bằng CLIP ảnh–text rồi chọn đoạn tốt nhất theo điểm đó; chưa thêm mô tả beam. Nó sử dụng cùng nguồn mô tả, tập ứng viên và mask hợp lệ với FirstValid, để kiểm tra ảnh hưởng của bước ranking. CLIP score được tối ưu không phải bằng chứng chất lượng độc lập.

Cấu trúc `full_both` và toàn bộ train/test, loss, optimizer, augmentation giữ nguyên như method 1–2. Vẫn dùng text cho cả query và candidate, selector ảnh + text, official split 1, seed/pool seed 42, K=1 và 50 epoch. CSV đã có, không sinh lại và không cần Florence cho bước train này.

Attach CSV thứ 3 cùng DTD/DINOv3/CLIP như trước rồi Run All. Có thể để trống `CAPTIONS_INPUT_PATH` để tự tìm file theo đúng tên và hash, hoặc đặt:

```python
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_DATASET/captions_single_grounded_control.csv"
```

Output riêng:

```text
/kaggle/working/outputs/mtacs_dtd_split1_caption_03_florence_singlegrounded_vitl16/
```

Các file kết quả mang tiền tố `caption_03_florence_singlegrounded`: `results.csv`, `report.md`, `training.png`. Checkpoint riêng trong `runs/full_both/seed_42/`; không nạp checkpoint hoặc cache từ method 1–2. Method-specific biến môi trường: `DTD_SINGLE_GROUNDED_CAPTIONS_INPUT_PATH`, `DTD_CAPTION_03_SINGLE_GROUNDED_OUTPUT`, `DTD_CAPTION_03_SINGLE_GROUNDED_EPOCHS`, `DTD_CAPTION_03_SINGLE_GROUNDED_SEED`.

## Notebook Đã Tạo: Method 4

`mtacs-train-vitl16-dtd-caption-ablation-04-florence-multigrounded.ipynb`

Notebook nhận đúng `captions_multibeam_grounded.csv`, metadata `multibeam_grounded`. Caption đã được chọn từ một mô tả greedy và ba mô tả beam bằng cùng cách trích đoạn hợp lệ và xếp hạng CLIP ảnh–text. So với SingleGrounded, thay đổi là thêm phương án mô tả để lựa chọn; hai CSV có cùng độ phủ hợp lệ. Ba beam không phải ba ảnh context, M-TACS vẫn K=1.

Giữ mô hình `full_both`, text query + text candidate, selector ảnh + text, official split 1, seed/pool seed 42, 50 epoch, batch theo VRAM, cùng loss/optimizer/augmentation và chọn checkpoint theo validation loss. Không sinh lại caption, không fine-tune Florence trong notebook này và không nạp checkpoint của ba method trước.

Attach CSV thứ 4 cùng DTD/DINOv3/CLIP như trước rồi Run All. Để trống `CAPTIONS_INPUT_PATH` để tự tìm đúng tên/hash, hoặc đặt:

```python
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_DATASET/captions_multibeam_grounded.csv"
```

Output riêng:

```text
/kaggle/working/outputs/mtacs_dtd_split1_caption_04_florence_multigrounded_vitl16/
```

Các file kết quả mang tiền tố `caption_04_florence_multigrounded`: `results.csv`, `report.md`, `training.png`. Checkpoint riêng trong `runs/full_both/seed_42/`. Biến môi trường riêng: `DTD_MULTI_GROUNDED_CAPTIONS_INPUT_PATH`, `DTD_CAPTION_04_MULTI_GROUNDED_OUTPUT`, `DTD_CAPTION_04_MULTI_GROUNDED_EPOCHS`, `DTD_CAPTION_04_MULTI_GROUNDED_SEED`.

Hiện đã tạo đủ bốn notebook tương ứng bốn CSV. Mỗi notebook train độc lập từ đầu trong thư mục riêng, không train nối tiếp trên cùng checkpoint. Chưa có kết quả phân loại mới được kiểm chứng; điểm CLIP cao hơn do ranking không đảm bảo F1 tốt hơn. Không file CSV gốc hoặc notebook method 1–3 nào bị sửa.
