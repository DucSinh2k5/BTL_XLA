# SUN397: xuất bundle web từ best.pt, không train lại

Hai notebook độc lập:

| Notebook | Model đầu vào | Bundle đầu ra |
|---|---|---|
| `sun397-tacs-visual-vitl16-export-only.ipynb` | TACS visual L/16, 384, K=1 | `sun397_tacs_visual_vitl16_web_bundle.zip` |
| `sun397-mtacs-full-both-vitl16-export-only.ipynb` | M-TACS full_both L/16, 384, K=1 | `sun397_mtacs_full_both_vitl16_web_bundle.zip` |

Đây là notebook mới, không sửa output hoặc checkpoint của các notebook train.
Không có vòng train, optimizer hoặc bước sinh lại toàn bộ captions.

## Input Chung

Attach Output đã train như một Kaggle Input. Đặt `SOURCE_RUN_DIR` tới **đúng thư mục có best.pt**, không phải thư mục Input ngoài cùng.

Các file cần giữ cùng best.pt:

```text
best.pt
protocol.json
class_names.json
candidate_pool.csv
train_split.csv
validation_split.csv
embedded_runtime/ hoặc embedded_runtime_v2/  # nên giữ source gốc
history.csv                                # tùy chọn, chỉ đọc thông tin
training_complete.json                     # tùy chọn, không tạo giả
test_split.csv                             # cần nếu bật full test
```

Không cần `last.pt` và optimizer state. Không cần file DINOv3 pretrained `.pth`: model được dựng với `pretrained=False`, sau đó nạp toàn bộ trọng số từ `best.pt` bằng `strict=True`.

Attach thêm:

1. SUN397 images, để tìm đúng ảnh theo image_id của candidate pool và một ảnh validation làm smoke test.
2. Đúng code DINOv3 đã dùng khi train, bao gồm LICENSE/hub/backbone/layers. Hash code được đối chiếu protocol; không tự thay sang phiên bản khác.

Thứ tự lớp và candidate lấy từ artifact đã lưu. Notebook **không lấy ngẫu nhiên pool mới**. Các CSV train/val/pool được dùng để kiểm tra lại run identity, không được đưa vào vòng train.

Checkpoint phải thuộc đúng kiến trúc notebook web tương ứng. `best.pt` từ một bản B/16, gated classifier khác, ablation khác hoặc checkpoint adapter không được tự đổi tên key để nạp. Nếu source runtime thiếu, fallback chỉ được dùng khi hash source inference khớp protocol.

## Input Riêng M-TACS

Giữ thêm:

```text
test_split.csv
shared/captions.csv
shared/text_embeddings.pt
shared/caption_protocol.json
shared/caption_settings.json
```

`test_split.csv` cũng cần khi tắt full test vì CSV/cache caption nguyên bản bao phủ cả train, val và test; notebook kiểm tra chính xác coverage này.

Attach các model đóng băng đã dùng để tạo cache:

- Florence-2-base Microsoft legacy, đủ weights/tokenizer/processor/Python model code.
- CLIP ViT-B/32, đủ weights/tokenizer/config.
- Offline wheels tương thích Python/Linux của Kaggle, khi cần chạy Florence trong môi trường con Transformers 4.49.x.

Không cần copy `shared/caption_runtime/legacy_packages/` từ phiên cũ. Notebook có thể chuẩn bị môi trường con từ offline wheels; không downgrade kernel và không bật Internet.

Model text có thể nằm ở mount mới nhưng hash nội dung phải giống cache cũ. Các đường dẫn trong caption settings nguyên bản được giữ cho provenance; inference thực tế dùng model trong bundle. CSV và vectors không bị viết lại. Vector candidate được lấy theo đúng image_id, không lấy theo vị trí tùy ý của CSV.

Hai chỗ cập nhật inplace vào inference tensor của caption worker được đổi sang masked_fill tạo tensor mới trong **source xuất ra**, có ghi hash trước/sau. Đây không phải đổi prompt, caption policy hoặc chạy lại caption dataset.

## Cách Chạy

1. Restart kernel để tránh giữ module/model từ notebook train hoặc notebook export còn lại.
2. Attach các Input và sửa section 2: SOURCE_RUN_DIR, DATASET_ROOT, DINOV3_REPO_DIR; với M-TACS còn FLORENCE_MODEL_DIR, CLIP_MODEL_DIR, OFFLINE_ASSETS_ROOT.
3. Để `FEATURE_BATCH_SIZE=8`, `RUN_FULL_TEST=False` trước. GPU được khuyến nghị; CPU sẽ chậm khi encode L/16.
4. Run All. Notebook xác minh checkpoint/source/pool/cache, dựng model và encode pool một lần.
5. Đợi cold reload kiểm tra dự đoán và attention. M-TACS còn sinh caption mới cho **một ảnh** để xác minh pipeline ảnh ngoài dataset.
6. Tải ZIP ở cell cuối hoặc thư mục exports trong Output.

Output mặc định:

```text
/kaggle/working/exports_sun397_tacs_visual_vitl16/exports/...
/kaggle/working/exports_sun397_mtacs_full_both_vitl16/exports/...
```

Nếu bị OOM ở bước encode, giảm FEATURE_BATCH_SIZE và restart; không đổi backbone hoặc pool. Các model/bank đã xuất xong trong staging được dùng lại sau khi kiểm tra hash. File staging chưa được ghi nhận hoặc có hash khác sẽ bị từ chối, không bị xóa tự động. Bundle đã hoàn chỉnh có thể được dùng lại để tạo ZIP còn thiếu mà không encode lại.

Không ghi đè source run. Không ghi giả `training_complete.json`, không đổi epoch horizon và không gọi train để vượt guard export cũ. Notebook này xuất snapshot best.pt theo một protocol export riêng.

## File Trong Bundle

```text
model.pt                         # full best weights, định dạng cho adapter web
candidate_features.pt            # pool encode bằng chính best.pt, fp32
candidate_text.pt                # chỉ M-TACS: vectors và mask đúng thứ tự pool
candidate_pool.json
config.json / class_names.json
candidates/...                   # ảnh pool thật, không phụ thuộc mount SUN397 sau giải nén
runtime/...                      # adapter/CLI và cold checker
vendor/dinov3/...                 # code và license gốc
attention_config.json
examples/prediction.json          # prediction và hai attention readout thật
smoke_test.json
metrics.json
export_provenance.json
manifest.json
provenance/original_run/...
```

M-TACS có thêm model Florence/CLIP, toàn bộ CSV/cache đã lưu và các offline wheel được cung cấp. Model attention riêng không cần thiết: heatmap lấy từ backbone đã nạp trọng số.

ZIP chứa toàn bộ artifact inference, không chứa last.pt/optimizer để train tiếp. Web host vẫn cần môi trường Python/PyTorch phù hợp OS/CUDA; wheel Linux của Kaggle không cài được trên Windows. Giữ điều kiện sử dụng model/dataset gốc. Khi tích hợp cả hai predictor vào web, cần quản lý source DINOv3 chung hoặc worker tách biệt để tránh xung đột module giữa hai vendor root.

## Metrics Và Trung Thực Báo Cáo

Mặc định không chạy lại 19.850 ảnh test. `metrics.json` ghi:

```json
{"test": null, "test_status": "not_run_export_only"}
```

Điểm validation trong checkpoint được ghi với tên `validation_from_checkpoint`. Test metrics/predictions cũ, nếu có, được giữ trong `provenance/unverified_original_metrics/` như tài liệu tham khảo, không giả là được notebook export đánh giá lại.

Muốn tính lại metrics, bật RUN_FULL_TEST trước khi bắt đầu một export mới. Dùng output riêng để không trộn protocol của bản demo nhanh với bản đã đánh giá test đầy đủ.

Best epoch nhỏ hơn epoch cuối là bình thường. Ví dụ best epoch 42, history chạy tới 100 không có nghĩa chỉ train 42 epoch. Nếu thiếu marker hoàn tất, notebook vẫn xuất snapshot và ghi rõ tình trạng kiểm chứng, không bắt train lại để tạo marker.

Smoke test kiểm tra logits/context selection và hai attention map, không thay thế đánh giá toàn bộ test set, không chứng minh accuracy/macro-F1 hoặc causal attribution.

## Mã Nguồn

- Exporter: `src/deployment/sun397_export_only.py`.
- Builder: `tools/notebook_builders/build_sun397_export_only_notebooks.py`.
- Tests: `tests/test_sun397_export_only.py`, dùng DINOv3 nhỏ thật trên CPU và caption service fixture được đánh dấu rõ; không giả đó là kiểm chứng chất lượng Florence hoặc full SUN397/L/16 trên GPU.
