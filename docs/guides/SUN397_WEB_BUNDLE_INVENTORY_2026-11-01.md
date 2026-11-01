# Kiểm kê hai bundle SUN397 để tích hợp demo web

Ngày kiểm tra: 01/11/2026. Dữ liệu trong tài liệu được đọc trực tiếp từ hai bundle đã giải nén và mã web hiện tại. Không thay đổi trọng số, pool, caption, manifest hoặc backend.

**Cập nhật sau kiểm kê:** theo yêu cầu tích hợp web, hai bundle đã được di chuyển nguyên vẹn vào `TACS/artifacts/deployment/sun397/tacs_visual/` và `TACS/artifacts/deployment/sun397/mtacs_full_both/`. Backend đã được nối với worker tự nạp model khi khởi động. Các phần mô tả “chưa tích hợp” bên dưới ghi lại trạng thái tại thời điểm kiểm kê; hướng dẫn hiện hành ở `TACS/apps/sun397_demo/README.md`.

## 1. Đường dẫn cần dùng

Hai thư mục người dùng cung cấp đều có thêm một thư mục cùng tên bên trong. Bundle root phải là nơi có trực tiếp `model.pt`, `config.json` và `manifest.json`:

```text
F:/Documents/CODE/Python/cv_project/XLA/TACS/
├── sun397_tacs_visual_vitl16_web_bundle/
│   └── sun397_tacs_visual_vitl16_web_bundle/       <-- TACS_BUNDLE_DIR
└── sun397_mtacs_full_both_vitl16_web_bundle/
    └── sun397_mtacs_full_both_vitl16_web_bundle/   <-- MTACS_BUNDLE_DIR
```

Các tên biến trên là tên cấu hình đề xuất cho bước tích hợp. Backend hiện tại chưa đọc chúng.

| Nội dung | TACS visual | M-TACS full_both |
|---|---|---|
| Backbone | DINOv3 ViT-L/16 | DINOv3 ViT-L/16 |
| Kích thước ảnh | 384 × 384 | 384 × 384 |
| Số lớp | 397 | 397 |
| Context mỗi query | 1 | 1 |
| Số ảnh candidate | 3.573 | 3.573 |
| Số file, gồm manifest | 3.624 | 3.661 |
| Dung lượng giải nén | Khoảng 2,393 GiB | Khoảng 4,674 GiB |
| Caption | Không dùng | Query + candidate, có mask |
| Selector | Đặc trưng ảnh | Ảnh + tương đồng caption hợp lệ |

Hai bundle có cùng thứ tự 397 lớp, cùng ảnh/thứ tự pool và cùng nội dung source DINOv3; model và feature bank được huấn luyện/xuất riêng. Mỗi pool chỉ thuộc train, không giao validation/test. Split lưu trong cả hai bundle là 17.865 train, 1.985 validation và 19.850 test.

Đã kiểm tra lại toàn bộ SHA-256 của 3.623 file được manifest TACS liệt kê và 3.660 file được manifest M-TACS liệt kê: không có file thiếu, sai kích thước hoặc sai hash. Manifest không liệt kê chính nó nên số file thực tế nhiều hơn một. Đã đối chiếu state metadata/config/classes, model hash, ảnh, thứ tự bank và các tensor cache; tất cả điều kiện kiểm tra đều đạt. Việc đọc metadata trọng số không tương đương chạy lại inference trên máy local.

## 2. Những file thực sự phục vụ inference

| File hoặc thư mục | Vai trò | Có trong |
|---|---|---|
| `model.pt` | Trọng số đầy đủ của backbone và các nhánh đã train; nạp bằng đúng runtime của bundle | Cả hai |
| `config.json` | Kiến trúc, resize, mean/std, số lớp, K và cấu hình text nếu có | Cả hai |
| `class_names.json` | Đổi class index thành nhãn SUN397 đúng thứ tự train | Cả hai |
| `candidate_features.pt` | Vector ảnh 3.573 × 1.024, ràng buộc với chính `model.pt` | Cả hai |
| `candidate_pool.json` | Ánh xạ vị trí vector sang image_id, ảnh, hash và lớp | Cả hai |
| `candidates/` | Ảnh thật của pool; dùng để encode context đã chọn và hiển thị lên web | Cả hai |
| `runtime/tacs_visual_web.py` | Preprocessing, kiến trúc visual, chọn context, phân loại và attention | Cả hai |
| `vendor/dinov3/` | Source DINOv3 đi cùng model và license | Cả hai |
| `candidate_text.pt` | CLIP caption vectors/mask của pool, cùng thứ tự với bank ảnh | M-TACS |
| `captions.csv` | Caption đã sinh và kết quả audit; tra caption đúng ảnh candidate | M-TACS |
| `text_embeddings.pt` | Cache CLIP text cho các ảnh đã có caption; phục vụ tái sử dụng caption query đã xác minh | M-TACS |
| `caption_settings.json` | Prompt, policy, danh sách lớp và định danh Florence/CLIP | M-TACS |
| `caption_protocol.json` | Lưu ràng buộc/provenance của bước tạo caption | M-TACS |
| `runtime/mtacs_multimodal_web.py` | Selector ảnh + text, fusion query/candidate text và API inference | M-TACS |
| `runtime/sun397_caption_web.py` | Worker sinh/audit caption và encode CLIP cho ảnh mới | M-TACS |
| `models/florence/` | Florence-2-base Microsoft legacy: weights, processor, tokenizer và Python model code | M-TACS |
| `models/clip/` | CLIP ViT-B/32: PyTorch weights, tokenizer, config | M-TACS |
| `attention_config.json` | Mô tả cách trích heatmap | Cả hai |
| `manifest.json` | Danh sách file, kích thước và SHA-256; runtime kiểm tra trước khi nạp | Cả hai |

Các file `.pt` ở trên đã có. Không cần train thêm model attention, model selector độc lập hoặc tải lại DINOv3 pretrained `.pth`. `best.pt` gốc vẫn nên lưu để tiếp tục nghiên cứu; bundle dùng `model.pt` khi inference.

Không cần mang toàn bộ SUN397 sang máy chạy web: những ảnh context cần thiết đã có trong `candidates/`. Ảnh query được người xem upload. Với ảnh query mới, M-TACS sinh caption bằng model đã có trong bundle.

Giữ nguyên toàn bộ bundle khi triển khai. Dù một số file chỉ phục vụ kiểm chứng, runtime hiện kiểm SHA-256 tất cả file trong manifest. Xóa `provenance/`, `examples/`, file model phụ hoặc wheel trong bundle sẽ làm kiểm tra không đạt. Nếu muốn thu gọn, cần một quy trình đóng gói lại và kiểm tra lại riêng.

## 3. Kết quả kiểm thử đã lưu trong output

`smoke_test.json` của cả hai ghi:

- `passed: true`, `cold_subprocess_reload: true`.
- `source_is_bundle_only: true`, `test_injected_services: false`.
- `max_abs_logit_error: 0.0` so với model checkpoint trong kiểm thử export.
- Có hai vai trò attention: `query_classifier`, `candidate_selector`.

Riêng M-TACS có `fresh_florence_clip_integration: true`: bước export đã thử sinh caption mới bằng Florence/CLIP thật. TACS không dùng text nên trường này là false.

Đã đọc và giải mã ảnh overlay lưu trong `examples/prediction.json`: cả bốn overlay là PNG hợp lệ, có lưới attention 24 × 24 và giá trị biến thiên. Đây là dữ liệu kiểm thử đã lưu từ Kaggle, không phải một lần chạy lại inference trên Windows trong đợt kiểm kê này.

Heatmap là attention từ CLS tới patch ở block cuối, trung bình qua các head. Nó không phải Grad-CAM, không điều kiện theo lớp và không chứng minh quan hệ nhân quả. Ở M-TACS, heatmap candidate mô tả nhánh ảnh của selector, không biểu diễn mức đóng góp từng từ CLIP.

## 4. Luồng dữ liệu để nối vào web

### Trang TACS `/`

1. Backend nhận và kiểm tra bytes ảnh query.
2. Gọi `tacs_visual_web.BundlePredictor(...).predict(image_bytes, explain=True)`.
3. Predictor resize/normalize theo config, chấm toàn bộ pool, loại query trùng đúng ID/hash nếu có và chọn một context.
4. Phân loại query + context, trả nhãn/top-k, ảnh candidate và hai attention overlay.
5. Frontend hiển thị nhãn, confidence, context và heatmap; caption có trạng thái `not_used`.

### Trang M-TACS `/mtacs`

1. Backend nhận và kiểm tra bytes ảnh query.
2. Gọi `mtacs_multimodal_web.BundlePredictor(...).predict(image_bytes, explain=True)`.
3. Nếu hash ảnh khớp cache được xác minh thì tái sử dụng caption/vector; ảnh mới chạy Florence, lexical audit và CLIP.
4. Selector kết hợp tín hiệu ảnh và text hợp lệ; chọn một candidate rồi lấy đúng ảnh, caption và vector tương ứng.
5. Model phân loại với ảnh query/context và hai nhánh caption có mask/gate.
6. Trả nhãn/top-k, caption query/candidate, gate và hai heatmap.

Contract trả về đã tương thích phần lớn với frontend:

```text
prediction.label / confidence
top_k
evidence.query_caption.text / valid / status
evidence.candidate.image_url / image_id / caption / caption_valid
evidence.attention.query_classifier.image_url
evidence.attention.candidate_selector.image_url
evidence.fusion.q_gate / c_gate / semantic_mix     # M-TACS
```

Ảnh candidate và heatmap hiện được trả bằng data URL PNG, đúng loại URL frontend đã hỗ trợ. Không cần đưa toàn bộ model hoặc pool vào trình duyệt; chúng thuộc backend.

## 5. Những việc còn thiếu trong web hiện tại

Đã đối chiếu `TACS/apps/sun397_demo/app.py`, `assets/app.js`, `requirements.txt` và README:

1. `GET /api/status` vẫn hard-code `inference.available: false`.
2. `POST /api/classify` vẫn trả HTTP 503 và chưa gọi predictor.
3. Chưa có cấu hình đường dẫn bundle, bước nạp predictor hoặc worker inference.
4. Trang TACS đang mô tả ViT-S/16 trong sơ đồ; cần đổi thành ViT-L/16 và 384 theo config thực tế. Nên gọi đây là TACS visual với cấu hình L/16–384 của dự án, không mô tả là trùng toàn bộ cấu hình paper.
5. Frontend hiện đánh dấu caption sẵn sàng chỉ vì có chuỗi text. Cần đọc thêm `valid`/`caption_valid`/`status` để phân biệt caption được hiển thị với caption được model sử dụng.
6. Cần nối trạng thái loading/lỗi, giới hạn số lượt inference đồng thời và chọn device phù hợp.

Hai runtime đang bảo vệ source DINOv3 theo đường dẫn import. Nạp tuần tự hai predictor mặc định từ hai `vendor/dinov3` khác nhau trong cùng process sẽ vướng guard, dù nội dung code giống nhau. Khi tích hợp, dùng worker riêng cho từng model, hoặc một factory dùng source DINOv3 chung sau khi xác minh hash; không xóa guard một cách tùy ý. Nên giữ model trong worker để tránh nạp L/16 lại sau mỗi request.

## 6. Môi trường local còn thiếu

`.venv/Scripts/python.exe` hiện là Python 3.12.10 và có FastAPI/uvicorn, nhưng kiểm tra tìm module chưa thấy `torch`, `transformers`, `accelerate`, `einops`, `timm`, `torchvision`.

- Web shell có thể chạy, nhưng môi trường đó chưa đủ để chạy hai predictor.
- Cần PyTorch phù hợp Windows và GPU/CUDA của máy chạy demo; CPU có thể được chọn nhưng phải đo thời gian thực tế trước khi trình diễn.
- Florence legacy của bundle cần môi trường Transformers 4.49.x cùng các dependency tương ứng. `requirements.txt` trong M-TACS bundle ghi `transformers==4.49.0`.
- `environment.json` ghi Transformers 5.0.0 là môi trường kernel Kaggle; worker Florence đã chạy bằng môi trường con 4.49.x. Không lấy số 5.0.0 này làm yêu cầu cho Florence legacy.
- `offline_wheels/` có tokenizers/safetensors mang tag Linux `manylinux`; những wheel đó không cài trực tiếp được trên Windows. Cần bộ dependency phù hợp Windows hoặc một môi trường Linux tương thích.

Không cài package hoặc thay đổi môi trường trong đợt kiểm kê này.

## 7. Lưu ý về caption và metrics

Ví dụ đã lưu của M-TACS sinh caption query chứa từ `cathedral`, bị mask (`valid: false`, `status: masked`, `q_gate: 0`). Candidate caption trong ví dụ hợp lệ. Khi hiển thị, web nên ghi rõ caption query bị loại khỏi nhánh text; không khẳng định mọi caption nhìn thấy đều đóng góp vào dự đoán. Không thay bộ lọc hoặc dùng caption bị loại chỉ để phần demo trông đầy đủ hơn, vì sẽ đổi pipeline đã train.

CSV M-TACS có đủ 39.700 dòng, nhưng chỉ 15.828 caption có mask hợp lệ, khoảng 39,9%. Trong pool có 1.416/3.573 caption hợp lệ. Các caption bị mask có vector zero đúng quy định; bank text 3.573 × 512 khớp từng image_id với cache đầy đủ 39.700 × 512. Đây không phải thiếu CSV hoặc thiếu model: pipeline cố ý bỏ tín hiệu text bị loại, trong khi ảnh vẫn được sử dụng. Giao diện cần thể hiện trạng thái này đúng với lần dự đoán thực tế.

`metrics.json` hiện chỉ chứa validation lấy từ checkpoint:

| Model | Best epoch | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|
| TACS visual | 29 | 0,811083 | 0,808249 |
| M-TACS full_both | 19 | 0,802015 | 0,798399 |

Cả hai ghi `test: null`, `test_status: not_run_export_only`. Không dùng các điểm validation trên làm test score. History được đóng gói hiện ghi đến epoch 43 và 42 tương ứng; không suy từ `config.epochs = 100` rằng output này đã chứng minh hoàn tất 100 epoch. Điều này không cản trở dùng snapshot `best.pt` đã xuất để demo.

Không có test metrics gốc trong `provenance/unverified_original_metrics/` của hai bundle này. Muốn đưa test score lên web/báo cáo cần một lần đánh giá test đúng protocol hoặc cung cấp output test tương ứng.

## 8. Phạm vi kiểm tra

Đợt kiểm kê đọc config, manifest, provenance, history, state metadata, bank/cache, caption và ví dụ dự đoán; kiểm tra ảnh overlay cùng source frontend/backend. Kết quả kiểm toàn vẹn và đối chiếu tensor đã được ghi tại `TACS/tmp/sun397_web_bundle_audit_2026-11-01.json`. Cả hai bundle có danh sách lỗi rỗng và tất cả điều kiện trong trường `checks` đều là true.

Chưa thực hiện nạp hai model L/16 và suy luận mới trên máy Windows, chưa nối backend, chưa chạy benchmark latency, chưa đánh giá lại test set.
