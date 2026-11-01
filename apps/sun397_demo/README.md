# SUN397: TACS / M-TACS demo với model thật

Hai trang dùng **DINOv3 ViT-L/16, ảnh 384 × 384, K=1, 397 lớp**:

- `/`: TACS visual, chỉ ảnh query và ảnh context.
- `/mtacs`: M-TACS full_both, dùng caption query/candidate cho selector và fusion khi caption hợp lệ.

Đây là cấu hình L/16–384 của dự án, khác cấu hình S/16–256 trong paper.

## Chạy

Trong thư mục `TACS`:

```powershell
.\start_web.ps1
```

Hoặc từ thư mục `XLA`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir TACS/apps sun397_demo.app:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000/> hoặc <http://127.0.0.1:8000/mtacs>. Workers tự khởi động khi máy chủ khởi động, không cần bấm dự đoán để nạp model. HTTP và giao diện vẫn phản hồi trong lúc nạp; đợi badge **Sẵn sàng** trước khi dự đoán.

Chỉ chạy một process Uvicorn. Nhiều server hoặc `--workers` lớn hơn 1 sẽ nhân bản model trong RAM/VRAM. Không cần `--reload` khi demo.

## Bundle

Giữ nguyên toàn bộ hai thư mục:

```text
TACS/artifacts/deployment/sun397/
├── tacs_visual/
│   ├── model.pt
│   ├── candidate_features.pt
│   ├── candidate_pool.json
│   ├── candidates/
│   ├── runtime/
│   ├── vendor/dinov3/
│   └── ...
└── mtacs_full_both/
    ├── model.pt
    ├── candidate_features.pt
    ├── candidate_text.pt
    ├── captions.csv
    ├── text_embeddings.pt
    ├── models/florence/
    ├── models/clip/
    └── ...
```

Các file khác trong manifest cũng phải được giữ. Model, caption policy, class order, pool và trọng số không bị sửa. Di chuyển trong cùng ổ không nhân đôi dữ liệu; `relocation.json` ghi đường dẫn trước/sau.

## Nạp sẵn và inference

`inference.py` hỗ trợ worker riêng hoặc worker chung. Máy hiện tại dùng `shared_process: true`: một process Python nạp hai model để giảm bộ nhớ thư viện bị lặp. Chỉ dùng source DINOv3 chung sau khi kiểm tra hai bundle có cùng hash source DINOv3 và visual runtime; vẫn kiểm tra đầy đủ manifest từng bundle. Trọng số, feature bank và đường dự đoán của TACS/M-TACS tách biệt. Florence-2 và CLIP được nạp sẵn cho nhánh M-TACS và tái sử dụng cho ảnh mới. TACS visual không gọi chúng.

Workers nạp lần lượt để giảm bộ nhớ đỉnh. Một hàng đợi dùng chung giới hạn một lượt inference tại một thời điểm. Ảnh upload được kiểm tra trước, giữ nguyên bytes cho cache/self-exclusion, chỉ ghi tạm trong runtime và xóa sau lượt suy luận. Không ghi thêm file vào bundle.

Checkpoint backbone được memory-map; model dựng qua Accelerate với parameter rỗng rồi nạp strict bằng tensor đã map. Các buffer RoPE được khởi tạo thật và giữ nguyên. Cách này giảm bản sao trọng số trên CPU, không đổi precision, kiến trúc, trọng số hoặc caption policy. Worker vẫn kiểm tra toàn bộ manifest và mọi ràng buộc model/bank/cache như runtime gốc.

Trên CPU Windows, trọng số dùng mapping chỉ đọc để tránh đặt chỗ bộ nhớ ảo cho bản sao ghi được. Florence có thêm cache FP32 chuyển chính xác từ F16 bằng từng khối nhỏ, khoảng 0,9 GiB, ràng buộc SHA-256 với file nguồn. Đây là cùng phép chuyển F16 → FP32 của loader cũ, không lượng tử hóa hoặc huấn luyện lại. CLIP được nạp theo chế độ giảm RAM với trọng số chỉ đọc. Processor, tokenizer, generation, audit caption và kiến trúc model giữ nguyên.

Máy hiện tại đặt cache tại `C:/Users/Dell/AppData/Local/TACS/caption_fp32` vì ổ F gần đầy; đường dẫn nằm trong `configs/web.local.json`. Cache có thể tạo lại, nhưng xóa nó sẽ khiến lần mở sau phải chuẩn bị lại khoảng 0,9 GiB. Các file model gốc vẫn ở `artifacts/deployment/sun397/`. Worker theo dõi process API và tự kết thúc khi API dừng để giải phóng model.

Web hiển thị nhãn, xác suất, ảnh candidate và hai attention overlay thật. M-TACS có caption/mask của query và candidate. Caption bị loại được đánh dấu không sử dụng trong nhánh text. Heatmap là CLS-to-patch self-attention block cuối, không phải Grad-CAM hoặc giải thích nhân quả.

### Nhãn context và so sánh ground truth

Cả hai trang hiển thị nhãn annotation của context đã chọn, nhãn query mà model dự đoán, độ tin cậy softmax, ground truth query và kết quả đối chiếu đúng/sai. Nhãn context lấy từ chính image_id/vị trí trong candidate pool đã lưu.

`labels.py` xác minh hash các file annotation, danh sách lớp, pool và CSV caption theo manifest rồi tạo chỉ mục SHA-256 nội dung ảnh. Annotation lớp lấy từ train/validation/test CSV; caption CSV chỉ cung cấp liên kết image_id ↔ hash ảnh, không được dùng để suy ra nhãn từ câu caption. Với hai bundle hiện tại, chỉ mục bao phủ 39.700 ảnh SUN397, dùng chung cho hai trang. Đổi tên file không làm thay đổi nhãn đã xác minh; ảnh bị resize/re-encode có hash khác sẽ không tự động được nhận là ảnh dataset.

Ảnh ngoài chỉ mục hiển thị “Chưa xác định” cho ground truth. Người dùng có thể chọn nhãn đối chiếu trong danh sách SUN397 trước khi dự đoán; nhãn này được đánh dấu **người dùng cung cấp, chưa xác minh**, không được hiển thị như annotation dataset. Không cho ghi đè nhãn dataset đã xác minh bằng nhãn nhập khác. Không lấy tên file, caption hay prediction làm ground truth.

Metadata nhãn được xử lý ở API; phần so sánh được tính sau inference. Nhãn đối chiếu không được truyền vào worker/selector/classifier, nên không đổi đầu vào mô hình hoặc dùng nhãn query để chọn context.

## Môi trường

Máy hiện tại có `.venv` Python 3.12 cho API, trong khi PyTorch CPU đã có tại `F:/Python3_10/python.exe`. `configs/web.local.json` chọn Python 3.10 này cho workers và overlay riêng `.runtime/dependencies_py310/`. Không cài đè package toàn cục hoặc sao chép PyTorch sang `.venv`.

Trên máy khác, dùng môi trường inference riêng và cài:

```powershell
python -m pip install -r apps/sun397_demo/requirements-inference.txt
```

Chọn cặp torch/torchvision phù hợp OS/CUDA theo PyTorch trước. Reference Kaggle: torch 2.10.0 / torchvision 0.25.0; máy hiện tại tái sử dụng torch 2.13.0 CPU / torchvision 0.28.0 CPU. Florence legacy dùng Transformers 4.49.0. Wheel Linux trong bundle không cài trực tiếp lên Windows. Inference không tải model hoặc gọi API ngoài.

Biến môi trường ghi đè `configs/web.local.json`:

| Biến | Ý nghĩa |
|---|---|
| `SUN397_WORKER_PYTHON` | Python chạy model, có thể khác Python của API |
| `SUN397_WORKER_DEPENDENCIES` | Overlay của đúng Python worker, tùy chọn |
| `SUN397_TACS_BUNDLE`, `SUN397_MTACS_BUNDLE` | Bundle root |
| `SUN397_DEVICE` | `auto` (mặc định), `cpu`, `cuda`, `cuda:0` |
| `SUN397_CPU_THREADS` | Thread CPU mỗi worker, mặc định 4 |
| `SUN397_SHARED_PROCESS` | 1 dùng một process cho hai model khi source đã xác minh giống nhau; máy hiện tại bật chế độ này và dùng 2 thread CPU |
| `SUN397_CAPTION_WEIGHT_CACHE` | Thư mục cache FP32 của Florence trên ổ còn đủ chỗ |
| `SUN397_AUTOLOAD` | Mặc định 1; 0 dành cho test/phát triển không nạp model |
| `SUN397_WEB_RUNTIME` | Thư mục log, cache và request tạm |
| `SUN397_STARTUP_TIMEOUT` | Timeout mỗi worker, mặc định 1800 giây |
| `SUN397_PREDICTION_TIMEOUT` | Timeout một lượt, mặc định 600 giây |

Mẫu: `configs/web.example.json`. Không đưa model vào static frontend; không đưa local config và dữ liệu lớn lên Git.

## Trạng thái và API

- `GET /healthz`: API còn hoạt động; không khẳng định model đã nạp.
- `GET /readyz`: 200 khi cả hai model sẵn sàng, 503 khi đang nạp/lỗi.
- `GET /api/status?model=tacs|mtacs`: trạng thái riêng và kiến trúc L/16–384.
- `GET /api/classes`: danh sách nhãn cho ô chọn nhãn đối chiếu.
- `POST /api/images/inspect`: kiểm JPEG/PNG/WebP, tối đa 12 MB / 36 triệu pixel.
- `POST /api/classify?model=tacs|mtacs`: multipart field `file`, trả prediction và evidence thật.

`/api/images/inspect` trả thêm `ground_truth` khi nhận diện được ảnh. `/api/classify` nhận tùy chọn `reference_class_id`, trả `ground_truth`, `comparison` và `evidence.candidate.label`. Trường `verified` phân biệt annotation đã xác minh với nhãn người dùng cung cấp.

Ở chế độ worker riêng, một worker lỗi không chặn model còn lại. Ở chế độ worker chung, lỗi khi nạp làm cả hai trang chưa sẵn sàng. Xem `.runtime/sun397_demo/logs/both.log` (worker chung) hoặc `tacs.log` / `mtacs.log` (worker riêng), sửa dependency/tài nguyên rồi khởi động lại. Không trả prediction mẫu khi suy luận thất bại.

Bundle chỉ có validation metrics từ best checkpoint, không có test score từ lần export. Ảnh minh họa frontend là ảnh nhân tạo, không phải ảnh SUN397 có nhãn chuẩn.

## Kiểm thử

```powershell
python -m pytest apps/sun397_demo/tests -q
node apps/sun397_demo/tests/test_frontend_labels.cjs
node apps/sun397_demo/tests/test_frontend_navigation.cjs
```

Test API/IPC dùng fixture có đánh dấu. Model thật được kiểm riêng qua HTTP sau khi workers sẵn sàng.

### Cập nhật sửa bộ nhớ và đối chiếu nhãn ngày 01/11/2026

- 41 test đạt; kiểm tra DOM/cú pháp JavaScript đạt cho cả hai trang.
- `/readyz` trả 200 với hai model trong worker chung; Florence và CLIP đã nạp.
- Đã chạy thật ba lượt HTTP: TACS với ground truth đã xác minh, M-TACS với ground truth đã xác minh, M-TACS sinh caption mới với nhãn đối chiếu nhập tay.
- Các response có nhãn context, nhãn dự đoán, confidence, ground truth/source/verified, comparison và hai heatmap 24 × 24. Nhãn/context vẫn trùng bản export.
- Trên máy đang thiếu bộ nhớ, cấu hình 2 thread CPU đo được khoảng 42 giây cho TACS, 43 giây cho M-TACS dùng caption cache và 71 giây cho M-TACS sinh caption mới. Đây là kiểm tra chức năng trên một ảnh, không phải benchmark hoặc đánh giá accuracy/F1.
- Bằng chứng mới: `manifests/sun397_web_ground_truth_20261101.json`.

### Bằng chứng lần tích hợp đầu ngày 01/11/2026

- 20 test đạt; JavaScript qua kiểm tra cú pháp.
- `/readyz` trả 200, cả hai model ready và M-TACS xác nhận đã nạp Florence/CLIP.
- TACS dự đoán thật qua HTTP khoảng 22,3 giây trên CPU; nhãn/context trùng export, chênh xác suất khoảng 5,1e-7.
- M-TACS sinh caption mới từ ảnh upload rồi phân loại, khoảng 28,4 giây trên CPU; nhãn/context trùng export, chênh xác suất khoảng 1,4e-6.
- Cả hai trả hai heatmap thật 24 × 24. Caption query trong ví dụ M-TACS bị mask đúng policy và q_gate bằng 0.

Đây là hai lượt kiểm tra chức năng trên một ảnh ví dụ, không phải benchmark tốc độ hoặc đánh giá lại accuracy/F1. Bằng chứng: `manifests/sun397_web_integration_20261101.json` và các response JSON được dẫn trong file đó.

### Giữ trạng thái khi chuyển mô hình

Nhấn liên kết TACS hoặc M-TACS trong web để chuyển trang trong cùng tài liệu. Mỗi trang giữ riêng ảnh đã chọn, nhãn đối chiếu, kết quả, confidence, ảnh context, caption và heatmap. Yêu cầu dự đoán đang chạy vẫn hoàn thành ở trang đã gửi; quay lại để xem kết quả. Nút Back/Forward cũng giữ trạng thái.

Dữ liệu này chỉ nằm trong bộ nhớ của tab trình duyệt. Reload/F5 hoặc đóng tab sẽ bắt đầu phiên mới cho cả hai trang; mở liên kết trong tab mới cũng là một phiên riêng. Nút xóa ảnh chỉ xóa dữ liệu của mô hình đang mở. Sau khi cập nhật frontend, reload một lần để nhận mã mới.

Kiểm tra phiên bản này: 21 test API đạt, kiểm tra hiển thị nhãn cho cả hai mô hình và 3 test giữ trạng thái đạt, JavaScript hợp lệ. Các test điều hướng dùng DOM/network fixture và không chạy suy luận thật.

Kiểm tra trực tiếp trên trình duyệt: mỗi trang giữ riêng ảnh minh họa và nhãn đối chiếu; TACS hoàn thành suy luận thật trong khi đang mở M-TACS, quay lại bằng liên kết và Back vẫn thấy kết quả/context. Reload xác nhận cả hai trang trở về trạng thái mới. Ảnh bằng chứng: 	mp/web_model_state_20261101.jpg. Đây là kiểm tra chức năng bằng ảnh minh họa, không phải đánh giá accuracy.
