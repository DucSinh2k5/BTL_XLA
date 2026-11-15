# TACS / M-TACS Research Project

Project nghiên cứu **Task-Aligned Context Selection**: tái lập paper, mở rộng ngữ cảnh ảnh-văn bản, đánh giá ablation và demo phân loại SUN397.

Thư mục này được tổ chức lại từ `XLA`. **78 notebook giữ nguyên nội dung, tên gốc và output**. Những thư mục chưa có triển khai chỉ có `.gitkeep`, không có notebook, checkpoint hay metrics giả.

## Bắt đầu từ đâu?

| Nhu cầu | Điểm bắt đầu |
| --- | --- |
| Tìm notebook theo task / method / dataset | [Danh mục notebook](notebooks/README.md) |
| Tái lập từng bảng trong paper | [Bản đồ các bảng 2–5, Figure 4](notebooks/01_paper_reproduction/README.md) |
| Chọn thí nghiệm cần chạy để viết báo cáo | [Rà soát kết quả và kế hoạch train ưu tiên](reports/paper_readiness/KE_HOACH_KET_QUA_BAO_CAO_TACS.md) |
| Các hướng cải tiến M-TACS | [Danh mục cải tiến](notebooks/02_improvements/README.md) |
| Tám notebook ablation text DTD | [Text-context ablation](notebooks/02_improvements/text_context_ablation/) |
| Bốn notebook so sánh phương pháp caption | [Caption-method ablation](notebooks/02_improvements/caption_ablation/) |
| Tạo / fine-tune caption | [Caption generation](notebooks/02_improvements/caption_generation/) |
| Segmentation ISIC + caption Qwen8B full_both | [Notebook tích hợp sinh caption và train segmentation 448](notebooks/02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/README.md) |
| Đọc kết quả tám thí nghiệm đã chạy | [Báo cáo output thật](experiments/completed/text_context_ablation/dtd/analysis/DANH_GIA_ABLATION_DTD_01_07.md) |
| Các CSV caption và reference | [Caption DTD](data/captions/dtd/) / [Reference DTD](data/annotations/dtd/) |
| Demo web (SUN397 + ISIC) | [Chạy web demo](#chạy-web-demo) / [Backend, frontend](apps/sun397_demo/) |
| Tiến độ và phần còn thiếu | [Trạng thái project](docs/progress/STATUS.md) |
| Tìm lại đường dẫn cũ | [Nhật ký di chuyển](docs/MIGRATION.md) / [JSON đối chiếu](manifests/migration_plan.json) |

## Cấu trúc

```text
TACS/
  apps/                    Web demo FastAPI: phân loại SUN397 + phân đoạn ISIC (TACS và M-TACS)
  configs/                 Config có sẵn, ví dụ đường dẫn dataset/model
  data/
    raw/                   Chỗ dành cho dataset; dữ liệu lớn hiện để ngoài project
    processed/             Chỗ dành cho dữ liệu đã xử lý
    captions/dtd/          CSV caption, tách decoder LoRA và grounded controls
    annotations/dtd/       Reference caption và thông tin nguồn gán nhãn
    archives/              Bundle đầu vào cũ
  docs/                    Paper, quy trình, hướng dẫn, lịch sử và tiến độ
  notebooks/
    00_data_preparation/   Dành cho notebook chuẩn bị dữ liệu theo dataset
    01_paper_reproduction/
      classification/     table -> method -> dataset -> notebook
      segmentation/       table -> method -> dataset -> notebook
      figure_04_attention/
    02_improvements/       hướng cải tiến -> method -> dataset -> notebook
    03_analysis_and_evaluation/
    90_archive/            Bản lịch sử; một số vẫn là source của builder
  src/
    captioning/            Code Florence, screening, fine-tune, grounded rerank
    training/              Code nguồn các cell ablation
    models/                Chưa tách model khỏi notebook
    selection/             Chưa tách selector khỏi notebook
    evaluation/            Chưa tách evaluator khỏi notebook
  tools/                   Builder, caption CLI, Figure 4, quản lý cấu trúc
  experiments/
    planned/               Chưa có run cụ thể
    running/               Chưa nhập tiến trình Kaggle đang chạy
    completed/             Notebook đã có output và bằng chứng phân tích
  artifacts/               Checkpoint / feature bank / embedding / captioner
  reports/                 Tổng hợp nghiên cứu, hình và bằng chứng kiểm duyệt
  tests/                   Regression tests của caption và ablation
  manifests/               Catalog, SHA-256, migration và kết quả kiểm tra
```

## Quy ước quan trọng

- `visual_only` trong ablation M-TACS **vẫn có ảnh context**; không đồng nhất với `no_context` của paper.
- Notebook trong khu vực tái lập là implementation liên quan đến paper, chưa tự động được coi là tái lập đúng toàn bộ protocol. Có các backbone, split và phiên bản selector khác nhau.
- Tên file gốc có thể sai dataset; thư mục mới và catalog dùng cấu hình bên trong notebook. Ví dụ `tacs-chuanpaper-sun-b16 (2).ipynb` thực sự dùng Stanford Cars.
- Các file trong `experiments/completed` là bằng chứng lưu từ lần chạy trước. Không Run All để “cập nhật” kết quả lịch sử.
- Không suy ra phương pháp tốt nhất chỉ từ tên folder hoặc một seed. Giữ nguyên CSV, split, candidate pool, budget và metric khi so sánh.
- Caption reference hiện có nguồn teacher AI, không phải toàn bộ đã được người độc lập kiểm duyệt. Giữ nguyên provenance khi viết báo cáo.
- Checkpoint nhập vào `artifacts/checkpoints/imported_unverified` chưa được xác nhận tương thích với web. Demo chưa giả lập dự đoán hay attention.

## Chạy web demo

Web gồm 4 trang trong một giao diện studio, đều dùng DINOv3 ViT-L/16 và mô hình thật (không giả lập kết quả):

| Trang | Nội dung |
| --- | --- |
| `http://127.0.0.1:8000/` | Phân loại SUN397, TACS chỉ dùng ảnh (ảnh 384) |
| `http://127.0.0.1:8000/mtacs` | Phân loại SUN397, M-TACS full_both (sinh caption Florence-2 + CLIP cho ảnh mới) |
| `http://127.0.0.1:8000/isic` | Phân đoạn ISIC 2017, TACS chỉ dùng ảnh (ảnh 448) |
| `http://127.0.0.1:8000/isic/mtacs` | Phân đoạn ISIC 2017, M-TACS full_both |

### 1. Cài môi trường

Cài `torch` + `torchvision` đúng với CUDA của máy trước (bản dùng khi xuất bundle: torch 2.10.0, torchvision 0.25.0), sau đó:

```powershell
python -m pip install -r apps/sun397_demo/requirements-inference.txt
```

File này cố định `transformers==4.49.0` cho Florence-2. Nếu môi trường chính cần Transformers bản khác, hãy tạo môi trường riêng cho web.

### 2. Đặt model bundle

Bundle (trọng số, kho ứng viên, Florence-2, CLIP) rất lớn nên **không có trên GitHub**. Lấy 4 file `.zip` từ output Kaggle của các notebook `notebooks/04_deployment/`, giải nén rồi đặt đúng tên thư mục:

```text
artifacts/deployment/
├── sun397/
│   ├── tacs_visual/        <- sun397_tacs_visual_vitl16_web_bundle.zip
│   └── mtacs_full_both/    <- sun397_mtacs_full_both_vitl16_web_bundle.zip
└── isic2017/
    ├── tacs_visual/        <- isic2017_tacs_vitl16_448_web_bundle.zip
    └── mtacs_full_both/    <- isic2017_mtacs_vitl16_448_web_bundle.zip
```

Mỗi file zip có sẵn một thư mục gốc (ví dụ `sun397_tacs_visual_vitl16_web_bundle/`): đổi tên thư mục đó thành tên ở trên. Giữ nguyên toàn bộ file bên trong; web kiểm tra mã băm của từng file khi nạp.

Nếu bundle nằm ở chỗ khác, khai báo đường dẫn bằng biến môi trường (hoặc cùng tên khóa viết thường trong `configs/web.local.json`, sao từ `configs/web.example.json`):

```powershell
$env:SUN397_TACS_BUNDLE  = "D:\bundles\sun397\tacs_visual"
$env:SUN397_MTACS_BUNDLE = "D:\bundles\sun397\mtacs_full_both"
$env:ISIC_TACS_BUNDLE    = "D:\bundles\isic2017\tacs_visual"
$env:ISIC_MTACS_BUNDLE   = "D:\bundles\isic2017\mtacs_full_both"
```

Khóa tương ứng trong `configs/web.local.json`: `tacs_bundle`, `mtacs_bundle`, `isic_tacs_bundle`, `isic_mtacs_bundle`. Tùy chọn khác: `SUN397_DEVICE` (`auto` / `cuda` / `cpu`), `SUN397_SHARED_PROCESS=1` để hai mô hình SUN397 dùng chung một tiến trình (tiết kiệm RAM).

### 3. Khởi động

Từ thư mục `TACS`:

```powershell
.\start_web.ps1
```

Script dùng `..\.venv\Scripts\python.exe` nếu có, ngược lại dùng `python` trong PATH. Đổi cổng: `.\start_web.ps1 -Port 8080`. Lệnh tương đương:

```powershell
python -m uvicorn --app-dir apps sun397_demo.app:app --host 127.0.0.1 --port 8000
```

- Mô hình được nạp sẵn khi khởi động (có thể mất vài phút). Đợi badge **Sẵn sàng** trên trang rồi mới dự đoán.
- Chỉ chạy **một** tiến trình server; không dùng `--workers` lớn hơn 1 hoặc `--reload`, vì mỗi tiến trình sẽ nạp lại toàn bộ mô hình.
- Kiểm tra nhanh: `http://127.0.0.1:8000/healthz`, `/readyz`, `/api/status`.
- Thiếu bundle thì trang vẫn mở, nhưng API dự đoán trả về `503` kèm thông báo, không giả lập kết quả.
- Ảnh tải lên chỉ được ghi tạm trong `.runtime/` lúc suy luận rồi bị xóa.

### 4. Test của web

```powershell
python -m pytest -q apps/sun397_demo/tests
node apps/sun397_demo/tests/test_studio_navigation.cjs
```

Các file `.cjs` khác trong `apps/sun397_demo/tests/` chạy tương tự bằng `node`.

## Kiểm tra nhanh

Chạy trong thư mục `TACS`:

```powershell
python tools/project_management/validate_layout.py
python -m pytest -q
```

Cần cài dependency trước khi chạy test/web; xem [hướng dẫn môi trường](docs/guides/RUN_PROJECT.md). Việc tổ chức thư mục không tự cài package, tải model, chạy train hoặc thay đổi phiên bản Transformers.

Các dataset lớn, môi trường `.venv` và các bài không thuộc TACS vẫn ở `XLA`. `configs/paths.local.example.json` ghi lại vị trí dataset hiện có; notebook Kaggle tiếp tục dùng config `/kaggle/input/...` như trước.
