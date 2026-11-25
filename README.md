# TACS / M-TACS Research Project

Project nghiên cứu **Task-Aligned Context Selection** (TACS, Guo et al., CVPR 2026): tái lập paper, mở rộng ngữ cảnh ảnh–văn bản (M-TACS), đánh giá ablation và demo web phân loại SUN397 + phân đoạn ISIC 2017.

## Tóm tắt nhanh (dành cho người chấm)

- **TACS:** một bộ chọn học được lấy 1 ảnh ngữ cảnh (K = 1) từ kho 20% tập train, ghép với ảnh truy vấn rồi đưa vào ViT (DINOv3). Bộ chọn học bằng Gumbel-Softmax + REINFORCE, phần thưởng = loss khi không có ngữ cảnh − loss khi có ngữ cảnh.
- **Cải tiến 1:** nâng backbone ViT-S → **ViT-L/16** và ảnh 256 → **384/448**, kèm cách tiết kiệm bộ nhớ (batch logic 64 / batch tính 8, phát lại RNG, gradient checkpointing).
- **Cải tiến 2 (M-TACS):** thêm caption (Florence-2 / Qwen3-VL → CLIP) của ảnh ngữ cảnh, trộn qua cổng học được; bộ lọc chống lộ nhãn `dtd_lexical_v2`.

| Kết quả chính (DTD split 1, ViT-L/16, 384, seed 42) | Test Macro-F1 (%) |
| --- | :---: |
| TACS chỉ dùng ảnh (A01) | 84.82 |
| M-TACS tốt nhất, caption ứng viên + gated residual (S01) | **85.30** |
| Caption ở cả ảnh truy vấn (A05, `full_both`) | 80.91 |

Mức chênh 0.48 điểm mới đo với 1 seed, chưa đủ để khẳng định. Kết luận chính của phần ablation: văn bản chỉ có ích khi gắn với ảnh ứng viên; gắn vào ảnh truy vấn làm giảm điểm. Toàn bộ bảng số liệu đã đối chiếu output có trong [Cẩm nang hệ thống](CAM_NANG_HE_THONG_TACS.md), mục 5.

Nguyên tắc của repo: notebook đã chạy giữ nguyên tên gốc và output. Thư mục chưa có triển khai chỉ có `.gitkeep`; không có checkpoint hay metric giả. Mọi kết quả chưa chạy test (ví dụ bundle web, study06) đều được ghi rõ là chưa có.

## Bắt đầu từ đâu?

| Nhu cầu | Điểm bắt đầu |
| --- | --- |
| Hiểu toàn bộ hệ thống và mọi điểm số | [Cẩm nang hệ thống TACS & M-TACS](CAM_NANG_HE_THONG_TACS.md) |
| Slide, kịch bản thuyết trình, kiến thức ôn tập | [reports/presentation/](reports/presentation/) |
| Bảng mọi notebook theo thời gian (cấu hình + điểm) | [Tổng hợp notebook](reports/presentation/TONG_HOP_NOTEBOOK_THEO_THOI_GIAN.md) |
| Sáu nghiên cứu kiến trúc M-TACS trên DTD (S01–S06) | [Multimodal architecture ablation](notebooks/02_improvements/multimodal_architecture_ablation/README.md) |
| Khảo sát độ mới so với các công trình liên quan | [Khảo sát độ mới của M-TACS](reports/Khảo%20sát%20độ%20mới%20của%20M%20TACS.md) |
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

Quy ước đặt tên: số đầu tên thư mục notebook là **thứ tự công việc** (00 chuẩn bị dữ liệu → 01 tái lập paper → 02 cải tiến → 03 phân tích → 04 triển khai web, 90 lưu trữ). Bên trong luôn đi theo `nhiệm vụ / phương pháp / dataset / notebook`.

```text
TACS/
  CAM_NANG_HE_THONG_TACS.md  Cẩm nang: kiến trúc, lịch sử cải tiến, MỌI điểm số đã xác minh
  README.md                  File này
  start_web.ps1              Khởi động web demo
  *.ipynb (ở gốc)            Notebook chạy gần đây, giữ nguyên chỗ để không gãy đường dẫn (bảng bên dưới)

  notebooks/                 Toàn bộ notebook, chia theo giai đoạn
    00_data_preparation/       Chuẩn bị dữ liệu theo dataset (hiện mới có chỗ trống)
    01_paper_reproduction/     Tái lập paper TACS: classification / segmentation theo đúng số bảng
                               trong paper (table_02_main_results, ...), figure_04_attention
    02_improvements/           Các hướng cải tiến (M-TACS), mỗi thư mục là một hướng:
      text_context_ablation/     Chuỗi A: đặt caption ở đâu (truy vấn / ứng viên / cả hai), 8 bản
      caption_ablation/          Chuỗi C: so sánh nguồn caption (Florence, Qwen3-VL, LoRA)
      caption_generation/        Sinh và fine-tune caption
      multimodal_architecture_ablation/  Chuỗi S: cách trộn văn bản, study 01–06 (candidate_only)
      segmentation/              M-TACS cho phân đoạn ISIC 2017
      adaptive_context/, visual_top3/, multimodal_top3/, query_text/, query_candidate_text/,
      zero_shot_siglip/          Các hướng thử phụ (K = 3, ngữ cảnh thích nghi, zero-shot)
    03_analysis_and_evaluation/  Dành cho notebook phân tích riêng (hiện trống)
    04_deployment/             Train + xuất bundle cho web (sun397/, isic2017/)
    90_archive/                Bản lịch sử; một số vẫn là nguồn của builder

  experiments/               Bằng chứng chạy thí nghiệm
    completed/                 Notebook đã chạy có output + CSV + phân tích (chuỗi A, C, LoRA)
    planned/, running/         Chỗ trống dành cho thí nghiệm dự kiến / đang chạy

  src/                       Code Python dùng chung, được nhúng vào notebook
    captioning/                Sinh caption Florence/Qwen, bộ lọc chống lộ nhãn, rerank CLIP
    training/                  Code các cell ablation DTD (A, S01–S06), train ISIC, train web SUN397 + bản vá bộ nhớ
    models/, deployment/       Dựng model và kiểm tra bundle cho web
    selection/, evaluation/    Chưa tách khỏi notebook
  tools/                     Script hỗ trợ
    notebook_builders/         Sinh notebook từ src/ (mỗi study có một builder, có test đi kèm)
    captioning/, visualization/, deployment/, project_management/  CLI caption, vẽ attention, kiểm tra cấu trúc
  tests/                     Test tự động cho builder, caption và ablation
  apps/sun397_demo/          Web demo FastAPI: phân loại SUN397 + phân đoạn ISIC (TACS và M-TACS)

  data/                      Caption CSV và reference (dataset lớn để ngoài repo)
    captions/, annotations/    Caption đã sinh và caption tham chiếu
    raw/, processed/, archives/  Chỗ dành cho dataset gốc / đã xử lý / bundle đầu vào cũ
  artifacts/                 Checkpoint, feature bank, embedding, bundle web (không đưa lên GitHub vì quá lớn)
  configs/                   Đường dẫn dataset/model, cấu hình web và train

  reports/                   Báo cáo và bằng chứng
    presentation/              Slide chuẩn, kịch bản người nói, kiến thức ôn tập, bảng notebook theo thời gian
    bao_cao_bai_tap_lon/       Đề cương và nội dung báo cáo bài tập lớn (md + docx)
    cam_nang_nhom/             Cẩm nang học tập và bàn giao cho nhóm (md + docx)
    multimodal_architecture_ablation/, caption_ablation/, caption_quality/, caption_decoder_lora/
                               Báo cáo đánh giá của từng chuỗi thí nghiệm (kèm JSON bằng chứng)
    paper_tables/, paper_readiness/  Bảng so sánh kiểu paper, kế hoạch kết quả cho báo cáo
    annotation_review/, figures/     Kiểm duyệt caption tham chiếu, hình
    Khảo sát độ mới của M TACS.md    Khảo sát công trình liên quan
  research_notes/            Ghi chú nguồn của bài khảo sát
  docs/                      Paper gốc, giao thức, hướng dẫn (spec kiến trúc, sửa lỗi bộ nhớ), tiến độ
  manifests/                 Danh mục notebook, SHA-256, nhật ký di chuyển thư mục
```

Các notebook ở thư mục gốc (5 bản đầu đã chạy và giữ output; mã là tên gọi dùng trong cẩm nang):

| Notebook | Nội dung | Mã |
| --- | --- | :---: |
| `dtd-architecture-study-01.ipynb` | DTD, caption ứng viên + gated residual: **bản tốt nhất 85.30** | S01 |
| `stydu2_da_train.ipynb` | DTD, S01 + chuẩn hóa L2 vector caption | S02 |
| `study3_datrain.ipynb` | DTD, S02 + loss căn chỉnh Sinkhorn (m-LTM) | S03 |
| `tacs_isic_segement_448.ipynb` | ISIC 2017, TACS chỉ dùng ảnh, ảnh 448 | I01 |
| `isic-segmen-captions.ipynb` | ISIC 2017, M-TACS `full_both` + caption Qwen3-VL-8B | I02 |
| `tacs_web.ipynb` | SUN397, train TACS ViT-L/16 và xuất bundle web (đã vá lỗi bộ nhớ; bản này chưa có output) | — |
| `mtacs-sun.ipynb` | SUN397, M-TACS cho web (chưa có output) | — |

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
