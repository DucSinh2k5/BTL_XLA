# ISIC-2017 segmentation: train và xuất bộ model cho web

Hai notebook độc lập, chưa có output train thực tế:

| Notebook | Đầu vào mô hình | Caption |
| --- | --- | --- |
| [01 TACS visual](01_tacs_visual/isic2017-tacs-visual-vitl16-448-train-export-web.ipynb) | Ảnh query và một ảnh context do selector chọn | Không dùng |
| [02 M-TACS full_both](02_mtacs_full_both/isic2017-mtacs-full-both-florence-vitl16-448-train-export-web.ipynb) | Query + context; caption cả hai trong fusion và selector | Tự tạo/resume bằng Florence-Prefix, encode CLIP |

Mặc định hai bản cùng DINOv3 **ViT-L/16, ảnh 448, 100 epoch, seed 42, effective batch 8, context K=1**.
Sửa ngân sách ngay cell cấu hình trước khi chạy lần đầu nếu cần 50 epoch. Không đổi ngân sách dưới cùng output khi resume.
Đây là bản mở rộng L/16-448, không phải kết quả nguyên bản S/16-256 trung bình ba seed của paper.

Backbone đóng băng. Sau **từng** transformer block có gated cross-attention query nhận context; bốn độ sâu vào DPT nhẹ.
Selector học utility trên CLS của encoder dùng chung. Huấn luyện kết hợp hard Gumbel ST và categorical policy gradient,
reward bằng mức giảm BCE+Dice so với query không context. Lambda=0.5, tau=0.1; entropy bổ sung mặc định bằng 0.
Các chi tiết utility projection, DPT và gate được triển khai rõ trong runtime; không tuyên bố đồng nhất từng chi tiết code chưa công bố của tác giả.

## Kaggle Inputs

Cho cả hai notebook:

- ISIC-2017 **Part1 segmentation**, gồm ảnh và mask ở ba split chính thức: train 2.000, val 150, test 600.
- Local DINOv3 source có `dinov3/hub/backbones.py` và các file license.
- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` đúng bản L/16 LVD1689M.
- GPU; môi trường có PyTorch tương thích DINOv3/SDPA, NumPy, Pillow, tqdm, Matplotlib.

Riêng M-TACS thêm:

- Complete **microsoft/Florence-2-base legacy**: config, toàn weights, tokenizer, processor và Python remote code.
- Complete **CLIP ViT-B/32** (projection 512), gồm tokenizer và weights.
- Nếu kernel không có Transformers 4.49: offline wheels cho Transformers 4.49.0, tokenizers 0.21.x, huggingface-hub <1,
  và einops/timm/accelerate nếu thiếu. Wheels phải tương thích đúng Python/platform của Kaggle.
- Có thể dùng Input `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi` đã dùng trước đây.
- **Không cần upload captions.csv**: notebook tạo toàn bộ 2.750 caption từ ảnh và resume cache theo từng ảnh.

Không bật Internet. Chỉ caption child dùng một dependency overlay riêng; không downgrade kernel train.
Florence và CLIP dùng FP32 trong cả bước tạo caption và adapter demo; precision được ghi trong provenance.
Tokenizer parallelism được tắt từ preflight; DataLoader mặc định không fork worker.

Dataset có thể là thư mục chuẩn bị `train/images,masks`, `val/images,masks`, `test/images,masks`, hoặc các thư mục ISIC gốc
`ISIC-2017_Training_Data`, `ISIC-2017_Training_Part1_GroundTruth` và bản Validation/Test tương ứng.
Nếu tự dò thấy nhiều bản dataset/model, đặt chính xác các biến đường dẫn trong cell cấu hình.

Pool cố định là 400 ảnh train. Self và pixel trùng bị loại khỏi pairing. Nếu có patient/lesion group metadata, cung cấp
CSV `image_id,group_id` để loại cả cùng nhóm. Không có metadata thì protocol ghi rõ giới hạn này; không suy diễn patient từ filename.

## Output cần tải

Chạy đến cell export cuối. Chỉ bundle có `manifest.json: status=ready` sau cold reload mới được coi là kiểm tra hoàn tất.
Tải nguyên ZIP `OUTPUT_DIR/exports/..._web_bundle.zip`.

| Thành phần | Công dụng |
| --- | --- |
| `models/<official DINO checkpoint>.pth` | Backbone frozen đầy đủ |
| `models/selector.pt` | Utility projections; thêm semantic mix với M-TACS |
| `models/segmentor.pt` | Toàn bộ gated cross-attention, DPT; thêm query/candidate text projections và gates với M-TACS |
| `candidate_bank.pt`, `candidate_pool.json`, `candidate_images/` | Bank đúng thứ tự, ảnh context thật, metadata pool train |
| `ground_truth_masks/`, `ground_truth_index.json` | Hiển thị/đối chiếu GT sau suy luận, không làm đầu vào mô hình |
| `runtime/`, `vendor/dinov3/` | Kiến trúc, preprocessing, inference Python/CLI và source backbone |
| `config.json`, `manifest.json`, `training_identity.json` | Cấu hình, binding checkpoint/data/source và SHA256 |
| `metrics.json`, `history.csv`, `example/` | Metrics thực sau train; ví dụ mask, probability, overlay, attention grids |
| `caption_models/`, `captions.csv`, `text_embeddings.pt`, caption metadata | M-TACS: đầy đủ Florence, CLIP, caption/cache/policy cho ảnh mới |

`best.pt`/`last.pt` của training chỉ lưu phần học được vì backbone được freeze và khóa hash. Để resume, giữ **toàn bộ OUTPUT_DIR**;
deployment ZIP không chứa optimizer. Runtime dựng lại predictor từ ba thành phần model, không cần train attention riêng.
ZIP chứa model/source/data, không chứa Python interpreter; máy chạy web vẫn cần dependency đúng hệ điều hành/Python.

Metric chính: mean per-image binary Dice/IoU tại 448×448, threshold cố định 0.5, best chọn bằng val Dice.
Metrics đối chiếu trong demo tính ở kích thước ảnh gốc và được ghi riêng, không thay bảng benchmark.
Attention là CLS→patch của block cuối, không phải Grad-CAM, giải thích nhân quả, hay mức ác tính.

Đây là bước chuẩn bị notebook/artifact. Route/giao diện segmentation của web sẽ nối adapter này ở bước tiếp theo;
không thay đổi hai trang SUN397 hiện tại.

Nguồn kiến trúc: [TACS §3–4.2](https://arxiv.org/html/2512.00489v1), [DINOv3](https://github.com/facebookresearch/dinov3).
