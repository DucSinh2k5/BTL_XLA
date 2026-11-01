# Chạy lại notebook TACS web sau lỗi bộ nhớ

Notebook đã sửa: `TACS/tacs_web.ipynb`.

Bản trước khi sửa, bao gồm output và traceback, được giữ tại
`TACS/tmp/notebook_backups/tacs_web.before_memory_repair.ipynb`.

## Nguyên nhân xác nhận từ output

- GPU là NVIDIA RTX PRO 6000 Blackwell Server Edition, gần 95 GiB VRAM.
- Batch thực được đổi thành 64 và `gradient_checkpointing=False`.
- OOM xảy ra trong forward cặp query/context ở batch đầu của epoch 1, không phải lỗi đọc dataset.
- Pixel bank CPU trước đây phải được chuyển theo chunk ở mỗi backward. Đây là một chi phí triển khai thêm, không phải yêu cầu phải chạy chậm của TACS.

## Cách chạy

1. Giữ lại các output cũ. Restart kernel trên Kaggle để giải phóng model, graph và optimizer còn sống sau OOM.
2. Dùng notebook `tacs_web.ipynb` đã sửa và các Input SUN397, DINOv3 code, weights L/16 cũ. Không cần Internet hoặc cài thêm thư viện cho bản sửa này.
3. Kiểm tra đường dẫn trong section 2. Đường dẫn code DINOv3 của người dùng được giữ nguyên dưới owner `sinhnguync`.
4. Giữ `LOGICAL_BATCH_SIZE=64`, `EFFECTIVE_BATCH_SIZE=64`, `COMPUTE_BATCH_SIZE=8` để bắt đầu trên GPU nêu trên. Checkpointing luôn bật.
5. Run All. Kiểm tra log pixel bank dùng `cuda` hay fallback `cpu`, sau đó xem ba dòng `Profile batch` đầu.
6. Nếu vẫn OOM, restart kernel và giảm **COMPUTE_BATCH_SIZE**, không giảm logical batch. Có thể tăng compute batch sau khi đo VRAM, nhưng không có bảo đảm 16/32/64 sẽ vừa.

`COMPUTE_BATCH_SIZE` là kích thước chunk tính toán. Nó không cần chia hết 64. Logical batch mới xác định nhóm ảnh để chuẩn hóa reward; effective batch xác định tích lũy gradient.

## Tiếp tục checkpoint cũ

Output mặc định mới:

```text
/kaggle/working/outputs/sun397_tacs_visual_vitl16_web_v1_repaired_v2
```

`RESUME_FROM_DIR` mặc định trỏ về output cũ không có hậu tố. Nếu chuyển sang session Kaggle mới, attach Output đã lưu và đổi biến này sang đúng thư mục có `protocol.json`, `last.pt`, `best.pt`. Không trỏ vào file ZIP.

- Có checkpoint hợp lệ: kiểm tra cấu hình, thứ tự lớp/pool, nguồn code và hashes Input; sao chép checkpoint sang output mới và tiếp tục epoch sau. Trọng số, optimizer, scheduler, scaler, RNG, history và best checkpoint đều được giữ. File gốc không bị ghi đè.
- Chỉ có notebook hoặc log lỗi batch đầu: không có trọng số của một epoch hoàn tất để resume. Notebook thông báo bắt đầu epoch 1.
- Có `best.pt` nhưng thiếu `last.pt`: dừng và báo thiếu trạng thái resume, không giả vờ đã tiếp tục đủ optimizer state.
- Nếu run trước có logical batch khác 64, cấu hình đã đăng ký của run đó được giữ, không âm thầm đổi thống kê reward.
- Trong bản đã sửa, đổi compute batch không thay định danh thí nghiệm. Không đổi seed, logical batch, số epoch, learning rate hoặc pool trong cùng output.

Migration được ghi trong `migration.json` và `protocol.json`. Thứ tự lấy mẫu ngẫu nhiên mới không phải bản replay bitwise của vòng train cũ; báo cáo cần ghi rõ nếu kết quả gồm các epoch trước và sau sửa.

## Cơ chế được giữ và thay đổi

Giữ nguyên inference runtime và kiến trúc L/16, ảnh 384, K=1, shared backbone, ghép patch token, fixed train-only candidate pool và self masking. Giữ task CE qua straight-through Gumbel và policy REINFORCE, reward detached. Không thay bằng `argmax` không có task gradient.

Thay đổi vòng thực thi: prepass không giữ graph để lấy reward cho đủ logical batch; sau đó replay RNG của selector, tính log probability của chính policy đã lấy mẫu và backward từng chunk nhỏ. Vì vậy không cần giữ toàn bộ activation của 64 cặp ảnh đồng thời. Prepass có thêm một lượt encode selector; tốc độ cải thiện đến từ bỏ chuyển bank lặp lại và tận dụng batch tính toán hợp lý, không phải bỏ loss.

Khi đủ VRAM, bank training được chuyển một lần sang GPU FP32. Giá trị vẫn là pixel FP16 cũ được chuyển sang FP32; gradient straight-through không bị thay bằng index-only. Auto fallback CPU được thông báo rõ. Validation/test/export vẫn preprocess ảnh gốc FP32 để giữ tính nhất quán với adapter web.

File `history.csv` lưu throughput, compute batch, vị trí pixel bank, peak VRAM, thời gian refresh bank và validation. Các cột mới cũng được giữ khi nối tiếp history cũ.

## Kiểm chứng và giới hạn

Đã chạy bộ test CPU cho visual export, M-TACS export và bản sửa: **40 passed, 2 skipped**. Có kiểm tra gradient, replay RNG, normalize reward theo logical batch, tiếp tục checkpoint thực của DINOv3 nhỏ, export và nạp lại attention/prediction.

Hai test skip là đối chiếu sklearn khi thiếu thư viện và kiểm thử CUDA khi máy local không có CUDA. Chưa chạy full SUN397 trên RTX PRO 6000; không khẳng định số phút mỗi epoch hoặc mức batch tối đa từ kiểm thử CPU.

Bản sửa này không biến notebook thành tái hiện paper tuyệt đối: epoch-refreshed detached candidate keys, validation holdout, một seed và augmentation hiện có vẫn được giữ. ZIP deploy cuối và attention thật vẫn được xuất như trước.

Mã tạo lại notebook: `TACS/tools/notebook_builders/repair_sun397_tacs_web_notebook.py`.
Các hàm train sửa riêng: `TACS/src/training/sun397_tacs_web_training_patch.py`.
Không chỉnh các notebook ablation hay runtime M-TACS hiện có.
