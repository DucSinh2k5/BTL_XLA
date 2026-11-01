# Fine-Tune Caption DTD: Kiểm Duyệt Trước, Training Sau

Notebook: `florence2-dtd-caption-finetune.ipynb`.

Đây là thí nghiệm mới, tách riêng tám bản ablation M-TACS. Hiện chưa có caption mục tiêu được duyệt, nên notebook **không tự train trên caption Florence cũ** và không hứa trước điểm F1 sẽ tăng.

## Bước 1: Tạo Mẫu Kiểm Duyệt

1. Đưa notebook lên Kaggle và gắn dataset DTD như trước. Chưa cần GPU hoặc PEFT cho bước kiểm duyệt.
2. Chạy từ đầu đến mục **Create the Review Template**. Mặc định lấy 4 ảnh mỗi lớp, tổng cộng 188 ảnh, chỉ từ official train split 1.
3. Mẫu được lưu tại:

```text
/kaggle/working/outputs/florence2_dtd_split1_caption_lora_v1/review/caption_references.csv
```

`target_caption` để trống; `review_status=pending`. Nếu muốn xem caption cũ như gợi ý, đặt `DRAFT_CAPTIONS_CSV` trong cell cấu hình thành đường dẫn CSV cũ trước khi tạo mẫu. Gợi ý luôn mang tên `draft_caption_unverified`, không phải ground truth.

## Bước 2: Xem Ảnh Và Ghi Target

Dùng cell xem ảnh:

```python
current_review_id = show_caption_review(0)
```

Đổi chỉ số để xem ảnh tiếp theo. Viewer chỉ hiện ảnh và mã kiểm duyệt, không hiện nhãn hay tên file. Thứ tự kiểm duyệt được xáo trộn cố định. CSV vẫn chứa `image_id` để nối dữ liệu, nên tránh đọc tên thư mục lớp khi viết mô tả.

Trong cell ngay dưới, điền `REVIEWED_TARGET` và `REVIEWER_ID`, rồi chạy để lưu. Mỗi target gồm **5–8 từ tiếng Anh**, tả điều nhìn thấy: bề mặt, vật liệu, màu, hình dạng nhỏ, cách sắp xếp, khoảng cách hoặc sự lặp lại. Tránh câu mở đầu chung chung như “The image is” và chi tiết không thấy trong ảnh. Không nhìn nhãn rồi viết lại nhãn thành câu.

Nếu không chắc, để pending và ghi `notes`; không cố phê duyệt cho đủ số lượng. Nếu một VLM teacher đề xuất caption, người kiểm duyệt vẫn phải xem ảnh; ghi `caption_source=teacher` và `teacher_model`. Notebook không tự gọi teacher hay API nào.

Mốc tối thiểu là **50 caption đã duyệt**, chỉ là điều kiện chặn training quá sớm, không chứng minh đủ dữ liệu. Nên duyệt cả mẫu 188 ảnh để có độ phủ tốt hơn. Nếu chưa đủ, cell **Validate References Before Any Fine-Tuning** sẽ dừng có chủ đích. Các target đã lưu không mất khi chạy lại notebook.

## Bước 3: Fine-Tune Florence

Sau khi duyệt, tiếp tục từ cell validate references. Notebook chia lại ảnh train đã duyệt thành tập fit và tập caption holdout, không dùng official val/test để cập nhật captioner.

- Base: Microsoft Florence-2-base legacy tại `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`.
- Huấn luyện: LoRA ở attention decoder; vision encoder và trọng số base đóng băng.
- Mặc định 5 epoch cho captioner, khác với 50 epoch train M-TACS.
- Chọn adapter bằng token NLL trên caption holdout. Không dùng test macro-F1 để chọn hyperparameter captioner.
- Chạy Transformers 4.49.x và PEFT 0.14.0 trong subprocess offline, không thay Transformers của kernel.
- Nếu Input cũ chưa có PEFT, cần gắn thêm wheel `peft-0.14.0-py3-none-any.whl`. Các wheel phụ thuộc phải phù hợp Python/platform của Kaggle. Notebook không tải Internet; thiếu wheel sẽ báo rõ.

`last.pt` giữ trạng thái cuối epoch để resume. Epoch đang chạy dở được chạy lại từ đầu. `best.pt` và `best_adapter/` lưu adapter tốt nhất theo caption holdout. Đổi target, dữ liệu, code hoặc hyperparameter sau khi đã bắt đầu phải dùng output directory mới. LoRA adapter cần đi cùng base Florence, không phải model độc lập.

Xem `history.csv` và `holdout_before_after.csv`; đánh giá trực quan ảnh tương ứng trước khi kết luận caption tốt hơn. NLL giảm chưa chứng minh mô tả đúng hơn hoặc phân loại tốt hơn.

## Bước 4: Xuất CSV Mới Và So Sánh M-TACS

Mục cuối áp dụng adapter đã cố định lên 5.640 ảnh và xuất:

```text
/kaggle/working/outputs/florence2_dtd_split1_caption_lora_v1/exports/captions.csv
```

CSV được ghi/resume theo batch, dùng policy **dtd_lexical_v2**. `raw_caption` chứa caption mới của adapter, tránh việc pipeline M-TACS đọc lại raw caption cũ. Tên lớp chỉ được dùng sau sinh để audit; CLIP 0.30 chỉ là thông tin chẩn đoán, không tự loại caption. Bộ kiểm tra không chứng nhận hết mọi dạng semantic leakage.

Gắn CSV này vào **một lần chạy M-TACS mới**. Giữ cùng cấu hình, DTD split, seed, candidate pool và 50 epoch như đối chứng; đặt output/cache mới để không tái dùng checkpoint hoặc CLIP embeddings cũ. Tám kết quả ablation hiện có được giữ nguyên làm baseline. Muốn khẳng định cải thiện đáng tin cậy, cần kiểm tra nhiều seed và không lựa chọn biến thể theo test lặp đi lặp lại.

## Trạng Thái Kiểm Chứng

Notebook và code đã được kiểm thử bằng dữ liệu tổng hợp/mô hình nhỏ trên CPU. Chưa thực hiện fine-tune Florence thật trên GPU, chưa có target được người dùng kiểm duyệt, chưa có CSV cải tiến thật và chưa xác minh mức tăng F1.
