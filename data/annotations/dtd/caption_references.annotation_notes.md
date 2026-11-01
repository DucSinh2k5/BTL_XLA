# Nguồn Gán Nhãn Caption DTD

Đã xem 188 ảnh trong `archive (23)/dtd/images`, đối chiếu chính xác `image_id` và ghi caption riêng cho từng ảnh. Cả 188 ảnh đều đọc được và thuộc official DTD train split 1. Ảnh được xem trong bảng ảnh chỉ có số thứ tự; một số ảnh được mở lại ở độ phân giải gốc để kiểm tra chi tiết.

CSV giữ nguyên header, thứ tự dòng, `review_id`, `image_id` và `draft_caption_unverified`. Có 188 caption khác nhau, mỗi caption 7–8 từ tiếng Anh, không có tiền tố bị cấm và không khớp tên lớp hoặc các dạng gốc/biến thể trong bộ kiểm tra từ vựng. Không có dòng nào suy ra caption chỉ từ tên thư mục.

## Ý Nghĩa Metadata

- `caption_source=teacher`: caption do trợ lý AI đọc ảnh và viết, không phải nhãn do người gán trực tiếp.
- `review_status=approved`: hoàn tất gán nhãn bằng AI theo yêu cầu và đã qua kiểm tra cấu trúc/từ vựng. Không biểu thị đã có kiểm duyệt thủ công độc lập từng ảnh.
- `reviewer_id=ducsinh`: mã người yêu cầu và ủy quyền gán nhãn AI, giữ theo yêu cầu của người dùng. Không khẳng định ducsinh đã xem và duyệt từng ảnh.
- `teacher_model=openai-codex:model-id-unavailable`: nguồn là OpenAI Codex. Phiên làm việc không cung cấp mã model backend chính xác, nên không tự ghi một mã chưa được xác minh.
- `notes` để trống vì không có ảnh không đọc được; giới hạn nguồn gán nhãn được ghi chung tại đây và trong báo cáo JSON.

Đây là **teacher-generated reference captions**, chưa phải bộ gold captions được người kiểm duyệt độc lập. Kiểm tra từ vựng không chứng minh hết semantic leakage hay độ đúng tuyệt đối của mô tả. Khi báo cáo thí nghiệm, cần giữ đúng nguồn AI này; không mô tả dữ liệu là đã được người kiểm duyệt toàn bộ. Nếu code huấn luyện vẫn ghi cố định `reference_origin=human_reviewed_train_only`, cần đổi thông tin provenance đó cho phù hợp trước khi dùng các dòng AI-only này.

## Kiểm Tra Pipeline

Đã chạy bộ kiểm tra caption của pipeline hiện tại: 188 dòng vượt qua các điều kiện cấu trúc/provenance được kiểm tra bằng code. Chia reference train/holdout với seed 42 cho 141/47 ảnh, không giao nhau và đều lấy từ official train. Việc vượt guard không tự chứng nhận đã có người kiểm duyệt ảnh.

Báo cáo: `caption_references.annotation_report.json`. Bản CSV đầu vào nguyên trạng và ánh xạ ảnh/hash nằm trong `tmp/annotate_dtd_references/`. Chưa thực hiện fine-tune, sinh CSV toàn bộ DTD hay đánh giá F1 ở bước gán nhãn này.

## Dùng Trên Kaggle

Đưa CSV đã hoàn thành lên Kaggle và nhập nó làm reference CSV, không phải CSV caption cho toàn bộ 5.640 ảnh. Nếu notebook đã tạo `review/caption_references.csv` trống trước đó, chỉ đặt `REFERENCE_INPUT_CSV` sẽ không ghi đè mẫu đang có. Hãy sao lưu mẫu đang có rồi thay bằng CSV đã gán nhãn này, sau đó chạy lại cell validate references. Giữ thông tin nguồn AI khi tiếp tục thí nghiệm.
