# Kịch bản người nói: chỉ các phần vừa sửa

> Đối chiếu với [KICH_BAN_NGUOI_NOI_TACS_MTACS.md](KICH_BAN_NGUOI_NOI_TACS_MTACS.md). Các slide không nêu ở đây giữ nguyên lời nói cũ.

## Tóm tắt thay đổi

| Vị trí | Loại thay đổi |
|---|---|
| Bảng khối lượng công việc | Thêm 2 dòng: kiểm soát chất lượng, web demo |
| Slide 10 | Thêm 1 đoạn về caption ISIC |
| Slide 14 | Viết lại lời nói, thời lượng 0:30 → 0:40 |
| Slide 15 | Viết lại lời nói 5 bước demo, thêm 2 tình huống |
| Slide 16 | Thêm 1 hướng phát triển vào câu nói |
| Hỏi đáp nhanh | Thêm 3 câu |

---

## Bảng khối lượng công việc: 2 dòng mới

| Hạng mục | Số lượng |
|---|---|
| Kiểm soát chất lượng | Notebook sinh tự động bằng builder, caption khóa bằng SHA-256, **56 file test tự động** |
| Web demo | 4 trang (SUN397 TACS / M-TACS, ISIC TACS / M-TACS), sinh caption trực tiếp cho ảnh mới |

---

## Slide 10: đoạn chèn thêm (nói sau câu "cao hơn chưa lọc 1.93 điểm F1")

"Với ảnh da liễu ISIC, em dùng chính sách riêng: **chỉ mô tả thứ nhìn thấy được** như màu sắc, đường viền, **cấm hoàn toàn thuật ngữ chẩn đoán** như melanoma; nhờ vậy 100% ảnh có caption hợp lệ."

---

## Slide 14: Triển khai hệ thống (0:40), lời nói mới

"Mỗi mô hình được **đóng gói thành một bundle tự chứa**: trọng số, mã nguồn backbone, kho ứng viên, cấu hình, và với M-TACS là cả Florence-2 và CLIP. Bundle được kiểm tra nạp lại và mã băm trước khi xuất.

Web dùng **FastAPI**, gồm **4 trang** trong một giao diện: phân loại SUN397 và phân đoạn ISIC, mỗi bài có TACS và M-TACS. Với M-TACS, **ảnh mới tải lên được sinh caption trực tiếp** trên server. Mô hình được nạp sẵn khi khởi động, và ảnh tải lên bị xóa sau khi xử lý.

Để kết quả tái lập được, notebook được sinh tự động từ mã nguồn, caption khóa bằng mã băm, và dự án có **56 file test tự động**."

**Chuyển ý:** "Em xin demo trực tiếp."

---

## Slide 15: Demo web (3:00), lời nói mới

1. *(Mở studio)* "Đây là giao diện demo với 4 trang: phân loại SUN397 và phân đoạn ISIC, mỗi bài có TACS và M-TACS. Mô hình đã được nạp sẵn, badge báo **Sẵn sàng**."
2. *(Trang SUN397 TACS, tải ảnh bãi biển)* "Em tải ảnh một bãi biển. Mô hình dự đoán đúng với độ tin cậy cao; vì ảnh thuộc bộ dữ liệu nên **nhãn đúng hiện ngay bên cạnh** để đối chiếu. **Đây là ảnh ngữ cảnh mà bộ chọn chọn từ kho 3,573 ảnh**, và bản đồ chú ý cho thấy vùng mô hình tập trung."
3. *(Chuyển sang trang M-TACS, cùng ảnh)* "Ở M-TACS, hệ thống **sinh caption trực tiếp** cho ảnh vừa tải lên bằng Florence-2, đồng thời hiện caption của ảnh ngữ cảnh. Toàn bộ pipeline ảnh và văn bản chạy ngay trên ảnh mới."
4. *(Mở trang ISIC)* "Với phân đoạn tổn thương da, đây là mặt nạ dự đoán, ảnh hỗ trợ được chọn và vùng mô hình chú ý."
5. "Em xin quay lại slide."

**Tình huống mới**
- **Nếu thiếu ảnh:** bấm nút **ảnh mẫu** có sẵn trên trang.
- **Nếu bị hỏi về cấu hình M-TACS trong demo:** "Bản web là full_both, được train trước khi ablation trên DTD chốt thiết kế caption chỉ ở ảnh ứng viên. Đóng gói lại theo thiết kế đó là bước tiếp theo của em."

---

## Slide 16: câu nói về hướng phát triển (đã sửa)

"Hướng phát triển: **đóng gói lại bản web theo thiết kế M-TACS tốt nhất**, **trộn ảnh và văn bản ở mức token**, dùng **văn bản mang thông tin mới** như mô tả lớp của ảnh ngữ cảnh, dùng backbone có sẵn không gian chung ảnh–văn bản, và chạy thêm seed."

---

## Hỏi đáp nhanh: 3 câu mới

| Câu hỏi | Trả lời |
|---|---|
| Sao demo không dùng bản M-TACS tốt nhất? | "Bundle web là full_both, train trước khi ablation chốt thiết kế. Demo cho thấy pipeline đa phương thức chạy trực tiếp; đóng gói lại theo thiết kế tốt nhất là bước tiếp theo." |
| Ảnh tải lên có bị lưu không? | "Không; ảnh chỉ ghi tạm trong lúc suy luận rồi bị xóa." |
| Làm sao đảm bảo kết quả tái lập được? | "Notebook sinh tự động từ mã nguồn, caption và kết quả khóa bằng SHA-256, có 56 file test tự động kiểm tra giao thức." |
