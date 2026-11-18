# Bản slide: chỉ các phần vừa sửa

> Đối chiếu với [SLIDE_TIEU_CHUAN_TACS_MTACS.md](SLIDE_TIEU_CHUAN_TACS_MTACS.md). Các slide không nêu ở đây giữ nguyên.
> Mục đích của đợt sửa: bổ sung phần web demo và các phần dự án còn thiếu.

## Tóm tắt thay đổi

| Vị trí | Loại thay đổi |
|---|---|
| Đầu file | Tổng thời lượng 14:35; slide dự phòng tăng từ 4 lên 6 |
| Bảng thời gian | Slide 14: 0:30 → 0:40; lũy kế từ slide 14 dời thêm 10 giây |
| Slide 10 | Thêm 1 gạch đầu dòng về chính sách caption ISIC |
| Slide 14 | Viết lại toàn bộ nội dung |
| Slide 15 | Viết lại kịch bản demo, thêm lưu ý về bản M-TACS full_both |
| Slide 16 | Thêm hướng phát triển số 1 |
| Slide dự phòng | Thêm B5 và B6 |
| Hỏi đáp | Thêm 2 câu |

---

## Đầu file và bảng thời gian

- **Thời lượng:** 15 phút = khoảng 10.5 phút slide + 3 phút demo + 1 phút kết luận (tổng **14:35**).
- **Số slide:** 16 slide chính + **6** slide dự phòng.

| # | Slide | Thời gian | Lũy kế |
|---|---|---|---|
| 14 | Triển khai hệ thống và web | **0:40** | 10:25 |
| 15 | Demo web | 3:00 | 13:25 |
| 16 | Hạn chế, hướng phát triển, kết luận | 1:10 | 14:35 |

Dư 25 giây cho chuyển cảnh. Nếu bị dồn thời gian: rút slide 10 và 14 xuống 25 giây mỗi slide.

---

## Slide 10: thêm 1 dòng (đặt sau dòng "Hiệu quả: +1.93")

- **ISIC 2017** dùng chính sách riêng `isic_visible_attributes_v1`: chỉ mô tả thứ nhìn thấy được (màu sắc, đường viền, cấu trúc bề mặt), **cấm thuật ngữ chẩn đoán** (melanoma, nevus…); phủ 100% trên 2,750 ảnh với Qwen3-VL-8B.

---

## Slide 14: Triển khai hệ thống và web (0:40), nội dung mới

- **Train ViT-L không tràn bộ nhớ:** tách batch logic 64 / batch tính toán 8; lượt đầu tính phần thưởng không lấy đạo hàm, lượt sau chạy lại đúng trạng thái ngẫu nhiên (RNG replay) + gradient checkpointing.
- **Model bundle tự chứa (.zip):** checkpoint, mã DINOv3, kho ứng viên, cấu hình, **kèm Florence-2 và CLIP cho M-TACS**; kiểm tra nạp lại nguội và mã băm trước khi xuất.
- **Web FastAPI, 4 trang** trong một giao diện studio: SUN397 TACS / M-TACS, ISIC TACS / M-TACS.
  - Ảnh mới tải lên được **sinh caption trực tiếp** (Florence-2 + CLIP nạp sẵn trên server).
  - Nạp sẵn mô hình khi khởi động, hàng đợi một lượt suy luận mỗi lần, **ảnh tải lên bị xóa sau khi xử lý**.
- **Đảm bảo tái lập:** notebook sinh tự động bằng builder, caption và kết quả khóa bằng SHA-256, **56 file test tự động**.

**Gợi ý hình:** sơ đồ Trình duyệt → FastAPI → worker suy luận (DINOv3 + Florence-2 + CLIP) → bundle mô hình.

---

## Slide 15: Demo web (3:00), kịch bản mới

1. (0:20) Mở studio, chỉ 4 trang (SUN397 TACS / M-TACS, ISIC TACS / M-TACS) và badge "Sẵn sàng".
2. (0:50) Trang SUN397 TACS: tải một ảnh khung cảnh rõ ràng (ví dụ bãi biển). Chỉ ra: nhãn dự đoán, độ tin cậy, **nhãn đúng hiện ngay bên cạnh**, **ảnh ngữ cảnh được chọn**, bản đồ chú ý "Mô hình nhìn vào đâu?".
3. (0:50) Chuyển sang trang M-TACS với cùng ảnh: **caption được sinh trực tiếp** cho ảnh truy vấn và caption của ảnh ngữ cảnh; so sánh dự đoán với TACS.
4. (0:50) Trang ISIC: tải ảnh tổn thương da, hiển thị mặt nạ phân đoạn, ảnh hỗ trợ được chọn và bản đồ chú ý.
5. (0:10) Quay lại slide.

Nếu thiếu ảnh: dùng nút **ảnh mẫu** có sẵn trên trang. Phần "Bên trong TACS" cuối trang có thể cuộn qua nhanh nếu dư giờ.

**Lưu ý về bản M-TACS trong demo (mới)**
- Bundle web M-TACS (SUN397 và ISIC) là cấu hình **full_both**: caption của cả ảnh truy vấn và ảnh ứng viên, ở cả bộ chọn và khâu trộn. Bundle này được train trước khi ablation DTD chốt thiết kế "caption chỉ ở ảnh ứng viên".
- Điểm mạnh để nhấn khi demo: **toàn bộ pipeline đa phương thức chạy trực tiếp** trên ảnh mới. Đóng gói lại theo thiết kế tốt nhất là bước triển khai tiếp theo.

**Chuẩn bị bắt buộc (đã cập nhật)**
- Chọn sẵn 3–4 ảnh đã thử trước: hai mô hình dự đoán đúng, ảnh ngữ cảnh có ý nghĩa. Nhãn đúng chỉ hiện khi mã băm của ảnh nằm trong dữ liệu đã lưu của bundle, nên hãy thử trước và chọn ảnh mà web hiện được nhãn đúng.
- Kiểm tra đường dẫn bundle (biến môi trường hoặc `artifacts/deployment/`) và chạy `start_web.ps1` trước giờ trình bày; chỉ chạy một tiến trình server.
- Mở sẵn server và tab trình duyệt, đợi badge "Sẵn sàng"; quay sẵn video demo dự phòng nếu máy lỗi.

---

## Slide 16: danh sách hướng phát triển mới

1. **Đóng gói lại bản web M-TACS theo thiết kế tốt nhất (caption chỉ ở ảnh ứng viên).** *(mới)*
2. Trộn ảnh–văn bản ở mức token (cross-attention giữa patch và từ).
3. Dùng văn bản mang thông tin mới: nhãn hoặc mô tả lớp của ảnh ngữ cảnh.
4. Backbone có sẵn không gian chung ảnh–văn bản (SigLIP 2, DINOv3 dino.txt).
5. Chạy 3 seed cho các cấu hình chính; mở rộng K = 3 ảnh ngữ cảnh.

---

## Slide dự phòng mới

**B5. Hướng thử thêm: số ngữ cảnh thay đổi theo ảnh (SUN397, ViT-S/16, ảnh 384)**
- TACS adaptive chọn tối đa 3 ảnh ngữ cảnh, trung bình khoảng 2: test 75.88% Acc / 75.80% F1. Chưa có bản K = 1 cùng ViT-S, ảnh 384 để so trực tiếp.

**B6. Cấu hình của các bundle web**
- SUN397 và ISIC: TACS chỉ dùng ảnh và M-TACS **full_both** (caption ảnh truy vấn + ảnh ứng viên, ở bộ chọn và khâu trộn), DINOv3 ViT-L/16.
- ISIC M-TACS có tùy chọn thay bộ caption thử nghiệm (attributes v3) mà không train lại checkpoint; mặc định tắt.

---

## Hỏi đáp: 2 câu mới

| Câu hỏi | Câu trả lời gợi ý |
|---|---|
| "Sao demo không dùng bản M-TACS tốt nhất?" | "Bundle web được train trước khi ablation DTD chốt thiết kế, nên đang là full_both. Demo cho thấy toàn bộ pipeline đa phương thức chạy trực tiếp; đóng gói lại theo thiết kế tốt nhất là bước tiếp theo." |
| "Ảnh tải lên có được lưu lại không?" | "Không; ảnh chỉ ghi tạm trong lúc suy luận rồi bị xóa, không ghi vào bundle." |
