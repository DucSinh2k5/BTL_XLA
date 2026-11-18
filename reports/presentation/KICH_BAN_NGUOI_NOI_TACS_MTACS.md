# Kịch bản người nói: TACS → M-TACS (15 phút)

> Đi kèm [SLIDE_TIEU_CHUAN_TACS_MTACS.md](SLIDE_TIEU_CHUAN_TACS_MTACS.md), đánh số slide trùng nhau.
> Mỗi slide gồm: **thời gian**, **lời nói** (đọc tự nhiên, không cần thuộc từng chữ) và **chuyển ý** sang slide sau.
> Chữ **in đậm** là ý bắt buộc phải nói. Phần *(nếu dư giờ)* có thể bỏ.

---

## Số liệu khối lượng công việc (thuộc lòng để nói ở slide 3, 7 và khi được hỏi)

| Hạng mục | Số lượng |
|---|---|
| Tổng số lượt train có lưu kết quả | **khoảng 45** |
| Tái lập baseline đúng cấu hình bài báo | **12 notebook** (CUB 3 seed + 1 lượt chạy thêm, Stanford Cars 5 phương pháp, SUN397 2, ISIC 1), cộng 3 phiên bản sửa lỗi |
| Cải tiến 1 (backbone lớn + ảnh lớn) | **5 lượt** (Cars ViT-B, SUN397 ViT-B và ViT-L, DTD ViT-L, ISIC ViT-L) |
| Cải tiến 2 (M-TACS), ablation trên DTD | **20 cấu hình**: 8 vị trí văn bản + 7 nguồn caption + 5 kiến trúc trộn |
| Cải tiến 2 trên SUN397 và ISIC | 3 lượt SUN397 + 2 lượt ISIC |
| Sinh caption | Florence-2 (DTD, SUN397, ISIC), Qwen3-VL 8B và 32B, Florence fine-tune LoRA |
| Kiểm soát chất lượng | Notebook sinh tự động bằng builder, caption khóa bằng SHA-256, **56 file test tự động** |
| Web demo | 4 trang (SUN397 TACS / M-TACS, ISIC TACS / M-TACS), sinh caption trực tiếp cho ảnh mới |

**Khó khăn về tài nguyên (nói ngắn, không than thở):**
- Chủ yếu dùng GPU Kaggle: mỗi phiên tối đa 12 giờ, có giới hạn quota theo tuần.
- Vì vậy **mỗi thí nghiệm là một notebook độc lập**, chia thành các giai đoạn: sinh caption → mã hóa CLIP → train → đánh giá. Giai đoạn nào xong thì lưu lại, phiên sau chạy tiếp từ checkpoint.
- Có những lượt bị cắt giữa chừng do hết giờ (bản web ISIC dừng ở epoch 93–94/100, bản web SUN397 ở epoch 42–43/100). Em dùng checkpoint tốt nhất theo tập val tại thời điểm dừng.

---

## Slide 1: Tiêu đề (0:15)

**Lời nói**
"Em chào thầy/cô. Em xin trình bày đề tài **chọn ngữ cảnh theo nhiệm vụ, TACS**, một phương pháp của CVPR 2026, cùng hai hướng cải tiến của em: **mở rộng quy mô mô hình** và **bổ sung ngữ cảnh văn bản**, gọi là M-TACS."

**Chuyển ý:** "Trước hết, em xin nói về bài toán."

---

## Slide 2: Đặt vấn đề và khó khăn (0:55)

**Lời nói**
"Khi gặp một ảnh khó, con người thường **so sánh với một ví dụ tham khảo**: bác sĩ xem lại ca bệnh cũ, người chọn vải đặt hai mẫu cạnh nhau. Mô hình thị giác cũng có thể làm vậy: nhìn ảnh cần đoán cùng một ảnh ngữ cảnh.

Cách phổ biến là lấy **ảnh giống nhất**. Nhưng **ảnh giống nhất chưa chắc là ảnh giúp đoán đúng nhất**. Ngay trong bài báo, lấy ảnh giống nhất bằng DINO còn thấp hơn không dùng ngữ cảnh trên CUB và DTD.

Khi làm đề tài, em gặp ba khó khăn chính: **phép chọn ảnh là rời rạc** nên không lấy đạo hàm trực tiếp được; **mô hình rất lớn so với GPU** em có; và **caption có thể lộ nhãn**, làm kết quả bị sai lệch."

**Chuyển ý:** "Từ đó, em đặt ra các mục tiêu sau."

---

## Slide 3: Mục tiêu và đóng góp (0:35)

**Lời nói**
"Em làm theo ba bước. Một, **tái lập TACS cho đúng** bài báo. Hai, **Cải tiến 1: mở rộng quy mô**, dùng backbone ViT-L và ảnh lớn hơn. Ba, **Cải tiến 2: M-TACS**, thêm ngữ cảnh văn bản. Cuối cùng là đóng gói thành web demo.

Tổng cộng em đã chạy **khoảng 45 lượt train có lưu kết quả**, trong đó có **20 cấu hình ablation** cho M-TACS."

**Chuyển ý:** "Em xin giới thiệu nhanh TACS hoạt động thế nào."

---

## Slide 4: TACS, kiến trúc (0:55)

**Lời nói**
"TACS có ba khối. **Kho ứng viên** là 20% tập train, cố định. **Bộ chọn** tính điểm giữa ảnh truy vấn và từng ứng viên, rồi chọn **đúng một ảnh**. **Mạng tác vụ** nhận cặp ảnh và đưa ra dự đoán.

Với phân loại, patch của hai ảnh được ghép thành một chuỗi và đi chung qua ViT. Với phân đoạn, backbone được giữ cố định, thông tin từ ảnh ngữ cảnh đi vào qua các lớp cross-attention có cổng.

Một chi tiết quan trọng: nếu ảnh truy vấn cũng nằm trong kho, nó **không được tự chọn chính nó**, để tránh chép đáp án."

**Chuyển ý:** "Câu hỏi là: làm sao dạy bộ chọn khi phép chọn không lấy đạo hàm được?"

---

## Slide 5: TACS, thuật toán (1:05)

**Lời nói**
"TACS dùng **tối ưu lai** gồm hai nhánh.

Nhánh thứ nhất là **Gumbel-Softmax straight-through**: lượt xuôi vẫn chọn cứng một ảnh, nhưng lượt ngược cho đạo hàm đi qua một phân phối mềm.

Nhánh thứ hai là **REINFORCE**. Phần thưởng bằng loss khi không có ngữ cảnh trừ loss khi có ảnh ngữ cảnh. **Nếu ảnh ngữ cảnh giúp giảm loss, phần thưởng dương, và bộ chọn được khuyến khích chọn ảnh đó.** Nghĩa là mô hình học chọn ảnh *có ích*, không phải ảnh *giống*.

Bài báo cũng cho thấy TACS chọn ảnh **khác lớp** nhiều hơn 40–70%, tức là học cách chọn ví dụ tương phản."

**Chuyển ý:** "Trước khi cải tiến, em phải chắc chắn mình tái lập đúng."

---

## Slide 6: Tái lập baseline (0:50)

**Lời nói**
"Đây là bảng tái lập trên **CUB-200**, đúng cấu hình bài báo: ViT-S/16, ảnh 256. Em chạy **3 seed** giống bài báo. Trung bình đạt **82.42%**, so với **82.1%** của bài báo. Chênh lệch chỉ 0.3 điểm, các seed lệch nhau khoảng 0.1 điểm.

Em cũng tái lập trên Stanford Cars và SUN397. Riêng SUN397 với TACS em đạt **74.01**, so với 71.8 của bài báo.

Tổng cộng phần tái lập gồm **12 notebook** và 3 phiên bản sửa lỗi, ví dụ sửa cách tách tập val và sửa bộ chọn cho đúng bài báo."

**Chuyển ý:** "Baseline đã đúng. Tiếp theo là dữ liệu dùng cho các cải tiến."

---

## Slide 7: Dữ liệu và thiết lập (0:35)

**Lời nói**
"Em thử nghiệm trên **cả DTD và SUN397**. **DTD ít dữ liệu và nhẹ hơn**, chỉ 5,640 ảnh, nên mỗi lần train nhanh. Để **tiết kiệm GPU và thời gian**, em dùng DTD cho phần lớn thử nghiệm và ablation. **Demo thì dùng SUN397**, vì ảnh lớn, rõ nét, nhãn khung cảnh dễ phân biệt bằng mắt.

Về tài nguyên: em chủ yếu dùng **GPU Kaggle, mỗi phiên tối đa 12 giờ**. Vì vậy mỗi thí nghiệm là một notebook riêng, **chia thành nhiều giai đoạn** (sinh caption, mã hóa, train, đánh giá), có checkpoint để phiên sau chạy tiếp.

Mọi so sánh đều giữ cùng giao thức: cùng split, cùng kho ứng viên, 50 epoch, seed 42, chọn checkpoint theo tập val."

**Chuyển ý:** "Sau đây là Cải tiến 1."

---

## Slide 8: Cải tiến 1, ViT-L/16 + tăng kích thước ảnh (0:50)

**Lời nói**
"Bài báo dùng ViT-S với ảnh 256. Em nâng lên **ViT-B rồi ViT-L**, và tăng ảnh lên **384** cho phân loại, **448** cho phân đoạn.

Kết quả tăng rõ: **DTD từ 77.7 lên 84.95**, tăng 7.3 điểm. **SUN397 tăng dần qua từng bước: 74.01, 79.42, rồi 81.11.** ISIC tăng lên 88.62 Dice.

Cái khó là **ViT-L với ảnh 384 tràn cả GPU 95 GB** khi train SUN397. Em phải tách batch logic 64 thành các phần nhỏ 8 ảnh, chạy lại đúng trạng thái ngẫu nhiên và dùng gradient checkpointing. Như vậy vẫn giữ đúng thuật toán TACS mà vừa bộ nhớ.

Bản TACS ViT-L này là **baseline mạnh** để M-TACS so sánh."

**Chuyển ý:** "Có một backbone mạnh rồi, câu hỏi tiếp theo là: thêm văn bản có giúp được không?"

---

## Slide 9: Cải tiến 2, M-TACS (0:55)

**Lời nói**
"Ý tưởng của M-TACS: ngoài ảnh ngữ cảnh, đưa thêm **mô tả văn bản** của ảnh đó, ví dụ 'sọc dọc dày đặc', 'chấm tròn nhỏ'.

Caption được sinh bởi mô hình thị giác–ngôn ngữ **Florence-2 hoặc Qwen3-VL**, qua bộ lọc chống lộ nhãn, rồi mã hóa bằng CLIP. Vector văn bản được cộng vào đặc trưng ảnh qua một **cổng học được**. Nếu caption không hợp lệ, cổng bằng 0, nên **không phá đặc trưng ảnh gốc**.

Em còn thử hai biến thể nâng cao: đưa văn bản vào **bộ chọn**, và đưa caption thành **một token bên trong ViT**."

**Chuyển ý:** "Một vấn đề quan trọng khi dùng caption là lộ nhãn."

---

## Slide 10: Pipeline caption và chống lộ nhãn (0:35)

**Lời nói**
"Bộ lọc ban đầu so caption với tên lớp bằng CLIP, nhưng trên DTD nó **loại nhầm gần như toàn bộ caption**. Em thiết kế bộ lọc mới: cắt câu mở đầu chung chung, **chặn trực tiếp tên lớp và từ đồng nghĩa**, và **không xóa ảnh** khi caption lỗi mà chỉ che đi.

Bộ lọc có hiệu quả rõ: caption đã lọc cao hơn chưa lọc **1.93 điểm F1**.

Với ảnh da liễu ISIC, em dùng chính sách riêng: **chỉ mô tả thứ nhìn thấy được** như màu sắc, đường viền, **cấm hoàn toàn thuật ngữ chẩn đoán** như melanoma; nhờ vậy 100% ảnh có caption hợp lệ.

Về kỹ thuật: Florence-2 cần thư viện bản cũ, nên em chạy nó trong **tiến trình con riêng**. Việc sinh caption tự giảm batch khi tràn bộ nhớ và lưu dần, để khi Kaggle hết giờ thì phiên sau chạy tiếp."

**Chuyển ý:** "Tiếp theo là ablation: đặt văn bản ở đâu thì tốt nhất?"

---

## Slide 11: Ablation 1, vị trí văn bản (0:45)

**Lời nói**
"Em chạy **8 cấu hình** trên DTD với cùng giao thức, chỉ đổi vị trí đặt văn bản.

Kết quả rất rõ: **chỉ đặt caption ở ảnh ứng viên** cho **85.03**, vượt TACS chỉ dùng ảnh (84.82). Ngược lại, **đặt caption ở ảnh truy vấn làm giảm 2.5 điểm**. Em cũng có đối chứng xáo trộn caption để kiểm tra.

Kết luận: **vị trí đặt văn bản quan trọng hơn lượng văn bản**. Đây là cơ sở cho thiết kế cuối cùng."

**Chuyển ý:** "Sau khi chốt vị trí, em tối ưu nguồn caption và cách trộn."

---

## Slide 12: Ablation 2, nguồn caption và kiến trúc trộn (0:45)

**Lời nói**
"Về nguồn caption, em thử **7 phương án**. **Qwen3-VL-8B tốt hơn Florence khoảng 1 điểm**, còn bản 32B không hơn bản 8B, nên 8B là lựa chọn hiệu quả.

Về kiến trúc trộn, em thử **5 phiên bản**: chuẩn hóa L2, căn chỉnh bằng Optimal Transport, đưa văn bản vào bộ chọn, và token văn bản trong ViT. **Phiên bản S01, đơn giản nhất, cho điểm cao nhất: 85.30.**

Ở phiên bản đưa văn bản vào bộ chọn, khi tắt caption thì **5–11% ảnh ngữ cảnh được chọn thay đổi**. Như vậy bộ chọn thật sự có dùng thông tin văn bản."

**Chuyển ý:** "Tổng hợp lại, đây là kết quả chính."

---

## Slide 13: Kết quả chính (0:45)

**Lời nói**
"Trên DTD: bài báo đạt 77.7. **Cải tiến 1** đưa lên **84.82** F1. **Cải tiến 2, M-TACS**, tăng tiếp lên **85.03** rồi **85.30**, tức **+0.48** so với TACS ViT-L.

Trên phân đoạn ISIC, M-TACS cũng **vượt TACS**: **88.83 so với 88.62 Dice** trên tập val.

Đây là kết quả của 1 seed; em đã lên kế hoạch chạy thêm seed để khẳng định."

**Chuyển ý:** "Toàn bộ hệ thống đã được triển khai thành web."

---

## Slide 14: Triển khai hệ thống (0:40)

**Lời nói**
"Mỗi mô hình được **đóng gói thành một bundle tự chứa**: trọng số, mã nguồn backbone, kho ứng viên, cấu hình, và với M-TACS là cả Florence-2 và CLIP. Bundle được kiểm tra nạp lại và mã băm trước khi xuất.

Web dùng **FastAPI**, gồm **4 trang** trong một giao diện: phân loại SUN397 và phân đoạn ISIC, mỗi bài có TACS và M-TACS. Với M-TACS, **ảnh mới tải lên được sinh caption trực tiếp** trên server. Mô hình được nạp sẵn khi khởi động, và ảnh tải lên bị xóa sau khi xử lý.

Để kết quả tái lập được, notebook được sinh tự động từ mã nguồn, caption khóa bằng mã băm, và dự án có **56 file test tự động**."

**Chuyển ý:** "Em xin demo trực tiếp."

---

## Slide 15: Demo web (3:00)

**Lời nói theo từng bước**
1. *(Mở studio)* "Đây là giao diện demo với 4 trang: phân loại SUN397 và phân đoạn ISIC, mỗi bài có TACS và M-TACS. Mô hình đã được nạp sẵn, badge báo **Sẵn sàng**."
2. *(Trang SUN397 TACS, tải ảnh bãi biển)* "Em tải ảnh một bãi biển. Mô hình dự đoán đúng với độ tin cậy cao; vì ảnh thuộc bộ dữ liệu nên **nhãn đúng hiện ngay bên cạnh** để đối chiếu. **Đây là ảnh ngữ cảnh mà bộ chọn chọn từ kho 3,573 ảnh**, và bản đồ chú ý cho thấy vùng mô hình tập trung."
3. *(Chuyển sang trang M-TACS, cùng ảnh)* "Ở M-TACS, hệ thống **sinh caption trực tiếp** cho ảnh vừa tải lên bằng Florence-2, đồng thời hiện caption của ảnh ngữ cảnh. Toàn bộ pipeline ảnh và văn bản chạy ngay trên ảnh mới."
4. *(Mở trang ISIC)* "Với phân đoạn tổn thương da, đây là mặt nạ dự đoán, ảnh hỗ trợ được chọn và vùng mô hình chú ý."
5. "Em xin quay lại slide."

**Nếu thiếu ảnh:** bấm nút **ảnh mẫu** có sẵn trên trang.
**Nếu bị hỏi về cấu hình M-TACS trong demo:** "Bản web là full_both, được train trước khi ablation trên DTD chốt thiết kế caption chỉ ở ảnh ứng viên. Đóng gói lại theo thiết kế đó là bước tiếp theo của em."

**Nếu có sự cố:** chuyển ngay sang video demo đã quay sẵn, nói: "Em xin phép chiếu bản ghi để tiết kiệm thời gian."
**Nếu hai mô hình đoán khác nhau:** "Đây là một ảnh khó; hai mô hình chọn ảnh ngữ cảnh khác nhau nên kết quả khác nhau." Rồi chuyển ảnh tiếp theo.

---

## Slide 16: Hạn chế, hướng phát triển, kết luận (1:10)

**Lời nói**
"Về hạn chế: **mức tăng của M-TACS còn nhỏ** và mới có 1 seed. Quan trọng hơn, **kênh văn bản chưa được tận dụng tối đa**: caption ngắn mô tả toàn ảnh trùng nhiều thông tin với những gì ViT-L đã thấy, nên cổng văn bản học được còn nhỏ.

Hướng phát triển: **đóng gói lại bản web theo thiết kế M-TACS tốt nhất**, **trộn ảnh và văn bản ở mức token**, dùng **văn bản mang thông tin mới** như mô tả lớp của ảnh ngữ cảnh, dùng backbone có sẵn không gian chung ảnh–văn bản, và chạy thêm seed.

Tóm lại: em đã **tái lập TACS sát bài báo**; **Cải tiến 1** tăng mạnh hiệu năng nhờ mở rộng quy mô; **Cải tiến 2, M-TACS**, vượt baseline mạnh đó và qua ablation chỉ ra yếu tố quyết định là vị trí và chất lượng văn bản. Toàn bộ đã chạy thành web demo.

**Em xin cảm ơn thầy/cô đã lắng nghe.**"

---

## Hỏi đáp nhanh (câu trả lời 1–2 câu)

| Câu hỏi | Trả lời |
|---|---|
| Train mất bao lâu, dùng GPU gì? | "Chủ yếu GPU Kaggle, mỗi phiên tối đa 12 giờ, nên em chia mỗi thí nghiệm thành nhiều giai đoạn có checkpoint. Riêng bản ViT-L SUN397 cần GPU 95 GB và em phải tối ưu bộ nhớ." |
| Đã chạy bao nhiêu thí nghiệm? | "Khoảng 45 lượt có lưu kết quả: 12 notebook tái lập, 5 lượt mở rộng quy mô, 20 cấu hình ablation M-TACS trên DTD, và các lượt trên SUN397, ISIC." |
| Sao một số lượt không đủ 100 epoch? | "Do giới hạn thời gian phiên GPU; em dùng checkpoint tốt nhất theo tập val tại thời điểm dừng, và ghi rõ đó là điểm val." |
| Mức tăng 0.48 có ý nghĩa không? | "Còn nhỏ và mới 1 seed; em sẽ chạy thêm seed. Điều chắc chắn là ablation chỉ ra thiết kế đúng: đặt caption ở ảnh truy vấn giảm 2.5 điểm." |
| Caption thật sự giúp không? | "Ở bộ chọn, tắt caption làm đổi 5–11% ảnh ngữ cảnh, nên mô hình có dùng văn bản. Ở khâu trộn, cổng văn bản còn nhỏ; đây là hạn chế em đã nêu." |
| Có lộ nhãn qua caption không? | "Bộ lọc chặn tên lớp và từ đồng nghĩa; caption chưa lọc cho điểm thấp hơn 1.93, chứng tỏ bộ lọc có tác dụng." |
| Sao demo không dùng bản M-TACS tốt nhất? | "Bundle web là full_both, train trước khi ablation chốt thiết kế. Demo cho thấy pipeline đa phương thức chạy trực tiếp; đóng gói lại theo thiết kế tốt nhất là bước tiếp theo." |
| Ảnh tải lên có bị lưu không? | "Không; ảnh chỉ ghi tạm trong lúc suy luận rồi bị xóa." |
| Làm sao đảm bảo kết quả tái lập được? | "Notebook sinh tự động từ mã nguồn, caption và kết quả khóa bằng SHA-256, có 56 file test tự động kiểm tra giao thức." |
| Bảng CUB chỉ có No-Context? | "CUB dùng để kiểm chứng pipeline với 3 seed như bài báo; TACS có ngữ cảnh em kiểm chứng trên SUN397 (74.01 so với 71.8) và Stanford Cars." |
