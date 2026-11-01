# Đánh giá 8 ablation DTD với captions hiện tại

Ngày đánh giá/cập nhật: 18/10/2026. Nguồn: toàn bộ source và output được lưu trong 8 notebook ở `output_ablation/`, đã bổ sung bản 08 `full_redacted`. Không chạy lại training, không sửa notebook gốc. Giữ tên file `DANH_GIA_ABLATION_DTD_01_07.md` để các liên kết cũ tiếp tục hoạt động; nội dung hiện bao gồm đủ 01-08.

## 1. Kết luận chính

**Với caption pipeline và cách tích hợp hiện tại, 8 run này chưa chứng minh text giúp M-TACS tốt hơn baseline chỉ dùng ảnh.** `candidate_only` nhỉnh hơn một chút trên test, nhưng thấp hơn trên validation và gate candidate rất nhỏ. Các cấu hình dùng caption query đều thấp hơn baseline trên cả validation và test.

**Bản 08 bổ sung một kết quả quan trọng:** che tên lớp/alias đạt 81.11% test macro-F1, cao hơn bản 07 unscreened 2.13 điểm nhưng vẫn thấp hơn baseline 3.71 điểm. Coverage 07/08 đều 100%, không caption nào bị che rỗng. Đây là bằng chứng phù hợp với việc loại các từ này làm cấu hình full tốt hơn trong run hiện tại, không phải bằng chứng text đã giúp vượt baseline hay đã chứng minh không có leakage.

Điều này là một kết quả ablation có ích, không phải thí nghiệm thất bại. Nó không đồng nghĩa mọi caption đều sai, CLIP không có ích, hay mọi cách dùng text đều không hiệu quả. Bằng chứng hiện có áp dụng cho DTD split 1, seed 42, cấu hình ViT-L/16 và ngân sách 50 epoch đã chạy.

**Không nên gọi `full_both` là bản tốt nhất của đợt DTD này.** Baseline visual có validation tốt nhất; `candidate_only` có test tốt nhất với biên rất nhỏ. Không nên chọn biến thể tiếp theo chỉ dựa vào thứ hạng test.

## 2. Bảng kết quả

Accuracy và macro-F1 tính theo phần trăm. Delta F1 là điểm phần trăm so với `visual_only`, không phải phần trăm cải thiện tương đối. Val dùng bảng report làm tròn 4 chữ số; test lấy bảng ở cell final-test với 6 chữ số để tính delta. Best epoch được chọn bằng **validation loss**, không phải macro-F1 lớn nhất.

| Bản | Cấu hình | Val Acc | Val macro-F1 | Test Acc | Test macro-F1 | Delta test F1 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | `visual_only` | **84.68** | **84.66** | 84.95 | 84.82 | 0.00 | 38 |
| 02 | `query_only` | 80.85 | 80.64 | 82.34 | 82.31 | -2.51 | 29 |
| 03 | `candidate_only` | 84.10 | 83.98 | **85.16** | **85.03** | +0.22 | 38 |
| 04 | `both_fusion` | 79.89 | 79.74 | 81.17 | 81.07 | -3.75 | 39 |
| 05 | `full_both` | 80.43 | 80.25 | 81.01 | 80.91 | -3.90 | 42 |
| 06 | `full_shuffle_both` | 82.87 | 82.75 | 83.40 | 83.34 | -1.48 | 21 |
| 07 | `full_unscreened` | 77.66 | 77.61 | 79.04 | 78.98 | -5.84 | 41 |
| 08 | `full_redacted` | 80.74 | 80.56 | 81.28 | 81.11 | -3.71 | 39 |

Giá trị test gốc trong bảng final-test:

| Cấu hình | Accuracy | Macro-F1 | Loss | Số ảnh đúng suy ra / 1,880 |
| --- | ---: | ---: | ---: | ---: |
| `visual_only` | 0.849468 | 0.848168 | 1.156826 | 1,597 |
| `query_only` | 0.823404 | 0.823067 | 1.277675 | 1,548 |
| `candidate_only` | 0.851596 | 0.850321 | 1.162605 | 1,601 |
| `both_fusion` | 0.811702 | 0.810661 | 1.318655 | 1,526 |
| `full_both` | 0.810106 | 0.809124 | 1.307802 | 1,523 |
| `full_shuffle_both` | 0.834043 | 0.833392 | 1.223320 | 1,568 |
| `full_unscreened` | 0.790426 | 0.789774 | 1.381790 | 1,486 |
| `full_redacted` | 0.812766 | 0.811105 | 1.313813 | 1,528 |

Số ảnh đúng được làm tròn từ accuracy hiển thị nhân 1,880, chưa phải kiểm đếm từ file prediction. `candidate_only` chỉ hơn baseline **4 ảnh đúng**, tăng 0.2128 điểm accuracy và 0.2153 điểm macro-F1. Chưa có đủ cơ sở gọi đây là cải thiện ổn định/có ý nghĩa thống kê; validation macro-F1 của nó còn thấp hơn baseline 0.68 điểm.

## 3. Đối chiếu từng câu hỏi ablation

### Caption query có giúp không? 02 so với 01

`query_only` thấp hơn baseline **2.51 điểm test macro-F1**, và thấp hơn khoảng 4.02 điểm validation macro-F1. Chưa có bằng chứng lợi ích cho cách đưa caption query vào gated fusion hiện tại.

Đây là tác động của cả quá trình huấn luyện với nhánh text, không chỉ tác động của caption ở inference. Không được kết luận rằng chỉ cần tắt gate của checkpoint 02 là sẽ lấy lại điểm của 01.

### Caption candidate có giúp không? 03 so với 01

Trên test, 03 đạt 85.03 F1 so với 84.82 của 01; trên validation, 03 đạt 83.98 so với 84.66. Kết luận hợp lý là **gần mức baseline, lợi ích chưa chắc chắn**, không phải candidate captions đã được chứng minh hiệu quả.

Cần ít nhất kiểm tra nhiều seed và prediction từng ảnh để xem những lỗi được sửa/phát sinh. Một seed không đo được độ biến thiên do training; điểm tổng hợp không đủ cho kiểm định paired bootstrap/McNemar.

### Dùng cả hai nhánh fusion có tốt hơn không? 04 so với 02 và 03

- Thêm candidate text vào query-text fusion: 04 thấp hơn 02 **1.24 điểm test F1**.
- Thêm query text vào candidate-text fusion: 04 thấp hơn 03 **3.97 điểm test F1**.

Trong run này, kết hợp hai nhánh không tạo lợi ích cộng gộp. Có thể là thông tin không phù hợp, dư thừa, cách fusion hoặc động lực tối ưu; chưa thể xác định một nguyên nhân duy nhất từ các bảng metrics.

### Text trong selector có giúp không? 05 so với 04

Hai bản giữ cả hai nhánh text-fusion, chỉ bật semantic selector ở 05. Test F1 giảm **0.15 điểm**, trong khi validation F1 tăng khoảng 0.51 điểm. Chênh lệch nhỏ và không đồng nhất giữa hai split: **chưa thấy lợi ích rõ ràng của semantic selection**.

Hệ số semantic ở checkpoint 05 là 0.497676, gần khởi tạo 0.5. Điều này không có nghĩa selector bỏ text: công thức vẫn cộng gần `0.5 * semantic_z`. Nó cũng không chứng minh gradient bị lỗi; cần gradient/logit statistics hoặc can thiệp tắt semantic để kết luận sâu hơn.

### Alignment ảnh-caption có đang mang lại lợi ích? 06 so với 05

`full_shuffle_both` đạt 83.34 F1, cao hơn 80.91 của `full_both` **2.43 điểm**, nhưng vẫn thấp hơn baseline 1.48 điểm.

Audit shuffle trong notebook cho thấy mask được giữ nguyên (rate 1.0), nguồn caption hợp lệ đã đổi 100%, và nội dung text đổi khoảng 99.60-100% tùy nhóm. Vì vậy phép shuffle thực sự có tác dụng, không phải chỉ đổi tên mode.

Kết quả này **không ủng hộ giả thuyết rằng alignment caption hiện tại đang tạo lợi ích trong cấu hình full**. Một khả năng là mô hình học giảm sử dụng text không phù hợp, nhưng chưa chứng minh được cơ chế đó chỉ bằng gate trung bình.

Đây là hai model được train lại; 06 đồng thời đảo caption ở hai vai trò, tác động cả fusion lẫn selector và có thể thay ảnh context được chọn. Nó không phải phép can thiệp giữ nguyên checkpoint/context, không tách được vai trò từng nhánh, và không kết luận được riêng 02 hay 03 có cần alignment không. Mask được giữ cũng không có nghĩa mọi tín hiệu thống kê từ mask đã bị loại bỏ.

### Bỏ lọc có làm tăng điểm do từ chỉ lớp không? 07 so với 05

`full_unscreened` đạt 78.98 F1, thấp hơn bản screened 05 **1.94 điểm**, thấp hơn baseline 5.84 điểm. Bỏ lọc và dùng thêm caption không làm điểm tăng trong run này.

Tuy nhiên 07 thay đổi đồng thời **nội dung được phép dùng và coverage**. Vì thế không được diễn giải chênh lệch là hiệu ứng thuần của từ chỉ lớp. Điểm thấp hơn cũng không chứng minh không có shortcut/leakage trong bất kỳ caption nào.

### Che tên lớp/alias có giúp không? 08 so với 07, 05 và 01

08 giữ cả hai nhánh fusion và semantic selector, không đảo caption. CLIP mã hóa text **đã che** cho cả hai vai trò; không dùng vector caption gốc để lách phép can thiệp. Source model/loss giống các bản khác, và audit ghi không còn từ nào khớp vocabulary tên lớp/alias đã định nghĩa.

| Đối chiếu | Delta val macro-F1 | Delta test macro-F1 | Delta test accuracy | Chênh ảnh đúng suy ra |
| --- | ---: | ---: | ---: | ---: |
| 08 trừ 07 unscreened | +2.95 | **+2.13** | +2.23 | +42 |
| 08 trừ 05 screened | +0.31 | +0.20 | +0.27 | +5 |
| 08 trừ 01 visual-only | -4.10 | **-3.71** | -3.67 | -69 |

Các delta tính theo điểm phần trăm; val lấy số đã làm tròn trong report. Test F1 08 là 0.811105 so với 0.789774 của 07 và 0.809124 của 05. Checkpoint 08 ở epoch 39 có validation loss 1.338365; test loss 1.313813.

Audit redaction:

| Vai trò/split | Caption trước che có sẵn | Caption sau che có sẵn | Caption thay đổi | Bị che rỗng | Còn từ khớp vocabulary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query train | 1,880 | 1,880 | 580 | 0 | 0 |
| Query val | 1,880 | 1,880 | 573 | 0 | 0 |
| Query test | 1,880 | 1,880 | 570 | 0 | 0 |
| Candidate pool train | 376 | 376 | 105 | 0 | 0 |

**Lần này không có nhiễu do khác coverage giữa 07 và 08:** tất cả query/pool captions của cả hai bản đều khả dụng. Vì thế đối chiếu này tách vấn đề coverage tốt hơn so với 07/05. Tuy nhiên CSV 08 được sinh mới, chưa có hash để xác minh toàn bộ caption nguồn giống 07; vẫn là hai lần train với một seed, không phải cùng checkpoint và context cố định. Việc xóa từ cũng thay đổi độ dài/biểu diễn text và có thể đổi ảnh context được selector chọn.

Kết luận phù hợp: redaction cải thiện hiệu năng quan sát được so với unscreened trên cả val/test, nhưng **chưa chứng minh tên lớp là nguyên nhân duy nhất**, chưa chứng minh hiệu ứng ổn định qua seed, và không đủ kết luận vi phạm train/test protocol. Mức hơn 05 chỉ 0.20 điểm F1/5 ảnh đúng chưa phải lợi ích chắc chắn, đặc biệt coverage 05 thấp hơn 08. 08 cũng vẫn thấp hơn shuffle khoảng 2.23 điểm F1 và thấp hơn visual-only 3.71 điểm.

## 4. Gate và coverage

Đây là gate trung bình trên toàn bộ query đã đánh giá, bao gồm các gate bị mask về 0. Không phải attention, xác suất đúng, hay tỷ lệ phần trăm đóng góp thông tin.

| Cấu hình | Query caption rate test | Caption của context được chọn có sẵn | q-gate test | c-gate test | Semantic mix |
| --- | ---: | ---: | ---: | ---: | ---: |
| `visual_only` | 0% | 0% | 0 | 0 | Không dùng |
| `query_only` | 84.63% | 0% | 0.012443 | 0 | Không dùng |
| `candidate_only` | 0% | 82.18% | 0 | 0.001122 | Không dùng |
| `both_fusion` | 84.63% | 82.87% | 0.014603 | 0.001574 | Không dùng |
| `full_both` | 84.63% | 85.74% | 0.014092 | 0.002006 | 0.497676 |
| `full_shuffle_both` | 84.63% | 83.24% | 0.007151 | 0.000118 | 0.498174 |
| `full_unscreened` | 100% | 100% | 0.019150 | 0.004411 | 0.498047 |
| `full_redacted` | 100% | 100% | 0.014921 | 0.003190 | 0.498708 |

Các nhánh được bật không bị vô hiệu hóa hoàn toàn bởi mask. Ở 02-06, phần lớn caption vẫn được dùng; không đúng nếu giải thích kết quả thấp là "tất cả caption đã bị lọc hết".

Gate candidate đặc biệt nhỏ. Ngay cả khi chỉ xét caption hợp lệ, gate candidate trung bình suy ra của 03 vẫn khoảng 0.001365, không phải chỉ do 17.82% context thiếu text. Điều này gợi ý hệ thống đang giảm trọng số nhánh text, nhưng cần đo chuẩn của residual `gate * text_features` so với visual features và thử can thiệp để biết ảnh hưởng thực tế.

Gate query đầu epoch 1 khá lớn: 02 khoảng 0.289, 04 khoảng 0.416, 05 khoảng 0.326, 07 khoảng 0.500, 08 khoảng 0.458; sau đó giảm xuống khoảng 0.01-0.02. Vì nhánh text đã ảnh hưởng quá trình tối ưu ở giai đoạn đầu, gate nhỏ cuối run không làm quá trình train tương đương baseline.

Coverage screened của query: train 1,573/1,880 = 83.67%; val 1,585/1,880 = 84.31%; test 1,591/1,880 = 84.63%. Pool có 324/376 caption hợp lệ = 86.17%. 07 và 08 dùng 100% ở cả query/pool.

Ở 08, q-gate/c-gate nhỏ hơn 07 nhưng không về 0; semantic mix vẫn gần 0.5. Kết quả không chứng minh model đã bỏ text hay selector đã ngừng dùng semantic. Cũng không được coi 0.003190 là phần trăm thông tin đóng góp của candidate.

Không nhầm 86.17% caption có sẵn trong toàn pool với `candidate_caption_rate` trong bảng test: chỉ số sau là tỷ lệ trên **context thực sự được chọn**, nên có thể đổi khi selector đổi.

## 5. Đọc các biểu đồ học

Đã xem cả 8 PNG được nhúng trong cell report; chúng được trích nguyên bản vào thư mục analysis, không vẽ lại thành kết quả mới.

- Baseline và candidate-only tiến nhanh rồi gần bão hòa ở vùng validation F1 khoảng 0.84.
- Query-only bão hòa gần 0.81; both-fusion/full-both gần 0.80; unscreened gần 0.776. Không thấy mô hình bị kẹt ở mức đoán ngẫu nhiên.
- Redacted có validation F1 tiến tới khoảng 0.81, cao hơn unscreened nhưng chưa tới baseline. Log có đủ epoch 1-50; epoch 39 được chọn theo loss, dù F1 cuối epoch 50 khoảng 0.8081 cao hơn F1 của checkpoint được chọn. Đây là đúng quy tắc chọn checkpoint đã khai báo, không phải lỗi lấy số.
- Shuffle đạt validation tốt hơn full-both, nhưng validation loss tốt nhất ở epoch 21 và không cải thiện đáng kể về cuối. Vì chọn checkpoint theo loss, không lấy điểm epoch 50 hoặc đỉnh F1 trên đường cong để thay số test đã báo cáo.
- Hầu hết đường validation đã phẳng về cuối. Chưa có bằng chứng chỉ tăng epoch sẽ khắc phục được khoảng cách lớn của các cấu hình query-text.
- Train loss và validation loss không phải cùng đại lượng hoàn toàn: train còn có SupCon, policy loss và entropy; validation dùng cross-entropy với label smoothing. Không lấy riêng khoảng cách hai đường để kết luận chắc chắn overfit hoặc model có lỗi.

## 6. Mức độ so sánh công bằng

**Đã kiểm tra được từ source/output:**

- Cả 8 report ghi completed, 50 epoch, LR horizon 50, seed 42, K=1; dữ liệu 1,880/1,880/1,880 và pool 376 ảnh train. Batch train đều 32, retrieval 128, feature-refresh 64; GPU memory log đều 95 GiB.
- Source model classes, `make_ablation_model_class`, `run_forward`, `create_ablation_model`, `train_ablation_epoch`, optimizer và loader split có cùng cấu trúc AST. Các notebook đã chạy khớp builder hiện tại, ngoại trừ đường dẫn DINOv3 repo/weights được đổi owner ở 01/03/07/08 và các cell rỗng thêm cuối 07. Không thấy sửa kiến trúc/loss ngoài mục đích ablation.
- Learning rate khai báo giống nhau: backbone 3e-6, head và selector 1e-4; label smoothing 0.1, SupCon weight 0.05, warmup 5, không freeze backbone. Pool seed mặc định 42; bảng sample pool đầu giống nhau.
- Best epoch trong report khớp epoch có validation loss thấp nhất trong log còn hiển thị. Bảng test ở cell final-test khớp report cuối trong sai số làm tròn.

**Chưa kiểm chứng độc lập được:**

- Thư mục cung cấp có 8 notebook, không có protocol JSON, CSV captions riêng từng run, cache embeddings, prediction CSV hay checkpoint. Không xác nhận được hash toàn bộ pool, split, CSV, weights và thư viện thực tế chỉ từ vài hàng sample/tên file giống nhau.
- Log cho thấy 03/04/05/08 sinh CSV mới từ đầu với cùng model/pipeline; 02/06/07 không hiển thị sinh lại từ đầu. Các số audit thống kê khớp nhau là tín hiệu tốt, nhưng **không phải bằng chứng CSV giống hệt từng dòng**. Không nên gộp thành "một CSV được khóa bằng hash" trước khi đối chiếu artifact.
- Log dòng epoch của query-only chỉ lưu 1-41. Bảng trạng thái báo đủ 50 và biểu đồ có phần cuối tới 50; chưa có bằng chứng run dừng ở 41, nhưng muốn xác minh chi tiết 42-50 cần `history.csv`.
- Có lỗi/cảnh báo cleanup DataLoader như `can only test a child process`, `Bad file descriptor` hoặc semaphore trong stream, nhưng không có cell `output_type=error`; các run đều có kết quả final-test. Chúng không tự làm metrics vô hiệu, cũng không nên gọi log hoàn toàn sạch. Cần xử lý worker trước khi chạy lại nhiều seed; không sửa giữa các run để tăng điểm.
- Không có prediction từng ảnh nên chưa tính được độ bất định paired, per-class F1, accuracy theo nhóm caption valid/invalid, hay mức đổi context khi bật semantic selector.

`visual_only` ở đây vẫn dùng query + ảnh context được chọn, không phải no-context classifier và cũng không phải tái lập nguyên cấu hình paper. Không dùng các điểm này để tuyên bố hơn/kém paper nếu backbone, budget, split và cách tổng hợp khác nhau.

## 7. Kết luận về leakage và chất lượng captions

Ba câu hỏi cần tách biệt:

1. **Caption có giúp downstream không?** Chưa có bằng chứng đáng tin cậy về lợi ích trong 8 run; các bản full, kể cả redacted, thấp hơn baseline.
2. **Caption có khớp nội dung ảnh không?** Metrics phân loại không trả lời trực tiếp. Chưa kiểm duyệt một tập ảnh-caption nên chưa kết luận Florence mô tả sai hay đúng đến mức nào.
3. **Có leakage/shortcut không?** Chưa thể chốt. Automatic caption từ pixel chứa từ trùng tên lớp không đồng nghĩa đã đưa ground-truth test vào model; trên DTD, tên lớp cũng là từ mô tả texture hợp lệ. Cần phân biệt thông tin thị giác hợp lệ, phụ thuộc từ vựng và vi phạm protocol thật sự.

Shuffle không làm giảm điểm so với full-both, unscreened không làm tăng điểm, và redacted cải thiện so với unscreened khi coverage đều 100%. Hiện không có mẫu kết quả "full tăng mạnh nhờ caption đúng hoặc từ chỉ lớp". Nhưng điểm thấp không phải chứng nhận không leak, và redaction tốt hơn cũng chưa là bằng chứng causal về từng từ hay bằng chứng đã hết thông tin lớp. Nó chỉ loại các tên/alias thuộc vocabulary hữu hạn, không bảo đảm loại mọi hàm ý về texture. Các kiểm tra artifact và nhiều seed vẫn cần thiết.

Các nguyên nhân khả dĩ như caption chung chung, mất thông tin khi rút ngắn, mismatch CLIP-text với texture, tối ưu nhánh fusion hay semantic retrieval chỉ là **giả thuyết**, chưa phải nguyên nhân được xác nhận. Không nên vội fine-tune captioner rồi quy mọi thay đổi cho chất lượng caption.

## 8. Việc nên làm tiếp theo

1. Đã đủ 8 run: giữ nguyên kết quả và protocol của đợt này. Xác minh CSV/weights/pool bằng artifact gốc, đặc biệt giữa 07 và 08, trước khi coi đây là đối chứng khóa hoàn toàn cùng dữ liệu. Không cần train lại chỉ vì bảng report làm tròn số.
2. Thu gọn artifact cần gửi: `protocol.json`, `<mode>_results.csv`, `runs/<mode>/seed_42/history.csv`, `val_metrics.json`, `test_metrics.json`, `val_predictions.csv`, `test_predictions.csv`, `shared/caption_model_inputs.csv` và báo cáo coverage/shuffle/redaction. Gửi hash hoặc CSV gốc để kiểm tra dữ liệu. Không cần gửi mọi checkpoint chỉ để lập bảng.
3. Sau đó ưu tiên xác nhận visual-only/candidate-only bằng thêm seed; chưa coi +4 ảnh là cải thiện chắc chắn. Nếu tiếp tục phát triển full, cần kiểm tra lại nhánh query/fusion bằng validation thay vì chọn theo test.
4. Khi có checkpoint và dữ liệu, một kiểm tra rẻ hơn retrain là tắt/đảo từng nhánh text của **cùng model**, giữ context cố định để đo fusion, rồi riêng rẽ cho chọn lại context để đo selector. Đây là đề xuất, chưa được chạy trong đánh giá này.
5. Kiểm duyệt một nhóm ảnh train-caption trước khi cải thiện captioner. Caption gốc vẫn là pseudo-caption; VLM/LoRA chỉ là bước sau khi hiểu chất lượng dữ liệu và giới hạn fusion.

### Cách viết kết luận nghiên cứu hiện tại

> Trên DTD split 1 với seed 42 và 50 epoch, baseline visual đạt 84.82% test macro-F1. Candidate-caption fusion đạt 85.03%, một mức tăng nhỏ chưa được xác nhận qua nhiều seed, trong khi các cấu hình có query-caption fusion giảm hiệu năng. Đảo caption ở cả hai vai trò cho kết quả cao hơn full alignment. Full-unscreened đạt 78.98%; che tên lớp/alias nâng lên 81.11% với coverage vẫn 100%, nhưng chưa vượt baseline. Các kết quả này chưa hỗ trợ lợi ích của semantic caption context trong thiết lập hiện tại và chưa đủ để kết luận có hoặc không có leakage. Cần xác minh tính đồng nhất dữ liệu và độ ổn định qua nhiều seed trước khi quy chênh lệch cho một cơ chế cụ thể.

## 9. Bằng chứng và tái lập phần đọc output

- Source: `../visual-only.ipynb`, `../query-only.ipynb`, `../candidate-only.ipynb`, `../both-fusion.ipynb`, `../full-both.ipynb`, `../full-shuffle-both.ipynb`, `../full-unscreened.ipynb`, `../full-redacted.ipynb`.
- `saved_output_evidence.json`: metrics, bảng audit, cấu hình source, hash AST của model/helper, hash notebook, log epoch còn lưu và đường dẫn PNG.
- `*_streams.txt`: text output nguyên được trích, phục vụ kiểm tra log bị cắt/cảnh báo. `*_cell*_output*.png`: PNG biểu đồ nguyên từ notebook.
- Chạy `python -X utf8 output_ablation/analysis/extract_evidence.py` từ project root để đọc lại. Script không thực thi code notebook, không tải model, không chạy training và không ghi vào notebook đầu vào.

Độ chính xác số liệu giới hạn bởi số chữ số được lưu trong output HTML; muốn phân tích thống kê sâu hơn cần metrics JSON/predictions gốc.
