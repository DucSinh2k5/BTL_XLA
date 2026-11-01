# Đề Cương Báo Cáo Bài Tập Lớn TACS / M-TACS

Ngày lập: 25/10/2026. Đây là khung báo cáo học phần, không phải bố cục bài báo khoa học và chưa phải bản báo cáo hoàn chỉnh.

## A. Định Hướng Chung

### A.1. Tên đề tài đề xuất

**Tìm hiểu và triển khai phương pháp TACS cho các bài toán thị giác máy tính và khảo sát mở rộng ngữ cảnh ảnh - văn bản.**

Tên ngắn hơn: **Triển khai TACS và khảo sát vai trò của ngữ cảnh văn bản trong phân loại ảnh và phân đoạn ảnh.**

Nên dùng từ “khảo sát mở rộng” thay vì khẳng định “cải tiến hiệu suất” ngay ở tên đề tài. Các kết quả hiện có cho thấy bổ sung caption có thể tăng hoặc giảm điểm tùy vị trí sử dụng, phương pháp sinh caption và bài toán.

### A.2. Trọng tâm báo cáo

Báo cáo cần thể hiện được chuỗi công việc: hiểu phương pháp gốc, xây dựng pipeline, triển khai trên một số bộ dữ liệu, đề xuất mở rộng, làm thí nghiệm có đối chứng và xây dựng giao diện minh họa. Không bắt buộc tái lập toàn bộ dataset, bảng và cấu hình của paper để hoàn thành một bài tập lớn có chất lượng.

Phần thuyết phục nhất của project hiện tại là nghiên cứu thực nghiệm về vai trò của caption: caption giúp ở đâu, gây nhiễu ở đâu, các đối chứng nói được gì và chưa nói được gì. Kết quả không cải thiện vẫn có giá trị nếu cách so sánh và phân tích rõ ràng.

### A.3. Phạm vi nên công bố

- Phân loại ảnh: trình bày các dataset thực sự có triển khai và kết quả sử dụng trong báo cáo; DTD là nhóm khảo sát ngữ cảnh văn bản chính. SUN397 phục vụ thí nghiệm liên quan và demo; Cars/CUB chỉ đưa vào bảng khi có bằng chứng và cấu hình phù hợp.
- Phân đoạn ảnh: khảo sát mở rộng trên ISIC 2017; ghi rõ các điều chỉnh so với paper và không suy rộng thành kết luận ứng dụng lâm sàng.
- Demo: giao diện phân loại SUN397 gồm hai luồng TACS và M-TACS; mô tả trung thực phần đã hoạt động và phần chưa kết nối checkpoint.
- Những hướng alignment, semantic selector mới, token fusion/resampler và ASWP chưa có kết quả chỉ thuộc kế hoạch kiểm nghiệm hoặc hướng phát triển.

### A.4. Quy mô gợi ý

Khoảng **35-50 trang nội dung chính**, chưa tính bìa, mục lục, tài liệu tham khảo và phụ lục; điều chỉnh theo mẫu của khoa/giảng viên. Không cần kéo dài bằng cách chép lại toàn bộ mã nguồn hoặc liệt kê mọi notebook lịch sử.

| Phần | Số trang tham khảo |
| --- | ---: |
| Chương 1: Tổng quan đề tài | 3-4 |
| Chương 2: Cơ sở lý thuyết và phương pháp TACS | 6-8 |
| Chương 3: Phân tích, thiết kế và triển khai | 5-7 |
| Chương 4: Mở rộng ngữ cảnh ảnh - văn bản | 6-8 |
| Chương 5: Thực nghiệm và đánh giá | 10-14 |
| Chương 6: Chương trình demo | 3-5 |
| Chương 7: Kết luận và hướng phát triển | 2-3 |

## B. Phần Đầu Báo Cáo

### B.1. Bìa và trang thông tin

Ghi tên trường, khoa, môn học, đề tài, giảng viên hướng dẫn, sinh viên/nhóm thực hiện và thời gian. Nếu làm nhóm, thêm bảng phân công thực tế; không điền vai trò hoặc mức đóng góp khi chưa được xác nhận.

### B.2. Tóm tắt báo cáo

Viết khoảng nửa đến một trang, nêu bài toán, ý tưởng TACS, phạm vi triển khai, mở rộng caption và nhận xét chính từ thí nghiệm. Chỉ đưa số liệu đã đối chiếu với output; không biến một mức tăng nhỏ ở một seed thành kết luận cải thiện ổn định.

### B.3. Mục lục và danh mục

Thêm mục lục, danh mục hình, danh mục bảng và bảng từ viết tắt. Các từ cần giải thích gồm TACS, M-TACS, ViT, DINOv3, VLM, CLIP, candidate pool, query, context, Macro-F1, Dice và IoU.

### B.4. Lời cảm ơn và các trang theo mẫu

Lời cảm ơn, phiếu nhận xét hoặc cam đoan chỉ thêm nếu học phần yêu cầu. Không cần dành nhiều trang cho phần nghi thức; ưu tiên nội dung triển khai và kết quả.

## Chương 1. Tổng Quan Đề Tài

### 1.1. Đặt vấn đề

Giới thiệu phân loại và phân đoạn ảnh, cùng khó khăn khi mô hình chỉ xử lý một ảnh độc lập. Từ đó đặt câu hỏi liệu các ảnh liên quan có cung cấp ngữ cảnh hữu ích cho dự đoán hay không.

### 1.2. Lý do lựa chọn TACS

Giải thích ý tưởng chọn ngữ cảnh phù hợp với tác vụ thay vì chỉ chọn ảnh giống nhau về biểu diễn. Nêu vì sao hướng tiếp cận này phù hợp để tìm hiểu retrieval, representation learning và cách phối hợp nhiều nguồn thông tin trong một bài tập lớn.

### 1.3. Mục tiêu thực hiện

Nêu các mục tiêu cụ thể: hiểu và triển khai TACS trong phạm vi lựa chọn; xây dựng thí nghiệm phân loại/phân đoạn; bổ sung caption; đánh giá bằng ablation; tổ chức mã nguồn và xây dựng demo. Phân biệt mục tiêu đã hoàn thành với các mục còn tiếp tục thực hiện.

### 1.4. Phạm vi và giới hạn

Liệt kê dataset, tác vụ và nhóm mô hình thực sự được báo cáo. Ghi rõ giới hạn tài nguyên, số seed, số split, khác biệt backbone/resolution/budget so với paper và trạng thái tích hợp web.

### 1.5. Các công việc chính và bố cục báo cáo

Tóm tắt những gì sinh viên trực tiếp triển khai, điều chỉnh và kiểm nghiệm trên nền phương pháp gốc. Kết thúc bằng một đoạn giới thiệu nội dung các chương; không gọi toàn bộ thuật toán TACS hoặc các mô hình pretrained là đề xuất mới của mình.

## Chương 2. Cơ Sở Lý Thuyết Và Phương Pháp TACS

### 2.1. Phân loại ảnh và phân đoạn ảnh

Trình bày đầu vào, đầu ra và sự khác nhau giữa gán nhãn toàn ảnh với dự đoán mặt nạ theo pixel. Dùng một ví dụ phân loại DTD/SUN397 và một ví dụ phân đoạn ISIC để gắn lý thuyết với nội dung triển khai.

### 2.2. Vision Transformer và đặc trưng DINOv3

Giải thích patch, token, attention và cách lấy biểu diễn ảnh từ backbone pretrained ở mức đủ để hiểu hệ thống. Nêu các backbone S/16, B/16, L/16 xuất hiện trong project và lý do không so sánh trực tiếp các run khác backbone để quy chênh lệch cho một thay đổi khác.

### 2.3. Query, candidate pool và context

Định nghĩa ba thành phần bằng hình minh họa: query là ảnh đang cần dự đoán, pool là tập ảnh có thể được chọn và context là ảnh được chọn cho query đó. Giải thích K là số ảnh ngữ cảnh được sử dụng, không phải K-fold cross-validation.

### 2.4. Quy trình TACS

Mô tả pipeline từ trích xuất đặc trưng, chấm điểm candidate, chọn context đến xử lý query cùng context để dự đoán. Đặt bên cạnh các cách đối chứng như No-Context, Random Context và chọn bằng độ tương đồng đặc trưng.

### 2.5. Học bộ chọn ngữ cảnh

Trình bày mục tiêu để selector chọn ảnh hữu ích cho tác vụ và cơ chế cập nhật selector trong phương pháp gốc, sau đó nêu phiên bản thực tế đang dùng trong nhóm notebook báo cáo. Chỉ đưa công thức/loss tương ứng với implementation đã kiểm tra; không coi mọi biến thể Gumbel, categorical policy hoặc shortlist là cùng một cấu hình.

### 2.6. Mô hình sinh caption và CLIP

Phân biệt Florence/Qwen3-VL là mô hình nhận ảnh để sinh văn bản, còn CLIP trong pipeline dùng để mã hóa hoặc so sánh biểu diễn. Giải thích caption không do CLIP tự sinh ra và việc hai vector có cùng kích thước sau projection không tự bảo đảm chúng đã được căn chỉnh ngữ nghĩa tối ưu.

### 2.7. Tổng kết cơ sở lý thuyết

Chốt lại các khái niệm cần dùng trong chương thiết kế và chỉ ra cơ sở cho việc bổ sung ngữ cảnh văn bản. Các bài báo multimodal khác có thể được giới thiệu ngắn như nguồn tham khảo cho hướng phát triển; không cần biến chương này thành một khảo sát literature dài như paper.

## Chương 3. Phân Tích, Thiết Kế Và Triển Khai Hệ Thống

### 3.1. Yêu cầu và kiến trúc tổng thể

Nêu các yêu cầu: chạy offline trên Kaggle, đọc dữ liệu/split chính xác, lưu kết quả để resume, kiểm tra nguồn caption và có giao diện minh họa. Vẽ sơ đồ gồm dữ liệu, caption pipeline, feature/text bank, huấn luyện, đánh giá và web demo.

### 3.2. Dữ liệu và tổ chức train/validation/test

Mô tả từng dataset được dùng, số lớp hoặc loại mặt nạ, các file split và số mẫu thực tế. Với DTD cần ghi rõ split chính thức 1 và số lượng 1.880/1.880/1.880; nếu chỉ chạy split 1 thì không gọi đó là kết quả trung bình 10 split.

### 3.3. Tiền xử lý và xây dựng candidate pool

Trình bày resize, augmentation, normalization và cách lấy pool từ training data. Giải thích rằng ảnh thuộc pool vẫn có thể là training query, nhưng phải loại chính nó khỏi lựa chọn context; validation/test không được bổ sung vào pool trong các thí nghiệm này.

### 3.4. Triển khai TACS cho phân loại ảnh

Mô tả backbone, selector, cách đưa context vào classifier và đầu phân loại trong phiên bản thực tế. Có bảng tách phần kế thừa paper và phần điều chỉnh của project, bao gồm backbone, kích thước ảnh, cơ chế fusion, losses và training budget.

### 3.5. Triển khai phân đoạn ISIC 2017

Mô tả cặp query/context, phần kết hợp đặc trưng, segmentation head và đầu ra mặt nạ. Ghi đúng cấu hình 448 và kiến trúc của run đã có, thay vì mặc định nó giống hoàn toàn segmentation trong paper hoặc giống backbone L/16 của nhóm DTD.

### 3.6. Quy trình huấn luyện và suy luận

Trình bày một epoch, cập nhật feature bank, validation, tiêu chí chọn best checkpoint và final test. Làm rõ khác biệt giữa training query có nhãn để tính loss với ảnh mới khi inference không có nhãn; không mô tả việc dùng ảnh test để học tiếp như một phần của đánh giá hiện tại.

### 3.7. Tổ chức mã nguồn và khả năng chạy lại

Giới thiệu ngắn vai trò của `notebooks`, `src`, `data`, `experiments`, `artifacts`, `reports`, `tests` và `apps`. Chọn vài cơ chế thực sự quan trọng như seed, split/pool manifest, CSV hash, checkpoint identity, resume và cấu hình offline; phần cây thư mục đầy đủ để ở phụ lục.

## Chương 4. Khảo Sát Mở Rộng TACS Với Ngữ Cảnh Văn Bản

### 4.1. Động cơ bổ sung caption

Đặt giả thuyết caption có thể bổ sung mô tả chất liệu, màu sắc, cấu trúc hoặc thông tin ngữ nghĩa mà biểu diễn ảnh chưa khai thác thuận lợi. Đồng thời nêu nguy cơ caption sai, dư thừa, mất chi tiết hoặc đưa shortcut vào mô hình; vì vậy cần kiểm nghiệm chứ không mặc định caption sẽ tăng điểm.

### 4.2. Pipeline tạo captions.csv

Mô tả luồng ảnh -> Florence/Qwen3-VL -> raw caption -> xử lý và audit -> CSV. Giải thích các cột quan trọng, cơ chế tạo/resume theo ảnh và yêu cầu mô hình sinh caption chỉ nhận pixels, không nhận nhãn thật hoặc tên thư mục lớp làm prompt.

### 4.3. Các phương pháp tạo và lựa chọn caption

Giới thiệu Florence-Prefix, FirstValid, SingleGrounded, MultiGrounded và hai nguồn Qwen3-VL 8B/32B theo đúng thủ tục đã triển khai. Phân biệt thay đổi cách chọn/rút gọn caption với fine-tune captioner: không phải cả sáu phương pháp đều có cập nhật trọng số mô hình sinh caption.

Nếu đưa nhánh decoder-LoRA vào báo cáo, trình bày riêng dữ liệu reference, nguồn teacher, cách chia dữ liệu và quá trình fine-tune có bằng chứng. Không lấy output classifier dùng caption LoRA để thay cho log training hoặc đánh giá chất lượng riêng của captioner.

### 4.4. Chính sách kiểm tra caption và kiểm soát shortcut

Giải thích kiểm tra số từ, bỏ boilerplate, phát hiện tên lớp/alias và mask caption không đủ điều kiện. Với policy `dtd_lexical_v2`, mô tả đúng rằng điểm CLIP semantic được dùng để audit, không áp ngưỡng 0.30 làm điều kiện loại cứng như policy trước.

Nêu rõ lexical/semantic audit không chứng minh caption hoàn toàn không chứa thông tin lớp. `caption_valid` chỉ là điều kiện sử dụng theo policy, không phải chứng nhận caption là ground truth đúng và độc lập.

### 4.5. Vị trí sử dụng text trong M-TACS

Trình bày ba vai trò khác nhau: caption query vào fusion, caption candidate được chọn vào fusion và caption tham gia chấm điểm selector. Giải thích không tồn tại một caption riêng của “selector”; selector dùng biểu diễn caption của query/candidate tùy cấu hình.

Minh họa projection, gate và cách kết hợp vector trong implementation đang dùng. Với `candidate_only`, selector chỉ dùng ảnh rồi lấy caption ở đúng chỉ số candidate đã chọn; không dùng caption query ở train hoặc inference.

### 4.6. Thiết kế tám cấu hình ablation

Đưa bảng bật/tắt từng thành phần và giải thích câu hỏi mà mỗi đối chứng trả lời. Phần này mô tả thiết kế; số liệu và nhận xét để ở Chương 5.

| Cấu hình | Text query vào fusion | Text candidate vào fusion | Text vào selector | Mục đích đối chứng |
| --- | --- | --- | --- | --- |
| `visual_only` | Không | Không | Không | Mức tham chiếu có context ảnh, không có text |
| `query_only` | Có | Không | Không | Tách tác động của caption query |
| `candidate_only` | Không | Có | Không | Tách tác động của caption context |
| `both_fusion` | Có | Có | Không | Đánh giá kết hợp hai nhánh text trong fusion |
| `full_both` | Có | Có | Có | Đánh giá khi text còn tham gia chọn context |
| `full_shuffle_both` | Có | Có | Có | Đảo caption, giữ mask, kiểm tra vai trò khớp ảnh - text |
| `full_unscreened` | Có | Có | Có | Dùng caption trước lọc để khảo sát ảnh hưởng policy/shortcut |
| `full_redacted` | Có | Có | Có | Che tên lớp/alias ở cả hai vai trò |

`visual_only` vẫn có context ảnh, không phải baseline `No-Context` của paper. Shuffle, unscreened và redacted là đối chứng phục vụ diễn giải; riêng từng đối chứng không thể xác nhận tuyệt đối có hoặc không có leakage.

### 4.7. Mở rộng caption cho segmentation

Mô tả cách bổ sung caption query/candidate và semantic selector cho ISIC 2017, cùng điểm khác so với phân loại. Trình bày chính sách mô tả đặc trưng nhìn thấy được; không đưa nhãn chẩn đoán, mask thật hoặc thông tin không có trong ảnh vào caption đầu vào.

### 4.8. Thiết kế so sánh kiến trúc với CSV cố định

Giới thiệu mục đích notebook mới `candidate_only` + Florence-Prefix: tạo baseline trên đúng một file CSV trước khi thay alignment/selector/fusion/pooling. Run được gửi ngày 25/10/2026 đã có test Macro-F1 0.853032; xem [audit baseline 01](../multimodal_architecture_ablation/dtd/BASELINE_01_FLORENCE_PREFIX_CANDIDATE_ONLY.md). Không gán điểm candidate_only cũ cho nó và không mô tả các bước nâng cấp dự kiến như đã triển khai xong.

## Chương 5. Thực Nghiệm Và Đánh Giá Kết Quả

### 5.1. Môi trường và cấu hình thực nghiệm

Liệt kê GPU, thư viện chính, phiên bản quan trọng, backbone, resolution, optimizer, learning rate, batch size, epoch, seed và cách chọn checkpoint. Dùng bảng theo nhóm thí nghiệm; không ghi một cấu hình L/16/50 epoch chung cho các run ISIC hoặc các baseline lịch sử có thiết lập khác.

### 5.2. Quy tắc so sánh và cách thu thập kết quả

Nêu nguồn số liệu là output notebook, metric JSON, prediction CSV hoặc log đã lưu; ghi tên run và cấu hình đi kèm. Tách bảng so sánh cùng protocol khỏi bảng kết quả lịch sử khác protocol; không lấy kết quả paper điền vào ô kết quả triển khai của mình.

### 5.3. Chỉ số đánh giá

Giới thiệu Accuracy và Macro-F1 cho phân loại, Dice và IoU cho phân đoạn, kèm công thức và cách tính/aggregation thực tế trong code. Phân biệt Macro-F1 với Accuracy và mean per-image IoU với IoU gộp nếu các nguồn báo cáo dùng định nghĩa khác nhau.

### 5.4. Kết quả TACS cho phân loại ảnh

Trình bày các method baseline và TACS trên dataset đã lựa chọn, kèm backbone/split/seed. Có thể lấy ý tưởng từ Table 2 của paper nhưng chỉ giữ cột thực sự có kết quả phù hợp; các run khác cấu hình nên đặt thành nhóm tham khảo riêng.

Nếu đã có thí nghiệm loại context hoặc cơ chế tối ưu, thêm các bảng theo tinh thần Table 4 và Table 5. Nếu chưa có output thì ghi “chưa thực hiện” trong kế hoạch/phụ lục, không cần ép đầy đủ mọi bảng của paper trong bản báo cáo nộp.

### 5.5. Kết quả segmentation ISIC 2017

So sánh run visual với full_both dùng caption Qwen8B theo các bằng chứng đã có; nêu cả validation và test khi cần giải thích khác biệt. Kết quả giảm trên test vẫn phải báo cáo, và cần ghi các khác biệt kiến trúc/capacity/training trước khi quy toàn bộ nguyên nhân cho nội dung caption.

Florence-Prefix ISIC hiện có notebook để chạy nhưng chưa có metrics được xác nhận trong nhóm bằng chứng này. Chỉ thêm hàng kết quả sau khi có output thật; không suy rằng Prefix tốt trên DTD thì cũng tốt trên ISIC.

### 5.6. Kết quả tám cấu hình text ablation trên DTD

Đây là phần phân tích chính: đưa bảng tám cấu hình, Accuracy/Macro-F1 và chênh lệch so với `visual_only`. Trả lời text query, text candidate và text-aware selector tác động ra sao, đồng thời dùng shuffle/unscreened/redacted để thảo luận ảnh hưởng của độ khớp và screening.

Mức tăng nhỏ của candidate_only ở một seed nên mô tả là “ghi nhận trong lần chạy này”, chưa khẳng định bền vững hoặc có ý nghĩa thống kê. Gate/coverage giúp kiểm tra nhánh text có hoạt động nhưng không tự chứng minh nhánh đó hữu ích cho tổng quát hóa.

### 5.7. So sánh nguồn và phương pháp tạo caption

Trình bày sáu phương pháp Florence-Prefix, FirstValid, SingleGrounded, MultiGrounded, Qwen8B và Qwen32B dưới cùng nhóm classifier full_both đã chạy. Phân tích chất lượng downstream, tỷ lệ caption hợp lệ và độ khớp ảnh - text; mô hình lớn hơn hoặc caption dài/đẹp hơn không tự đồng nghĩa điểm phân loại cao hơn.

Đưa decoder-LoRA vào hàng hoặc bảng riêng nếu cấu hình và dữ liệu so sánh được giải thích đầy đủ. Không diễn giải bảng full_both này thành bảng xếp hạng caption cho candidate_only, vì vị trí sử dụng text cũng là một biến thí nghiệm.

### 5.8. Kiểm nghiệm kiến trúc mới trên Florence-Prefix

Khi có kết quả, mở đầu bằng baseline `candidate_only` + Prefix rồi so sánh các thay đổi độc lập về alignment, selector, token fusion và pooling. Khi chưa có kết quả, giữ mục này trong đề cương/kế hoạch và bỏ khỏi phần kết quả chính của bản nộp; không cần chờ hoàn tất mọi ý tưởng để viết báo cáo.

### 5.9. Phân tích định tính và các trường hợp sai

Đưa ví dụ gồm query, caption, context được chọn, caption context, dự đoán và nhãn thật; phân tích cả trường hợp đúng và sai. Với segmentation, đặt ảnh gốc, ground-truth mask và predicted mask cạnh nhau; chỉ dùng attention/heatmap khi đã trích xuất bằng mô hình thật và nêu phương pháp.

### 5.10. Chi phí tính toán

Nếu có log, báo cáo thời gian sinh CSV, thời gian train, VRAM và kích thước model/cache. So sánh thời gian chỉ có ý nghĩa khi kèm GPU, batch size, số ảnh và cấu hình; không dùng ký ức “20 phút” hay “8 giờ” làm số đo công bằng nếu thiếu điều kiện chạy.

### 5.11. Thảo luận và giới hạn

Tổng hợp câu trả lời cho các câu hỏi: context ảnh có tác dụng gì, text ở vị trí nào phù hợp, caption tốt hơn có luôn cải thiện downstream không và fusion hiện tại còn hạn chế gì. Thừa nhận giới hạn single-seed/split1, nguy cơ shortcut, khác biệt protocol và thiếu một số artifact; không dùng test để chọn ngưỡng hoặc chọn phiên bản tốt nhất rồi coi đó là đánh giá độc lập.

## Chương 6. Xây Dựng Chương Trình Demo

### 6.1. Mục tiêu và phạm vi demo

Giới thiệu mục đích minh họa một ảnh đầu vào đi qua TACS/M-TACS để phân loại SUN397, không thay thế đánh giá trên test set. Trình bày hai trang TACS và M-TACS, và giải thích vì sao chỉ demo classification trong phạm vi hiện tại.

### 6.2. Kiến trúc frontend và backend

Mô tả giao diện upload, kiểm tra ảnh, API trạng thái, API dự đoán và cấu trúc response chứa kết quả/evidence. Vẽ một sơ đồ request-response ngắn; không cần biến chương này thành tài liệu API đầy đủ.

### 6.3. Các chức năng và luồng sử dụng

Trình bày chọn ảnh, xem cấu trúc mô hình, yêu cầu dự đoán và vị trí hiển thị query caption, candidate được chọn, candidate caption và kết quả phân loại. Tách chức năng giao diện đã xây dựng khỏi chức năng inference thật còn cần checkpoint, class map, preprocessing và candidate bank tương thích.

### 6.4. Trực quan hóa ngữ cảnh và attention

Giải thích mục đích hiển thị query/context và các vùng chú ý theo vai trò tương ứng, tham khảo Figure 4 của paper. Ô heatmap đang để trống phải được mô tả là phần chờ tích hợp; không dùng ảnh minh họa ngẫu nhiên làm evidence của mô hình và không gọi điểm CLIP là attention map.

### 6.5. Kiểm thử và trạng thái hoàn thiện

Nêu các ca kiểm tra ảnh không hợp lệ, trạng thái model chưa sẵn sàng, response API và hiển thị frontend. Theo trạng thái project hiện tại, checkpoint tương thích chưa được nối vào demo; không tuyên bố hệ thống đã phân loại được ảnh bên ngoài hoặc đã có demo segmentation end-to-end.

## Chương 7. Kết Luận Và Hướng Phát Triển

### 7.1. Các kết quả thực hiện được

Tóm tắt mức độ hoàn thành mục tiêu: triển khai pipeline, tổ chức project, thí nghiệm TACS, khảo sát caption, các bộ đối chứng và giao diện demo. Nhận xét kết quả theo bằng chứng đã thu được, không chỉ chọn hàng có điểm cao nhất để đại diện toàn bộ project.

### 7.2. Hạn chế còn tồn tại

Nêu giới hạn dữ liệu, tài nguyên, số lần chạy, độ tin cậy caption, sự khác nhau giữa các protocol và trạng thái kết nối checkpoint cho web. Đặc biệt làm rõ caption model-generated/teacher-generated chưa mặc nhiên là ground truth được người kiểm duyệt độc lập.

### 7.3. Hướng phát triển

Đề xuất kiểm nghiệm alignment ảnh - text, semantic selection từ query image sang candidate caption, token-level fusion/resampler và adaptive pooling/ASWP theo từng ablation. Đặt notebook `candidate_only` + Prefix làm baseline cho nhóm mới, giữ CSV/protocol cố định rồi mới đánh giá khả năng áp dụng với caption nguồn khác.

Các hướng tiếp theo có thể gồm nhiều seed/split, kiểm duyệt reference, hoàn thiện inference/attention thật và demo segmentation. “Học tiếp tục từ ảnh đưa vào test” chỉ là hướng khác cần thiết kế protocol riêng; không gộp vào điểm test cố định hiện tại.

## C. Tài Liệu Tham Khảo Và Phụ Lục

### C.1. Tài liệu tham khảo

Trích dẫn paper TACS, các backbone và captioner sử dụng, CLIP, nguồn dataset, thư viện và các paper multimodal làm cơ sở cho hướng phát triển. Dùng một kiểu trích dẫn nhất quán theo yêu cầu môn học và kiểm tra metadata từ nguồn gốc khi viết bản hoàn chỉnh.

### C.2. Phụ lục cấu hình và hướng dẫn chạy

Đưa danh sách input Kaggle, đường dẫn mẫu, lệnh chạy web, cấu hình từng nhóm và mô tả output/checkpoint. Chỉ trích những đoạn code ngắn thật sự cần giải thích; liên kết repository/notebook cho phần code đầy đủ.

### C.3. Phụ lục kết quả chi tiết

Đặt bảng đầy đủ theo lớp, confusion matrix lớn, learning curves phụ, audit caption, các run lịch sử và các ô chưa thực hiện ở đây. Không để danh sách kết quả thiếu hoặc khác protocol làm loãng bảng kết quả chính.

## D. Danh Sách Bảng Và Hình Nên Chuẩn Bị

### D.1. Bảng

| Bảng gợi ý | Nội dung | Vị trí |
| --- | --- | --- |
| Dữ liệu thực nghiệm | Dataset, task, số lớp/mặt nạ, split, số mẫu, vai trò trong project | Chương 3 |
| Phương pháp gốc và bản triển khai | Thành phần kế thừa và điều chỉnh; không mặc định tương đương paper | Chương 3 |
| Tám cấu hình ablation | Query fusion, candidate fusion, semantic selector, caption view | Chương 4 |
| Cấu hình huấn luyện | Backbone, resolution, pool, K, loss, epoch, seed, checkpoint criterion | Chương 5 |
| Phân loại TACS và baseline | Kết quả dự án trên các dataset có protocol so sánh được | Chương 5 |
| Segmentation ISIC | Dice/IoU, visual và các caption variant đã có output | Chương 5 |
| Text-context ablation DTD | Tám cấu hình; Accuracy, Macro-F1, delta và coverage khi có | Chương 5 |
| Caption-method ablation DTD | Sáu phương pháp caption trên full_both; LoRA có chú thích riêng nếu đưa vào | Chương 5 |
| Architecture ablation với CSV cố định | Baseline Prefix và từng thay đổi mới; chỉ thêm sau khi chạy | Chương 5 hoặc kế hoạch |
| Trạng thái demo | Chức năng đã có, đã kiểm thử và còn chờ artifact | Chương 6 |

Đánh số bảng theo chương của báo cáo, ví dụ Bảng 5.1, 5.2; không cần giữ tên Table 2/3/4/5 của paper. Khi đối chiếu số liệu công bố, tách rõ cột “paper” và “triển khai của đề tài”, cùng chú thích khác biệt cấu hình.

### D.2. Hình

- Sơ đồ TACS: query, candidate pool, selector và downstream model.
- Ví dụ dataset phân loại và ảnh/mask ISIC.
- Sơ đồ sinh caption, screening, CLIP encoding và lưu CSV/cache.
- Sơ đồ M-TACS phân biệt ba vai trò của text.
- Learning curves của các run đại diện, không cần dán đường cong của mọi notebook.
- Biểu đồ so sánh tám text ablation và sáu caption methods.
- Ví dụ đúng/sai kèm query, context và caption; ví dụ segmentation visual so với caption.
- Ảnh chụp hai trang demo với trạng thái model được ghi rõ.

Chỉ đưa hình attention thật nếu đã có artifact phù hợp. Mỗi hình phải có chú thích và được phân tích trong nội dung, không dùng chỉ để tăng số trang.

## E. Bằng Chứng Có Thể Dùng Để Viết

Các liên kết dưới đây chỉ dẫn đến file hiện có; notebook template không đồng nghĩa đã có kết quả GPU.

| Nhóm | Nguồn hiện có | Cách sử dụng |
| --- | --- | --- |
| Tám text ablation DTD | [Notebook kết quả và phân tích](../../experiments/completed/text_context_ablation/dtd/) | Cơ sở chính cho mục 5.6 |
| Sáu caption methods DTD | [Báo cáo và bằng chứng](../caption_ablation/dtd/) và [notebook kết quả](../../experiments/completed/caption_ablation/dtd/) | Mục 5.7; giữ classifier full_both và protocol đi kèm |
| Decoder-LoRA-caption classifier | [Run đã lưu](../../experiments/completed/caption_decoder_lora/dtd/) | Kết quả classifier; không thay cho log fine-tune captioner |
| Segmentation ISIC | [So sánh visual và Qwen8B](../paper_tables/SO_SANH_ISIC_VISUAL_VA_QWEN8B.md) | Mục 5.5, kèm giới hạn so sánh |
| Các phần liên quan paper | [Bản đồ bảng và notebook](../../notebooks/01_paper_reproduction/README.md) | Xác định vai trò implementation; không chứng nhận mọi run đúng paper |
| Baseline Prefix candidate_only mới | [Notebook và hướng dẫn](../../notebooks/02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/README.md) | Đã có output split1/seed42: test Macro-F1 0.853032; cần giữ các giới hạn từ audit |
| Web demo | [Frontend/backend và trạng thái tích hợp](../../apps/sun397_demo/README.md) | Chương 6, không giả lập kết quả khi thiếu checkpoint |
| Các kết quả còn thiếu | [Rà soát độ sẵn sàng](../paper_readiness/KE_HOACH_KET_QUA_BAO_CAO_TACS.md) | Kiểm tra cập nhật mới ở đầu file; không coi snapshot cũ là trạng thái cuối |

## F. Thứ Tự Viết Và Quy Tắc Chốt Bản Nộp

1. Chốt phạm vi dataset và các run dùng trong báo cáo, lập bảng kiểm nguồn metrics/config trước khi viết nhận xét.
2. Viết Chương 3-4 từ implementation thực tế; hoàn thiện bảng/hình Chương 5 từ output đã lưu.
3. Viết Chương 2 vừa đủ để đọc hiểu các pipeline đã triển khai, rồi viết Chương 1 theo đúng phạm vi đó.
4. Chụp và mô tả trạng thái demo thực tế cho Chương 6, nêu riêng phần chờ nối checkpoint.
5. Viết kết luận, tóm tắt và các giới hạn sau cùng; không hứa kết quả cho thí nghiệm đang chạy hoặc mới dự kiến.
6. Kiểm tra mọi bảng phân biệt rõ val/test, accuracy/F1, phần trăm/tỷ lệ và từng backbone/split/seed; đối chiếu lần cuối với file nguồn.

**Bản nộp không cần chờ tất cả hướng phát triển hoàn thành.** Một báo cáo có triển khai rõ, thí nghiệm có đối chứng và phân tích trung thực cả kết quả tăng lẫn giảm phù hợp hơn việc cố lấp mọi ô của paper bằng các run không cùng điều kiện.
