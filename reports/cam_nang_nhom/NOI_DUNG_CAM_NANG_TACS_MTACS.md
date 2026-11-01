# Lời mở đầu và cách sử dụng

Cẩm nang này dành cho thành viên nhóm chưa theo dõi toàn bộ quá trình làm dự án. Mục tiêu là giúp người đọc giải thích được TACS làm gì, văn bản được thêm vào đâu, vì sao phải làm ablation, kết quả hiện có cho phép kết luận đến mức nào, và cần mở file nào khi muốn chạy lại hoặc sửa báo cáo.

Điểm cần nhớ ngay từ đầu là **caption đã làm thay đổi mô hình, nhưng chưa tạo ra lợi ích nhất quán**. Trong tám ablation DTD ban đầu, candidate_only đạt Test Macro F1 0.850321 so với visual_only 0.848168. Trong chuỗi kiến trúc dùng CSV Prefix cố định, baseline S01 đạt 0.853032; các bước chuẩn hóa, alignment và semantic selector chưa vượt baseline này. Trên ISIC, bản Qwen full_both giảm so với visual. Đây là các kết quả có giá trị để phân tích, không phải lý do phải tìm cách viết thành một cải tiến luôn thành công.

Một cách đọc tốt là tách ba câu hỏi: nguyên lý có hợp lý không, code có thực hiện đúng nguyên lý không, và thực nghiệm có chứng minh lợi ích không. Ba câu hỏi có liên quan nhưng không đồng nghĩa. Ví dụ, đưa hai vector về cùng số chiều giải quyết được yêu cầu cộng tensor; điều đó chưa chứng minh hai modality đã có cùng ý nghĩa, và càng chưa bảo đảm điểm phân loại tăng.

**Lộ trình 30 phút.** Đọc chương 1, 2, 5, 9, 12 và 21. Người đọc sẽ nắm sơ đồ hệ thống, ba vai trò của text, kết quả chính và các giới hạn cần giữ khi thuyết trình.

**Lộ trình một buổi.** Đọc tiếp chương 3, 4, 6, 8, 10, 11, 15 và 16. Người đọc có thể theo dõi một query, chọn đúng notebook và chuẩn bị một run Kaggle.

**Lộ trình người tiếp quản.** Đọc toàn bộ, đặc biệt chương 17–20 và phụ lục. Người tiếp quản cần biết kiểm tra danh tính CSV, khôi phục training, đối chiếu output và sửa bảng mà không làm mất nguồn gốc kết quả.

Các ví dụ số học, ảnh giả định và caption minh họa trong phần giải thích được ghi rõ là minh họa. Chúng không phải ảnh đã gán nhãn thật hay kết quả của notebook. Các bảng thực nghiệm dùng output đã lưu và mã nguồn đối chiếu ở phụ lục. Mã A chỉ ablation vai trò text; C chỉ nguồn caption; S chỉ nghiên cứu kiến trúc; I chỉ phân đoạn; V và U chỉ một số run visual và query text lịch sử.

Đường dẫn trong cẩm nang được viết tương đối từ thư mục dự án `TACS`, trừ những lệnh có ghi rõ đường dẫn Windows hoặc Kaggle. Không đổi tên notebook output cũ chỉ để phù hợp với một diễn giải mới. Tên sai chính tả như `stydu2_da_train.ipynb` vẫn được giữ trong danh mục nguồn để tìm đúng file đã chạy.

# Mục lục

[[CONTENTS]]

# 1 Bản đồ toàn bộ dự án

## 1.1 Từ một ảnh đến một quyết định

Bài toán phân loại nhận một ảnh query và trả một nhãn trong tập lớp đã biết. TACS bổ sung một bước trước phân loại: tìm trong candidate pool một ảnh context có thể giúp xử lý query. Mô hình sau đó sử dụng query cùng context, thay vì chỉ nhìn query độc lập. Chọn context tốt ở đây nghĩa là giúp giảm lỗi tác vụ, không nhất thiết là ảnh giống nhất theo cảm nhận của người xem.

Mở rộng M TACS thêm mô tả văn bản. Captioner như Florence hoặc Qwen nhìn pixel để sinh câu; CLIP biến câu thành vector; selector có thể dùng thêm điểm ngữ nghĩa; fusion có thể kết hợp text với đặc trưng visual. Bộ phân loại cuối vẫn phải học từ nhãn của training set. Captioner không tự thay thế classifier và CLIP cũng không phải một giám khảo biết chắc caption đúng hay sai.

Luồng khái niệm có thể đọc thành một dòng: **dataset và split → candidate pool → caption và audit → embedding bank → selector → context được chọn → visual và text fusion → head → loss hoặc dự đoán**. Không phải cấu hình nào cũng dùng mọi khối. visual_only bỏ ảnh hưởng của text; candidate_only bỏ caption query; semantic selector là một lựa chọn riêng với việc đưa text vào fusion.

Phân đoạn khác ở đầu ra: mô hình tạo mask theo pixel thay vì một nhãn duy nhất. Candidate context vẫn có thể hữu ích, nhưng một câu về màu sắc hoặc tổn thương không tự mang tọa độ biên. Vì vậy, thành công hoặc thất bại ở phân loại không được chuyển nguyên sang kết luận về segmentation.

## 1.2 Ba tầng thay đổi đã được kiểm nghiệm

[[TABLE:map]]

Nhóm A hỏi **nên dùng text ở đâu**. Nhóm C hỏi **thay nguồn hoặc cách chọn caption có làm kết quả khác không**, với kiến trúc full_both giữ chung. Nhóm S hỏi **cách xử lý và kết hợp vector có phải nút thắt không**, nên khóa CSV Prefix và dùng candidate_only. Nếu đổi caption, kiến trúc, pool và số epoch cùng lúc, không thể biết yếu tố nào gây chênh lệch.

Các nhánh SUN397, adaptive context, top3, zero shot SigLIP và demo là phần lịch sử hoặc triển khai liên quan. Chúng cần được mô tả theo đúng mức bằng chứng. Một notebook có code đầy đủ nhưng chưa có output không phải một kết quả đã xác nhận. Một mô hình có tên “chuẩn paper” cũng cần đọc source, không thể dùng tên làm bằng chứng.

## 1.3 Bốn loại trạng thái cần phân biệt

**Đã đo.** Có output cuối, xác định được dataset, split, seed, cấu hình và metric. Mức mạnh cao hơn khi còn prediction, history và checksum. **Đã cài đặt.** Có code thực thi nhưng chưa có một run hoàn tất để đánh giá. **Prototype.** Đã minh họa một luồng nhưng còn thiếu adapter hoặc tài nguyên, như web chưa nối checkpoint. **Đề xuất.** Ý tưởng chưa triển khai, như Partial OT hoặc attention resampler trong lộ trình sau.

Khi họp nhóm, mỗi kết luận nên có đủ bốn thành phần: đang nói về nhóm run nào, so với baseline nào, thay đổi điều gì, và bằng chứng nằm ở đâu. Câu “thêm caption tăng điểm” quá rộng; câu “candidate_only trong A03 cao hơn A01 0.2153 điểm phần trăm Macro F1 trên DTD split 1 seed 42” kiểm tra được.

# 2 TACS gốc dễ hiểu từ nguyên lý đến training

## 2.1 Vì sao cần chọn ngữ cảnh theo tác vụ

Giả sử query là một ảnh bề mặt có nhiều chi tiết nhỏ. Hai candidate có thể cùng giống về màu; một ảnh giúp mô hình phân biệt cấu trúc, ảnh còn lại chỉ lặp màu nền. Truy hồi theo độ giống visual chưa biết candidate nào giúp classifier. TACS đặt việc lựa chọn dưới mục tiêu tác vụ để ưu tiên context làm mô hình xử lý query tốt hơn.

Ví dụ trên chỉ minh họa ý nghĩa của task alignment. TACS không dùng nhãn thật của query test để đi tìm context cùng lớp. Nhãn training giúp tính loss và reward khi học; lúc test, selector chỉ có những đầu vào được phép. “Selected same class rate” là số đo sau khi đối chiếu nhãn để phân tích, không phải tiêu chí đưa nhãn vào selector.

Backbone biến ảnh thành đặc trưng. Với biểu diễn query zq và candidate zc, utility visual có thể dựa trên tích vô hướng. Vì các biểu diễn và thành phần học được thay đổi trong training, utility dần phản ánh lợi ích cho nhiệm vụ. Không nên diễn giải rằng paper bắt buộc dùng một MLP riêng cho mọi phép so điểm.

## 2.2 Vì sao lựa chọn rời rạc khó học

Chọn chỉ số lớn nhất là thao tác rời rạc. Thay đổi nhỏ một score thường không đổi ảnh được chọn, nên gradient thông thường không truyền qua argmax như truyền qua phép cộng. Paper kết hợp hai nguồn tín hiệu: Gumbel Softmax straight through để có đường xấp xỉ gradient từ task loss, và policy gradient để thưởng hành động chọn context có ích.

Gumbel Softmax straight through dùng một lựa chọn cứng ở forward nhưng dùng dạng mềm để xấp xỉ backward. Policy gradient xem context được chọn là một action, rồi nhân log xác suất action với advantage. Hai cơ chế không đồng nghĩa và không thể chứng minh chúng tồn tại chỉ bằng việc trong notebook có import `F.gumbel_softmax`.

Điểm cần nhìn trong code là ảnh context được lấy như thế nào. Nếu lấy một ma trận trọng số lựa chọn rồi nhân với image bank, đường gradient qua trọng số còn có thể được giữ. Nếu lấy index rồi `index_select`, task loss không có đường đạo hàm qua index đó. Phần policy vẫn có thể học qua log probability nếu được cài đặt đúng.

## 2.3 Reward nói rằng context đã giúp được gì

[[EQ:R = L(query only) − L(query and selected context)]]

Ví dụ minh họa: loss chỉ query bằng 2.1, loss có context bằng 1.5, reward bằng +0.6. Context này có lợi theo phép đo đó. Nếu loss tăng lên 2.4 thì reward bằng −0.3. Reward không phải độ giống ảnh, không phải số từ trong caption, cũng không phải tỷ lệ cùng lớp.

Trong policy update, reward hoặc advantage thường được tách khỏi đồ thị tác vụ bằng detach để vai trò của nó là tín hiệu đánh giá action. Chuẩn hóa theo batch giúp giảm biến thiên; entropy khuyến khích khám phá thay vì chọn một ảnh cố định quá sớm. Cách scale reward, nhiệt độ và trọng số entropy đều ảnh hưởng training, nên không nên sửa tùy tiện rồi vẫn coi là cùng cấu hình.

## 2.4 Paper kết hợp ảnh ra sao

Ở classification, paper ghép token patch của query và context trước các transformer block. Context vì vậy có thể tác động đến biểu diễn query qua attention. Đây không đơn thuần là lấy trung bình hai vector CLS sau khi encode độc lập. Ở segmentation, paper dùng backbone đóng băng, gated cross attention theo block và một DPT head nhẹ.

Dự án đã triển khai nhiều biến thể và có những khác biệt quan trọng: DTD dùng L/16 với ảnh 384; ISIC visual dùng S/16 ảnh 448, cross attention ở token cuối và DPT lite; đường ablation DTD chọn index bằng categorical policy. Phải gọi đúng đó là tái triển khai và mở rộng theo tinh thần TACS, không khẳng định tất cả chi tiết đều giống paper.

[[TABLE:protocol]]

# 3 Kiến thức nền để đọc code đa phương thức

## 3.1 Tensor và embedding không phải nhãn

Tensor là mảng số nhiều chiều. Một batch ảnh thường có dạng B × 3 × H × W. Embedding là vector học được để biểu diễn nội dung. Vector không có vị trí “tên lớp” rõ ràng như cột CSV; ý nghĩa nằm trong quan hệ giữa các vector và head đã học. Một embedding 512 chiều không phải 512 từ và một mô hình 8B không có nghĩa embedding 8 tỷ chiều.

Batch size B là số mẫu xử lý cùng một lượt. K là số ảnh context chọn cho một query. Seed điều khiển một số nguồn ngẫu nhiên. DTD split là bộ file phân hoạch chính thức. K = 1 trong dự án không phải one fold cross validation; seed 42 cũng không phải split thứ 42.

Với DINOv3 L/16 ở nhánh DTD, biểu diễn visual dùng 1.024 chiều; CLIP ViT B/32 text dùng 512 chiều. Projection là lớp học được để chuyển vector text sang không gian mà nhánh visual có thể tiếp nhận. Đưa cả hai về 1.024 chiều mới chỉ bảo đảm phép toán hợp lệ, không bảo đảm các trục có cùng ngữ nghĩa.

## 3.2 Transformer Florence Qwen và CLIP khác nhau thế nào

Transformer là họ kiến trúc dùng attention, không phải tên của một mô hình duy nhất. Florence là mô hình vision language có thể sinh văn bản theo task prompt. Qwen3 VL Instruct là mô hình đa phương thức làm theo chỉ dẫn, có thể đưa ra mô tả linh hoạt hơn. Thư viện `transformers` cung cấp hạ tầng tải model và processor; đổi phiên bản thư viện không tự biến Florence thành Qwen hoặc nâng chất lượng caption.

CLIP học biểu diễn ảnh và text để so độ tương đồng trong không gian đã huấn luyện. Trong pipeline caption audit và embedding, dự án dùng CLIP để mã hóa hoặc so vector; nó không đọc caption rồi suy luận chắc chắn như một người phản biện. Cosine cao với prompt lớp có thể chỉ phản ánh đặc trưng bề mặt chung, đặc biệt khi DTD chính là tập nhãn texture.

Captioner lớn có khả năng mô tả phong phú nhưng vẫn có thể hallucinate, nói tên vật thể thay vì cấu trúc, hoặc dùng từ gần nhãn. Chất lượng diễn đạt và ích lợi cho classifier là hai tiêu chí khác nhau. Câu văn “thông minh” không tự bảo đảm feature hữu ích hơn một mô tả ngắn, đúng và ít nhiễu.

## 3.3 Freeze fine tune và cache

Freeze nghĩa là không cập nhật trọng số của thành phần đó. Fine tune nghĩa là cập nhật một phần hoặc toàn bộ trọng số theo dữ liệu mục tiêu. LoRA thêm các ma trận cập nhật hạng thấp để giảm số tham số cần học. Sinh nhiều kiểu caption bằng cùng Florence đóng băng không phải bốn lần fine tune.

Cache lưu kết quả tính toán để tránh làm lại. Text encoder đóng băng cho phép cache embedding, miễn caption, tokenizer, weights, policy và mask không đổi. Candidate visual feature bank có thể được refresh khi backbone trainable; nếu không refresh, selector sẽ chấm query mới với đặc trưng candidate cũ. Chiến lược refresh là một phần của protocol, không chỉ tối ưu tốc độ.

Tốc độ 20 phút hay 7–8 giờ còn phụ thuộc GPU, batch, độ dài decode, số beam, model, số lần retry, IO và cache. Không được suy ra caption tốt hơn hoặc chạy thiếu chỉ từ thời gian. Muốn kết luận, phải kiểm đủ số image ID, các lỗi inference, nội dung caption và metadata model thực tế.

# 4 Dataset split và candidate pool

## 4.1 Phân biệt ba vai trò dữ liệu

Training set cung cấp query và nhãn để cập nhật model. Validation set dùng chọn checkpoint và quyết định cấu hình trong phạm vi kế hoạch. Test set dành đánh giá cuối; không dùng test để chọn ngưỡng leakage, chọn CSV thắng cuộc hoặc điều chỉnh head rồi vẫn gọi là đánh giá độc lập.

Candidate pool là kho context lấy từ train. Việc lấy 20% training set làm pool không có nghĩa mặc định bỏ 20% đó khỏi query training. Trong triển khai của dự án, một ảnh có thể vừa thuộc train query vừa nằm trong pool; khi ảnh đó xuất hiện làm query, phải mask vị trí chính nó trong selector. Đây là quy tắc chống tự lấy chính ảnh, không phải loại cả lớp của query.

Nếu dữ liệu có nhiều ảnh cùng bệnh nhân hoặc những mẫu liên quan chặt, cần loại theo nhóm liên quan thay vì chỉ theo tên file. Self exclusion theo image ID không tự bảo đảm patient leakage đã được xử lý. Với mask hoặc ảnh augmentation, identity gốc cần được xác định trước khi augmentation.

[[TABLE:datasets]]

## 4.2 DTD chuẩn chính thức và giới hạn của split 1

DTD là Describable Textures Dataset, gồm 47 lớp texture, 120 ảnh mỗi lớp, tổng 5.640 ảnh. Mỗi phân hoạch chính thức có 40 ảnh mỗi lớp cho train, 40 cho validation và 40 cho test. Có 10 bộ phân hoạch. `DTD_SPLIT = 1` chọn `train1.txt`, `val1.txt`, `test1.txt`; không phải lấy ngẫu nhiên 1% hay chạy k fold trong một epoch.

Nhóm DTD chính của dự án dùng đúng file train, val và test của split 1. Pool có 376 ảnh, tức 20% của 1.880 ảnh train. Code còn kiểm số lượng, ảnh trùng giữa split, phân bố 40 ảnh mỗi lớp và đường dẫn thực. Để so một benchmark yêu cầu trung bình 10 split, phải train lại theo từng split và tổng hợp; không được coi một split là kết quả trung bình của DTD.

Việc có split chuẩn chưa đủ làm kết quả ngang hàng paper. Backbone, image size, số epoch, số seed, policy checkpoint và metric cũng phải khớp hoặc công bố rõ khác biệt. Dự án dùng Macro F1 để phân tích nhiều phần; Table 2 paper dùng Accuracy, nên phải so Accuracy với Accuracy.

## 4.3 Đi theo một query có trong pool

Ví dụ minh họa: pool có các ID P0, P1, P2, P3; query chính là P2. Selector tính một hàng bốn score. Trước khi sampling hay argmax, score của P2 bị mask thành giá trị không thể chọn, thường là −inf. Nếu chọn P1, model nhận ảnh P1 và, trong cấu hình candidate text, caption gắn với P1. Không được chọn P2 rồi chỉ giấu nhãn, vì pixel đã tự lặp query.

Khi K lớn hơn 1, cần chọn không hoàn lại và bảo đảm đủ candidate hợp lệ sau exclusion. Metric same class rate phải chia cho tổng số context thực tế đã chọn, không luôn chia cứng cho K nếu code có fallback K nhỏ hơn. Nếu mọi score đều bị mask hoặc pool rỗng, nên dừng với lỗi có thông tin thay vì nhận một index tùy ý.

Ở test, query không có trong pool train theo split chuẩn. Pool và caption bank được giữ cố định, không tự thêm từng ảnh test vào pool sau dự đoán. Ý tưởng học tiếp từ ảnh mới là một bài toán online hoặc continual learning khác, phải có protocol và đánh giá riêng.

# 5 Caption được tạo kiểm tra và biến thành vector ra sao

## 5.1 Tách sinh văn bản khỏi training classifier

Trong luồng offline, captioner đọc từng ảnh và ghi `captions.csv`. Classifier sau đó đọc CSV, kiểm image ID, audit text, mã hóa bằng CLIP rồi train. Captioner không cần chạy lại mỗi epoch. Nếu notebook tích hợp cả hai bước, điều đó chỉ giúp thao tác thuận tiện; về logic vẫn có hai giai đoạn với tài nguyên và output khác nhau.

Raw caption là chuỗi model sinh ban đầu. Final caption là chuỗi sau quy tắc xử lý, ví dụ bỏ phần mở đầu và chọn 5–8 từ. Validity là điều kiện pipeline cho phép dùng text. Một ảnh có caption không hợp lệ vẫn được dùng ở nhánh ảnh; text embedding hoặc mask bị đặt về 0. Không được xóa ảnh để làm độ phủ caption trông đẹp hơn rồi so với baseline trên tập khác.

CSV cần image ID ổn định, split, raw/final caption, validity, các cờ leakage, score audit, model và prompt version. `retry_count` hoặc tên model trong một cột không tự chứng minh model đã chạy đúng; nên đi cùng log, hash source và kiểm số lượng output. Sau mỗi batch, ghi và flush giúp resume khi Kaggle dừng.

## 5.2 Vì sao policy ban đầu loại quá nhiều

Pipeline ban đầu cắt vài từ đầu caption. Một số câu còn phần mở đầu như “The image is a close up of a”, làm đủ số từ nhưng gần như không mô tả texture. Đồng thời, threshold cosine 0.30 với các tên lớp DTD gắn cờ gần như toàn bộ caption. Kết quả có đủ 5.640 dòng nhưng tất cả text bị loại, nên classifier thực chất không có nhánh text hữu ích.

Sửa policy thành `dtd_lexical_v2` giải quyết hai vấn đề riêng: bỏ boilerplate để lấy từ mô tả thật, và giữ kiểm lexical tên lớp trực tiếp trong khi dùng score CLIP như **audit**, không tự động loại theo threshold 0.30. Đây là đổi chính sách dữ liệu, không phải chứng minh mọi caption đã hết leakage. Cần ghi policy vào output để không trộn CSV trước và sau sửa.

Ví dụ minh họa: “The image is a close up of pale stones with tiny dark gaps” có thể được xử lý thành “pale stones with tiny dark gaps”. Câu này gồm bảy từ alphanumeric và mô tả vật liệu, màu, khoảng cách. Nhưng đúng hay sai với một ảnh cụ thể vẫn cần xem ảnh; không được lấy ví dụ làm nhãn thật cho bất kỳ ID nào.

## 5.3 Leakage có nhiều mức

Lexical leakage là caption chứa trực tiếp tên lớp hoặc alias được cấm. Semantic shortcut có thể xảy ra dù không có đúng từ cấm: một mô tả quá gần định nghĩa nhãn có thể giúp đoán lớp. Trên DTD, ranh giới đặc biệt nhạy cảm vì texture chính là nội dung cần mô tả. Không thể cấm mọi dấu hiệu texture mà vẫn kỳ vọng text hữu ích.

Mục tiêu hợp lý là không đưa ground truth hoặc tên thư mục lớp vào prompt, không dùng nhãn để tạo câu cho test, công bố quy tắc từ cấm và đánh giá vai trò text bằng các đối chứng. Việc caption ảnh test bằng model đóng băng từ pixel là xử lý đầu vào, không tự động là train trên test. Tuy nhiên, xem test score nhiều lần rồi chỉnh caption policy vẫn làm yếu tính độc lập của test.

Redaction che tên lớp hoặc alias là một phép kiểm tra, không phải máy bảo đảm không leak. Unscreened là đối chứng trước lọc, không phải cách khuyến nghị để tối đa hóa test. Một threshold tốt phải có lý do từ train/val hoặc dữ liệu kiểm duyệt; không chọn threshold vì test cao rồi coi là hợp lệ.

## 5.4 Ground truth teacher và reference CSV

Caption Florence hoặc Qwen là nhãn yếu do model tạo, không tự thành ground truth. Một reference được con người xem ảnh, sửa và duyệt có thể là mục tiêu giám sát đáng tin hơn trong phạm vi tiêu chí duyệt. Nếu AI điền rồi đặt `approved`, đó vẫn là teacher annotation, không nên báo cáo như một cuộc gán nhãn độc lập bởi chuyên gia.

File `caption_references.csv` của bước fine tune có 188 ảnh, là tập mục tiêu nhỏ để xem và duyệt, khác hoàn toàn full captions.csv có 5.640 ảnh. Các cột review ID và image ID giữ identity; `caption_source`, `reviewer_id`, `teacher_model`, `review_status` ghi nguồn gốc. Không nên điền tên một model không biết chính xác hoặc để “human_corrected” khi chưa có người kiểm thật.

Quy trình nên là xem đúng pixel ứng với ID, viết câu 5–8 từ, tránh boilerplate và tên lớp, kiểm từ cấm, duyệt nội dung, rồi tách phần dùng train và kiểm chất lượng theo protocol. Không dùng cấu trúc tên thư mục để suy câu; không dùng caption cũ làm mục tiêu “chuẩn” chỉ vì cần có CSV đủ cột.

# 6 Ba vị trí đưa text vào mô hình

## 6.1 Text query trong fusion

Text query là caption của chính ảnh đang phân loại. Sau CLIP và projection, vector này đi vào fusion với biểu diễn ảnh query và visual context. Nó có thể giúp khi mô tả bổ sung đặc trưng mà backbone chưa khai thác tốt, nhưng cũng có thể làm mô hình tập trung vào vật thể, màu chung hoặc lỗi captioner.

Ở test, nếu train dùng text query thì inference cũng phải có caption query theo cùng policy, hoặc có cơ chế fallback đã kiểm nghiệm. Không thể train full_both rồi deploy chỉ ảnh và kỳ vọng hành vi giống hệt. Với ảnh ngoài dataset, pipeline phải tự sinh caption mới; không có dòng CSV trùng tên không có nghĩa lấy caption từ ảnh gần nhất tùy ý.

## 6.2 Text candidate trong fusion

Selector chọn candidate nào thì lấy ảnh và caption gắn đúng với candidate đó. Candidate text không phải một câu độc lập do selector sáng tác. Cần giữ thứ tự image bank, text bank, mask và candidate ID bank hoàn toàn khớp. Sai lệch một index có thể làm ảnh đúng nhưng text thuộc ảnh khác, tạo ra một dạng shuffle không chủ đích.

Với candidate_only, selector chỉ dùng ảnh để chọn; sau đó caption của ảnh đã chọn mới đi vào fusion. Vì pool lấy từ train và cố định, caption candidate có thể sinh và cache từ trước. Khi suy luận ảnh mới, bản này không bắt buộc sinh caption cho query, giúp giảm độ trễ so với full_both.

## 6.3 Text trong selector

Trong full_both lịch sử, selector kết hợp score visual với tương đồng giữa caption query và caption candidate. Text có thể thay đổi thứ hạng candidate trước khi fusion. Do đó, chỉ tắt candidate text ở fusion vẫn chưa làm model thành visual_only nếu selector còn dùng text.

S04 dùng một ý tưởng khác: query **image** được so với candidate **caption** trong biểu diễn đã alignment, không dùng caption query. Đây vẫn là candidate_only về nguồn text ở fusion và query text vẫn tắt. Cần phân biệt với full_both, nơi selector dựa vào caption query và caption candidate.

Nếu caption candidate không valid, candidate đó vẫn còn trong kho ảnh, nhưng không được nhận phần semantic score không đáng tin. Cơ chế fallback phải giữ một mức score trung tính sau chuẩn hóa; không đẩy mọi caption invalid xuống cuối chỉ vì biểu diễn 0 bị so với một hằng số khác.

[[TABLE:roles]]

## 6.4 Hai hiểu nhầm thường gặp

**Có dùng text không đồng nghĩa có dùng text để chọn context.** both_fusion dùng text hai phía trong fusion nhưng selector visual. **Có text selector không đồng nghĩa có caption của selector.** Selector là bộ chấm điểm; text đầu vào là caption của query hoặc candidate, không phải chú thích của một ảnh mang tên selector.

Thêm cả query và candidate text cũng không có nghĩa K tăng. K chỉ đếm số ảnh context. Bản K = 1 có thể dùng hai caption, một của query và một của candidate. Bản top3 có thể dùng query caption và ba candidate caption nếu fusion thực sự được nối; cần đọc forward, không chỉ cấu hình bank.

# 7 Theo dõi một batch và hiểu gate

## 7.1 Ví dụ tensor cho candidate_only

Ví dụ minh họa theo cấu hình DTD: batch B = 32 chứa ảnh query `32 × 3 × 384 × 384`. Candidate image bank chứa 376 ảnh. Selector tạo score `32 × 376` rồi chọn một index cho mỗi query, thành vector index `32`. Ảnh context được gather thành `32 × 3 × 384 × 384`; caption embedding tương ứng là `32 × 512`, validity mask là `32`.

Text projection chuyển 512 sang 1.024 chiều. Nhánh visual tạo biểu diễn phù hợp với head. Một dạng khái niệm của residual text là thêm phần text đã chiếu, nhân gate và mask. Source có thêm normalization và thành phần cụ thể theo từng study, nên công thức bên dưới là sơ đồ để hiểu, không phải bản thay thế nguyên hàm forward.

[[EQ:h = visual representation + gq mq Pq(tq) + gc mc Pc(tc)]]

Trong candidate_only, mq bằng 0 và nhánh query text không đóng góp. Trong visual_only, cả hai mask text bị tắt. Nhãn class chỉ dùng để tính supervised loss hoặc metric, không làm đầu vào cho projection text.

## 7.2 Gate nói điều gì và không nói điều gì

q gate điều tiết text của query; c gate điều tiết text của candidate. Gate có thể được tính theo từng sample rồi báo trung bình. Giá trị 0 trong visual_only là chủ đích, không phải lỗi cần chỉnh tay. Trong candidate_only, q gate bằng 0 cũng đúng. Muốn biết c gate bằng 0 do lỗi hay do mask, phải xem caption valid rate, mode và branch thực thi.

Gate không phải tỷ lệ phần trăm thông tin text. Ví dụ minh họa: visual có norm 1, projected text có norm 32, gate bằng 0.001. Residual text có norm khoảng 0.032 trước các tương tác tiếp theo, không phải 0.1% thông tin. Khi chuẩn hóa norm text về 1, cùng mức residual có thể đòi gate lớn hơn. Vì vậy c gate S01 khoảng 0.001 và S02 khoảng 0.062 không chứng minh S02 “sử dụng text gấp 62 lần”.

Muốn đọc gate có ý nghĩa, xem đồng thời norm trước và sau projection, residual norm so với visual norm, tỷ lệ caption valid, phân phối gate theo ảnh, thay đổi dự đoán khi tắt hoặc đảo text, và metric tác vụ. Gate có giá trị khác 0 chỉ cho thấy nhánh được mở, chưa chứng minh nhánh đang giúp.

## 7.3 Các tín hiệu loss trong training

Classification CE phạt dự đoán sai nhãn. Label smoothing làm mục tiêu mềm hơn one hot, nên CE giữa các chế độ phải cùng định nghĩa. SupCon là thành phần supervised contrastive trong một số source, làm feature có cấu trúc theo nhãn. Policy loss học hành động selector. Entropy regularization điều tiết khám phá. Alignment loss ở S03/S04 căn quan hệ giữa ảnh và text candidate.

Không cộng các loss bằng cảm tính rồi so độ lớn thô. Một loss đã chuẩn hóa theo batch, một loss theo cặp, một loss theo token có scale khác. Trọng số, detach, nơi cập nhật optimizer và số bước auxiliary đều thuộc định nghĩa phương pháp. Nếu sửa chúng, cần lưu một run identity mới thay vì resume dưới tên cũ.

## 7.4 Pseudocode để kiểm tra luồng

Đoạn dưới là minh họa, không dùng để thay nguyên cell training. Mục đích là kiểm xem index, caption và mask có đi cùng nhau không.

```text
scores = selector(query_images, candidate_banks)
scores = mask_self_candidate(scores, query_ids)
indices = sample_during_train_or_argmax_during_eval(scores)
context_images = image_bank[indices]
context_text = text_bank[indices]
context_valid = valid_bank[indices]
logits = classifier(query_images, context_images,
                    query_text, context_text, role_masks)
task_loss = supervised_loss(logits, train_labels)
policy_loss = action_log_prob_and_detached_advantage(...)
```

Một điểm kiểm code hữu ích: khi đảo caption để làm đối chứng, phải biết đang đảo trước selector hay chỉ tại fusion, giữ mask theo ảnh nào và có chọn lại context hay không. Những chi tiết đó quyết định câu hỏi khoa học của phép thử.

# 8 Nhật ký phát triển từ đầu đến hiện tại

## 8.1 Sinh caption SUN397 bằng Florence

Điểm xuất phát là bổ sung mô tả visual ngắn cho SUN397. Công cụ `tools/captioning/generate_sun397_captions.py` tải Florence local, xử lý ảnh theo batch, ghi CSV sau mỗi batch và hỗ trợ resume. Lọc lexical cùng audit CLIP được thêm để giảm nguy cơ câu mô tả nhắc thẳng tên lớp. Đây là hạ tầng tạo dữ liệu, không phải một classifier mới.

Những lỗi đầu tiên liên quan tài nguyên và môi trường: không tìm thấy weights Florence; Transformers 5 không phù hợp checkpoint remote code legacy; đường dẫn Kaggle bị gõ sai; notebook chạy ngoài thứ tự. Bài học là kiểm đúng thư mục, `config.json`, weights, processor và runtime trước khi chạy hàng chục nghìn ảnh. Đổi code không thể thay cho một model chưa được gắn vào Input.

## 8.2 Thêm text query vào M TACS

Caption query được encode bằng CLIP, chiếu sang visual dimension và đưa vào gated fusion. Giả thuyết là ngôn ngữ giúp tổng hợp các thuộc tính thị giác hữu ích cho phân loại. Source và output SUN có nhiều backbone khác nhau, nên không thể gom tất cả thành “bản ViT L/16”.

Run query text SUN U01 có output xác nhận dùng DINOv3 S/16 ảnh 384, Test Accuracy 0.761159, Macro F1 0.760189. Các số 0.8237 và 0.8108 từng được trao đổi cho nhánh khác không có artifact tương ứng được khóa trong bộ bằng chứng chính; không dùng chúng thay cho output đã kiểm. Kết quả lịch sử cần giữ nhưng không ép vào một phép so sánh đồng cấu hình chưa có.

## 8.3 Thử selector visual và semantic với top3

Prototype đầu thêm caption bank candidate và dùng tương đồng text trong selector; chọn ba context. Nó kiểm ý tưởng text có thể đổi lựa chọn context, nhưng việc có candidate text bank không tự có nghĩa classifier đã nhận candidate text trong fusion. Đường forward vẫn phải được theo đến head.

Thư mục `multimodal_top3/selector_text_only` giữ dấu vết của prototype này. Đây là nguồn gốc của hiểu nhầm top1 và top3 trong trao đổi. K tăng do thiết kế thử nghiệm, không do việc thêm một loại modality. Sau đó cần một bản riêng thực sự đưa caption candidate vào fusion.

## 8.4 Nối candidate caption vào fusion rồi trở về K bằng 1

Bản context text fusion lấy đúng caption gắn với ảnh được selector chọn và thêm nó vào đầu vào classifier. Các biến thể top3 và top1 được tách để tránh nhập nhằng. Bản K = 1 là cơ sở tự nhiên để kiểm tác động text mà không đồng thời đổi số ảnh context.

Câu hỏi thay đổi từ “text có chọn ảnh khác không” thành “sau khi đã chọn ảnh, caption của ảnh đó có cung cấp thông tin không”. Về sau câu hỏi này được tách thành candidate_only, both_fusion và full_both. Không có căn cứ mặc định full_both tốt nhất chỉ vì sử dụng nhiều đầu vào nhất.

## 8.5 Làm rõ candidate pool và suy luận ảnh mới

Quy trình được thống nhất: pool chỉ từ train; ảnh thuộc pool vẫn có thể làm training query nhưng phải loại chính nó; test không cập nhật pool. Ảnh mới ngoài dataset phải đi cùng preprocessing, class mapping, pool và policy text của model đã train.

Ý tưởng hệ thống “test xong rồi học luôn ảnh đó” được thảo luận nhưng chưa triển khai. Nó đòi nhãn thật hoặc pseudo label, quy tắc chống quên và đánh giá streaming. Nếu thêm test chuẩn vào train rồi tiếp tục báo test cũ, đó không còn là đánh giá giữ nguyên benchmark.

## 8.6 Chuyển sang DTD và tích hợp tạo CSV

DTD được chọn để kiểm texture với split chính thức và quy mô nhỏ hơn SUN397. Notebook mới đọc train1, val1, test1, lấy 20% train làm pool, tạo hoặc resume 5.640 caption rồi train. Mục tiêu tích hợp là giảm thao tác thủ công và tránh quên bước tạo CSV, không thay đổi nguyên tắc tách giai đoạn sinh text và học classifier.

Các lỗi `DTD_SPLIT` chưa định nghĩa và legacy/native checkpoint cho thấy cần Run All đúng thứ tự và preflight rõ. DTD dùng 47 texture classes nên danh sách từ cấm và đánh giá caption phải theo texture, không sao chép nguyên scene labels SUN397.

## 8.7 Xây web demo phân loại

Web có hai trang, một cho visual TACS và một cho M TACS. Giao diện cho phép chọn ảnh và dành chỗ hiển thị query caption, candidate image/caption, cấu trúc luồng và attention cho query/candidate. Những phần chưa có checkpoint được để trống thay vì tạo kết quả giả.

Trạng thái hiện tại là giao diện và API kiểm tra ảnh, chưa phải sản phẩm inference đã nối model. Tính năng segmentation web vẫn là kế hoạch. Chưa có ảnh attention thật thì không được dựng heatmap màu ngẫu nhiên rồi ghi là attention giống Figure 4 paper.

## 8.8 Thiết kế tám ablation vai trò text

Một notebook gộp tám mode làm thời gian chạy dài và dễ nhầm checkpoint, nên được tách thành tám notebook độc lập. Giữ 50 epoch thay vì 100 để có ngân sách chạy các đối chứng. Các mode lần lượt là visual_only, query_only, candidate_only, both_fusion, full_both, shuffle, unscreened và redacted.

Không yêu cầu visual_only có q gate và c gate khác 0. Hai gate bằng 0 là kết quả đúng khi text bị mask. Bảng ablation chỉ hoàn chỉnh khi từng cấu hình đã có checkpoint và test output tương ứng; không dùng `visual_only/best.pt` để chạy intervention cho full_both.

## 8.9 Sửa policy caption thành dtd lexical v2

Sau khi tạo đủ caption, log cho thấy tỷ lệ valid bằng 0. Kiểm CSV phát hiện hai nguyên nhân: score CLIP threshold 0.30 loại quá mạnh và nhiều câu cắt đầu chỉ còn boilerplate. Policy mới xử lý phần mở đầu và giữ score semantic như audit thay vì hard rejection.

Sửa này có ý nghĩa phương pháp vì nó thay phân phối text thực sự được classifier dùng. Vì vậy phải ghi policy mới, không coi việc thay ngưỡng là một sửa kỹ thuật vô hại. Caption sạch hơn về định dạng chưa tự đồng nghĩa đúng hơn với ảnh; các ablation sau mới giúp đánh giá vai trò downstream.

## 8.10 Đọc đủ tám output thay vì dự đoán kết quả

Khi đủ tám bản đã train, candidate_only nhỉnh hơn visual_only; query_only và full_both thấp hơn; shuffle cao hơn full_both đúng cặp nhưng vẫn thấp hơn visual. Những con số này làm suy yếu giả thuyết “càng nhiều text càng tốt”. Chúng cũng không cho phép quy mọi suy giảm cho leakage.

Nhóm bắt đầu phân biệt kết quả tác vụ, độ phủ caption và tính đúng của cặp ảnh–text. Đây là lý do các bước sau không chỉ đổi một model captioner lớn hơn mà còn xét normalization, alignment và control selector.

## 8.11 Tạo reference và thử fine tune decoder

Vì caption Florence chưa phải ground truth, một mẫu reference 188 ảnh được tạo để kiểm duyệt. Hướng LoRA nhắm điều chỉnh decoder từ mục tiêu mô tả texture, thay vì chỉ cắt câu. Nguồn teacher và người duyệt phải được ghi rõ; một cột approved không thay cho kiểm ảnh độc lập.

Classifier dùng CSV từ decoder LoRA đạt Macro F1 0.806707, thấp hơn full_both Florence lịch sử 0.809124. Đây là downstream metric, không phải BLEU, grounding score hay bằng chứng mô tả ngôn ngữ đã kém đi. Chưa có bộ đánh giá caption độc lập đủ để quy nguyên nhân cho fine tune.

## 8.12 Thử bốn cách chọn caption Florence đóng băng

Một hướng khác không update captioner: sinh giả thuyết rồi lấy đoạn ngắn theo Prefix, FirstValid, SingleGrounded và MultiGrounded. Các CSV được tạo để kiểm tác động extraction và reranking. FirstValid lấy span hợp lệ đầu tiên; Single/Multi dùng score ảnh–text để chọn span, khác nhau ở nguồn một hay nhiều giả thuyết.

Kết quả cho thấy hai control đơn giản đứng trên Single/Multi trong full_both. Nó gợi ý độ khớp ảnh–text mà CLIP ưa thích chưa chắc là mục tiêu phù hợp để cung cấp context cho classifier. Không nên đổi tên Single/Multi thành “ground truth tốt hơn” chỉ vì có chữ grounded.

## 8.13 Sắp xếp repository và tăng khả năng truy vết

Notebook được phân theo task, dataset, phương pháp và trạng thái; completed output tách khỏi template; helper, builder, caption CSV và report có nơi riêng. Manifest giữ quan hệ tên cũ–mới, catalog và dấu vết chuyển file. Một số thư mục là ranh giới dự kiến nên còn trống.

Lợi ích là giảm nhầm notebook có cấu hình gần giống nhau và tìm được đúng nguồn số liệu. Việc sắp xếp không đồng nghĩa mọi source đã được refactor thành thư viện production; nhiều model và selector vẫn nằm trong notebook hoặc cell helper nhúng.

## 8.14 Thử Qwen3 VL 8B và 32B

Notebook sinh hai CSV từ Qwen3 VL Instruct được thiết kế để làm đối chứng nguồn mô tả lớn hơn. Sau đó hai notebook full_both đọc CSV riêng để train classifier. Không fine tune Qwen trong các classifier run này; Qwen là nguồn text đầu vào đã tính trước.

Qwen 8B đạt Test Macro F1 0.821445 và 32B đạt 0.820788, cao hơn Florence Prefix trong full_both nhưng chưa vượt visual_only lịch sử 0.848168. 32B chưa thể được gọi tốt hơn 8B từ số liệu hiện có. Độ phủ Qwen thấp hơn các control Florence 100%, nên chất lượng và mask vẫn còn lẫn nhau.

## 8.15 Mở rộng sang segmentation ISIC

Visual ISIC S/16 ảnh 448 là mốc so sánh. Một notebook Qwen 8B tích hợp sinh caption rồi dùng text cho query, candidate và selector. Tất cả 2.750 ảnh có caption valid trong output run này, khác với coverage DTD.

Kết quả Qwen giảm Dice và IoU so với visual. Một notebook Florence Prefix ISIC được chuẩn bị để thử nguồn khác nhưng chưa có output cuối. Không được điền điểm Florence từ DTD sang ô ISIC hoặc ghi cải thiện segmentation chỉ vì nhánh classification từng nhỉnh hơn.

## 8.16 Đặt lại câu hỏi về không gian vector

Từ kết quả text không tăng ổn định, nhóm đặt câu hỏi: cộng hai vector sau projection đã phải cách kết hợp tốt chưa, liệu scale và semantic mismatch có làm mất thông tin không. Hai paper về transport alignment và omni modality retrieval được đọc để tìm hướng mới.

Các ý tưởng được chuyển thành giả thuyết có thể kiểm: kiểm norm, học quan hệ ảnh–caption, đưa phần semantic vào selector một cách có kiểm soát; token resampling, ASWP và Partial OT để sau. Không khẳng định hai paper gốc đã chứng minh lợi ích trên TACS hoặc DTD.

## 8.17 S01 khóa Prefix và dựng baseline candidate only

S01 dùng CSV Florence Prefix đã định danh và candidate_only để tránh query caption trở thành biến phụ. Cấu hình DTD L/16, 384, K = 1, 50 epoch và seed 42 được giữ làm điểm xuất phát cho chuỗi kiến trúc. Test Macro F1 bằng 0.853032.

Mục đích của bước này không phải lấy điểm cao nhất bằng mọi cách, mà tạo một baseline có dữ liệu text cụ thể để so những thay đổi vector sau đó. S01 và A03 cùng tên candidate_only nhưng không tự là cùng run; phải đọc danh tính CSV và source.

## 8.18 S02 chuẩn hóa text trước fusion

S02 chuẩn hóa norm projected text để residual dễ kiểm soát hơn. Nếu text có norm lớn còn visual đã chuẩn hóa, gate rất nhỏ có thể vẫn tạo ảnh hưởng đáng kể. S02 kiểm giả thuyết scale là nguyên nhân của kết quả text không tốt.

Gate tăng lên khoảng 0.062366 nhưng Test Macro F1 bằng 0.849329, thấp hơn S01. Điều này không làm normalization vô nghĩa; nó cho thấy sửa scale chưa đủ tạo lợi ích tác vụ trong run đã thử. Các chẩn đoán tắt/đảo text giúp xem mô hình có phụ thuộc đúng semantics không.

## 8.19 S03 học alignment ảnh và caption

S03 thêm auxiliary alignment lấy cảm hứng từ mLTM. Nó dùng ảnh candidate với caption tương ứng, balanced log Sinkhorn, mục tiêu có xử lý caption trùng và metric Mahalanobis chéo dương. Mục tiêu là khuyến khích projection text phản ánh quan hệ ảnh–caption thay vì chỉ được điều chỉnh gián tiếp qua classifier.

Alignment fit cải thiện trên training pool nhưng Test Macro F1 bằng 0.848722, chưa vượt S02 hoặc S01. Đây là ví dụ rõ của việc một auxiliary objective học được không đồng nghĩa tác vụ đích tăng. Không gọi implementation này là Partial OT hoặc một bản chép nguyên thuật toán audio–text trong paper.

## 8.20 S04 thêm semantic selector từ query image

S04 giữ nhánh S03 và thêm điểm query image–candidate caption vào selector. Query text vẫn tắt. Visual và semantic score được chuẩn hóa, chặn biên z score, trộn bằng hệ số có giới hạn, warmup rồi ramp; self candidate được mask trước thống kê.

S04 đạt Test Macro F1 0.850717, cao hơn S03 khoảng 0.1995 điểm phần trăm nhưng còn thấp hơn S01 khoảng 0.2315 điểm phần trăm. Can thiệp selector làm thay đổi context ở một phần query, song chưa có bằng chứng chắc về lợi ích F1. Đây là kết quả của notebook thứ tư trong chuỗi kiến trúc, không phải both_fusion của bộ A.

## 8.21 Điểm dừng hiện tại và các hướng chưa làm

Hiện đã có mạch tái triển khai, ablation text, thay captioner, thử segmentation và bốn bước nghiên cứu kiến trúc. Với mục tiêu bài tập lớn là xây lại cấu trúc paper và thử một số cải tiến, đây là khối lượng đủ để viết một báo cáo có phân tích, miễn công bố khác biệt và không nhận là exact reproduction đầy đủ 18 dataset.

Chưa có kết quả cho Partial OT, token resampler, ASWP, continual learning, benchmark nhiều seed/split đầy đủ hoặc web inference đã nối checkpoint. Nếu tiếp tục, nên mở từng run có hypothesis và control thay vì thêm tất cả trong một notebook rồi không xác định được nguyên nhân.

# 9 Đọc tám ablation text đúng cách

## 9.1 Kết quả chính

[[TABLE:ablations]]

A03 candidate_only cao hơn A01 0.002153 trên thang F1, tức 0.2153 điểm phần trăm. Accuracy tăng 0.002128, tương đương khoảng bốn ảnh đúng thêm trên 1.880 ảnh test. F1 không đếm số ảnh tuyến tính như Accuracy, vì nó tổng hợp precision và recall từng lớp. Chênh nhỏ như vậy cần nhiều seed và so prediction trước khi gọi là ổn định.

A02 query_only thấp hơn visual_only; A04 both_fusion và A05 full_both tiếp tục thấp. Có thể caption query hoặc cách fusion không phù hợp trong thiết lập này, nhưng chưa được phép kết luận query caption luôn có hại trên mọi dataset, encoder hoặc captioner.

## 9.2 Tại sao shuffle lại cao hơn đúng cặp

A06 đảo caption đạt 0.833392, cao hơn A05 0.809124 nhưng vẫn dưới visual_only. Một cách giải thích cần kiểm là text đúng cặp đang tạo nhiễu hoặc shortcut không bền, còn shuffle làm model giảm phụ thuộc vào nhánh text. Tuy nhiên, đây là các model được train lại với pipeline khác, không chỉ thay caption ở cùng checkpoint.

Training noise, phân phối mask, lựa chọn context và optimization đều có thể tham gia. Vì thế, kết quả shuffle không chứng minh “ngữ nghĩa text không cần thiết” hay “dữ liệu nhãn bị leak chắc chắn”. Nó là dấu hiệu để làm các intervention giữ checkpoint, query và context cố định nhằm tách nguyên nhân.

## 9.3 Unscreened và redacted trả lời câu gì

A07 unscreened dùng caption trước lọc, đạt 0.789774. A08 redacted che tên lớp/alias, đạt 0.811105. Số này không hỗ trợ giả thuyết nhắc thẳng tên lớp đang làm model đạt điểm rất cao. Nhưng không thể suy ngược rằng pipeline hoàn toàn không có shortcut, bởi nội dung và cách xử lý đã đổi.

Một confound quan trọng là coverage: full_both screened chỉ dùng được khoảng 84.63% caption query test, trong khi unscreened/redacted đạt 100%. Redacted có thể thay cả cấu trúc câu và lượng text sử dụng. Muốn tách hiệu ứng từ nhãn trực tiếp, cần cùng source caption, cùng mask, cùng từ còn lại, chỉ thay đúng đoạn bị can thiệp.

## 9.4 Phát biểu đủ thận trọng để đưa vào báo cáo

Có thể viết: “Trong ablation DTD split 1, candidate_only là cấu hình text tốt nhất của lô, nhưng lợi ích so với visual_only nhỏ. Sử dụng text ở cả hai vai trò hoặc trong selector chưa cải thiện. Shuffle và redaction cho thấy cần kiểm kỹ semantics và policy, song chưa cô lập hoàn toàn nguyên nhân.”

Không nên viết: “M TACS chứng minh text làm tăng hiệu suất”, “caption đã sạch 100%”, hoặc “full_both là bản tốt nhất”. Những câu đó không phản ánh bảng hiện có. Không xóa các cấu hình thấp khỏi bảng chỉ vì chúng trái kỳ vọng ban đầu.

# 10 Các phương pháp cải tiến nguồn caption

## 10.1 Prefix FirstValid Single và Multi thực sự làm gì

Prefix xử lý và lấy phần đầu mô tả theo control. FirstValid tìm span 5–8 từ hợp lệ đầu tiên trong giả thuyết đầu. SingleGrounded xét các span hợp lệ từ một giả thuyết và chọn theo độ khớp ảnh–text CLIP. MultiGrounded mở rộng tập span qua nhiều giả thuyết, khử trùng và chọn theo cùng tiêu chí grounding.

Trong helper hiện có, span không ghép qua ranh giới câu, có kiểm đầu/cuối câu, lặp từ, từ cấm và giới hạn số span. Class audit score được ghi để chẩn đoán nhưng không dùng làm tiêu chí chọn thắng. Khi không có span, cơ chế fallback trở về control; cần xem tỷ lệ fallback thay vì tưởng mọi ảnh đều có ứng viên tốt.

Các phương pháp này giữ Florence đóng băng. Chúng thay cách trích và chọn văn bản, không phải thay kiến trúc captioner. Độ khớp CLIP dùng trong selection không phải một đánh giá độc lập caption: nếu chọn vì score CLIP cao rồi lại dùng score đó để tuyên bố tốt hơn, phép đánh giá sẽ thiên vị tiêu chí đã tối ưu.

## 10.2 Bảng downstream và độ phủ

[[TABLE:captions]]

[[TABLE:coverage]]

Qwen 8B cao hơn Prefix 0.010794 F1, tức 1.0794 điểm phần trăm trong full_both. Qwen 32B cao hơn Prefix 1.0137 điểm phần trăm nhưng thấp hơn 8B 0.0657 điểm phần trăm. Đây là chênh trong một lô downstream; không đủ để xếp hạng tổng quát năng lực của hai model ngôn ngữ thị giác.

Prefix và FirstValid gần nhau, còn Single/Multi thấp hơn dù có 100% coverage. Như vậy, có nhiều caption hơn chưa chắc tốt hơn; reranking theo image–text similarity có thể giữ mô tả generic hoặc những tín hiệu không giúp head phân loại. Đó là giả thuyết giải thích, cần xem caption từng ảnh và prediction theo lớp để kiểm.

## 10.3 Vì sao hai CSV khác nhau cho metric khác nhau

CSV quyết định nội dung, tính valid và vector CLIP của từng ảnh. Nó có thể thay fusion trực tiếp và thay cả ảnh context nếu selector dùng text. Hai file cùng 5.640 dòng, cùng tên lớp và cùng `prompt_version` vẫn có thể khác câu, mask, thứ tự, hoặc metadata. Chỉ so tên file không bảo đảm cùng dữ liệu.

Muốn so caption content công bằng hơn, dùng chung architecture, seed, split, pool, budget và mask. Có thể chọn giao của các ảnh có caption valid ở mọi nguồn, rồi đặt cùng mask để giảm confound coverage. Cần ghi rõ kết quả matched mask là một thí nghiệm mới; không sửa bảng cũ thành số từ tập con mà quên đổi protocol.

## 10.4 Vì sao không chọn “CSV tốt nhất” bằng test rồi báo thắng

Nếu chạy nhiều CSV và chọn file có test cao nhất, test đã tham gia model selection. Bảng vẫn hữu ích để khảo sát nhưng kết luận phải mô tả tính exploratory. Một kế hoạch chặt hơn là chọn trên val, khóa pipeline rồi đánh giá một lần trên test hoặc trên split bổ sung chưa dùng.

CSV được gọi là Prefix ở S01–S04 vì dễ khóa, có provenance và là control tương đối tốt trong lô Florence, không phải vì đã được chứng minh là ground truth hay tốt nhất cho mọi kiến trúc. Việc khóa một đầu vào giúp tìm nguyên nhân tốt hơn là liên tục thay caption trong lúc sửa fusion.

# 11 Bốn study kiến trúc ảnh và text

## 11.1 S01 và S02 kiểm scale chứ chưa giải quyết semantics

S01 tạo baseline candidate_only với Prefix, projection text và gate. S02 thêm L2 normalization trước fusion để đưa magnitude về quy mô dễ so. Câu hỏi của S02 là “text có bị scale lấn át hoặc gate bù scale không”, không phải “hai vector đã cùng ngữ nghĩa chưa”.

[[TABLE:studies]]

S02 có gate lớn hơn, CE hơi thấp hơn nhưng F1 thấp hơn S01. Đây không mâu thuẫn: CE đánh giá cả xác suất trên nhãn thật, F1 dựa nhãn argmax. Mô hình có thể cải thiện xác suất ở nhiều ảnh nhưng đổi một vài argmax theo hướng bất lợi cho F1. Cần đọc curve, per class score và prediction thay vì chỉ nhìn một metric.

## 11.2 S03 học quan hệ ảnh và text bằng transport

Transport là cách phân phối khối lượng ghép giữa hai tập biểu diễn theo chi phí. S03 lấy một minibatch candidate image feature và caption embedding đã chiếu, xây cost, rồi dùng balanced Sinkhorn trong log space để tính một coupling có ràng buộc hàng/cột. Loss KL khuyến khích coupling phù hợp các cặp mong muốn.

Vì hai ảnh có thể dùng cùng câu caption, target không nên coi mọi phần tử ngoài đường chéo là âm tuyệt đối. Implementation có xử lý nhóm caption trùng để giảm false negative. Metric chéo dương kiểu Mahalanobis điều chỉnh trọng số các chiều nhưng không làm lật chiều âm; chuẩn hóa trung bình trọng số về một giúp kiểm scale.

Image anchor lấy từ bank ở đầu epoch và detach. Shared text projection được dùng trong auxiliary nhưng không chạy dropout của nhánh phụ khi cần deterministic alignment. Alignment có weight 0.05, warmup 5 epoch; batch 32; epsilon 0.1; tối đa 100 Sinkhorn iteration với tiêu chí dừng tương đối 0.005. Các giá trị này giải thích run đã thử, không phải mặc định tối ưu cho mọi bài.

Balanced OT buộc ghép hết khối lượng; nó có thể không phù hợp khi một caption thiếu hoặc sai nhiều. Partial OT cho phép chỉ vận chuyển một phần là hướng sau, chưa cài trong S03. Không gọi một loss có chữ Sinkhorn là đã thực hiện đầy đủ mọi ý tưởng robust matching của mLTM.

## 11.3 S04 nối alignment vào lựa chọn context

S04 so query visual feature với caption candidate trong biểu diễn và metric đã học, rồi trộn semantic residual vào score visual. Hệ số semantic bị giới hạn tối đa 0.25 qua sigmoid và ramp. Có năm epoch visual warmup rồi năm epoch tăng dần; z score được chặn khoảng ±3 để giảm outlier. Alpha ở checkpoint tốt nhất khoảng 0.130996, không phải confidence classifier.

Self exclusion phải diễn ra trước khi tính thống kê z score để ảnh query không làm lệch mean/std. Candidate caption invalid hoặc score gần hằng số cần fallback về visual. Khi inference, score mỗi query được tính độc lập với pool; không có OT ghép giữa các query trong một batch test. Điều này tránh việc thay batch size hoặc trộn ảnh khác làm thay nguyên lý truy hồi.

S04 tăng F1 so với S03 nhưng chưa vượt S01. Kết quả chỉ cho thấy thêm semantic selector có tác động và có hồi phục một phần trong chuỗi, không chứng minh alignment là yếu tố mang lợi ích. Muốn tách rõ cần S02 + selector không alignment hoặc capacity matched control, cùng seeds và runtime.

## 11.4 Auxiliary học được nhưng tác vụ vẫn không tăng

Chẩn đoán S03 trên 324 cặp caption valid thuộc train pool cho caption group top 1 khoảng 0.746914; S04 khoảng 0.697531. Alignment loss training cũng giảm. Đây là fit trên tập đã dùng học auxiliary, không phải held out retrieval và không phải chất lượng caption được người đánh giá.

So S03 với S02 trên test, bảng prediction có sáu trường hợp correctness được cải thiện và tám trường hợp bị hại; tổng cộng 22 nhãn dự đoán thay đổi. Kiểm McNemar cho correctness cho p khoảng 0.790527, không hỗ trợ khác biệt chắc chắn. Đây không phải kiểm định Macro F1, cũng không thay thế bất định do train nhiều seed.

S02 và S03 còn khác phiên bản môi trường đã ghi: torch 2.10 so với 2.11, numpy 2.0.2 so với 2.1.3. Điều này không chứng minh runtime là nguyên nhân, nhưng ngăn khẳng định chỉ một dòng alignment giải thích mọi chênh lệch. Chưa xác minh toàn bộ pretrained weights giống byte giữa mọi run.

# 12 Can thiệp để biết model có thực sự dùng text

## 12.1 Ablation train lại và intervention khác nhau

Ablation train lại tạo một model mới với cấu hình thay đổi. Nó đo khả năng đạt được của cả quy trình sau training, nhưng có ngẫu nhiên optimization và các tương tác khác. Intervention giữ một checkpoint rồi thay một đầu vào hoặc nhánh tại evaluation. Nó đo mức phụ thuộc cục bộ của chính model đó, không phải kết quả model sẽ đạt nếu được train lại từ đầu không có nhánh đó.

Ví dụ minh họa: cùng một checkpoint, cùng query và context, tắt text làm một prediction đổi sai thành đúng. Điều đó cho thấy text gây hại ở mẫu đó trong model này; chưa chứng minh mọi text hoặc mọi classifier đều hại. Cần tổng hợp nhiều ảnh, khoảng tin cậy và kiểm mask.

## 12.2 Fusion intervention giữ ảnh context cố định

[[TABLE:interventions]]

Ở S02–S04, tắt hoặc đảo text fusion không làm F1 giảm nhất quán; một số mode tăng nhẹ. Đây là dấu hiệu chưa có lợi ích semantics đúng cặp rõ trong F1. CE khi tắt/đảo thường cao hơn aligned, cho thấy xác suất và quyết định nhãn đang phản ánh các khía cạnh khác nhau.

Với S04, khoảng tin cậy bootstrap cho chênh F1 của text off khoảng [−0.004476, 0.005086], shuffle khoảng [−0.003611, 0.007703], đều chứa 0. Có thay đổi prediction, nhưng bằng chứng về tăng hay giảm F1 ổn định còn yếu. Bootstrap theo query chỉ phản ánh mẫu đánh giá có điều kiện trên checkpoint đã train, không tạo ra ba seed training.

## 12.3 Selector intervention phải chọn lại context

Nếu tắt semantic score ở selector, context phải được chọn lại rồi fusion dùng caption đúng của context mới. Nếu vẫn cố giữ context cũ, đó không còn kiểm tác động lựa chọn. Ngược lại, khi kiểm riêng fusion, context cần giữ cố định để tránh trộn hai cơ chế.

S04 aligned val F1 là 0.838806; selector visual only là 0.838311; selector shuffle là 0.836615. Context thay đổi khoảng 5.21% và 10.59% query tương ứng. Interval chênh F1 vẫn chạm hoặc chứa 0. Selected caption coverage cũng đổi theo context, nên đây chưa phải matched mask test hoàn hảo.

## 12.4 Các phép kiểm nên dùng khi tiếp tục

Một bộ kiểm gọn có bốn phần: giữ context và tắt text fusion; giữ context và đảo text fusion theo mask; bỏ semantic selector rồi chọn lại context; đảo riêng semantic selector rồi vẫn fusion đúng caption của context mới. Mỗi phần phải ghi rõ index trước/sau, mask, prediction và score.

Không dùng can thiệp trên test để liên tục chỉnh architecture rồi gọi test độc lập. Khi đang thiết kế, ưu tiên validation. Đến khi khóa giả thuyết và cấu hình, chạy test cuối với quyết định đã ghi trước. Các thí nghiệm hiện có có giá trị khám phá, nên nói đúng phạm vi đó.

# 13 Phân đoạn ISIC và những khác biệt cần nhớ

## 13.1 Phân đoạn không chỉ là phân loại nhiều lần

Classification nén thông tin để chọn một nhãn ảnh. Segmentation cần giữ cấu trúc không gian để xác định vùng tổn thương. Head phải khôi phục độ phân giải và loss làm việc với mask; caption toàn cục không chỉ trực tiếp pixel nào thuộc tổn thương. Một mô tả đúng về màu cũng có thể bỏ qua biên, vùng chuyển tiếp hoặc chi tiết rất nhỏ.

Run ISIC visual dùng S/16 đóng băng, ảnh 448, context 400 ảnh train và DPT lite. Nó có hai gated cross attention ở mức token cuối, không phải cùng vị trí blockwise attention như paper. Cấu hình Qwen mở rộng thêm text query/candidate/selector nhưng giữ phần lõi visual theo source đối chiếu.

## 13.2 Số liệu và cách đọc

[[TABLE:segmentation]]

Qwen giảm khoảng 0.73 điểm phần trăm Dice và 0.84 điểm phần trăm IoU. Kết quả không chứng minh caption không bao giờ dùng được trong segmentation; nó cho biết cách full_both đã thử chưa giúp. Candidate only, vùng mô tả cục bộ, hoặc text token có thể là các giả thuyết khác, nhưng chưa có số để bổ sung như kết quả.

Các metric ở đây là trung bình theo ảnh của hard mask sau threshold 0.5 ở kích thước 448, không phải soft Dice loss. Paper có protocol riêng và dùng kích thước khác. Không lấy tỷ lệ tăng so với paper để tuyên bố text tốt hơn khi bản visual khác backbone, resolution và head.

## 13.3 Caption cho ISIC không dùng policy texture DTD

Mô tả tổn thương cần quan sát màu, hình dạng, vùng và biên theo thông tin ảnh, không tự chẩn đoán bệnh từ tên file hoặc metadata. Pipeline dùng policy `isic_visible_attributes_v1`, không áp dụng nguyên danh sách 47 texture class cấm của DTD. Mục tiêu nghiên cứu không biến output thành tư vấn y khoa.

Notebook Florence Prefix ISIC đã có trong thư mục cải tiến nhưng chưa có output đo. Khi nhóm chạy sau, cần cùng split, ngưỡng mask, image size, epoch budget, pool và visual checkpoint provenance để so với I01/I02. Ghi riêng caption coverage và thời gian sinh; không trộn với thời gian train segmenter.

# 14 Những nhánh visual và thử nghiệm liên quan

## 14.1 Vì sao không gọi mọi run là ViT L16

Trong dự án có S/16, B/16 và L/16; ảnh 256, 384 và 448; pool shortlist và full pool; K bằng 1 hoặc nhiều context. Cấu hình L/16 384 thuộc mạch DTD chính, không phải nhãn chung cho toàn bộ repository. Số /16 là kích thước patch, không phải số lớp hay số context.

Ví dụ, SUN visual S/16 256 có run Test Accuracy 0.7401 và Macro F1 0.7385. SUN B/16 384 có run 0.7942 và 0.7927. Query text SUN U01 là S/16 384 với 0.761159 và 0.760189. Khác backbone và image size khiến các hàng không tạo thành ablation text thuần.

## 14.2 Các mốc tái triển khai khác

Ba run CUB no context S/16 256 seed 42, 43, 44 có Accuracy lần lượt 0.8246, 0.8227, 0.8253. Đây là một phần có lặp seed, nhưng không thể lấy độ ổn định CUB chuyển sang kết luận DTD cũng ổn định. Cars có no context S/16 256 Accuracy 0.9159; các run context khác dùng 448 và cấu hình khác.

Một số Cars visual có điểm cao nhưng còn khác shortlist, pool 100% hay train full. Một run random context 256 cho Accuracy 0.0179 là bất thường, cần điều tra chứ không lấy làm baseline thấp để làm nổi bật cải tiến. Một notebook có lỗi sau final test vẫn có thể giữ metric test, nhưng phải ghi rõ lỗi xảy ra trước hay sau phép đánh giá.

## 14.3 Adaptive context top3 và zero shot

Các notebook adaptive/diverse context, visual top3 và zero shot SigLIP thể hiện những hướng đã được thiết kế hoặc thử trong lịch sử. Chỉ đưa metric vào bảng khi có output khớp danh tính run. Code hiện diện không tự trả lời được “tăng bao nhiêu”.

Top K lớn hơn có thể cung cấp đa dạng context nhưng tăng bộ nhớ, số token và nguy cơ nhiễu; adaptive K đặt câu hỏi nên dùng bao nhiêu context theo query. Zero shot SigLIP có cơ chế scoring khác với classifier supervised. Những hướng này không được trộn vào bảng caption method nếu đồng thời đổi visual pipeline và mục tiêu học.

## 14.4 Các bảng của paper đã tái lập đến đâu

Table 2 cần các baseline classification trên cùng dataset và protocol. Table 3 cần segmentation. Table 4 cần control về context. Table 5 cần tách cơ chế gradient và policy. Dự án có bằng chứng từng phần nhưng chưa có một ma trận đầy đủ, đồng cấu hình cho tất cả ô của paper.

Bảng A, C và S của dự án là các bảng mở rộng riêng. Chúng không điền thay ô còn thiếu của Table 4 hoặc Table 5 paper chỉ vì cùng gọi ablation. Có thể trình bày một bảng đối chiếu phạm vi, đánh dấu “đã có output”, “có source”, “chưa chạy”, thay vì lấp số dự đoán hoặc số từ paper vào cột thực nghiệm nhóm.

# 15 Tìm đúng notebook CSV và checkpoint

## 15.1 Bản đồ repository

[[TABLE:folders]]

Thư mục gốc trên máy hiện tại là `F:/Documents/CODE/Python/cv_project/XLA/TACS`. Một thành viên dùng máy khác có thể đặt root khác; phần tương đối sau `TACS/` mới là mốc dùng trong danh mục này. Đường `/kaggle/input/...` chỉ tồn tại trong notebook Kaggle có gắn Input tương ứng, không phải thư mục có sẵn trên Windows.

Các thư mục `src/models`, `src/selection`, `src/evaluation` là ranh giới dự kiến, chưa phải package đã bao hết mô hình. Hiện helper thật tập trung ở `src/captioning` và `src/training`; source model, backbone adapter và selector còn nằm trong notebook hoặc cell source nhúng. Đừng import một class từ thư mục trống theo suy đoán từ tên.

## 15.2 Template để chạy và notebook để lấy kết quả

Các template text ablation nằm ở `notebooks/02_improvements/text_context_ablation/`, chia tám mode rồi đến `dtd/`. Notebook đã chạy và output A01–A08 ở `experiments/completed/text_context_ablation/dtd/`. Các template caption method ở `notebooks/02_improvements/caption_ablation/`; output C01–C06 ở `experiments/completed/caption_ablation/dtd/` theo method.

Template S01–S04 ở `notebooks/02_improvements/multimodal_architecture_ablation/`. Bốn output đã train hiện còn mang tên gốc ở root: `dtd-architecture-study-01.ipynb`, `stydu2_da_train.ipynb`, `study3_datrain.ipynb`, `study4.ipynb`. Tên root này là nguồn số liệu, không phải hướng dẫn đặt tên cho run mới. Khi bàn giao, giữ chúng cùng evidence thay vì chạy đè để có tên đẹp.

Các file xây notebook trong `tools/notebook_builders/` tái tạo cell từ source chung và template. Sửa builder có thể làm lần build sau khác source đã train; vì vậy không chạy lại builder vô thức trong lúc chỉ muốn đọc output. Lưu bản notebook đã xuất cùng hash tại thời điểm đưa lên Kaggle.

## 15.3 CSV Prefix đã khóa và các export quan trọng

CSV dùng thống nhất S01–S04 là `data/captions/dtd/grounded_controls/captions_prefix_control.csv`. SHA256 đầy đủ là `b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011`. File có 5.640 dòng, 4.749 caption valid, 324 caption valid trong pool 376. Chỉ copy lại cùng file không đổi nội dung mới giữ được danh tính này.

Artifacts S02 ở `dtd_study02_study03_results_20261026_050338/study02_norm_fusion/`; S03 ở `dtd_study03_results_20261026_050358/study03_mltm_alignment/`. Trong `runs/candidate_only/seed_42/` có best checkpoint, history, val/test metrics và prediction. Hai bộ này cho phép kiểm lại Accuracy và Macro F1 trực tiếp từ 1.880 dự đoán mỗi split.

Không nên nạp checkpoint pickle chỉ để lấy một con số đã có trong JSON. Ngoài rủi ro nguồn file, nó còn tải nhiều bộ nhớ và có thể đòi đúng code class. Khi cần inference, xác minh checkpoint do nhóm tạo, kiến trúc, class mapping, preprocessing và toàn bộ assets trước khi load.

## 15.4 Cách lần ngược một con số trong báo cáo

Ví dụ thực hành: muốn kiểm S03 Test Macro F1 0.848722. Mở bảng test của `study3_datrain.ipynb`, đối chiếu `test_metrics.json` trong export S03, rồi đọc `test_predictions.csv` và tính lại. Sau đó kiểm seed 42, split 1, CSV Prefix và checkpoint best epoch 38. Nếu notebook và JSON khác nhau, chưa sửa báo cáo ngay; trước hết xác định chúng có cùng run identity hay không.

Các báo cáo phân tích nằm trong `reports/caption_ablation/dtd`, `reports/multimodal_architecture_ablation/dtd`, `reports/paper_tables` và `reports/paper_readiness`. Evidence của báo cáo chính nằm ở `reports/bao_cao_bai_tap_lon/report_verified_evidence.json`. Evidence riêng cẩm nang nằm cạnh tài liệu này. Chúng giúp kiểm nguồn, không thay thế full output khi điều tra lỗi.

Mức truy vết không ngang nhau giữa các lô. Bốn CSV Florence control đã được kiểm hash tại máy; các nguồn Qwen trong bộ kết quả hiện có chủ yếu có binding và hash ghi trong runtime, chưa có bản CSV local để kiểm độc lập lại toàn bộ nội dung. CSV cũ của tám ablation cũng chưa được chứng minh giống byte ở tất cả run. Các giới hạn này không xóa metric đã in, nhưng cần giữ khi diễn giải nguyên nhân.

# 16 Chuẩn bị và chạy trên Kaggle offline

## 16.1 Chọn đúng loại tác vụ trước khi gắn Input

[[TABLE:assets]]

Các notebook so CSV cố định cần CSV full, không phải reference 188 dòng. Chúng không cần Florence/Qwen weights nếu không sinh lại text. Ngược lại, notebook tạo/resume captions thiếu những ID nào thì phải có captioner đúng runtime để sinh phần thiếu. Một notebook có cả bước generation không bảo đảm nó sẽ bỏ qua generation nếu đường dẫn CSV hoặc metadata không khớp.

Input DTD thường có cấu trúc `dtd/images`, `dtd/labels`, `dtd/imdb`. Notebook phải trỏ đến thư mục `dtd`, không chỉ thư mục người dùng dataset bên ngoài. Code DINO và weights DINO là hai tài nguyên khác nhau. CLIP phải có tokenizer, config và weights đầy đủ; Florence/Qwen cũng cần processor và những file của checkpoint tương ứng.

## 16.2 Checklist trước khi Run All

- Ghi run name, seed, DTD split, mode và câu hỏi của run; kiểm output directory không đè run cũ.
- Bật GPU phù hợp và kiểm dtype; đọc tên thiết bị, VRAM, phiên bản torch và transformers.
- Kiểm dataset có images và labels; số lượng split đúng; pool chỉ từ train; self exclusion hoạt động.
- Kiểm `CAPTIONS_INPUT_PATH` là file CSV tồn tại; xem header, đủ ID, đúng policy và hash mong đợi.
- Kiểm DINO repository, đúng backbone weights và CLIP assets; không thay model do tự discovery chọn nhầm.
- Kiểm 50 epoch, scheduler horizon, batch, best criterion và resume identity trước khi train.
- Đặt biến môi trường tokenizer trước lúc import hoặc sử dụng tokenizer; kiểm worker không tạo lỗi thật.
- Chạy smoke test một batch để kiểm shape, mask, finite loss và mapping candidate ảnh–caption.

Smoke test chỉ giúp bắt lỗi triển khai. Điểm của một batch hoặc vài epoch không được đưa vào bảng như final test. Khi đã đổi cấu hình sau smoke test, dọn đúng cache phụ thuộc cấu hình hoặc chọn output mới; không xóa checkpoint gốc để tiện làm lại.

## 16.3 Chạy notebook có sẵn CSV

Đầu tiên gắn CSV đã xuất lên Input, rồi copy đường dẫn đúng từ Kaggle vào biến cấu hình. Không dùng ví dụ `/kaggle/input/YOUR_ATTACHED_DATASET/captions.csv` nguyên xi. Khi gán path một thư mục DINO vào `CAPTIONS_INPUT_PATH`, `is_file()` sẽ fail; đây là sai loại tài nguyên chứ không phải thiếu pip package.

Chạy các cell theo thứ tự: preflight, config, dataset và split, staging/validation CSV, text embedding cache, dataset/loader, model/banks, training, evaluation/interventions, export. Đọc log valid caption trước khi bắt đầu 50 epoch. Nếu text mode cần caption mà valid bằng 0, dừng để kiểm policy, không tiếp tục train rồi kỳ vọng gate tự sửa dữ liệu.

Sau staging, giữ nguyên file cached trong shared output và lưu binding gồm source path/hash. Không chỉnh thủ công final caption trong cached CSV rồi dùng embedding cache cũ. Một thay đổi câu đơn lẻ cũng có thể làm embedding và model behavior khác; cần invalidate cache theo fingerprint hoặc rebuild.

## 16.4 Sinh hoặc resume CSV trong notebook

Generation phải tạo một hàng cho đúng image ID và split, kiểm không trùng và ghi sau batch. Nếu dừng sau một phần, lần sau chỉ sinh các ID chưa có, với cùng model và prompt/policy. Khi model hoặc prompt đổi, dùng output directory mới thay vì append vào CSV cũ vì cùng schema.

Sau generation, kiểm đủ 5.640 DTD ID, thống kê valid theo split và pool, cờ lexical, độ dài câu, tần suất boilerplate, duplicate caption và ví dụ ngẫu nhiên. Đủ số dòng chỉ chứng minh độ phủ record, không chứng minh mô tả đúng. Nếu có ảnh unreadable hoặc inference lỗi, log phải chỉ rõ ID để sửa hoặc chạy lại, không lặng lẽ coi hoàn tất.

## 16.5 Khi runtime Transformers không khớp

Checkpoint Microsoft Florence legacy có `auto_map` trỏ remote code như `configuration_florence2` và `AutoModelForCausalLM`. Các notebook lịch sử này dùng runtime 4.49.x để chạy checkpoint đó. Checkpoint native dạng community đi với các API mới hơn. Không đổi duy nhất chuỗi phiên bản trong guard để ép một checkpoint khác định dạng chạy.

Với Internet off, nếu Input notebook đã mang offline wheels và assets phù hợp, dùng đúng phần bootstrap/subprocess mà notebook thiết kế. Nó có thể tạo môi trường caption worker riêng để không làm hỏng runtime classifier. Chỉ trỏ thư mục chứa model không thay đổi thư viện đang import trong kernel. Cài package vào kernel thường cần restart để import phiên bản mới; subprocess phải tự in version để chứng minh đang dùng đúng môi trường.

Các mô tả tương thích trên giải thích những run đã dùng, không phải quy tắc vĩnh viễn rằng mọi Florence chỉ chạy 4.49. Khi nâng stack sau này, phải kiểm model card chính thức, định dạng checkpoint và smoke test lại; ghi environment mới như một run mới. Không cho notebook offline tự tải package từ Internet hoặc tự thay weights không thông báo.

# 17 Dừng resume và giữ checkpoint đúng cách

## 17.1 Best không phải epoch cuối

Best checkpoint là trạng thái có validation criterion tốt nhất, không mặc định epoch 50. DTD chọn min validation CE; ISIC chọn max validation hard Dice. Nếu epoch có F1 cao nhất khác epoch có CE thấp nhất, bảng final test vẫn phải dùng criterion đã định trước, không đổi sang checkpoint cho test đẹp hơn.

History ghi toàn bộ epoch để biết model học và dừng ra sao. Best checkpoint dùng đánh giá/inference; last hoặc resume checkpoint nếu được hỗ trợ cần optimizer, scheduler, epoch, scaler và trạng thái ngẫu nhiên để tiếp tục gần đúng training. Một file chỉ `model.state_dict()` thường không đủ khôi phục lịch học chính xác.

## 17.2 Đổi 100 epoch thành 50 khi đang train

Cần phân biệt một run đã lên kế hoạch 50 từ đầu với run có cosine horizon 100 rồi dừng ở epoch 50. Learning rate đi qua hai lịch khác nhau, nên không xem chúng là cùng ngân sách tối ưu hóa chỉ vì cùng số lượt dữ liệu. Nếu notebook đang train và sửa biến dưới cell, biến dùng trong scheduler đã tạo có thể không tự đổi.

Để giữ checkpoint visual_only đang có, sao lưu output nguyên trạng và ghi epoch đã hoàn tất, scheduler config cũ, best criterion và identity. Không đổi identity trong checkpoint bằng tay để vượt guard. Có thể báo đây là run 35/50 epoch theo trạng thái thực hoặc chạy cấu hình mới riêng. Nếu muốn tiếp tục đúng run, đọc logic resume có sẵn và xác minh nó tải đủ trạng thái.

Khi Kaggle bị interrupt, kiểm history và checkpoint cuối được ghi xong chưa. Không mặc định một epoch đang in progress đã hoàn tất. Nếu chỉ có best epoch 20 từ một training dừng ở 35, final model là best epoch 20; không ghi checkpoint epoch 35. Bảng ablation cần mỗi mode có output của chính mode đó.

## 17.3 Chạy intervention trước khi mọi mode hoàn tất

Cell intervention cho full_both cần `runs/full_both/seed_42/best.pt`; file visual_only/best.pt không đáp ứng. Cách sửa đúng là giới hạn danh sách vào những mode đã có đúng checkpoint và mode có nhánh text cần kiểm. Có thể xuất bảng metric tạm của những run hoàn tất, ghi ô còn lại chưa có.

Không đổi đường dẫn trong `validation_interventions` sang visual_only để né FileNotFoundError rồi giữ nhãn full_both. Việc đó làm bảng sai provenance và nhiều gate text vẫn bằng 0. Nếu model identity guard từ chối, điều tra config/source/CSV mismatch thay vì gỡ kiểm tra để chạy bằng được.

## 17.4 Gói bàn giao sau training

[[TABLE:handover]]

Nên đặt một thư mục cho một run hoặc một phương pháp với seed dưới nó. File zip bàn giao cần giữ cấu trúc tương đối, kèm manifest liệt kê đường dẫn và SHA256. Không cần nhét model captioner rất lớn vào mỗi zip nếu đã có Input được định danh; nhưng phải ghi repo/revision hoặc hash để người khác lấy đúng.

Đừng chỉ gửi ảnh chụp bảng: người tiếp quản không thể tính per class score, paired changes hay kiểm nhãn từ ảnh bảng. Ngược lại, không cần gửi hàng chục GB cache có thể tái tạo nếu đã có config và fingerprint. Ưu tiên checkpoint, CSV nguồn, prediction, history và manifest trước.

# 18 Sổ xử lý các lỗi đã gặp

## 18.1 Không tìm thấy Florence weights

Triệu chứng là `FileNotFoundError` trước generation dù DTD và CLIP đã có. Kiểm đường dẫn là thư mục có thật, bên trong có config và weight files không. Trong tình huống đã gặp, đường đúng là `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`; chữ `notebooks` và tên notebook là một phần path, không tự suy từ ảnh giao diện.

Nếu thư mục không tồn tại, sửa path hoặc gắn đúng Input trước. Không cần cài lại Transformers chỉ vì path sai. Nếu `config: False` và `weights: False`, chưa biết do thiếu file hay kiểm sai folder; phải liệt kê thư mục đúng trước khi quyết định tải lại.

## 18.2 Tên kaggle hoặc DTD SPLIT chưa định nghĩa

Gõ một đường dẫn `/kaggle/input/...` trần vào cell Python có thể được parse thành biểu thức và báo `NameError: kaggle is not defined`. Đường dẫn phải nằm trong chuỗi hoặc dùng shell command đúng cú pháp. Không đặt một dòng path trần giữa vòng đọc config.

`NameError: DTD_SPLIT is not defined` thường do cell config chưa chạy hoặc dừng vì preflight lỗi. Chạy từ đầu đúng thứ tự, giải quyết lỗi preflight trước. Không chỉ thêm `DTD_SPLIT=1` rải rác vào nhiều cell vì có thể tạo hai cấu hình không nhất quán.

## 18.3 Runtime Florence không tương thích

Nếu log nói weights có nhưng định dạng không khớp Transformers, path đã được tìm thấy; vấn đề lúc này khác lỗi thiếu weights. Kiểm `auto_map`, backend legacy/native và runtime worker. Đúng hướng là dùng môi trường tương thích với checkpoint hoặc checkpoint tương thích môi trường, không xóa điều kiện guard để lộ lỗi sâu hơn trong generate.

Khi Internet off, chỉ cài từ wheel có sẵn hoặc dùng subprocess được đóng gói phù hợp. In `transformers.__version__` ngay trong tiến trình thực sự tạo caption. Một kernel in 5.x không phủ nhận worker có thể dùng 4.49.x, nhưng phải có log worker và import path để kiểm chứng.

## 18.4 UnboundLocalError candidate labels

Trong một hàm, viết `candidate_labels_tensor = candidate_labels_tensor.to(DEVICE)` làm Python coi tên đó là biến local, nên vế phải chưa có giá trị và gây UnboundLocalError. Dùng biến device bank được tạo bên ngoài hoặc truyền tensor vào hàm, không gán lại một tên global bên trong như vậy.

Đồng thời, index và tensor được index phải ở device phù hợp. Cách sửa sạch là tạo bank label trên device cần dùng một lần, rồi gather có kiểm device. Không chuyển toàn bộ bank qua GPU trong mỗi vòng `for j` chỉ để tính metric; điều đó vừa chậm vừa che cấu trúc dữ liệu.

## 18.5 Có đủ CSV nhưng valid bằng 0

Kiểm raw/final caption trước, histogram score CLIP, cờ lexical và số từ. Với DTD đã gặp, threshold 0.30 hard rejection và boilerplate là nguyên nhân chính. Policy v2 đã được ghi riêng để semantic score là audit, không coi thay đổi đó là chỉnh ngưỡng theo test.

Nếu tất cả mask bằng 0, gate text có thể bằng 0 và training vẫn chạy, nhưng không kiểm được hypothesis text. Cần dừng từ preflight với cảnh báo rõ cho mode cần text. Không sửa c gate thành một hằng số lớn để ép dùng caption invalid.

## 18.6 Hugging Face tokenizers và fork

Cảnh báo “The current process just got forked, after parallelism has already been used” báo tokenizers tự tắt parallelism sau fork để giảm nguy cơ deadlock. Bản thân dòng cảnh báo không đồng nghĩa training thất bại. Tuy nhiên, stderr dài có thể che lỗi worker thật, nên đọc đến cuối log và kiểm epoch/test output.

Đặt biến môi trường trước khi dùng tokenizer và trước DataLoader worker. Nếu kernel đã import và tạo worker, cần tạo lại loader hoặc restart để thiết lập sạch. Một phương án chẩn đoán là `num_workers=0`; chỉ tăng worker sau khi ổn. Không đặt `true` để ép bỏ cảnh báo mà không hiểu fork.

```python
# Dat o cell dau, truoc import hoac su dung tokenizer.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

Đoạn trên là cài đặt môi trường nhỏ có thể dùng, nhưng không sửa những exception khác trong multiprocessing. Nếu worker exit, lỗi ảnh, lỗi pickle dataset hoặc bộ nhớ, cần xử lý nguyên nhân đó riêng.

## 18.7 File best pt không có và gate luôn bằng 0

FileNotFoundError ở full_both sau khi chỉ train visual_only là thiếu run tương ứng, không phải checkpoint bị đổi tên sai. Giới hạn evaluation vào mode đã chạy hoặc train tiếp mode cần có. Không tái sử dụng checkpoint khác kiến trúc bằng `strict=False` để lấp bảng.

Gate bằng 0 có thể đúng theo mode. Với visual_only, cả q/c bằng 0; candidate_only, q bằng 0; query_only, c bằng 0. Nếu gate của nhánh cần bật vẫn 0, kiểm mask, source CSV, branch, gradient và optimizer trước khi chỉnh hyperparameter.

## 18.8 CUDA OOM và lỗi không phải OOM

Với OOM thật, giảm batch, giảm số context hoặc dùng dtype phù hợp có thể giúp, nhưng K và image size là biến phương pháp nếu thay trong run so sánh. Sinh caption có thể giảm batch sau OOM rồi resume mà giữ nội dung decode deterministic, nhưng cần log effective batch và bảo đảm đủ output.

Không coi mọi exception là OOM rồi cứ giảm batch. Lỗi API, shape hoặc model compatibility không hết khi batch bằng 1. Nên để lỗi không thuộc OOM hiện ra rõ trong caption worker để tránh chạy rất lâu và bỏ qua mọi ảnh.

## 18.9 Không có uvicorn khi chạy web

Trong virtual environment của web, cài requirements của app rồi chạy uvicorn từ đúng root. Đây là dependency server, không liên quan Transformers captioner. Lệnh minh họa dưới dùng PowerShell từ thư mục TACS; cần active đúng môi trường trước.

```powershell
cd F:\Documents\CODE\Python\cv_project\XLA\TACS
python -m pip install -r apps/sun397_demo/requirements.txt
python -m uvicorn apps.sun397_demo.app:app --host 127.0.0.1 --port 8000
```

Nếu port đã dùng, chọn port khác. Server hiện cho kiểm ảnh và giao diện; chưa có classifier checkpoint nên nhấn phân loại không trả prediction thật. Lỗi 503 cho model chưa sẵn sàng là hành vi chủ đích của prototype, không phải bằng chứng classifier đã fail.

# 19 Đọc metric và làm bảng không nhầm

## 19.1 Accuracy Precision Recall và Macro F1

Accuracy là tỷ lệ dự đoán đúng trên tổng ảnh. Với một lớp, precision là tỷ lệ đúng trong những ảnh model gọi là lớp đó; recall là tỷ lệ tìm được trong các ảnh thật thuộc lớp đó. F1 là trung bình điều hòa của precision và recall; Macro F1 lấy trung bình F1 của từng lớp để các lớp có vai trò ngang nhau.

Ví dụ minh họa nhị phân theo một lớp: TP=30, FP=10, FN=20. Precision=30/40=0.75; recall=30/50=0.60; F1=60/(60+10+20)=0.6667. Không thể thay Macro F1 bằng Accuracy dù dataset có số ảnh cân bằng mỗi lớp. Một vài lớp yếu có thể làm Macro F1 giảm trong khi tổng ảnh đúng gần như không đổi.

[[EQ:F1c = 2 TPc / (2 TPc + FPc + FNc)]]

Khi tính lại từ prediction, dùng cùng tập label và xử lý zero division. DTD test có đủ 47 lớp, nên trung bình trên 47. Nếu lọc tập con để phân tích, phải giữ cách xác định class set rõ; không để thư viện tự bỏ lớp không xuất hiện rồi so con số đó với full test.

## 19.2 Điểm phần trăm khác phần trăm tương đối

A03 F1 0.850321 trừ A01 0.848168 bằng 0.002153. Đổi sang thang phần trăm, chênh là 0.2153 **điểm phần trăm**, không phải tăng 21.53%. Tăng tương đối là 0.002153/0.848168, khoảng 0.254%, một đại lượng khác và thường không cần dùng trong bảng này.

Ghi rõ cột thang 0–1 hay 0–100. Không trộn 0.85 với 85.0 trong cùng cột. Output chỉ in 4 số thập phân thì không bịa thành 6 số có độ chính xác mới; có thể thêm số 0 để định dạng nhưng cần hiểu đó không phải thông tin đo thêm. Bảng evidence giữ precision mà source cung cấp.

## 19.3 CE và F1 có thể đi ngược chiều

Ví dụ minh họa: model tăng xác suất nhãn thật từ 0.1 lên 0.3 ở nhiều ảnh nhưng vẫn dự đoán một lớp khác có 0.4. CE tốt hơn, Accuracy/F1 chưa đổi. Ngược lại, chỉ một argmax sát biên đổi lớp có thể cải thiện F1 dù confidence của nhiều ảnh khác kém hơn. Không dùng CE giảm để nói F1 cũng chắc tăng.

DTD chọn checkpoint theo min val CE, nên best epoch có thể không tối ưu val F1. Chọn lại checkpoint theo test F1 sau khi xem test là sai quy trình. Nếu muốn dùng max val F1 làm criterion mới, định nghĩa trước và train/evaluate như một protocol mới cho mọi baseline liên quan.

## 19.4 Dice IoU và trung bình theo ảnh

Dice đo hai lần vùng giao chia tổng diện tích mask dự đoán và mask thật. IoU đo giao chia hợp. Với một cặp mask, chúng có quan hệ đại số; nhưng trung bình Dice theo ảnh không thể đổi thành trung bình IoU chỉ bằng áp công thức lên số trung bình. Phân phối diện tích và lỗi khác nhau theo ảnh làm hai cách tổng hợp khác nhau.

[[EQ:Dice = 2 |P ∩ G| / (|P| + |G|)]]

[[EQ:IoU = |P ∩ G| / |P ∪ G|]]

Phải phân biệt hard threshold mask với soft loss, mean per image với tổng pixel toàn tập, đánh giá sau resize với kích thước gốc. Trường hợp mask rỗng cũng cần quy ước. Đừng so một IoU global của thư viện này với mean IoU của notebook khác mà không kiểm protocol.

## 19.5 Bất định từ ảnh và từ training

Bootstrap query trên checkpoint cố định ước lượng độ nhạy với tập ảnh đánh giá. Nhiều seed training ước lượng biến thiên do initialization, data order và optimization. Nhiều DTD split kiểm biến thiên do phân hoạch. Ba thứ này trả lời ba câu khác nhau; bootstrap một model không thay được trung bình 3 seed.

So hai model trên cùng ảnh nên dùng paired analysis, vì cùng một ảnh khó hoặc dễ tác động cả hai. Có thể thống kê ảnh improved/harmed và bootstrap chênh F1. McNemar áp cho correctness từng mẫu, không trực tiếp kiểm Macro F1. Khi nhiều cấu hình đều được xem test, còn có yếu tố multiple comparisons và lựa chọn hậu nghiệm cần thừa nhận.

## 19.6 Quy tắc lập bảng cho nhóm

Mỗi bảng cần một câu hỏi, các biến được giữ, metric, split, seed và nguồn output. Tách bảng vai trò text, bảng nguồn caption, bảng kiến trúc và bảng segmentation. Không xếp mọi F1 từ backbone khác nhau vào một ranking duy nhất rồi chọn hàng cao nhất.

Ô chưa có dùng “chưa có output”, không dùng 0.0000 vì 0 là một số đo. Số từ paper đặt cột riêng và dẫn nguồn. Không tính mean ± std nếu chỉ có một run. Chênh lệch nhỏ phải viết là quan sát của run, không gọi có ý nghĩa thống kê khi chưa kiểm.

# 20 Suy luận ảnh mới và trạng thái web demo

## 20.1 Hợp đồng inference của model phân loại

Một classifier deploy cần kiến trúc đúng, checkpoint, danh sách lớp theo đúng index, preprocessing, candidate pool, visual/text bank tương ứng và policy caption. Chỉ có file best.pt chưa đủ nếu thiếu candidate ID/order hoặc text projection weights. Không tải checkpoint full_both vào model candidate_only rồi bỏ qua các key không khớp để coi là deploy thành công.

Với candidate_only, ảnh mới được chuẩn hóa, encode visual, chọn context từ train pool, lấy text của context, fusion và phân loại. Không cần caption query. Với full_both, ảnh mới cần được caption bằng model/prompt/policy đã chọn, encode CLIP rồi dùng cả selector và fusion theo training. Nếu caption invalid, fallback phải đúng mask và đã được kiểm.

Ở evaluation dataset, caption có thể được tính trước từ pixel test. Ở ảnh người dùng mới, có thể tính online rồi cache theo content hash và captioner identity. Không lấy caption theo basename đơn lẻ vì hai ảnh khác nhau có thể cùng tên. Không dùng path chứa nhãn thật làm prompt.

## 20.2 Ảnh ngoài tập nhãn có thể không thuộc lớp nào

Classifier SUN397 chỉ có 397 nhãn của bài toán đã train. Một ảnh hoàn toàn ngoài miền vẫn bị softmax phân phối vào các nhãn đó nếu chưa cài open set detection. Điểm softmax cao không đảm bảo dự đoán đúng hoặc model hiểu ảnh. Cần hiển thị phạm vi dataset và có thể thêm kiểm uncertainty trong một thí nghiệm riêng.

Trong demo nghiên cứu, nên cho xem ảnh query, caption nếu dùng, ảnh context, caption đúng ID và top predictions thật. Hiển thị version model và policy để giải thích khác biệt. Nếu thiếu tài nguyên, trả trạng thái chưa sẵn sàng; không tạo nhãn random hoặc một top1 mặc định.

## 20.3 Web hiện có và phần còn thiếu

FastAPI phục vụ hai trang visual và M TACS, API status, inspect ảnh và classify. Backend kiểm format JPEG/PNG/WebP, kích thước file, số pixel và content hash. `POST /api/classify` hiện trả 503 khi model chưa được nối; prediction và evidence còn trống. Trang không phải bằng chứng một model đã chạy inference.

Để hoàn thiện, cần adapter tải đúng checkpoint của mỗi trang, build/load pool và caption bank, xử lý caption query nếu cần, response schema chứa context ID/caption, confidence và attention thật. Cần test ảnh không hợp lệ, thiếu model, caption fail, pool fail, CPU/GPU mismatch và thứ tự class. Segmentation web để một giai đoạn riêng.

## 20.4 Attention như Figure 4 không tự sinh từ checkpoint

Attention hoặc saliency phải được trích từ một cơ chế xác định: block nào, head nào, CLS/patch hay cross attention, cách tổng hợp, resize overlay và query/candidate mapping. Hai heatmap khác cơ chế không nên cùng ghi “mô hình chú ý” mà không chú thích. Attention chỉ là một công cụ giải thích, không tự là chứng cứ nhân quả hoàn chỉnh.

Không có checkpoint và adapter tương ứng thì để vùng heatmap trống là đúng. Khi làm sau, xuất ảnh gốc, raw map, normalization và caption của hình. Tránh chọn chỉ ví dụ đẹp mà giấu lỗi; nên có cả đúng, sai và trường hợp caption làm prediction đổi để nhóm học được điều gì đang diễn ra.

# 21 Sửa báo cáo và thuyết trình có căn cứ

## 21.1 Cấu trúc nội dung nên được giữ

Báo cáo chính có vai trò trình bày bài toán, nền tảng TACS, thiết lập, phương pháp mở rộng, thực nghiệm, phân tích và kết luận. Cẩm nang này bổ sung cách học và vận hành. Không cần bê toàn bộ nhật ký lỗi vào phần kết quả, nhưng những lỗi dẫn đến thay đổi policy hoặc protocol phải được mô tả ở thiết lập/phương pháp.

Mạch kể hợp lý là: tái triển khai TACS trên một số dataset → thêm text → kiểm vai trò text → thử nguồn caption → nhận ra fusion/selector là một nút thắt → dựng baseline kiểm soát và các study → đánh giá giới hạn. Mạch này giải thích vì sao đã thử nhiều phương án mà không cần giả vờ mỗi bước đều tăng điểm.

## 21.2 Những câu có thể dùng và những câu phải sửa

Có thể viết “Chúng tôi sử dụng L/16 và ảnh 384 cho các thí nghiệm DTD, khác S/16 và ảnh 256 trong paper”. Cần sửa câu “tái lập y hệt TACS” nếu đường DTD không có Gumbel ST task gradient thực thi. Có thể viết “lấy cảm hứng từ transport alignment”; không nhận đã triển khai Partial OT khi chỉ có balanced Sinkhorn.

Có thể viết “candidate_only nhỉnh hơn visual_only trong lô A, một seed”. Không viết “text luôn cải thiện”. Có thể viết “Qwen tốt hơn các Florence control trong full_both đã thử”. Không viết “Qwen là ground truth tốt nhất” hoặc “32B chắc tốt hơn 8B”. Có thể viết “S04 hồi phục so với S03”; cần nói thêm S01 vẫn cao hơn.

Có thể viết “captioner chỉ nhận pixel, không nhận ground truth của ảnh”. Không viết “đã chứng minh không leak nhãn” chỉ từ lexical filter. Có thể viết “web đã có giao diện và API readiness”; không viết “hệ thống inference hoàn chỉnh” trước khi nối checkpoint và kiểm thực tế.

## 21.3 Quy trình cập nhật một bảng

Bước 1 xác định câu hỏi và baseline của bảng. Bước 2 mở notebook output thật, tìm final val/test và config. Bước 3 đối chiếu file metric/prediction nếu có; không lấy dòng epoch tốt nhất trên test. Bước 4 kiểm source CSV, hash, policy, seed, split, pool, backbone và kích thước. Bước 5 cập nhật bảng và nhận xét cùng lúc, ghi thiếu artifact nếu có.

Bước 6 tính delta bằng precision đủ và chỉ làm tròn ở lúc trình bày. Bước 7 kiểm đơn vị phần trăm, đpt, Dice hay F1. Bước 8 lưu bản report mới cùng nguồn, không sửa output notebook gốc. Bước 9 nhờ thành viên khác lần lại một con số. Bước 10 render Word/PDF và kiểm bảng, heading, hình, reference sau mọi chỉnh lớn.

File báo cáo chính nằm ở `reports/bao_cao_bai_tap_lon/BAO_CAO_TACS_VA_MO_RONG_NGU_CANH_ANH_VAN_BAN.docx`. Nguồn nội dung và evidence nằm cạnh nó. Nếu sửa Word trực tiếp, builder từ Markdown sẽ không biết các sửa đó; cần thống nhất một nguồn chính để tránh lần build sau ghi đè bản đã biên tập. Không chạy builder báo cáo trong lúc chưa sao lưu bản nhóm đã sửa.

## 21.4 Bổ sung hình như thế nào cho đúng

Hình kiến trúc phải phân biệt phần paper và phần nhóm thêm; ghi backbone, K và vị trí text. Hình training curve lấy history tương ứng best checkpoint. Hình caption so sánh phải cùng ảnh với ID được ẩn nhãn thư mục nếu cần tránh dẫn dắt người đọc. Hình segmentation phải có ảnh, ground truth mask, prediction và metric/caption nguồn tương ứng.

Heatmap chỉ dùng map thật từ checkpoint. Hình minh họa tự vẽ phải ghi là sơ đồ hoặc ví dụ, không trình bày như output model. Hình trích paper cần dẫn nguồn và giữ bối cảnh. Không dùng biểu đồ thiếu baseline hoặc trục bị cắt để phóng đại chênh 0.2 điểm phần trăm.

## 21.5 Phân vai trong nhóm

Một người giữ protocol và cấu hình; một người kiểm dữ liệu/caption; một người kiểm model/training; một người đối chiếu metric và viết bảng; người trình bày cần hiểu ít nhất inference và kết luận chính. Có thể một người làm nhiều vai nhưng không để mọi chỉnh sửa thiếu người kiểm chéo.

Khi họp, dùng một trang run card: mục tiêu, input identity, mode, config khác baseline, output path, best criterion, metric, kết luận và giới hạn. Không chỉ gửi “bản mới cao hơn”. Đặc biệt, người sửa báo cáo phải biết không dùng A03 làm baseline trực tiếp của S04 mà bỏ S01 vì S01 là baseline cùng chuỗi.

## 21.6 Khi giảng viên hỏi đã làm đúng và đủ chưa

Với yêu cầu xây lại cấu trúc paper trên một số dataset và thử cải tiến, nhóm đã có nhiều thành phần đúng hướng: selector context, visual fusion, phân loại/phân đoạn, hạ tầng offline, ablation text, thay nguồn caption và nghiên cứu alignment/selector. Giá trị học thuật nằm ở câu hỏi được kiểm, code có truy vết và phân tích cả kết quả âm.

Điểm phải tự nhận là khác protocol gốc, chưa đầy đủ bảng paper, nhiều nhánh chỉ một seed/split, và caption text chưa có đánh giá chất lượng độc lập. Dừng thêm kiến trúc để hoàn thiện báo cáo có thể hợp lý hơn chạy tiếp không có control. Nếu còn ngân sách, ưu tiên làm mạnh bằng chứng của một kết luận nhỏ hơn là mở thêm nhiều biến cùng lúc.

# 22 Lộ trình tiếp tục và các bài tập cho nhóm

## 22.1 Ưu tiên nếu chỉ còn ít thời gian

Đầu tiên, khóa một bộ split/pool/CSV và tập trung S01 hoặc candidate_only cùng baseline visual đồng cấu hình. Thứ hai, chạy thêm seed nếu tài nguyên cho phép để biết lợi ích 0.2 đpt có bền không. Thứ ba, hoàn thiện prediction và artifact thay vì chỉ thêm notebook. Thứ tư, chốt báo cáo với giới hạn rõ và demo đúng trạng thái.

Đây là ưu tiên về độ tin cậy, không phải cam kết sẽ tăng score. Nếu nhiều seed cho kết quả text không giúp, đó vẫn là kết luận quan trọng: tín hiệu caption và fusion đã thử chưa cải thiện trong phạm vi được đo. Một báo cáo cẩn thận không cần luôn có một hàng thắng paper.

## 22.2 Các hướng mô hình chưa được xác nhận

**Matched masks và caption quality review.** Giữ cùng mask để tách content khỏi coverage; đánh giá một mẫu caption độc lập về grounding, thuộc tính và leakage. Đây là kiểm dữ liệu trước khi thay model.

**Partial transport.** Chỉ ghép phần đáng tin giữa ảnh và text, có thể giảm tác hại của cặp nhiễu. Cần control balanced OT và kiểm held out; không nhận lợi ích robust từ paper audio–text mà chưa đo ở DTD.

**Token level fusion hoặc attention resampler.** Giữ nhiều token text/visual hơn một vector tổng hợp, cho phép truy cập chi tiết theo query. Cần giới hạn số token, giữ budget/capacity control và kiểm ảnh hưởng độ trễ.

**Attention Sliced Wasserstein Pooling.** Ý tưởng từ OmniRet để tổng hợp biểu diễn giàu chi tiết; chưa có code hoặc metric trong chuỗi hiện tại. Không dùng tên này để đổi nhãn một phép mean pooling.

**Fine tune captioner với supervision tốt hơn.** Mở rộng reference có kiểm ảnh, phân chia dữ liệu caption task rõ, đánh giá độc lập trước khi chạy downstream. Không dùng nhãn test classifier để tạo mục tiêu caption.

**Continual learning.** Chỉ triển khai như một protocol streaming riêng, có chống quên, nhãn/pseudo label, xử lý drift và tập đánh giá giữ kín. Không tự học từng ảnh test benchmark rồi báo lại trên cùng ảnh.

## 22.3 Bài tập một theo dõi image ID

Chọn một ID thuộc candidate pool, tìm đúng ảnh, dòng CSV, validity, embedding index và vị trí trong bank. Giả sử nó là query training, chỉ ra vị trí bị mask. Sau khi chọn một context khác, kiểm ảnh và caption có cùng ID. Không cần train lại để làm bài này.

Đáp án cần có một chuỗi ánh xạ rõ, không chỉ screenshot ảnh. Nếu thứ tự CSV khác candidate_df, phải chứng minh lookup dùng ID rồi mới stack theo candidate_df. Đây là nơi phát hiện lỗi ghép caption phổ biến nhất.

## 22.4 Bài tập hai đọc một kết quả trái kỳ vọng

Từ bảng S, giải thích vì sao S02 c gate tăng nhưng F1 không tăng. Đáp án tốt nhắc normalization thay scale, gate không là phần trăm semantics, CE và F1 khác mục tiêu, và một seed chưa đủ cho kết luận tổng quát. Đáp án “gate lớn hơn nên text mạnh hơn và model tốt hơn” là chưa đúng.

Từ bảng A, giải thích shuffle cao hơn full_both mà không khẳng định leakage. Đáp án cần phân biệt train lại với intervention và chỉ ra mask/optimization/context selection có thể cùng đổi. Đề xuất kiểm tiếp nên giữ checkpoint và kiểm từng nhánh.

## 22.5 Bài tập ba kiểm một bảng trước khi nộp

Chọn một bảng, kiểm ngẫu nhiên ba hàng qua notebook output, rồi kiểm baseline, đơn vị, nguồn và kết luận. Nếu có 0.850321 so 0.848168, tính delta đúng 0.2153 đpt. Nếu bảng dùng mean ± std, yêu cầu tìm đủ run độc lập tạo ra std; không lấy bootstrap interval đổi tên thành std qua seed.

Kết quả bàn giao là ghi chú kiểm gồm ID, path, metric gốc, metric đã làm tròn và câu kết luận sửa nếu cần. Không chỉnh notebook evidence để khớp một con số viết nhầm trong Word.

# Phụ lục A Hỏi đáp nhanh để học và bảo vệ

## A1 Các câu về dữ liệu và phương pháp

**TACS khác nearest neighbor ở đâu.** TACS học selection theo lợi ích task; độ giống là công cụ chấm điểm, không phải mục tiêu cuối duy nhất. Random, frozen feature retrieval và feature averaging là các baseline khác cơ chế.

**Ảnh pool có làm query train không.** Có thể có trong triển khai; khi trùng ID phải loại chính ảnh đó, và loại nhóm liên quan nếu protocol yêu cầu. Không cần loại mọi ảnh cùng lớp.

**K bằng 1 là k fold không.** Không. Nó là số context cho mỗi query. DTD split 1 là một phân hoạch chính thức; training seed 42 là nguồn ngẫu nhiên của run.

**Visual only có context không.** Có trong bộ ablation A. Nó chỉ không dùng text. No context là một baseline khác và không được đổi tên hai cái cho nhau.

**Có ba text caption cho selector không.** Không có một caption riêng của selector. Text selector là thông tin từ caption query/candidate được dùng để chấm điểm context; S04 dùng image query đối chiếu caption candidate.

**Caption test có vi phạm không.** Sinh từ pixel bằng captioner đóng băng và không dùng nhãn test là xử lý đầu vào. Dùng test để chọn policy, fine tune hoặc cập nhật model lại là vấn đề đánh giá khác.

**Đưa về cùng 1.024 chiều có cùng semantic space chưa.** Chưa. Cùng shape chỉ cho phép tính toán; alignment là một mục tiêu học bổ sung và vẫn cần kiểm task performance.

**CLIP threshold bảo đảm không leak không.** Không. Nó là heuristic vector similarity. Lọc lexical, audit, redaction và đối chứng hỗ trợ đánh giá rủi ro chứ không chứng minh tuyệt đối.

**Gate 0 có phải hỏng training không.** Không nếu mode mask nhánh đó. Nếu nhánh được yêu cầu bật thì kiểm valid mask và gradient. Không chỉnh gate tay chỉ vì thấy 0.

**Dùng text candidate có cần sinh caption query mới không.** Candidate_only không cần; full_both có. Phải theo đúng train/inference contract.

## A2 Các câu về kết quả

**Text có tác động không.** Có thể quan sát prediction, context và metric thay đổi. Tác động khác với lợi ích; có thay đổi xấu và tốt.

**Cấu hình nào tốt nhất trong tám A.** Candidate_only có F1 cao nhất lô A; lợi ích so visual_only nhỏ, một seed. Không chuyển kết luận đó thành full_both tốt nhất.

**Nguồn caption nào tốt nhất trong C01–C06.** Qwen 8B cao nhất trong full_both của lô đó. Không đồng nghĩa tốt nhất ở candidate_only hoặc tốt hơn baseline visual mọi cấu hình.

**Florence nào dùng trong S.** Prefix đã khóa bằng hash. Nó là control để giữ dữ liệu cố định, không phải ground truth.

**Vì sao dùng 32B không thắng 8B.** Model lớn hơn chưa chắc tạo text phù hợp task hơn; mask, style, decoding và tương tác classifier đều ảnh hưởng. Một chênh nhỏ một seed không đủ xếp hạng model tổng quát.

**S04 có cải tiến không.** Về thiết kế có semantic selector mới; về kết quả cao hơn S03 nhưng thấp hơn baseline S01. Báo cáo cả hai vế.

**Alignment loss giảm có thành công không.** Nó cho thấy fit mục tiêu phụ trong phạm vi đo. Cần held out retrieval và task metric để nói thông tin hữu ích được cải thiện.

**Segmentation có tăng không.** Bản Qwen 8B full_both đã chạy giảm so visual ở cả Dice và IoU. Bản Florence chưa có output để kết luận.

**Có thể lấy số từng hàng tốt nhất từ nhiều protocol không.** Có thể liệt kê như lịch sử nếu ghi rõ, nhưng không tạo thành ablation công bằng. Muốn kết luận nhân quả phải có baseline đồng cấu hình.

**Bảng hiện tại dùng trong bài tập lớn hợp lý không.** Hợp lý khi gọi đúng là khảo sát và mở rộng có giới hạn. Không trình bày nó như benchmark đầy đủ paper hoặc kết quả nhiều seed khi chưa có.

## A3 Các câu về vận hành và báo cáo

**CAPTIONS INPUT PATH là reference hay full.** Với classifier so caption methods là full CSV 5.640 ảnh. Reference 188 ảnh dành giám sát/kiểm duyệt captioner, không đủ thay full data.

**Chỉ có best pt đủ bàn giao không.** Không. Cần config, class mapping, pool, preprocessing, CSV/bank identity, source và prediction để tái lập/kiểm số.

**Có thể train 35 epoch rồi chạy ablation luôn không.** Có thể đánh giá các run đã hoàn tất và ghi đúng 35 epoch hoặc best thực tế, nhưng không giả có checkpoint những mode chưa train. Giữ lịch scheduler và identity nguyên trạng.

**Cảnh báo tokenizers fork có làm sai metric không.** Bản thân cảnh báo không chứng minh metric sai. Nó báo điều chỉnh parallelism; cần đọc lỗi worker thật và xác nhận run hoàn tất.

**Web đã dự đoán được chưa.** Source hiện tại chưa nối checkpoint/pool adapter. Giao diện và inspect hoạt động nhưng classify trả chưa sẵn sàng.

**Có thể thêm ảnh test vào pool sau dự đoán không.** Không trong protocol benchmark đang dùng. Đó là một thử nghiệm online riêng nếu muốn làm.

**Nhóm có cần làm lại đủ 18 dataset không.** Theo mục tiêu bài tập lớn đã trao đổi, có thể chọn một số dataset. Cần nói phạm vi, không khẳng định tái lập toàn bài khi chưa có đủ bảng.

**Có nên giấu kết quả âm không.** Không. Nó giải thích vì sao cần ablation và giới hạn phương pháp. Một kết luận nhỏ đúng đáng tin hơn lời hứa cải thiện không được số liệu hỗ trợ.

# Phụ lục B Từ điển thuật ngữ và cách đọc

**Query.** Ảnh đang cần dự đoán. **Candidate.** Một ảnh có thể được chọn làm context. **Candidate pool.** Kho ảnh train cố định dùng để lựa chọn. **Context.** Candidate thực sự được chọn cho query cụ thể, không phải toàn bộ pool.

**Backbone.** Mạng trích feature ảnh. **Head.** Thành phần đổi feature thành logits hoặc mask. **Patch.** Mảnh ảnh đầu vào ViT. **CLS token.** Token tổng hợp biểu diễn trong một số kiến trúc. **DPT.** Head khôi phục dự đoán dày đặc từ feature transformer.

**Captioner hoặc VLM.** Mô hình nhận ảnh và sinh mô tả. **Processor.** Thành phần chuẩn bị ảnh/text theo yêu cầu model. **Tokenizer.** Chuyển văn bản thành token. **CLIP.** Mô hình biểu diễn ảnh và text được tiền huấn luyện tương phản.

**Embedding.** Vector biểu diễn nội dung. **Projection.** Biến đổi học được sang dimension/biểu diễn khác. **Normalization.** Điều chỉnh scale theo quy tắc; không tự là semantic alignment. **Fusion.** Kết hợp thông tin các nhánh. **Gate.** Hệ số điều tiết phần thông tin đi vào fusion.

**Selector.** Thành phần chấm và chọn context. **Categorical policy.** Phân phối xác suất trên candidate. **Argmax.** Chọn phần tử lớn nhất. **Gumbel Softmax ST.** Ước lượng cho lựa chọn rời rạc, hard forward và soft backward. **Entropy.** Độ phân tán phân phối, dùng trong exploration.

**Reward.** Tín hiệu đánh giá ích lợi action. **Advantage.** Reward đã trừ baseline hoặc chuẩn hóa để học policy ổn định hơn. **Detach.** Tách một tensor khỏi đường gradient. **Gradient clipping.** Giới hạn norm gradient để giảm bước cập nhật quá lớn.

**CE.** Cross entropy. **SupCon.** Supervised contrastive objective. **Label smoothing.** Làm mục tiêu class mềm hơn one hot. **Weight decay.** Regularization áp khi tối ưu. **Warmup.** Giai đoạn tăng dần một learning rate hoặc tác động auxiliary.

**OT.** Optimal transport, ghép khối lượng theo cost. **Sinkhorn.** Thuật toán tính coupling có regularization và ràng buộc phân phối. **KL.** Độ lệch giữa hai phân phối theo hướng xác định. **Mahalanobis metric.** Khoảng cách có trọng số hoặc ma trận đo. **Partial OT.** Chỉ ghép một phần khối lượng; chưa triển khai trong chuỗi S.

**Ablation.** Bỏ hoặc đổi thành phần để kiểm vai trò. **Intervention.** Can thiệp vào model đã cố định. **Control.** Đối chứng giúp tách tác động. **Confound.** Yếu tố đổi cùng biến đang xét nên làm nguyên nhân không rõ. **Matched mask.** Cùng điều kiện text valid ở các phương án.

**Checkpoint.** Trạng thái model và có thể kèm optimizer/scheduler. **Resume.** Tiếp tục trạng thái training. **Cache.** Kết quả trung gian được lưu. **Fingerprint hoặc SHA256.** Định danh nội dung file/cấu hình để phát hiện thay đổi. **Provenance.** Nguồn gốc và quá trình tạo một kết quả.

**Macro F1.** Trung bình F1 theo lớp. **Dice và IoU.** Chỉ số khớp vùng phân đoạn. **CI.** Khoảng tin cậy theo cách ước lượng đã chọn. **Seed.** Nguồn kiểm ngẫu nhiên. **Split.** Phân hoạch dữ liệu. **đpt.** Điểm phần trăm.

**Screened.** Đã qua quy tắc lọc của pipeline. **Unscreened.** Dùng text trước lọc. **Redacted.** Che phần từ/cụm bị quy định. **Grounded.** Có cơ chế gắn với ảnh; không tự là nhãn đúng. **Teacher annotation.** Mục tiêu do model teacher tạo. **Ground truth.** Mục tiêu tham chiếu theo quy trình gán nhãn được xác định, không phải mọi output AI.

# Phụ lục C Danh mục nguồn thực nghiệm đã đối chiếu

Danh mục dưới dùng đường dẫn tương đối từ TACS. Mã A01–A08, C01–C08, S01–S04, I01–I02, V và U01 cho phép tìm lại nguồn con số trong tài liệu. Có 32 mục evidence nhưng 31 notebook duy nhất, vì A05 và C07 trỏ cùng run. Chúng không phải 32 lần lặp độc lập. Checksum ở đây rút ngắn để dễ đọc; chuỗi đầy đủ trong `handbook_evidence.json` và evidence báo cáo chính.

[[SOURCES]]

Ngoài notebook, S02/S03 còn được đối chiếu JSON và prediction CSV trong hai thư mục export đã nêu ở chương 15. Các helper audit ở `tools/project_management/` kiểm output mà không train lại. Không xem dữ liệu cache hoặc file checkpoint chưa rõ nguồn chỉ vì tên giống nhau.

# Phụ lục D Tài liệu đọc và checklist bàn giao

## D1 Tài liệu gốc nên đọc

[1] Jingyu Guo và cộng sự. Learning What Helps Task Aligned Context Selection for Vision Tasks. CVPR 2026. Bản PDF trong `docs/paper/`; trang công bố https://arxiv.org/abs/2512.00489. Đọc phần selector, reward và các bảng trước khi mô tả là tái lập nguyên bản.

[2] DINOv3 repository chính thức. https://github.com/facebookresearch/dinov3. Dùng để kiểm tên backbone, weights và cách lấy feature. Cần khóa revision/weights theo run thay vì chỉ ghi “DINO”.

[3] Alec Radford và cộng sự. Learning Transferable Visual Models From Natural Language Supervision. https://arxiv.org/abs/2103.00020. Nền tảng để hiểu CLIP embedding và giới hạn suy diễn từ cosine score.

[4] Florence 2 model card. https://huggingface.co/microsoft/Florence-2-base. Đọc định dạng checkpoint, processor, task prompt và hướng dẫn runtime tương ứng; không đồng nhất tên thư viện với model.

[5] Qwen3 VL 8B Instruct và 32B Instruct model cards. https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct và https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct. Dùng khi chuẩn bị đúng model/processor và prompt; bản classifier hiện có đọc CSV đã xuất.

[6] Describable Textures Dataset. https://www.robots.ox.ac.uk/~vgg/data/dtd/. Nguồn chuẩn về 47 lớp, 5.640 ảnh và 10 phân hoạch. Tên folder trong dataset là nhãn, không dùng làm đầu vào captioner.

[7] Manh Luong và cộng sự. Revisiting Deep Audio Text Retrieval Through the Lens of Transportation. https://arxiv.org/abs/2405.10084. Nguồn ý tưởng alignment/ground metric; domain gốc audio text, không phải chứng minh sẵn cho ảnh DTD.

[8] Chuong Huynh, Manh Luong và Abhinav Shrivastava. Efficient and High Fidelity Omni Modality Retrieval. https://arxiv.org/abs/2603.02098. Nguồn ý tưởng token resampling và ASWP; các phần đó chưa được đo trong dự án.

[9] Edward J Hu và cộng sự. LoRA Low Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685. Dùng để giải thích adapter low rank, không nhầm một lần extraction caption với fine tune.

[10] ISIC 2017 challenge. https://challenge.isic-archive.com/landing/2017/. Nguồn bối cảnh benchmark phân đoạn; quy tắc threshold, resize và mean theo ảnh của run phải kiểm source dự án riêng.

## D2 Kiểm trước khi giao cho thành viên mới

- Mở được root TACS và tìm đúng template, completed notebook, CSV và evidence theo danh mục.
- Giải thích được visual_only vẫn có ảnh context; K không phải fold; candidate text đi với context đã chọn.
- Đọc được một hàng bảng A, một hàng bảng C và một hàng bảng S mà không trộn baseline.
- Nói rõ DTD L/16 384, ISIC S/16 448 và các khác biệt với paper; không gán chung backbone cho toàn dự án.
- Xác định được CSV full 5.640 khác reference 188; caption AI không tự là human ground truth.
- Kiểm được một image ID từ split đến bank và self mask; không dùng nhãn test để chọn context.
- Tìm được best criterion, best epoch, số epoch thực và scheduler của run; không đổi identity để vượt guard.
- Có history, prediction, metrics, checkpoint phù hợp, config, environment và hash nguồn khi bàn giao.
- Báo cáo cả kết quả chưa tăng, confound coverage, giới hạn seed/split và phần chưa triển khai.
- Web không dùng prediction hoặc heatmap giả; mô tả đúng phần frontend/backend đã có và adapter còn thiếu.

## D3 Kết luận để cả nhóm thống nhất

Dự án không chỉ là thêm một câu vào ảnh. Nó đã hình thành một pipeline lựa chọn context có task supervision, một hệ thống tạo và kiểm text, các phép tách vai trò text, đối chứng nguồn caption và một chuỗi kiểm tra scale, alignment, semantic selection. Những bước đó tạo ra kiến thức về nơi text có thể hữu ích và nơi thiết kế hiện tại chưa tận dụng được nó.

Kết luận hiện tại cần giữ ở mức bằng chứng: text candidate có một lợi ích nhỏ trong lô ablation ban đầu; Qwen thay đổi downstream score trong full_both; các bước kiến trúc chưa vượt baseline Prefix candidate_only; Qwen segmentation chưa cải thiện visual. Bước tiếp theo có thể là làm mạnh protocol hoặc chốt báo cáo, không bắt buộc phải chạy thêm cho đến khi có một số đẹp.

Người tiếp quản làm đúng khi giữ được quan hệ giữa dữ liệu, code, model và con số, giải thích được cả kết quả thuận và trái giả thuyết, rồi chỉ thay một nhóm biến có kiểm soát. Đó là nền tảng để nhóm sửa báo cáo, bảo vệ bài tập lớn và tiếp tục nghiên cứu mà không đánh mất những gì đã thực sự làm được.
