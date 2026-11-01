# TÓM TẮT

Báo cáo trình bày quá trình tái triển khai có điều chỉnh phương pháp Task Aligned Context Selection (TACS) và khảo sát mở rộng ngữ cảnh ảnh bằng mô tả văn bản. Mục tiêu của bài tập lớn là hiểu và xây dựng các thành phần cốt lõi của paper, vận hành trên một số bộ dữ liệu, sau đó đề xuất và kiểm nghiệm các thay đổi có cơ sở. Phạm vi triển khai gồm phân loại ảnh tự nhiên, trọng tâm là SUN397 và Describable Textures Dataset (DTD), cùng phân đoạn tổn thương da trên ISIC 2017. Các notebook CUB và Stanford Cars được sử dụng như bằng chứng thực nghiệm visual bổ sung, không ghép thành một đối chứng thống nhất khi thiết lập khác nhau.

Phần mở rộng, gọi là M-TACS trong project, dùng Florence-2 hoặc Qwen3-VL để sinh caption từ ảnh, CLIP để mã hóa văn bản và các cổng học được để kết hợp text với biểu diễn thị giác. Nghiên cứu được triển khai theo ba lớp: tám ablation xác định vị trí và nội dung text; so sánh các nguồn caption Florence và Qwen; và bốn nghiên cứu kiến trúc với một CSV Florence Prefix cố định. Chuỗi kiến trúc lần lượt kiểm tra chuẩn hóa vector, auxiliary alignment lấy cảm hứng từ mini-batch Learning-to-match và semantic selector giữa ảnh query với caption candidate. Một nhánh ISIC được mở rộng theo cấu hình full_both để kiểm tra khả năng chuyển từ phân loại sang phân đoạn.

Trên DTD split 1, seed 42, DINOv3 ViT-L/16, ảnh 384 × 384 và 50 epoch, ablation candidate_only đạt test Macro-F1 0.850321, nhỉnh hơn visual_only 0.848168 nhưng chưa thể khẳng định ổn định qua nhiều lần huấn luyện. Trong nhóm full_both so sánh nguồn caption, Qwen3-VL-8B và 32B đạt lần lượt 0.821445 và 0.820788, cao hơn Florence Prefix 0.810651 nhưng vẫn thấp hơn visual control lịch sử. Chuỗi kiến trúc Prefix có baseline candidate_only đạt 0.853032; các bản chuẩn hóa, alignment và semantic selector lần lượt đạt 0.849329, 0.848722 và 0.850717. Trên ISIC, bản visual đạt Dice 0.8560 và IoU 0.7738; bản Qwen 8B full_both đạt 0.848689 và 0.765407. Những kết quả này cho thấy caption có tác động, song thêm text, tăng kích thước captioner hoặc làm tốt hơn mục tiêu matching chưa tự bảo đảm cải thiện tác vụ cuối.

Báo cáo phân biệt rõ cấu trúc TACS nguyên bản với những điều chỉnh của project, không tuyên bố tái lập toàn bộ benchmark của tác giả hoặc vượt paper trong điều kiện tương đương. Kết quả chủ yếu là thăm dò trên một seed và một split DTD; một số nhóm còn thiếu prediction CSV, checkpoint hoặc baseline cùng protocol. Giá trị của đề tài nằm ở hệ thống thực nghiệm có truy vết, các đối chứng chỉ ra giới hạn của fusion hiện tại và một lộ trình phát triển tiếp có thể kiểm chứng.

**Từ khóa:** TACS, lựa chọn ngữ cảnh, DINOv3, mô tả ảnh, CLIP, ablation, căn chỉnh đa phương thức, DTD, ISIC.

# MỤC LỤC

[[CONTENTS]]

# THUẬT NGỮ VÀ QUY ƯỚC

**TACS** là Task Aligned Context Selection, phương pháp lựa chọn ảnh ngữ cảnh theo lợi ích đối với tác vụ. **M-TACS** là tên làm việc của các biến thể đa phương thức trong project, không phải tên mô hình do tác giả paper gốc công bố. **Query** là ảnh cần dự đoán; **candidate pool** là bộ ảnh train cố định để chọn ngữ cảnh; **context** là ảnh được chọn cho một query. **Caption** là mô tả do mô hình hoặc người gán; **fusion** là bước kết hợp thông tin; **selector** là mạng chấm điểm và chọn context.

**VLM** là mô hình ngôn ngữ thị giác; **CLIP** là mô hình biểu diễn ảnh và văn bản được tiền huấn luyện tương phản. **ST** chỉ straight-through estimator; **CE** là cross-entropy; **SupCon** là supervised contrastive loss; **LoRA** là low-rank adaptation. **OT** là optimal transport; **KL** là Kullback–Leibler divergence; **DPT** là dense prediction transformer head. **ASWP** là Attention Sliced Wasserstein Pooling. **CI** là khoảng tin cậy. **đpt** là điểm phần trăm.

Các bảng ghi Acc và F1 theo phần trăm nếu tiêu đề có ký hiệu %. Những bảng ghi 0.85 dùng thang 0–1. Chênh lệch 0.01 trên thang F1 tương đương 1 điểm phần trăm, không đồng nghĩa tăng tương đối 1%. K = 1 trong báo cáo là số ảnh context được chọn cho mỗi query, không phải K-fold. DTD split 1 là phân hoạch chính thức số 1 trong mười phân hoạch của bộ dữ liệu.

Mã nguồn kết quả được ký hiệu A01–A08 cho ablation text, C01–C08 cho nguồn caption, S01–S04 cho nghiên cứu kiến trúc, I01–I02 cho phân đoạn, V cho visual lịch sử và U01 cho caption query SUN397. Danh mục notebook và checksum nằm ở Phụ lục A. Số liệu được đọc từ output đã lưu; không phát sinh thêm lần huấn luyện khi tổng hợp báo cáo.

# CHƯƠNG 1 GIỚI THIỆU ĐỀ TÀI

## Bài toán và động cơ

Phân loại ảnh thường ánh xạ một ảnh đầu vào thành một nhãn mà không tham chiếu trực tiếp tới những ví dụ khác. Cách tiếp cận này hiệu quả khi đặc trưng phân biệt rõ ràng, nhưng có thể gặp khó ở các lớp gần nhau hoặc khi một vùng ảnh chứa nhiều tín hiệu gây nhiễu. Một ví dụ ngữ cảnh phù hợp có thể cung cấp phép so sánh về cấu trúc, vật liệu, bố cục hoặc hình thái. Vấn đề quan trọng không chỉ là tìm ảnh giống query nhất, mà là xác định ảnh nào thực sự làm giảm lỗi dự đoán.

TACS đặt retrieval vào mục tiêu học của tác vụ thay vì coi đó là một tiền xử lý cố định [1]. Ảnh có embedding gần nhau chưa chắc mang lại thông tin bổ sung. Ngược lại, ảnh khác lớp hoặc khác vẻ ngoài vẫn có thể tạo một đối chiếu hữu ích. Vì vậy selector cần được đánh giá theo thay đổi của task loss, không chỉ theo mức tương đồng thị giác. Đây là ý tưởng trung tâm mà phần tái triển khai của đề tài hướng tới.

Sau khi xây dựng pipeline ảnh–ảnh, đề tài đặt câu hỏi liệu caption có thể mô tả những thuộc tính mà vector ảnh chưa khai thác hiệu quả hay không. Với SUN397, đó có thể là vật thể, ánh sáng và bố cục cảnh. Với DTD, mô tả cần tập trung vào màu sắc, hình dạng cục bộ, khoảng cách và cấu trúc bề mặt. Với ISIC, mô tả ngắn mang tính toàn cục có thể hữu ích cho nhận biết hình thái, nhưng chưa chắc đủ cho việc xác định biên từng pixel. Sự khác nhau này là lý do cần đánh giá theo tác vụ và không mặc định text sẽ giúp mọi trường hợp.

## Mục tiêu thực hiện

Mục tiêu thứ nhất là xây dựng lại các khối query encoder, candidate bank, learned selector, mô hình tác vụ nhận context và quy trình train–validation–test. Phần lý thuyết phải giải thích được hai đường tối ưu của TACS gốc và đối chiếu chúng với đường thực thi trong source. Việc một notebook chạy ra metric chưa đủ chứng minh mọi chi tiết của paper đã được tái lập.

Mục tiêu thứ hai là thiết kế các mở rộng nhẹ có thể kiểm tra từng bước: tăng backbone hoặc resolution, bổ sung text ở query và candidate, kiểm soát caption, thay nguồn sinh caption, chuẩn hóa vector và thêm ràng buộc alignment. Các thay đổi phải có tên, giả thuyết và đối chứng tương ứng. Không đồng nhất số lượng notebook với số đóng góp thuật toán; các sửa lỗi môi trường, caching và tổ chức thư mục là đóng góp kỹ thuật phục vụ thực nghiệm.

Mục tiêu thứ ba là đưa ra kết luận dựa trên output thật, bao gồm cả trường hợp điểm giảm. Báo cáo ưu tiên sự chính xác của điều kiện so sánh, provenance của CSV và checkpoint, sự khác biệt giữa validation và test, cũng như mức độ bất định chưa được đo. Đối với bài tập lớn, một cải tiến có giả thuyết hợp lý nhưng chưa tăng điểm vẫn mang lại bài học thực nghiệm nếu được phân tích và công bố đúng phạm vi.

## Phạm vi và câu hỏi nghiên cứu

Phần phân tích định lượng tập trung vào DTD official split 1 vì có đủ tám cấu hình ablation, nhiều nguồn caption và chuỗi nghiên cứu kiến trúc. ISIC 2017 cung cấp một kiểm tra mở rộng sang dense prediction. SUN397 thể hiện giai đoạn triển khai và phát triển ban đầu; CUB và Cars bổ sung các kết quả visual. Đề tài không chạy lại toàn bộ 18 dataset của paper, không báo cáo trung bình nhiều split cho DTD và không coi web demo là bằng chứng tăng độ chính xác.

Các câu hỏi nghiên cứu gồm: text nên được dùng ở query hay candidate; lợi ích có phụ thuộc caption đúng ảnh hay chỉ là thay đổi mask và quá trình tối ưu; captioner lớn hơn có tốt hơn trong cùng classifier hay không; phép cộng vector sau projection có xử lý thang độ lớn và modality gap đủ tốt hay không; và alignment hoặc semantic retrieval có chuyển thành lợi ích phân loại hay không. Mỗi câu hỏi được trả lời bằng một nhóm đối chứng riêng thay vì một bảng xếp hạng chung cho mọi run.

## Phần kế thừa và phần thực hiện

TACS, DINOv3, CLIP, Florence-2, Qwen3-VL và ý tưởng m-LTM/OmniRet là các công trình kế thừa, được dẫn nguồn trong tài liệu tham khảo. Phần thực hiện của đề tài là thích nghi các thành phần vào pipeline notebook offline, xây dựng các nhánh fusion/selector, tổ chức ablation, kiểm soát dữ liệu và phân tích các output. Auxiliary matching và semantic selector mới là những thiết kế thích nghi cho bài toán đang xét, không được giới thiệu như tái hiện nguyên thuật toán hoặc kết quả của hai paper multimodal.

Sản phẩm phần mềm gồm notebook theo task/dataset, dữ liệu caption có định danh, công cụ sinh và kiểm tra CSV, các export thực nghiệm, báo cáo phân tích và web demo phân loại. Trạng thái hoàn thiện của từng phần được công bố riêng. Những hướng như Partial OT, token resampling, ASWP hay học liên tục mới ở mức đề xuất, không có số liệu xác nhận trong báo cáo này.

# CHƯƠNG 2 CƠ SỞ LÝ THUYẾT VÀ TACS NGUYÊN BẢN

## Biểu diễn ảnh bằng Vision Transformer

Vision Transformer chia ảnh thành patch và biến mỗi patch thành một token. Các lớp self-attention trao đổi thông tin giữa token để hình thành biểu diễn toàn cục và cục bộ. DINOv3 cung cấp backbone thị giác tiền huấn luyện tự giám sát được dùng để khởi tạo các mạng của project [2]. Tiền huấn luyện và huấn luyện downstream là hai giai đoạn khác nhau: báo cáo không huấn luyện DINOv3 từ đầu.

Ký hiệu S/16, B/16 và L/16 mô tả quy mô backbone với patch size 16. Tăng từ S lên L thay đổi năng lực và chi phí; tăng kích thước ảnh làm tăng số patch. Với ảnh vuông 256 và 384, số patch lần lượt là 16² = 256 và 24² = 576. Nếu ghép một query với một context trước self-attention, riêng phần patch token tăng từ khoảng 512 lên 1.152, chưa kể các token đặc biệt. Do đó độ phân giải không chỉ là một thay đổi tên cấu hình, mà ảnh hưởng cả compute lẫn lượng chi tiết đầu vào.

Trong nhánh phân loại, biểu diễn cuối được đưa vào classifier và tính CE với nhãn lớp. Trong phân đoạn, cần giữ thông tin không gian để decoder tạo mask. Một vector toàn cục có thể tốt cho chọn context nhưng không thay thế được patch features hoặc cấu trúc decoder cần thiết cho dense prediction. Điều này quan trọng khi mở rộng caption toàn cục sang ISIC.

## Query và candidate pool

Cho tập training D và một pool P gồm M ảnh được lấy từ D, selector nhận query x và chấm điểm từng candidate c trong P. Pool của paper được cố định trong train và evaluation; thông thường chiếm 20% training data, riêng DRIVE có xử lý do dữ liệu nhỏ [1, mục 4.2]. Candidate không phải một tập test bổ sung và không lấy từ validation/test để tăng số ví dụ truy xuất.

Ảnh trong pool vẫn có thể xuất hiện như query khi duyệt training set. Điều cần loại là pairing giữa query với chính nó hoặc các mẫu liên quan không được phép, chẳng hạn cùng bệnh nhân trong dữ liệu y khoa. Exclusion phải xảy ra trước sampling hoặc argmax. Với validation/test, nếu split thực sự rời nhau thì không có self ID trong pool, nhưng vẫn cần kiểm tra trùng ảnh hoặc quan hệ nguồn dữ liệu khi có metadata phù hợp.

Một context không nhất thiết cùng lớp query. Selector chỉ được nhận tín hiệu tác vụ qua loss trong training; nhãn candidate không được đưa như một đặc trưng trực tiếp vào dự đoán trong các cấu hình DTD đang phân tích. Chỉ số cùng lớp giữa query/context là một diagnostic, không phải tiêu chí được dùng để chọn ảnh test. Cần phân biệt class label dùng cho loss/đánh giá với label prior được đưa vào input; nhánh prior bị tắt trong các run DTD.

## Learned selector và mô hình tác vụ

Selector biến đặc trưng query và candidate thành utility scores s. Công thức 1 của paper dùng tích vô hướng giữa biểu diễn query và candidate; điểm khác similarity retrieval cố định nằm ở việc biểu diễn được học theo tác vụ. Softmax tạo phân phối trên pool để sampling lúc train; inference chọn candidate có điểm cao nhất. Các projection và cách tính điểm bổ sung trong project là điều chỉnh implementation, không phải công thức nguyên bản mặc định của paper.

[[EQ:sᵢ = zqᵀ zᵢ     πᵢ = softmax(s)ᵢ     i* = argmaxᵢ sᵢ]]

Đối với phân loại trong paper, patch embeddings của query và context được ghép trước các transformer blocks, sau đó toàn mạng được fine-tune. Như vậy ảnh context có thể tương tác với query bên trong transformer, khác với cách lấy hai vector độc lập rồi cộng ở cuối. Đối với segmentation, paper đóng băng backbone, gắn gated cross-attention vào mỗi transformer block và sử dụng lightweight DPT head [1, mục 4.2]. Project có nhánh final-token cross-attention riêng; khác biệt này được mô tả ở chương 3.

[[FIG:Hình 2.1 Kiến trúc TACS gốc|Bổ sung sơ đồ query encoder, pool, utility scores, nhánh Gumbel ST, nhánh policy, context được chọn và task network. Nếu dùng Figure 2 của paper phải ghi rõ nguồn [1]; không ghi sơ đồ này là kiến trúc M-TACS.]]

## Hai đường tối ưu trong paper

Đường differentiable sử dụng Gumbel-Softmax với straight-through estimator. Forward có thể chọn một context cứng nhưng backward sử dụng xấp xỉ khả vi từ trọng số mềm. Task loss nhờ đó truyền tín hiệu tới selector qua lựa chọn. Điều kiện thực thi quan trọng là downstream phải phụ thuộc vào tensor selection_weights theo cách giữ graph; nếu chỉ dùng argmax rồi gather bằng index, đường gradient trực tiếp này bị cắt tại thao tác rời rạc.

Đường policy đánh giá mức cải thiện khi thêm context so với chỉ có query. Với cùng nhãn y, reward dương khi loss của query có context nhỏ hơn loss query-only. Reward được detach để cập nhật policy từ tín hiệu đánh giá, không biến phép trừ loss thành một đường gradient không mong muốn của task network. Advantage được chuẩn hóa trong minibatch để giảm phương sai [1, công thức 8–10].

[[EQ:r = Ltask(xq, ∅, y) − Ltask(xq, c, y)]]

[[EQ:Lpolicy = − E [log πθ(a | xq, P) · A]     LTACS = Lgrad + λ Lpolicy]]

Hai nhánh bổ sung cho nhau: task gradient cung cấp tín hiệu trơn, còn policy reward hướng tới utility của lựa chọn rời rạc. Paper đặt nhiệt độ Gumbel τ = 0,1 và hybrid weight λ = 1. Một biến thể chỉ sampling categorical và policy vẫn là một phương án hợp lý để khảo sát, nhưng không được gọi là tái hiện đầy đủ hybrid optimization. Sự phân biệt này được giữ xuyên suốt khi đọc các notebook DTD.

## Captioner và CLIP có vai trò khác nhau

Florence-2 là mô hình thị giác–ngôn ngữ có các task prompt cho caption, detailed caption và những nhiệm vụ khác [4]. Trong pipeline ban đầu, task MORE_DETAILED_CAPTION tạo raw caption từ pixels. Bộ xử lý sau đó rút mô tả theo policy của thí nghiệm. Qwen3-VL Instruct là lựa chọn captioner khác; các bản 8B và 32B được dùng để xuất CSV độc lập [5, 6]. Các notebook phân loại nạp CSV không đồng nghĩa đang fine-tune captioner trong vòng train TACS.

CLIP được dùng để ánh xạ text thành embedding và, ở những bước nhất định, tính similarity [3]. CLIP không phải một LLM thứ hai đọc và chứng nhận caption an toàn. Cosine similarity cao với tên lớp có thể phản ánh đúng thuộc tính thị giác, từ đồng nghĩa, sự gần ngữ nghĩa không đặc hiệu hoặc shortcut. Vì vậy một ngưỡng duy nhất không thể chứng minh không còn leakage, cũng không thể dùng test score cao để hợp thức hóa việc điều chỉnh ngưỡng sau khi xem test.

## Động cơ tham khảo các hướng multimodal

m-LTM đề xuất học ground metric để matching giữa modality bằng inverse optimal transport trên minibatch và có biến thể Partial OT cho cặp nhiễu [7]. Paper gốc xét audio–text; đề tài chỉ mượn động cơ căn chỉnh và học metric để thiết kế auxiliary ảnh–caption, không suy rằng lợi ích đã chuyển sẵn sang DTD. OmniRet nhấn mạnh biểu diễn token gọn qua resampling và bảo toàn chi tiết bằng Attention Sliced Wasserstein Pooling [8]. Hai thành phần này gợi ý hướng vượt giới hạn một vector text, nhưng chưa được triển khai trong các run được báo cáo.

# CHƯƠNG 3 TÁI TRIỂN KHAI VÀ THIẾT LẬP THỰC NGHIỆM

## Tổ chức hệ thống

Project tách notebook tái triển khai paper, notebook cải tiến và notebook kết quả đã chạy. Các thư mục src và tools chứa helper dùng lại; data/captions lưu CSV có provenance; experiments/completed chứa kết quả đã hoàn thành; reports chứa bảng phân tích. Checkpoint và export tiến trình được quản lý theo experiment, dataset, seed và cấu hình thay vì đặt chung một tên best.pt cho mọi nhánh. Cách tổ chức này nhằm giảm nhầm lẫn giữa template chưa chạy và file có output thật.

Một run đi qua các bước: kiểm tra môi trường, đọc split, tạo pool, chuẩn bị bank/caption nếu cần, khởi tạo model, train, chọn checkpoint bằng validation, rồi đánh giá test. Những notebook có chức năng sinh caption ghi CSV theo từng batch và có thể tiếp tục phần còn thiếu. Resume huấn luyện cần đi kèm identity và scheduler horizon; đổi số epoch rồi nạp bất kỳ checkpoint cũ không tự tạo ra một đối chứng cùng lịch học.

Nguồn dữ liệu cho báo cáo là output notebook đã lưu và các export local được cung cấp. Nhóm S02/S03 có metrics JSON, history và đầy đủ prediction CSV, cho phép kiểm lại Accuracy/Macro-F1. Với các nhóm chỉ có bảng và stdout, báo cáo giữ độ chính xác hiển thị, không suy thêm chữ số hoặc tự tạo độ lệch chuẩn. Những thông báo “Saved ...” trong Kaggle chỉ xác nhận log đã ghi đường dẫn, không đồng nghĩa mọi artifact đó đều đã được chuyển về local.

## Ánh xạ các khối paper sang source

Nhánh SUN visual có các hàm tiêu biểu `_forward_token_concat_features`, `select_full_pool`, `select_bank_images_from_weights` và `train_one_epoch`. Source ghép token query/context và có đường selection_weights nhân image bank để giữ straight-through graph, cùng reward/policy loss. Đây là một cơ sở để giải thích việc xây dựng cấu trúc TACS. Tuy nhiên source còn thêm query/context gate và SupCon, nên bản triển khai không chỉ là sao chép nguyên mọi chi tiết trong paper.

Các template SUN chưa có output không được dùng thay cho run đã train. Ví dụ kết quả S/16–256 true validation được đọc từ notebook đã chạy `tacs-sun-da-sua (2).ipynb`; source chuẩn tên `tacs-sun-true-val-split.ipynb` phục vụ đối chiếu kiến trúc nhưng chưa tự cung cấp số test. Một hạn chế ở source SUN được đối chiếu là API selector chưa nhận query ID/excluded positions dù pool nằm trong cùng train queries. Vì vậy không chứng nhận self-exclusion cho mọi run SUN lịch sử. Đây không đồng nghĩa dùng test labels để train, nhưng có thể tạo shortcut khi query tự làm context.

Trong nhánh DTD ablation, đường thực thi `run_forward → select_context → candidate_images_at(indices)` chọn context bằng categorical index. Mạng tác vụ vẫn được train bằng CE và các loss được khai báo, còn selector được cập nhật từ policy/entropy; các nghiên cứu alignment còn thêm tín hiệu qua projection chung. Không có task gradient Gumbel-ST đi qua gather index rời rạc này. Việc lớp cha còn giữ một hàm Gumbel không chứng minh hàm đó được gọi. Vì vậy báo cáo gọi nhóm DTD là biến thể M-TACS lấy cảm hứng từ TACS, không dùng nó làm bằng chứng thay thế toàn bộ hybrid optimization nguyên bản.

[[TABLE:protocol]]

## Dữ liệu và phân chia

SUN397 được dùng theo Training_01 và Testing_01, mỗi tập có 19.850 ảnh. 10% ảnh trong training partition được tách có phân tầng để tạo validation, còn 17.865 ảnh train và 1.985 ảnh val; test giữ 19.850. Đây là protocol project đã chọn để chọn checkpoint bằng validation thật, khác với một số run lịch sử train trên full train rồi dùng monitor nội bộ. Không ghép hai kiểu validation này thành cùng điều kiện.

DTD có 47 thuộc tính kết cấu và 5.640 ảnh. Nhóm thí nghiệm chính sử dụng `labels/train1.txt`, `val1.txt`, `test1.txt`: mỗi lớp có 40 ảnh ở từng split, tương ứng 1.880 ảnh mỗi phần [9]. Loader kiểm tra file tồn tại, phân bố lớp, ID không trùng và split không giao nhau. DTD_SPLIT là lựa chọn phân hoạch chính thức 1 đến 10, không phải seed. Do mới chạy split 1, báo cáo không gọi các số hiện có là trung bình benchmark mười split.

ISIC 2017 được dùng cho segmentation với 2.000 cặp image–mask train, 150 val và 600 test, phù hợp quy mô các split của challenge [10]. Pool có 400 ảnh train. Source có self-exclusion theo ID, nhưng chưa có kiểm chứng độc lập loại mọi quan hệ cùng bệnh nhân vì thiếu patient metadata/manifest tương ứng. Các kết quả chỉ phản ánh đánh giá mô hình nghiên cứu trên split đã sử dụng, không mang ý nghĩa kiểm định lâm sàng.

[[TABLE:datasets]]

## Pool và sự nhất quán của ảnh với caption

Trong DTD, pool gồm 376 ảnh được lấy ngẫu nhiên từ train với pool seed 42. Tập query vẫn là toàn bộ train; một pool image khi làm query bị mask khỏi tập context hợp lệ của chính nó. Mỗi index trong bank phải trỏ tới cùng image ID, ảnh, text vector, validity mask và metadata. Nếu selected index là i thì fusion phải lấy caption của candidate i, không lấy caption hàng i của một CSV có thứ tự khác.

Ảnh candidate có caption không hợp lệ vẫn ở trong pool và vẫn có thể đóng góp ngữ cảnh thị giác. Text vector/gate tương ứng được mask về 0. Đây là lựa chọn giúp tách việc thiếu text khỏi việc xóa ảnh training, nhưng cũng tạo ra availability pattern mà mô hình có thể khai thác. Vì vậy các đối chứng shuffle giữ mask và báo cáo coverage riêng; giữ mask chưa đồng nghĩa đã loại bỏ mọi shortcut thống kê từ mask.

## Tiền xử lý và tối ưu

Các nhóm DTD chính dùng resize và augmentation training đã khai báo, gồm random resized crop, horizontal flip, color jitter, grayscale, blur và random erasing; evaluation resize xác định về 384 × 384 rồi chuẩn hóa. Mục tiêu của augmentation là giảm phụ thuộc vào biến thiên không thiết yếu, nhưng báo cáo không tách được tác dụng riêng của từng phép biến đổi vì chưa có ablation tương ứng. Caption tạo từ ảnh gốc không được sinh lại cho từng crop ngẫu nhiên; đây là một nguồn lệch có thể xảy ra giữa mô tả toàn ảnh và view train.

Backbone dùng LR nhỏ hơn head/selector để fine-tune thận trọng. Label smoothing, SupCon và entropy regularization là các điều chỉnh trong project, không được ngầm coi là nguyên cấu hình tối giản của paper. Các số ở bảng 3.3 mô tả nhóm DTD đã chạy, không thay thế việc lưu toàn bộ protocol của từng run. Batch caption generation và batch training là hai giá trị độc lập.

[[TABLE:hypers]]

## Khác biệt backbone và độ phân giải

Việc dùng ViT-L/16 và 384 thay cho S/16 và 256 phải được công bố như một điều chỉnh thực nghiệm. Mô hình lớn hơn có thể biểu diễn mạnh hơn nhưng cũng thay capacity, memory, tốc độ và động lực tối ưu. Không có cặp S/L hoặc 256/384 chỉ thay đúng một yếu tố trong nhóm DTD hiện tại, nên chưa thể định lượng phần tăng do từng thay đổi. Không gọi chênh lệch DTD so với số paper là cải tiến thuật toán TACS.

Nhánh ISIC vẫn dùng frozen S/16 ở 448, không phải L/16 như DTD. Nó áp dụng hai gated cross-attention blocks sau final patch tokens rồi DPT-lite, khác vị trí fusion ở mỗi transformer block của paper. Các khác biệt này được giữ trong tên bảng và phần thảo luận để tránh kết luận quá mức khi một số Dice/IoU cao hơn reference paper về mặt số học.

## Chọn checkpoint và đánh giá

Nhóm DTD chọn checkpoint có validation loss nhỏ nhất trong budget 50 epoch. Best epoch không nhất thiết là epoch có validation F1 cao nhất; lấy một checkpoint khác sau khi xem test sẽ làm thay đổi protocol. Tám ablation được tách notebook nhưng dùng cùng horizon scheduler 50, tránh so model dừng 35/100 với model hoàn thành 50/50 mà coi lịch học giống nhau. Những checkpoint cũ chỉ được resume khi identity và quy tắc ngân sách phù hợp.

Accuracy là tỷ lệ ảnh đúng. Macro-F1 tính F1 riêng cho từng lớp rồi lấy trung bình không trọng số, nhằm phản ánh lỗi giữa các lớp thay vì chỉ tổng số đúng. Với ISIC, notebook tính hard foreground Dice và IoU theo từng ảnh rồi lấy trung bình, threshold sigmoid lớn hơn 0,5 và mask ở resolution 448. Trung bình Dice không được chuyển thành trung bình IoU bằng một công thức đại số của riêng một ảnh.

[[EQ:Accuracy = số dự đoán đúng / số ảnh     Macro F1 = (1 / C) Σ F1c]]

[[EQ:Dice = 2 |P ∩ G| / (|P| + |G|)     IoU = |P ∩ G| / |P ∪ G|]]

## Kiểm soát tính hợp lệ của so sánh

Một so sánh nội bộ có sức thuyết phục khi giữ dataset, split, seed, pool, preprocessing, backbone, budget, scheduler, checkpoint criterion và metric. Tên file giống nhau hoặc AST giống nhau chỉ xác nhận một phần cấu trúc, không chứng minh weights/runtime và dữ liệu thực tế giống byte. Các nhóm cũ còn thiếu export nên mức kết luận phải thấp hơn nhóm có protocol và predictions đầy đủ.

Mỗi thay đổi được trình bày với đối chứng gần nhất. Full_both dùng cả query text và semantic selector không phải một đối chứng chỉ thêm candidate text so với candidate_only. S04 giữ S03 rồi thêm selector ảnh–caption; nó không so thuần selector với S01 vì còn khác normalization/alignment. Can thiệp tại cùng checkpoint trả lời câu hỏi sử dụng text lúc inference, không thay thế việc retrain một visual-only model.

# CHƯƠNG 4 CÁC HƯỚNG MỞ RỘNG VÀ CẢI TIẾN

## Pipeline sinh caption và quản lý CSV

Pipeline ban đầu sinh raw caption bằng Florence-2-base với task MORE_DETAILED_CAPTION. Processor nhận ảnh và task prompt, không được nhận class label hay tên thư mục làm chỉ dẫn mô tả. Sau sinh, văn bản được chuẩn hóa và chọn đoạn ngắn mô tả thuộc tính thị giác. CSV gắn caption với image_id và split, đồng thời giữ raw_caption, final_caption, validity, leakage flags, semantic score/class, model ID, prompt version và retry metadata.

Việc lưu cả raw và final text cho phép phân biệt lỗi của captioner với lỗi extraction hoặc policy. Batch hoàn thành được append và flush để có thể resume. Các file cũ thiếu schema hoặc sai prompt/model không được nối âm thầm vào file mới. Một CSV đủ 5.640 dòng chỉ xác nhận đủ ID; nếu caption_valid đều false thì classifier vẫn không nhận text hữu ích. Vì vậy phải kiểm tra coverage trước training, thay vì chỉ nhìn số dòng.

Mixed precision, torch inference mode, batching, DataLoader và cache giúp giảm thời gian sinh và mã hóa; giảm batch khi OOM giúp giữ tiến trình. Những tối ưu này có thể giải thích khác biệt tốc độ chạy, nhưng không có benchmark thời gian đồng nhất giữa các phiên để báo mức tăng tốc định lượng. Log “20 phút” hoặc “7–8 giờ” do quan sát rời rạc không đủ để quy toàn bộ chênh lệch cho một tham số hay khẳng định caption thông minh hơn.

[[FIG:Hình 4.1 Luồng tạo caption và đưa vào M TACS|Bổ sung sơ đồ ảnh → captioner → raw text → policy → CSV → frozen CLIP → cache vector và mask → selector hoặc fusion. Ghi rõ class labels không đi vào captioner và test labels không dùng cho tối ưu.]]

## Điều chỉnh policy cho DTD

Policy đầu tiên rút các từ đầu caption rồi loại theo lexical terms và CLIP threshold 0,30. Khi kiểm tra CSV thực tế, toàn bộ caption bị semantic filter loại; nhiều câu sau cắt chỉ còn boilerplate như “The image is a close up of a”. Mô tả như vậy đủ số từ nhưng không chứa nội dung kết cấu. Lỗi này nằm ở pipeline sử dụng text, không thể khắc phục bằng cách mặc định captioner hoặc TACS sẽ tự hiểu phần mô tả thiếu.

Policy `dtd_lexical_v2` xử lý phần mở đầu, chọn mô tả ngắn 5–8 từ theo tokenizer của pipeline và kiểm tra tên lớp/alias trực tiếp. Semantic CLIP score/flag vẫn được lưu để audit, nhưng không còn tự loại cứng mọi caption khi vượt 0,30. Trong chuỗi Prefix, có trường hợp toàn bộ pool mang semantic flag nhưng 324/376 caption vẫn được dùng theo policy v2. Điều này là chủ đích của policy, không phải quên áp dụng mask.

Tên lớp DTD vốn là tính từ mô tả texture nên phải phân biệt ba tình huống: mô tả thuộc tính có thể quan sát; model dựa quá mạnh vào một từ khóa; và vi phạm train/test thật sự do đưa ground-truth test vào input hoặc tối ưu. Trùng từ với tên lớp không tự chứng minh vi phạm protocol, nhưng là lý do thiết kế redaction/unscreened controls. Ngược lại, xóa hết từ trong vocabulary đã định nghĩa cũng không bảo đảm mọi từ đồng nghĩa, suy diễn hoặc shortcut đã biến mất.

## Text cho query và text cho context

Caption được mã hóa bằng frozen CLIP text tower thành vector 512 chiều. Projection đưa text về số chiều tương thích với representation thị giác; gate học được kiểm soát residual text trước classifier. Text query và text candidate là hai nhánh riêng. Query caption có thể hỗ trợ mô tả ảnh cần dự đoán; candidate caption mô tả chính ảnh context được selector chọn. Việc dùng caption candidate không yêu cầu đưa nhãn thật của candidate vào classifier.

Trong candidate_only, selector chỉ xem ảnh. Sau khi chọn index, hệ thống lấy ảnh context và caption đúng index đó để fuse; caption query bị tắt. Trong full_both cũ, text ảnh hưởng cả selection qua caption–caption similarity và fusion qua hai nhánh. “Text vào selector” không phải caption do một mô hình selector mới sinh ra; đó là text embedding tham gia công thức điểm chọn context.

Run SUN397 query-caption U01 dùng S/16 ở 384 × 384, K = 1, pool 3.573 ảnh train. Bảng test lưu Accuracy 0.761159, Macro-F1 0.760189 và caption-valid rate 0.605542; best epoch 34, early stopping ở 44. Đây là bằng chứng nhánh text ban đầu đã chạy, không phải đối chứng chỉ thêm text so với SUN visual S/16–256 vì resolution và pipeline khác nhau. Các điểm SUN L/16 từng trao đổi nhưng chưa gắn chắc artifact không được thay vào kết quả này.

[[EQ:h = LayerNorm(v + mq gq Pq(tq) + mc gc Pc(tc))]]

Trong công thức khái quát, v là biểu diễn visual sau xử lý query/context, tq và tc là text vectors, mq và mc là validity masks. Pq, Pc là projection còn gq, gc là cổng học được. Công thức minh họa vai trò các nhánh, không đồng nhất mọi notebook phân đoạn hoặc mọi thử nghiệm token fusion với phép cộng này. Với candidate_only, thành phần query text bằng 0 theo thiết kế.

Gate không là attention map, xác suất dự đoán đúng hay phần trăm thông tin văn bản đóng góp. Một gate nhỏ nhân vector norm lớn vẫn có thể tạo residual đáng kể. Caption candidate còn tác động selector ở full_both ngay cả khi c_gate trong fusion nhỏ. Do đó kết luận “mô hình bỏ text” phải dựa thêm vào norm ratio, can thiệp và thay đổi dự đoán, không chỉ nhìn một số gate làm tròn ba chữ số.

## Từ prototype nhiều context đến đối chứng một context

Giai đoạn prototype đã khảo sát kết hợp visual scores với similarity caption query–candidate và chọn top 3 context. Ở một bản đầu, candidate text mới tham gia selector, chưa thực sự vào downstream fusion. Sau đó nhánh lấy caption của context được chọn được bổ sung, rồi tạo bản K = 1 theo yêu cầu giữ một ảnh context. Vì vậy “dùng candidate caption” và “K = 3” là hai thiết kế độc lập, không nên suy từ tên notebook multimodal rằng cả hai đã được kiểm tra cùng nhau.

K = 3 làm tăng token, compute và số ví dụ hỗ trợ cho mỗi query; đồng thời thay đổi quy trình sampling không hoàn lại và loss/policy liên quan. Nhóm DTD có kết quả trong báo cáo chốt K = 1 để kiểm tra vai trò text. Không có bộ matched K1/K3 được kiểm chứng ở đây để tuyên bố tăng số context làm tốt hơn hoặc xấu hơn. Các chỉ số SUN từng báo trong trao đổi nhưng chưa khóa đúng artifact không được dùng làm bảng định lượng chính.

## Tám ablation về vị trí text và shortcut

Tám cấu hình chia thành hai lớp câu hỏi. Bốn cấu hình đầu xác định text hỗ trợ ở query, context hay cả hai trong khi selector visual. Full_both bật thêm semantic score. Ba control sau làm thay đổi quan hệ giữa caption và ảnh hoặc sự xuất hiện tên lớp. Chúng giúp kiểm tra một kết quả đẹp có phụ thuộc đúng nội dung hay chỉ do quá trình tối ưu và availability pattern.

[[TABLE:modes]]

Ở full_shuffle_both, caption ở cả query và candidate được đảo nhưng giữ mask. Đây là retraining một model với dữ liệu text biến đổi, không phải lấy checkpoint full_both rồi tắt text. Ở full_unscreened, các mô tả trước lọc được dùng cho cả hai vai trò; coverage do đó cũng thay đổi. Ở full_redacted, tên lớp và alias bị xóa trước CLIP encoding cho cả fusion và selector; không dùng embedding gốc để vô tình lách phép che.

Redaction control có thể giữ mô tả còn lại dù ngắn hơn 5 từ sau xóa, theo availability rule của bản này. Vì vậy không áp đặt lại một rule độ dài khác khi giải thích số coverage 100%. Mỗi biến thể phải giữ đúng policy của nó; việc gọi tất cả là screened sẽ làm mất ý nghĩa kiểm chứng. Thí nghiệm không sửa ngưỡng hoặc chọn prompt bằng test để tạo thứ hạng mong muốn.

## Cải tiến nguồn caption và dữ liệu tham chiếu

Một hướng là fine-tune captioner để mô tả texture phù hợp hơn thay vì chỉ đổi classifier. Bộ caption_references được chuẩn bị trên 188 ảnh, tương đương một tập tham chiếu nhỏ, không phải full CSV DTD. Trường draft_caption_unverified được tách khỏi target_caption, cùng source/reviewer/teacher metadata nhằm tránh dùng caption Florence cũ làm ground truth mặc định. Quy trình teacher hỗ trợ cần xem ảnh thật, không suy caption chỉ từ tên folder.

Reference do AI teacher tạo là supervision yếu cần kiểm duyệt, không tự trở thành human ground truth chỉ vì review_status có chữ approved. Báo cáo không xác nhận đã có bộ tham chiếu người đánh giá độc lập toàn diện hoặc một chỉ số caption quality chuẩn. Khi fine-tune bằng LoRA, mục tiêu là điều chỉnh một phần decoder với số tham số cập nhật thấp hơn full fine-tuning [11]; rủi ro là tập nhỏ, văn phong hẹp và việc tối ưu teacher text có thể không chuyển thành thông tin hữu ích cho classifier.

Hướng khác giữ captioner đóng băng nhưng cải tiến extraction/selection. Prefix là control rút đoạn; FirstValid tăng khả năng lấy được một mô tả đạt ràng buộc; SingleGrounded và MultiGrounded dùng khớp ảnh–text để chọn mô tả từ các giả thuyết. Chúng không phải bốn lần fine-tune. CLIP score dùng để lựa chọn có thể thiên về tín hiệu mà CLIP ưa thích, nên không dùng lại score đó như bằng chứng độc lập caption đúng hơn.

[[TABLE:caption_methods]]

Qwen3-VL-8B-Instruct và 32B-Instruct được chạy thành hai nguồn CSV, sau đó đưa vào cùng classifier full_both để đánh giá downstream. So sánh này kiểm tra cả pipeline nguồn caption cùng policy/mask đi kèm, không tách riêng tác dụng số tham số. Qwen 32B không được giả định tốt hơn trước khi có kết quả. Caption sinh từ pretrained model cũng không trở thành nhãn thật của texture; nhãn thật cho classification vẫn là class labels của DTD.

## Chuỗi nghiên cứu kiến trúc với Prefix cố định

Sau các ablation, đề tài chuyển câu hỏi từ chọn captioner sang cách kết hợp modality. Đưa vector về cùng số chiều chỉ là điều kiện hình thức để cộng/attention, không bảo đảm chúng có cùng ngữ nghĩa. Đặc trưng DINO và CLIP text được học bằng mục tiêu khác nhau; có thể khác phân bố, thang norm và thông tin giữ lại. Do đó một phép cộng có projection chưa thể được gọi là tối ưu hay không làm mất đặc trưng từng nhánh.

Chuỗi S01–S04 khóa CSV Prefix bằng SHA256, giữ candidate_only, tắt query text, giữ split/pool/seed và K = 1. Baseline mới phải được train riêng vì candidate_only lịch sử chưa có đủ bằng chứng CSV byte-identical với Prefix control. Cách đặt baseline này tránh lấy một điểm cũ bất kỳ rồi quy mọi khác biệt sau đó cho một module mới.

### Baseline S01

S01 dùng vector visual đã L2-normalize và projected text qua Linear, LayerNorm, Dropout trước gated additive fusion. Projected text chưa L2-normalize ở điểm fusion. Với chiều 1.024, LayerNorm không ràng buộc norm cuối giống 1; tham số học được và dropout còn làm thay đổi độ lớn. Vì vậy gate cỡ 0.001 không đủ kết luận ảnh hưởng text gần 0. S01 cung cấp mốc thực nghiệm và động cơ đo tỷ lệ norm residual thay vì sửa gate thủ công.

[[EQ:ρ = ‖gc Pc(tc)‖₂ / (‖v‖₂ + ε)]]

### Chuẩn hóa fusion S02

S02 L2-normalize text sau projection ngay trước gate/fusion để đưa độ lớn hai nhánh về thang dễ so sánh hơn. Selector vẫn chỉ dùng ảnh, query caption vẫn tắt và chưa có auxiliary alignment loss. Thay đổi này kiểm tra vấn đề scale, không tự căn chỉnh ngữ nghĩa. Nếu model điều chỉnh gate sau normalization thì trị số gate có thể tăng nhưng task performance chưa chắc tăng.

S02 bổ sung diagnostic tại cùng checkpoint trên validation: dùng đúng caption, mask text và shuffle text hợp lệ. Context image được giữ cố định trong ba trường hợp. Bên cạnh F1 và CE, có thể đo thay đổi prediction, xác suất và norm. Mục đích là biết text có tác dụng trực tiếp hay không và tác dụng đó có phụ thuộc caption đúng ảnh hay không, tách khỏi thay đổi retrieval.

### Auxiliary alignment S03

S03 giữ fusion của S02 và thêm matching loss lấy cảm hứng từ m-LTM. Các cặp là ảnh candidate train với caption của chính ảnh đó, không gán mọi query và caption context được retrieve thành positive. Image anchors được lấy từ feature bank ở đầu epoch và detach. Projection text dùng chung với nhánh fusion nên loss ràng buộc một biểu diễn thực sự đi vào model, không chỉ học một đầu phụ rồi bỏ đi khi dự đoán.

Khoảng cách sử dụng metric diagonal Mahalanobis dương, chuẩn hóa mean-one để tránh thay đổi scale tùy ý. Cost tạo transport plan bằng balanced log-Sinkhorn, sau đó KL matching với target có xử lý các caption trùng câu. Hai ảnh có caption giống nhau không bị ép thành negative độc nhất chỉ vì khác source ID. Cấu hình dùng weight 0,05, warmup 5 epoch, batch alignment 32, epsilon 0,1 và tối đa 100 Sinkhorn iterations; các tham số này là lựa chọn implementation, chưa có sweep tối ưu đầy đủ.

[[EQ:dM(v, t) = Σⱼ mⱼ (vⱼ − tⱼ)²     mⱼ > 0     mean(m) = 1]]

[[EQ:L = Ltask + λpolicy Lpolicy − λentropy H(π) + λalign LKL]]

S03 dùng balanced transport, không phải Partial OT; không suy từ việc cùng nhắc OT rằng tính robust với noisy pairs đã được kiểm chứng. Chẩn đoán caption-group top 1 trên pool đo fit của auxiliary objective, không phải retrieval held-out. Một matching loss giảm có thể cho thấy module đang học mục tiêu phụ nhưng chưa đủ để chứng minh phân loại sẽ tốt hơn.

### Semantic selector ảnh và caption S04

S04 giữ normalization/alignment của S03 và thêm semantic residual chấm ảnh query với caption candidate trong không gian projection/metric đã học. Đây là thay đổi khác full_both cũ vốn dùng caption query–caption candidate. S04 không cần caption query ở train hoặc inference. Nó kiểm tra khả năng dùng text candidate để cải thiện lựa chọn context trong khi giữ thiết kế candidate_only ở fusion.

Visual score và semantic score được chuẩn hóa trước kết hợp. Semantic residual được chặn độ lớn, dùng warmup visual 5 epoch và ramp thêm 5 epoch; alpha bị giới hạn bởi 0,25 nhân sigmoid của tham số học được và hệ số ramp. Candidate caption không hợp lệ hoặc semantic score không có phương sai sẽ fallback về visual. Self candidate bị loại trước khi tính thống kê semantic. Không có inference-time OT giữa các query trong cùng batch; mỗi query vẫn có thể dự đoán độc lập với thứ tự batch.

[[EQ:scombined = z(svisual) + α · clip(z(ssemantic), −3, 3)     0 ≤ α ≤ 0.25]]

S04 có thêm can thiệp selector: tắt semantic residual hoặc đảo text chỉ trong selector rồi chọn lại context. Sau reselect, fusion vẫn dùng caption đúng của context mới. Nhờ vậy phép kiểm tra không vô tình đồng thời đưa caption sai vào fusion. Alpha khoảng 0,131 trong checkpoint tốt nhất là hệ số phối trộn điểm, không phải độ tin cậy 13,1% hay mức lợi ích causal của text.

[[FIG:Hình 4.2 Chuỗi bốn nghiên cứu kiến trúc|Bổ sung bốn sơ đồ cạnh nhau hoặc một sơ đồ tăng dần: S01 candidate caption fusion, S02 thêm L2, S03 thêm own-pair alignment, S04 thêm ảnh-query/caption-candidate residual selector. Giữ query text tắt trong cả bốn sơ đồ.]]

## Mở rộng sang phân đoạn ISIC

Nhánh ISIC full_both sinh/resume caption Qwen 8B trong notebook rồi encode CLIP. Query và candidate text được đưa vào fusion và semantic selector cùng với visual pipeline. Policy mô tả thuộc tính thấy được không nhận ground-truth mask làm chỉ dẫn sinh caption. Mask thật chỉ dùng cho training loss hoặc đánh giá đúng split, không phải nội dung được chuyển cho captioner.

Các thành phần visual chính giữa hai bản ISIC được giữ theo source: frozen S/16, ảnh 448, final-token cross-attention và DPT-lite, BCE cộng Dice loss. Tuy nhiên full_both vẫn tăng tham số trainable, thay ảnh context được chọn và quá trình tối ưu, nên đây không phải một can thiệp chỉ thay một câu caption ở inference. Một template Florence Prefix ISIC được tạo để khảo sát tiếp, nhưng chưa có output xác nhận và không được điền số dự đoán.

## Cải tiến kỹ thuật phục vụ khả năng tái chạy

Trong môi trường Internet off, model folder và thư viện phải được chuẩn bị đầy đủ. Florence checkpoint legacy có remote code không mặc định tương thích runtime Transformers mới; project phân biệt backend/checkpoint và có hướng chạy subprocess với bộ thư viện offline phù hợp. Chỉ sửa đường dẫn không thể tạo ra weights còn thiếu; ngược lại weights đã có không bảo đảm phiên bản thư viện phù hợp. Mục tiêu kỹ thuật là làm lỗi rõ và không tải ngầm qua mạng.

Các notebook cần tokenizer đặt TOKENIZERS_PARALLELISM=false trước khi dùng tokenizer hoặc fork workers, và có thể dùng caption workers bằng 0 trong bước sinh offline. Điều này xử lý cảnh báo fork/tokenizers; nó không bảo đảm mọi lỗi multiprocessing khác biến mất. Một số output vẫn có QueueFeederThread, Bad file descriptor hoặc semaphore khi cleanup. Báo cáo ghi nhận run completed/test nếu có bằng chứng, đồng thời không gọi log hoàn toàn sạch lỗi.

Định danh run kết hợp cấu hình, caption hash, split/pool và seed để hạn chế nạp nhầm checkpoint. Caching embeddings giảm việc chạy lại CLIP; refresh candidate visual features giúp bank phù hợp backbone đang học. Tách notebook theo từng ablation làm compute và tiến trình rõ hơn, nhưng tổng chi phí vẫn là nhiều lần train độc lập, không phải một lần 50 epoch cho toàn bộ tám cấu hình.

## Những hướng đã thảo luận nhưng chưa có kết quả

Token-level fusion/resampler và ASWP được giữ làm hướng phát triển để bảo toàn nhiều chi tiết hơn một vector toàn cục. Partial OT là một control tiềm năng khi ảnh–caption không khớp hoặc mô tả thiếu, nhưng chưa được triển khai trong chuỗi S01–S04. Học liên tục bằng cách nhận ảnh inference vào train cũng mới là ý tưởng; nếu thực hiện sẽ cần nhãn hoặc pseudo-label, chống quên và protocol streaming riêng. Không được đưa ảnh test chuẩn vào train rồi tiếp tục báo metric trên cùng test như một benchmark độc lập.

Việc dùng teacher mạnh hơn, tăng dữ liệu reference, kiểm duyệt con người, thử nhiều seed/split và matched-mask caption comparisons có thể làm kết luận chắc hơn. Các hướng này chưa được ghi thành thành quả đã đạt. Cải tiến của đề tài gồm cả thiết kế thử nghiệm và học được giới hạn của thiết kế hiện tại, không chỉ những thay đổi có điểm test cao hơn.

# CHƯƠNG 5 KẾT QUẢ THỰC NGHIỆM VÀ THẢO LUẬN

## Cách đọc các nhóm kết quả

Kết quả chia thành bốn nhóm chính: vị trí text, nguồn caption, kiến trúc với CSV cố định và segmentation. Mỗi nhóm có đối chứng gần nhất riêng. Các bảng đều đọc output thật, nhưng mức kiểm chứng artifact khác nhau: notebook tables/stdout cho phần lớn run, full metrics/prediction exports cho S02/S03. Không cộng các điểm test từ nhóm khác nhau để tạo một mức tăng tổng hợp của “mô hình cuối”.

Bảng 5.1 ghi số tác giả công bố để người đọc hiểu bối cảnh, không trộn với project. Paper dùng S/16–256, budget 100 và trung bình ba seed; DTD mở rộng dùng L/16–384, 50 epoch, split 1 và seed 42. Macro-F1 là metric bổ sung của project, không được so trực tiếp với cột Accuracy của paper. Ngay cả khi cùng Accuracy, khác protocol cũng ngăn kết luận causal về mức vượt phương pháp gốc.

[[TABLE:paper_reference]]

## Ablation vị trí và nội dung text trên DTD

[[TABLE:ablation_results]]

Visual_only có test Macro-F1 0.848168. Candidate_only đạt 0.850321, hơn 0.002153, tương đương 0,2153 đpt; Accuracy hiển thị tương ứng khoảng bốn ảnh đúng nhiều hơn trên 1.880 test. Tuy nhiên validation F1 của candidate_only thấp hơn visual_only trong run lịch sử. Biên nhỏ và chỉ một seed không cho phép kết luận có cải thiện ổn định. Visual_only ở đây vẫn có context image, nên bảng không trả lời TACS hơn No-Context bao nhiêu.

Query_only thấp hơn visual_only khoảng 2,51 đpt F1. Thêm cả hai nhánh fusion giảm xuống 0.810661, thấp hơn candidate_only gần 3,97 đpt. Khi bật thêm caption–caption selector, full_both đạt 0.809124, chưa cải thiện so với both_fusion. Các kết quả này không ủng hộ giả thuyết hai nhánh text tạo lợi ích cộng gộp trong thiết kế additive fusion hiện tại. Chưa thể xác định nguyên nhân duy nhất là caption sai, query text dư thừa hay optimization; cần những phép can thiệp tách riêng các yếu tố đó.

Full_shuffle_both đạt 0.833392, cao hơn full_both khoảng 2,43 đpt dù caption không còn khớp ảnh. Audit của shuffle giữ validity mask và thay đổi nguồn/nội dung text thực sự, nên đây không chỉ là đổi tên cấu hình. Tuy vậy shuffle tác động đồng thời fusion và selector, có thể đổi context và cả quá trình training; nó không chứng minh mọi caption đều vô ích, cũng không cô lập tác dụng của riêng candidate_only. Kết luận hẹp là kết quả hiện tại chưa ủng hộ lợi ích của alignment caption trong cấu hình full này.

Full_unscreened thấp nhất với F1 0.789774. Dùng text trước lọc không làm điểm tăng theo kỳ vọng shortcut đơn giản. Full_redacted đạt 0.811105, cao hơn unscreened khoảng 2,13 đpt nhưng vẫn thấp hơn visual_only 3,71 đpt. Trong output của hai control này coverage đều 100% và không caption nào bị xóa rỗng ở redacted; đối chiếu vì thế ít nhiễu coverage hơn screened/unscreened. Tuy nhiên nguồn CSV từng run chưa được khóa byte đầy đủ và có hai lần train riêng, nên không quy toàn bộ khác biệt cho riêng tên lớp.

Các số trên không chứng minh “không có leakage”. Metric thấp có thể cùng tồn tại với một số shortcut; metric cao cũng có thể đến từ thông tin hợp lệ trong pixels. Muốn kết luận sâu cần kiểm tra ảnh–text độc lập, per-class errors, nhóm caption valid/invalid, matched masks và kế hoạch đánh giá không điều chỉnh bằng test. Chỉ thay threshold đến khi test ổn định/cao là dùng test để lựa chọn, không thay thế kiểm chứng tính đúng của dữ liệu.

[[TABLE:ablation_gates]]

Coverage query screened khoảng 84,63% trên test, trong khi context rate thay đổi theo selector. Candidate_only có c_gate khoảng 0.001122, nhưng phần lớn context test vẫn có text hợp lệ; không thể giải thích toàn bộ kết quả bằng “mọi caption bị mask”. Gate query giảm từ mức lớn lúc đầu về khoảng 0,01–0,02 ở nhiều bản. Nhánh text có thể đã ảnh hưởng quỹ đạo tối ưu trước khi gate giảm, nên gate nhỏ cuối train không làm model tương đương visual_only được train từ đầu.

Giới hạn của nhóm A là các run chủ yếu có notebook output, chưa có đủ caption/pool/prediction exports cho paired audit. Thống kê caption giống nhau không chứng minh từng CSV giống byte. Query_only có trạng thái completed 50 nhưng stdout epoch lưu chưa đầy đủ đến 50; cần history để xác nhận từng dòng. Các lỗi cleanup worker xuất hiện trong một số stderr nhưng final-test vẫn có output; không tự loại bỏ metric cũng không che đi tình trạng môi trường này.

[[FIG:Hình 5.1 Đường học của tám cấu hình text|Bổ sung đường validation CE và Macro-F1 từ history đã export; không thay test bằng điểm cao nhất trên đường validation. Dùng cùng trục và ghi best checkpoint chọn theo CE. Nếu chỉ có ảnh nhúng notebook, ghi rõ nguồn và độ chính xác đọc được.]]

## So sánh các phương pháp caption

[[TABLE:caption_results]]

Trong nhóm sáu run full_both, Qwen 8B đạt F1 0.821445 và Qwen 32B đạt 0.820788, hơn Prefix khoảng 1,08 và 1,01 đpt. Chênh giữa hai Qwen chỉ khoảng 0,066 đpt, không chứng minh 8B luôn tốt hơn 32B hoặc việc tăng tham số không có giá trị. Đây là một seed với masks và caption distribution khác nhau, không phải scaling law có kiểm soát.

Prefix và FirstValid gần tương đương: 0.810651 và 0.809382. SingleGrounded đạt 0.779611 và MultiGrounded 0.790596, thấp hơn hai control đơn giản dù coverage cao. Reranking bằng một thước khớp ảnh–text có thể tối ưu tiêu chí khác với việc chọn context giúp classifier. Kết quả không đủ để kết luận CLIP ranking sai, nhưng phản bác việc mặc định ranking score tốt hơn sẽ làm downstream F1 tốt hơn.

[[TABLE:coverage]]

Độ phủ là yếu tố đồng biến cần được công bố. Prefix có 4.749 caption dùng được, trong khi FirstValid/Single/Multi có 5.640; Qwen 8B và 32B có 4.211 và 4.111. Natural mask vừa thay số ảnh nhận text vừa có thể tạo tín hiệu thống kê. Do đó bảng này đánh giá pipeline caption hoàn chỉnh, chưa cô lập thuần chất lượng câu chữ. Để làm phép thử chất lượng nội dung tốt hơn, cần một mask chung trên giao các caption hợp lệ hoặc một matched availability design được chốt trước.

Cả hai Qwen vẫn thấp hơn visual_only lịch sử 0.848168. Điều đúng có thể nói là đổi nguồn caption giúp full_both tốt hơn những nguồn Florence đã thử; điều chưa thể nói là thêm text đã vượt nhánh visual. Khi so sánh với candidate_only Prefix mới, còn thay vị trí text và selector nên càng không được quy toàn bộ khoảng cách cho captioner.

[[TABLE:caption_history]]

CSV decoder LoRA cho classifier F1 0.806707, chưa vượt full_both Florence lịch sử 0.809124. Hai số là downstream metrics, không phải loss/độ chính xác của captioner. Chúng không phủ định mọi cách fine-tune, nhưng cho thấy bộ reference nhỏ và cách thích nghi đã thử chưa tạo lợi ích quan sát được trong classifier này. Chưa có đánh giá caption độc lập với người duyệt để kết luận câu mô tả đã tốt hơn hay xấu hơn về mặt ngôn ngữ hoặc grounding.

Giữa sáu classifier mới, 57 định nghĩa function/class chung có AST giống nhau trong audit, hỗ trợ nhận định đã giữ phần model/training cốt lõi. Bốn Florence CSV có hash đối chiếu local; hai Qwen có hash runtime nhưng chưa có byte CSV local độc lập ở nhóm evidence này. Source giống nhau không xác nhận toàn bộ weights, thư viện và attachment đã giống nhau. Bảng chỉ có một split/seed, không có mean ± std và không tự suy p-value từ thứ hạng aggregate.

## Bốn nghiên cứu kiến trúc với CSV cố định

[[TABLE:study_results]]

S01 đạt test F1 0.853032, cao nhất trong chuỗi Prefix candidate_only đã thử, nhưng validation F1 0.840162 chưa vượt visual control lịch sử. S02 đạt 0.849329, giảm khoảng 0,370 đpt so với S01. Chuẩn hóa giúp giải quyết sự khác biệt scale theo thiết kế nhưng chưa cho thấy tăng task score. Không suy từ kết quả này rằng L2 luôn có hại; có thể gate hoặc các siêu tham số cần điều chỉnh, nhưng chưa có sweep kiểm chứng và không nên điều chỉnh theo test.

S03 đạt F1 0.848722, thấp hơn S02 khoảng 0,061 đpt. S04 đạt 0.850717, hơn S03 khoảng 0,199 đpt nhưng vẫn thấp hơn S01 khoảng 0,232 đpt. Các biên này nhỏ so với yêu cầu xác nhận nhiều seed. S04 là sự phục hồi nhẹ trong chuỗi, không phải bằng chứng đã tìm được mô hình tối ưu hoặc alignment/semantic selection chắc chắn có lợi.

[[TABLE:study_gates]]

Sau normalization, c_gate tăng từ cỡ 0.001 ở S01 lên 0,06–0,07 ở S02–S04. Đây là phản ứng hợp lý khi thang text đổi, nhưng không được diễn giải là lượng thông tin text tăng hàng chục lần. Cần xem residual norm và can thiệp thực tế. q_gate bằng 0 theo thiết kế candidate_only, không phải một lỗi cần chỉnh tay để ép khác 0.

S02/S03 có đầy đủ predictions, history và metrics JSON. Kiểm lại Accuracy và Macro-F1 trên từng split 1.880 ảnh khớp số export. Tuy vậy runtime không hoàn toàn đồng nhất: S02 ghi Torch 2.10.0+cu128, NumPy 2.0.2; S03 ghi Torch 2.11.0+cu128, NumPy 2.1.3. Attachment code/weights có đường dẫn hoặc metadata khác, chưa có toàn bộ pretrained weight hashes để chứng minh byte-identical. Các hạn chế này cần giữ khi đánh giá chênh lệch rất nhỏ.

Paired audit S03 so với S02 trên test ghi 6 ảnh từ sai thành đúng và 8 ảnh từ đúng thành sai, với 22 prediction thay đổi. McNemar exact p khoảng 0,7905 chỉ liên quan correctness của cặp predictions này, không phải kiểm định Macro-F1 hoặc biến thiên training seeds. Số này không ủng hộ một cải thiện accuracy đã xác nhận. Không áp dụng phép kiểm định này cho S04 vì hiện chưa có prediction CSV độc lập của S04.

S01 và S04 có kết quả từ bảng notebook sáu chữ số. S04 lưu đủ 50 dòng epoch và trạng thái completed, best epoch 38; stderr vẫn chứa lỗi multiprocessing/cleanup. S01 thiếu một dòng epoch stdout dù trạng thái ghi 50. Những quan sát này được công bố để phân biệt độ đầy đủ của log với độ chính xác bảng, không tự bù history còn thiếu hoặc khẳng định run hoàn toàn sạch.

## Can thiệp trực tiếp vào fusion

[[TABLE:fusion_interventions]]

Ở S02, tắt hoặc đảo text đều cho F1 nhỉnh hơn aligned trên validation. Ở S03 xu hướng tương tự nhưng biên nhỏ. S04 có text_off F1 0.839080 so với aligned 0.838806; shuffle đạt 0.841044. Đồng thời CE của S04 tăng từ 1.174964 lên 1.190771 và 1.193906 khi tắt/đảo. F1 và CE đo các khía cạnh khác nhau: nhãn argmax có thể tốt hơn một ít dù phân bố xác suất theo CE kém hơn.

S04 tắt text làm đổi prediction khoảng 1,70%, shuffle khoảng 2,02%, nên nhánh text không hoàn toàn vô tác dụng. Tuy nhiên CI 95% của delta F1 so aligned lần lượt là [−0.004476; 0.005086] và [−0.003611; 0.007703], đều chứa 0. Không có bằng chứng rõ rằng đúng caption giúp F1 cao hơn ở checkpoint này. Các CI resample query đánh giá độ bất định trên tập validation của một checkpoint, không thay thế bất định do retrain bằng seed khác.

Đây là đối chứng inference cùng checkpoint/context, khác với full_shuffle_both được train lại trong bộ A. Một model học với text rồi tắt text lúc suy luận không tương đương visual_only được train riêng. Hai loại thí nghiệm bổ sung nhau: retraining đo ảnh hưởng xuyên suốt quá trình học, intervention giữ model cố định để kiểm tra đường sử dụng thông tin.

## Can thiệp selector của S04

[[TABLE:selector_interventions]]

Tắt semantic selector làm F1 giảm khoảng 0.000495, còn shuffle text chỉ trong selector làm giảm khoảng 0.002191. Tỷ lệ đổi context tương ứng 5,21% và 10,59%, khoảng 98 và 199 trên 1.880 query theo số hiển thị. Đây là tín hiệu phù hợp với việc semantic score có ảnh hưởng tới retrieval, nhưng cả hai CI vẫn chứa 0. Do đó chưa đủ kết luận semantic retrieval tạo lợi ích ổn định.

Sau reselect, tỷ lệ context có text hợp lệ cũng đổi: aligned khoảng 84,89%, visual-only selector 82,07% và selector shuffle 81,81%. Bởi vậy semantic intervention chưa hoàn toàn matched về selected-text availability. Cần matched coverage hoặc phân tích nhóm để tách hiệu ứng tìm context khác với việc chọn context có caption dùng được. Kết quả hiện tại nên được mô tả là tín hiệu nhỏ cần xác nhận, không phải xác nhận thành công toàn bộ module.

## Alignment học được gì

[[TABLE:alignment_fit]]

Training minibatch alignment loss ở S03 giảm từ khoảng 3.41 xuống 0.86 cuối run, cho thấy module tối ưu được mục tiêu phụ. S04 cũng giảm nhưng profile khác khi selector thay đổi. Tại best checkpoint, các chẩn đoán pool trong bảng phản ánh correspondence trên chính dữ liệu đã dùng cho alignment. Không được viết caption-group top 1 khoảng 74,69% như một retrieval accuracy trên held-out test.

S03 có matching diagnostics tốt hơn S04 ở một số chỉ số pool nhưng test classification thấp hơn. Quan sát này minh họa sự khác nhau giữa representation matching và utility cho tác vụ. Việc caption bám đúng ảnh là điều kiện hữu ích, song classifier còn phụ thuộc fusion, visual features, context selection và độ nhiễu của mô tả. Muốn quy nguyên nhân cần held-out alignment diagnostics và đối chứng giữ các yếu tố còn lại.

[[FIG:Hình 5.2 Chẩn đoán gate và matching của chuỗi kiến trúc|Bổ sung norm ratio trước và sau chuẩn hóa, alignment loss theo epoch, cùng một số query/context đúng và sai. Chỉ dùng diagnostics của checkpoint thật; không tự dựng heatmap minh họa như attention được đo.]]

## Phân đoạn ISIC

[[TABLE:isic]]

Qwen 8B full_both thấp hơn visual khoảng 0,73 đpt Dice và 0,84 đpt IoU trên test. Best validation Dice lại nhỉnh hơn, khoảng 87,55% so với 87,31%. Vì checkpoint được chọn theo validation, kết quả cho thấy lợi ích nhỏ trên validation chưa chuyển thành test gain; không đủ dữ liệu để kết luận nguyên nhân duy nhất là overfitting.

Nhánh caption sinh đủ 2.750 mô tả và coverage theo policy là 100% ở train/val/test/pool. Gate ở best validation khác 0 và semantic mix khoảng 0,5026. Vì vậy không thể giải thích điểm giảm bằng việc toàn bộ text bị mask. Tuy nhiên coverage 100% chỉ xác nhận policy, không chứng nhận mô tả đúng 100%. Một giả thuyết là text toàn cục thiếu thông tin không gian về biên; giả thuyết khác là semantic retrieval thay context nhưng chưa có lợi cho segmentation. Chưa có ablation riêng query/candidate/selector ISIC để phân biệt.

Source chính của visual và full_both có cùng decoder, loss/metric helpers, nhưng thêm text làm số trainable parameters tăng từ 6.306.754 lên 6.704.325 và thay quá trình training/selection. Cặp run chỉ có một seed, chưa có full predictions/masks export độc lập cho paired audit trong nhóm hiện tại. Không kết luận Qwen hoặc text luôn có hại cho segmentation.

So với paper, visual ISIC 85,60% Dice và 77,38% IoU cao hơn reference 85,0/76,5 về mặt số học. Đây không phải một so sánh cùng điều kiện: resolution 448 thay 256, final-token fusion thay blockwise, số seed và quy tắc early stopping khác nhau. Báo cáo không dùng mức chênh này để tuyên bố vượt TACS gốc.

[[FIG:Hình 5.3 Ví dụ phân đoạn ISIC đúng và sai|Bổ sung mỗi hàng gồm query, ground truth, visual prediction, caption prediction, context được chọn và caption. Chọn cả ca biên rõ, biên mờ và có nhiễu; ghi checkpoint, threshold 0,5 và split. Không dùng mask thật làm evidence của mô hình.]]

## Các kết quả visual lịch sử và phạm vi tái triển khai

[[TABLE:visual_history]]

Các notebook visual cho thấy pipeline đã được chạy trên nhiều dữ liệu, nhưng điều kiện chưa đồng nhất. Ba run CUB No-Context cung cấp nhiều seed ở một cấu hình; không vì vậy gán ba seed cho tất cả các thí nghiệm caption/segmentation. Cars Random/Frozen/TACS 448 có khác pool, K, shortlist và cách monitor; Cars No-Context 256 không là baseline cùng độ phân giải. Một run Random 256 có Accuracy 1,79% và F1 0,61% là kết quả bất thường cần chẩn đoán, không được chọn làm đối chứng để thổi phồng mức tăng của TACS.

SUN true-validation S/16 đạt Accuracy 74,01% và F1 73,85%; bản B/16–384 đạt 79,42% và 79,27%. Ngoài backbone/resolution, cần kiểm soát training và protocol trước khi phân bổ nguyên nhân. Một số run SUN cũ dùng monitor không tách khỏi train và có validation rất cao; chúng không được dùng làm bằng chứng khái quát hóa tương đương held-out validation. Các kết quả test được lưu không tự sửa được hạn chế của quy trình lựa chọn model.

Nhóm này được giữ trong báo cáo để thể hiện phạm vi triển khai thật và quá trình điều chỉnh, không gộp thành “TACS luôn hơn mọi baseline”. Những Tables 4–5 của paper cần matched context/optimization controls riêng. Bộ ablation text không thay thế ablation no-context/blank/duplicate/noisy hoặc gradient-only/policy-only. Đặt tên bảng theo tinh thần paper cần kèm phạm vi thu hẹp, không điền số tác giả vào ô kết quả project còn thiếu.

[[TABLE:scope]]

## Tổng hợp câu trả lời nghiên cứu

**Vị trí text:** trong nhóm đã chạy, candidate_only là lựa chọn khả quan hơn các cấu hình có query text, nhưng lợi ích so visual còn nhỏ và chưa ổn định được xác nhận. **Nguồn caption:** Qwen cải thiện full_both so Florence controls, song chưa vượt visual; model lớn hơn và coverage cao hơn không tự làm điểm tốt hơn. **Fusion và alignment:** normalization giải quyết scale theo thiết kế, auxiliary matching học được mục tiêu phụ, nhưng chưa tạo mức tăng F1 so baseline Prefix. **Selector mới:** có tín hiệu đổi retrieval và biên F1 nhỏ, chưa có CI loại 0 hoặc multi-seed confirmation.

Các kết quả không chứng minh caption vô dụng. Chúng chỉ ra rằng quality của text, availability, vị trí sử dụng, norm/semantics và utility retrieval phải được kiểm tra cùng nhau. Một mô tả chính xác có thể dư thừa với visual features; một caption thiếu hoặc lệch có thể gây nhiễu; một fusion học được có thể chọn giảm nhánh text. Những cơ chế này là giả thuyết phân tích, chưa được coi là các nguyên nhân đã chứng minh bởi aggregate metrics.

## Hạn chế và nguy cơ ảnh hưởng kết luận

Hạn chế lớn nhất là phần lớn kết quả DTD chỉ có một seed và split 1. Chưa có standard deviation hoặc độ ổn định qua mười split; không tự tạo mean/std. Nhóm caption có natural masks khác nhau, còn nhóm lịch sử thiếu hash toàn bộ dữ liệu/weights. Một số lỗi worker và version runtime khác nhau giảm mức chắc chắn khi chênh lệch chỉ vài phần nghìn F1. S04 thiếu export predictions/checkpoint để kiểm lại độc lập như S02/S03.

Dataset test đã được quan sát qua nhiều vòng phát triển. Nếu dùng các thứ hạng test này để quyết định cải tiến kế tiếp, báo cáo phải coi đây là kết quả khám phá, không còn là một xác nhận cuối hoàn toàn chưa nhìn thấy. Các vòng sau nên chốt hypothesis, CSV, masks và hyperparameters bằng train/validation trước khi mở test, hoặc dùng một split mới chưa tham gia quyết định. Không sửa policy sau khi xem test rồi gọi đó là đánh giá độc lập của cùng thí nghiệm.

Caption là weak annotation, chưa có bộ đánh giá ngôn ngữ/grounding độc lập đầy đủ. Tập 188 reference không bảo đảm phủ hết biến thiên DTD; nguồn teacher cần được lưu chính xác và không nhận nhãn thật qua tên file. Các điểm test của classifier không thay thế caption quality. Với ISIC, chưa kiểm định lâm sàng, không dùng mô hình demo để chẩn đoán hoặc đưa ra quyết định y khoa.

Chưa có benchmark latency, VRAM, FLOPs và caption generation time được đo cùng phần cứng/protocol giữa mọi method. Vì vậy báo cáo không đưa bảng tốc độ suy đoán, không coi 32B kém hiệu quả chi phí chỉ từ một chỉ số F1 và tên model. Có thể ghi GPU/VRAM thực tế khi log có, nhưng không đồng nhất memory total với peak allocated hoặc chi phí training đo được.

# CHƯƠNG 6 DEMO VÀ KHẢ NĂNG TRIỂN KHAI

## Mục đích và hai chế độ

Web demo minh họa phân loại SUN397 với backend và frontend. Trang TACS visual mô tả query image cùng context image; trang M-TACS bổ sung caption query/candidate và CLIP. Người dùng chọn hoặc kéo thả ảnh rồi gửi yêu cầu dự đoán. Hai trang phân biệt kiến trúc, không phải hai benchmark mới.

Giao diện có vị trí hiển thị caption query, ảnh context được chọn cùng caption tương ứng, nhãn dự đoán và attention cho query/candidate theo vai trò. Khi chưa nối checkpoint/pool đúng kiến trúc, các ô dự đoán/evidence để trống. Không sinh caption, confidence, ảnh candidate hay heatmap giả để tạo cảm giác inference đã chạy.

## Backend và hợp đồng dữ liệu

Backend FastAPI kiểm tra định dạng, kích thước và hash ảnh; upload hợp lệ không đồng nghĩa đã inference. Endpoint phân loại có tham số chọn model. Source hiện tại chưa nối checkpoint, trả HTTP 503 với prediction null và evidence chưa kết nối, không trả kết quả mô phỏng.

[[TABLE:api]]

Để suy luận thật, mỗi model cần checkpoint đúng class definition, class_names đúng thứ tự, backbone/preprocessing, pool và ánh xạ ID. Nhánh M-TACS cần thêm captioner, policy, CLIP cùng image/text bank và validity masks. Không nạp một best.pt bất kỳ chỉ vì extension giống nhau; notebook architecture04 DTD hoặc checkpoint segmentation không thể thay cho SUN397 classifier 397 lớp.

## Quy trình cho một ảnh mới ngoài dataset

Một query mới đi qua kiểm tra ảnh và preprocessing đúng lúc train. Với model có query-caption branch, hệ thống sinh caption theo đúng captioner/prompt/policy đã dùng và mã hóa CLIP. Với candidate_only thì không cần caption query; các caption candidate đã có trong bank. Selector chấm điểm với pool training cố định, loại self nếu nhận diện trùng ID/ảnh theo quy tắc được hỗ trợ, rồi lấy ảnh và caption của cùng candidate để forward.

Logits đi qua softmax cho thứ hạng trong 397 lớp đã biết, không phải một cơ chế open-set tự nhận mọi loại ảnh ngoài thế giới. Ảnh ngoài phân bố vẫn có thể nhận confidence cao; demo nên thể hiện giới hạn này và không gán confidence như xác suất model chắc chắn đúng trong mọi tình huống. Test label không tồn tại ở inference; mọi nhãn thật dùng chấm điểm phải được cung cấp riêng sau dự đoán.

## Hiển thị attention và giới hạn diễn giải

Thiết kế giao diện dành hai vùng attention: classifier trên query và selector trên candidate. Chúng phản ánh hai vai trò khác nhau như động cơ Figure 4 của paper, không đồng nhất semantic cosine CLIP với spatial attention. Trích attention cần dùng đúng checkpoint/layer/head và phương pháp đã công bố, sau đó overlay lên đúng ảnh đã được model xử lý.

Hiện các trường attention có image_url null và trạng thái chưa nối checkpoint. Khi hoàn thiện, phải kiểm tra alignment giữa crop/resolution và overlay, cùng ID context đang hiển thị. Attention chỉ là một chẩn đoán, không tự giải thích causal model dựa vào đâu. Muốn chứng minh một vùng ảnh cần thêm perturbation/occlusion controls; không dùng heatmap minh họa như bằng chứng cơ chế.

[[FIG:Hình 6.1 Giao diện demo TACS và M TACS|Bổ sung ảnh chụp hai trang, khu vực upload, query/context caption và trạng thái model. Nếu chưa inference thật, giữ nhãn chưa kết nối và ô attention trống trong hình chụp.]]

## Tái chạy và chuyển giao

Một gói chuyển giao tối thiểu gồm notebook source, protocol JSON, environment, split/pool manifests, captions.csv và hash, history, val/test metrics, predictions và checkpoint tốt nhất. Khi báo cáo chỉ dùng output, không cần thực thi lại notebook để tạo số mới. Khi người khác tái chạy, họ phải giữ cấu hình và được biết các dependency offline, đường dẫn asset và các khác biệt runtime còn tồn tại.

Gói export được tách theo run. Cache caption chỉ tái dùng khi image/model/prompt/policy identity phù hợp; split, pool và training state vẫn phải riêng. Template chưa chạy và output completed không ghi đè lên nhau. Không thêm test vào train hoặc thay CSV rồi giữ nguyên experiment identity.

# CHƯƠNG 7 KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## Những phần đã thực hiện

Đề tài đã xây dựng pipeline học với ảnh context, các thành phần selector/task network, dữ liệu pool và quy trình evaluation trên một số dataset. Source visual thể hiện token concatenation và hai hướng tối ưu, còn nhánh DTD được công bố rõ là adaptation categorical-policy. Phân đoạn ISIC tạo được mask và test Dice/IoU thực tế. Các khác biệt so paper về backbone, resolution, fusion, loss và ngân sách được ghi thành một phần của phương pháp thay vì ẩn sau tên TACS.

Phần mở rộng đi từ caption query, candidate fusion/selector tới một bộ tám ablation, các nguồn Florence/LoRA/grounded/Qwen và chuỗi chuẩn hóa–alignment–semantic selection. Engineering offline, cache/resume, hash và web demo hỗ trợ vận hành và truy vết. Các cải tiến không phải tất cả đều tăng điểm, nhưng đã được biến từ ý tưởng thành những thí nghiệm có câu hỏi và kết quả cụ thể.

## Kết luận định lượng chính

Trong bộ tám DTD ablation, candidate_only đạt 0.850321 test Macro-F1 so visual_only 0.848168, một mức hơn nhỏ chưa được xác nhận nhiều seed. Các cấu hình dùng text query và full_both chưa vượt visual. Trong so sánh nguồn caption full_both, Qwen 8B/32B đạt 0.821445/0.820788, tốt hơn Prefix 0.810651 nhưng chưa làm text có lợi hơn visual control. Kết quả không hỗ trợ giả định cứ captioner lớn hơn hoặc coverage cao hơn là classifier tốt hơn.

Với Prefix cố định, baseline candidate_only S01 đạt 0.853032. Chuẩn hóa S02, alignment S03 và selector S04 đạt 0.849329, 0.848722 và 0.850717. Auxiliary alignment cải thiện fit mục tiêu matching nhưng chưa chuyển thành tăng F1; semantic selector có tác động lựa chọn context và tín hiệu nhỏ cần xác nhận. ISIC full_both Qwen cũng chưa cải thiện test so visual. Nhìn tổng thể, vấn đề hiện tại không chỉ nằm ở nguồn caption mà còn ở cách học và sử dụng thông tin đa phương thức.

## Ý nghĩa đối với mục tiêu học phần

Theo mục tiêu xây dựng lại cấu trúc trong paper và đề xuất một vài cải tiến, đề tài đã có phạm vi triển khai và kiểm nghiệm mở rộng đáng kể. Tuy vậy báo cáo không tuyên bố tái lập nguyên bản toàn bộ TACS hoặc hoàn thành mọi ablation Tables 2–5. Phần nền cần được đánh giá theo việc ánh xạ đúng module, xác nhận đường gradient/pairing và đối chứng visual cùng protocol; phần caption được đánh giá như một nghiên cứu mở rộng riêng.

Kết quả âm và không chắc chắn vẫn có giá trị: chúng ngăn việc chọn mô hình chỉ từ một điểm test, chỉ ra nguy cơ nhầm mask với chất lượng caption và phân biệt matching objective với task utility. Báo cáo giữ những giới hạn này để kết luận phù hợp với bằng chứng, thay vì chỉ tuyển các run cao điểm hoặc gán tên “cải tiến” như một bảo đảm hiệu quả.

## Công việc tiếp theo được ưu tiên

Ưu tiên trước hết là chốt một bộ No-Context, Frozen Similarity và TACS cùng protocol trên dataset trọng tâm, kiểm tra self-exclusion và cả hai gradient paths nếu muốn xác nhận hybrid reproduction. Tiếp theo là export đầy đủ S04 và thực hiện nhiều seed hoặc split độc lập cho một số cấu hình đã chọn bằng validation. Các bước này đem lại giá trị kiểm chứng cao hơn mở thêm nhiều captioner chưa có đối chứng.

Về text, nên dùng matched masks, caption review độc lập và các control tách fusion khỏi selector. Nếu tiếp tục kiến trúc, token-level fusion/resampling, Partial OT và adaptive pooling là các hướng cần triển khai từng yếu tố, có capacity/budget controls. Captioner fine-tune nên có tập tham chiếu nguồn rõ, kiểm duyệt và chia train/validation cho caption task. Học liên tục hoặc demo inference thật chỉ được đưa vào kết quả sau khi có protocol và artifact tương ứng.

# TÀI LIỆU THAM KHẢO

[1] J. Guo, E. Konuk, F. Strand, C. Matsoukas và K. Smith. Learning What Helps: Task-Aligned Context Selection for Vision Tasks. CVPR 2026. Bản paper lưu trong project, các mục 3–4 và Tables 2–5. Bản arXiv: https://arxiv.org/abs/2512.00489.

[2] Meta FAIR. DINOv3 reference PyTorch implementation and pretrained models. Kho mã chính thức: https://github.com/facebookresearch/dinov3. Truy cập ngày 27/10/2026.

[3] A. Radford và cộng sự. Learning Transferable Visual Models From Natural Language Supervision. 2021. https://arxiv.org/abs/2103.00020.

[4] Microsoft. Florence-2-base model card và tài liệu task prompts. https://huggingface.co/microsoft/Florence-2-base. Truy cập ngày 27/10/2026. Cách nạp runtime của project được khóa riêng theo checkpoint/dependency, không suy từ model card hiện tại.

[5] Qwen Team. Qwen3-VL-8B-Instruct model card. https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct. Truy cập ngày 27/10/2026.

[6] Qwen Team. Qwen3-VL-32B-Instruct model card. https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct. Truy cập ngày 27/10/2026.

[7] M. Luong, K. Nguyen, N. Ho, R. Haf, D. Phung và L. Qu. Revisiting Deep Audio-Text Retrieval Through the Lens of Transportation. 2024. https://arxiv.org/abs/2405.10084.

[8] C. Huynh, M. Luong và A. Shrivastava. Efficient and High-Fidelity Omni Modality Retrieval. CVPR 2026; arXiv v2 ngày 24/03/2026. https://arxiv.org/abs/2603.02098.

[9] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed và A. Vedaldi. Describing Textures in the Wild. CVPR 2014. Trang DTD và official splits: https://www.robots.ox.ac.uk/~vgg/data/dtd/.

[10] ISIC Challenge. ISIC 2017 Challenge, Part 1 Lesion Segmentation. Mô tả training/validation/test: https://challenge.isic-archive.com/landing/2017/. Truy cập ngày 27/10/2026.

[11] E. J. Hu và cộng sự. LoRA: Low-Rank Adaptation of Large Language Models. 2021. https://arxiv.org/abs/2106.09685.

[12] Bộ notebook và export thực nghiệm của đề tài TACS/M-TACS. Nguồn A01–A08, C01–C08, S01–S04, I01–I02, U01 và visual lịch sử được liệt kê trong Phụ lục A. Số liệu tổng hợp đến ngày 27/10/2026; nguồn output và độ chính xác được ghi tại từng nhóm bảng.
