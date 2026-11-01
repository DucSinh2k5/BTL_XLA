# Tổng hợp các hướng mở rộng M-TACS so với paper TACS

Cập nhật: 14/10/2026. Phạm vi: những ý tưởng và notebook SUN397/DTD đã trao đổi trong hội thoại này. M-TACS ở đây là biến thể dùng captions, không phải mô hình TACS gốc của tác giả.

## Mốc tham chiếu từ paper

[Learning What Helps: Task-Aligned Context Selection for Vision Tasks, mục 4.2 và bảng 2](https://arxiv.org/pdf/2512.00489) dùng ảnh query và **một** ảnh context do selector chọn từ pool cố định gồm 20% ảnh train. Selector và mạng tác vụ dùng DINOv3 ViT-S/16. Paper huấn luyện 100 epoch, báo cáo trung bình ba random seed, và không dùng caption query hay caption candidate trong mô hình phân loại.

Trong phần phân loại, TACS gốc ghép patch embeddings của hai ảnh trước các transformer block và học selector bằng cả nhánh khả vi (Gumbel-Softmax) lẫn nhánh thưởng/chính sách. Các notebook M-TACS giữ ý tưởng chọn context theo tác vụ, nhưng thêm CLIP text embeddings/gate, thay backbone hoặc số context ở từng bản; vì thế chúng là **biến thể nghiên cứu**, không phải bản sao nguyên cấu hình paper.

| Dataset | TACS gốc | No-context trong paper | Metric paper |
| --- | ---: | ---: | --- |
| SUN397 | 71,8% | 68,0% | Top-1 accuracy |
| DTD | 77,7% | 76,1% | Top-1 accuracy |

Đây là **accuracy**, không phải macro-F1. Các số M-TACS dưới đây không thể coi là so sánh trực tiếp với paper nếu backbone, split, seed, cách chọn checkpoint, caption pipeline hoặc metric khác nhau.

## Các mốc đã thử và kết quả

| Mốc | Thay đổi so với TACS gốc | Kết quả đã có | Nguồn và mức độ chắc chắn |
| --- | --- | --- | --- |
| M-TACS SUN397 cũ, ViT-S/16 | Thêm caption của **query** vào nhánh downstream bằng CLIP embedding và gate; selector vẫn dùng ảnh. Một context image. | `test_accuracy = 0.7612` (76,12%); `test_macro_f1 = 0.7602` (76,02%). `test_valid_caption_rate = 0.6055`. | Output lưu trong [`mtacs-train-captions.ipynb`](./mtacs-train-captions.ipynb), cell test. Đây là run cũ, không cùng cấu hình với các run ViT-L/16 bên dưới. |
| M-TACS SUN397, ViT-L/16, query caption | Đổi backbone sang DINOv3 ViT-L/16; ảnh query + một ảnh context được chọn; **chỉ caption của query** được fuse vào downstream. | `test_macro_f1 = 0.8237` (82,37%). Chưa có accuracy/log gốc của run này trong workspace. | Số bạn báo trong hội thoại. Notebook cấu hình: [`mtacs-train-captions-vitl16.ipynb`](./mtacs-train-captions-vitl16.ipynb); output trong file hiện tại đã được xóa. |
| Candidate captions trong selector, top-3 | Tính điểm chọn candidate từ visual score và độ tương đồng CLIP giữa caption query/candidate; chọn ba ảnh context không lặp. **Bản top-3 đầu tiên chưa fuse caption candidate vào downstream**, dù có dùng nó để chọn ảnh. | Bạn báo `test_macro_f1 = 0.8108` (81,08%) cho bài “dùng cả text cho context”. Chưa có log/checkpoint để xác nhận chính xác notebook tạo ra số này. | [`mtacs-train-captions-vitl16-multimodal-top3.ipynb`](./mtacs-train-captions-vitl16-multimodal-top3.ipynb) là notebook top-3 được tạo ở giai đoạn đó. Không gán `0.8108` cho bản fuse caption context khi chưa kiểm tra log. |
| Fuse cả query caption và selected context captions, top-3 | Ngoài selector visual + semantic, đưa caption của ba candidate được chọn vào downstream; caption không hợp lệ được mask. | **Chưa có metric được cung cấp/xác minh.** | [`mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb`](./mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb). |
| Fuse cả query caption và selected context caption, top-1 | Giữ đúng **một** ảnh context như yêu cầu sau khi làm rõ; caption của ảnh đó cùng caption query đều vào downstream. Candidate captions cũng tham gia selector. | **Chưa có metric được cung cấp/xác minh.** | [`mtacs-train-captions-vitl16-context-text.ipynb`](./mtacs-train-captions-vitl16-context-text.ipynb). Không đồng nhất với notebook top-3. |
| Chuyển sang DTD | Dùng official `trainN.txt`, `valN.txt`, `testN.txt`; tạo/resume `captions.csv` ngay trong notebook rồi huấn luyện M-TACS query-caption top-1. | **Chưa có `test_accuracy` hoặc `test_macro_f1` được cung cấp.** | [`mtacs-train-captions-vitl16-dtd.ipynb`](./mtacs-train-captions-vitl16-dtd.ipynb). Lần trao đổi gần nhất mới xác nhận checkpoint Florence-2 đang ở định dạng legacy; chưa có log train/test thành công. |

Hai số bạn báo `0.8237` và `0.8108` lệch **0.0129 macro-F1**, tức **1,29 điểm phần trăm**. Đây là chênh lệch quan sát được, **không chứng minh candidate text làm giảm chất lượng**: bản query-only là top-1, còn bản candidate-text được tạo trước đó là top-3; phiên bản prompt caption và cách selector hoạt động cũng khác. Chưa có log gốc, nhiều seed hay xác nhận cùng split cho hai run này.

## Các hướng cải tiến và kiểm soát chưa có metric riêng

1. **Tạo và lọc caption:** [`generate_sun397_captions.py`](./generate_sun397_captions.py) dùng Florence-2-base với `<MORE_DETAILED_CAPTION>`, rút về 5-8 từ; kiểm tra từ khóa tên lớp và cosine CLIP ViT-B/32 (ngưỡng mặc định `0.30`). Caption bị đánh dấu không được dùng làm text feature, nhưng ảnh vẫn có thể làm query/context. Đây là xử lý dữ liệu, chưa phải một ablation có accuracy riêng. CLIP là heuristic, không phải LLM đọc hiểu và chứng minh “không leak”.
2. **Pool, query và tránh tự ghép:** Candidate pool là tập con 20% của train, không phải val/test. Ảnh trong pool vẫn có thể xuất hiện như query train; khi chọn context cần loại chính ảnh query đó. Notebook DTD và các bản `context-text` sau có xử lý loại self-candidate. Cần kiểm tra điều này trong từng run SUN cũ trước khi so sánh.
3. **Inference ảnh ngoài dataset:** Cần tạo caption của ảnh mới bằng cùng VLM/prompt, encode bằng cùng CLIP, rồi chọn context từ pool train đã đóng băng và phân loại. Không có ground truth thì không tính được test metric. Đây là workflow được thảo luận, chưa có kết quả đo trong hội thoại.
4. **Học liên tục sau mỗi lần test:** Suy luận rồi đưa ảnh vào train/pool là một giao thức online/continual-learning mới, khác TACS benchmark tĩnh. Nếu không có nhãn thật, dùng pseudo-label có thể khuếch đại lỗi; không cập nhật từ nhãn test khi đang báo cáo kết quả benchmark chuẩn. Chưa có notebook/thực nghiệm/metric cho hướng này.
5. **Thời gian tạo caption 20 phút so với 7-8 giờ:** Không phải metric phân loại. Cần đối chiếu số ảnh thực sự sinh mới (`cached/queued`), model/weights, task prompt, batch và số dòng CSV trước khi suy ra hai run tương đương. Chưa đủ dữ liệu để kết luận.

## 1. Từ ảnh đến caption và kiểm soát leak

Bạn đã hỏi hai mô hình nào tham gia, chúng có phải Transformer/LLM “thông minh” hay không, tại sao file mới chạy nhanh, và có thể chỉ sửa ngưỡng để được điểm test cao hay không. Có ba bước khác nhau:

```text
Ảnh -> Florence-2-base + <MORE_DETAILED_CAPTION> -> raw_caption
raw_caption -> rút 5-8 từ + so khớp tên lớp -> lexical_leakage_detected
final_caption -> CLIP text encoder -> so với vector tên lớp -> semantic_leakage_detected
```

- **Florence-2** là mô hình vision-language dựa trên Transformer dùng để sinh mô tả từ pixel ảnh. Trong các script/notebook ở đây, nó không nhận nhãn đúng, tên lớp của chính ảnh hay đáp án test làm input. Đây không phải API của một LLM tổng quát hiện đại; khả năng mô tả phụ thuộc checkpoint, prompt và dữ liệu.
- **CLIP ViT-B/32** ở bước kiểm tra không sinh câu trả lời dạng “có/không leak” như một LLM thứ hai. Nó mã hóa caption và các prompt tên lớp thành vector, tính cosine similarity và gắn cờ theo ngưỡng `0.30`. Một caption hàm ý tên lớp nhưng không vượt ngưỡng có thể bị bỏ sót; câu vô hại vẫn có thể bị gắn cờ. So khớp từ khóa cũng không bắt hết từ đồng nghĩa. Cả hai phép kiểm tra chỉ **giảm rủi ro**, không bảo đảm không leak.
- Một caption bị đánh dấu `caption_valid=False` thì text embedding được vô hiệu hóa; ảnh vẫn thuộc tập dữ liệu và vẫn có thể là context hình ảnh. Cần phân biệt **tỷ lệ caption hợp lệ** với **độ chính xác phân loại**. Run ViT-S/16 cũ lưu `test_valid_caption_rate=0.6055`, `test_leakage_detected_rate=0.3929`, `test_mean_gate=0.0163`. Các số này không chứng minh riêng CLIP audit làm mô hình tốt hơn.
- Ngưỡng semantic là **hyperparameter của bộ lọc dữ liệu**. Hạ hoặc nâng ngưỡng làm thay đổi những caption được đưa vào train *và* test. Chọn ngưỡng dựa trên `test_accuracy`/`test_macro_f1` sẽ làm bẩn phép đánh giá; nên chọn bằng phân tích caption thủ công và validation, rồi khóa ngưỡng trước lần đánh giá test cuối.
- File CSV mới có các cột như `image_id`, `split`, `raw_caption`, `final_caption`, `caption_valid`, hai cờ leak, điểm CLIP, `vlm_model` và `prompt_version`. Cache theo `image_id` cho phép tiếp tục khi ngắt. File caption cũ và mới có thể khác schema/prompt, nên phải kiểm tra version và số dòng thay vì gộp ngầm.

**Về 20 phút so với 7-8 giờ:** Một lần chạy nhanh có thể đã resume từ CSV/cache, chỉ xử lý `--limit`, dùng batch lớn hơn, dùng checkpoint/prompt khác, hoặc bỏ qua một công đoạn tốn thời gian. Tốc độ một mình không cho biết caption đã được sinh cho đủ ảnh, đúng mô hình và cùng mức audit. Cần đối chiếu `cached`, `queued`, tổng dòng CSV, `vlm_model`, `prompt_version` và vài caption mẫu. Trong hội thoại chưa có log đầy đủ để quy nguyên nhân cụ thể hay gán một metric accuracy cho thay đổi này.

## 2. Candidate pool, train query và tự chọn chính mình

Paper TACS lấy ngẫu nhiên **20% ảnh train ban đầu** làm candidate pool cố định để phục vụ cả train và evaluation; đây không phải val/test và không tự động là 20% “bị rút khỏi train”. Các notebook SUN chọn candidate từ `train_df` sau khi tách validation; vì vậy tỷ lệ 20% có mẫu số là **train thực sự dùng để huấn luyện**, cần ghi rõ khi báo cáo. Ảnh nằm trong candidate pool vẫn có thể được dùng như query train.

Khi một ảnh trong pool trở thành query, quy trình đúng cho phiên bản này là:

1. Vẫn dùng ảnh đó và nhãn của nó để tính loss của **query train**.
2. Khi xếp hạng candidate, mask chính `image_id` của query để không chọn nó làm context của nó; nếu còn quan hệ trùng lặp/patient-group thì cũng loại theo nhóm tương ứng.
3. Chọn context từ những candidate train còn lại. Nhãn của ảnh context không được đưa thẳng vào model ở lúc inference; nhãn ground truth của query chỉ dùng để tính loss/metric.
4. Khi đánh giá val/test, giữ nguyên pool train và trọng số mô hình. Không đưa ảnh val/test vào pool chỉ để lấy điểm cao.

Về mã hiện tại: bản DTD và các bản `context-text` sau có mask self-candidate; bản top-3 prototype và các run SUN cũ cần kiểm tra lại điều này trong đúng checkpoint đã chạy. Không thể suy ra đã chống self-match chỉ từ tên notebook. Nếu pool được lấy trước/sau bước train-val khác nhau thì đó cũng là một biến thực nghiệm.

## 3. Caption nào đi vào đâu?

Đây là chỗ dễ bị gọi chung là “dùng text context” mặc dù có **hai vị trí sử dụng text**:

| Thành phần | Dùng text để làm gì? | Có trong bản nào? |
| --- | --- | --- |
| Caption của query ở downstream | Chiếu CLIP embedding rồi dùng gate cộng vào đặc trưng thị giác trước classifier. | Bản M-TACS query-only, top-3 prototype và các bản `context-text`. |
| Caption của candidate ở selector | Tính similarity caption query-candidate và trộn với visual score để **chọn ảnh context**. | Top-3 prototype và các bản `context-text`. |
| Caption của context đã chọn ở downstream | Lấy đúng caption tương ứng `selected_indices`, mask caption không hợp lệ, fuse cùng đặc trưng query/context trước classifier. | Chỉ các notebook có hậu tố `context-text`. |

Luồng của bản query-only ViT-L/16:

```text
Query image ----> visual selector ----> 1 selected candidate image
      |                                  |
      +------------ visual task network -+--> visual features
Query caption --> frozen CLIP --> projection + gate --> classifier
```

Caption của ảnh context **không** đi vào selector hoặc classifier trong bản này. Lý do ban đầu là giữ phần chọn context hình ảnh gần TACS gốc, rồi đo riêng tác dụng text của query. Đây là quyết định ablation, không phải một quy tắc cấm dùng caption context.

Luồng của bản top-3 prototype: caption query/candidate tham gia **selector**, selector chọn ba ảnh; downstream nhận ba ảnh và caption của query, **không nhận ba caption context**. Vì vậy một kết quả của prototype này không phải kết quả “fuse query + context captions”.

Luồng của bản bạn thực sự muốn sau khi làm rõ: **một ảnh context**, caption query và caption của đúng ảnh context đó đều được dùng; text candidate vừa tham gia selector vừa tham gia fusion downstream. Đó là [bản top-1 context-text](./mtacs-train-captions-vitl16-context-text.ipynb). Cũng có [bản top-3 context-text](./mtacs-train-captions-vitl16-multimodal-top3-context-text.ipynb) để thử số lượng context, nhưng nó không phải yêu cầu top-1 ban đầu. Dùng caption context không tự vi phạm protocol nếu caption được tạo từ ảnh train, không chứa nhãn/metadata cấm và cùng pipeline khả dụng khi inference; vẫn phải audit leak và mask caption lỗi.

**Đọc lại hai kết quả bạn báo:** `0.8237` là `test_macro_f1` của bài query-text ViT-L/16 theo lời bạn. `0.8108` là `test_macro_f1` của bài bạn gọi là có text context, nhưng trong hội thoại chưa có output/checkpoint để xác định đó là prototype selector-only top-3 hay bản context-fusion. Không có cơ sở ghi rằng bản top-1 context-fusion đạt `0.8108`. Chênh lệch 1,29 điểm phần trăm không phải thí nghiệm kiểm soát một biến vì có thể đổi cả `K`, selector, prompt version, self-exclusion và các yếu tố khác.

## 4. Train, test và ảnh mới ngoài dataset

- **Trong train:** Mỗi query có ảnh, nhãn và (nếu hợp lệ) caption của chính query; selector tìm context trong pool train. Với bản context-text, lấy thêm caption tương ứng của context. Loss phân loại dùng nhãn query; các thành phần selector được tối ưu theo cơ chế của notebook.
- **Trong validation/test:** Dùng cùng mô hình đã train, cùng pool train, cùng CLIP text encoder và quy tắc `caption_valid`. Caption của ảnh val/test có thể được tạo trước từ pixel của chính ảnh đó, miễn là không dùng nhãn val/test hay kết quả test để chỉnh mô hình/prompt/threshold. Chọn context rồi dự đoán, **không cập nhật trọng số hoặc pool**. `captions.csv` là kho mô tả đã tính sẵn, không phải đáp án nhãn.
- **Với một ảnh hoàn toàn mới:** Tiền xử lý ảnh đúng như train; chạy cùng Florence-2 và prompt để lấy caption, kiểm tra leak, encode bằng CLIP hoặc dùng zero/mask nếu caption không hợp lệ; chọn ảnh/caption context từ pool train đã lưu; chạy classifier, ánh xạ chỉ số dự đoán qua `class_names`. Muốn đánh giá accuracy/F1 thì phải có ground truth cho một tập ảnh mới, không phải chỉ một ảnh không nhãn.

Hướng “test xong lấy ảnh đó học tiếp” bạn hỏi là **continual/online learning**, không còn là inference tĩnh của paper. Có ít nhất ba giao thức khác nhau: (a) nhận nhãn thật sau dự đoán rồi fine-tune, (b) dùng pseudo-label do model tự đoán, (c) chỉ thêm ảnh chưa nhãn vào pool truy xuất mà không cập nhật classifier. Chúng phải được tách thành thí nghiệm riêng, có thứ tự thời gian, ngân sách nhãn, replay/kiểm soát quên và metric prequential (chấm **trước** khi cập nhật). Dùng nhãn test để cập nhật rồi vẫn báo điểm trên chính tập test là leakage. Hiện chưa có kết quả thực nghiệm cho hướng này.

## 5. DTD: thử khả năng chuyển sang dataset khác

Bạn hỏi DTD là gì, lấy ở đâu, liệu split đã chuẩn chưa, và muốn notebook DTD tự tạo `captions.csv`. [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) là Describable Textures Dataset với nhãn thuộc tính texture. Notebook DTD dùng ảnh trong `/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd/images` và file chia tập trong `.../labels`; `DTD_SPLIT=1` có nghĩa đọc `train1.txt`, `val1.txt`, `test1.txt`. Các giá trị 1..10 là **10 cách chia chính thức**, không phải seed huấn luyện.

Notebook tự đọc official split, lập pool 20% từ train, tạo caption cho 5.640 ảnh (resume theo batch), audit bằng CLIP, cache text embeddings, train rồi test. Việc chuyển từ SUN397 sang DTD là **đánh giá khả năng tổng quát hóa** của cùng ý tưởng, không phải mặc nhiên là tăng điểm SUN397. Chưa có DTD test metric nào bạn gửi để báo cáo.

Các lỗi Kaggle đã gặp thuộc **môi trường/model**, chưa phải kết quả huấn luyện:

1. Lúc đầu `microsoft/Florence-2-base` bị gọi qua Hugging Face và lỗi DNS khi Kaggle offline; chỉ có DTD và CLIP input không cung cấp weights Florence-2.
2. Đường dẫn notebook output đúng bạn tìm được là `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`, không phải đường dẫn owner khác từng thử. Checkpoint ở đó có `config.json`, `model.safetensors` và `auto_map` cho `AutoModelForCausalLM`, tức dạng legacy Microsoft.
3. Runtime của bạn báo `transformers 5.0.0`; notebook phân biệt checkpoint legacy (thiết kế cho 4.49.x) và checkpoint Florence-2 native cho 5.x. Cần **ghép đúng phiên bản thư viện với định dạng checkpoint** và đặt đúng `FLORENCE_MODEL_REF`; chỉ sửa đường dẫn sẽ không giải quyết xung đột API. Môi trường cài đặt cũng có thể cần các phụ thuộc `accelerate`, `einops`, `timm`.
4. `DTD_SPLIT is not defined` nghĩa là cell cấu hình chưa chạy thành công trước cell dataset. Lỗi `NameError: kaggle is not defined` trong cell kiểm tra xuất phát từ việc một đường dẫn được dán như mã Python thay vì chuỗi.

Các lỗi trên không sinh ra metric DTD. Chỉ sau khi caption cell hoàn tất, kiểm tra số caption hợp lệ và huấn luyện/evaluation thành công mới điền được kết quả DTD vào bảng trên.

## DTD và độ công bằng khi đối chiếu paper

Theo [DTD Oxford](https://www.robots.ox.ac.uk/~vgg/data/dtd/), DTD có 47 lớp và 5.640 ảnh; mỗi official split có 1.880 train, 1.880 val, 1.880 test. Notebook hiện mặc định `DTD_SPLIT=1`; pool 20% train là 376 ảnh. Đây là **một split chính thức**, nhưng [giao thức DTD của Oxford](https://www.robots.ox.ac.uk/~vgg/data/dtd/eval.html) báo cáo trung bình 10 split. [Paper TACS](https://arxiv.org/pdf/2512.00489) nói dùng standard split và ba seed, nhưng phần phương pháp không nêu rõ DTD được tổng hợp trên cả 10 split hay split nào. Vì vậy không tự động coi một lần `DTD_SPLIT=1` là tái lập đúng `77,7%`.

Để kết luận text candidate có giúp hay không, nên chạy ablation **cùng SUN397 split, cùng caption CSV/prompt, ViT-L/16, candidate pool, K=1, seed và quy tắc chọn checkpoint**:

| Run đề xuất | Selector | Downstream text | Mục đích |
| --- | --- | --- | --- |
| A | Visual | Query caption | Mốc so sánh query-only. |
| B | Visual + query/candidate text | Query caption | Đo riêng tác dụng text candidate trong selector. |
| C | Như B | Query + selected candidate caption | Đo thêm tác dụng fuse context text. |
| D (tùy chọn) | Như B/C, K=3 | Giữ cấu hình text nhất quán | Đo riêng tác dụng số lượng context. |

Báo cáo `test_accuracy` và `test_macro_f1` tách riêng, mean/std qua nhiều seed. Không chỉnh ngưỡng leak, chọn checkpoint hay chọn biến thể dựa trên test; dùng validation cho các quyết định đó. Khi so paper, chỉ đối chiếu **accuracy với accuracy**, và ghi rõ M-TACS ViT-L/16 + captions là mở rộng, không phải tái lập TACS ViT-S/16.

## Nhật ký đầy đủ các câu hỏi và hướng xử lý

Phần này ghi lại **cả những gì chưa chạy thành thí nghiệm**. “Đã giải thích” không đồng nghĩa “đã xác nhận bằng metric”; “đã tạo notebook” cũng không đồng nghĩa “Kaggle đã chạy xong”.

### A. Tạo dữ liệu văn bản và chất lượng caption

1. **Vì sao generator cũ tạo được CSV, generator mới báo lỗi mạng?** Bạn gặp lỗi DNS khi gọi Hugging Face. Hướng xử lý là dùng checkpoint Florence-2 local đã attach hoặc, nếu được phép có Internet, tải đúng model; không thể giải quyết bằng cách đổi tên CLIP/SUN397 input. **Trạng thái:** đã xác định nguyên nhân, không có metric phân loại riêng.
2. **Các Kaggle Inputs cũ có đủ để chạy generator mới không?** Ảnh SUN397 và CLIP không thay được weights Florence-2. Cần kiểm tra thư mục model thực, `config.json`, processor/tokenizer và weights. **Trạng thái:** bạn tìm được checkpoint ở notebook output, nhưng môi trường/phiên bản Transformers là vấn đề tiếp theo.
3. **Có `captions.csv` rồi thì làm gì để train?** Gắn CSV làm Kaggle Input hoặc đặt đúng `CAPTIONS_INPUT_PATH`; kiểm tra `image_id` khớp partition, `split`, `caption_valid`, schema và `prompt_version`; tạo/cache frozen CLIP embeddings; chạy các cell cấu hình, dataset, model, train, test theo thứ tự. Không phải sinh lại mọi caption nếu CSV và pipeline tương thích. **Trạng thái:** notebook SUN tách caption preprocessing khỏi train; các bản notebook mới không chứa log của chính run bạn chạy.
4. **Vì sao run cũ 7-8 giờ còn run mới khoảng 20 phút?** Đề xuất kiểm toán số ảnh `queued` thật, resume cache, `--limit`, batch, tốc độ GPU, model/prompt và số dòng CSV. Không suy ra run ngắn là “thông minh hơn” hoặc “kém hơn”. **Trạng thái:** chưa có log đủ để chốt nguyên nhân.
5. **Dùng model nào để viết và kiểm tra leak? Có giống LLM không?** Florence-2 sinh mô tả; CLIP text encoder + luật từ khóa gắn cờ rủi ro, không phải LLM thứ hai suy luận tự do. **Đề xuất:** đọc thủ công mẫu caption bị gắn cờ/bỏ sót và lưu tỷ lệ hợp lệ theo split. **Trạng thái:** pipeline đã có; chưa có đánh giá thủ công định lượng.
6. **Chỉ chỉnh ngưỡng CLIP tới khi điểm test cao/ổn định có chứng minh model tốt không?** Không. **Đề xuất:** khóa test, chọn ngưỡng trên validation và một tập audit độc lập, báo độ nhạy theo vài ngưỡng đã định trước. **Trạng thái:** chưa có sweep được báo cáo; tránh tối ưu trực tiếp trên test.

### B. Caption của query, caption của context và TACS

7. **Caption chỉ dùng train hay test cũng cần?** Bản M-TACS dùng text query ở train thì khi test/inference cũng cần tạo hoặc lấy caption query bằng cùng pipeline; caption lỗi phải được mask theo cùng quy tắc. **Trạng thái:** đây là hợp đồng đầu vào của mô hình, không phải một biến thể có metric riêng.
8. **Vì sao không dùng caption của ảnh reference/context ngay từ đầu?** Để trước tiên cô lập lợi ích của text query trong khi selector/context hình ảnh gần TACS gốc. Sau đó bạn yêu cầu thêm text candidate/context thành ablation mới. **Trạng thái:** đã có notebook cho cả hai cách.
9. **Notebook `mtacs-train-captions-vitl16.ipynb` dùng loại ngữ cảnh nào?** Bản này đặt `CONTEXT_TOP_K=1`: query image + query caption + **một context image**; không fuse context caption. Không phải top-3. **Metric bạn báo:** `test_macro_f1=0.8237`.
10. **Có thể dùng cả caption của ảnh context không, có vi phạm không?** Có thể nếu caption chỉ được sinh từ ảnh, có sẵn ở inference và đã kiểm tra leak; cần tách hiệu quả của text trong selector khỏi text trong classifier. **Trạng thái:** có bản top-1 và top-3 context-text, nhưng chưa có metric xác minh cho riêng chúng.
11. **Bạn muốn query + text query + đúng một ảnh context + text context, sao lại ra ba?** Bản top-3 đầu là một mở rộng sai về `K` so với yêu cầu top-1 của bạn. **Hướng sửa:** giữ `K=1`, thêm caption context đã chọn vào fusion; tạo notebook context-text top-1 riêng, không ghi đè run cũ. **Trạng thái:** notebook đã tạo, chưa có metric xác minh.
12. **Kết quả hai hướng gần nhau, nhưng cụ thể là 0.8237 so với 0.8108?** Ghi đúng hai số là `test_macro_f1`, chênh 1,29 điểm phần trăm; không nói “bằng nhau” hoặc kết luận text context vô dụng. **Đề xuất:** ablation cùng `K=1`, seed, CSV và split để loại nhiễu. **Trạng thái:** có số tự báo, chưa có output/checkpoint đối chiếu; chưa xác định chắc `0.8108` thuộc notebook nào.
13. **Có nên thử top-3 context?** Có, nhưng đó là **một giả thuyết riêng** về số lượng context, không được trộn vào phép đo tác dụng candidate text. Bản top-3 selector-only và bản top-3 context-fusion phải là hai run riêng. **Trạng thái:** notebook có, chưa có metric xác minh.

### C. Pool, train/test và triển khai bên ngoài benchmark

14. **20% candidate pool có bị tách khỏi train, hay còn làm ảnh query?** Paper lấy pool từ train; ở cách triển khai hiện tại ảnh đó vẫn có thể là query train. **Đề xuất:** tài liệu hóa mẫu số của 20%, seed và danh sách `image_id` pool, giữ pool cố định khi validation/test. **Trạng thái:** đã giải thích; đây là protocol, không phải metric.
15. **Nếu query chính là một ảnh thuộc pool thì sao?** Vẫn train query đó nhưng không được chọn nó làm context của chính nó; mask self-ID trước softmax/top-k. Với dữ liệu có nhóm liên quan, có thể cần loại cả nhóm. **Trạng thái:** bản DTD và context-text có logic này; cần audit run SUN cũ.
16. **Inference của bản query-caption trên test diễn ra thế nào?** Load checkpoint validation tốt nhất, dùng pool train cố định, lấy caption/test text embedding, chọn context và phân loại; không cập nhật tham số trên test. **Trạng thái:** notebook có bước test; số `0.8237` là kết quả bạn tự báo cho run SUN.
17. **Đưa ảnh hoàn toàn mới ở ngoài SUN397/DTD vào được không?** Được nếu nó thuộc cùng bài toán nhãn; phải có đúng image transform, captioner, auditor, CLIP, candidate bank, class mapping và checkpoint. **Đề xuất:** tách pipeline inference thành hàm/CLI riêng, có nhánh caption invalid, xuất dự đoán cùng context đã chọn và độ tin cậy. **Trạng thái:** mới thảo luận quy trình, chưa có metric hay artifact inference độc lập.
18. **Sau mỗi lần test lấy ảnh về “học tiếp tục” có hợp TACS không?** Đây là online/continual-learning extension, không phải đánh giá TACS gốc. **Đề xuất:** chọn rõ nhãn thật/pseudo-label/chỉ thêm pool, đánh giá trước khi cập nhật, duy trì replay buffer và một test set bất biến để đo quên và drift. **Trạng thái:** ý tưởng, chưa triển khai và chưa có metric.

### D. Debug notebook và chuyển domain

19. **Lỗi `UnboundLocalError: candidate_labels_tensor` trong validation top-3?** Đây là lỗi scope Python: gán lại tên global bên trong hàm khiến Python coi nó là local trước khi được khởi tạo. **Hướng sửa:** dùng tên khác như `candidate_labels_device` hoặc helper gather, giữ tensor nhãn readonly trong eval. **Trạng thái:** builder/notebook sau đã dùng tên device riêng; đây không phải kết quả chất lượng mô hình.
20. **DTD trong paper là dataset nào và tìm ở đâu?** Đó là Describable Textures Dataset của Oxford, 47 lớp texture. **Hướng thử:** chuyển M-TACS sang dataset khác để đo generalization thay vì chỉ SUN397. **Trạng thái:** notebook DTD đã tạo, chưa có test metric.
21. **Notebook DTD có thể tự sinh `captions.csv` khi chạy một lượt không?** Có: thêm cell Florence-2 + audit CLIP + resume CSV trước text embedding/training. **Trạng thái:** notebook đã có cell này; những lỗi môi trường phát sinh trước khi có kết quả cuối.
22. **`DTD_SPLIT=1` nghĩa là gì và train/val/test có chuẩn không?** Đây là bộ `train1/val1/test1.txt` chính thức. **Đề xuất:** báo rõ một split hay trung bình 10 split, không ngầm so một run với điểm paper. **Trạng thái:** loader có kiểm tra phân bố 40 ảnh/lớp/tập và không trùng split.
23. **Lỗi `DTD_SPLIT` chưa định nghĩa, Transformers 5.0.0, thiếu model, sai đường dẫn local, `config/weights=False`?** Đó là lỗi thứ tự cell, phiên bản checkpoint/API hoặc path chứ không phải bằng chứng mô hình kém. **Hướng xử lý:** chạy lại từ cell cấu hình, dùng path Kaggle đúng, kiểm tra weights và ghép legacy Florence-2 với Transformers phù hợp. **Trạng thái:** đã xác minh checkpoint local có `auto_map` legacy; chưa có log DTD train/test thành công.
24. **Muốn đối chiếu với TACS paper thì đọc metric nào?** Với SUN397/DTD dùng **test accuracy** của cùng protocol; `test_macro_f1` là chỉ số bổ sung, không được trừ trực tiếp từ 71,8%/77,7% trong paper. **Đề xuất:** làm baseline visual-only cùng backbone/split/seed, sau đó so các ablation text; với DTD báo theo official splits. **Trạng thái:** chưa có bộ kết quả kiểm soát hoàn chỉnh.

## Bộ thực nghiệm đề xuất nếu tiếp tục nghiên cứu

Các mode `no_context`, `original_tacs`, `text_only`, `mtacs`, `mtacs_shuffled` đã được định nghĩa trong notebook SUN ViT-L/16, nhưng **sự tồn tại của mode không phải bằng chứng nó đã được chạy**. Để truy nguyên lợi ích của text, nên chạy theo thứ tự:

1. **Mốc visual:** no-context và original TACS cùng backbone ViT-L/16, cùng split/seed/candidate pool. Đây là baseline nội bộ công bằng hơn con số ViT-S/16 của paper.
2. **Text query:** text-only, M-TACS query-caption và shuffled-query-caption. Shuffled caption giữ số lượng text tương tự nhưng phá liên hệ ảnh-câu; nếu điểm không đổi, chưa chứng minh text có thông tin hữu ích.
3. **Text candidate:** giữ `K=1`; so visual selector với visual+semantic selector trong khi downstream vẫn chỉ nhận query caption.
4. **Text context:** giữ nguyên selector ở bước 3; bật/tắt fusion caption của candidate đã chọn. Báo thêm tỷ lệ caption context hợp lệ và gate để biết text có thực sự tham gia.
5. **Số lượng context:** chỉ sau các bước trên mới đổi `K=1` sang `K=3`, giữ các yếu tố khác cố định.
6. **Chuyển domain DTD:** chạy ít nhất các mốc visual và M-TACS tương ứng trên cùng official split; nếu muốn công bố theo giao thức DTD, tổng hợp 10 split, nhiều seed và phương sai.

Cho mỗi run, lưu `dataset/split/seed`, hash hoặc path CSV, `prompt_version`, checkpoint Florence-2/CLIP/DINOv3, `K`, cách mask self-match, ngưỡng leak, số epoch tốt nhất theo validation, `test_accuracy`, `test_macro_f1`, valid-caption rate và selected-context same-class rate. Kết quả thiếu log nên ghi **“chưa đo/chưa xác minh”**, không nội suy từ phiên bản gần giống.
