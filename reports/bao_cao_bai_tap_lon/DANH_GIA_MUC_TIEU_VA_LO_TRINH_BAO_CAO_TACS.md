# Đánh Giá Mục Tiêu Và Lộ Trình Báo Cáo TACS

Cập nhật: 27/10/2026. Mục tiêu học phần do sinh viên cung cấp: **xây dựng lại cấu trúc trong paper gốc, đưa ra một vài cải tiến nếu có**. Đây là tư vấn phạm vi và cách trình bày bài tập lớn, không phải xác nhận điểm số hoặc chứng nhận tái lập nguyên bản.

## 1. Đã Làm Đúng Và Đủ Chưa?

**Đúng hướng, và phần mở rộng đã đủ phong phú để viết báo cáo. Tuy nhiên, chưa nên tuyên bố đã tái lập nguyên bản TACS chỉ dựa vào các notebook caption.** Việc cần ưu tiên lúc này là chốt một nhánh TACS visual làm minh chứng cho phần tái triển khai, rồi tách phần mở rộng multimodal thành nghiên cứu riêng.

Không cần chạy đủ 18 dataset, mọi bảng trong paper hay tiếp tục tìm một cấu hình tăng F1 mới để có thể nộp bài. Ngược lại, nhiều notebook không tự chứng minh cấu trúc gốc đã được thực hiện đúng. Một cải tiến được đề xuất và kiểm nghiệm nhưng chưa tăng điểm vẫn có giá trị học phần, miễn là trình bày trung thực về mục tiêu, implementation và kết quả.

| Yêu cầu | Bằng chứng hiện có | Đánh giá và việc cần chốt |
| --- | --- | --- |
| Hiểu query, pool và context | Đã triển khai pool train, truy xuất context, inference, self-exclusion trong DTD | Có nền tảng rõ; trình bày bằng sơ đồ và ID/pool manifest |
| Selector học theo tác vụ | Nhánh SUN visual có utility scores, Gumbel-Softmax, task loss và policy reward | Có cấu trúc trong source; chọn đúng một phiên bản và kiểm chứng đường gradient đang được gọi |
| Mạng phân loại nhận query/context | Source SUN có concat patch tokens trước transformer blocks | Có phần cốt lõi; công bố thêm gate/head/loss của project khác paper |
| Tối ưu hybrid của paper | Paper có task-gradient qua straight-through Gumbel và policy-gradient từ reward | Không thay bằng bằng chứng của nhánh DTD chỉ sampling/gather theo index và policy loss |
| Thực nghiệm TACS gốc | Có nhiều output SUN/Cars/CUB, nhưng một số khác backbone/resolution/budget | Cần một nhóm đối chứng cùng protocol, không ghép điểm khác thiết lập thành một bảng nhân quả |
| Đề xuất mở rộng | Caption query/candidate, selector, caption policies, LoRA/grounded/Qwen, alignment | Đã vượt mức chỉ nêu ý tưởng; nhiều hướng đã có output thật |
| Kiểm nghiệm mở rộng | Tám text ablation, sáu caption-method run mới, architecture studies01..04, ISIC | Đủ nội dung phân tích; chưa phải mọi thay đổi đều cải thiện ổn định |
| Demo và kỹ thuật triển khai | Frontend/backend, hai trang TACS/M-TACS, vị trí hiển thị caption/context/attention | Là minh họa bổ trợ; chưa được coi là inference hoàn chỉnh khi chưa nối checkpoint phù hợp |

### 1.1. Điểm cần nói rõ về cấu trúc gốc

Theo mục 3.2 và 4.2 của [paper gốc](<../../docs/paper/18. Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.pdf>):

- Selector và mạng tác vụ chia sẻ backbone ViT-S/16 DINOv3.
- Phân loại ghép patch embeddings query/context trước các transformer blocks.
- Selector được tối ưu bằng cả straight-through Gumbel-Softmax và policy objective với reward là mức giảm task loss so với query-only.
- Pool cố định lấy từ training data, loại query-associated samples khỏi pairing.
- Segmentation dùng gated cross-attention ở từng transformer block và DPT head; khác với chỉ ghép final tokens.

Trong source [SUN true-validation](../../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-sun-true-val-split.ipynb), có `_forward_token_concat_features`, `select_full_pool`, `select_bank_images_from_weights` và `train_one_epoch`: concat tokens, Gumbel-ST qua trọng số nhân image bank, cùng reward/policy loss. Đây là một nhánh thích hợp để bắt đầu chứng minh cấu trúc, **không phải xác nhận mọi run SUN đã đạt reproduction chuẩn**. Bản này còn có query/context gate, SupCon và các điều chỉnh khác cần công bố.

Một điểm phải kiểm tra trước khi dùng run SUN cũ làm bằng chứng chuẩn: pool lấy từ cùng train queries, nhưng API `select_full_pool` của source được đối chiếu chưa nhận query IDs/excluded positions để mask self. Không suy ra tất cả notebook SUN đều giống vậy; phải chọn đúng phiên bản/checkpoint và kiểm tra pairing thực tế. Self-selection không đồng nghĩa dùng nhãn test để train, nhưng là khác biệt protocol có thể tạo shortcut train.

Trong nhánh DTD ablation/architecture hiện tại, đường thực thi `run_forward -> select_context -> candidate_images_at(indices)` lấy context qua chỉ số categorical. Nhánh selector học bằng policy loss, entropy và backbone/shared projection; task CE vẫn train mạng phân loại nhưng không đi qua thao tác chọn index rời rạc để thành đường Gumbel-ST. Việc class cha còn chứa một hàm Gumbel không chứng minh đường đó đang được gọi. Vì vậy gọi nhóm này là **biến thể M-TACS/TACS-inspired có khai báo điều chỉnh**, không dùng nó thay toàn bộ phần trình bày hybrid TACS gốc.

### 1.2. Mức hoàn thành hợp lý để nộp

Chốt các bằng chứng sau là thực tế hơn việc mở thêm captioner/architecture:

1. Một sơ đồ TACS gốc và bảng ánh xạ từng khối sang class/function của phiên bản visual được chọn.
2. Một cấu hình TACS visual có train/evaluation hoạt động, cùng một No-Context đối chứng trên cùng dataset, backbone, split, preprocessing, budget và quy tắc checkpoint. Thêm Frozen Similarity nếu muốn kết luận selector học được tốt hơn similarity retrieval.
3. Kiểm tra self-exclusion, candidate pool chỉ gồm train, và gradient hai đường nếu tuyên bố tái lập hybrid optimization.
4. Một hoặc vài mở rộng có thiết kế, đối chứng, số liệu và nhận xét. Project hiện đã có nhiều hơn yêu cầu này.
5. Ghi rõ khác biệt so với paper, giới hạn single-seed/split và nguồn số liệu. Không bắt buộc nhiều seed cho mọi hàng nếu học phần không yêu cầu, nhưng không được bịa mean/std hoặc tuyên bố độ ổn định.

Nếu đã có mục 1..3 bằng artifact phù hợp, có thể chốt thí nghiệm và viết ngay. Nếu chưa có, chỉ bổ sung phần cốt lõi còn thiếu, không cần train lại toàn bộ các notebook cải tiến.

## 2. Dòng Thời Gian Các Hướng Đã Trao Đổi

Thứ tự dưới đây theo tiến trình trong hội thoại, không phải mỗi hàng đều là một thuật toán mới. Phân biệt **đề xuất**, **đã tạo implementation** và **đã có kết quả**. Các công việc chia folder/offline/resume là cải tiến kỹ thuật, không tự có accuracy riêng.

### 01. Thêm mô tả văn bản cho ảnh SUN397

- Giả thuyết: caption bổ sung màu sắc, cấu trúc và thuộc tính nhìn thấy được cho biểu diễn ảnh.
- Thực hiện: Florence-2 sinh raw caption từ pixels; lưu CSV theo image ID, batch, resume và provenance model/prompt.
- Bổ sung kiểm tra lexical class terms, CLIP semantic audit, quy tắc 5..8 từ và mask caption không hợp lệ. CLIP không phải mô hình sinh caption hay LLM thứ hai chứng nhận không leak.
- Đây là khởi đầu pipeline dữ liệu, không phải fine-tune Florence hoặc một ablation có metric độc lập.

### 02. M-TACS với caption query

- Đưa caption query qua frozen CLIP text encoder, projection và gate, kết hợp với biểu diễn query/context ảnh trước classifier.
- Selector vẫn dùng ảnh, một context image; caption candidate chưa vào fusion.
- SUN ViT-S/16 cũ có output Accuracy 0.7612, Macro-F1 0.7602. SUN ViT-L/16 query-caption có F1 0.8237 do sinh viên báo trong chat, chưa có đầy đủ artifact cùng run để xác nhận như các nhóm DTD mới.

### 03. Cho caption candidate tham gia selector

- Dùng visual score cộng similarity CLIP giữa caption query và caption candidate để hỗ trợ chọn ảnh context.
- Prototype đầu có K=3, nhưng chưa tự động đồng nghĩa đã fuse caption của candidate vào classifier.
- F1 0.8108 được sinh viên báo ở giai đoạn này, chưa xác định chắc notebook/protocol của run. Không gán số này cho bản top1 context-fusion đã được kiểm chứng.

### 04. Thêm caption của context đã chọn vào fusion

- Lấy image/caption ở cùng selected index, cho caption query và candidate cùng tham gia downstream fusion.
- Làm rõ hai vị trí khác nhau: text chọn context và text hỗ trợ classifier sau lựa chọn.
- Tạo biến thể top3 và sau đó top1 theo yêu cầu một context. Đây là hai phép thử khác nhau; thay K và thêm text cùng lúc không cô lập được tác dụng candidate text.

### 05. Chuẩn hóa pool và quy trình inference

- Làm rõ pool là tập con train; một pool image vẫn có thể là training query nhưng không được chọn chính mình làm context.
- Tách train/val/test, giữ pool cố định khi evaluation; inference ảnh ngoài dataset phải tuân đúng hợp đồng caption của từng model.
- Ý tưởng test xong đưa ảnh về học tiếp được thảo luận dưới online/continual learning, nhưng chưa thành thí nghiệm benchmark hiện tại. Không coi đó là tính năng đã triển khai.

### 06. Chuyển sang DTD và tích hợp sinh CSV trong notebook

- Dùng official split1 với 47 lớp, 1.880 train/1.880 val/1.880 test; pool 376 ảnh train.
- Notebook tạo/resume captions.csv rồi encode/cache text và train, nhằm giảm thao tác thủ công.
- DTD_SPLIT là chỉ số official split 1..10, không phải seed hoặc K-fold của selector.

### 07. Xây dựng web demo TACS/M-TACS

- Tạo frontend/backend phân loại SUN397, tách trang mô hình visual và trang multimodal.
- Thêm vị trí hiển thị caption query, candidate/caption tương ứng và attention kiểu Figure4.
- Chưa có checkpoint/pool tương thích nối adapter nên prediction/attention chưa phải chức năng hoàn chỉnh. Đây là phần triển khai, không phải minh chứng tăng accuracy.

### 08. Thiết kế ablation vị trí và nội dung caption

- Tạo bộ tám cấu hình để tách text query, text candidate, text selector và các đối chứng shortcut.
- Ban đầu chạy chung một notebook; sau đó tách thành tám notebook riêng, chốt 50 epoch để quản lý compute/resume và thu bảng so sánh.
- Bổ sung q_gate/c_gate, coverage và caption source IDs để biết nhánh nào thực sự được dùng. Gate không phải xác suất đúng hay phần trăm đóng góp vào prediction.

### 09. Sửa policy DTD sau khi toàn bộ caption bị loại

- Bộ lọc CLIP ngưỡng 0.30 cứng loại hết caption; nhiều caption bị cắt còn boilerplate như “The image is a close up of a”.
- Chuyển sang `dtd_lexical_v2`: xử lý phần mở đầu, giữ mô tả ngắn và kiểm tra lexical; semantic CLIP score là thông tin audit, không tự loại cứng theo 0.30.
- Đây là điều chỉnh policy sử dụng dữ liệu, không chứng minh caption là ground truth hoặc không còn mọi loại leakage. Giữ policy cố định trong các đối chứng mới.

### 10. Hoàn thành tám text-context ablation DTD

| Cấu hình | Query text fusion | Candidate text fusion | Text selector | Test Macro-F1 |
| --- | --- | --- | --- | ---: |
| visual_only | Không | Không | Không | 0.848168 |
| query_only | Có | Không | Không | 0.823067 |
| candidate_only | Không | Có | Không | 0.850321 |
| both_fusion | Có | Có | Không | 0.810661 |
| full_both | Có | Có | Có | 0.809124 |
| full_shuffle_both | Có | Có | Có, caption đảo | 0.833392 |
| full_unscreened | Có | Có | Có, caption trước lọc | 0.789774 |
| full_redacted | Có | Có | Có, tên lớp/alias đã che | 0.811105 |

Candidate-only nhỉnh hơn visual-only khoảng 0.2153 điểm phần trăm trong run này. Không coi đó là cải thiện ổn định từ một seed. Visual-only vẫn dùng context image, không phải No-Context của paper. Đây là nhóm CSV/run lịch sử, không thay bằng điểm của các run Prefix hash-locked về sau.

### 11. Xây dữ liệu tham chiếu và thử fine-tune captioner

- Khi chưa có caption mục tiêu, tạo mẫu reference để gán/duyệt; bộ mẫu 188 ảnh là tập con, không phải full CSV 5.640 ảnh.
- Đã trao đổi sử dụng VLM/LLM teacher để hỗ trợ gán mô tả nhìn thấy được và chuẩn bị decoder-LoRA Florence.
- Reference teacher AI không tự trở thành ground truth do con người kiểm duyệt độc lập. Báo cáo đúng nguồn và quy trình thực sự có bằng chứng, không tự nhận đã có human verification đầy đủ.
- Classifier full_both với CSV LoRA có test F1 0.806707, so với full_both lịch sử 0.809124. Đây là downstream metric, không phải captioner validation score và không tự xác nhận toàn bộ quá trình fine-tune đã được audit.

### 12. Thay cách chọn caption bằng grounded selection

- Tạo bốn CSV: Prefix, FirstValid, SingleGrounded và MultiGrounded.
- Prefix là control rút gọn; FirstValid chọn mô tả đủ điều kiện; SingleGrounded lựa chọn theo khớp ảnh-text từ nguồn giả thuyết đơn; MultiGrounded mở rộng nguồn giả thuyết rồi lựa chọn.
- Các pipeline selection/rerank không đồng nghĩa bốn lần fine-tune Florence. CLIP ranking score không phải đánh giá caption độc lập.
- Tạo bốn classifier notebook cùng full_both để so sánh CSV.

### 13. Chuẩn hóa project và khả năng tái chạy

- Phân tách notebooks tái triển khai paper, cải tiến, caption generation, caption ablation, kết quả completed và archive theo dataset/task.
- Bổ sung src/tools/tests/protocol, CSV hash, manifests, checkpoint identity và các báo cáo bảng kết quả.
- Sửa môi trường offline Florence legacy/native, tách subprocess phù hợp khi cần. Đây là engineering support, không phải một method có F1 mới.

### 14. Mở rộng captioner sang Qwen3-VL 8B và 32B

- Tạo pipeline xuất hai CSV, sau đó dùng classifier full_both tương ứng để so sánh với Florence.
- Không fine-tune Qwen trong hai notebook classifier đã đọc; trọng số captioner và trọng số TACS là hai phần khác nhau.

| CSV/method trong nhóm full_both | Test Macro-F1 |
| --- | ---: |
| Florence-Prefix | 0.810651 |
| Florence-FirstValid | 0.809382 |
| Florence-SingleGrounded | 0.779611 |
| Florence-MultiGrounded | 0.790596 |
| Qwen3-VL-8B-Instruct | 0.821445 |
| Qwen3-VL-32B-Instruct | 0.820788 |

Qwen nhỉnh hơn các Florence pipeline trong nhóm full_both, nhưng chưa vượt visual-only lịch sử. 32B chưa cho thấy lợi thế so với 8B. Natural masks/coverage khác nhau, nên không quy mọi chênh lệch cho chất lượng câu chữ hoặc số tham số.

### 15. Mở rộng sang segmentation ISIC2017

- Đọc run visual segmentation448, sau đó tạo bản Qwen8B full_both có sinh/resume CSV và dùng caption query/candidate trong fusion/selector.
- Visual có test Dice 0.8560, IoU 0.7738; Qwen8B full_both đạt Dice 0.848689, IoU 0.765407. Chưa có lợi ích test quan sát được.
- Run visual ISIC dùng ViT-S/16 và DPT-lite/final-token cross-attention; không gọi chung là ViT-L/16 như DTD hoặc segmentation blockwise nguyên bản paper.
- Đã tạo hướng Florence-Prefix ISIC, nhưng chưa có metric xác nhận trong nhóm bằng chứng hiện tại. Không điền số dự đoán cho hướng này.

### 16. Đặt lại câu hỏi về kiến trúc fusion

- Từ các kết quả text thường không tăng điểm, chuyển câu hỏi từ “captioner nào tốt hơn?” sang “mô hình sử dụng hai modality như thế nào?”.
- Projection về cùng kích thước không tự chứng minh hai vector cùng không gian ngữ nghĩa; gate nhỏ còn phụ thuộc thang norm, không tự chứng minh mô hình bỏ text.
- Tham khảo m-LTM và OmniRet để đề xuất alignment/metric, semantic retrieval, token-level fusion/resampling và adaptive pooling. Đây là nguồn ý tưởng, không tự biến các module của hai paper thành đóng góp nguyên bản của project.

### 17. Chốt baseline mới với đúng Florence-Prefix CSV

- Tạo architecture study01: candidate_only, query text off, image-only selector, ViT-L16/384/K1, split1/seed42, 50 epoch.
- Khóa exact Prefix CSV bằng hash để không nhầm với candidate_only lịch sử.
- Kết quả val/test F1 0.840162/0.853032. Đây là baseline của chuỗi kiến trúc mới, không phải No-Context hoặc điểm TACS nguyên bản paper.

### 18. Architecture study02: chuẩn hóa vector text trước fusion

- L2-normalize projected candidate text trước gate/fusion để kiểm tra ảnh hưởng thang độ lớn.
- Giữ image-only selector, chưa thêm own-pair alignment loss; có đối chứng fixed-context text off/shuffle và norm/gate statistics.
- Val/test F1 0.835824/0.849329. Chưa cải thiện so với study01.

### 19. Architecture study03: auxiliary image-caption alignment

- Giữ study02, thêm m-LTM-inspired KL matching qua balanced log-Sinkhorn và metric diagonal Mahalanobis dương.
- Dùng ảnh candidate train và caption của chính ảnh đó; image anchors detached, projection text dùng chung với fusion, có duplicate-aware target. Không dùng query/context do retrieval chọn làm positive giả.
- Val/test F1 0.834947/0.848722. Học tốt hơn mục tiêu matching trên train pool nhưng chưa cải thiện classification. Training-pool retrieval không phải held-out retrieval quality.

### 20. Architecture study04: aligned query-image/candidate-caption selector

- Giữ study03 làm đối chứng cấu trúc, thêm semantic residual chấm query image với candidate caption qua projection/metric alignment. Không dùng caption query.
- Khác full_both cũ: full_both dùng caption-query/caption-candidate similarity; study04 dùng ảnh-query/caption-candidate. Có visual warmup, bounded alpha, fallback và self-exclusion.
- Val/test F1 0.838806/0.850717; tăng nhẹ so với 03 nhưng chưa vượt 01.
- Tắt semantic selector cho val F1 0.838311; đảo caption chỉ trong selector cho 0.836615. Tín hiệu hữu ích còn yếu, CI95 của delta vẫn chứa 0.
- Giữ ảnh context rồi tắt/đảo fusion caption cho F1 0.839080/0.841044: chưa có lợi ích F1 rõ từ fusion. Đây là same-checkpoint interventions, không phải retrain từng control.
- Nguồn study04 hiện là bảng output notebook, hiển thị sáu chữ số thập phân; chưa có prediction CSV/history/checkpoint export bản04 để audit độc lập như 02/03. Có lỗi stderr multiprocessing dù run hiển thị đủ 50 epoch và final test.

### 21. Những hướng chưa thực hiện, không ghi thành kết quả

- Online/continual learning sau mỗi inference.
- Partial OT để xử lý noisy correspondence trong kiến trúc DTD mới.
- Token-level fusion/resampler thay cộng vector.
- Attention Sliced Wasserstein Pooling/adaptive pooling theo hướng OmniRet.
- Xác nhận nhiều seed/split và matched-mask caption controls đầy đủ.
- Kết quả Florence-Prefix ISIC và tích hợp inference/attention thật vào web, nếu chưa có artifact bổ sung.

## 3. Bố Cục Báo Cáo Theo Yêu Cầu Giảng Viên

Không cần kể mọi lỗi Kaggle hay mọi vòng chat trong thân báo cáo. Dùng dòng thời gian ở trên làm nhật ký/phụ lục, còn thân bài phải ưu tiên **TACS gốc -> triển khai -> mở rộng -> thực nghiệm**.

### Phần đầu

Bìa, phân công nếu làm nhóm, tóm tắt, mục lục, danh mục bảng/hình/từ viết tắt. Tóm tắt nêu phạm vi tái triển khai và kết quả chính, không khẳng định vượt paper nếu chưa có cùng protocol.

### Chương 1. Giới thiệu đề tài

1.1. Bài toán và động cơ sử dụng context ảnh. Giải thích vì sao ảnh hữu ích cho tác vụ có thể khác ảnh tương tự nhất.

1.2. Mục tiêu học phần. Tái triển khai các khối TACS trong phạm vi dataset lựa chọn; thử mở rộng caption và đánh giá bằng đối chứng.

1.3. Phạm vi và đóng góp thực hiện. Ghi dataset/task, nguồn pretrained, phần tự triển khai và phần kế thừa. Không nhận TACS/DINO/CLIP/m-LTM là thuật toán mới do mình sáng tạo.

### Chương 2. Cơ sở lý thuyết và paper TACS gốc

2.1. ViT/DINOv3, query/candidate/context và retrieval.

2.2. Selector: đặc trưng và utility scores, candidate probabilities, sampling và argmax khi inference.

2.3. Downstream classification: ghép query/context tokens trước transformer blocks; segmentation: blockwise gated cross-attention và DPT trong paper.

2.4. Hybrid optimization: Gumbel-ST/task gradient, policy reward/advantage và joint objective. Giải thích rõ mỗi đường truyền gradient chứ không chỉ chép công thức.

2.5. Thực nghiệm paper và giới hạn phạm vi tái triển khai. Tóm tắt các baseline; số paper đặt ở cột tham khảo riêng, không thay cho kết quả project.

### Chương 3. Tái triển khai TACS trong project

3.1. Kiến trúc phần mềm và ánh xạ module paper sang code/notebook canonical.

3.2. Dataset/split/preprocessing/pool. Công bố pool seed, IDs, self-exclusion và nguyên tắc val/test không vào candidate pool.

3.3. Forward, training và inference thực tế. Chỉ rõ nhánh visual nào có hai đường optimization; nhóm caption nào dùng categorical-policy adaptation.

3.4. Khác biệt so với paper. Backbone, resolution, epochs, augmentation, extra gate/SupCon, budget/seeds và segmentation head/fusion.

3.5. Đối chứng tái triển khai. No-Context và TACS cùng protocol là nền tối thiểu; Frozen Similarity/Random/Feature-Average bổ sung theo claim và tài nguyên.

### Chương 4. Các hướng mở rộng đã đề xuất và triển khai

4.1. Pipeline sinh/audit CSV, policy dtd_lexical_v2 và provenance; caption là synthetic weak input.

4.2. Vai trò text query, candidate fusion và selector. Phân biệt việc chọn context với việc dùng caption sau khi chọn.

4.3. Thiết kế tám text ablation và mục đích shuffle/unscreened/redacted.

4.4. Caption references/LoRA và các pipeline Prefix/FirstValid/SingleGrounded/MultiGrounded; Qwen8B/32B. Phân biệt fine-tune captioner, thay source caption và reranking.

4.5. Kiến trúc với CSV cố định: baseline01, norm02, alignment03, selector04. Nêu hypothesis và công thức đủ để tái chạy từng thay đổi; không đổi số thứ tự với notebook04 both_fusion cũ.

4.6. Mở rộng ISIC segmentation. Mô tả nguồn caption, fusion/selector và khác biệt với DTD, không trộn các backbone.

### Chương 5. Thực nghiệm, phân tích và thảo luận

5.1. Protocol và metric. Bảng riêng theo nhóm, Accuracy/Macro-F1 phân loại, Dice/IoU phân đoạn; checkpoint chọn bằng validation criterion thực tế.

5.2. TACS visual so với baseline. Đây là bảng chứng minh phần tái triển khai, không để các bảng caption thay thế.

5.3. Tám cấu hình text-context DTD. Phân tích vị trí text và các đối chứng, không coi visual_only là No-Context.

5.4. Các CSV/caption method trong full_both. Báo coverage và natural masks, không suy rằng captioner lớn hơn là tốt hơn.

5.5. Architecture01..04 với Prefix cố định. So sánh đúng đối chứng từng cặp, giữ cả kết quả giảm, nêu runtime differences và giới hạn single-seed.

5.6. Selector/fusion interventions. Tách context cố định khỏi context được chọn lại; CI query-resampling không thay training-seed uncertainty.

5.7. ISIC visual/caption và định tính. Chỉ dùng predicted masks/attention từ checkpoint thật, lấy cả ví dụ đúng và sai.

5.8. Chi phí và giới hạn. Compute/VRAM nếu có log, nguy cơ shortcut, test đã xem nhiều vòng, thiếu artifact hoặc bất đồng protocol. Tránh tiếp tục chọn prompt/hyperparameter bằng test.

### Chương 6. Demo và khả năng triển khai

6.1. Backend/frontend và hai chế độ TACS visual/M-TACS.

6.2. Quy trình upload ảnh, retrieval, caption/context và kết quả. Phân biệt phần chạy thật với placeholder.

6.3. Checkpoint/pool/class map cần để inference, attention Figure4 và trạng thái tích hợp. Không có checkpoint phù hợp thì không mô tả demo như hệ thống phân loại đang hoạt động.

Chương này có thể rút thành một mục ngắn nếu rubric chỉ tập trung vào paper; không cần làm web hoàn chỉnh mới được viết phần mô hình.

### Chương 7. Kết luận và hướng phát triển

7.1. Đã tái triển khai khối nào, trên dataset nào, với điều chỉnh gì.

7.2. Kết luận từ các mở rộng: caption có tác động nhưng không mặc định tăng F1; alignment fit không bảo đảm task benefit; semantic selector có tín hiệu nhỏ chưa xác nhận ổn định.

7.3. Việc chưa làm: kiểm chứng nhiều seed, matched-mask, token fusion, Partial OT/pooling, continual learning và demo hoàn chỉnh. Không viết các mục này như thành quả đã có.

### Phụ lục

Cây thư mục, danh sách notebook/run ID, config/hash/pool manifests, bảng chi tiết, lỗi môi trường và cách tái chạy, nhật ký 21 mốc ở trên. Tài liệu tham khảo phải ghi rõ paper TACS và hai paper multimodal nguồn ý tưởng.

## 4. Bộ Bảng Và Hình Ưu Tiên

| Bảng/hình | Nội dung | Vai trò |
| --- | --- | --- |
| Sơ đồ TACS gốc | Selector, hai đường tối ưu, downstream, train/inference | Bắt buộc để giải thích paper |
| Bảng paper/project mapping | Khối, code, cấu hình, điểm khác | Chứng minh tái triển khai có kiểm soát |
| Bảng No-Context/Frozen/TACS | Cùng dataset và protocol | Phần nền cần chốt, không thay bằng text ablation |
| Bảng tám text ablation | Vai trò query/candidate/selector và shortcut controls | Đã có output single-seed |
| Bảng caption methods | Sáu run mới, reference lịch sử tách rõ | Đã có output; có confound coverage |
| Bảng architecture01..04 | CSV cố định, norm/alignment/selector | Đã có output, bản04 hiện ở độ chính xác bảng notebook |
| Hai bảng interventions | Fusion fixed-context và selector reselected-context | Bằng chứng cơ chế, không phải retraining |
| Bảng ISIC | Visual vs caption, Dice/IoU, đúng backbone | Mở rộng có kết quả, chưa chứng minh caption tăng điểm |
| Minh họa query/context/caption và mask | Ví dụ đúng/sai, attention khi có | Giúp giải thích, không thay metric |

Không phải lấp đầy Tables2..5 nguyên kích thước paper. Chỉ đặt tên “theo tinh thần Table...” khi phạm vi và protocol đã thu hẹp; ghi “chưa thực hiện” thay vì điền số không có bằng chứng.

## 5. Kết Luận Để Quyết Định Việc Tiếp Theo

**Nên bắt đầu viết báo cáo ngay, dừng mở thêm kiến trúc mới, và dành phần thời gian còn lại để kiểm chứng nhánh TACS gốc/đối chứng và hoàn thiện artifact.** Phần cải tiến đã đủ nội dung cho học phần. Nếu giảng viên chấm trọng tâm cấu trúc paper, vài bằng chứng đúng về token fusion, hybrid gradients, self-mask và bảng baseline cùng protocol quan trọng hơn một captioner mới hoặc vài phần nghìn F1.

Tên đề tài phù hợp: **“Tái triển khai TACS trên một số bộ dữ liệu và khảo sát mở rộng ngữ cảnh ảnh–văn bản.”** M-TACS là tên biến thể trong project; không tuyên bố đó là model của tác giả paper hoặc phương pháp đã vượt TACS trên mọi thiết lập.

## Nguồn Đối Chiếu

- [Paper readiness và các khoảng thiếu protocol](../paper_readiness/KE_HOACH_KET_QUA_BAO_CAO_TACS.md), snapshot cũ được bổ sung bởi các báo cáo mới, không coi mọi mục “chưa có output” trong snapshot còn đúng.
- [Tám text-context ablation](../../experiments/completed/text_context_ablation/dtd/analysis/DANH_GIA_ABLATION_DTD_01_07.md).
- [Caption-method comparison](../caption_ablation/dtd/KET_QUA_SO_SANH_PHUONG_PHAP_CAPTION_DTD.md).
- [Baseline01](../multimodal_architecture_ablation/dtd/BASELINE_01_FLORENCE_PREFIX_CANDIDATE_ONLY.md).
- [Study02/03 audit](../multimodal_architecture_ablation/dtd/STUDY_02_03_REVIEW_AND_STUDY_04_PLAN.md).
- [Study04 đã chạy](../../study4.ipynb), đọc source/output, không sửa hoặc chạy lại notebook.
- [ISIC visual/Qwen8B](../paper_tables/SO_SANH_ISIC_VISUAL_VA_QWEN8B.md).
- [Đề cương chi tiết đã có](DE_CUONG_BAO_CAO_TACS_MTACS.md), các mục architecture được cập nhật bằng kết quả ở tài liệu này.
