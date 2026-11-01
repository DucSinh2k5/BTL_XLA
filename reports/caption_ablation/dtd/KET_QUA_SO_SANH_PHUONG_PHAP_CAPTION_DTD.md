# So Sánh Tám Phương Pháp Caption Trên DTD

Ngày cập nhật: 24/10/2026. Đã bổ sung hai notebook Qwen vào bốn run Florence mới và hai reference lịch sử. Sáu notebook kết quả mới được đổi tên, chuyển vào `experiments/completed/caption_ablation/dtd/` theo từng method. SHA-256 toàn file và saved outputs được kiểm tra trước/sau; không sửa cell, xóa output hoặc chạy lại training.

## 1. Phạm vi và cách lấy số

Đây là **metrics phân loại của M-TACS sau khi sử dụng từng CSV**, không phải phép đo trực tiếp độ đúng của captioner.

Sáu run mới đều dùng DTD split 1, train/val/test 1.880 ảnh mỗi split, seed 42, DINOv3 ViT-L/16, ảnh 384, batch 32, 50 epoch, K=1 và candidate pool 376 ảnh train. Classifier dùng `full_both`: text query/candidate đều vào fusion và selector kết hợp ảnh-text. K=1 là chọn một ảnh context cho mỗi query, không phải k-fold. Checkpoint được chọn theo **validation loss nhỏ nhất**, không lấy epoch cuối hoặc chọn bằng test F1.

Cả sáu run khai báo outer policy `dtd_lexical_v2`; CLIP semantic score chỉ phục vụ audit, không tự loại caption bằng ngưỡng 0.30. Qwen thay nguồn captioner/prompt và cách tạo mô tả; đây không phải phép so sánh chỉ thay số tham số trong cùng một pipeline sinh caption. Hai notebook Qwen đã cung cấp là notebook **train classifier từ CSV có sẵn**, không phải notebook fine-tune Qwen.

Bằng chứng: [caption_method_output_evidence.json](caption_method_output_evidence.json). File này lưu metric gốc, source/output hashes, CSV hashes, log 50 epoch của từng run và đối chiếu AST model/training. Đối chiếu tên cũ/mới cùng hash bảo toàn nằm trong [relocation_manifest.json](../../../experiments/completed/caption_ablation/dtd/relocation_manifest.json).

## 2. Bảng đầy đủ

Accuracy/F1 và coverage đều tính theo phần trăm; loss không đổi đơn vị. Số hiển thị được lấy từ bảng validation/test có sáu chữ số thập phân, không lấy bảng report cuối đã làm tròn bốn chữ số.

| Phương pháp caption | Val Macro-F1 (%) | Test Accuracy (%) | Test Macro-F1 (%) | Test loss | Best epoch | Coverage query test (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence cũ, reference lịch sử | 80.2483 | 81.0106 | 80.9124 | 1.307802 | 42 | 84.6277 |
| Florence + decoder LoRA, reference lịch sử | 79.6779 | 80.7447 | 80.6707 | 1.311028 | 39 | 84.3085 |
| Florence-Prefix | 80.2258 | 81.1170 | 81.0651 | 1.307839 | 39 | 84.6277 |
| Florence-FirstValid | 80.2282 | 81.0638 | 80.9382 | 1.327073 | 35 | 100.0000 |
| Florence-SingleGrounded | 78.1723 | 78.1915 | 77.9611 | 1.398994 | 45 | 100.0000 |
| Florence-MultiGrounded | 79.0217 | 79.3085 | 79.0596 | 1.378851 | 45 | 100.0000 |
| Qwen3-VL-8B-Instruct | **81.1412** | **82.2872** | **82.1445** | **1.275098** | 36 | 74.0957 |
| Qwen3-VL-32B-Instruct | 80.9204 | 82.1809 | 82.0788 | 1.275700 | 40 | 71.5957 |

Chữ đậm đánh dấu Accuracy/F1 cao nhất hoặc test loss thấp nhất **trong sáu run mới**, không hàm ý có ý nghĩa thống kê. Hai hàng đầu là reference lịch sử đã đọc trực tiếp từ notebook đã chạy, không phải hai control thuộc cùng bộ CSV hash-locked mới. Không đổi nhãn Florence cũ thành Prefix. Bảng có tám phương pháp caption; visual-only được giữ riêng ở mục 5 vì là control kiến trúc.

Coverage query test là phần query có text thực sự khả dụng trên test. Đây **không phải** tỷ lệ caption được con người xác nhận đúng, và khác tỷ lệ caption hợp lệ trên toàn bộ 5.640 ảnh hoặc trên toàn pool.

## 3. Các đối chiếu chính trong sáu run mới

| So với Prefix | Delta test Accuracy (điểm %) | Delta test Macro-F1 (điểm %) | Số ảnh đúng suy ra trên 1.880 ảnh test |
| --- | ---: | ---: | ---: |
| Prefix | 0 | 0 | 1.525 |
| FirstValid | -0.0532 | -0.1269 | 1.524 |
| SingleGrounded | -2.9255 | -3.1040 | 1.470 |
| MultiGrounded | -1.8085 | -2.0055 | 1.491 |
| Qwen3-VL-8B-Instruct | +1.1702 | +1.0794 | 1.547 |
| Qwen3-VL-32B-Instruct | +1.0639 | +1.0137 | 1.545 |

Số ảnh đúng là số nguyên suy ra từ accuracy hiển thị và N=1.880, không thay cho prediction CSV đầy đủ. Không biết các method sai/đúng trên cùng những ảnh nào nên chưa tính được paired test hoặc phân tích lỗi theo ảnh.

### Qwen cải thiện so với các pipeline Florence đã thử

Qwen 8B đạt test macro-F1 82.1445%, cao hơn Florence-Prefix 1.0794 điểm phần trăm và Florence cũ 1.2321 điểm. Qwen 32B đạt 82.0788%, tương ứng cao hơn 1.0137 và 1.1664 điểm. Test loss giảm từ 1.307839 của Prefix xuống 1.275098/1.275700. Validation F1 cũng cao hơn nhóm Florence mới, nên thứ hạng không chỉ xuất hiện trên test.

Đây là **cải thiện downstream quan sát được khi thay pipeline caption**, chưa phải bằng chứng caption Qwen đúng hơn ở mọi ảnh, đã hết leak hoặc chắc chắn cải thiện qua nhiều seed. Hai Qwen có natural mask khác nhau và khác các Florence controls.

### 32B chưa thể hiện lợi thế so với 8B

8B nhỉnh hơn 32B **0.0657 điểm test F1** và **0.1063 điểm Accuracy**, tương ứng hai ảnh đúng chênh lệch tổng số. Test loss gần nhau. Trên validation, 8B có F1 cao hơn (81.1412% so với 80.9204%), nhưng 32B có loss thấp hơn một chút (1.299287 so với 1.301636); các chỉ số không đồng loạt ủng hộ một model. Không đủ bằng chứng kết luận 8B tốt hơn có ý nghĩa thống kê, nhưng kết quả hiện có cũng không hỗ trợ tuyên bố tăng lên 32B mang lại lợi ích phân loại.

Coverage query test của 32B là 71.5957%, thấp hơn 8B 2.5000 điểm; cả hai thấp hơn Prefix và các grounded controls. Điều này cho thấy tỷ lệ text khả dụng không quyết định một mình chất lượng phân loại. Khác biệt mask là một yếu tố gây nhiễu cần đo bằng matched-mask control, không phải cơ chế suy giảm đã được chứng minh. Chưa có log thời gian/VRAM của bước sinh hai CSV để định lượng trade-off chi phí captioner.

### Prefix và FirstValid gần tương đương trong run này

Prefix có F1 cao hơn FirstValid 0.1269 điểm phần trăm; accuracy chỉ hơn một ảnh đúng. Validation F1 lần lượt 80.2258 và 80.2282, gần như bằng nhau, thậm chí FirstValid nhỉnh hơn rất ít trên validation.

Vì vậy nên viết **Prefix và FirstValid cho hiệu năng gần tương đương trong thiết lập hiện tại**, không khẳng định Prefix chắc chắn tốt hơn. Chỉ có một seed/split và hai method khác coverage.

FirstValid tăng coverage query test từ 84.6277% lên 100%, nhưng không làm classifier tốt hơn rõ ràng. Đây là bằng chứng rằng có thêm caption hợp lệ theo policy không tự động tạo thêm thông tin hữu ích; không phải bằng chứng caption được thêm vào đều sai.

### Grounded ranking chưa cải thiện downstream

SingleGrounded và FirstValid dùng cùng nguồn mô tả greedy và availability, nhưng khác cách chọn đoạn. SingleGrounded thấp hơn FirstValid 2.9771 điểm test F1. Do đó bước ranking theo ảnh/CLIP trong pipeline này chưa mang lại lợi ích classification quan sát được.

MultiGrounded cao hơn SingleGrounded 1.0985 điểm F1 và 21 ảnh đúng, cho thấy thêm giả thuyết caption có cải thiện quan sát được so với single-grounded. Tuy nhiên nó vẫn thấp hơn Prefix 2.0055 điểm F1 và thấp hơn FirstValid 1.8786 điểm F1; chưa có lợi ích cộng dồn vượt các control đơn giản hơn.

Thứ hạng này cũng xuất hiện trên validation: hai Grounded thấp hơn nhóm Prefix/FirstValid. Không nên chỉ quy chênh lệch cho một lần chọn epoch ngẫu nhiên trên test. Tuy nhiên nhiều seed vẫn cần thiết để đánh giá độ ổn định.

Một cách giải thích khả dĩ là mức khớp ảnh-caption theo CLIP không trùng hoàn toàn với thông tin phân biệt 47 texture classes. Ranking có thể ưu tiên nội dung nổi bật nhưng không hữu ích cho classifier. Đây là **giả thuyết**, chưa phải cơ chế đã được chứng minh bằng kiểm duyệt ảnh-caption hoặc can thiệp cùng checkpoint.

Các giá trị `Stored CLIP score range` trong stdout là điểm audit ngữ nghĩa theo class prompts, không được tự gọi là điểm image-text ranking hay chất lượng caption độc lập.

## 4. Coverage, gate và selector

| Method mới | Caption hợp lệ toàn dataset | Query test có text (%) | Context thực sự được chọn có text trên test (%) | Query gate trung bình | Candidate gate trung bình | Semantic mix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefix | 4.749/5.640 | 84.6277 | 85.5851 | 0.013987 | 0.001836 | 0.497618 |
| FirstValid | 5.640/5.640 | 100.0000 | 100.0000 | 0.014163 | 0.001893 | 0.497948 |
| SingleGrounded | 5.640/5.640 | 100.0000 | 100.0000 | 0.017658 | 0.004731 | 0.498157 |
| MultiGrounded | 5.640/5.640 | 100.0000 | 100.0000 | 0.017174 | 0.004452 | 0.498488 |
| Qwen3-VL-8B-Instruct | 4.211/5.640 | 74.0957 | 82.9255 | 0.015749 | 0.002957 | 0.498105 |
| Qwen3-VL-32B-Instruct | 4.111/5.640 | 71.5957 | 78.5638 | 0.013951 | 0.002096 | 0.498710 |

Caption khả dụng theo split/pool, đọc từ bảng output trước training:

| Method | Train / 1.880 | Val / 1.880 | Test / 1.880 | Candidate pool / 376 |
| --- | ---: | ---: | ---: | ---: |
| Prefix | 1.573 | 1.585 | 1.591 | 324 |
| FirstValid | 1.880 | 1.880 | 1.880 | 376 |
| SingleGrounded | 1.880 | 1.880 | 1.880 | 376 |
| MultiGrounded | 1.880 | 1.880 | 1.880 | 376 |
| Qwen 8B | 1.418 | 1.400 | 1.393 | 295 |
| Qwen 32B | 1.391 | 1.374 | 1.346 | 285 |

Pool coverage là thống kê trên toàn bộ pool cố định; `candidate_caption_rate` là tỷ lệ context **thực sự được chọn** có caption khi đánh giá. Hai đại lượng có mẫu số khác nhau, không được đánh đồng. Caption bị mask không có nghĩa ảnh tương ứng bị bỏ khỏi train/test/pool.

Các gate nhỏ nhưng không bằng không. Không được diễn giải chúng như phần trăm đóng góp thông tin hoặc kết luận model bỏ hoàn toàn text. Semantic mix gần 0.5 cũng không chứng minh text không được dùng hoặc gradient đã hỏng.

Đổi CSV trong `full_both` có thể thay đổi đồng thời text fusion, ảnh context được semantic selector chọn và cả quá trình tối ưu. Do đó đây là **ablation cả pipeline sử dụng caption**, không phải một phép đo tách biệt chất lượng câu chữ.

## 5. So với các mốc lịch sử

- Florence cũ full_both đạt 80.9124 F1; LoRA full_both đạt 80.6707. Chênh lệch lịch sử là -0.2417 điểm F1, chưa hỗ trợ tuyên bố fine-tune đã cải thiện classifier.
- Prefix mới đạt 81.0651 F1, gần Florence cũ; chênh +0.1527 điểm F1 không đủ để quy thành một cải tiến ổn định, nhất là chưa khóa hoàn toàn bộ provenance giữa run lịch sử và run mới.
- Visual-only DTD từ bộ ablation trước đạt Accuracy 84.9468%, Macro-F1 84.8168%, test loss 1.156826. Qwen 8B/32B vẫn thấp hơn mốc này lần lượt **2.6723/2.7380 điểm F1**. Prefix thấp hơn 3.7517 điểm.
- Visual-only vẫn có ảnh context, không phải No-Context hoặc kết quả TACS paper. Mốc này thuộc nghiên cứu kiến trúc text-context; không được trộn thành phương pháp sinh caption thứ chín. Nguồn: [visual-only.ipynb](../../../experiments/completed/text_context_ablation/dtd/visual-only.ipynb).

Nhận xét đúng phạm vi là: các phương pháp caption được khảo sát chưa giúp cấu hình full_both vượt visual control trong các run đang có. Không suy rộng thành mọi caption đều vô ích, mọi cơ chế fusion đều kém, hoặc grounded captions mô tả ảnh sai.

## 6. Đã kiểm tra và chưa thể kiểm tra

Đã kiểm tra:

- Từng notebook báo completed, 50 epoch, seed 42, LR schedule horizon 50; có đủ 50 dòng log epoch. Không có cell `output_type=error`.
- GPU log đều 95 GiB; batch train 32, retrieval 128, feature refresh 64; official DTD 1.880/1.880/1.880 và pool 376 ảnh train.
- Bốn Florence CSV được đối chiếu byte hash với file local, metadata và runtime. Hai Qwen ghi đầy đủ CSV SHA-256 trong stdout; chưa được cung cấp hai CSV đó ở local để kiểm tra byte hash độc lập. Metadata Qwen ghi `computed_and_locked_at_runtime`, không phải chuỗi hash thực tế.
- Giữa cả sáu run, **57/57 định nghĩa function/class chung có AST giống nhau**, gồm các class model, forward, optimizer, training/evaluation và data loader. Riêng bốn Florence có 65/67 định nghĩa chung giống nhau; hai khác biệt là helper kiểm tra/tìm CSV theo method. Helper nhập dữ liệu Qwen khác Florence, không kết luận toàn bộ notebook giống nhau.
- Các LR khai báo: backbone 3e-6, head/selector 1e-4; label smoothing 0.1, SupCon 0.05, warmup 5, policy weight 1.0, entropy weight 0.01 giống nhau trong source cả sáu run.
- Bảng report cuối khớp validation/test riêng trong sai số làm tròn; best epochs lần lượt Prefix 39, FirstValid 35, SingleGrounded 45, MultiGrounded 45, Qwen 8B 36, Qwen 32B 40. Chúng cũng khớp minimum loss trong log 50 epoch đã hiển thị. Không thay số test bằng F1 cao nhất trên đường cong validation.
- Sáu file sau di chuyển có SHA-256 toàn file, dung lượng và saved outputs hash khớp bản trước di chuyển; metadata, widgets, hình và execution counts được giữ nguyên. Không suy luận chất lượng/run completion từ kích thước file khác nhau.

Chưa thể kết luận:

- Chỉ một seed và một official DTD split, chưa có mean/std qua nhiều lần train hoặc nhiều split.
- Chưa có toàn bộ prediction CSV để paired bootstrap/McNemar, phân tích per-class, theo mask hoặc xác định ảnh bị sửa/sai mới.
- Source giống nhau không tự chứng minh hash weight thực tế, hash toàn bộ pool hoặc mọi version/runtime asset giống hệt; cần protocol/checkpoint/pool export để xác minh độc lập đầy đủ.
- Prefix và Qwen khác coverage so với ba grounded controls; Qwen 8B và 32B cũng khác nhau. Muốn tách tác dụng nội dung khỏi availability cần matched-mask control riêng, không ghi đè các CSV/run hiện có.
- Qwen và Florence khác họ model, prompt và extraction; bảng này không tách riêng tác dụng scale, instruction tuning, fusion và semantic retrieval. Muốn kiểm tra riêng nội dung caption cần thêm đánh giá độc lập và can thiệp giữ nguyên checkpoint/context.
- Lexical policy và coverage 100% không chứng minh đã hết mọi shortcut, cũng không tương đương caption chính xác 100%.
- Không chọn lại hyperparameter/prompt bằng test. Các method tiếp theo cần chốt từ train/validation trước khi mở test; kết quả hiện tại là bằng chứng khám phá nếu dùng chúng để đề xuất vòng cải tiến tiếp theo.

## 7. Đoạn nhận xét dùng cho báo cáo

> Trên DTD split 1, seed 42 và budget 50 epoch, thay pipeline Florence bằng caption Qwen3-VL cải thiện kết quả của classifier M-TACS full_both. Qwen3-VL-8B-Instruct và 32B-Instruct đạt lần lượt 82.14% và 82.08% test macro-F1, cao hơn Florence-Prefix khoảng 1.08 và 1.01 điểm phần trăm. Chênh lệch giữa hai Qwen chỉ 0.066 điểm F1, chưa cho thấy lợi thế của model 32B. Trong nhóm Florence, Prefix và FirstValid gần tương đương, còn SingleGrounded/MultiGrounded chưa cải thiện downstream dù có coverage cao hơn. Tuy nhiên cả hai Qwen vẫn thấp hơn visual-only control 84.82% F1; do đó kết quả mới hỗ trợ khả năng cải thiện pipeline caption so với các Florence đã thử, chưa chứng minh thêm text tốt hơn nhánh visual trong thiết lập này. Vì mỗi phương pháp mới được đánh giá trên một seed/split và natural mask khác nhau, cần nhiều seed/split, matched-mask control và kiểm tra caption độc lập trước khi kết luận về độ ổn định, chất lượng mô tả hoặc label leakage.

## 8. Nguồn notebook

| Method | Notebook đã đọc |
| --- | --- |
| Prefix | [01_florence_prefix](../../../experiments/completed/caption_ablation/dtd/01_florence_prefix/mtacs-dtd-full-both-florence-prefix-split1-seed42-50ep.ipynb) |
| FirstValid | [02_florence_firstvalid](../../../experiments/completed/caption_ablation/dtd/02_florence_firstvalid/mtacs-dtd-full-both-florence-firstvalid-split1-seed42-50ep.ipynb) |
| SingleGrounded | [03_florence_singlegrounded](../../../experiments/completed/caption_ablation/dtd/03_florence_singlegrounded/mtacs-dtd-full-both-florence-singlegrounded-split1-seed42-50ep.ipynb) |
| MultiGrounded | [04_florence_multigrounded](../../../experiments/completed/caption_ablation/dtd/04_florence_multigrounded/mtacs-dtd-full-both-florence-multigrounded-split1-seed42-50ep.ipynb) |
| Qwen 8B | [05_qwen3_vl_8b](../../../experiments/completed/caption_ablation/dtd/05_qwen3_vl_8b/mtacs-dtd-full-both-qwen3-vl-8b-split1-seed42-50ep.ipynb) |
| Qwen 32B | [06_qwen3_vl_32b](../../../experiments/completed/caption_ablation/dtd/06_qwen3_vl_32b/mtacs-dtd-full-both-qwen3-vl-32b-split1-seed42-50ep.ipynb) |
| Florence cũ | [full-both.ipynb](../../../experiments/completed/text_context_ablation/dtd/full-both.ipynb) |
| Decoder LoRA | [caption-cai-tien.ipynb](../../../experiments/completed/caption_decoder_lora/dtd/caption-cai-tien.ipynb) |

Tên cũ `first-multi` được nhận diện là MultiGrounded bằng `caption_method=multibeam_grounded` và CSV hash, không suy đoán từ tên file. Hai thư mục `05_qwen3_vl_8b`/`06_qwen3_vl_32b` là số thứ tự **caption method**, cả hai vẫn train cấu hình kiến trúc `full_both`, không phải ablation `full_shuffle_both`.

Notebook chưa chạy dùng để train lại vẫn nằm trong `notebooks/02_improvements/caption_ablation/`; không ghi đè chúng bằng bản đã chạy. Chạy lại bộ trích xuất bằng `python tools/project_management/audit_dtd_caption_results.py` từ project TACS chỉ đọc notebook và cập nhật JSON bằng chứng, không thực thi cell.
