# Rà Soát Kết Quả Cần Cho Báo Cáo TACS

Ngày rà soát: 22/10/2026. Phạm vi: project `TACS`, 78 notebook hiện có trong `notebooks/` và `experiments/`, các artifact và báo cáo lưu kèm.

> Cập nhật 24/10/2026: sáu notebook caption-method (bốn Florence và hai Qwen) đã có kết quả, xem [bảng so sánh và nhận xét đầy đủ](../caption_ablation/dtd/KET_QUA_SO_SANH_PHUONG_PHAP_CAPTION_DTD.md). Qwen 8B/32B đạt test Macro-F1 0.821445/0.820788; các notebook kết quả đã được chuyển vào `experiments/completed/caption_ablation/dtd/` theo method, giữ nguyên byte và output. Các nhận định "chưa có output" ở mục 9.2 bên dưới là trạng thái của lần rà soát trước khi nhận các file này, không phải yêu cầu train lại chúng.

> Bổ sung segmentation 24/10/2026: `tacs_isic_segement_448.ipynb` đã có test Dice 85.60%, IoU 77.38%. Xem [Table 3 segmentation đã cập nhật](../paper_tables/TABLE_03_SEGMENTATION.md): nhóm hàng TACS, hai cột ISIC 2017, ghi rõ biến thể 448/final-token cross-attention/DPT-lite và single-seed. Các nhận định chưa có output segmentation trong snapshot rà soát cũ không còn áp dụng cho run mới này.

> Bổ sung 25/10/2026: đã đọc `isic-segmen-captions.ipynb` dùng Qwen3-VL-8B full_both. Test Dice 84.8689%, IoU 76.5407%, thấp hơn visual khoảng 0.73/0.84 điểm phần trăm dù best validation Dice nhỉnh hơn. [So sánh và giới hạn](../paper_tables/SO_SANH_ISIC_VISUAL_VA_QWEN8B.md). Không dùng kết quả này để tuyên bố caption cải thiện segmentation.

## 1. Kết luận ngắn

**Không cần train lại cả 78 notebook hoặc chạy đủ 18 dataset của paper.** Cần hoàn thành một nhóm thí nghiệm có kiểm soát để trả lời: context có giúp không, selector học được có tốt hơn similarity không, và hai đường tối ưu có đóng góp gì.

Đề xuất phạm vi mặc định:

- **Phần tái lập/điều chỉnh TACS:** SUN397 và Stanford Cars. Hai dataset này đã có nhiều code liên quan, đại diện scene và fine-grained object classification. Đây là lựa chọn theo độ sẵn sàng của project, không phải bắt buộc phải chọn hai dataset này.
- **Phần cải tiến caption/M-TACS:** dùng DTD làm nghiên cứu mở rộng riêng, vì đã có tám run ablation và bốn CSV control. Không trộn kết quả ViT-L/16 DTD với bảng TACS ViT-S/16 để quy chênh lệch cho caption.
- **CUB:** đã có ba notebook no-context theo seed; có thể giữ làm sanity check hoặc thay Cars nếu muốn, nhưng hiện thiếu kết quả các method context tương ứng.
- **Segmentation/medical:** không bắt buộc nếu báo cáo tuyên bố rõ chỉ đánh giá classification trên một số dataset. Không tuyên bố kết luận cho segmentation khi chưa có thí nghiệm.

**Việc cần làm trước khi train: thống nhất protocol và sửa/kiểm tra các điểm không đồng nhất. Không chạy ngay mọi notebook có chữ `tacs`.** Tên file, tên folder, hoặc điểm test cao không tự chứng minh run phù hợp với bảng chính.

## 2. Cơ sở và giới hạn rà soát

- Đọc JSON/source/output của 78 notebook, cấu hình và lỗi cell; không thực thi cell, không tải model và không mở checkpoint `.pt` bằng pickle.
- Đối chiếu các mục 4.1–4.6 và Tables 2–5, Figure 4 trong [paper local](<../../docs/paper/18. Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.pdf>).
- Trích lại metric từ output stream và bảng HTML. Stream bị Jupyter chia thành nhiều mảnh được nối theo stdout/stderr, tránh đọc nhầm `test_accuracy: 0.` là kết quả hoàn chỉnh.
- Bằng chứng máy đọc được: [notebook_evidence_inventory.json](notebook_evidence_inventory.json). Cấu hình source chỉ là cấu hình được lưu, không chứng minh toàn bộ trạng thái runtime cũ; các cảnh báo tĩnh là đầu mối kiểm tra, không phải kết luận lỗi tự động.
- Đây là audit độ sẵn sàng cho báo cáo và các điểm protocol nổi bật, không phải chứng nhận từng dòng mô hình/gradient của mọi notebook đã đúng paper.
- Chưa tìm thấy metric JSON/prediction CSV/protocol gốc của các run chính trong phần project đã khảo sát; phần lớn điểm hiện nằm trong output notebook. Không khẳng định chúng không tồn tại ở Kaggle hoặc vị trí khác ngoài project.

## 3. Kết quả thật đang có

Các điểm dưới đây là **điểm hiển thị trong output**, chưa phải các hàng của một bảng so sánh công bằng. Accuracy và F1 trong bảng này quy đổi sang phần trăm.

| Nhóm | Bằng chứng hiện có | Accuracy | Macro-F1 | Cách dùng hiện tại |
| --- | --- | ---: | ---: | --- |
| CUB no-context, notebook seed 42 | `baseline_no_tacs_image_classification (seed42).ipynb` | 82.46 | 82.57 | Giữ; xác minh artifact/seed trước khi tổng hợp |
| CUB no-context, notebook seed 43 | `baseline_no_tacs_image_classification (seed43).ipynb` | 82.27 | 82.32 | Giữ; không tính thêm bản basename chung như một seed mới |
| CUB no-context, notebook seed 44 | `baseline_no_tacs_image_classification_(seed_44).ipynb` | 82.53 | 82.54 | Giữ; không cần train lại chỉ vì đang sắp xếp folder |
| Cars no-context, seed 44 | `seed44-car.ipynb`, S/16, 256, full train, final checkpoint | 91.59 | 91.43 | Có baseline cũ; chưa cùng protocol với các bản context 448/validation monitor |
| Cars random, bản `baseline_random_context_ret_cars` | S/16, 256 | 1.79 | 0.61 | Run bất thường cần kiểm tra training/nhãn; không bỏ qua chỉ vì điểm thấp và không dùng làm bằng chứng TACS tốt hơn khi chưa hiểu nguyên nhân |
| Cars random, bản `ban-cuoi-Random Context Retrieval baseline` | S/16, 448, full train | 94.38 | 94.29 | Giữ làm run lịch sử; chuẩn hóa trước khi so với baseline 256 |
| Cars frozen, bản `ban-cuoi-frozen-dino-car` trong folder frozen | S/16, 448, K=4, pool 100%; có KeyboardInterrupt/FileNotFoundError | 93.60 | 93.45 | Không phải frozen K=1/pool 20% trong bảng mong muốn; run cũ bị ngắt |
| Cars TACS, bản `ban-cuoi-frozen-dino-car` trong folder TACS | S/16, 448, shortlist 4; lỗi NameError tại cell inference sau đó | 94.98 | 94.90 | Điểm test có lưu; lỗi inference sau test không tự xóa điểm, nhưng variant/protocol chưa phù hợp bảng chính |
| Cars TACS, file mang tên `tacs-chuanpaper-sun-b16 (2)` | B/16, 384, có KeyboardInterrupt | 95.15 | 95.03 | Lưu như run B/16; không ghép với baseline S/16 |
| SUN TACS S/16 | `tacs-sun-da-sua (2).ipynb`, 256, true validation | 74.01 | 73.85 | Có điểm test, early stopping tại epoch 23; cần baseline tương ứng và audit self-exclusion |
| SUN TACS B/16 | `tacs-chuanpaper-sun-b16 (1).ipynb`, 384, true validation | 79.42 | 79.27 | Có điểm test; phải có baseline B/16/384 mới quy khác biệt cho selector |
| SUN query-caption S/16 | `mtacs-train-captions.ipynb` | 76.12 | 76.02 | Run mở rộng cũ; không thay cho kết quả ViT-L/16 hiện chưa có output |
| DTD tám text-context ablation | Tám notebook đã chạy, split 1, seed 42, L/16, 50 epoch | Xem bảng dưới | Xem bảng dưới | Đã đủ điểm cho bảng single-seed; thiếu artifact độc lập và độ ổn định qua seed |
| DTD classifier với caption decoder LoRA | `caption-cai-tien.ipynb`, full_both, seed 42, 50 epoch | 80.7447 | 80.6707 | Có điểm classifier, không phải output training LoRA captioner |

### ISIC 2017: bổ sung kết quả segmentation

| Run | Test Dice (%) | Test IoU (%) | Cách dùng |
| --- | ---: | ---: | --- |
| TACS tái triển khai điều chỉnh, 448, seed 42 | 85.60 | 77.38 | Điền hai ô TACS × ISIC 2017 trong bảng project; không gọi là nguyên bản paper hoặc đối chứng cùng protocol |

Train/val/test runtime = 2.000/150/600; candidate pool 400 ảnh train; best validation Dice tại epoch 49, early-stop ở epoch 64/100. Source dùng frozen ViT-S/16 và DPT-lite sau final-token cross-attention, khác paper 256/blockwise fusion. Chưa có baseline 448 cùng protocol để kết luận lợi ích selector. [Bảng và bằng chứng chi tiết](../paper_tables/TABLE_03_SEGMENTATION.md).

### DTD: giữ bảng ablation đã có

| Cấu hình | Test Accuracy (%) | Test Macro-F1 (%) |
| --- | ---: | ---: |
| visual_only | 84.9468 | 84.8168 |
| query_only | 82.3404 | 82.3067 |
| candidate_only | 85.1596 | 85.0321 |
| both_fusion | 81.1702 | 81.0661 |
| full_both | 81.0106 | 80.9124 |
| full_shuffle_both | 83.4043 | 83.3392 |
| full_unscreened | 79.0426 | 78.9774 |
| full_redacted | 81.2766 | 81.1105 |

Nguồn: [bằng chứng tám run](../../experiments/completed/text_context_ablation/dtd/analysis/saved_output_evidence.json) và [báo cáo đã phân tích](../../experiments/completed/text_context_ablation/dtd/analysis/DANH_GIA_ABLATION_DTD_01_07.md).

`visual_only` vẫn nhận ảnh context, **không phải No-Context của paper**. Candidate-only hơn visual-only khoảng 0.2153 điểm phần trăm F1, chưa đủ kết luận ổn định từ một seed. Full-both với caption LoRA hiện thấp hơn full-both cũ khoảng 0.2417 điểm F1; không gọi caption fine-tune là cải thiện downstream đã được chứng minh.

## 4. Những điểm phải thống nhất trước khi dùng GPU

### 4.1. Backbone, resolution, pool, budget đang khác nhau

- Paper dùng ViT-S/16 DINOv3, ảnh 256, budget 100 epoch và tổng hợp ba seed. Project có S/16, B/16, L/16, ảnh 256/384/448, budget 50/60/100 và nhiều batch khác nhau.
- Cars frozen 93.60% dùng pool toàn bộ train và K=4. Cars random/TACS bản 448 không thể ghép trực tiếp với no-context 256 để kết luận tác động riêng của retrieval.
- Không chọn variant theo điểm test cao nhất. Chọn protocol trước, giữ nguyên cho mọi method, rồi báo cáo cả kết quả tốt/lẫn kém.

### 4.2. Training monitor không phải validation độc lập

Trong các bản `TRAIN_ON_FULL_TRAIN=True`, source có nhánh lấy `monitor_val_df` từ `full_train_df`, sau đó dùng toàn bộ `full_train_df` để train và `monitor_val_df` làm `val_df`. Monitor này nằm trong training data.

Điều đó **không tự chứng minh đã dùng test để train**, nhưng monitor không được trình bày như validation độc lập để chọn model. Trái lại, no-context Cars hiện dùng full official train và checkpoint epoch cuối, không dùng validation/early stopping. Hai cách chọn checkpoint này đang không đồng nhất.

Chọn một trong hai hướng, không trộn:

1. **Khuyến nghị cho nhóm thí nghiệm mới:** chia validation từ official train, khóa cùng train/val IDs cho mọi method; candidate chỉ từ actual-train; chọn checkpoint theo cùng validation criterion; official test giữ nguyên.
2. Hoặc dùng toàn bộ official train, fixed budget và final checkpoint cho mọi method; không dùng monitor trong train để tuyên bố khả năng generalization, không tune trên test. Nếu dùng hướng này phải ghi rõ trong báo cáo.

### 4.3. Self-exclusion và cấu trúc selector cần kiểm tra

- Các source full-pool SUN và Cars B/L đã xem có `select_full_pool(query_images, candidate_feature_bank, use_gumbel=True)` nhưng không nhận query IDs/excluded positions tại API này. Cần xác nhận/hoàn thiện mask theo image ID để query train không chọn chính nó khi nó là candidate. Không suy ra đã xử lý chỉ vì có hàm tên `select_full_pool`.
- Shortlist 4 candidate, frozen embedding selector và full-pool shared-backbone selector là các variant khác nhau. Không dùng chung tên TACS mà không mô tả khác biệt.
- Giữ head/fusion, augmentation, loss phụ như SupCon, scheduler và LR nhất quán khi ablate; nếu một method thêm loss/gate thì phải công bố và giải thích.
- Nếu GPU buộc phải đổi batch, khóa cùng thiết lập giữa method. Gradient accumulation không làm SupCon trên các microbatch tự động tương đương một batch lớn.
- Các baseline Cars dùng định dạng CSV hoặc MATLAB khác nhau: cần khóa cùng ảnh, class mapping và split manifest; không chỉ kiểm tra tổng số ảnh.

## 5. P0: bảng kết quả chính cần hoàn thành

### Phạm vi tối thiểu có trọng tâm

Hai dataset SUN397 và Cars, ba method **No-Context / Frozen DINO Similarity / TACS**. Đây là sáu ô kết quả, chưa phải sáu notebook hiện đều sẵn sàng Run All.

| Method | SUN397 | Stanford Cars | Vai trò |
| --- | --- | --- | --- |
| No-Context | Chưa có output của notebook SUN | Có seed 44 cũ; cần xác minh/đồng bộ protocol mới | Mốc không dùng context |
| Frozen DINO Similarity | Chưa có notebook thực nghiệm tương ứng trong folder SUN | Có code và run cũ khác K/pool/size | Kiểm tra learned selection có hơn similarity |
| TACS | Có run S/16 và B/16, cần baseline/protocol tương ứng | Nhiều variant, chưa có bộ cùng protocol với hai baseline | Method cần đánh giá |

Nên dùng **S/16, 256** cho nhóm tái lập chính vì gần paper và giảm chi phí so với B/L. Nếu chỉ đủ 50 epoch, dùng 50 cho toàn nhóm và ghi rõ đây là thí nghiệm điều chỉnh, không gọi là tái lập nguyên cấu hình 100 epoch.

- Chạy một seed pilot để kiểm tra split, loss, gradient selector và xuất file; pilot không dùng để chọn method theo test.
- Sau khi code/protocol ổn định, chạy seeds 42/43/44 cho các ô chính: **6 cấu hình × 3 seed = 18 run**. Chỉ tái sử dụng run cũ nếu xác minh được cùng protocol; nếu đổi protocol thì phải train mới, không resume checkpoint không tương thích.
- Nếu chỉ báo cáo một seed, vẫn có thể viết một báo cáo phạm vi hẹp nhưng phải ghi single-seed, không bịa std hoặc tuyên bố ổn định qua seed.

### Bảng đầy đủ hơn theo tinh thần Table 2

Thêm **Random Context** và **Feature-Averaged Context** trên hai dataset: tổng **5 method × 2 dataset × 3 seed = 30 run**. Không cộng lại 18 run tối thiểu; 30 đã bao gồm chúng.

| Baseline bổ sung | SUN397 hiện tại | Cars hiện tại |
| --- | --- | --- |
| Random Context | Chưa có notebook trong slot SUN | Có nhiều bản; cần chuẩn hóa và kiểm tra run 1.79% |
| Feature-Averaged Context | Chưa có notebook trong slot SUN | Có source, chưa có output |

Không cần thêm CUB/DTD vào bảng chính chỉ để tăng số cột nếu phải tạo nhiều implementation chưa được kiểm tra.

## 6. Notebook nguồn và hành động cụ thể

Đây là **nguồn để chuẩn hóa**, không phải xác nhận mọi file sẵn sàng train ngay. Mỗi nhóm method chỉ cần một bản canonical cho mỗi dataset/protocol, không chạy mọi bản trùng vai trò.

| Việc | Notebook nguồn hiện có | Hành động trước khi train |
| --- | --- | --- |
| SUN No-Context | [baseline_no_tacs_cars.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/sun397/baseline_no_tacs_cars.ipynb) | Tên cũ gây nhầm nhưng dataset là SUN; đồng bộ split/checkpoint criterion với TACS |
| SUN TACS S/16 | [tacs-sun-true-val-split.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/tacs-sun-true-val-split.ipynb) | Kiểm tra self-exclusion, gradient và xuất artifact; không chạy thêm mọi bản B/L |
| SUN TACS có xuất checkpoint Figure 4 | [tacs_train_with_checkpoints.ipynb](../../notebooks/01_paper_reproduction/figure_04_attention/tacs/sun397/tacs_train_with_checkpoints.ipynb) | Có thể chọn làm run TACS chính nếu cùng protocol; không cần train hai lần chỉ vì nằm ở hai folder |
| SUN Frozen / Random / Feature-Average | Chưa có notebook tương ứng trong slot dataset | Cần tạo/adapt từ baseline Cars, kiểm tra dataset/class IDs/fusion trước khi train |
| Cars No-Context | [seed44-car.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/stanford_cars/seed44-car.ipynb) | Dùng làm source; không chỉ đổi SEED rồi chạy nếu protocol mới dùng true validation |
| Cars Frozen S/16 256 | [baseline_frozen_dino_similarity_ret_cars.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/frozen_dino_similarity/stanford_cars/baseline_frozen_dino_similarity_ret_cars.ipynb) | Khóa K/pool/split/selection rule tương ứng; không lấy run K=4/pool 100% làm baseline K=1 |
| Cars Random S/16 256 | [baseline_random_context_ret_cars.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/random_context/stanford_cars/baseline_random_context_ret_cars.ipynb) | Kiểm tra train/labels/forward vì run cũ 1.79%; không chọn bản 448 chỉ do có điểm cao hơn |
| Cars Feature-Average | [baseline_feature_averaged_context_cars.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/feature_averaged_context/stanford_cars/baseline_feature_averaged_context_cars.ipynb) | Chưa có output; kiểm tra aggregation K và đồng bộ training recipe |
| Cars TACS S/16 256 | [baseline_tacs_cars.ipynb](../../notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/stanford_cars/baseline_tacs_cars.ipynb) | Source cũ; audit selector/backbone sharing và training path; tốt hơn là adapt bản full-pool SUN đã chuẩn hóa sang Cars thay vì coi các variant tương đương |
| CUB no-context | [folder CUB](../../notebooks/01_paper_reproduction/classification/table_02_main_results/no_context/cub200/) | Giữ ba file seed 42/43/44; thu metrics gốc, không cần train lại chỉ để lập bảng |

## 7. P1: ablation quan trọng của TACS

### 7.1. Hai đường tối ưu: ưu tiên cao nếu giải thích cơ chế TACS

Slot [table_05_optimization](../../notebooks/01_paper_reproduction/classification/table_05_optimization/) hiện chỉ có `.gitkeep`, **chưa có notebook chuyên biệt và chưa có output**.

Cần trên một dataset đại diện, ví dụ SUN397:

| Variant | Cần thêm train? |
| --- | --- |
| Frozen retriever | Dùng lại run baseline của bảng chính |
| Differentiable selection only, không policy objective | Cần notebook và run riêng |
| Policy selection only, không đường gradient qua Gumbel | Cần notebook và run riêng |
| Full dual-path TACS | Dùng lại run TACS của bảng chính |

Như vậy thêm **2 variant × 3 seed = 6 run**, không phải train lại bốn hàng. Không chỉ đổi tên mode: cần kiểm tra loss và đường gradient thực sự bị tắt đúng, còn các thành phần khác giữ nguyên. Bảng trên một dataset không thay thế hàng trung bình 11 dataset trong paper.

### 7.2. Cách ghép context: tốt cho giải thích lợi ích context

Ba notebook SUN đã có source nhưng **chưa có output**:

- [Blank image](../../notebooks/01_paper_reproduction/classification/table_04_context_pairs/blank_image/sun397/table4_blank_image_sun.ipynb).
- [Duplicate query](../../notebooks/01_paper_reproduction/classification/table_04_context_pairs/duplicate_query/sun397/table4_duplicate_query_sun.ipynb).
- [Noisy query](../../notebooks/01_paper_reproduction/classification/table_04_context_pairs/noisy_query/sun397/table4_noisy_query_sun.ipynb).

Chạy cùng protocol với SUN TACS/baselines: thêm **3 variant × 3 seed = 9 run**. No-context, similar image và TACS context dùng lại từ bảng chính. Cần đảm bảo cùng head/token/fusion để không quy tác động kiến trúc cho việc thay context. Duplicate/noisy là đối chứng cố ý dựa vào query, không áp mask như khi chọn từ candidate pool thật.

## 8. P1/P2: hình và phân tích, không mặc định cần retrain

### Figure 4: một nhóm hình dễ hiểu cho báo cáo

- Cần checkpoint TACS đã xác minh, class map, candidate pool và feature bank đúng với checkpoint. Hai bundle hiện ở `artifacts/checkpoints/imported_unverified` chưa đủ metadata để xác nhận tương thích; không tự coi chúng là checkpoint của run mới.
- Từ cùng run TACS chính, lưu query, selected candidate, nhãn thật/dự đoán và attention query/context. Lấy cả ví dụ đúng và sai theo quy tắc công bố trước, không chỉ các ảnh đẹp nhất.
- Không cần thêm training nếu checkpoint hiện có phù hợp hoặc run chính đã xuất đủ. Nếu chưa có thì thêm export vào run chính, không mở một nghiên cứu mới chỉ để có hình.
- Công cụ đã có ở `tools/visualization/`; attention hiện là cách trực quan hóa kiểu Figure 4, không chứng minh causality hay mặc định đúng hệt quy trình tác giả.

### Figure 3 / phân tích selector: bổ sung nếu đủ thời gian

- So sánh same-class/cross-class selection rate của Frozen và TACS trên cùng query IDs/candidate pool.
- LPIPS có thể đánh giá từ các cặp đã chọn, không cần train lại model; công bố backbone LPIPS/preprocessing và sample/query set cố định.
- Nếu hạn chế thời gian, dùng bảng same-class rate + một số cặp minh họa. Không tuyên bố đã tái lập LPIPS khi chưa tính.

## 9. Phần caption: làm gì và không cần làm gì

### 9.1. Tám cấu hình text-context DTD

**Không cần mặc định train lại cả tám bản.** Đã có output completed seed 42, 50 epoch cho bảng thăm dò. Cần tải/import artifact gốc trước: protocol, history, predictions, caption input CSV và hash, pool IDs, coverage, shuffle/redaction audit. Notebook/biểu đồ không thay thế tất cả các artifact đó.

Nếu muốn phát biểu có độ ổn định, ưu tiên thêm seeds 43/44 cho **visual_only, candidate_only, full_both**: thêm sáu run. Việc chọn nhóm này là kế hoạch xác nhận/giới hạn compute dựa trên kết quả đã quan sát, không phải một benchmark mới còn hoàn toàn mù test. Công bố tính exploratory và khóa cấu hình; không tiếp tục chọn CSV/hyperparameter theo test. Nếu báo cáo mọi control với mean/std thì cần đủ seed cho tất cả các hàng.

### 9.2. Bốn CSV-method classifier

Các notebook dưới [caption_ablation](../../notebooks/02_improvements/caption_ablation/) **đều chưa có output lưu trong project**:

1. Florence-Prefix.
2. Florence-FirstValid.
3. Florence-SingleGrounded.
4. Florence-MultiGrounded.

Đây là **bốn run ưu tiên nếu báo cáo có đóng góp về caption**. Chạy seed 42 cùng DTD split 1, L/16, 50 epoch, K=1 và full_both theo notebook hiện có; sau đó xác nhận bằng thêm seed theo budget đã chọn. Không cần sinh lại bốn CSV chỉ để chạy classifier, không gọi đây là bốn lần fine-tune Florence.

Không tự điền điểm full_both cũ vào hàng Florence-Prefix mới dù coverage giống nhau; cần trùng CSV hash và protocol. Prefix có coverage 4.749/5.640 còn ba control sau 5.640/5.640: phải báo cáo khác biệt mask/coverage, không quy mọi delta cho chất lượng câu chữ.

### 9.3. Caption LoRA và grounded generator

- Hai notebook [decoder LoRA](../../notebooks/02_improvements/caption_generation/decoder_lora/dtd/florence2-dtd-caption-finetune.ipynb) và [grounded rerank](../../notebooks/02_improvements/caption_generation/grounded_rerank/dtd/florence2-dtd-caption-grounded-rerank.ipynb) hiện là bản source không lưu output.
- CSV đã có không buộc phải train/sinh lại captioner. Ưu tiên tìm notebook đã chạy, adapter/config, hash references, fit/holdout IDs, training history và export summary từ Kaggle.
- `caption-cai-tien.ipynb` chứng minh classifier đã chạy với một CSV, không chứng minh chi tiết quá trình fine-tune captioner.
- Reference hiện có nguồn teacher AI; không tự mô tả thành ground truth do con người kiểm duyệt độc lập. CLIP score được dùng để rerank không phải metric chất lượng caption độc lập.

## 10. Các phần có thể hoãn

| Nhóm | Vì sao chưa cần train ngay |
| --- | --- |
| TACS B/16, L/16 | Không cần nhiều backbone nếu câu hỏi chính là context/selector; nếu dùng phải có baseline cùng backbone |
| Visual top-3 / multimodal top-3 / adaptive context | Là các câu hỏi mở rộng khác, không cần để chứng minh bản K=1 tái lập được |
| SigLIP zero-shot fusion | Nhánh mở rộng riêng, không thay cho các baseline TACS |
| CUB đầy đủ context methods | Chỉ làm khi chọn CUB vào phạm vi chính; baseline ba seed riêng chưa chứng minh hiệu quả TACS trên CUB |
| ISIC2017 segmentation | Đã bổ sung run TACS 448: Dice 85.60%, IoU 77.38%; xem Table 3 mới. Nếu báo cáo segmentation, còn cần No-Context/Frozen cùng protocol để đánh giá tác dụng retrieval; không bắt buộc mở rộng thêm dataset nếu scope hẹp |
| `90_archive` | Không train lại các bản lịch sử để tăng số lượng kết quả |
| Web demo | Không phải thí nghiệm thay thế accuracy/ablation; dùng checkpoint đã kiểm chứng từ run chính |

## 11. Bộ artifact cần xuất sau mỗi run

Tên file có thể khác giữa notebook; các trường thông tin mới là bắt buộc. Những mục dưới đây là checklist cần có, không phải khẳng định project đã có chúng.

| Nhóm | Cần lưu |
| --- | --- |
| Danh tính run | Dataset, method, seed, split number, train/val/test IDs hoặc hash; class mapping |
| Protocol | Backbone/weights hash, resolution, transforms, batch/accumulation, LR, loss, budget, early stopping/checkpoint criterion, thư viện, candidate fraction/K |
| Metrics | Test top-1 accuracy chính; macro-F1 bổ sung; validation metric/criterion và best epoch; không lấy test-best epoch |
| Learning curves | History từng epoch: train/val loss và accuracy/F1; TACS thêm policy loss, reward, entropy và gradient loss |
| Predictions | Mỗi ảnh: image_id, ground truth, predicted class, probability/logits nếu có; tất cả ảnh, không chỉ `head()` |
| Retrieval | Candidate pool IDs, selected context IDs trên cùng query list, selector scores, self-mask audit, selected same-class rate |
| Chi phí | Thời gian train/inference, GPU và peak VRAM nếu báo cáo đánh đổi hiệu năng/chi phí |
| Checkpoint | Checkpoint được chọn + config + class map; giữ last checkpoint/RNG/optimizer nếu cần resume |
| Text, nếu dùng | CSV hash, caption policy, input masks, text embedding identity, coverage cho query/pool, nguồn caption/teacher |

Không bắt buộc vẽ toàn bộ confusion matrix 397×397 trong thân báo cáo; per-class CSV và phân tích một số nhóm lỗi có thể dễ đọc hơn. Prediction đầy đủ giúp tính paired bootstrap/McNemar và phân tích lỗi mà không train lại. CI từ một tập test không thay thế độ biến thiên qua training seed.

## 12. Bộ bảng/hình đủ cho báo cáo phạm vi hẹp

1. **Bảng protocol:** dataset sizes/splits, candidate pool, backbone, resolution, epochs, seeds và các khác biệt so với paper.
2. **Bảng chính:** ít nhất No-Context/Frozen/TACS trên hai dataset; đầy đủ hơn thì năm method của Table 2. Report mean ± sample std nếu có ba seed; nếu một seed thì ghi rõ.
3. **Bảng tối ưu selector:** một dataset, hai ablation mới và hai hàng dùng lại.
4. **Bảng context-pair:** ba control SUN mới và ba hàng dùng lại, nếu scope/budget cho phép.
5. **Hình learning curves + cặp query/context/attention:** dùng checkpoint của bảng chính.
6. **Nếu có phần cải tiến:** bảng DTD tám text-control và bảng bốn CSV-method là hai câu hỏi khác nhau; báo cáo riêng, kể cả khi không cải thiện.

Đối chiếu số paper trong cột riêng có ghi rõ thiết lập, không lấy các số paper làm số đo của mình. Với SUN/Cars/CUB/DTD trong Table 2, dùng accuracy tương ứng; không đặt macro-F1 vào cùng cột để kết luận vượt paper.

## 13. Thứ tự hành động đề xuất

1. Chốt phạm vi chính SUN397 + Cars, hoặc thay Cars bằng CUB nếu đó là dataset muốn đầu tư; xác định caption là phần mở rộng có đưa vào báo cáo hay không.
2. Chuẩn hóa một bộ S/16/256 và shared split/pool/checkpoint rule; kiểm tra self-exclusion và selector gradient, chạy smoke test trước khi commit GPU dài.
3. Lấy bảng chính tối thiểu ba method; giữ đủ artifact ngay từ seed đầu, sau đó thêm seed theo budget.
4. Bổ sung Table 5 trên một dataset; thêm Table 4 và hai baseline nếu cần báo cáo đầy đủ hơn. Không lặp lại các hàng đã có.
5. Tạo hình/selector analysis từ checkpoint của các run trên, không mặc định retrain.
6. Song song, thu artifact gốc tám run DTD; nếu làm caption, chạy bốn CSV-method notebook chưa có output. Chưa cần tạo một captioner mới.

**Tóm lại:** khoảng thiếu lớn nhất không phải số lượng notebook mà là một tập kết quả khóa cùng protocol và có prediction/manifest đi kèm. Một báo cáo trung thực với hai dataset, đối chứng đúng và artifact đầy đủ có giá trị hơn nhiều bảng ghép điểm của các backbone/split khác nhau.
