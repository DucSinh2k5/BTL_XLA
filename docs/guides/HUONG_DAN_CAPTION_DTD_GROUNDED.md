# Caption DTD C: Florence Gốc + CLIP Chọn Mô Tả Theo Ảnh

Notebook: `florence2-dtd-caption-grounded-rerank.ipynb`.

## 1. Khác Gì Hai CSV Trước?

| Bản | Cách tạo caption |
| --- | --- |
| Florence ban đầu | Sinh một mô tả, làm sạch phần mở đầu và lấy tối đa 8 từ |
| Florence decoder-LoRA | Fine-tune một phần mô hình bằng caption mục tiêu rồi sinh mô tả |
| Bản C mới | Giữ nguyên trọng số Florence; sinh nhiều mô tả, dùng CLIP so sánh từng đoạn ngắn với chính ảnh để chọn |

Bản C là cải tiến ở bước suy luận/chọn caption, **không phải một lần fine-tune mới**. Không cần `caption_references.csv`, PEFT, dịch vụ LLM hay Internet. Đây là một giả thuyết cần kiểm nghiệm, chưa có bằng chứng sẽ tăng macro-F1.

Luồng xử lý cho mỗi ảnh:

1. Florence gốc nhận pixel và task `<MORE_DETAILED_CAPTION>`, sinh một mô tả greedy cùng ba mô tả beam search.
2. Trích các đoạn liên tiếp 5–8 từ trong từng câu; loại tên lớp trực tiếp, một số đoạn mở/đóng không trọn ý và lặp từ.
3. CLIP mã hóa ảnh và từng đoạn văn bản, chọn đoạn có cosine ảnh–text cao nhất.
4. Xuất `captions.csv` theo policy `dtd_lexical_v2`. Điểm text–tên lớp chỉ để audit, không chọn caption và không dùng ngưỡng 0.30 để loại text.

Tên file, thư mục lớp, nhãn thật và caption mục tiêu không đi vào Florence hoặc điểm xếp hạng ảnh–text. Danh sách toàn bộ lớp chỉ được dùng sau generation để lọc từ lớp trực tiếp và audit, giống policy cũ. Alias vẫn chỉ là chẩn đoán; không cam kết loại hết gợi ý ngữ nghĩa về lớp.

**Ba beam là ba phương án mô tả của một ảnh, không phải ba ảnh context. M-TACS vẫn giữ K=1.**

## 2. Chạy Trên Kaggle Offline

Attach các input đã có:

- DTD: `/kaggle/input/datasets/jmexpert/describable-textures-dataset-dtd/dtd`
- Florence gốc: `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`
- Offline wheels: `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi`
- CLIP đầy đủ: `/kaggle/input/datasets/nguyenducxu/vit-32/model_clip_vit`

Bật GPU, mở notebook mới và chạy từ trên xuống. Cấu hình nằm ở cell code thứ hai, sau preflight. Nếu đường dẫn input khác, sửa tại đây.

Notebook tạo worker riêng với Transformers 4.49.x từ wheel offline; không hạ phiên bản Transformers trong kernel đang dùng. Wheel phải khớp Python/Linux của Kaggle. Không cần PEFT. Caption generation không dùng DataLoader multiprocessing và đặt `TOKENIZERS_PARALLELISM=false` từ đầu.

CLIP phải có **cả nhánh ảnh lẫn nhánh text**, không chỉ text encoder. Thiếu trọng số nhánh ảnh sẽ dừng, không dùng trọng số khởi tạo ngẫu nhiên. Nếu thiếu `preprocessor_config.json`, chỉ cấu hình CLIP chuẩn 224px/patch32 được phép dùng preprocessing mặc định có ghi lại trong `clip_preprocessing_used.json`.

Mặc định chạy đủ 5.640 ảnh của official DTD split 1. Để thử trước, đặt `MAX_NEW_IMAGES = 32`. Sau đó trả về `0` rồi chạy lại cấu hình và cell generation để tiếp tục toàn bộ ảnh còn thiếu. Không đổi protocol giữa chừng. Thay model, dataset, settings hoặc code thì dùng thư mục output mới.

CSV của smoke test nằm riêng trong `preview_incomplete`, không dùng để train. CSV chính chỉ xuất khi đủ ảnh. Mỗi ảnh hoàn tất được cache nguyên tử; nếu bị ngắt, chạy lại sẽ bỏ qua ảnh đã xong. Notebook này sinh và chấm nhiều phương án nên có thể chậm hơn generation greedy cũ; chưa có thời gian GPU đo thực tế.

## 3. Các Output

Thư mục mặc định:

`/kaggle/working/outputs/florence2_dtd_split1_grounded_caption_v1/`

| Đường dẫn bên trong | Ý nghĩa |
| --- | --- |
| `exports/captions.csv` | CSV chính của phương pháp mới |
| `controls/prefix_control/captions.csv` | Greedy + cách lấy phần đầu cũ, được sinh mới để đối chứng |
| `controls/first_valid_control/captions.csv` | Chọn đoạn hợp lệ đầu tiên trong cùng mô tả greedy, không CLIP ranking |
| `controls/single_grounded_control/captions.csv` | CLIP ranking trên cùng tập đoạn từ mô tả greedy |
| `caption_generation_summary.csv` | Độ phủ, từ lớp/alias, fallback và thống kê theo split |
| `candidate_audit.jsonl` | Các mô tả, đoạn ứng viên và điểm của chúng theo ảnh |
| `generation_manifest.json` | Hash ảnh, assets, code, runtime và protocol |

`raw_caption` và `final_caption` đều chứa đúng đoạn ngắn đã chọn, vì M-TACS ưu tiên đọc `raw_caption`. Mô tả dài gốc được lưu ở `source_long_caption`, kèm vị trí đoạn trích. Không có đoạn phù hợp thì ghi fallback rõ ràng; không tự điền một câu mặc định để tăng tỷ lệ hợp lệ.

CSV prefix đối chứng là bản **mới sinh theo quy trình cũ**, không khẳng định giống từng byte CSV lịch sử.

## 4. So Sánh Với Hai CSV Đã Có

Attach cả CSV Florence gốc và CSV decoder-LoRA vào Kaggle, điền đường dẫn chính xác tại cấu hình:

```python
OLD_CAPTIONS_CSV = "/kaggle/input/YOUR_OLD_DATASET/captions.csv"
LORA_CAPTIONS_CSV = "/kaggle/input/YOUR_LORA_DATASET/captions.csv"
```

Chạy phần comparison cuối notebook. Không bắt buộc hai file này để sinh caption mới. File cũ chỉ được đọc, không sửa. Notebook kiểm tra đủ image ID/split, so sánh text và mask **thực sự đi vào M-TACS theo dtd_lexical_v2**, không chỉ đếm cột `caption_valid` cũ.

Reports nằm trong `comparisons/<hash>/`. Các CSV `matched_masks/<method>/captions.csv` giữ text bật/tắt trên cùng tập giao hợp lệ giữa các phương pháp. Điều này giúp tách ảnh hưởng của độ phủ khỏi ảnh hưởng nội dung caption. Không phải phép can thiệp giữ nguyên context được chọn, vì selector semantic vẫn có thể chọn ảnh khác.

Muốn so sánh chỉ ba CSV chính thay vì kèm controls, sửa tập `comparisons` trước khi chạy phần comparison và giữ nguyên tập đó cho các lần train liên quan. Thay tập CSV làm thay đổi mask giao.

## 5. Kiểm Nghiệm Bằng M-TACS

Trước tiên dùng CSV chính mới với notebook `mtacs-train-vitl16-dtd-ablation-05-full-both.ipynb`, train classifier **từ đầu**, cache text và output mới. Không tiếp tục checkpoint classifier đã train bằng CSV cũ.

Trong cấu hình notebook 05:

```python
CAPTIONS_INPUT_PATH = "/kaggle/input/YOUR_NEW_DATASET/captions.csv"
CAPTION_INPUT_POLICY = "dtd_lexical_v2"
GENERATE_CAPTIONS_IF_MISSING = False
os.environ["DTD_FULL_BOTH_OUTPUT"] = "/kaggle/working/outputs/dtd_full_both_grounded_seed42"
```

Giữ official split 1, training seed 42, pool seed 42, batch 32, K=1, 50 epoch và chọn checkpoint theo validation loss. Kiểm tra batch thực tế vì notebook có cấu hình theo VRAM. So sánh độ phủ query/candidate, gate, context được chọn và metrics, không chỉ một số test F1.

| Kết quả đã quan sát, split 1 / seed 42 | Test macro-F1 |
| --- | --- |
| Visual-only | 0.848168 |
| Full-both + CSV gốc | 0.809124 |
| Full-both + CSV LoRA | 0.806707 |
| Full-both + CSV mới này | Chưa chạy |

CSV LoRA hiện thấp hơn CSV gốc ở lần chạy đã có, không nên gọi là đã cải thiện downstream. Với bản mới, điểm CLIP cao hơn là hệ quả của chính tiêu chí chọn, **không phải kiểm chứng chất lượng độc lập**. Mô tả vẫn có thể chung chung, sai chi tiết hoặc còn shortcut. Chọn protocol bằng train/validation, không tối ưu lặp lại theo test; dùng paired seeds để kiểm tra độ ổn định.

Nếu muốn phân tích cơ chế sâu hơn sau phép so sánh chính:

- `prefix_control` so với `first_valid_control`: ảnh hưởng của chọn lại đoạn ngắn.
- `first_valid_control` so với `single_grounded_control`: ảnh hưởng CLIP ranking trên cùng tập đoạn và cùng độ phủ.
- `single_grounded_control` so với CSV chính: ảnh hưởng của thêm beam hypotheses.
- So sánh tự nhiên và cùng mask: ảnh hưởng thay đổi độ phủ caption.

Notebook mới chỉ sinh/audit/so sánh CSV, không chạy lại tám ablation hay train M-TACS tự động. Các notebook và CSV cũ được giữ nguyên.

## 6. Tình Trạng Kiểm Thử

Đã kiểm thử bằng dữ liệu/mô hình giả lập CPU: số từ và bộ lọc, ghép beam với đúng ảnh, CLIP ranking theo ảnh thay vì nhãn, sinh tiếp sau ngắt/OOM, cache/fingerprint, mask tương thích policy thực tế, notebook tự chứa và giữ nguyên thí nghiệm cũ.

Chưa chạy Florence/CLIP thật trên GPU tại máy này, chưa đo tốc độ hoặc macro-F1 của CSV mới.
