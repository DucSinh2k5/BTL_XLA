"""Build a standalone DTD notebook for query/candidate caption ablations."""
from __future__ import annotations

from _project_bootstrap import PROJECT_ROOT, project_path

import ast
import copy
import hashlib
import json
import re
import textwrap
from pathlib import Path


ROOT = PROJECT_ROOT
SOURCE = project_path('mtacs-train-captions-vitl16-dtd.ipynb')
HELPERS = project_path('mtacs_dtd_ablation_cells.py')
TARGET = project_path('mtacs-train-captions-vitl16-dtd-text-ablation.ipynb')


def source_of(cell):
    return "".join(cell.get("source", []))


def make_cell(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one source marker: {old!r}")
    return source.replace(old, new, 1)


def helper_sections():
    parts = re.split(r"^# %% (\w+)\s*$", HELPERS.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


INTRO = """
# DTD: ablation text cho query và candidate pool

Notebook được tách từ `mtacs-train-captions-vitl16-dtd.ipynb`. Giữ nguyên dataset, ViT-L/16,
preprocessing, candidate pool 20% train và captioner hiện tại; bổ sung caption của candidate vào
selector/fusion để kiểm tra vai trò của text cho cả query và candidate. Bản nguồn chỉ fuse caption query.
Mọi cấu hình dùng cùng một lớp model, classifier head, khởi tạo, ngân sách train và pool; nhánh text
bị tắt bằng mask. `visual_only` là baseline cùng kiến trúc cho ablation này, không phải tái lập chính xác
mọi lựa chọn của paper TACS.

DTD có 47 lớp, 40 ảnh/lớp cho mỗi train, validation, test trong một official split: 1.880/1.880/1.880.
Candidate pool gồm 376 ảnh chỉ từ train; query nằm trong pool bị loại khỏi chính danh sách candidate của nó.
Nguồn split: https://www.robots.ox.ac.uk/~vgg/data/dtd/. Một lần chạy chỉ là một split; so sánh benchmark
10 splits cần train độc lập split 1..10 và tổng hợp mean/std, không gộp các split vào một tập train.

Không thay prompt/model sinh caption ở thí nghiệm này. Ưu tiên tái sử dụng `captions.csv` DTD cũ;
tự tạo/resume CSV nếu còn thiếu khi `GENERATE_CAPTIONS_IF_MISSING=True` (mặc định). Model sinh caption nhận pixels, không nhận
label, tên file hay đường dẫn ảnh. Các từ chỉ lớp được dùng cho audit/redaction toàn bộ 47 lớp, không dùng
nhãn thật của từng ảnh để xây caption. Không đưa label candidate vào selector hoặc fusion.
"""

EXPERIMENTS = """
## Các thí nghiệm đã định trước

| Cấu hình | Text query vào fusion | Text candidate vào fusion | Text vào selector | Caption |
|---|---|---|---|---|
| `visual_only` | Không | Không | Không | Nhánh text bị mask |
| `query_only` | Có | Không | Không | Screened |
| `candidate_only` | Không | Có | Không | Screened |
| `both_fusion` | Có | Có | Không | Screened |
| `full_both` | Có | Có | Có | Screened |
| `full_shuffle_both` | Có | Có | Có | Đảo caption cả hai vai trò, giữ mask |
| `full_unscreened` | Có | Có | Có | Caption trước lọc, đối chứng shortcut |
| `full_redacted` | Có | Có | Có | Che tên lớp/alias trong cả query và candidate |

`query_only`/`candidate_only`/`both_fusion` tách vai trò text downstream trong cùng họ selector thị giác.
`full_both` so với `both_fusion` kiểm tra đóng góp thêm của semantic selector. Các model train độc lập có thể
học cách chọn ảnh khác nhau; cell intervention giữ nguyên ảnh context trên cùng checkpoint để tách riêng ảnh hưởng fusion.

`screened` dùng nguyên caption đã tạo, mask CSV, lexical check và ngưỡng CLIP cố định 0.30.
Giữ policy lọc của bản nguồn; alias mở rộng dùng cho audit/redaction và term-only probe.
`unscreened` giữ `final_caption` trước áp mask, bỏ các cờ từ chối chỉ cho đối chứng; đây không phải
`raw_caption` đầy đủ và không phục hồi được text đã bị xóa khỏi CSV. `redacted` xóa các tên lớp và
alias trong toàn bộ caption. Raw/redacted dùng cùng availability mask; redacted được phép ngắn hơn 5 từ
để phép che từ không bị biến thành phép loại cả caption. Danh sách alias chỉ là heuristic có thể mở rộng trước khi mở test.
So screened với unscreened thay đổi tỷ lệ caption được dùng; cần đọc cùng coverage, không coi chênh lệch đó
là ảnh hưởng thuần của nội dung từ chỉ lớp. Cặp unscreened/redacted đã khớp mask để kiểm tra việc che từ.

Shuffle không dùng nhãn: chỉ hoán vị giữa caption hợp lệ, trong cùng split và cùng nhóm pool/non-pool,
caption candidate chỉ lấy từ pool train. Giữ nguyên mask từng ảnh, lưu ID caption donor. Nhóm chỉ có một
caption hợp lệ không thể derange; trường hợp này được ghi ra manifest, không tự đổi tỷ lệ caption hợp lệ.
Shuffle query trên val/test là đối chứng cố ý phá liên kết ảnh-text giữa các ảnh trong cùng split,
không phải pipeline inference độc lập từng ảnh để triển khai; candidate vẫn chỉ lấy caption từ train pool.

**Cách đọc leak:** DTD dùng chính các thuộc tính texture làm tên lớp. Việc caption sinh từ pixels nói đúng
"striped" không chứng minh model đã đọc nhãn test. Drop sau redaction có thể do mất tín hiệu thị giác hợp lệ.
Caption đảo, text-only probe, mask-only probe, tỷ lệ từ chỉ lớp và intervention giúp phát hiện shortcut;
không phép kiểm riêng lẻ nào chứng minh hoàn toàn không leak. CLIP similarity không phải xác suất leak.
"""

CONFIG = r'''
# Protocol: finalize the plan on validation before looking at final test results.
AB_CONFIG = {
    "visual_only": dict(query_fusion=False, candidate_fusion=False, semantic_selector=False),
    "query_only": dict(query_fusion=True, candidate_fusion=False, semantic_selector=False),
    "candidate_only": dict(query_fusion=False, candidate_fusion=True, semantic_selector=False),
    "both_fusion": dict(query_fusion=True, candidate_fusion=True, semantic_selector=False),
    "full_both": dict(query_fusion=True, candidate_fusion=True, semantic_selector=True),
    "full_shuffle_both": dict(query_fusion=True, candidate_fusion=True, semantic_selector=True,
                              q_transform="shuffle", c_transform="shuffle"),
    "full_unscreened": dict(query_fusion=True, candidate_fusion=True, semantic_selector=True, view="unscreened"),
    "full_redacted": dict(query_fusion=True, candidate_fusion=True, semantic_selector=True, view="redacted"),
    # Optional finer training controls; default fixed-context interventions already test these roles.
    "selector_text_only": dict(query_fusion=False, candidate_fusion=False, semantic_selector=True),
    "full_shuffle_query": dict(query_fusion=True, candidate_fusion=True, semantic_selector=True, q_transform="shuffle"),
    "full_shuffle_candidate": dict(query_fusion=True, candidate_fusion=True, semantic_selector=True, c_transform="shuffle"),
}
DEFAULT_ABLATIONS = ["visual_only", "query_only", "candidate_only", "both_fusion", "full_both",
                     "full_shuffle_both", "full_unscreened", "full_redacted"]
ABLATION_NAMES = [name.strip() for name in os.getenv("DTD_ABLATIONS", ",".join(DEFAULT_ABLATIONS)).split(",") if name.strip()]
TRAIN_SEEDS = [int(seed) for seed in os.getenv("DTD_ABLATION_SEEDS", "42").split(",")]
if not ABLATION_NAMES or len(ABLATION_NAMES) != len(set(ABLATION_NAMES)) or set(ABLATION_NAMES) - set(AB_CONFIG):
    raise ValueError("DTD_ABLATIONS contains duplicate/unknown experiment names")
if not TRAIN_SEEDS or len(TRAIN_SEEDS) != len(set(TRAIN_SEEDS)):
    raise ValueError("Use distinct training seeds")
SELECTED_ABLATIONS = [{"name": name, "view": "screened", "q_transform": "aligned", "c_transform": "aligned", **AB_CONFIG[name]}
                     for name in ABLATION_NAMES]
CAPTIONS_INPUT_PATH = os.getenv("DTD_CAPTIONS_INPUT_PATH", "")  # Set the exact existing DTD captions.csv path here.
GENERATE_CAPTIONS_IF_MISSING = True  # Create/resume the shared CSV automatically; reuse complete captions without loading Florence-2.
RESUME_ABLATIONS = True
KEEP_LAST_CHECKPOINT = False  # Keep best.pt per run; discard only completed runs' redundant resume state.
RUN_VALIDATION_INTERVENTIONS = True
RUN_FINAL_TEST = True  # Final held-out evaluation starts only after all declared training runs finish.
BOOTSTRAP_REPEATS = 200
EXTRA_FORBIDDEN_TERMS_PATH = os.getenv("DTD_EXTRA_FORBIDDEN_TERMS", "")
'''

CAPTION_SETUP = r'''
SHARED_DIR = OUT_PUT_DIR / "shared"
RUNS_DIR = OUT_PUT_DIR / "runs"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_CSV = SHARED_DIR / "captions.csv"
if CAPTIONS_INPUT_PATH:
    requested_csv = Path(CAPTIONS_INPUT_PATH).expanduser()
    if not requested_csv.is_file():
        raise FileNotFoundError(f"DTD_CAPTIONS_INPUT_PATH not found: {requested_csv}")
    if CAPTION_CSV.is_file() and requested_csv.resolve() != CAPTION_CSV.resolve():
        if requested_csv.read_bytes() != CAPTION_CSV.read_bytes():
            raise ValueError("Requested captions differ from this experiment's cached CSV; choose a new output directory")
if not CAPTION_CSV.is_file():
    old_output = PROJECT_DIR / "outputs" / f"mtacs_dtd_split{DTD_SPLIT}_captions_vitl16" / "captions.csv"
    source_csv = Path(CAPTIONS_INPUT_PATH).expanduser() if CAPTIONS_INPUT_PATH else old_output
    if source_csv.is_file():
        shutil.copy2(source_csv, CAPTION_CSV)
        print("Reusing existing DTD captions:", source_csv)
    elif CAPTIONS_INPUT_PATH:
        raise FileNotFoundError(f"DTD_CAPTIONS_INPUT_PATH not found: {source_csv}")
    elif not GENERATE_CAPTIONS_IF_MISSING:
        raise FileNotFoundError(
            "Set CAPTIONS_INPUT_PATH to the existing DTD captions.csv, or enable GENERATE_CAPTIONS_IF_MISSING "
            "with matching local Florence-2 and CLIP assets. No caption improvements are applied here."
        )
'''

CAPTION_STATUS = r'''
caption_expected_ids = {
    str(image_id) for frame in (train_df, val_df, test_df) for image_id in frame["image_id"]
}
caption_cached_rows = load_existing(CAPTION_CSV)
caption_unexpected_ids = set(caption_cached_rows) - caption_expected_ids
if caption_unexpected_ids:
    raise ValueError("Cached captions contain images outside DTD; choose the correct CSV or another output directory")
caption_missing_ids = caption_expected_ids - set(caption_cached_rows)
NEEDS_CAPTION_GENERATION = bool(caption_missing_ids)
if NEEDS_CAPTION_GENERATION and not GENERATE_CAPTIONS_IF_MISSING:
    raise FileNotFoundError(
        f"DTD captions incomplete: {len(caption_cached_rows)}/{len(caption_expected_ids)}. "
        "Enable GENERATE_CAPTIONS_IF_MISSING=True to create/resume captions.csv with the attached local models."
    )
if NEEDS_CAPTION_GENERATION:
    print(f"Creating/resuming DTD captions: cached={len(caption_cached_rows):,}, missing={len(caption_missing_ids):,}")
else:
    print(f"Reusing complete DTD captions: {len(caption_cached_rows):,}; Florence-2 generation skipped.")
print("Caption output:", CAPTION_CSV)
'''


def caption_components(source_cells):
    source = source_of(source_cells[7])
    start = source.index("CAPTION_CSV = OUT_PUT_DIR")
    definitions = source[:start]
    definitions = replace_once(definitions,
        'local_model = resolve_local_model(model_id, expected_format)',
        'local_model = resolve_local_model(model_id, expected_format, extra_roots=(\n'
        '        Path("/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base"),\n'
        '    ))')
    definitions = replace_once(definitions,
        '    if local_model is None:\n        checkpoint = ',
        '''    if local_model is None:
        other_format = "florence2_legacy" if FLORENCE_BACKEND == "native" else "florence2_native"
        other_model = resolve_local_model(model_id, other_format)
        if other_model is not None:
            required_version = "4.49.x" if other_format == "florence2_legacy" else "5.x"
            raise RuntimeError(
                f"Florence-2 weights exist at {other_model}, but their format does not match Transformers {_transformers_version}. "
                f"Use Transformers {required_version} with that checkpoint, or attach a checkpoint matching the current runtime. "
                "After changing Transformers, restart the Kaggle session and Run All. No packages were installed automatically."
            )
        checkpoint = ''')
    run = replace_once(source[start:],
                       "CAPTION_CSV = OUT_PUT_DIR / 'captions.csv'", "CAPTION_CSV = SHARED_DIR / 'captions.csv'")
    return definitions, run


PREPARE_TEXT = r'''
SPLIT_FRAMES = {"train": train_df, "val": val_df, "test": test_df}
CANDIDATE_IDS = candidate_df.image_id.astype(str).tolist()
CANDIDATE_POSITION = {image_id: i for i, image_id in enumerate(CANDIDATE_IDS)}
caption_df = load_ablation_captions(CAPTION_CSV, SPLIT_FRAMES)
SPLIT_BY_ID = dict(zip(caption_df.image_id, caption_df.split))
extra_terms = []
if EXTRA_FORBIDDEN_TERMS_PATH:
    extra_terms = [line.strip() for line in Path(EXTRA_FORBIDDEN_TERMS_PATH).read_text(encoding="utf-8-sig").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
caption_audit, caption_views = prepare_caption_views(caption_df, class_names, CAPTION_SEMANTIC_THRESHOLD, extra_terms)
caption_audit.to_csv(SHARED_DIR / "caption_audit.csv", index=False)
audit_summary = caption_audit.groupby("split").agg(
    count=("image_id", "size"), screened_valid_rate=("screened_valid", "mean"),
    class_term_rate=("exact_class_term_hit", "mean"), alias_term_rate=("alias_or_class_term_hit", "mean"),
    matched_raw_redacted_rate=("raw_redacted_matched_valid", "mean"))
audit_summary.to_csv(SHARED_DIR / "caption_audit_summary.csv")
display(audit_summary)
if not caption_audit.screened_valid.any():
    print("WARNING: screened caption coverage is zero. Screened text ablations will equal visual-only inputs.")
if caption_audit.groupby("split").screened_valid.mean().min() < 0.10:
    print("Low caption coverage: inspect the audit before interpreting a small text effect. Do not tune on test.")
clip_asset = resolve_local_model(CLIP_MODEL_LOCAL_PATH, "clip")
if clip_asset is None:
    raise FileNotFoundError("Attach the complete local CLIP ViT-B/32 folder to encode all caption ablation views")
TEXT_CACHE = encode_view_cache(caption_views, clip_asset, SHARED_DIR / "text_ablation_embeddings.pt")
CLIP_EMBED_DIM = int(TEXT_CACHE["embedding_dim"])
print("Cached caption views:", list(TEXT_CACHE["views"]), "| CLIP dimension:", CLIP_EMBED_DIM)
'''

PROTOCOL = r'''
if EPOCHS < 1 or PATIENCE < 1 or BOOTSTRAP_REPEATS < 0:
    raise ValueError("Invalid epochs, patience, or bootstrap count")
if CONTEXT_TOP_K != 1:
    raise ValueError("This controlled ablation uses exactly one selected context per query")
ensure_dinov3_assets()
weights_path = resolve_dinov3_weights()
protocol = {
    "schema": "dtd_text_ablation_v1", "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "helper_source_sha256": HELPER_SOURCE_SHA256, "dtd_split": DTD_SPLIT, "pool_seed": SEED,
    "split_image_ids": {split: frame.image_id.astype(str).tolist() for split, frame in SPLIT_FRAMES.items()},
    "candidate_ids": CANDIDATE_IDS, "class_names": class_names, "caption_cache": TEXT_CACHE["fingerprint"],
    "caption_csv_sha256": hashlib.sha256(CAPTION_CSV.read_bytes()).hexdigest(),
    "weights_path": str(weights_path), "weights_bytes": Path(weights_path).stat().st_size,
    "weights_mtime_ns": Path(weights_path).stat().st_mtime_ns,
    "preprocessing": {"train": repr(train_transform), "eval": repr(eval_transform)},
    "training": {name: globals()[name] for name in (
        "DINOV3_MODEL_NAME", "FEATURE_DIM", "IMAGE_SIZE", "CONTEXT_TOP_K", "CANDIDATE_POOL_FRACTION",
        "EPOCHS", "BATCH_SIZE", "NUM_WORKERS", "BACKBONE_LR", "HEAD_LR", "TACS_SELECTOR_LR", "BACKBONE_WEIGHT_DECAY",
        "HEAD_WEIGHT_DECAY", "TACS_SELECTOR_WEIGHT_DECAY", "LABEL_SMOOTHING", "TOKEN_HEAD_DROPOUT",
        "FREEZE_BACKBONE_EPOCHS", "WARMUP_EPOCHS", "PATIENCE", "GRAD_CLIP_NORM", "SUPCON_WEIGHT", "SUPCON_TEMPERATURE",
        "TACS_POLICY_WEIGHT", "TACS_ENTROPY_WEIGHT", "TACS_SELECTOR_TEMPERATURE",
        "QUERY_CONTEXT_GATE_DROPOUT", "QUERY_CONTEXT_GATE_BIAS", "CAPTION_SEMANTIC_THRESHOLD")},
    "selector_sampling": "categorical_policy_on_visual_z_plus_optional_semantic_z",
    "checkpoint_selection": "minimum_validation_cross_entropy", "test_hyperparameter_tuning": False,
}
PROTOCOL_FINGERPRINT = stable_hash(protocol)
protocol_path = OUT_PUT_DIR / "protocol.json"
if protocol_path.is_file() and read_json(protocol_path) != protocol:
    raise ValueError("Existing output uses another protocol; choose a new DTD_ABLATION_OUTPUT directory")
write_json(protocol_path, protocol)
write_json(OUT_PUT_DIR / "environment.json", {"torch": torch.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "device": str(DEVICE), "transformers": _transformers_version,
           "gpu": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None})
plan = {"protocol": PROTOCOL_FINGERPRINT, "experiments": SELECTED_ABLATIONS, "seeds": TRAIN_SEEDS}
plan_path = OUT_PUT_DIR / "experiment_plan.json"
if (OUT_PUT_DIR / "locked_test_plan.json").is_file() and plan_path.is_file() and read_json(plan_path) != plan:
    raise ValueError("Cannot extend or modify the plan after test was opened; start another experiment directory")
write_json(plan_path, plan)

AblationModel = make_ablation_model_class(MTACSModel, DinoV3TACSClassifier)
candidate_image_bank, _candidate_labels_unused, candidate_fnames = build_candidate_image_bank(
    candidate_df, eval_transform, device=torch.device(CANDIDATE_IMAGE_BANK_DEVICE), dtype=CANDIDATE_IMAGE_BANK_DTYPE)
del _candidate_labels_unused  # Candidate labels do not enter any model branch or selection function.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("Run plan:", len(SELECTED_ABLATIONS), "configurations x", len(TRAIN_SEEDS), "seeds;")
print("Maximum training epochs:", EPOCHS * len(SELECTED_ABLATIONS) * len(TRAIN_SEEDS), "(early stopping enabled)")
print("One best ViT-L checkpoint is retained per run; active last.pt also contains optimizer state.")
print("Output:", OUT_PUT_DIR, "| Candidate bank:", tuple(candidate_image_bank.shape))
display(pd.DataFrame(SELECTED_ABLATIONS))
'''

REPORT = r'''
validation_summary = summarize_ablation("val")
display(validation_summary)
if RUN_FINAL_TEST:
    display(summarize_ablation("test"))

def markdown_table(frame):
    if frame.empty:
        return "No completed metrics yet."
    frame = frame.copy().round(5).fillna("")
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)

report = [f"# DTD split {DTD_SPLIT}: text ablation", "", f"Protocol: `{PROTOCOL_FINGERPRINT}`", "",
          "Official train/val/test: 1880/1880/1880; candidate pool: 20% train; K=1.", "",
          "## Validation", "", markdown_table(validation_summary), "",
          "## Caption-only / availability-only validation probes", "", markdown_table(probe_results), "",
          "## Interpretation", "",
          "- Compare query_only, candidate_only and both_fusion with visual_only to separate downstream text roles.",
          "- Compare full_both with both_fusion for the semantic selector contribution.",
          "- Compare full_both with full_shuffle_both; masks are preserved, so availability is not the shuffled variable.",
          "- Compare full_unscreened with full_redacted: their eligibility masks are identical.",
          "- Inspect val_interventions.csv: fixed-context perturbations isolate fusion; reselect also changes retrieval.",
          "- A strong term-only probe indicates a class-word shortcut; it does not establish access to test annotations.",
          "- DTD category names are valid texture attributes; losing accuracy after redaction is not alone proof of leakage.",
          "- Paired confidence intervals resample images within classes. They do not estimate training-seed variability.",
          "- A single training seed/split is exploratory; use multiple seeds and official splits for robust benchmark claims."]
report += ["", "## Caption audit / label-word diagnostic", "", markdown_table(audit_summary.reset_index()), "",
           markdown_table(pd.DataFrame([label_word_diagnostics]))]
for name, title in (("ablation_val_paired_vs_visual.csv", "Paired validation vs visual baseline"),
                    ("validation_interventions.csv", "Validation interventions")):
    path = OUT_PUT_DIR / name
    if path.is_file() and path.stat().st_size > 2:
        report += ["", "## " + title, "", markdown_table(pd.read_csv(path))]
if RUN_FINAL_TEST:
    report += ["", "## Locked final test", "", markdown_table(pd.read_csv(OUT_PUT_DIR / "ablation_test_results.csv"))]
(OUT_PUT_DIR / "ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
for axis, metric in zip(axes, ("accuracy", "macro_f1")):
    aggregate = validation_summary.groupby("experiment")[metric].agg(["mean", "std"]).reindex(ABLATION_NAMES)
    axis.bar(aggregate.index, aggregate["mean"], color="#217c75", yerr=aggregate["std"].fillna(0), capsize=3)
    axis.set_ylabel("Validation " + metric)
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", labelrotation=70)
fig.savefig(OUT_PUT_DIR / "ablation_validation.png", dpi=160, bbox_inches="tight")
plt.show()
print("Saved report:", OUT_PUT_DIR / "ablation_report.md")
'''


def build_notebook():
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_cells = original["cells"]
    helpers = helper_sections()
    config = replace_once(source_of(source_cells[3]),
        'EXPERIMENT_MODE = "mtacs"  # no_context | original_tacs | text_only | mtacs | mtacs_shuffled',
        'EXPERIMENT_MODE = "dtd_text_ablation_suite"')
    config += "\n" + CONFIG
    settings = replace_once(source_of(source_cells[5]), "SEED = 42", 'SEED = int(os.getenv("DTD_POOL_SEED", "42"))')
    settings = replace_once(settings,
        "OUT_PUT_DIR = PROJECT_DIR / 'outputs' / f'mtacs_dtd_split{DTD_SPLIT}_captions_vitl16'",
        "OUT_PUT_DIR = Path(os.getenv('DTD_ABLATION_OUTPUT', str(PROJECT_DIR / 'outputs' / f'mtacs_dtd_split{DTD_SPLIT}_text_ablation_vitl16')))")
    settings = replace_once(settings, "EPOCHS = 100", 'EPOCHS = int(os.getenv("DTD_ABLATION_EPOCHS", "100"))')

    caption_definitions, caption_run = caption_components(source_cells)
    caption_cell = (caption_definitions + "\n" + CAPTION_SETUP + "\n" + CAPTION_STATUS
                    + "\nif NEEDS_CAPTION_GENERATION:\n" + textwrap.indent(caption_run, "    "))
    base_model = source_of(source_cells[14])
    base_model = base_model[:base_model.index("\nensure_dinov3_assets()")]
    provenance = ("SOURCE_NOTEBOOK_SHA256 = " + repr(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
                  + "\nHELPER_SOURCE_SHA256 = " + repr(hashlib.sha256(HELPERS.read_bytes()).hexdigest()))

    cells = [
        make_cell("markdown", INTRO), make_cell("markdown", EXPERIMENTS),
        make_cell("markdown", """## Cách chạy

1. Gắn DTD, DINOv3 code/ViT-L weights và CLIP ViT-B/32 như notebook nguồn.
2. Đặt `CAPTIONS_INPUT_PATH` đến CSV DTD đã tạo. Nếu CSV cũ còn trong working, notebook tự lấy từ output bản nguồn.
   Nếu chưa có CSV, giữ `CAPTIONS_INPUT_PATH=""`: mặc định tự tạo/resume bằng Florence-2 và CLIP local.
   Model cũ của bạn được tìm trong `/kaggle/input/notebooks/nguyenducxu/notebook-mt-moi/Florence-2-base`.
   Checkpoint Microsoft có `AutoModelForCausalLM` cần Transformers 4.49.x; checkpoint native cần runtime tương thích.
   CSV đầy đủ được dùng lại ngay, không nạp captioner; khi resume CSV thiếu, cấu hình model/prompt phải khớp các dòng đã tạo.
3. Giữ 8 cấu hình mặc định hoặc chốt `ABLATION_NAMES` trước khi train. Mặc định seed 42, tối đa 100 epoch/cấu hình;
   `DTD_ABLATION_SEEDS=42,43,44` cho multi-seed. Toàn bộ suite lâu hơn một lần train thông thường.
4. Chạy lần lượt: caption cache -> audit/CLIP -> probes -> train từng cấu hình -> validation interventions -> final test.
5. Resume từng run từ last.pt, tự bỏ qua run đã xong. Chỉ giữ best.pt sau khi hoàn tất để giảm dung lượng Kaggle.
   Dành chỗ cho nhiều ViT-L checkpoints; có thể chạy một subset nếu giới hạn tài nguyên, nhưng chốt plan trước final test.
   Nhiều seeds làm tăng cả thời gian và dung lượng; có thể cần tách session/output thay vì lưu 24 checkpoint trong Kaggle working.

Kết quả/charts/CSV ở output riêng, không ghi đè notebook hay output nguồn. Không có metric giả trong notebook.
`RUN_FINAL_TEST=False` cho giai đoạn chỉ phát triển trên validation. Khi bật, test chỉ mở sau khi toàn bộ plan đã train xong;
plan và best epoch được khóa trước khi tính test metrics, kết quả được cache để rerun không đánh giá lại.
"""),
        make_cell("code", source_of(source_cells[2])), make_cell("code", config),
        make_cell("code", settings),
        make_cell("markdown", "## Tái sử dụng caption; giữ nguyên captioner và audit của notebook nguồn"),
        make_cell("code", caption_cell),
        make_cell("markdown", "## Caption views, hoán vị không dùng nhãn và CLIP cache riêng"),
        make_cell("code", helpers["utilities"] + "\n" + provenance),
        make_cell("code", helpers["captions"] + "\n\n" + helpers["text_banks"]),
        make_cell("code", PREPARE_TEXT),
        make_cell("markdown", """## Cùng kiến trúc cho mọi cấu hình

Kế thừa backbone/visual fusion từ notebook nguồn; thêm nhánh caption candidate đối xứng với query.
Mọi chế độ dùng cùng classifier head kể cả visual_only; tránh thay head/capacity khi tắt text.
Selector dùng visual score chuẩn hóa và semantic score tùy cấu hình. Lựa chọn rời rạc được học bằng policy gradient
với cùng action đã dùng trong supervised forward; không giả định gradient đi qua index_select.
Reward so sánh query + text query với query + candidate image + các caption tương ứng, dropout tắt ở cả hai nhánh.
Candidate feature bank refresh trước train và sau cập nhật backbone để validation không dùng features cũ.
"""),
        make_cell("code", base_model), make_cell("code", helpers["model"]),
        make_cell("code", helpers["training"]), make_cell("code", helpers["diagnostics"]),
        make_cell("code", PROTOCOL),
        make_cell("markdown", """## Probes chỉ dùng text và dấu hiệu caption bị lọc

Logistic regression fit trên train labels, báo validation: screened, unscreened, redacted, từ chỉ lớp được tìm trong caption,
và chỉ các availability mask. Các probe này không nhận ảnh; tên file/nhãn thật không được chuyển thành text.
Probe tốt cho biết text hoặc mask có tín hiệu phân loại; cần đọc cùng audit để phân biệt mô tả hợp lệ với shortcut.
"""),
        make_cell("code", 'label_word_diagnostics = caption_label_diagnostics("val")\ndisplay(pd.DataFrame([label_word_diagnostics]))\nprobe_results = caption_probes()\ndisplay(probe_results)'),
        make_cell("markdown", "## Train độc lập từng cấu hình; chỉ validation chọn checkpoint"),
        make_cell("code", "for spec in SELECTED_ABLATIONS:\n    for training_seed in TRAIN_SEEDS:\n        train_one_configuration(spec, training_seed)\ndisplay(summarize_ablation('val'))"),
        make_cell("markdown", """## Intervention trên validation: cùng checkpoint, cùng ảnh context

Tắt/đảo query text, candidate text, cả hai; giữ nguyên context để kiểm tra fusion. Các dòng `reselect` cho phép đổi context
để đo cả ảnh hưởng retrieval. Gate trung bình và tỷ lệ đổi nhãn được lưu cùng delta macro-F1/CI 95%.
Redaction giữ nguyên mask caption của checkpoint gốc ở cả hai nhánh, nên chỉ thay nội dung text; nếu không còn từ,
vector text được đặt zero nhưng mask gốc vẫn được giữ cho phép can thiệp này. Đây là sensitivity test tại inference;
kết luận còn cần so với các cấu hình được train độc lập.
Mọi intervention ở cell này chỉ dùng validation, không dùng test để lựa chọn cấu hình.
"""),
        make_cell("code", "intervention_tables = []\nif RUN_VALIDATION_INTERVENTIONS:\n    for spec in SELECTED_ABLATIONS:\n        if spec['name'] in {'full_both', 'full_unscreened'}:\n            for training_seed in TRAIN_SEEDS:\n                intervention_tables.append(validation_interventions(spec, training_seed))\nif intervention_tables:\n    interventions = pd.concat(intervention_tables, ignore_index=True)\n    interventions.to_csv(OUT_PUT_DIR / 'validation_interventions.csv', index=False)\n    display(interventions)"),
        make_cell("markdown", "## Đánh giá test cuối: khóa plan trước khi tính metrics, không chọn cấu hình bằng test"),
        make_cell("code", 'if RUN_FINAL_TEST:\n    display(evaluate_locked_test())\nelse:\n    print("Validation-only development: final test has not been opened.")'),
        make_cell("markdown", "## Tổng hợp metrics thật, paired comparisons và báo cáo"),
        make_cell("code", REPORT),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"dtd-ablation-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse(source_of(cell), filename=f"notebook_cell_{index}")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": copy.deepcopy(original.get("metadata", {})), "cells": cells}
    notebook["metadata"].pop("widgets", None)
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["metadata"]["dtd_ablation"] = {"source": SOURCE.name, "default_experiments": 8, "schema": 1}
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {TARGET} ({len(notebook['cells'])} cells)")
