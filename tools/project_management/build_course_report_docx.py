"""Build the Vietnamese course report from saved notebook outputs, without execution."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Cm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/bao_cao_bai_tap_lon"
SOURCE = OUT / "NOI_DUNG_BAO_CAO_TACS_MTACS.md"
TARGET = OUT / "BAO_CAO_TACS_VA_MO_RONG_NGU_CANH_ANH_VAN_BAN.docx"
sys.path.insert(0, str(Path(__file__).parent))
from audit_dtd_caption_results import saved_tables_and_streams


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EVIDENCE = []


def notebook_record(path, code, expected_sha=None):
    path = ROOT / path
    digest = sha(path)
    if expected_sha and digest != expected_sha:
        raise ValueError(f"Notebook changed: {path}")
    nb = read_json(path)
    tables, streams, errors = saved_tables_and_streams(nb)
    EVIDENCE.append({"id": code, "path": str(path.relative_to(ROOT)), "sha256": digest,
                     "error_outputs": errors, "table_count": len(tables)})
    return nb, tables, streams


def rows_dict(table):
    rows = table["rows"]
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def find_metric(tables, split):
    if split == "test":
        candidates = [t for t in tables if "macro_f1" in t["rows"][0]
                      and "accuracy" in t["rows"][0]
                      and "intervention" not in t["rows"][0]
                      and "epochs_trained" not in t["rows"][0]
                      and len(t["rows"]) == 2]
    else:
        candidates = [t for t in tables if "macro_f1" in t["rows"][0]
                      and "epochs_trained" in t["rows"][0]
                      and "intervention" not in t["rows"][0]
                      and len(t["rows"]) == 2]
    if not candidates:
        raise ValueError(f"Missing {split} metric table")
    return rows_dict(candidates[0])[0]


def check_metrics(actual, expected):
    for key in ("accuracy", "macro_f1", "loss"):
        if key in expected:
            if abs(float(actual[key]) - float(expected[key])) > 1.1e-6:
                raise ValueError((key, actual[key], expected[key]))


def n(value, digits=4):
    return f"{float(value):.{digits}f}"


def percent(value, digits=2):
    return f"{100 * float(value):.{digits}f}"


def gather():
    data = {}
    old = read_json(ROOT / "experiments/completed/text_context_ablation/dtd/analysis/saved_output_evidence.json")
    ablations = []
    for index, (name, saved) in enumerate(old.items(), 1):
        if not isinstance(saved, dict) or "test_metrics" not in saved:
            continue
        original_name = Path(saved["notebook"].replace("\\", "/")).name
        path = Path("experiments/completed/text_context_ablation/dtd") / original_name
        _, tables, _ = notebook_record(path, f"A{index:02}", saved["sha256"])
        test = find_metric(tables, "test")
        check_metrics(test, saved["test_metrics"][0])
        val = next(r for r in saved["report_metrics"] if r.get("split") == "val")
        ablations.append({"name": name, "test": test, "val": val, "id": f"A{index:02}"})
    data["ablations"] = ablations
    cap_evidence = read_json(ROOT / "reports/caption_ablation/dtd/caption_method_output_evidence.json")
    captions = []
    for index, saved in enumerate(cap_evidence["new_runs"] + cap_evidence["historical_references"], 1):
        _, tables, _ = notebook_record(saved["source_notebook"], f"C{index:02}", saved["notebook_sha256"])
        test = find_metric(tables, "test")
        check_metrics(test, saved["test_metrics"])
        captions.append({"name": saved["name"], "test": test,
                         "val": saved["validation_metrics"], "id": f"C{index:02}"})
    data["captions"] = captions
    studies = []
    study_files = ["dtd-architecture-study-01.ipynb", "stydu2_da_train.ipynb", "study3_datrain.ipynb", "study4.ipynb"]
    expected = ["3254ec626634f7922472963e4ccb64c5016cdc2f43730ba6c3fb411e739e5e07",
                "9efd869857503cd73c023ff9475a562810c2c144f95ec23acdcacf49629e3d00",
                "9382caf28dc8f63eae7a4149e79508fc91bf05f197a4e57264ae7323eb51f6de",
                "10ad4dcc0ab82b9caaea2cf1a3b4f4d3af733a32b1261deb6c392c3714ffedae"]
    for index, (file, digest) in enumerate(zip(study_files, expected), 1):
        _, tables, _ = notebook_record(file, f"S{index:02}", digest)
        val, test = find_metric(tables, "val"), find_metric(tables, "test")
        if index in (2, 3):
            dirname = ("dtd_study02_study03_results_20261026_050338/study02_norm_fusion" if index == 2
                       else "dtd_study03_results_20261026_050358/study03_mltm_alignment")
            run = ROOT / dirname / "runs/candidate_only/seed_42"
            actual_val = read_json(run / "val_metrics.json")
            actual_test = read_json(run / "test_metrics.json")
            check_metrics(val, actual_val)
            check_metrics(test, actual_test)
            val, test = actual_val, actual_test
            for split, metrics in [("val", val), ("test", test)]:
                pred = run / f"{split}_predictions.csv"
                if not pred.exists():
                    raise FileNotFoundError(pred)
                rows = list(csv.DictReader(pred.open(encoding="utf-8-sig")))
                if len(rows) != 1880:
                    raise ValueError(f"Expected 1880 predictions: {pred}")
                ykey = next(k for k in ("ground_truth", "label", "class_id") if k in rows[0])
                pkey = next(k for k in ("prediction", "predicted_class_id", "pred") if k in rows[0])
                truth = [int(r[ykey]) for r in rows]
                predictions = [int(r[pkey]) for r in rows]
                accuracy = sum(y == p for y, p in zip(truth, predictions)) / len(rows)
                scores = []
                for label in range(47):
                    tp = sum(y == label and p == label for y, p in zip(truth, predictions))
                    fp = sum(y != label and p == label for y, p in zip(truth, predictions))
                    fn = sum(y == label and p != label for y, p in zip(truth, predictions))
                    scores.append(2 * tp / max(2 * tp + fp + fn, 1))
                f1 = sum(scores) / 47
                if abs(accuracy - metrics["accuracy"]) > 1e-12 or abs(f1 - metrics["macro_f1"]) > 1e-12:
                    raise ValueError(f"Prediction audit failed: {pred}")
        studies.append({"index": index, "val": val, "test": test, "tables": tables})
    data["studies"] = studies
    for index, file in enumerate(["tacs_isic_segement_448.ipynb", "isic-segmen-captions.ipynb"], 1):
        _, tables, streams = notebook_record(file, f"I{index:02}")
        dice = re.findall(r"test_dice:\s*([0-9.]+)", streams)
        iou = re.findall(r"test_iou:\s*([0-9.]+)", streams)
        if not dice or not iou:
            raise ValueError(f"Missing ISIC metrics: {file}")
        data[f"isic{index}"] = {"dice": dice[-1], "iou": iou[-1]}
    inventory = read_json(ROOT / "reports/paper_readiness/notebook_evidence_inventory.json")
    selected = [r for r in inventory if any(s in r["path"] for s in [
        "classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification (seed42)",
        "classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification (seed43)",
        "classification/table_02_main_results/no_context/cub200/baseline_no_tacs_image_classification_(seed_44)",
        "classification/table_02_main_results/no_context/stanford_cars/seed44-car",
        "classification/table_02_main_results/random_context/stanford_cars/ban-cuoi-Random Context Retrieval baseline",
        "classification/table_02_main_results/frozen_dino_similarity/stanford_cars/ban-cuoi-frozen-dino-car",
        "classification/table_02_main_results/tacs/stanford_cars/ban-cuoi-frozen-dino-car"])]
    for index, r in enumerate(selected, 1):
        notebook_record(r["path"], f"V{index:02}", r["sha256"])
    for index, file in enumerate(["tacs-sun-da-sua (2).ipynb", "tacs-chuanpaper-sun-b16 (1).ipynb"], 8):
        notebook_record("notebooks/01_paper_reproduction/classification/table_02_main_results/tacs/sun397/" + file, f"V{index:02}")
    sun_path="notebooks/02_improvements/query_text/sun397/mtacs-train-captions.ipynb"
    sun_saved=next(r for r in inventory if r["path"]==sun_path)
    _, tables, _=notebook_record(sun_path,"U01",sun_saved["sha256"])
    data["sun_query_caption"]=find_metric(tables,"test")
    check_metrics(data["sun_query_caption"],{"accuracy":.761159,"macro_f1":.760189,"loss":1.765304})
    return data


TABLES = {}


def make_tables(data):
    def add(key, title, headers, rows, widths=None, note=""):
        TABLES[key] = {"title": title, "headers": headers, "rows": rows, "widths": widths, "note": note}
    add("protocol", "Bảng 3.1 Đối chiếu thiết lập gốc và các nhánh của đề tài",
        ["Thành phần", "TACS gốc [1]", "DTD mở rộng", "ISIC mở rộng"], [
        ["Backbone", "DINOv3 S/16", "DINOv3 L/16", "DINOv3 S/16 đóng băng"],
        ["Đầu vào ảnh", "256 × 256", "384 × 384", "448 × 448"],
        ["Ngân sách", "100 epoch", "50 epoch cố định", "Tối đa 100; early stop"],
        ["Số seed báo cáo", "Trung bình 3 seed", "42; split DTD 1", "42"],
        ["Pool và K", "20% train; một context", "376 ảnh; K = 1", "400 ảnh; K = 1"],
        ["Fusion", "Patch concat; blockwise cross-attention cho segmentation", "Ảnh theo backbone + gated text vector", "Final-token cross-attention + DPT-lite; thêm text ở bản caption"],
        ["Tối ưu selector", "Task gradient Gumbel-ST + policy", "Categorical policy trong đường ablation thực thi", "Gumbel-ST + policy theo source visual"],
        ["Chọn checkpoint", "Chi tiết theo thí nghiệm gốc", "Min validation CE", "Max validation hard Dice"],
        ], [1.25, 1.55, 1.65, 1.95],
        "Phạm vi khác nhau phải được công bố. Không diễn giải chênh lệch với paper như tác dụng thuần của caption hoặc backbone.")
    add("datasets", "Bảng 3.2 Dữ liệu và phân chia được dùng trong các nhóm chính",
        ["Bộ dữ liệu", "Train", "Validation", "Test", "Pool", "Vai trò"], [
        ["SUN397 phân hoạch 01", "17.865", "1.985", "19.850", "20% train", "Triển khai visual và thử caption ban đầu"],
        ["DTD official split 1", "1.880", "1.880", "1.880", "376", "Ablation text và nghiên cứu kiến trúc"],
        ["ISIC 2017", "2.000", "150", "600", "400", "Phân đoạn tổn thương"],
        ], [1.3,.7,.85,.7,.8,2.05],
        "SUN tách validation 10% có phân tầng từ 19.850 ảnh training của phân hoạch 01; không áp dụng train+val DTD. Pool là tập con train, không phải split đánh giá thứ tư.")
    add("hypers", "Bảng 3.3 Siêu tham số chính của nhóm DTD",
        ["Hạng mục", "Giá trị", "Hạng mục", "Giá trị"], [
        ["Backbone / chiều vector", "L/16 / 1.024", "Text encoder / chiều", "CLIP B/32 / 512"],
        ["Epoch / seed", "50 / 42", "Batch thực tế", "32"],
        ["LR backbone", "3 × 10⁻⁶", "LR head và selector", "10⁻⁴"],
        ["Weight decay backbone", "0,05", "Weight decay head", "10⁻⁴"],
        ["Label smoothing", "0,1", "SupCon weight", "0,05"],
        ["Warmup", "5 epoch", "Gradient clipping", "1,0"],
        ["Policy / entropy weight", "1,0 / 0,01", "Scheduler horizon", "50 epoch"],
        ["Retrieval / feature batch", "128 / 64", "TTA và label prior", "Tắt"],
        ], [1.65,1.35,1.65,1.75],
        "Giá trị lấy từ source/log của run, không mặc định là cấu hình tác giả. Thông số alignment/selector mới nằm ở chương 4.")
    modes = [
        ["visual_only", "Không", "Không", "Ảnh", "Mask text"],
        ["query_only", "Có", "Không", "Ảnh", "Screened"],
        ["candidate_only", "Không", "Có", "Ảnh", "Screened"],
        ["both_fusion", "Có", "Có", "Ảnh", "Screened"],
        ["full_both", "Có", "Có", "Ảnh + text", "Screened"],
        ["full_shuffle_both", "Có", "Có", "Ảnh + text", "Đảo text hai vai trò; giữ mask"],
        ["full_unscreened", "Có", "Có", "Ảnh + text", "Trước lọc; không đảo"],
        ["full_redacted", "Có", "Có", "Ảnh + text", "Xóa tên lớp và alias"],
    ]
    add("modes", "Bảng 4.1 Tám cấu hình kiểm tra vai trò của văn bản",
        ["Cấu hình", "Query fusion", "Context fusion", "Selector", "Caption dùng"], modes,
        [1.65,.85,.9,1.0,2.0], "Mọi hàng đều có query image và ảnh context. visual_only không phải No-Context. ‘Text selector’ ở nhóm này là điểm caption–caption.")
    add("caption_methods", "Bảng 4.2 Các cách tạo hoặc chọn caption được khảo sát",
        ["Phương pháp", "Thay đổi chính", "Có cập nhật captioner"], [
        ["Florence ban đầu", "MORE_DETAILED_CAPTION; xử lý prefix và kiểm tra policy", "Không"],
        ["Decoder LoRA", "Học adapter decoder từ caption tham chiếu của teacher", "Có trong thiết kế fine-tune; metric bảng 5.6 là downstream"],
        ["Prefix", "Control lấy đoạn đầu của mô tả đã xử lý", "Không"],
        ["FirstValid", "Ưu tiên mô tả đầu tiên đạt các ràng buộc", "Không"],
        ["SingleGrounded", "Chọn mô tả hợp lệ từ nguồn giả thuyết đơn bằng khớp ảnh–text", "Không"],
        ["MultiGrounded", "Nhiều giả thuyết/beam rồi chọn bằng tiêu chí grounding", "Không"],
        ["Qwen3-VL 8B / 32B", "Đổi họ mô hình sinh caption; xuất hai CSV riêng", "Không fine-tune Qwen trong classifier"],
        ], [1.35,3.15,1.9])
    base = float(data["ablations"][0]["test"]["macro_f1"])
    add("ablation_results", "Bảng 5.2 Kết quả tám ablation text trên DTD",
        ["Cấu hình", "Val F1", "Test Acc", "Test F1", "Δ F1 đpt", "Best epoch"],
        [[r["name"], percent(r["val"]["macro_f1"]), percent(r["test"]["accuracy"]),
          percent(r["test"]["macro_f1"]), f"{(float(r['test']['macro_f1']) - base)*100:+.2f}", str(r["val"]["best_epoch"])] for r in data["ablations"]],
        [1.85,.85,.9,.85,1.05,.9],
        "Đơn vị Acc/F1 là %. Δ là điểm phần trăm so với visual_only. Test lấy bảng 6 chữ số; val của bộ A lấy report 4 chữ số nên không suy ra thêm độ chính xác. Nguồn A01–A08; một seed và một split.")
    add("ablation_gates", "Bảng 5.3 Tỷ lệ caption và gate trên test của bộ tám ablation",
        ["Cấu hình", "Query hợp lệ %", "Context hợp lệ %", "q gate", "c gate"],
        [[r["name"],percent(r["test"]["query_caption_rate"]),percent(r["test"]["candidate_caption_rate"]),
          n(r["test"]["mean_query_gate"],6),n(r["test"]["mean_candidate_gate"],6)] for r in data["ablations"]],
        [1.85,1.1,1.2,1.1,1.1],
        "Context rate tính trên ảnh thực sự được chọn, không phải tỷ lệ caption hợp lệ toàn pool. Gate trung bình có cả vị trí mask bằng 0; không phải phần trăm đóng góp.")
    add("caption_results", "Bảng 5.4 So sánh sáu nguồn caption trong cùng cấu hình full_both",
        ["Caption", "Val F1 %", "Test Acc %", "Test F1 %", "Test CE", "Best"],
        [[r["name"].replace("-Instruct", ""),percent(r["val"]["macro_f1"]),percent(r["test"]["accuracy"]),
          percent(r["test"]["macro_f1"]),n(r["test"]["loss"],4),str(r["val"]["best_epoch"])] for r in data["captions"][:6]],
        [2.05,.8,.9,.8,1.0,.6],
        "Nguồn C01–C06. Cùng full_both, L/16–384, split 1, seed 42, 50 epoch; natural masks khác nhau. Số làm tròn để trình bày, phụ lục lưu chỉ số 6 chữ số.")
    add("caption_history", "Bảng 5.6 Tham chiếu Florence lịch sử và decoder LoRA",
        ["Nguồn caption", "Val F1 %", "Test Acc %", "Test F1 %", "Best"],
        [[r["name"],percent(r["val"]["macro_f1"]),percent(r["test"]["accuracy"]),percent(r["test"]["macro_f1"]),str(r["val"]["best_epoch"])] for r in data["captions"][6:]],
        [2.2,1.1,1.1,1.1,.6],
        "Nguồn C07–C08. Đây là reference lịch sử, không phải hai seed lặp hoặc hai hàng trong nhóm bốn CSV control khóa hash. Không suy từ F1 classifier ra chất lượng caption tham chiếu.")
    add("coverage", "Bảng 5.5 Độ phủ caption theo nguồn",
        ["Nguồn", "Train", "Val", "Test", "Toàn bộ", "Pool"], [
        ["Prefix", "1.573", "1.585", "1.591", "4.749/5.640", "324/376"],
        ["FirstValid", "1.880", "1.880", "1.880", "5.640/5.640", "376/376"],
        ["SingleGrounded", "1.880", "1.880", "1.880", "5.640/5.640", "376/376"],
        ["MultiGrounded", "1.880", "1.880", "1.880", "5.640/5.640", "376/376"],
        ["Qwen 8B", "1.418", "1.400", "1.393", "4.211/5.640", "295/376"],
        ["Qwen 32B", "1.391", "1.374", "1.346", "4.111/5.640", "285/376"],
        ], [1.4,.7,.7,.7,1.4,1.0],
        "Số caption qua policy trong output; Qwen CSV chưa được kiểm tra byte độc lập ở local trong nhóm evidence này. 100% qua policy không tương đương 100% đúng.")
    names = ["S01 Candidate Prefix", "S02 + L2 fusion", "S03 + alignment", "S04 + semantic selector"]
    add("study_results", "Bảng 5.7 Nghiên cứu kiến trúc với một CSV Florence Prefix cố định",
        ["Nghiên cứu", "Val F1 %", "Test Acc %", "Test F1 %", "Test CE", "Δ F1 so S01"],
        [[names[i],percent(s["val"]["macro_f1"]),percent(s["test"]["accuracy"]),percent(s["test"]["macro_f1"]),
          n(s["test"]["loss"],4),f"{(float(s['test']['macro_f1'])-float(data['studies'][0]['test']['macro_f1']))*100:+.3f}"] for i,s in enumerate(data["studies"])],
        [1.95,.85,.9,.85,.8,1.0],
        "Tất cả best epoch 38, budget 50, seed 42, K = 1; query text tắt. S02/S03 lấy JSON và kiểm lại prediction CSV; S01/S04 lấy bảng notebook 6 chữ số. Runtime S02/S03 không hoàn toàn giống nhau.")
    add("study_gates", "Bảng 5.8 Gate và độ phủ context trong chuỗi kiến trúc",
        ["Nghiên cứu", "Val c gate", "Test c gate", "Test context có text %", "Semantic selector"],
        [[f"S{i+1:02}",n(s["val"]["mean_candidate_gate"],6),n(s["test"]["mean_candidate_gate"],6),percent(s["test"]["candidate_caption_rate"]), "Bật" if i==3 else "Tắt"] for i,s in enumerate(data["studies"])],
        [.8,1.2,1.2,1.8,1.3],
        "q gate bằng 0 theo thiết kế. S01 không L2-normalize projected text; không so trực tiếp trị số gate giữa S01 và S02 như cùng một thước đo đóng góp.")
    add("fusion_interventions", "Bảng 5.9 Can thiệp fusion trên validation với context ảnh cố định",
        ["Checkpoint", "Text đúng F1", "Text off F1", "Shuffle F1", "CE đúng / off / shuffle"], [
        ["S02", "0.835824", "0.839112", "0.839715", "1.178273 / xem artifact / xem artifact"],
        ["S03", "0.834947", "0.835391", "0.837773", "1.178478 / xem artifact / xem artifact"],
        ["S04", "0.838806", "0.839080", "0.841044", "1.174964 / 1.190771 / 1.193906"],
        ], [.85,1.1,1.1,1.1,2.05],
        "Cùng checkpoint, không retrain; F1 thang 0–1. Selector và context ảnh được giữ cố định. Nội dung text trong nhánh fusion là biến bị can thiệp.")
    # Replace diagnostic CE placeholders with the exact exported values.
    for i, dirname in enumerate(["dtd_study02_study03_results_20261026_050338/study02_norm_fusion", "dtd_study03_results_20261026_050358/study03_mltm_alignment"]):
        rows = list(csv.DictReader((ROOT/dirname/"validation_interventions.csv").open(encoding="utf-8-sig")))
        TABLES["fusion_interventions"]["rows"][i][-1] = " / ".join(n(r["loss"],6) for r in rows)
    add("selector_interventions", "Bảng 5.10 Can thiệp selector S04 trên validation",
        ["Can thiệp", "F1", "CE", "Đổi context %", "CI 95% của Δ F1"], [
        ["Aligned", "0.838806", "1.174964", "0.00", "Mốc"],
        ["Visual selector only", "0.838311", "1.175294", "5.21", "[−0.002116; 0.001036]"],
        ["Selector text shuffled", "0.836615", "1.176430", "10.59", "[−0.004760; 0.000043]"],
        ], [1.65,.85,.85,1.0,1.8],
        "CI lấy từ output paired bootstrap của notebook; Δ = can thiệp trừ aligned trên thang F1 0–1. Có reselect context; fusion dùng caption đúng của context mới, không dùng caption đảo trong fusion.")
    add("alignment_fit", "Bảng 5.11 Chẩn đoán matching trên training candidate pool tại best checkpoint",
        ["Chỉ số", "S03", "S04"], [
        ["Cặp caption hợp lệ / nhóm câu khác nhau", "324 / 289", "324 / 289"],
        ["Transport KL", "2.656227", "2.948433"],
        ["Positive transport mass", "0.093364", "0.073993"],
        ["Cosine cặp đúng trung bình", "0.429815", "0.391488"],
        ["Caption group top 1", "0.746914", "0.697531"],
        ["Exact source top 1", "0.672840", "0.629630"],
        ["Tỷ số max/min diagonal metric", "1.331802", "1.113216"],
        ], [3.3,1.5,1.5],
        "Đây là fit trên chính pool training dùng cho auxiliary loss, không phải đánh giá retrieval held-out hoặc caption quality.")
    add("isic", "Bảng 5.12 Kết quả phân đoạn ISIC 2017",
        ["Cấu hình", "Test Dice %", "Test IoU %", "Best epoch", "Đã chạy"], [
        ["Visual TACS điều chỉnh 448", percent(data["isic1"]["dice"]),percent(data["isic1"]["iou"]),"49","64/100"],
        ["Qwen 8B full_both 448",percent(data["isic2"]["dice"],4),percent(data["isic2"]["iou"],4),"26","41/100"],
        ["Florence Prefix full_both", "Chưa có", "Chưa có", "Chưa có", "Chưa có output"],
        ], [2.25,1.0,1.0,1.0,1.1],
        "Nguồn I01–I02. Mean hard foreground Dice/IoU trên từng ảnh test, threshold 0,5, resolution 448. Visual chỉ in 4 chữ số, nên chênh lệch dùng xấp xỉ.")
    add("paper_reference", "Bảng 5.1 Một số số liệu gốc dùng làm mốc tham khảo",
        ["Phương pháp trong paper", "CUB Acc %", "DTD Acc %", "Cars Acc %", "SUN Acc %", "ISIC Dice / IoU %"], [
        ["No Context", "82.1", "76.1", "92.4", "68.0", "83.3 / 74.4"],
        ["Random Context", "82.1", "74.6", "92.3", "69.0", "83.7 / 76.3"],
        ["Frozen DINO", "81.4", "75.4", "93.0", "69.2", "84.4 / 76.0"],
        ["Feature Averaged", "82.6", "76.3", "93.2", "66.7", "84.5 / 75.6"],
        ["TACS", "85.2", "77.7", "94.1", "71.8", "85.0 / 76.5"],
        ], [1.6,.8,.8,.8,.8,1.55],
        "Trích số công bố từ Tables 2–3 của [1], không phải số đo lại của đề tài. Backbone S/16–256, protocol và tổng hợp khác nhóm DTD/ISIC local.")
    add("visual_history", "Bảng 5.13 Kết quả visual lịch sử với cấu hình riêng từng run",
        ["Dataset và phương pháp", "Cấu hình", "Acc %", "F1 %", "Lưu ý"], [
        ["CUB No Context seed 42", "S/16–256", "82.46", "82.57", "Một run"],
        ["CUB No Context seed 43", "S/16–256", "82.27", "82.32", "Một run"],
        ["CUB No Context seed 44", "S/16–256", "82.53", "82.54", "Một run"],
        ["Cars No Context seed 44", "S/16–256", "91.59", "91.43", "Khác cấu hình các hàng Cars dưới"],
        ["Cars Random", "S/16–448; K1", "94.38", "94.29", "Full train; monitor nội bộ"],
        ["Cars Frozen", "S/16–448; K4; pool100%", "93.60", "93.45", "Không phải đối chứng pool20%"],
        ["Cars TACS", "S/16–448; shortlist4", "94.98", "94.90", "Có lỗi inference bổ sung sau test"],
        ["SUN visual true val", "S/16–256; K1", "74.01", "73.85", "Best13; early stop23"],
        ["SUN visual true val", "B/16–384; K1", "79.42", "79.27", "Best17; early stop27"],
        ], [1.85,1.65,.7,.7,1.8],
        "Các ô lấy stdout notebook đã chạy. Đây là catalog kết quả, không phải bảng đối chứng cùng điều kiện. Template source chưa chạy không thay thế notebook output. Các run bất thường được nêu trong văn bản, không dùng để quảng bá mức tăng.")
    add("scope", "Bảng 5.14 Mức hoàn thiện các nhóm kiểm chứng theo tinh thần paper",
        ["Nhóm", "Đã có", "Giới hạn còn lại"], [
        ["Table 2 classification", "CUB No Context; Cars/SUN visual lịch sử; DTD mở rộng", "Thiếu trọn bộ No Context/Frozen/TACS cùng protocol trên dataset trọng tâm"],
        ["Table 3 segmentation", "ISIC visual và Qwen full_both", "Thiếu No Context/Frozen cùng decoder và resolution"],
        ["Table 4 context controls", "Thiết kế và một số notebook", "Chưa đủ output cùng protocol cho blank/duplicate/noisy"],
        ["Table 5 optimization", "Source visual có hai đường tối ưu", "Chưa có bộ output tách gradient-only/policy-only/full cùng điều kiện"],
        ["Text / caption / kiến trúc", "8 text ablation; 6 CSV mới; 4 studies; can thiệp validation", "Một split/seed; masks/runtime và artifact chưa đồng đều"],
        ], [1.4,2.3,2.6])
    add("api", "Bảng 6.1 Hợp đồng API và trạng thái demo",
        ["Endpoint", "Mục đích", "Trạng thái"], [
        ["GET /api/status", "Chọn pipeline tacs hoặc mtacs; đọc trạng thái", "Có endpoint"],
        ["POST /api/images/inspect", "Kiểm tra định dạng, kích thước và hash ảnh", "Có kiểm tra input"],
        ["POST /api/classify", "Dự đoán và trả caption/context/attention", "503; prediction null khi chưa nối checkpoint"],
        ["GET /healthz", "Health check backend", "Có endpoint"],
        ], [2.05,2.65,1.6])


def xml_child(parent, tag, attrs=None):
    el = OxmlElement(tag)
    for key, val in (attrs or {}).items():
        el.set(qn(key), str(val))
    parent.append(el)
    return el


def field(paragraph, instruction, display=""):
    run = paragraph.add_run()
    xml_child(run._r, "w:fldChar", {"w:fldCharType": "begin"})
    t = xml_child(run._r, "w:instrText", {"xml:space": "preserve"})
    t.text = instruction
    xml_child(run._r, "w:fldChar", {"w:fldCharType": "separate"})
    xml_child(run._r, "w:t").text = display
    xml_child(run._r, "w:fldChar", {"w:fldCharType": "end"})


def set_font(style, size, bold=False):
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in list(fonts.attrib):
        if "theme" in key.lower():
            del fonts.attrib[key]
    for key in ["w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"]:
        fonts.set(qn(key), "Times New Roman")
    for spacing in style.element.findall(".//"+qn("w:spacing")):
        if spacing.getparent().tag == qn("w:rPr"):
            spacing.getparent().remove(spacing)


def setup(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.top_margin, s.bottom_margin = Cm(2.3), Cm(2.2)
    s.left_margin, s.right_margin = Cm(2.8), Cm(2.0)
    s.header_distance, s.footer_distance = Cm(1.0), Cm(1.1)
    for name,size,bold in [("Normal",12.5,False),("Title",19,True),("Subtitle",13,False),
                           ("Heading 1",16,True),("Heading 2",14,True),("Heading 3",12.5,True),
                           ("Caption",11.5,False)]:
        set_font(doc.styles[name],size,bold)
        p = doc.styles[name].paragraph_format
        p.line_spacing = 1.18 if name=="Normal" else 1.1
        p.space_after = Pt(7 if name=="Normal" else 8)
        p.space_before = Pt(14 if name.startswith("Heading") else 0)
        if name.startswith("Heading"):
            p.keep_with_next = True
    doc.styles["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.styles["Normal"].paragraph_format.widow_control = True
    doc.styles["Heading 1"].paragraph_format.page_break_before = True
    for name in ["TOC 1", "TOC 2"]:
        st=doc.styles[name] if name in doc.styles else doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        st.base_style=doc.styles["Normal"]
        set_font(st,11,False)
        st.paragraph_format.space_before=Pt(0)
        st.paragraph_format.space_after=Pt(2)
        st.paragraph_format.line_spacing=1.0
        st.paragraph_format.keep_with_next=False
        st.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
    # Remove template theme accents and paragraph rules from all heading styles.
    for style in doc.styles:
        if style.type == 1:
            for color in style.element.findall(".//"+qn("w:color")):
                for a in ["w:themeColor","w:themeTint","w:themeShade"]:
                    color.attrib.pop(qn(a),None)
            for border in style.element.findall(".//"+qn("w:pBdr")):
                border.getparent().remove(border)
    s.different_first_page_header_footer = True
    hp=s.header.paragraphs[0]
    hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=hp.add_run("TACS và mở rộng ngữ cảnh ảnh văn bản")
    r.font.name="Times New Roman";r.font.size=Pt(10);r.font.color.rgb=RGBColor(0,0,0)
    fp=s.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field(fp,"PAGE","1")
    doc.core_properties.title="Tái triển khai TACS và khảo sát mở rộng ngữ cảnh ảnh văn bản"
    doc.core_properties.subject="Báo cáo bài tập lớn với kết quả thực nghiệm từ notebook"
    doc.core_properties.author="Sinh viên thực hiện"
    doc.core_properties.keywords="TACS; DTD; SUN397; ISIC; multimodal; caption; ablation"
    xml_child(doc.settings.element,"w:updateFields",{"w:val":"true"})


def add_cover(doc):
    for text, size in [("[TÊN TRƯỜNG]",15),("[KHOA HOẶC VIỆN]",14),("",12),
                       ("BÁO CÁO BÀI TẬP LỚN",17),("[TÊN HỌC PHẦN]",14)]:
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=p.add_run(text);r.bold=True;r.font.size=Pt(size)
    p=doc.add_paragraph(style="Title");p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(26);p.paragraph_format.space_after=Pt(22)
    p.add_run("TÁI TRIỂN KHAI TACS\nVÀ KHẢO SÁT MỞ RỘNG\nNGỮ CẢNH ẢNH VĂN BẢN")
    p=doc.add_paragraph("Phân loại ảnh và phân đoạn với ngữ cảnh được chọn theo tác vụ",style="Subtitle")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for text in ["Giảng viên hướng dẫn: [Họ và tên giảng viên]",
                 "Sinh viên thực hiện: [Họ và tên]", "Mã sinh viên: [Mã sinh viên]",
                 "Lớp hoặc nhóm: [Lớp hoặc nhóm]", ""]:
        p=doc.add_paragraph(text);p.paragraph_format.left_indent=Cm(1.0)
        p.paragraph_format.space_after=Pt(10)
    p=doc.add_paragraph("Tháng 10 năm 2026");p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()


def text_runs(p,text):
    for part in re.split(r"(\*\*.*?\*\*|`[^`]+`)",text):
        if part.startswith("**") and part.endswith("**"):
            p.add_run(part[2:-2]).bold=True
        elif part.startswith("`") and part.endswith("`"):
            run=p.add_run(part[1:-1]);run.font.name="Consolas";run.font.size=Pt(10.5)
        else:
            p.add_run(part)


def table(doc,key):
    d=TABLES[key]
    cap=doc.add_paragraph(d["title"],style="Caption")
    cap.paragraph_format.keep_with_next=True
    cap.runs[0].bold=True
    t=doc.add_table(rows=1,cols=len(d["headers"]))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=d["widths"] or [6.35/len(d["headers"])]*len(d["headers"])
    scale=6.35/sum(widths)
    for col,w in zip(t.columns,widths): col.width=Inches(w*scale)
    pr=t._tbl.tblPr
    borders=xml_child(pr,"w:tblBorders")
    for edge in ["top","left","bottom","right","insideH","insideV"]:
        xml_child(borders,f"w:{edge}",{"w:val":"single","w:sz":"5","w:color":"D9D9D9"})
    margins=xml_child(pr,"w:tblCellMar")
    for edge in ["top","bottom"]: xml_child(margins,f"w:{edge}",{"w:w":"90","w:type":"dxa"})
    for edge in ["left","right"]: xml_child(margins,f"w:{edge}",{"w:w":"100","w:type":"dxa"})
    xml_child(t.rows[0]._tr.get_or_add_trPr(),"w:tblHeader",{"w:val":"true"})
    allrows=[d["headers"]]+d["rows"]
    for i,values in enumerate(allrows):
        row=t.rows[0] if i==0 else t.add_row()
        xml_child(row._tr.get_or_add_trPr(),"w:cantSplit")
        for j,value in enumerate(values):
            cell=row.cells[j];cell.width=Inches(widths[j]*scale)
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=cell.paragraphs[0]
            p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.line_spacing=1.08
            keep_table = len(d["rows"])<=4 or key=="alignment_fit"
            p.paragraph_format.keep_with_next=(i==0 or (keep_table and i<len(d["rows"])))
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or len(str(value))>23 else WD_ALIGN_PARAGRAPH.CENTER
            run=p.add_run(str(value));run.font.name="Times New Roman";run.font.size=Pt(10.5)
            if i==0:
                run.bold=True;run.font.color.rgb=RGBColor(255,255,255)
                xml_child(cell._tc.get_or_add_tcPr(),"w:shd",{"w:fill":"444444"})
            elif i%2==0:
                xml_child(cell._tc.get_or_add_tcPr(),"w:shd",{"w:fill":"F4F4F4"})
    if d["note"]:
        p=doc.add_paragraph(d["note"])
        p.paragraph_format.space_before=Pt(5);p.paragraph_format.space_after=Pt(10)
        p.paragraph_format.line_spacing=1.05
        for r in p.runs:r.font.size=Pt(10.5);r.italic=True
    else:
        p=doc.add_paragraph()
        p.paragraph_format.space_before=Pt(0)
        p.paragraph_format.space_after=Pt(5)
        p.paragraph_format.line_spacing=Pt(2)
        p.add_run().font.size=Pt(1)


def equation(doc,text):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(6);p.paragraph_format.space_after=Pt(10)
    # Linear Unicode math is inserted as editable OMML, not unparsed LaTeX.
    eq=OxmlElement("m:oMath");r=OxmlElement("m:r");t=OxmlElement("m:t");t.text=text
    r.append(t);eq.append(r);p._p.append(eq)


def figure_placeholder(doc,text):
    parts=text.split("|",1)
    label=parts[0].strip();description=parts[1].strip() if len(parts)>1 else ""
    p=doc.add_paragraph("[VỊ TRÍ BỔ SUNG HÌNH]")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(14);p.paragraph_format.space_after=Pt(9)
    p.paragraph_format.keep_with_next=True
    p.runs[0].italic=True;p.runs[0].font.color.rgb=RGBColor(90,90,90)
    p=doc.add_paragraph(label,style="Caption");p.paragraph_format.keep_with_next=True
    p.runs[0].bold=True
    p=doc.add_paragraph(description)
    p.paragraph_format.line_spacing=1.05;p.paragraph_format.space_after=Pt(12)
    for r in p.runs:r.font.size=Pt(10.5);r.italic=True


def source_appendix(doc,data):
    doc.add_heading("PHỤ LỤC A DANH MỤC KẾT QUẢ VÀ NGUỒN ĐỐI CHIẾU",level=1)
    doc.add_paragraph("Mã A dùng cho tám ablation text; C dùng cho nguồn caption; S dùng cho nghiên cứu kiến trúc; I dùng cho phân đoạn; V dùng cho visual lịch sử; U01 là run caption query SUN397. Đường dẫn dưới đây tương đối với thư mục gốc TACS. Các nguồn được đọc ở trạng thái đã lưu, không thực thi cell hoặc nạp pickle checkpoint khi viết báo cáo. A05 và C07 trỏ cùng run, không phải hai lần lặp độc lập.")
    for ev in EVIDENCE:
        p=doc.add_paragraph()
        p.add_run(ev["id"]+"  ").bold=True
        # Permit line breaks in long paths without changing underlying file names.
        p.add_run(ev["path"].replace("/","/\u200b").replace("_","_\u200b").replace("-","-\u200b"))
        p.paragraph_format.line_spacing=1.05
        for r in p.runs:r.font.size=Pt(10.5)
        p=doc.add_paragraph("SHA256: "+ev["sha256"][:16]+"…")
        p.paragraph_format.space_after=Pt(8)
        for r in p.runs:r.font.size=Pt(9.5)
    doc.add_heading("PHỤ LỤC B CHỈ SỐ TEST GỐC VÀ ĐỊNH DANH DỮ LIỆU",level=1)
    doc.add_paragraph("Các giá trị dưới đây giữ độ chính xác bảng final test đã lưu. S02/S03 có full precision trong JSON và prediction CSV; số ở đây thống nhất 6 chữ số để đối chiếu. Việc làm tròn không tạo thêm độ chính xác cho nguồn chỉ in 4 chữ số.")
    TABLES["raw"]={"title":"Bảng B.1 Chỉ số DTD thang 0 đến 1", "headers":["Mã","Cấu hình","Accuracy","Macro F1","CE"],
                  "rows":[[r["id"],r["name"],n(r["test"]["accuracy"],6),n(r["test"]["macro_f1"],6),n(r["test"]["loss"],6)] for r in data["ablations"]+data["captions"][:6]],
                  "widths":[.55,2.4,1.05,1.05,1.05],"note":"C07/C08 và S01–S04 được trình bày trong các bảng riêng ở chương 5; không gộp các nhóm thành một xếp hạng chung."}
    table(doc,"raw")
    p=doc.add_paragraph("CSV Prefix chuẩn cho chuỗi S01–S04: data/captions/dtd/grounded_controls/captions_prefix_control.csv. SHA256 đầy đủ: b71de4b211f5232b9f0aba85e874ddf293cbc8e770d83613719c1368f7ed5011.")
    for r in p.runs:r.font.size=Pt(11)
    doc.add_paragraph("Bộ evidence đọc lại gồm bảng output notebook, checksum toàn file và các export S02/S03. Chỉ S02/S03 trong chuỗi kiến trúc có prediction CSV đầy đủ được kiểm lại trực tiếp khi dựng tài liệu này. S01/S04 và các bộ A/C vẫn chịu giới hạn đã nêu trong chương 5.")


def build(data):
    doc=Document();setup(doc);add_cover(doc)
    for block in re.split(r"\n\s*\n",SOURCE.read_text(encoding="utf-8").strip()):
        block=block.strip()
        if not block:continue
        if block.startswith("[[TABLE:"):
            table(doc,block[8:-2]);continue
        if block.startswith("[[FIG:"):
            figure_placeholder(doc,block[6:-2]);continue
        if block.startswith("[[EQ:"):
            equation(doc,block[5:-2]);continue
        if block=="[[CONTENTS]]":
            p=doc.add_paragraph()
            field(p,'TOC \\o "1-2" \\h \\z \\u',"Mục lục tự động cập nhật theo các đề mục của tài liệu")
            continue
        if block.startswith("# "):
            if block[2:]=="MỤC LỤC":
                p=doc.add_paragraph(block[2:])
                p.paragraph_format.page_break_before=True
                p.paragraph_format.keep_with_next=True
                p.paragraph_format.space_before=Pt(14)
                p.paragraph_format.space_after=Pt(8)
                p.runs[0].bold=True;p.runs[0].font.size=Pt(16)
            else:
                doc.add_heading(block[2:],level=1)
            continue
        if block.startswith("## "):
            doc.add_heading(block[3:],level=2);continue
        if block.startswith("### "):
            doc.add_heading(block[4:],level=3);continue
        if block.startswith("- "):
            for line in block.splitlines():
                p=doc.add_paragraph(style="List Bullet");text_runs(p,line.removeprefix("- "))
            continue
        p=doc.add_paragraph();text_runs(p,block.replace("\n"," "))
    source_appendix(doc,data)
    doc.save(TARGET)
    report={"created":"2026-10-27","report":str(TARGET),"notebook_cells_executed":False,
            "checkpoint_pickle_loaded":False,"sources":EVIDENCE,
            "result_groups":{k:v for k,v in data.items() if k!="studies"},
            "studies":[{k:v for k,v in s.items() if k!="tables"} for s in data["studies"]]}
    (OUT/"report_verified_evidence.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"docx":str(TARGET),"bytes":TARGET.stat().st_size,"paragraphs":len(doc.paragraphs),
                      "tables":len(doc.tables),"sources":len(EVIDENCE)},ensure_ascii=False))


if __name__=="__main__":
    dataset=gather()
    if "--audit-only" in sys.argv:
        print(json.dumps({"verified_notebooks":len(EVIDENCE),"ablations":len(dataset["ablations"]),"caption_runs":len(dataset["captions"]),"studies":len(dataset["studies"])},ensure_ascii=False))
    else:
        make_tables(dataset);build(dataset)
