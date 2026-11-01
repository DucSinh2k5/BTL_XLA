"""Create the team handbook from prose and reverified saved experiment outputs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

import build_course_report_docx as evidence


ROOT = evidence.ROOT
OUT = ROOT / "reports/cam_nang_nhom"
SOURCE = OUT / "NOI_DUNG_CAM_NANG_TACS_MTACS.md"
TARGET = OUT / "CAM_NANG_HOC_TAP_VA_BAN_GIAO_TACS_MTACS.docx"


def register_tables(data):
    tables = evidence.TABLES

    def add(key, title, headers, rows, widths, note=""):
        tables[key] = dict(title=title, headers=headers, rows=rows, widths=widths, note=note)

    add("map", "Bảng 1 Ba loại thí nghiệm không được nhầm lẫn",
        ["Nhóm", "Biến được thay đổi", "Đối tượng giữ tương đối cố định"], [
            ["A01–A08 vai trò text", "Text query, candidate, selector; shuffle, screening, redaction", "DTD split 1, L/16, 384, 50 epoch, seed 42"],
            ["C01–C06 nguồn caption", "Florence Prefix, FirstValid, Single, Multi; Qwen 8B, 32B", "Kiến trúc full_both và định nghĩa training chung"],
            ["S01–S04 kiến trúc", "Chuẩn hóa, alignment, semantic selector", "CSV Florence Prefix đã khóa; candidate_only"],
        ], [1.35,2.65,2.35],
        "Tên notebook số 4 của ba nhóm không chỉ cùng một mô hình. C07/C08 là hai tham chiếu lịch sử, không phải hai method caption mới cùng lô.")
    add("protocol", "Bảng 2 Những thiết lập phải nhớ khi so sánh",
        ["Thuộc tính", "Paper TACS", "DTD của dự án", "ISIC của dự án"], [
            ["Backbone", "DINOv3 S/16", "DINOv3 L/16", "DINOv3 S/16 đóng băng"],
            ["Kích thước ảnh", "256 × 256", "384 × 384", "448 × 448"],
            ["Ngân sách", "100 epoch; 3 seed", "50 epoch; seed 42; split 1", "Tối đa 100; early stopping; seed 42"],
            ["Context", "20% train; K = 1", "376 candidate; K = 1", "400 candidate; K = 1"],
            ["Chọn best", "Theo thí nghiệm tác giả", "Min validation CE", "Max validation hard Dice"],
            ["Selector thực thi", "Gumbel ST và policy", "Categorical gather và policy", "Theo source visual và bản mở rộng"],
        ], [1.15,1.65,1.8,1.75],
        "Các số liệu của dự án không phải tái lập đồng cấu hình với paper. Mỗi run SUN397 phải đọc cấu hình riêng; không gán toàn bộ SUN thành L/16.")
    add("datasets", "Bảng 3 Các tập dữ liệu trong mạch chính",
        ["Dataset", "Train", "Val", "Test", "Candidate pool"], [
            ["SUN397 partition 01", "17.865", "1.985", "19.850", "3.573 trong run query text U01"],
            ["DTD official split 1", "1.880", "1.880", "1.880", "376 ảnh train"],
            ["ISIC 2017", "2.000", "150", "600", "400 ảnh train"],
        ], [1.65,.85,.85,.9,2.1],
        "SUN lấy 10% validation từ training partition. DTD dùng nguyên val chính thức, không áp dụng cách tách SUN. Pool không phải tập test thứ hai.")
    roles = [
        ("visual_only","Không","Không","Không","Mask text"),
        ("query_only","Có","Không","Không","Screened"),
        ("candidate_only","Không","Có","Không","Screened"),
        ("both_fusion","Có","Có","Không","Screened"),
        ("full_both","Có","Có","Có","Screened"),
        ("full_shuffle_both","Có","Có","Có","Đảo cả hai vai trò, giữ mask"),
        ("full_unscreened","Có","Có","Có","Trước lọc"),
        ("full_redacted","Có","Có","Có","Che tên lớp và alias"),
    ]
    add("roles", "Bảng 4 Ma trận tám ablation vai trò văn bản",
        ["Mode", "Text query fusion", "Text candidate fusion", "Text selector", "Caption"],
        roles, [1.45,1,1.15,.95,1.8],
        "Screened theo policy của run, không có nghĩa đã được con người xác nhận. visual_only vẫn có ảnh context.")
    add("ablations", "Bảng 5 Kết quả thật của tám ablation DTD",
        ["Mã và cấu hình", "Test Acc", "Test Macro F1", "CE", "Δ F1 so visual pp"], [
            [r["id"]+" "+r["name"], evidence.n(r["test"]["accuracy"],6),
             evidence.n(r["test"]["macro_f1"],6), evidence.n(r["test"]["loss"],6),
             f"{100*(float(r['test']['macro_f1'])-float(data['ablations'][0]['test']['macro_f1'])):+.4f}"]
            for r in data["ablations"]
        ], [2.15,1,1.05,1,1.15],
        "Tất cả thang 0–1 trừ cột chênh lệch điểm phần trăm. Một seed, một split; đây không phải bảng chứng minh có ý nghĩa thống kê qua nhiều lần train.")
    add("captions", "Bảng 6 So sánh nguồn caption trong full_both",
        ["Mã và caption", "Val F1", "Test Acc", "Test F1", "Best epoch"], [
            [r["id"]+" "+r["name"], evidence.n(r["val"]["macro_f1"],6),
             evidence.n(r["test"]["accuracy"],6), evidence.n(r["test"]["macro_f1"],6),
             str([39,35,45,45,36,40,42,39][i])]
            for i,r in enumerate(data["captions"])
        ], [2.45,1,1,1,.9],
        "C07 Florence old trùng run A05, không đếm như một lần lặp độc lập. C08 là tham chiếu decoder LoRA. Các method không có cùng tỷ lệ caption hợp lệ.")
    add("coverage", "Bảng 7 Độ phủ caption của sáu method chính",
        ["Nguồn", "Valid toàn bộ", "Train", "Val", "Test", "Pool"], [
            ["Prefix",4749,1573,1585,1591,324],
            ["FirstValid",5640,1880,1880,1880,376],
            ["SingleGrounded",5640,1880,1880,1880,376],
            ["MultiGrounded",5640,1880,1880,1880,376],
            ["Qwen 8B",4211,1418,1400,1393,295],
            ["Qwen 32B",4111,1391,1374,1346,285],
        ], [1.75,1.05,.9,.9,.9,.85],
        "Đây là độ phủ đầu vào, không phải độ chính xác caption. Candidate caption rate trong metric test là tỷ lệ trên context được chọn, có thể khác valid pool/376.")
    add("studies", "Bảng 8 Chuỗi nghiên cứu kiến trúc với cùng CSV Prefix",
        ["Study", "Thay đổi chính", "Val F1", "Test F1", "Test CE", "C gate"], [
            [f"S0{r['index']}", ["Baseline candidate_only","L2 text trước fusion","S02 + mLTM inspired alignment","S03 + image to caption selector"][r['index']-1],
             evidence.n(r["val"]["macro_f1"],6), evidence.n(r["test"]["macro_f1"],6),
             evidence.n(r["test"]["loss"],6),evidence.n(r["test"]["mean_candidate_gate"],6)]
            for r in data["studies"]
        ], [.55,2.05,1,1,1,.75],
        "Best epoch của cả bốn là 38. Query text tắt. S02/S03 đã kiểm lại metric từ 1.880 prediction mỗi split; S01/S04 dựa output đã lưu.")
    add("interventions", "Bảng 9 Can thiệp validation với checkpoint cố định",
        ["Study", "Fusion đúng F1", "Tắt text F1", "Đảo text F1"], [
            ["S02",.835824,.839112,.839715],
            ["S03",.834947,.835391,.837773],
            ["S04",.838806,.839080,.841044],
        ], [1.1,1.75,1.75,1.75],
        "Giữ context cố định cho can thiệp fusion. S04 selector còn có can thiệp riêng, phải chọn lại context; không gộp hai loại phép thử.")
    add("segmentation", "Bảng 10 Phân đoạn ISIC với và không có Qwen caption",
        ["Run", "Test Dice", "Test IoU", "Best epoch", "Dừng"], [
            ["I01 visual S/16 448",data['isic1']['dice'],data['isic1']['iou'],49,64],
            ["I02 Qwen 8B full_both",data['isic2']['dice'],data['isic2']['iou'],26,41],
        ], [2.2,1.15,1.15,.95,.9],
        "Mỗi run seed 42. Florence Prefix ISIC mới có notebook thiết kế, chưa có output thực nghiệm để thêm vào bảng.")
    add("folders", "Bảng 11 Bản đồ thư mục để bàn giao",
        ["Thư mục từ gốc TACS", "Chứa gì", "Lưu ý khi sửa"], [
            ["notebooks/00_data_preparation", "Chuẩn bị dữ liệu", "Không đưa test vào pool"],
            ["notebooks/01_paper_reproduction", "Các task và bảng tái triển khai", "Đọc đúng dataset, backbone, kích thước"],
            ["notebooks/02_improvements", "Text ablation, caption, architecture, segmentation", "Phân biệt template và run đã train"],
            ["notebooks/03_analysis_and_evaluation", "Phân tích và đánh giá", "Không ghi đè prediction gốc"],
            ["notebooks/90_archive", "Notebook cũ", "Không chọn theo tên file đẹp hơn"],
            ["experiments/completed", "Notebook đã có kết quả", "Giữ output và checksum"],
            ["experiments/running và planned", "Run đang chạy và dự kiến", "Không xem kế hoạch là kết quả"],
            ["data/captions và annotations", "CSV caption và reference", "Không nhầm full CSV với reference 188 ảnh"],
            ["artifacts", "Checkpoint, bank, embedding theo chỗ đã có", "Thư mục trống không nghĩa có mô hình"],
            ["src và tools", "Helper dùng chung; builder và audit", "Model/selector còn nhiều code trong notebook"],
            ["reports và manifests", "Báo cáo, evidence, catalog, lịch sử chuyển", "Kết luận đi cùng nguồn"],
            ["apps/sun397_demo", "Web FastAPI và giao diện", "Hiện chưa nối model suy luận"],
        ], [2.1,2.1,2.15])
    add("assets", "Bảng 12 Tài nguyên cần cho từng loại notebook",
        ["Loại run", "Bắt buộc", "Không tự động cần"], [
            ["Visual DTD", "DTD; DINO code và đúng weights; môi trường train", "Florence, Qwen, reference CSV"],
            ["Caption classification từ CSV", "Các mục visual; full caption CSV; CLIP text assets", "Captioner weights nếu không sinh lại"],
            ["Sinh Florence CSV", "Dataset; Florence đầy đủ; CLIP audit nếu dùng; runtime tương thích", "Classifier checkpoint"],
            ["Sinh Qwen CSV", "Dataset; Qwen 8B hoặc 32B đầy đủ; processor; runtime tương thích", "Florence weights"],
            ["Fine tune Florence", "Captioner; reference đã duyệt; image pixels; validation policy", "Nhãn test để hướng dẫn teacher"],
            ["ISIC full_both", "Ảnh và mask ISIC; visual assets; captioner nếu sinh CSV; CLIP", "DTD labels hoặc CSV DTD"],
        ], [1.45,3.2,1.7],
        "Một Input chứa DINO repository không thể thay cho captions.csv; folder CLIP không chứa Florence; file .pt bất kỳ không thay cho checkpoint đúng kiến trúc.")
    add("handover", "Bảng 13 Tối thiểu phải lưu sau mỗi run",
        ["Nhóm", "File hoặc thông tin", "Mục đích"], [
            ["Danh tính", "Config, seed, split, class mapping, pool IDs, source notebook", "Xác định đúng thí nghiệm"],
            ["Dữ liệu text", "CSV gốc và SHA256, policy, audit, mask, cache fingerprint", "Biết chính xác text đã dùng"],
            ["Training", "history.csv; best.pt; last/resume nếu notebook hỗ trợ", "Vẽ curve; suy luận; tiếp tục đúng trạng thái"],
            ["Đánh giá", "Val/test metrics và prediction CSV; intervention CSV", "Kiểm số; so sánh theo từng ảnh"],
            ["Môi trường", "Python, torch, transformers, CUDA, device và dependency log", "Giải thích sai khác runtime"],
            ["Phân đoạn", "Ngưỡng mask; resize; per image Dice/IoU; output minh họa", "Không trộn soft/hard hoặc micro/macro"],
        ], [1.15,3.3,1.9])


def make_cover(doc):
    p=doc.add_paragraph(style="Title")
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before=Pt(66)
    p.add_run("CẨM NANG HỌC TẬP\nVÀ BÀN GIAO DỰ ÁN\nTACS VÀ M TACS")
    p=doc.add_paragraph("Từ lựa chọn ngữ cảnh ảnh đến thí nghiệm ảnh và văn bản", style="Subtitle")
    p.paragraph_format.space_after=Pt(24)
    doc.add_paragraph("Dành cho thành viên nhóm cần hiểu nguyên lý, chạy lại thí nghiệm, đọc kết quả và chỉnh sửa báo cáo bài tập lớn.")
    doc.add_paragraph("Phạm vi chính  SUN397  DTD  ISIC 2017")
    doc.add_paragraph("Phiên bản bàn giao ngày 27 tháng 10 năm 2026")
    doc.add_paragraph("Nhóm thực hiện  [Bổ sung tên nhóm]")
    doc.add_paragraph("Tài liệu đi cùng báo cáo chính, không thay thế notebook, dữ liệu và checkpoint gốc.")
    doc.add_page_break()


def code_block(doc, text):
    for line in text.splitlines():
        p=doc.add_paragraph(style="Code")
        p.paragraph_format.keep_with_next=False
        p.add_run(line or " ")
    p=doc.add_paragraph()
    p.paragraph_format.space_after=Pt(3)
    p.paragraph_format.line_spacing=Pt(2)
    p.add_run().font.size=Pt(1)


def build(data):
    doc=Document()
    evidence.setup(doc)
    normal=doc.styles['Normal']
    normal.font.size=Pt(12)
    normal.paragraph_format.line_spacing=1.14
    normal.paragraph_format.space_after=Pt(7)
    normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
    for name in ('Heading 1','Heading 2','Heading 3'):
        doc.styles[name].paragraph_format.keep_with_next=True
    doc.styles['Heading 1'].font.size=Pt(17)
    doc.styles['Heading 1'].paragraph_format.page_break_before=True
    doc.styles['Heading 2'].font.size=Pt(13)
    doc.styles['Heading 3'].font.size=Pt(12)
    code=doc.styles['Code'] if 'Code' in doc.styles else doc.styles.add_style('Code',1)
    code.font.name='Consolas';code.font.size=Pt(9)
    code.paragraph_format.space_after=Pt(2)
    code.paragraph_format.line_spacing=1.05
    code.paragraph_format.left_indent=Inches(.12)
    doc.sections[0].header.paragraphs[0].text="Cẩm nang học tập và bàn giao TACS và M TACS"
    doc.core_properties.title="Cẩm nang học tập và bàn giao dự án TACS và M TACS"
    doc.core_properties.subject="Giải thích nguyên lý, dòng thí nghiệm, tái lập và biên tập báo cáo"
    doc.core_properties.author="Nhóm thực hiện dự án TACS"
    make_cover(doc)
    lines=SOURCE.read_text(encoding='utf-8').splitlines()
    i=0
    glossary=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith('```'):
            contents=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                contents.append(lines[i]);i+=1
            code_block(doc,'\n'.join(contents));i+=1;continue
        if line=='[[CONTENTS]]':
            p=doc.add_paragraph()
            evidence.field(p,'TOC \\o "1-1" \\h \\z \\u','Mục lục được cập nhật khi mở bằng Word')
            i+=1;continue
        if line.startswith('[[TABLE:'):
            key=line[8:-2]
            if key=='folders':
                for row in evidence.TABLES[key]['rows']:
                    row[0]=row[0].replace('/','/\u200b').replace('_','_\u200b')
            evidence.table(doc,key);i+=1;continue
        if line.startswith('[[EQ:'):
            evidence.equation(doc,line[5:-2]);i+=1;continue
        if line=='[[SOURCES]]':
            for ev in evidence.EVIDENCE:
                p=doc.add_paragraph()
                p.add_run(ev['id']+'  ').bold=True
                p.add_run(ev['path'].replace('\\','/').replace('/','/\u200b').replace('_','_\u200b').replace('-','-\u200b'))
                p.paragraph_format.line_spacing=1.04
                p.paragraph_format.keep_with_next=True
                for r in p.runs:r.font.size=Pt(10)
                p=doc.add_paragraph('SHA256  '+ev['sha256'][:16])
                p.paragraph_format.space_after=Pt(6)
                for r in p.runs:r.font.size=Pt(9)
            i+=1;continue
        if line.startswith('# '):
            glossary=line.startswith('# Phụ lục B ')
            if line[2:]=='Mục lục':
                p=doc.add_paragraph('Mục lục')
                p.paragraph_format.page_break_before=True
                p.paragraph_format.keep_with_next=True
                p.paragraph_format.space_after=Pt(10)
                p.runs[0].font.size=Pt(17);p.runs[0].bold=True
            else:
                doc.add_heading(line[2:],1)
            i+=1;continue
        if line.startswith('## '):
            doc.add_heading(line[3:],2);i+=1;continue
        if line.startswith('### '):
            doc.add_heading(line[4:],3);i+=1;continue
        if line.startswith('- '):
            p=doc.add_paragraph(style='List Bullet');evidence.text_runs(p,line[2:]);i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\[\[|```|- )',lines[i]):
            paragraph.append(lines[i].strip());i+=1
        p=doc.add_paragraph();evidence.text_runs(p,' '.join(paragraph))
        if glossary:
            p.paragraph_format.line_spacing=1.08
            p.paragraph_format.space_after=Pt(5)
            for run in p.runs:run.font.size=Pt(11.5)
    OUT.mkdir(parents=True,exist_ok=True)
    doc.save(TARGET)
    manifest={
        'created':'2026-10-27','artifact':str(TARGET),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'notebook_cells_executed':False,'checkpoint_pickle_loaded':False,
        'source_notebooks':evidence.EVIDENCE,'tables':len(doc.tables),'paragraphs':len(doc.paragraphs),
        'experiment_groups':{
            'ablations':data['ablations'],'caption_methods':data['captions'],
            'architecture_studies':[{k:v for k,v in s.items() if k!='tables'} for s in data['studies']],
        },
    }
    (OUT/'handbook_evidence.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'docx':str(TARGET),'bytes':TARGET.stat().st_size,'paragraphs':len(doc.paragraphs),
                      'tables':len(doc.tables),'verified_notebook_entries':len(evidence.EVIDENCE)},ensure_ascii=False))


if __name__=='__main__':
    saved=evidence.gather()
    register_tables(saved)
    build(saved)
