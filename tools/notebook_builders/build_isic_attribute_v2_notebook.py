"""Create a new caption-policy experiment; leave deployed bundles untouched."""
from __future__ import annotations

import ast
import copy
import hashlib
import json

from _project_bootstrap import PROJECT_ROOT
import build_isic_segmentation_web_notebooks as baseline

TARGET = PROJECT_ROOT / "notebooks/02_improvements/segmentation/full_both/florence_attributes_v2/isic2017/mtacs-isic2017-vitl16-448-florence-attributes-v2.ipynb"
POLICY = "isic_visible_attributes_v2"
METHOD = "florence_extractive_attributes_v2"


def definition(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise ValueError(name)


def caption_core():
    source = baseline.caption_core()
    source = source.replace('METHOD = "florence_prefix_control"', f'METHOD = {METHOD!r}')
    source = source.replace('"isic_visible_attributes_v1"', repr(POLICY))
    source = source.replace('"greedy_beam1_prefix_cleanup_first8"', '"greedy_beam1_extract_visible_attribute_phrase"')
    source = source.replace('def generate_isic_records(', 'def generate_prefix_records(', 1)
    source = source.replace('"code_sha256": payload["code_sha256"]}\n        output = Path(payload["output"])',
        '"code_sha256": payload["code_sha256"],\n                    "long_response_source": prior_caption_registration(payload)[0]}\n        output = Path(payload["output"])')
    helpers = (PROJECT_ROOT / "src/captioning/isic_caption_attributes_v2.py").read_text("utf-8")
    additions = '\n\n'.join(definition(helpers, name) for name in (
        "attribute_caption_details", "attribute_caption", "prior_caption_registration", "generate_attribute_records"))
    additions += '''

def clean_dtd_caption(text):
    return attribute_caption(text)

def generate_isic_records(payload, identity, factory=IsicFlorencePrefixCaptioner):
    return generate_attribute_records(payload, identity, factory)
'''
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.If) and '__name__' in ast.get_source_segment(source, n.test))
    lines = source.splitlines(keepends=True)
    source = ''.join(lines[:main.lineno - 1]) + additions + '\n\n' + ''.join(lines[main.lineno - 1:])
    ast.parse(source)
    if '"long_response_source": prior_caption_registration(payload)[0]' not in source:
        raise ValueError("Generation protocol source changed; inspect before rebuilding")
    return source


def sources():
    result = baseline.embedded_sources("mtacs")
    result["isic_caption_core.py"] = caption_core()
    service = result["isic_caption_web.py"]
    before = '    request = output / "request.json"'
    assert service.count(before) == 1
    result["isic_caption_web.py"] = service.replace(before,
        '    payload["prior_captions_source"] = os.getenv("ISIC_ATTRIBUTE_LONG_CAPTIONS_SOURCE", "")\n' + before)
    return result


def frame_helper_source():
    source = (PROJECT_ROOT / "src/deployment/isic_export_only.py").read_text("utf-8")
    return '''import csv, hashlib, json, os, re
from pathlib import Path
import numpy as np
from PIL import Image
''' + '\n\n'.join(definition(source, n) for n in ("sha", "read_rows", "relocate_frames"))


DATASET_RECOVERY = '''
def find_saved_run(reference):
    roots = [Path(reference).expanduser()] if reference else [Path('/kaggle/input')]
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            path = Path(current)
            dirs[:] = [n for n in dirs if n not in {'images','masks','candidate_images','ground_truth_masks','caption_models','generated','caption_runtime','vendor','.cache','__pycache__'}]
            if len(path.relative_to(root).parts) >= 10:
                dirs[:] = []
            binding = path / 'run_binding.json'
            if not binding.is_file():
                continue
            recorded = json.loads(binding.read_text('utf-8'))
            if recorded.get('config', {}).get('mode') == 'mtacs' and recorded['config'].get('schema') == 'isic2017_segmentation_web_v1':
                found.append(path.resolve())
    found = sorted(set(found))
    if len(found) != 1:
        raise FileNotFoundError(f'Attach the old ISIC M-TACS output/bundle and set SAVED_SPLITS_SOURCE to its exact directory. Found: {found}')
    return found[0]

SAVED_METADATA = find_saved_run(SAVED_SPLITS_SOURCE)
OLD_BINDING = json.loads((SAVED_METADATA / 'run_binding.json').read_text('utf-8'))
OLD_PROTOCOL = json.loads((SAVED_METADATA / 'protocol.json').read_text('utf-8'))
if OLD_PROTOCOL.get('identity') != workflow.object_hash(OLD_BINDING):
    raise ValueError('Original saved metadata binding no longer matches its protocol')
RECORDED_CONFIG = OLD_BINDING['config']
if any(RECORDED_CONFIG[k] != CONFIG[k] for k in ('backbone','feature_dim','image_size','context_top_k','pool_fraction','seed')):
    raise ValueError('Original saved run architecture/split settings do not match this experiment')
frames = {role: snapshot_io.read_rows(SAVED_METADATA / f'{role}_split.csv') for role in ('train','val','test')}
for role, rows in frames.items():
    compact = [{k:r[k] for k in ('image_id','image_sha256','mask_sha256','group_id')} for r in rows]
    if compact != OLD_BINDING['splits'][role]:
        raise ValueError('Original split was changed: ' + role)
_, FRAMES = snapshot_io.relocate_frames(frames, ISIC_ROOT, workflow, model_code.io_utils)
pool_rows = snapshot_io.read_rows(SAVED_METADATA / 'candidate_pool.csv')
if [r['image_id'] for r in pool_rows] != OLD_BINDING['pool_ids']:
    raise ValueError('Original candidate pool order changed')
by_id = {r['image_id']:r for r in FRAMES['train']}
CANDIDATE_POOL = [by_id[r['image_id']] for r in pool_rows]
if len(CANDIDATE_POOL) != round(len(FRAMES['train']) * CONFIG['pool_fraction']):
    raise ValueError('Unexpected saved pool size')
if not LONG_CAPTIONS_SOURCE:
    LONG_CAPTIONS_SOURCE = str(SAVED_METADATA.parents[1] if SAVED_METADATA.name == 'original_run' else SAVED_METADATA)
os.environ['ISIC_ATTRIBUTE_LONG_CAPTIONS_SOURCE'] = LONG_CAPTIONS_SOURCE
print('Exact saved train/val/test/pool:', *[len(FRAMES[r]) for r in ('train','val','test')], len(CANDIDATE_POOL))
print('Long responses source:', LONG_CAPTIONS_SOURCE)
'''


QUALITY_CELL = '''
import csv, collections, re
with (CAPTION_OUTPUT / 'captions.csv').open(encoding='utf-8-sig', newline='') as handle:
    new_caption_rows = list(csv.DictReader(handle))
counts = collections.Counter(r['final_caption'] for r in new_caption_rows if r['caption_valid'].lower() == 'true')
generic = "person's skin The skin appears to be"
quality = {
    'policy': caption_service.core.POLICY, 'method': caption_service.core.METHOD,
    'rows': len(new_caption_rows), 'valid': sum(counts.values()), 'unique_valid_captions': len(counts),
    'generic_prefix_count': sum(r['final_caption'] == generic for r in new_caption_rows),
    'most_common_valid_captions': counts.most_common(10),
    'top_valid_caption_fraction_all_images': counts.most_common(1)[0][1] / len(new_caption_rows) if counts else 0.,
    'not_caption_accuracy_or_segmentation_metrics': True,
    'human_review_required': 'Attribute extraction can retain Florence hallucinations. Diversity alone does not prove factual accuracy.',
}
if quality['generic_prefix_count']:
    raise ValueError('Generic prefix unexpectedly survived attributes-v2')
review = []
for row in new_caption_rows:
    review.append({k:row[k] for k in ('image_id','split','source_long_caption','final_caption','caption_valid')})
workflow.csv_write(OUTPUT_DIR / 'caption_review_v2.csv', review)
model_code.io_utils.write_json(OUTPUT_DIR / 'caption_quality_v2.json', quality)
display(quality)
display(review[:8])
from IPython.display import FileLink
for path in (CAPTION_OUTPUT / 'captions.csv', CAPTION_OUTPUT / 'text_cache/clip_text.pt',
             OUTPUT_DIR / 'caption_quality_v2.json', OUTPUT_DIR / 'caption_review_v2.csv'):
    display(FileLink(str(path)))
if CAPTIONS_ONLY:
    print('Caption-only phase complete. Review caption_review_v2.csv against images. Set CAPTIONS_ONLY=False and rerun to train a NEW M-TACS on this policy/cache.')
'''


def build_notebook():
    notebook = copy.deepcopy(baseline.build_notebook("mtacs"))
    by_section = {c["metadata"]["section"]: c for c in notebook["cells"]}
    def put(section, text):
        target = by_section[section]
        target.update(baseline.cell(target["cell_type"], text, section))
    def get(section):
        return ''.join(by_section[section]["source"])
    put("overview", '''# ISIC M-TACS · Florence attribute captions v2

Thí nghiệm mới để xử lý caption bị co về cùng phần mở đầu. Giữ ViT-L/16, ảnh 448, full_both, pool và split đã lưu.
**Không sửa CSV, cache, checkpoint hoặc bundle đang chạy web.** Không fine-tune Florence và không gọi caption này là ground truth.

V2 chọn cụm mô tả màu/hình dạng/biên/bề mặt 5–8 từ trong từng câu Florence, loại phần mở đầu chung chung.
Có thể sắp lại thuộc tính trước danh từ trong cùng câu; không mượn thuộc tính từ câu khác hay lấy nhãn/mask để viết caption.
Không tìm được cụm đủ thông tin thì mask text, không bịa câu để ép diversity. Từ chẩn đoán vẫn bị chặn theo policy cũ.

Mặc định **CAPTIONS_ONLY=True**: tạo CSV/cache/audit rồi bỏ qua toàn bộ model/training/test/export.
Sau khi xem ảnh và duyệt mẫu trong caption_review_v2.csv, đổi False và chạy lại để train mới, cùng 100 epoch mặc định của baseline.
Thay caption cần train mới; không nạp best.pt cũ vào run này để coi như đã học policy mới.

Attach: output/bundle ISIC M-TACS cũ (để giữ đúng split/pool và tái dùng câu dài đã xác minh hash), ISIC ảnh+mask,
DINOv3 source/ViT-L pretrained, Florence Microsoft, CLIP B/32 và offline wheels 4.49 như notebook trước.
Chỉ tái dùng câu dài từ cùng ảnh/model/split đã đăng ký; ảnh thiếu câu dài sẽ được Florence sinh thêm.
CLIP embeddings luôn tạo mới cho caption v2. Việc giảm trùng không chứng minh caption đúng hơn hoặc Dice tăng.
''')
    put("configuration_intro", '''## Cấu hình và Inputs

SAVED_SPLITS_SOURCE: folder run cũ hoặc bundle cũ; để rỗng sẽ dò đúng một run M-TACS ISIC trong Kaggle Inputs.
LONG_CAPTIONS_SOURCE: folder có captions.csv cùng protocol/status (bundle hoặc shared/caption_generation); để rỗng sẽ theo run tìm được.
Input cũ được đọc nguyên trạng. Output mới có tên riêng. CAPTIONS_ONLY=True giúp kiểm caption trước khi tốn thời gian train.
''')
    config = get("configuration")
    config = config.replace('"ISIC_MTACS_WEB_OUTPUT"', '"ISIC_MTACS_ATTRIBUTE_V2_OUTPUT"').replace(
        'outputs/isic2017_mtacs_vitl16_448_web_v1', 'outputs/isic2017_mtacs_attributes_v2_vitl16_448_web_v1')
    config = config.replace('RUN_TRAINING = True\nRUN_FINAL_TEST = True\nRUN_EXPORT = True',
        'CAPTIONS_ONLY = True\nRUN_TRAINING = not CAPTIONS_ONLY\nRUN_FINAL_TEST = not CAPTIONS_ONLY\nRUN_EXPORT = not CAPTIONS_ONLY')
    config += '''\nSAVED_SPLITS_SOURCE = os.getenv("ISIC_ATTRIBUTE_V2_SAVED_RUN", "")
LONG_CAPTIONS_SOURCE = os.getenv("ISIC_ATTRIBUTE_LONG_CAPTIONS_SOURCE", "")
'''
    put("configuration", config)
    runtime = get("runtime")
    tree = ast.parse(runtime)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'EMBEDDED_SOURCES' for x in n.targets))
    old = ast.get_source_segment(runtime, assignment)
    runtime = runtime.replace(old, 'EMBEDDED_SOURCES = ' + repr(sources()), 1)
    put("runtime", runtime)
    helper = '''\nimport importlib.util
snapshot_file = OUTPUT_DIR / 'snapshot_helpers.py'
snapshot_file.parent.mkdir(parents=True, exist_ok=True)
snapshot_file.write_text(''' + repr(frame_helper_source()) + ''', encoding='utf-8')
snapshot_spec = importlib.util.spec_from_file_location('isic_v2_snapshot_helpers', snapshot_file)
snapshot_io = importlib.util.module_from_spec(snapshot_spec)
snapshot_spec.loader.exec_module(snapshot_io)
'''
    put("dataset_intro", "## Giữ đúng split và pool baseline\n\nĐọc lại split/pool đã lưu, kiểm hash ảnh/mask và thứ tự ID. Không random chia lại dữ liệu hoặc lấy pool mới.")
    put("dataset", 'DINOV3_REPO_DIR, DINOV3_WEIGHTS = workflow.resolve_dino_assets(DINOV3_REPO_DIR, DINOV3_WEIGHTS, CONFIG)\n'
        + helper + DATASET_RECOVERY + '''
BACKBONE_SHA = model_code.io_utils.file_sha256(DINOV3_WEIGHTS)
ASSETS = {"backbone_sha256": BACKBONE_SHA, "dino_source": workflow.source_identity(DINOV3_REPO_DIR)}
if any(ASSETS[k] != OLD_BINDING['assets'][k] for k in ASSETS):
    raise ValueError('DINO source/weights differ from the original comparison run')
''')
    put("captions_intro", '''## Tạo caption v2 và CLIP cache mới

Tái xử lý source_long_caption được đăng ký từ run cũ để tiết kiệm inference; không thay câu gốc hay tạo nhãn bằng tên ảnh.
Những ảnh thiếu nguồn hợp lệ mới chạy Florence. Vẫn dùng FP32, task prompt và decoding frozen như baseline.
Phương pháp thay đổi **cách chọn nội dung câu ngắn**, không huấn luyện lại captioner.
Child Transformers 4.49 chạy offline bằng wheels; kernel giữ nguyên phiên bản.
''')
    index = notebook["cells"].index(by_section["captions"]) + 1
    notebook["cells"].insert(index, baseline.cell("code", QUALITY_CELL, "caption_quality_v2"))
    for section in ("model", "bank", "training", "evaluation", "export"):
        source = get(section)
        put(section, 'if not CAPTIONS_ONLY:\n' + ''.join('    ' + line if line.strip() else line for line in source.splitlines(keepends=True))
            + '\nelse:\n    print("Skipped ' + section + ': CAPTIONS_ONLY=True")\n')
    notebook["metadata"]["isic_caption_v2"] = {"policy": POLICY, "method": METHOD, "source": "frozen_florence_long_responses",
        "default_caption_only": True, "new_training_required": True, "clinical_accuracy_not_validated": True}
    return notebook


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + '\n', encoding="utf-8")
    print(TARGET)
