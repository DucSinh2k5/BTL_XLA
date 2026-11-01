# DTD: Review Study02/03 Va Thiet Ke Study04

Ngay doi chieu: 2026-10-26. Bao cao doc output da luu, khong chay lai training, khong thuc thi cell notebook hay nap pickle checkpoint.

## 1. Du Lieu Da Doc

- [Study02 da train](../../../stydu2_da_train.ipynb).
- [Study03 da train](../../../study3_datrain.ipynb).
- [Output study02](../../../dtd_study02_study03_results_20261026_050338/study02_norm_fusion/architecture_02_prefix_candidate_only_norm_fusion_results.csv).
- [Output study03](../../../dtd_study03_results_20261026_050358/study03_mltm_alignment/architecture_03_prefix_candidate_only_mltm_alignment_results.csv).
- [Bang doi chieu metrics](study02_03_results.csv), [interventions](study02_03_validation_interventions.csv), [bang chung JSON](study02_03_output_evidence.json).
- [Audit baseline01 truoc do](BASELINE_01_FLORENCE_PREFIX_CANDIDATE_ONLY.md).

Hai notebook da luu co them cell dong goi ZIP/blank. Cac function/class cua 26 cell thi nghiem goc khop AST voi template tuong ung. Khac biet source ngoai function chu yeu la duong dan Kaggle. Khong thay error output hoac stderr traceback trong hai notebook nay.

Moi goi ZIP chi chua mot study, vi study con lai nam o phien Kaggle khac. Canh bao thieu thu muc study kia trong missing_files_summary.txt khong co nghia ket qua cua study da chay bi thieu.

Audit kiem tra du 50 dong history, epoch 1..50, best epoch 38 theo minimum validation loss, 1.880 prediction dung tap ID cho moi split, va tinh lai accuracy/macro-F1 tu prediction. Metrics tai tinh khop JSON/CSV; nhan query va caption candidate duoc doi chieu theo ID, khong dua nhan vao selector.

## 2. Bang Ket Qua Chinh

Tat ca dung DTD official split1, seed42, ViT-L/16, 384, K=1, 376 candidate anh train va cung Florence-Prefix CSV. Baseline01 chi co do chinh xac so thap phan tu output notebook da audit truoc do.

| Study | Thay doi | Val accuracy | Val macro-F1 | Val CE | Test accuracy | Test macro-F1 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | Candidate-only Prefix, projection text chua L2 | 0.841489 | 0.840162 | 1.178962 | 0.854255 | **0.853032** | 38 |
| 02 | L2 candidate text truoc gate/fusion | 0.837234 | 0.835824 | 1.178273 | 0.850532 | 0.849329 | 38 |
| 03 | Study02 + own-image/caption m-LTM-inspired alignment | 0.836170 | 0.834947 | 1.178478 | 0.849468 | 0.848722 | 38 |

- Study02 thap hon 01 khoang 0.3703 diem phan tram test F1.
- Study03 thap hon 02 khoang 0.0607 diem phan tram test F1, thap hon 01 khoang 0.4310 diem phan tram.
- Test accuracy study03 thap hon 02 dung 2 anh trong 1.880 anh. Co 22 prediction thay doi, 6 anh sua dung va 8 anh tu dung thanh sai. Context thay doi o 70 anh test; val thay doi o 73 anh.
- Paired exact McNemar p=0.790527 cho correctness 02 vs 03, ca val/test. Day khong phai kiem dinh macro-F1 hay do bat dinh giua training seeds. Khong co co so khang dinh alignment thuc su lam tot/xau hon voi chenh lech nay.

**Ket luan:** hai huong moi chua chung minh cai thien phan loai. Khong nen goi study03 la model tot nhat. Val loss/CE va F1 la hai tieu chi khac nhau: checkpoint van duoc chon bang CE nhu protocol cu, khong doi sang epoch co F1 cao nhat sau khi doc test.

## 3. Caption Thuc Su Dang Giup Gi?

Cac intervention dung cung checkpoint, cung query va GIU NGUYEN anh context da chon; chi thay text trong fusion. Day khong phai train lai visual-only.

| Study | Aligned val F1 | Text off | Shuffled text | Off - aligned | Shuffle - aligned |
| --- | ---: | ---: | ---: | ---: | ---: |
| 02 | 0.835824 | 0.839112 | 0.839715 | +0.003287 | +0.003890 |
| 03 | 0.834947 | 0.835391 | 0.837773 | +0.000444 | +0.002826 |

- Tat/dao text khong lam F1 giam trong hai run; caption dung chua the hien loi ich ve hard-label prediction tren validation nay.
- Tuy nhien CE aligned thap hon off/shuffled. Caption co anh huong den phan bo xac suat, khong phai nhanh text hoan toan khong hoat dong.
- So prediction doi khi off/shuffle: study02 deu 23/1880; study03 la 28/1880 va 32/1880.
- Khoang bootstrap 95% delta F1 study02 shuffle: [0.000093, 0.008197], off: [-0.001235, 0.007371]. Study03 off/shuffle deu chua 0. Day la bootstrap query voi mot shuffle/seed train, khong du de suy rong ve tat ca run.
- Ca hai q_gate deu 0 dung thiet ke. c_gate val 0.061562 / 0.062848; voi caption dang hoat dong, mean gate khoang 0.075 / 0.076. Norm visual va text fusion gan 1; projection text truoc L2 gan 32.
- Khong so truc tiep gate ~0.06 cua study02/03 voi gate ~0.001 cua study01 de ket luan caption manh hon: thang do vector da thay doi. Gate khong phai confidence hay phan tram dong gop vao prediction.
- Khong co intervention nao chung minh khong leak nhan; policy lexical va caption da sinh van co sai/noise. Khong can chinh nguong CLIP theo test F1.

## 4. Alignment Co Hoc Khong?

**Co hoc muc tieu phu, nhung chua chuyen thanh loi ich phan loai.**

- Train auxiliary KL giam tu khoang 3.41 xuong 0.859 o epoch50.
- Positive transport mass minibatch tang tu khoang 0.034 len 0.447.
- Tai best epoch38, danh gia 324 own-image/caption pair train pool, 289 caption canonical khac nhau: caption-group top1=0.746914; exact-source top1=0.672840; mean pair cosine=0.429815.
- Metric duong, condition ratio=1.331802; Sinkhorn marginal error khoang 8.3e-7, khong co dau hieu bat on so tu output nay.
- Day la **training-pool fit**, khong phai held-out retrieval accuracy va khong phai chat luong ground-truth caption. Caption lap khien exact-source retrieval mo ho.
- CLS anchor detached, backbone van hoc tu task loss, projector text hoc ca task loss va alignment. Khong phai dang fine-tune Florence.

## 5. Gioi Han So Sanh

Split IDs, candidate order, caption CSV bytes, policy, augmentations, batch32, workers8, budget50, LR/loss/hyperparameters khop giua 02/03. Dung cung GPU RTX PRO 6000. Hai best.pt deu ton tai, ~1.268 GB moi file; audit khong nap chung.

Nhung runtime khac:

| Thu vien | Study02 | Study03 |
| --- | --- | --- |
| PyTorch | 2.10.0+cu128 | 2.11.0+cu128 |
| NumPy | 2.0.2 | 2.1.3 |
| pandas | 2.3.3 | 2.3.3 |

DINO weights cung ten file va dung luong, nhung khac attachment path/mtime; khong co SHA256 toan bo weights trong protocol de chung minh bytes dong nhat. CLIP cache fingerprint khac, co the do path/metadata, khong tu no chung minh embeddings khac hay giong. Source DINO tu hai attachment khac cung chua co hash doi chieu. Mot split/mot seed va runtime khac gioi han ket luan nhan qua voi effect rat nho.

## 6. Quyet Dinh Cho Notebook04

[Notebook04](../../../notebooks/02_improvements/multimodal_architecture_ablation/04_candidate_only_semantic_selector/dtd/mtacs-dtd-vitl16-prefix-candidate-only-semantic-selector.ipynb) giu study03 lam **doi chung cau truc**, khong chon no vi test cao hon. Cau hoi moi la: projector/metric da hoc own-pair co huu ich khi dung vao CONTEXT RETRIEVAL khong?

- Khong dung caption query. Selector dung anh query, anh candidate va caption candidate hop le tu pool train.
- Nhanh visual giu original z-score. Nhanh semantic la am khoang cach Mahalanobis giua query DINO CLS va candidate CLIP caption qua projector alignment, masked z-score tren valid nonself candidate, clip [-3,3].
- Them residual alpha <=0.25. Dung existing semantic_mix_logit, khong tang trainable parameter count so voi study03. Warmup image-only 5 epoch, ramp 5 epoch; alpha bat dau khoang 0.125 sau ramp neu logit chua thay doi.
- Epoch ramp duoc luu trong checkpoint. Alpha khac c_gate: alpha chon context; c_gate tron caption vao representation sau khi chon.
- Invalid/constant/khong du caption: semantic residual=0, visual fallback. Invalid caption khong loai anh candidate. Exact-self exclusion ap dung truoc semantic statistics va truoc selection.
- Giua query batch khong chay OT, tranh prediction phu thuoc query khac cung batch. Day la **adaptation moi**, khong phai tai hien nguyen he retrieval cua [m-LTM](https://arxiv.org/abs/2405.10084).
- Giu loss alignment 0.05, fusion, Prefix CSV/policy, split/pool/seed, K1, 50 epoch va checkpoint CE. Train doc lap tu pretrained, khong can nap 02/03 best.pt.

Them hai nhom validation diagnostics tach rieng:

| Bang | Can thiep | Anh context |
| --- | --- | --- |
| Fusion | Aligned / off / shuffled fusion caption | Co dinh cung context |
| Selector | Aligned / visual-selector-only / shuffled caption CHI trong selector | Duoc chon lai; fusion lay caption GOC dung anh moi chon |

So sanh co/khong selector tren cung checkpoint la dependence test, khong tuong duong train lai. Muon retrained control, dat SELECTOR_SEMANTIC_MAX_WEIGHT=0 vao thu muc output moi; test CPU kiem tra duong zero-weight khop study03.

## 7. Cach Chay Va Tieu Chi Doc

Upload notebook04 va 5 input cu: DTD, DINO code, DINO ViT-L16 weights, full CLIP ViT-B/32, exact captions_prefix_control.csv. Chinh duong dan trong config/settings neu can. Khong can Florence/Qwen, references.csv hay checkpoint cu; notebook khong sinh lai caption.

Run All. Moi training run doc lap 50 epoch, output rieng. Dung validation de danh gia nhanh selector; RUN_FINAL_TEST=False neu con thu hyperparameter. Chi ket luan them selector co ich khi validation co tin hieu nhat quan, cap nhat metric tu output thuc va sau do kiem chung nhieu seed. Khong tu dien metric moi vao bang.

De tai tao audit: python tools/project_management/audit_dtd_architecture_study02_03.py trong thu muc TACS. Script chi ghi bao cao JSON/CSV moi, khong doi ten/di chuyen notebook hay sua output nguon.
