# Retrieval-augmented image classification that uses neighbour labels (novelty background for M-TACS)

Student variant used for comparison (M-TACS): TACS picks **one** context image (K=1) per query from a train-set pool with a **learned selector trained by REINFORCE** (reward = CE loss without context − CE loss with context). The selected candidate's **class label is turned into text** ("a photo of a <class> texture" + caption), encoded by **CLIP text encoder**, and fused into the query image feature through a **learned gate**.

Axes used throughout: (R) how neighbours are retrieved: fixed kNN on frozen features vs learned; (L) how neighbour labels enter: voting/one-hot, label embedding, label text, attention/memory; (D) datasets.

---

## Q1. RAC (Long et al., CVPR 2022) and follow-ups

### Takeaway
RAC is the closest single prior work on the "label of retrieved neighbour as text" axis: it retrieves k=30 training-set neighbours with a frozen ViT + FAISS, encodes their **labels as text** with a CLIP-pretrained text encoder, and adds the resulting logits to a base classifier. It differs from M-TACS in that retrieval is fixed (non-learned), K is large, and fusion is logit addition rather than a gated feature fusion. Follow-ups (Iscen et al. CVPR 2023, RECO ICLR 2024) keep retrieval fixed and add learned attention over retrieved items.

### Cited Findings
- **RAC** — Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, Anton van den Hengel. "Retrieval Augmented Classification for Long-Tail Visual Recognition", CVPR 2022, pp. 6959-6969; arXiv 2202.11233 (Feb 2022). — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html); [arXiv](https://arxiv.org/abs/2202.11233)
  - (R) Keys from a **frozen** pretrained ViT-B/16; FAISS HNSW approximate kNN, cosine similarity; k=30 in experiments; the default index is the **training set itself**; the top-1 hit is dropped in training since it is often the query itself. — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2202.11233)
  - (L) Each stored image has associated text (labels / descriptions / captions); experiments use **labels as text**. Retrieved text is encoded by a CLIP-pretrained BERT-like text encoder (~63M params, 76 tokens), then a linear layer gives retrieval logits. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.11233)
  - Fusion: base-branch logits (ViT-B/16 + linear) and retrieval logits are L2-normalised and **summed** (no attention / no learned gate); trained jointly with logit-adjusted CE + label smoothing. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.11233)
  - (D) Places365-LT 47.17% top-1 (vs 41.2 PaCo; base-only 44.05); iNaturalist-2018 80.24% @224 and 85.56% @384. Retrieval branch helps mainly tail classes. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.11233)
- **Iscen, Fathi, Schmid — "Improving Image Recognition by Retrieving from Web-Scale Image-Text Data"**, CVPR 2023, arXiv 2304.05173. Retrieves from a 1B image-text memory; key contribution is a **memory attention module** that learns how much each retrieved example matters (filtering irrelevant retrievals). Evaluated on long-tail (ImageNet-LT, Places-LT), noisy labels (WebVision), fine-grained. — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Iscen_Improving_Image_Recognition_by_Retrieving_From_Web-Scale_Image-Text_Data_CVPR_2023_paper.html); [PwC](https://PapersWithCode.com/paper/improving-image-recognition-by-retrieving)
- **RECO — Iscen, Caron, Fathi, Schmid, "Retrieval-Enhanced Contrastive Vision-Text Models"**, ICLR 2024, arXiv 2306.07196. Frozen CLIP + light single-layer fusion transformer refines embeddings with **cross-modal** retrieved info (uni-modal search, e.g. image→image, then use the paired text). +10.9 Stanford Cars, +10.2 CUB, +7.3 OVEN. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/045c87def0c02e3ad0d3d849766d7f1e-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2306.07196)

### Inferences
- RAC already establishes "kNN neighbours' class labels → text encoder → fused with a base classifier, using the training set as memory". M-TACS cannot claim novelty for "using neighbour label text"; its differentiators are (a) the learned, reward-trained selector, (b) K=1, (c) gated feature-level fusion with CLIP text of a templated label + caption.
- Iscen 2023 and RECO show that *learned weighting of fixed-retrieved neighbours* (attention) is standard; a reviewer will likely ask how a learned selector over a pool compares to "fixed kNN top-K + learned attention" (soft selection).

### Gaps
- RAC paper does not clearly state how the k text embeddings are aggregated into one logit vector (per ar5iv extraction).
- I did not check Iscen 2023 / RECO for whether neighbour labels (vs captions) are used; RECO uses paired text of retrieved items (captions/alt-text), not class labels, per its abstract.

---

## Q2. kNN / non-parametric memory combined with parametric classifiers

### Takeaway
kNN over frozen features (with label voting) is a standard classifier / evaluation protocol and is often interpolated with a linear head. Retrieval in all these works is fixed nearest-neighbour; labels enter through voting or interpolated label distributions, never through a learned selector.

### Cited Findings
- **Nakata et al. (Kioxia), "Revisiting a kNN-based Image Classification System with High-capacity Storage"**, ECCV 2022 (oral), arXiv 2204.01186. Stores features, labels and images in external storage; kNN classifier over frozen pretrained features; knowledge updated by editing the database, not fine-tuning. 79.8% ImageNet top-1 without fine-tuning after pretraining; 90.8% Split CIFAR-100 (task-incremental). Neighbours used for interpretability and fixing stored knowledge. — [arXiv](https://arxiv.org/abs/2204.01186v2); [ECCV](https://ecva.net/papers/eccv_2022/papers_ECCV/html/1552_ECCV_2022_paper.php)
- **Jia et al., "Rethinking Nearest Neighbors for Visual Classification"**, arXiv 2112.08459 (Dec 2021). kNN on pretrained features integrated with a standard classifier: kNN predictions flag easy/hard examples in training, and at test time kNN and classifier distributions are **linearly interpolated** with λ. kNN helps most in low-data settings. — [arXiv](https://arxiv.org/abs/2112.08459v2)
- **Li et al., "Memory-Based Neighbourhood Embedding for Visual Recognition"**, ICCV 2019. Episodic memory stores CNN features **and labels**, updated during training; neighbours retrieved per image refine the embedding end-to-end (retrieval itself is fixed NN). — [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Memory-Based_Neighbourhood_Embedding_for_Visual_Recognition_ICCV_2019_paper.pdf)
- DINO (Caron et al., ICCV 2021) popularised weighted kNN on frozen self-supervised features as an evaluation protocol (later reused by DINOv2). — [arXiv 2104.14294](https://arxiv.org/abs/2104.14294) (not re-fetched in this session; known paper)
- Memorizing Transformers (Wu et al., ICLR 2022) add a kNN-retrieved key/value memory to attention, but for language modelling, not image classification. — [arXiv 2203.08913](https://arxiv.org/abs/2203.08913) (not re-fetched)

### Inferences
- These are the natural "non-learned retrieval" baselines for M-TACS: kNN-vote, kNN+linear interpolation (Jia), and RAC-style label-text retrieval. A fair novelty claim requires showing a reward-trained selector beats top-1 kNN with the same label-text fusion.

### Gaps
- No exact DINO/DINOv2 kNN hyperparameters verified here (k=20 is commonly cited but not re-checked).

---

## Q3. CLIP cache / adapter methods with labelled support features

### Takeaway
Cache methods store labelled support features as key-value pairs; labels enter as **one-hot value vectors** weighted by query–key affinity (soft kNN over *all* cached shots), combined with CLIP zero-shot text logits. Retrieval is fixed similarity; no per-query selection of one sample, no RL, and label *text* enters only through CLIP's zero-shot classifier, not as a neighbour-conditioned feature.

### Cited Findings
- **Tip-Adapter — Renrui Zhang et al.**, ECCV 2022, arXiv 2207.09519. Non-parametric key-value cache from few-shot set: keys = CLIP visual features, values = **one-hot labels**; affinity-weighted label sum added to zero-shot CLIP logits; training-free. Tip-Adapter-F makes keys learnable. 11 datasets. — [ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/154_ECCV_2022_paper.php); [arXiv](https://arxiv.org/pdf/2207.09519)
- **SuS-X (with TIP-X) — Udandarao, Gupta, Albanie**, ICCV 2023, arXiv 2211.16198. Name-only transfer: builds a support set (SuS) from Stable Diffusion generations or LAION retrieval using only class names, then TIP-X cache inference; SoTA zero-shot on 19 datasets. — [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Udandarao_SuS-X_Training-Free_Name-Only_Transfer_of_Vision-Language_Models_ICCV_2023_paper.html)
- **APE — Zhu, Zhang et al., "Not All Features Matter: Enhancing Few-shot CLIP with Adaptive Prior Refinement"**, ICCV 2023, arXiv 2304.01195. Refines which feature channels of the CLIP cache model matter for the domain; training-free version +1.59% avg 16-shot over second-best on 11 benchmarks; treats Tip-X as a key-value cache method. — [arXiv](https://arxiv.org/pdf/2304.01195); [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Not_All_Features_Matter_Enhancing_Few-shot_CLIP_with_Adaptive_Prior_ICCV_2023_paper.pdf)
- **CaFo — Renrui Zhang et al., "Prompt, Generate, Then Cache"**, CVPR 2023, arXiv 2303.02151. GPT-3 prompts + DALL-E synthetic images + **learnable cache model** blending CLIP and DINO predictions for few-shot classification. — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.html)

### Inferences
- In cache methods, the "neighbour label" is a one-hot vector aggregated over many shots; M-TACS's "label → templated text → CLIP text embedding → gated into image feature" is a different label channel. Still, Tip-Adapter-style cache with K=1 nearest key ≈ "copy neighbour label", a trivial baseline M-TACS should beat.
- A reviewer may point out that the CLIP text embedding of the neighbour's class is exactly the CLIP zero-shot classifier weight of that class; gating it into the image feature risks being near-equivalent to a label leak / prior toward the neighbour's class. Ablation vs "add neighbour's one-hot to logits" is worth having.

### Gaps
- Tip-X's precise affinity formulation (KL over image–text affinity signatures) not re-verified in this session.

---

## Q4. Visual in-context classification with labelled examples

### Takeaway
Visual ICL works feed labelled support examples (labels as learned/fixed embeddings) to a sequence model; selection of examples is either all-support, fixed kNN, or (for dense/regression tasks) a learned selector. CAML is the closest "label-conditioned context for classification" model, but uses a fixed support set rather than a selected neighbour.

### Cited Findings
- **CAML — Fifty et al., "Context-Aware Meta-Learning"**, ICLR 2024, arXiv 2310.10971. Frozen pretrained (e.g. CLIP) image encoder; labelled support + query fed to a non-causal sequence model in one forward pass; labels encoded with a fixed **ELMES** (equiangular) label encoding instead of one-hot; "universal meta-learning" (no fine-tuning on benchmark). — [arXiv](https://arxiv.org/pdf/2310.10971); [code](https://github.com/cfifty/CAML)
- **Hummingbird — Balazevic, Steiner, Parthasarathy, Arandjelović, Hénaff, "Towards In-context Scene Understanding"**, NeurIPS 2023, arXiv 2306.01667. Nearest-neighbour retrieval over a prompt of annotated image (patch) features acts as in-context learner for segmentation/depth; no task-specific params. Retrieval = fixed NN. — [NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/hash/c94a632545000531f0b47000e9caa5b6-Abstract-Conference.html)
- **Zhang, Zhou, Liu, "What Makes Good Examples for Visual In-Context Learning?"**, NeurIPS 2023, arXiv 2301.13670. Prompt choice changes visual-ICL results by >70% mIoU. Proposes unsupervised (off-the-shelf kNN) and **supervised prompt retrieval (SupPR)** that trains a retriever with contrastive learning, where positives/negatives are candidate examples ranked by the actual in-context performance. Tasks: foreground segmentation, single object detection, colorization — **not classification**. — [arXiv](https://arxiv.org/pdf/2301.13670); [NeurIPS](https://neurips.cc/virtual/2023/poster/70423)

### Inferences
- SupPR is the closest *concept* to TACS's selector: a retriever trained from downstream-performance signal on which context example helps. Difference: contrastive supervision from ranked candidates (not REINFORCE with a loss-difference reward), frozen inpainting ICL model, dense tasks, and the "label" is the paired output image rather than class label text.

### Gaps
- Did not find a visual-ICL classification paper that learns example selection and passes neighbour class labels as text.

---

## Q5. Retrieval-augmented recognition with text (REACT, RA-CLIP, K-LITE, ...)

### Takeaway
These methods inject retrieved **captions or external knowledge text** (not neighbour class labels) into CLIP-style models, mostly for zero-shot transfer; retrieval is fixed (CLIP similarity or dictionary lookup).

### Cited Findings
- **RA-CLIP — Xie, Sun, Xiong, Zheng, Zhao, Zhou**, CVPR 2023, pp. 19265-19274. Holds out part of image-text data as a reference set; retrieves related image-text pairs per input to enrich the image representation ("open-book exam"); +12.7% zero-shot classification over CLIP across 10 datasets, +2.8% zero-shot ROI classification. — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_RA-CLIP_Retrieval_Augmented_Contrastive_Language-Image_Pre-Training_CVPR_2023_paper.html)
- **REACT — Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, Chunyuan Li, "Learning Customized Visual Models with Retrieval-Augmented Knowledge"**, CVPR 2023. Retrieves relevant web image-text pairs (~3% of CLIP pretraining data) for a target domain and trains new modular blocks with frozen original weights; +5.4% ImageNet zero-shot, +3.7% ELEVATER. Retrieval used to build training data, not per-query context at inference. — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.html)
- **K-LITE — Shen et al.**, NeurIPS 2022, arXiv 2204.09222. Enriches class-name text with WordNet / Wiktionary definitions in training and at evaluation; 20 classification + 13 detection datasets. Knowledge is retrieved per *class name*, not per image neighbour. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/63fef0802863f47775c3563e18cbba17-Abstract.html)
- RECO (see Q1) — cross-modal fusion of retrieved text into CLIP embeddings with a frozen backbone. — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/045c87def0c02e3ad0d3d849766d7f1e-Abstract-Conference.html)

### Inferences
- M-TACS's "label template + caption of neighbour → CLIP text → gated fusion" sits between RAC (label text of neighbours) and RA-CLIP/RECO (caption text of neighbours, fused into image embedding). The combination of both kinds of text is a minor variation, not a new paradigm.

### Gaps
- Did not verify RA-CLIP's exact fusion module (attention-based, per my recollection; not confirmed here).

---

## Q6. Does any work LEARN the retriever with a downstream task reward (RL) while feeding neighbour labels for classification?

### Takeaway
I found **no** paper doing exactly this for **image classification**. The nearest neighbours of the idea: RL-trained demonstration selection for visual regression with MLLMs (LSD, CVPR 2026 Findings), RL retrievers for text ICL (RetICL 2023), performance-supervised visual prompt retrieval (SupPR, NeurIPS 2023, dense tasks), and a score-function-trained retriever over labelled history for non-stationary (non-image) classification (2026 workshop). Novelty of M-TACS therefore rests on the combination "REINFORCE-learned single-neighbour selector + neighbour label text fusion for image classification", not on any single component.

### Cited Findings
- **LSD — Eugene Lee, Yu-Chi Lin, Jiajie Diao, "Learning to Select Visual In-Context Demonstrations"**, CVPR 2026 Findings, arXiv 2603.26775. RL agent (Dueling DQN with query-centric Transformer decoder) builds demonstration sets for an MLLM, replacing unsupervised kNN; reward from prediction accuracy (e.g. MAE); five **visual regression** benchmarks; authors find kNN remains strong for subjective tasks, learned policy needed for objective regression. Classification not evaluated. — [arXiv](https://arxiv.org/abs/2603.26775); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Lee_Learning_to_Select_Visual_In-Context_Demonstrations_CVPRF_2026_paper.pdf)
- **RetICL — Scarlatos & Lan, "Sequential Retrieval of In-Context Examples with Reinforcement Learning"**, arXiv 2305.14502 (2023). Example selection as MDP; retriever trained with RL using task reward + confidence reward; math word problems and science QA (text only). — [arXiv](https://arxiv.org/html/2305.14502v2)
- **LLM-R — "Learning to Retrieve In-Context Examples for Large Language Models"** (Wang et al., Microsoft, arXiv 2307.07164). Reward model from LLM feedback, distilled into a dense bi-encoder retriever; notably effective on classification tasks with many training examples (text). — [arXiv-vanity mirror](https://www.arxiv-vanity.com/papers/2307.07164)
- **Gammell et al., "Learning to Query History: Nonstationary Classification via Learned Retrieval"**, arXiv 2604.07027, ICLR 2026 TSALM workshop. Learned discrete retrieval of historical (input, label) examples trained end-to-end with a **score-based gradient estimator**; classifier conditions on retrieved labelled history; synthetic + Amazon Reviews '23 — no images. — [arXiv](https://arxiv.org/abs/2604.07027)
- **SupPR** (Zhang, Zhou, Liu, NeurIPS 2023) — learned visual prompt retriever supervised by in-context performance, contrastive not RL, dense tasks. — [arXiv](https://arxiv.org/pdf/2301.13670)

### Inferences
- Closest conceptual ancestor for the *selector*: LSD (RL, visual, but regression + MLLM) and SupPR (visual, performance-supervised, dense tasks). Closest ancestor for the *label-text fusion*: RAC. M-TACS = RAC-style label-text conditioning with K=1 + a SupPR/LSD/RetICL-style task-reward-trained selector, applied to texture/image classification with a small trainable classifier. A defensible novelty claim is the combination plus the specific reward (loss reduction from context), not either half alone.
- Risk to flag: with the train set as the pool and the neighbour's label fed as text, the selector can learn to pick same-class images and the gate can learn to copy the label (RAC handles the analogous self-retrieval by dropping the top-1 hit). Leakage controls (exclude the query itself, same-image augmentations) and a "kNN top-1 + same fusion" baseline are needed to show the learned selector adds anything.
- Reward = loss_without − loss_with is a baseline-subtracted REINFORCE signal similar in spirit to LSD/RetICL accuracy rewards; I found no vision-classification paper using exactly this loss-difference reward.

### Gaps
- Search coverage is ~15 queries; a paper combining learned retrieval + neighbour labels for image classification may exist under different terms (e.g. "differentiable retrieval", "Gumbel top-k retrieval", "learning to retrieve for few-shot classification"). Not exhaustively checked.
- LSD's exact reward formula and whether demonstrations include target values were not stated on the arXiv abstract page.
