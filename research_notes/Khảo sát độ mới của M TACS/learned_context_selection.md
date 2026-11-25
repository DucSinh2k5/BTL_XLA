# Learned selection of context / reference / exemplar images for vision models (prior work for TACS / M-TACS)

Scope: 2019–2026 work on picking a context image (exemplar, support, prompt, reference) for a query. For each work we note what is selected, whether selection is fixed similarity or learned, the training signal, the task, and how it relates to TACS (one learned context image, Gumbel-ST + REINFORCE, reward = loss drop) and to the student extension (adds the candidate's VLM caption and class label text, CLIP-encoded with gated fusion).

Verification legend: entries marked **[verified]** were checked this session on a primary or near-primary page (arXiv, CVF, NeurIPS, ICLR, ACL Anthology). Entries marked **[prior knowledge]** have well-known arXiv IDs but their pages were not opened this session. The report writer should spot-check those.

---

## Q1. Which works do the TACS authors cite as closest?

### Takeaway
TACS frames itself against (a) NLP retrieval-augmented models with trainable retrievers, (b) vision RAG that uses *static similarity* retrieval, and (c) recent RL- or reward-trained RAG for generation. It cites **no visual in-context-learning prompt-selection work** (SupPR, Partial2Global, SCS, prompt-SelF) and no explicit ICL papers. That is a notable gap in its related work, and it matters when judging novelty.

### Cited Findings
- **TACS method (as described in the paper):** the selector is a ViT-S/16. A candidate's utility is the dot product of its embedding with the query's, over a pool of 20% of the train set, with argmax at inference. Training uses straight-through Gumbel-Softmax (τ=0.1) on the task cross-entropy, plus a policy-gradient term whose reward is CE(no context) − CE(with context), with advantages standardized per batch. The total loss is L_grad + 0.5·L_policy. Evaluation covers 18 datasets (fine-grained, medical classification, medical segmentation). — [TACS arXiv 2512.00489](https://arxiv.org/html/2512.00489)
- **Closest works cited in TACS, with the stated contrast:**
  - Long et al., *Retrieval Augmented Classification for Long-Tail Visual Recognition* (RAC), CVPR 2022. TACS calls its retrieval "static, precomputed similarity." — [TACS](https://arxiv.org/html/2512.00489); [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html)
  - Liu et al., *REACT: Learning customized visual models with retrieval-augmented knowledge*, CVPR 2023. Retrieval is static (CLIP similarity) and is used only to gather extra training data, not at inference. — [TACS](https://arxiv.org/html/2512.00489)
  - Liu et al., *SWAT: Few-shot recognition via stage-wise retrieval-augmented finetuning*, CVPR 2025. Same contrast as REACT. — [TACS](https://arxiv.org/html/2512.00489)
  - Zhao et al., *Retrieval-augmented few-shot medical image segmentation with foundation models*, IEEE TNNLS 2025. Uses frozen DINO similarity, which TACS replaces with learned selection. — [TACS](https://arxiv.org/html/2512.00489)
  - Li et al., *RAU: Reference-based anatomical understanding with VLMs*, arXiv 2509.22404, 2025. Uses frozen embeddings and assumes "similarity implies usefulness." — [TACS](https://arxiv.org/html/2512.00489)
  - Hou, Gould, Zheng, *Learning to Select Views for Efficient Multi-View Understanding*, CVPR 2024. An RL view selector, cited in the Introduction. — [TACS](https://arxiv.org/html/2512.00489)
  - Alzahrani et al., selective multi-view, CVPR 2024, cited in the Introduction. — [TACS](https://arxiv.org/html/2512.00489)
  - NLP trainable-retriever lineage: REALM (Guu et al., ICML 2020), DPR (Karpukhin et al., EMNLP 2020), RAG (Lewis et al., NeurIPS 2020), EMDR² (Singh et al., NeurIPS 2021), reader-to-retriever distillation (Izacard & Grave, ICLR 2021), Atlas (Izacard et al., JMLR 2023). — [TACS](https://arxiv.org/html/2512.00489)
  - RL- or reward-trained RAG for generation: Self-RAG (2310.11511), RAG-DDR (2410.13509), SmartRAG (2410.18141), DRO (2505.03075). TACS's contrast is that these target generative text tasks, while TACS targets discriminative vision. — [TACS](https://arxiv.org/html/2512.00489)
  - Multimodal or vision RAG: REVEAL (Hu et al., CVPR 2023), retrieval-augmented diffusion (Blattmann et al., NeurIPS 2022), VisRAG (2410.10594), ColPali (2024). — [TACS](https://arxiv.org/html/2512.00489)
  - Gumbel-Softmax: Jang, Gu & Poole, ICLR 2017. No Williams-1992 REINFORCE reference appears. — [TACS](https://arxiv.org/html/2512.00489)

### Inferences
- TACS's novelty claim is narrow. It is "learned, task-aligned *single-image* context selection used *at inference* for *discriminative* vision (classification and segmentation), trained jointly with the task net." It is not "the first learned exemplar selection in vision."
- The visual-ICL prompt-selection literature (Q2) is the closest uncited prior art. A reviewer of TACS or of M-TACS would likely raise it.

### Gaps
- I did not read TACS's full experimental section, so I don't know whether any visual-ICL selector (e.g., SupPR) was used as a baseline.

---

## Q2. Which works learn to retrieve or select exemplar, support or reference images for a vision task?

### Takeaway
There is a well-developed line, mainly 2023–2025, of learned prompt selection for visual in-context learning (VICL). These methods use a *frozen* inpainting-style generalist (MAE-VQGAN, SegGPT, Painter) and train a selector from downstream-performance feedback: contrastive (SupPR), list-wise ranking (Partial2Global and RH-Partial2Global), or RL (SCS). In multimodal LLMs, learned demonstration selection exists too (Lever-LM, DRUM, LSD). TACS's main differences are that the task network is **trained jointly** with the selector, rather than being a frozen generalist, and that the reward is the **loss drop versus no context**.

### Cited Findings
**Visual ICL prompt selection (frozen generalist model, learned selector)**
- **Zhang, Zhou, Liu, *What Makes Good Examples for Visual In-Context Learning?*, NeurIPS 2023 (arXiv 2301.13670)** **[verified]**
  - **What is selected:** one in-context image-label pair per query. SupPR uses a learned retriever. UnsupPR uses fixed off-the-shelf cosine kNN.
  - **Training signal:** contrastive. Examples that give better in-context results are pulled toward the query and worse ones pushed away, so the supervision comes from downstream performance.
  - **Tasks and setting:** foreground segmentation, single-object detection, colorization. The large vision model is black-box.
  - **Key finding:** the gap between the best and worst prompt can exceed 70% mIoU.
  - **Relation to TACS:** conceptually the closest precursor ("learn which example helps, not which looks similar"). It differs in that the generalist model is frozen, training is offline contrastive rather than Gumbel/REINFORCE joint training, the tasks differ, and it uses no text.
  - Sources: [arXiv](https://arxiv.org/pdf/2301.13670); [NeurIPS](https://neurips.cc/virtual/2023/poster/70423)
- **Sun et al., *Exploring Effective Factors for Improving Visual In-Context Learning* (prompt-SelF), arXiv 2304.04748 (2023), IEEE TIP 2025** **[verified]**
  - **What is selected:** a prompt chosen by pixel-level retrieval (fixed similarity, not learned), followed by prompt fusion and ensembling.
  - **Tasks:** single-object segmentation and detection.
  - **Relation to TACS:** it shows that selection matters, but its selector is not learned.
  - Sources: [arXiv](https://arxiv.org/pdf/2304.04748); [IEEE](https://ieeexplore.ieee.org/document/10945944/references)
- **Xu, Liu, Wang, Yao, Fu, *Towards Global Optimal Visual In-Context Learning Prompt Selection* (Partial2Global), NeurIPS 2024 (arXiv 2405.15279)** **[verified]**
  - **How it selects:** a learned transformer list-wise ranker plus a consistency-aware aggregator, aiming for the globally best prompt per query.
  - **Training signal:** ranking supervision derived from measured ICL performance.
  - **Tasks:** foreground segmentation, detection, colorization.
  - Sources: [arXiv](https://arxiv.org/pdf/2405.15279); [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2024/hash/8900e600e4f55be0a10cee4a14d7c3de-Abstract-Conference.html)
- ***Towards Reliable and Holistic Visual In-Context Learning Prompt Selection* (RH-Partial2Global), NeurIPS 2025 (arXiv 2509.25989)** **[verified, snippet only]**
  - Improves on Partial2Global's randomly sampled pairwise comparisons, which leave incomplete coverage and redundant comparisons.
  - Sources: [arXiv](https://arxiv.org/html/2509.25989v2); [NeurIPS](https://neurips.cc/virtual/2025/poster/119157)
- **Suo et al., *Rethinking and Improving Visual Prompt Selection for In-Context Learning Segmentation Framework* (Stepwise Context Search, SCS), ECCV 2024 (arXiv 2407.10233)** **[verified abstract]**
  - **Finding:** prompt *diversity* matters, and plain similarity sorting is suboptimal.
  - **Method:** builds a compact candidate pool, then adaptively searches it.
  - **Training signal:** a third-party review says the search is RL with an IoU reward. This is not confirmed on the primary page and should be checked.
  - **Relation to TACS:** the pool-then-select design and the RL reward are close to TACS. SCS selects for a frozen segmentation generalist and does not train jointly.
  - Sources: [arXiv](https://arxiv.org/abs/2407.10233); [ECCV](https://eccv2024.ecva.net/virtual/2024/poster/785)
- **E-InMeMo (arXiv 2504.18158, 2025)** **[verified snippet]**
  - Notes that retrieved in-context pairs are not always optimal and refines the pair with a learnable pixel prompt. It improves the context rather than selecting it.
  - Source: [arXiv](https://arxiv.org/html/2504.18158v1)

**Generalist reference-based models whose selection is fixed (they consume the prompt but do not choose it)** **[prior knowledge]**
- Bar et al., *Visual Prompting via Image Inpainting* (MAE-VQGAN), NeurIPS 2022 — [arXiv 2209.00647](https://arxiv.org/abs/2209.00647)
- Wang et al., *Images Speak in Images: A Generalist Painter* (Painter), CVPR 2023 — [arXiv 2212.02499](https://arxiv.org/abs/2212.02499)
- Wang et al., *SegGPT: Segmenting Everything in Context*, ICCV 2023 — [arXiv 2304.03284](https://arxiv.org/abs/2304.03284)
- All three take the context pair from the user or by random or kNN choice. They are the backbones that SupPR, Partial2Global and SCS select for.

**Vision RAG with fixed similarity retrieval**
- **Long et al., RAC, CVPR 2022** **[verified]**
  - A retrieval branch queries an external memory of pre-encoded images *and associated text snippets*, with fixed kNN.
  - Reported gains are +14.5% on Places365-LT and +6.7% on iNaturalist-2018.
  - **Relation to M-TACS:** a precedent for feeding the *text attached to retrieved neighbours* into a classifier, though the retrieval is not learned.
  - Source: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html)
- **Iscen, Fathi, Schmid, arXiv 2304.05173 (2023)** **[verified snippet]**
  - Retrieves from web-scale image-text data (WebLI, LAION, YFCC100M) to improve image recognition. Retrieval is fixed and the text is part of the retrieved knowledge.
  - Source: [arXiv](https://arxiv.org/pdf/2304.05173.pdf)

**Learned demonstration selection for multimodal LLMs (image + text demos)**
- **Yang, Peng et al., *Lever LM*, NeurIPS 2024 (arXiv 2312.10104)** **[verified]**
  - A 67M-parameter LM configures *which* demos are used and in what order for a 9B LVLM.
  - **Training signal:** a dataset of effective sequences scored by the LVLM's own confidence.
  - **Tasks:** VQA and captioning.
  - **Relation to M-TACS:** the demos here are image + text (question/caption) pairs, so text is part of the selected context. The task model is frozen.
  - Sources: [arXiv](https://arxiv.org/pdf/2312.10104); [NeurIPS](https://neurips.cc/virtual/2024/poster/95045)
- **DRUM, ACL SRW 2025** **[snippet only]**
  - Fine-tunes a VLM embedding retriever to fetch demos suited to an LVLM.
  - Source: [ACL Anthology PDF](https://preview.aclanthology.org/landing_page/2025.acl-srw.83.pdf)
- **LSD, *Learning to Select Demonstrations*, arXiv 2603.26775 (2026)** **[snippet only]**
  - Replaces kNN with an RL agent that builds demo sets sequentially.
  - Source: [arXiv](https://arxiv.org/html/2603.26775v1)
- **RAICL, arXiv 2505.02087 (2025)** **[snippet only]**
  - Retrieval-augmented ICL for multimodal disease classification. It is a medical relative of TACS.
  - Source: [arXiv](https://arxiv.org/abs/2505.02087v1)

**NLP template for "learn the retriever from the scorer's feedback"**
- **Rubin, Herzig, Berant, *Learning To Retrieve Prompts for In-Context Learning* (EPR), NAACL 2022 (arXiv 2112.08633)** **[verified]**
  - An LM scores each candidate by P(y | x, candidate). The scores label positives and negatives, and a dense retriever is trained contrastively.
  - SupPR is essentially EPR transposed to vision.
  - Source: [arXiv](https://arxiv.org/pdf/2112.08633)

### Inferences
- The idea of learning context selection from downstream utility is **not new**, either in vision (SupPR 2023, Partial2Global 2024, SCS 2024) or in NLP (EPR 2022, PromptPG 2023). TACS's distinguishing elements are:
  1. joint training of selector and task net, rather than a frozen generalist;
  2. the specific reward (loss with context vs. without), plus the combined Gumbel-ST + REINFORCE objective;
  3. discriminative classification and medical segmentation instead of inpainting-style VICL;
  4. a DINOv3 backbone with patch-token concatenation or gated cross-attention.
- The student's extension inherits this positioning. Its novelty has to come from the text side (Q4).

### Gaps
- The exact training objective of SCS (RL vs. heuristic search) was not confirmed on the primary page.
- I did not survey learned support-set selection for few-shot segmentation or medical few-shot learning in depth (e.g., active or learned support selection). Other researchers may cover it.

---

## Q3. Which works use Gumbel-Softmax or REINFORCE to make retrieval or selection trainable end-to-end?

### Takeaway
REINFORCE or policy gradient for example selection is established: PromptPG (ICLR 2023, text+table), SCS (ECCV 2024, likely RL), MVSelect (CVPR 2024, view selection), and LSD (2026). Gumbel-Softmax is widely used for discrete choices in vision networks (resolution, channels, patches, prototypes). I found **no prior vision paper that combines straight-through Gumbel-Softmax and REINFORCE in one loss to select a context image**. This combination looks specific to TACS (low-confidence negative; the search was limited).

### Cited Findings
- **Lu et al., *Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning* (PromptPG), ICLR 2023 (arXiv 2209.14610)** **[verified]**
  - **Method:** a frozen BERT plus a trainable linear layer forms a policy that selects few-shot examples for GPT-3, trained with policy gradient.
  - **Task:** TabMWP (text + tables), reaching 68.23% accuracy (+5.31%) with reduced variance. The authors claim it is the first use of RL to select in-context examples for GPT-3.
  - **Relation to TACS:** the same policy-over-pool idea with reward from task correctness. The task model is frozen and the setting is not vision.
  - Sources: [arXiv](https://arxiv.org/pdf/2209.14610); [ICLR](https://iclr.cc/virtual/2023/poster/12073)
- **Hou, Gould, Zheng, *Learning to Select Views for Efficient Multi-View Understanding* (MVSelect), CVPR 2024** **[verified]**
  - **Method:** a light RL module picks the next-best camera view from initial views.
  - **Results:** classification and detection using only 2–3 of N views.
  - **Relation to TACS:** cited by TACS. It selects among views of the *same* instance, not exemplars from a training pool.
  - Source: [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Learning_to_Select_Views_for_Efficient_Multi-View_Understanding_CVPR_2024_paper.html)
- **NLP retrieval trained end-to-end through latent-variable or marginal likelihood (not Gumbel):** REALM, RAG, EMDR², and Atlas distillation, all cited by TACS. — [TACS](https://arxiv.org/html/2512.00489)
- **RL or reward-trained retrieval for generation:** SmartRAG, RAG-DDR, DRO (2024–2025). — [TACS](https://arxiv.org/html/2512.00489)
- **Gumbel-Softmax or differentiable top-k for discrete choices in vision (not exemplar retrieval):**
  - patch selection as differentiable top-K: [arXiv 2104.03059](https://pdf.arxiv.org/pdf/2104.03059)
  - per-image resolution chosen with straight-through Gumbel, trained jointly: [arXiv 2106.02898](https://arxiv.org/pdf/2106.02898)
  - channel gating: [arXiv 1812.04180](https://arxiv.org/pdf/1812.04180v4.pdf)
  - prototype assignment via Gumbel: [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720346.pdf)

### Inferences
- The Gumbel-ST + REINFORCE mixture applied to choosing a training-pool context image looks specific to TACS, but each ingredient has clear precedent.
- For M-TACS, keeping TACS's selector training unchanged means it adds no novelty on this axis.

### Gaps
- I did not find or verify differentiable kNN or top-k retrieval papers in vision (e.g., Plötz & Roth, *Neural Nearest Neighbors Networks*, NeurIPS 2018, which predates the scope), and I did not verify Gumbel-top-k retrieval works. A targeted search on "differentiable nearest neighbour retrieval vision" would be worthwhile.

---

## Q4. Is there prior work that combines learned context selection with TEXT (captions or labels of the selected example)?

### Takeaway
There is **partial precedent but no exact match found**. Text attached to retrieved neighbours appears in fixed-similarity vision RAG: RAC (CVPR 2022) and Iscen et al. (2023). Learned selection of image+text demonstrations exists for multimodal LLMs: Lever-LM (NeurIPS 2024), DRUM (2025), and PromptPG (text+tables). I found **no work** that, like M-TACS, (i) learns to select a context image for a discriminative vision network jointly trained with it, and (ii) fuses a VLM-generated caption and/or the candidate's class label text into the task network via a gated residual. The label-text variant also overlaps conceptually with standard ICL, where demonstrations carry their labels, and with RAC's retrieved text, which weakens its novelty.

### Cited Findings
- **RAC:** the external memory holds pre-encoded images *and associated text snippets*, fused into the classifier. Retrieval is fixed, not learned. — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html)
- **Iscen et al. 2023:** retrieval from web image-text corpora for recognition, also fixed. — [arXiv 2304.05173](https://arxiv.org/pdf/2304.05173.pdf)
- **Lever-LM:** a learned configurator of image+text demos for VQA and captioning. The task model is a frozen LVLM. — [arXiv 2312.10104](https://arxiv.org/pdf/2312.10104)
- **PromptPG:** an RL-learned selector over text+table examples. — [arXiv 2209.14610](https://arxiv.org/pdf/2209.14610)
- **Visual ICL selectors** (SupPR, Partial2Global, SCS) select image–label-*map* pairs with no natural-language text. — [2301.13670](https://arxiv.org/pdf/2301.13670); [2405.15279](https://arxiv.org/pdf/2405.15279); [2407.10233](https://arxiv.org/abs/2407.10233)
- A targeted search for "caption/text-guided VICL prompt selection" found only text-guided *prompting* for segmentation (Prompt-DINO, arXiv 2508.06146). That work fuses text and visual prompts early but involves no learned exemplar selection. — [arXiv 2508.06146](https://arxiv.org/abs/2508.06146)
- REVEAL (CVPR 2023) is cited by TACS as multimodal retrieval with attentive fusion of retrieved knowledge (image-text). It is pre-training for VL tasks, not discriminative context selection. — [TACS](https://arxiv.org/html/2512.00489)

### Inferences
- **The defensible novelty of M-TACS** is adding caption and label text of the *learned-selected* candidate into a jointly trained discriminative vision model with gated fusion, evaluated on fine-grained or medical tasks. This is an incremental combination of TACS with RAC/REVEAL-style text-augmented retrieval.
- **Label-text variant caveat.** Feeding the context image's class label to a classifier gives a strong shortcut, close to a kNN-label vote. Reviewers will likely compare it with kNN or RAC baselines and check for label leakage when the selector picks same-class images.
- **What would strengthen the novelty claim:** showing that the text changes *which* candidate is selected (text-aware selector), not only the fusion. If text enters only after selection, the selector is unchanged from TACS.

### Gaps
- I did not search 2025–2026 medical VLM retrieval-augmented classification in depth (e.g., RAU and RAICL details), where report text attached to retrieved images is common. This is a likely source of closer precedents.
- The search was web-snippet based. Google Scholar or Semantic Scholar citation tracing from TACS (forward citations, Nov 2025–Nov 2026) was not done and could reveal concurrent work.
