# Learned in-context example selection (RL / task-reward) and DTD SOTA — background for judging M-TACS

## A1. Which works learn demonstration/exemplar selectors from task feedback, and how do they compare to TACS / the student's project?

### Takeaway
Learning to select in-context examples from model feedback is well established in NLP (2021–2024). It has moved into VLMs (2025) and visual ICL (2023, 2026). TACS itself (arXiv 2512.00489, Nov 2025) already brings the idea to vision classification and segmentation with RL. So the student's REINFORCE selector is **not new as a general idea**. Any novelty has to be in the specific reward, the architecture (in-ViT context token, captions) or the results.

### Cited Findings
**TACS itself (the student's base paper)**
- Guo, Konuk, Strand, Matsoukas, Smith, "Learning What Helps: Task-Aligned Context Selection for Vision Tasks", arXiv 2512.00489, submitted 29 Nov 2025. A selector network is trained jointly with the task model using a "hybrid optimization scheme combining gradient-based supervision and reinforcement learning". It picks paired context examples by their effect on task performance, not by visual similarity. Evaluated on 18 datasets (fine-grained, medical classification, medical segmentation), where it beats similarity-based retrieval. — [arXiv abs](https://arxiv.org/abs/2512.00489)

**NLP / LLM retrievers trained with LM feedback**
- **EPR**: Rubin, Herzig, Berant, "Learning to Retrieve Prompts for In-Context Learning", NAACL 2022 (arXiv Dec 2021). Each candidate training example is scored by the LM's probability of the correct output when that example is the prompt. High and low scorers become positives and negatives for training a dense (contrastive) retriever. Tasks: three utterance-to-meaning-representation (semantic parsing) seq2seq tasks. — [arXiv 2112.08633](https://arxiv.org/abs/2112.08633)
- **Active Example Selection (RL)**: Zhang, Feng, Tan, EMNLP 2022, pp. 9134–9148. Prompt construction is cast as a sequential decision problem (choose and annotate one demonstration at a time), learned with **Q-learning**. The reward is a scoring function of the LM given the prompt; in practice it is **accuracy on a labeled validation set**. Learned policies give +5.8% on average on GPT-2 and generalize to unseen tasks, but the gain shrinks on larger GPT-3 models. — [ACL Anthology](https://aclanthology.org/2022.emnlp-main.622), [arXiv 2211.04486](https://arxiv.org/pdf/2211.04486)
- **CEIL**: Ye et al., "Compositional Exemplars for In-context Learning", ICML 2023 (PMLR 202:39818–39833). Selection is treated as subset selection with a **determinantal point process** (relevance plus diversity of the whole set). It is trained by contrastive learning on LM preferences. Evaluated on 12 datasets across 7 NLP tasks. — [arXiv 2302.05698](https://arxiv.org/abs/2302.05698v3), [ICML](https://icml.cc/virtual/2023/poster/24017)
- **LLM-R**: Wang, Yang, Wei (Microsoft), "Learning to Retrieve In-Context Examples for Large Language Models", EACL 2024 (arXiv Jul 2023). Candidates are ranked by the **conditional LLM log-probability of the ground-truth output**. A cross-encoder reward model learns these rankings, a dense retriever is distilled from it with soft labels, and the loop iterates. Reports +7.8% average over random selection on 30 tasks with LLaMA-7B. Ablation: dropping the reward model costs ~0.8 points, and using raw LLM log-likelihood as the reward costs ~0.5. — [arXiv 2307.07164](https://arxiv.org/abs/2307.07164v2)
- **TopK + ConE**: Peng, Ding et al., "Revisiting Demonstration Selection Strategies in In-Context Learning", ACL 2024. TopK shortlists candidates, which are then reranked by how much each one **reduces the conditional entropy of the test input** under the inference model (computed on the input, not the label). It shows that the best demonstration differs from one inference model to another. — [arXiv 2401.12087](https://arxiv.org/html/2401.12087v2)
- **RDES**: "Demonstration Selection for In-Context Learning via Reinforcement Learning", arXiv 2412.03966 (Dec 2024). RL that selects demonstrations for both relevance and diversity (text only). — [listing](https://fugumt.com/fugumt/paper_check/2412.03966v2_enmode)

**VLM / multimodal**
- Chen et al., "Provoking Multi-modal Few-Shot LVLM via Exploration-Exploitation In-Context Learning", CVPR 2025. Multimodal demonstration *combinations* for LVLMs. In the exploitation phase, the sampling policy is optimized with **policy-gradient RL** on the reward of each combination. Evaluated on 4 VQA datasets. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Provoking_Multi-modal_Few-Shot_LVLM_via_Exploration-Exploitation_In-Context_Learning_CVPR_2025_paper.html), [arXiv 2506.09473](https://arxiv.org/html/2506.09473v1)

**Vision-only (non-LLM)**
- **SupPR**: Zhang, Zhou, Liu, "What Makes Good Examples for Visual In-Context Learning?", NeurIPS 2023. The visual ICL model is a MAE-VQGAN-style inpainting prompter. The supervised prompt retriever scores each candidate by the ICL result it produces (e.g., mIoU) and uses the best and worst as positives and negatives for contrastive training. Tasks: foreground segmentation, single-object detection, colorization. The best and worst prompts can differ by over 70% mIoU. — [arXiv 2301.13670](https://arxiv.org/pdf/2301.13670), [NeurIPS](https://neurips.cc/virtual/2023/poster/70423)
- Lee, Lin, Diao, "Learning to Select Visual In-Context Demonstrations", arXiv 2603.26775 (Mar 2026). Sequential decision formulation with a **Dueling DQN** agent that builds demonstration sets for visual regression tasks. The reward is a downstream error metric, so the selector is not tied to one model architecture. — [arXiv 2603.26775](https://arxiv.org/pdf/2603.26775)

### Inferences
- **How the reward is defined, by family:** (a) absolute likelihood or score of the correct answer given the example (EPR, LLM-R, SupPR); (b) task accuracy or metric of the prompted model (Active Selection, CVPR'25 LVLM, Lee et al. 2026); (c) *reduction* of an uncertainty quantity relative to a baseline (ConE: entropy reduction of the test input). The student's reward, **loss(no context) − loss(with chosen example)**, is a baselined form of (a). Because it subtracts a per-query, context-free baseline, it is closest in spirit to ConE (a reduction) and to the "advantage" baseline used in REINFORCE. Among the papers I read, **none uses exactly "task loss without context minus task loss with the example" as the RL reward**. I cannot rule out that TACS's own RL component uses something equivalent, because I did not read its method section (see Gaps).
- **Similarity to TACS / the student:** every work above shares the core claim that the "helpful" example ≠ the "similar" example, and learns that from feedback. Differences: (1) NLP methods keep the LLM frozen; TACS trains the selector *jointly* with the task model. (2) Most NLP work uses contrastive distillation, not REINFORCE; true RL appears in Active Selection (Q-learning), RDES, CVPR'25 (policy gradient) and Lee et al. 2026 (DQN). (3) Vision-only prior work (SupPR, Lee 2026) targets visual ICL for dense or regression tasks, not classification with a jointly trained backbone.
- **Novelty for M-TACS:** the learned, reward-driven selector is prior art (TACS, plus the NLP/VLM literature). Claims should focus on the delta: the zero-context-baselined reward (if it differs from TACS's), the in-ViT candidate token or caption conditioning, and the empirical gains.

### Gaps
- Did not read the TACS method section, so its exact RL reward (whether it is already baselined against no context) is unverified. Check this before claiming the reward is new.
- RDES and Lee et al. 2026 reward details were taken only from search snippets.
- Did not check: Chang & Jia 2023 (CondAcc / datamodels "data curation"), Nguyen & Wong 2023 (influence-based selection), UDR (Li et al. 2023), Liu et al. 2023 (max information gain). These may contain "gain vs. zero-shot" style scores and are worth a quick look.

## A2. (merged into A1) Any vision-only work with the same idea?
### Takeaway
Yes: SupPR (NeurIPS 2023), Lee et al. (arXiv 2026, DQN) and TACS itself (2025) — see A1.
### Cited Findings
- See A1 (SupPR — [arXiv 2301.13670](https://arxiv.org/pdf/2301.13670); Lee et al. — [arXiv 2603.26775](https://arxiv.org/pdf/2603.26775); TACS — [arXiv 2512.00489](https://arxiv.org/abs/2512.00489)).
### Inferences
- The vision-classification setting with retrieved paired context is essentially TACS's own niche, so TACS is the main baseline to beat.
### Gaps
- Retrieval-augmented image classification (e.g., kNN-memory approaches) was not surveyed.

## B1. DTD protocol: what do papers report?

### Takeaway
The official protocol has 10 predefined splits (train/val/test of equal size, 40 images per class each), and results should be averaged over the 10 splits. Many SSL and foundation-model papers instead report split 1 only, with a logistic-regression linear probe.

### Cited Findings
- DTD comes with 10 preset splits into equal-sized train, validation and test sets; results are repeated per split and averaged. Some recent works (e.g., BYOL) use only the first split. — [HF dataset card (mcimpoi/dtd_split_1)](https://huggingface.co/datasets/mcimpoi/dtd_split_1), [original DTD paper](https://arxiv.org/pdf/1311.3618)
- DINOv2 follows the SimCLR/Chen et al. 2020 linear-eval protocol: logistic regression on precomputed frozen features over 12 datasets. — [DINOv2 arXiv 2304.07193](https://arxiv.org/html/2304.07193)
- DINOv3 uses the same 12 datasets ("Fine-S"), with scikit-learn LogisticRegression (L-BFGS) on frozen features. Resolution is set to give 1024 patch tokens (448px for patch 14, 512px for patch 16), using resize plus center crop. — [DINOv3 arXiv 2508.10104](https://arxiv.org/html/2508.10104v1)

### Inferences
- The student's result (split 1, 1 seed) is in line with the common "split 1" practice of SSL papers, but it is **not** the official 10-split mean. A single split with one seed has non-trivial variance; the original DTD paper reported about ±0.9 std across splits for its baseline (secondary summary, unverified). Stating "split 1, 1 seed" explicitly is needed.

### Gaps
- Did not confirm whether DINOv2/DINOv3 use split 1 or the 10-split mean for DTD (their text excerpts do not say).

## B2. Best reported DTD accuracies for strong backbones

### Takeaway
Frozen linear probes of the largest foundation models reach **~84–89% on DTD**: DINOv2 ViT-g 84.5 (paper) or 86.8 (DINOv3 re-eval at 448px), DINOv3 7B 87.2, EVA-CLIP-18B 87.1, PE-core-G 88.2, AM-RADIOv2.5-g 88.6 (best found). DTD is a hard, saturating benchmark. An unverified community leaderboard lists 91.7% (SFG).

### Cited Findings
Linear probe, frozen features (logistic regression):

| Model | DTD | Source / note |
|---|---|---|
| DINOv2 ViT-S/14 | 80.6 | [DINOv2 Table 8](https://arxiv.org/html/2304.07193) |
| DINOv2 ViT-L/14 | 84.0 | [DINOv2 Table 8](https://arxiv.org/html/2304.07193) |
| DINOv2 ViT-g/14 | 84.5 | [DINOv2 Table 8](https://arxiv.org/html/2304.07193) |
| OpenCLIP ViT-G/14 | 86.0 | [DINOv2 Table 8](https://arxiv.org/html/2304.07193) |
| DINOv2 ViT-g/14 (re-eval, 448px) | 86.8 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| DINOv3 ViT-7B/16 (512px) | 87.2 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| EVA-CLIP 18B/14 | 87.1 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| AIMv2 3B/14 | 87.4 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| SigLIP 2 g/16 | 87.7 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| PE-core G/14 | 88.2 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| AM-RADIOv2.5 g/14 | 88.6 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| Web-DINO 7B/14 | 79.9 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |
| Franca g/14 | 80.6 | [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1) |

- CLIP zero-shot ViT-L/14@336 on DTD is ~50.7% with the prompt "a photo of a [c] pattern" (zero-shot, not comparable). — [CLIP texture study (ECCV'22 workshop)](https://computer-vision-in-the-wild.github.io/eccv-2022/static/eccv2022/accepted_papers/texture_clip_abstract_eccv22.pdf)
- A community leaderboard lists 91.7% for "SFG" on DTD; protocol and backbone unverified. — [wizwand leaderboard](https://www.wizwand.com/sota/texture-classification-on-dtd-test)
- Note: DINOv2 ViT-g scores 84.5 in its own paper and 86.8 in the DINOv3 re-evaluation, a ~2.3-point swing from evaluation settings alone (resolution, etc.). — [DINOv2](https://arxiv.org/html/2304.07193) vs. [DINOv3](https://arxiv.org/html/2508.10104v1)

### Inferences
- **Placing the numbers:**
  - TACS paper 77.7 (ViT-S/16, 256px): below the DINOv2 ViT-S/14 frozen linear probe (80.6), but backbone, pretraining and resolution differ, so it is not apples to apples.
  - Student 84.95 (TACS ViT-L/16, 384px, split 1, 1 seed) is roughly equal to DINOv2 ViT-L/14 (84.0) and ViT-g/14 (84.5) linear probes from the DINOv2 paper.
  - Student 85.30 with captions is about 1.5–3.3 points below the 2025 frontier of frozen giant models (86.8–88.6), which use 1B–18B-parameter backbones at 448–512px.
  - So the result is **strong for ViT-L but not SOTA**. The +0.35 gain from captions is within the plausible split/seed noise (single split, single seed) and needs multiple splits and seeds to be called significant.
- The student's gain over the TACS paper (77.7 to 84.95) is driven mostly by backbone scale and resolution (ViT-S/256 to ViT-L/384). It should be compared against a plain ViT-L/16 baseline (linear probe or fine-tune, no context) at the same resolution to isolate the selector's contribution.
- Which pretrained weights the student's ViT-L/16 uses (DINOv2? DINOv3-distilled? supervised IN-21k?) determines which row above is the right reference.

### Gaps
- Could not extract CLIP ViT-L/14 linear-probe DTD numbers from the CLIP paper (Radford et al. 2021, Table 10); the PDF was not text-readable. From memory, about 82–83%, **unverified**.
- No verified **full fine-tuning** DTD SOTA was found; most foundation-model papers report only linear probes. Texture-specific methods (RADAM, the SFG leaderboard entry, a 2026 chaotic contrastive learning preprint [alphaxiv 2605.05012](https://www.alphaxiv.org/abs/2605.05012)) were not verified.
- DINOv3 distilled ViT-L DTD per-dataset numbers were not found (Table 22 lists only the 7B model).
- Papers with Code has been shut down / redirected; its DTD leaderboard could not be checked directly.
