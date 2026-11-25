# Language descriptions / generated captions as auxiliary signals for image classification (incl. texture recognition)

Context for comparison (from the student project, not a source): VLM captions (Florence-2, Qwen3-VL; 5–8 words, class names filtered) -> frozen CLIP text encoder -> learned gate into a DINOv3 ViT-L feature inside TACS. On DTD: query captions -2.5 Macro-F1, candidate captions ~neutral, shuffled captions do not hurt, gate learns ~0, filtered captions alone are at chance (2.13% / 47 classes), captions only change which context image is picked; Qwen-8B captions beat Florence by ~1 pt.

## Q1. LLM-description methods: LaBo, Label-free CBM, Menon & Vondrick, CuPL, WaffleCLIP

### Takeaway
All of these get text from an LLM (per-class descriptions), not from a captioner looking at the test image, and all score text against images inside CLIP's joint space. The gains are small and shrink as data or backbone grows. The sanity-check papers (WaffleCLIP, Yan et al.) show that random words or descriptors match LLM descriptors, so much of the "language gain" is ensembling or extra dimensions, not meaning. On DTD in particular, the gain over plain CLIP is under 1 point, and random descriptors give exactly the baseline.

### Cited Findings
- **Menon & Vondrick, "Visual Classification via Description from Large Language Models"** (arXiv 2210.07183, Oct 2022; published at ICLR 2023 per the conference listing; the arXiv page itself names no venue). GPT-3 (text-davinci-002, temp 0.7) is asked for "useful features" of each class. CLIP scores each descriptor, and the class score is the mean over its descriptors. Zero-shot only, no training. — [arXiv](https://arxiv.org/abs/2210.07183)
  - Table 1 (zero-shot top-1): ImageNet ViT-L/14 is 71.58 -> 75.00. **DTD** ViT-B/32 41.38 -> 44.26; ViT-L/14 51.33 -> 54.36; ViT-L/14@336 52.39 -> 54.95 (about +2.6 to +3 pts). — [arXiv HTML](https://arxiv.org/html/2210.07183)
- **CuPL** — Pratt, Covert, Liu, Farhadi, "What does a platypus look like? Generating customized prompts for zero-shot image classification", ICCV 2023. An LLM writes many descriptive sentences per class, which are used as CLIP prompt ensembles. Training-free; "over one percentage point gain on ImageNet". — [arXiv 2209.03320](https://arxiv.org/abs/2209.03320)
- **LaBo** — Yang, Panagopoulou, Zhou, Jin, Callison-Burch, Yatskar, "Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification", CVPR 2023. GPT-3 writes factual sentences per class. A submodular objective picks 50 concepts per class. Concept score = CLIP image · CLIP text embedding, with no learned concept predictor. A linear layer maps concept scores to classes, initialized from the LLM class-concept associations. Default backbone CLIP ViT-L/14. — [arXiv](https://arxiv.org/abs/2211.11158)
  - 1-shot: +11.7% over a linear probe, averaged over 11 datasets. With more data it is only "comparable". — [arXiv](https://arxiv.org/abs/2211.11158)
  - **DTD (Table 10): 1-shot LaBo 53.61 vs linear probe 41.67 (+11.9). Full data: LaBo 77.30 vs linear probe 81.68 (-4.4).** The paper says the LM prior "is more important for low shot settings", that concept selection's impact "decreases with more supervision", and that LaBo "marginally underperforms" a linear probe with large data. — [arXiv HTML](https://arxiv.org/html/2211.11158v2)
- **Label-free CBM** — Oikarinen, Das, Nguyen, Weng, ICLR 2023. GPT-3 concepts come from three prompts (features, surroundings, superclasses). **Filtering explicitly removes concepts too similar to a class name** (cosine > 0.85), plus long, duplicate and non-activating concepts. A CLIP-Dissect-aligned bottleneck feeds a sparse linear layer. — [arXiv HTML](https://arxiv.org/html/2304.06129)
  - Accuracy is always below the dense standard model: CIFAR-10 86.40 vs 88.80; CIFAR-100 65.13 vs 70.10; CUB 74.31 vs 76.70; Places365 43.68 vs 48.56; ImageNet 71.95 vs 76.13. — [arXiv HTML](https://arxiv.org/html/2304.06129)
- **WaffleCLIP** — Roth, Kim, Koepke, Vinyals, Schmid, Akata, "Waffling around for Performance: Visual Classification with Random Words and Broad Concepts", ICCV 2023. Replaces LLM descriptors with random character and word strings, and gets gains comparable to LLM descriptors. The authors propose this as a sanity check for LLM-based VL extensions. — [arXiv](https://arxiv.org/abs/2306.07282)
  - ViT-B/32, Table 2: **DTD CLIP 43.18 / DCLIP (Menon) 43.99 / WaffleCLIP 43.16±0.25**. Average over datasets: 57.34 / 58.56 / 58.57. — [arXiv HTML](https://arxiv.org/html/2306.07282v2)
  - The authors read the gains as "structured noise ensembling": averaging over descriptor variations is a key driver. Picking the max-similarity descriptor instead of averaging lowers accuracy, which they take as evidence against semantics. — [arXiv HTML](https://arxiv.org/html/2306.07282v2)
  - Table 4 controls: scrambled (word-shuffled) DCLIP descriptors give DTD 41.58, avg 57.55. Random descriptors 5x give DTD 43.41, avg 58.02. Swapping descriptor lists between classes gives DTD 38.42, avg 55.03, the largest drop. So shuffling words costs little, but attaching a different class's descriptions hurts. — [arXiv HTML](https://arxiv.org/html/2306.07282v2)
  - LLMs are judged more useful for high-level concepts that resolve class-name ambiguity. — [arXiv](https://arxiv.org/abs/2306.07282)
- **Yan et al., "Learning Concise and Descriptive Attributes for Visual Recognition"**, ICCV 2023. On 8 datasets, "LLM-generated attributes in a large quantity perform almost the same as random words". 32 learned attributes on CUB come close to about 10k LLM attributes. — [arXiv 2308.03685](https://arxiv.org/abs/2308.03685)

### Inferences
- The project's "shuffled captions do not hurt" result fits WaffleCLIP and Yan et al.: when text is a weak, noisy add-on, word order or meaning is not what drives the effect. The difference is that the project uses per-image captions, not per-class descriptors, so the WaffleCLIP "ensembling" explanation does not carry over directly. In the project the effect is close to zero, not a gain.
- LaBo's full-data DTD result (-4.4 vs linear probe on CLIP ViT-L/14) is the closest published analogue to "text hurts with a strong backbone on DTD". The project's -2.5 Macro-F1 from query captions has the same direction.
- LF-CBM's class-name filtering is a direct precedent for the project's filtering step, so filtering by itself is not novel.

### Gaps
- No exact DTD numbers were checked for CuPL or LF-CBM. LF-CBM does not report DTD.
- None of these works uses DINOv2/v3-type self-supervised backbones. All are CLIP-space or ResNet. No published result was found for LLM or caption text fused into a DINOv3 feature on DTD.

## Q2. Caption-augmented / image–text fusion classifiers (per-image generated captions as input)

### Takeaway
Few papers feed per-image generated captions into a supervised classifier. The ones found report small gains, mostly in few-shot or OOD zero-shot settings, and gain more from human captions than machine ones. No paper found used a gated residual fusion of captions into a strong self-supervised ViT and reported a sizeable full-data gain. Nor did any paper found report the negative result.

### Cited Findings
- **LIDE** — Nishida, Nishida, Nishioka, "Improving Few-Shot Image Classification Using Machine- and User-Generated Natural Language Descriptions", Findings of NAACL 2022. A text decoder generates descriptions, a text encoder encodes machine or user descriptions, and these are fused with image features. Machine-generated descriptions beat the baselines in few-shot, and higher-quality user descriptions do better. — [arXiv 2207.03133](https://arxiv.org/abs/2207.03133)
- **ICE (Image-Caption Encoding)** — Yu, Liao, Ravi, Tsiligkaridis, Kulis, arXiv 2402.02662 (WACV 2025 per CVF open access). At test time it makes image-conditioned and caption-conditioned CLIP predictions agree over the Top-K classes. Training-free. Gain: +0.5% average Top-1 OOD, up to +3% on hard datasets. — [arXiv](https://arxiv.org/abs/2402.02662); [CVF](https://openaccess.thecvf.com/content/WACV2025/papers/Yu_Image-Caption_Encoding_for_Improving_Zero-Shot_Generalization_WACV_2025_paper.pdf)
- **CVL** — He & Peng, "Fine-grained Image Classification via Combining Vision and Language", CVPR 2017. A two-stream vision + language model on CUB-200-2011, using natural-language descriptions of discriminative parts. Reported best among 12 methods at the time. The text is human-written class descriptions (not VLM captions), with weak 2017-era CNN backbones. — [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/He_Fine-Grained_Image_Classification_CVPR_2017_paper.pdf); [arXiv 1704.02792](https://arxiv.org/abs/1704.02792)
- Related but not classifier input: BLIP2-generated captions used as **training data** for CLIP are more faithful than raw web captions and improve CLIP (NeurIPS 2023 Datasets & Benchmarks, "Improving Multimodal Datasets with Image Captioning"). — [arXiv 2307.10350](https://arxiv.org/abs/2307.10350)

### Inferences
- The gains reported for caption fusion come from settings where the visual side is weak or data-poor: few-shot, zero-shot CLIP, or old CNNs. That fits the project's null result with DINOv3 ViT-L and full supervision.
- The project's caption-input design (frozen CLIP text encoder, learned gate) is a standard fusion pattern, not a novelty. The more distinctive contribution is the careful negative analysis: gate going to ~0, a shuffle control, and a text-only chance probe.

### Gaps
- No primary-source paper found that systematically measures when generated captions help a strong supervised visual classifier. A 2025 Preprints.org BLIP+Swin/BEiT CIFAR study appeared in search but was not checked and is low-profile.
- No gated-residual caption fusion paper was fetched and verified.

## Q3. Texture-specific language work (DTD, DTD2)

### Takeaway
Texture and language have been linked from the start. DTD's 47 classes are themselves describable attributes, and DTD2 adds free-form phrase descriptions. Because DTD's class labels already are visual adjectives ("striped", "dotted"), any honest caption of a DTD image tends to restate the label. That is why the project had to filter, and why text after filtering carries little information.

### Cited Findings
- **Cimpoi, Maji, Kokkinos, Mohamed, Vedaldi, "Describing Textures in the Wild"**, CVPR 2014 (arXiv 1311.3618). It defines 47 describable texture terms and the DTD dataset. Describable attributes transfer across datasets. Combined with IFV features they beat the prior SOTA by more than 8% on FMD and KTH-TIPS-2b. — [arXiv](https://arxiv.org/abs/1311.3618)
- **Wu, Timm, Maji, "Describing Textures using Natural Language"**, ECCV 2020 (DTD2). 5,369 DTD images, 24,697 descriptions written as comma-separated property phrases. VL models capture some texture properties but fail on compositional ones (e.g., the colour of dots). Texture attributes learned from DTD2 beat expert attributes for fine-grained CUB recognition. — [arXiv 2008.01180](https://arxiv.org/abs/2008.01180); [ECVA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/384_ECCV_2020_paper.php)
- Menon & Vondrick's DTD gain (about +3 pts with ViT-L/14) and WaffleCLIP's DTD numbers (DCLIP +0.8, random +0.0 on ViT-B/32) are the published LLM-descriptor results on DTD (see Q1). — [Menon](https://arxiv.org/html/2210.07183); [WaffleCLIP](https://arxiv.org/html/2306.07282v2)

### Inferences
- In DTD, the 47 class names are the attribute vocabulary. Removing class names from captions removes most of the label-correlated content. A chance-level text-only probe (2.13%) is the expected result, which also suggests there is little leakage left.
- DTD2's finding that VL models miss compositional texture properties suggests that 5–8 word captions from Florence-2 or Qwen3-VL are unlikely to add fine texture detail beyond what DINOv3 already encodes. The ~1 pt Qwen-8B > Florence gap fits "better captioner, slightly more signal".

### Gaps
- No paper found that feeds VLM captions of DTD images into a supervised DTD classifier.
- DTD2 phrase descriptions as auxiliary input for DTD classification with modern backbones were not checked.

## Q4. Redundancy, information-theoretic arguments, and leakage

### Takeaway
The CBM literature openly documents leakage: the text/concept layer encodes hidden class information beyond its semantics, so random concepts score well. The general argument for captions is a data-processing-inequality one. A caption is a function of the image, so it cannot add information about the label beyond what the image holds. Captions can only help as a better inductive bias (prior knowledge from VLM pretraining) when the visual encoder or data is limited. No primary source was found that states this DPI argument explicitly for generated captions in classification. It is a standard argument and is labelled as inference below.

### Cited Findings
- **VLG-CBM** — Srivastava, Yan, Weng, NeurIPS 2024. Prior work showed that concept values "encode unintended information", and "even a set of random concepts could achieve comparable test accuracy to state-of-the-art CBMs". The paper proposes the Number of Effective Concepts (NEC) metric to control leakage, and reports +4.27% to +51.09% at NEC=5 over existing CBMs. — [arXiv 2408.01432](https://arxiv.org/abs/2408.01432)
- Yan et al. (ICCV 2023): large LLM attribute sets perform about the same as random words, which shows the text dimension acts as extra capacity rather than meaning. — [arXiv 2308.03685](https://arxiv.org/abs/2308.03685)
- WaffleCLIP: gains come from averaging (noise ensembling), and random descriptors match LLM descriptors. — [arXiv HTML](https://arxiv.org/html/2306.07282v2)
- Label-free CBM removes class-name-like concepts (cos > 0.85), an explicit anti-leakage step. — [arXiv HTML](https://arxiv.org/html/2304.06129)
- LaBo and LF-CBM both lose to plain visual/linear baselines with full data: DTD -4.4; ImageNet -4.2 for LF-CBM. — [LaBo](https://arxiv.org/html/2211.11158v2); [LF-CBM](https://arxiv.org/html/2304.06129)

### Inferences
- DPI (inference, not sourced): for a deterministic captioner C, I(Y; X, C(X)) = I(Y; X). Captions can help only through optimisation or inductive bias, i.e. the VLM's knowledge, not new information. With a strong DINOv3 ViT-L feature, the extra bias is small. The learned gate going to ~0 is the model finding this out.
- Query-caption harm (-2.5) fits added noise and overfitting from a high-variance input with no new information. Candidate-caption neutrality and the effect on context selection fit text acting as a weak similarity prior.
- Leakage control: filtering class names plus a text-only probe at chance is stronger leakage control than most CBM papers report (they use similarity filtering only). It is worth presenting as a methodological strength.
- Novelty judgement: "captions help classification" is not new (LIDE 2022, CVL 2017, ICE 2024, LaBo/CuPL/DCLIP 2022–23). "Random or shuffled text works as well" is also known (WaffleCLIP, Yan et al. 2023). What is new to the extent found is (a) caption fusion into a context-selection model on top of a DINOv3 backbone, and (b) a clean negative result with ablations on DTD. It is best framed as a confirmation and extension of the WaffleCLIP/LaBo-style evidence to strong self-supervised backbones, not as a positive method contribution.

### Gaps
- "Promises and Pitfalls of Black-Box Concept Learning Models" (Mahinpei et al., 2021, arXiv 2106.13314) is the commonly cited origin of CBM leakage. It was not fetched in this session, so it is unverified here.
- No primary source found that measures caption redundancy versus visual backbone strength (e.g., CLIP-B vs ViT-L vs DINOv2) for per-image generated captions.
- 2025–2026 work on VLM-caption fusion with DINOv2/v3 was not searched exhaustively. A targeted search is recommended before claiming novelty for the DINOv3 + caption combination.
