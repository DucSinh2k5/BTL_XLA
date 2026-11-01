# Project Status After Organization

- Cataloged notebooks: 78; preserved original outputs and source cells.
- Eight DTD text-context execution notebooks and analysis are under `experiments/completed/text_context_ablation/dtd/`.
- A saved decoder-LoRA-caption classifier run is under `experiments/completed/caption_decoder_lora/dtd/`.
- Four new caption-method training templates are present; no new training was performed by this migration.
- Paper-related notebooks exist for CUB, Stanford Cars, SUN397 and ISIC 2017; published equivalence remains to be audited.
- Web classification frontend/backend exists; model adapter is not connected. Segmentation demo is an empty future slot.
- Checkpoints were imported without unpickling/activating; architecture/class-map/pool compatibility is unverified.
- `experiments/running/` is empty: local files do not establish whether an external Kaggle session is still running.

## Next Evidence To Collect

New DTD architecture-study baseline: [candidate_only with the frozen Florence-Prefix CSV](../../notebooks/02_improvements/multimodal_architecture_ablation/01_candidate_only_prefix/dtd/README.md).
Standalone split1/ViT-L16/384/K1/50-epoch template; query text and semantic selector are off.
The user's saved baseline run now reports val/test Macro-F1 0.840162/0.853032, best epoch 38, status completed 50 epochs.
See the [read-only audit and next steps](../../reports/multimodal_architecture_ablation/dtd/BASELINE_01_FLORENCE_PREFIX_CANDIDATE_ONLY.md).
No trained token-fusion/pooling result is available yet. Tiny text gate requires scale-aware diagnostics;
stderr worker exceptions and a missing epoch-48 stdout line remain noted. Historical scores are not substituted for this baseline.

Next controlled template: [study02 norm-calibrated candidate-only fusion](../../notebooks/02_improvements/multimodal_architecture_ablation/02_candidate_only_norm_fusion/dtd/README.md).
It retains the frozen Prefix CSV and recipe, L2-normalizes projected candidate text before gate/fusion,
and adds same-context validation aligned/off/shuffled interventions plus norm-ratio exports.
Saved study02 val/test F1 is 0.835824/0.849329, completed50/best38.

Study03 template adds [auxiliary m-LTM-inspired image-caption alignment](../../notebooks/02_improvements/multimodal_architecture_ablation/03_candidate_only_mltm_alignment/dtd/README.md)
to study02 fusion, using detached training-pool image anchors, the shared text projection, a positive diagonal metric
and duplicate-aware balanced transport targets. It does not change the image-only selector or train the captioner.
Saved study03 val/test F1 is 0.834947/0.848722, completed50/best38. Its training-pool matching diagnostics
must not be reported as held-out retrieval quality. Both studies' fixed-image validation text-off/shuffle F1 did not decrease.
See the [read-only audit, runtime caveats and plan](../../reports/multimodal_architecture_ablation/dtd/STUDY_02_03_REVIEW_AND_STUDY_04_PLAN.md).

Study04 is a new standalone [aligned query-image/candidate-caption selector](../../notebooks/02_improvements/multimodal_architecture_ablation/04_candidate_only_semantic_selector/dtd/README.md).
It retains study03 alignment/fusion as a structural control, adds a bounded semantic residual with visual warmup,
keeps query captions disabled and exports separate selector/fusion validation interventions. No study04 GPU metrics yet.

New segmentation extension: [ISIC 448 Qwen3-VL-8B full_both](../../notebooks/02_improvements/segmentation/full_both/qwen3_vl_8b/isic2017/README.md)
remains an unexecuted standalone template with caption generation/resume. The executed result was supplied separately
as `isic-segmen-captions.ipynb`: test Dice 84.8689%, IoU 76.5407%, below the historical visual 85.60%/77.38%.
See the [saved-output comparison](../../reports/paper_tables/SO_SANH_ISIC_VISUAL_VA_QWEN8B.md); one seed, not a significance claim.

Alternate captioner ready to run: [ISIC 448 Florence-Prefix full_both](../../notebooks/02_improvements/segmentation/full_both/florence_prefix/isic2017/README.md).
This keeps the Qwen segmentation model/training rules but generates new Florence captions in an isolated offline 4.49
worker. Prefix was best among the measured Florence controls on DTD, not proven best on ISIC. No Florence ISIC metrics yet.

1. Export each new caption-method run with protocol, input CSV hash, training history and best checkpoint identity.
2. Match split/backbone/batch/seed/budget across comparisons; report multiple seeds when available.
3. Verify a compatible SUN397 checkpoint and candidate pool before connecting the demo adapter.
4. Fill missing paper-table methods only after implementations and real outputs exist.
