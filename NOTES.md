# Teaching notes

## Learner preferences

- Goal priority: defend the model in DALI review first, then the web demo.
- Knows CNNs/backprop; R-CNN family is new. Don't re-teach convolutions.
- Ground everything in their own code and run outputs, not toy examples.

## Planned arc (adjust as records come in)

1. Anatomy: follow one tile through Mask R-CNN + PointRend; map each config knob to a stage.
2. PointRend in depth: coarse 7×7 → subdivision to 224×224; biased point sampling in training.
3. The training run: transfer from COCO, head re-init, iterations vs epochs, LR/warmup, augmentation, resume vs --weights.
4. Reading the results: AP/AP50/AP75/APs/APm/AR, maxDets=600, the 200→600 curve, best@500 vs final, threshold vs count.
5. Review rehearsal: interleaved retrieval across 1–4; honest limitations (provisional labels, one photo per split).
6+. Web inference: where can the model run (browser ONNX vs hosted API), export, tiling + dedup in JS, UI, GitHub Pages deploy.

## Key facts from their run (from notebook outputs)

- Val segm AP: 200 steps 9.60 | 300 best 11.31 | 400 14.30 | 500 best 15.01 | 600 14.46.
- Val segm AP50: 29.84 (200) → 41.43 (400) → 40.41 (600). AP75: 2.01 → 6.04 → 6.16.
- Val AR@600: 0.353 (400), 0.344 (600). bbox AP 10.28 → 14.10 → 15.50.
- Unseen demo used `model_final.pth` (step 600), not `model_best.pth` (step 500).
- Val tile counts at score ≥0.5: 133 (200-step) vs 62 (600-step); train tile 435 vs 346.
- Unseen img2: 164 predicted at 0.3; median mask area 341.5 px (√ ≈ 18 px).
- Max GT per tile 396; default DETECTIONS_PER_IMAGE is 100.
