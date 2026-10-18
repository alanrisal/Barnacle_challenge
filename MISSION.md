# Mission

## Why

I built a barnacle annotation pipeline for the DALI challenge: region selection, tiled COCO
data, and a Mask R-CNN + PointRend model fine-tuned in Colab for 600 steps. The next step is
the **DALI review**, where I need to explain and defend the model: what each part does, why
each setting was chosen, and what the 200 → 400 → 600-step results really show (and don't).

After that, I want to **ship a public web demo**: the trained model behind a small UI hosted on
GitHub Pages, where anyone can upload a barnacle photo and get back annotated masks plus the
data (count, sizes, coverage).

## What success looks like

1. I can trace a 640 px tile through Mask R-CNN + PointRend and name what every stage outputs.
2. For every non-default setting in `training/train_pointrend.py`, I can explain the barnacle-specific reason.
3. I can read the COCO metrics from my run and make honest claims about them in review.
4. I can explain my architecture for web inference (where the model runs, and why).
5. A working GitHub Pages demo exists that returns masks + JSON for an uploaded image.

## Starting point

- Comfortable with CNNs, losses and backprop.
- New to the R-CNN family (RPN, RoIAlign, two-stage detectors) and to PointRend.
- Built the pipeline itself, so the code and run outputs are the best teaching material.

## Constraints

- Windows machine; Detectron2 training/inference has only run in Colab (T4).
- GitHub Pages serves static files only (to be verified in the web-inference lessons).
