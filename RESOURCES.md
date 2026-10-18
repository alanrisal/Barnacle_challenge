# Resources

## Primary papers (highest trust)

- **Mask R-CNN**: He, Gkioxari, Dollár, Girshick (2017). https://arxiv.org/abs/1703.06870
  Read §3 (Mask R-CNN; RoIAlign; the loss L = Lcls + Lbox + Lmask). The baseline your model extends.
- **PointRend: Image Segmentation as Rendering**: Kirillov, Wu, He, Girshick (2019/2020). https://arxiv.org/abs/1912.08193
  Read §3 (point selection at inference and training) and §4 (architecture, Table 1). The exact head you trained.
  Key facts verified: coarse 7×7 head on P2; point MLP with 3×256 hidden layers; training samples 14² points with k=3, β=0.75;
  inference upsamples 7→224 in 5 steps, N=28² points per step; COCO R50-FPN mask AP 35.2 → 36.3 (224×224).

## Official code and docs

- Detectron2 PointRend project README: https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
  Checkpoint you started from: `pointrend_rcnn_R_50_FPN_3x_coco`, model id 164955410, mask AP 38.3.
- Detectron2 config defaults: https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
- Base-RCNN-FPN config: https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml
- PointRend config defaults: https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/config.py
- PointRend mask head (loss names, subdivision loop): https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/mask_head.py
- Detectron2 training tutorial (DefaultTrainer, hooks): https://detectron2.readthedocs.io/en/latest/tutorials/training.html
- COCO evaluation code (IoU thresholds, area ranges, maxDets): https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

## Your own evidence

- `training/train_pointrend.py`: every config override.
- `notebooks/barnacle_training_colab.ipynb`: the actual 200/400/600-step metrics and prediction counts.
- `docs/COLAB_TRAINING.md`: the reasoning written while building it.

## Web inference (to be researched before those lessons)

- GitHub Pages limits, ONNX export of Detectron2, onnxruntime-web, hosted-backend options (e.g. Hugging Face Spaces).

## Communities (for wisdom)

- Detectron2 GitHub Discussions/Issues: https://github.com/facebookresearch/detectron2/issues
- r/computervision: https://www.reddit.com/r/computervision/
