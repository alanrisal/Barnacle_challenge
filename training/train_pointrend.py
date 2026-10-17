"""Single-GPU PointRend fine-tuning on the prepared barnacle COCO export.

No training libraries are imported by --inspect-data. Unreviewed labels require
an explicit diagnostic opt-in and are recorded as provisional in every run.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from barnacle_challenge.io import read_json, sha256, write_json
from barnacle_challenge.validation import validate_dataset


WEIGHTS = ("https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/"
           "pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl")


def inspect_data(dataset):
    report = validate_dataset(dataset)
    if not report["valid"]:
        raise ValueError(f"Dataset failed structural validation: {report['errors']}")
    stats = {}
    for split in ("train", "val"):
        coco = read_json(dataset / "annotations" / f"{split}.json")
        counts = Counter(annotation["image_id"] for annotation in coco["annotations"])
        stats[split] = {"tiles": len(coco["images"]),
                        "maximum_instances_per_tile": max(counts.values(), default=0)}
    return {"validation": report, "split_statistics": stats}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detectron2-root", type=Path, default=Path("/content/detectron2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "finetune"), default="smoke")
    parser.add_argument("--steps", type=int, help="Default: smoke 50, finetune 600; iterations, not epochs")
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--max-detections", type=int, default=600)
    parser.add_argument("--allow-provisional", action="store_true",
                        help="Exploratory training only; results are not validated scientific measurements")
    parser.add_argument("--tile-index", type=int, default=0, help="Single training tile used by smoke mode")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--resume-from", type=Path, help="Resume full state from an existing checkpoint into a NEW output directory")
    parser.add_argument("--log-period", type=int, default=5, help="Write live loss/progress every N steps")
    parser.add_argument("--preview-threshold", type=float, default=0.5)
    return parser


def resume_start(checkpoint: dict, target_steps: int) -> int:
    state = checkpoint.get("trainer", {})
    if "optimizer" not in state.get("_trainer", {}):
        raise ValueError("Checkpoint lacks trainer/optimizer state. Use --weights for a weights-only new run.")
    iteration = checkpoint.get("iteration", state.get("iteration"))
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Checkpoint is missing a valid iteration counter")
    completed = iteration + 1  # Detectron2 stores the last completed zero-based iteration.
    if target_steps <= completed:
        raise ValueError(f"Checkpoint already completed {completed} steps; --steps must exceed that total")
    return completed


def train(args, report):
    import cv2
    import numpy as np
    import torch
    from detectron2.config import get_cfg
    from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                                build_detection_test_loader, build_detection_train_loader)
    from detectron2.data import transforms as T
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer, HookBase, hooks
    from detectron2.evaluation import COCOEvaluator
    from detectron2.projects.point_rend import add_pointrend_config
    from detectron2.utils.visualizer import Visualizer
    from barnacle_challenge.coco import encode_rle
    from barnacle_challenge.io import read_image, write_image

    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU unavailable; select a Colab T4 runtime")
    if args.output is None:
        raise ValueError("--output is required for training")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Run output already contains files. Use a new run name to preserve checkpoints.")
    config_path = args.detectron2_root / "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    if not config_path.is_file():
        raise ValueError("PointRend checkout is missing. Run colab_setup.py first.")
    steps = args.steps if args.steps is not None else (50 if args.mode == "smoke" else 600)
    if steps < 1 or args.learning_rate <= 0 or not 0 <= args.preview_threshold <= 1 or args.log_period < 1:
        raise ValueError("Invalid steps, learning rate or preview threshold")
    completed = 0
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = args.resume_from.resolve()
        if not resume_checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {resume_checkpoint}")
        source_manifest = read_json(resume_checkpoint.parent / "run_manifest.json")
        if source_manifest["mode"] != args.mode:
            raise ValueError("Resume must retain the original mode. To switch from one-tile smoke to all tiles, "
                             "use --mode finetune --weights CHECKPOINT instead (new optimizer and counter).")
        if source_manifest.get("training_tile_index") not in (None, args.tile_index):
            raise ValueError("Resume must use the same --tile-index as the original smoke run")
        if source_manifest["dataset_summary_sha256"] != sha256(args.dataset / "dataset_summary.json"):
            raise ValueError("Resume dataset differs from the original run. Use --weights for a new experiment.")
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        completed = resume_start(state, steps)
        del state
        print(f"Resuming after {completed} steps, targeting {steps} total: {steps - completed} additional steps", flush=True)
    maximum_gt = max(s["maximum_instances_per_tile"] for s in report["split_statistics"].values())
    if args.max_detections < maximum_gt:
        raise ValueError(f"Detection cap {args.max_detections} is below {maximum_gt} known instances in a tile")
    names = {split: f"barnacle_{split}" for split in ("train", "val")}
    for split, name in names.items():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, str(args.dataset / "annotations" / f"{split}.json"), str(args.dataset))
    train_records = DatasetCatalog.get(names["train"])
    if not 0 <= args.tile_index < len(train_records):
        raise ValueError("--tile-index is outside the training dataset")
    if args.mode == "smoke":
        single = train_records[args.tile_index]
        DatasetCatalog.register("barnacle_single_tile", lambda: [single])
        MetadataCatalog.get("barnacle_single_tile").set(thing_classes=["barnacle_annotated_region"])
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.DATASETS.TRAIN = ("barnacle_single_tile",) if args.mode == "smoke" else (names["train"],)
    cfg.DATASETS.TEST = (names["val"],)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.FORMAT = "BGR"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = steps
    cfg.SOLVER.WARMUP_ITERS = min(20, max(1, steps // 10))
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.CHECKPOINT_PERIOD = min(100, steps)
    cfg.TEST.EVAL_PERIOD = 100 if args.mode == "finetune" else 0
    cfg.TEST.DETECTIONS_PER_IMAGE = args.max_detections
    cfg.SEED = 42
    if resume_checkpoint:
        # Retain architecture, optimizer hyperparameters, warmup and dataset settings.
        cfg.merge_from_file(str(resume_checkpoint.parent / "config.yaml"))
        cfg.SOLVER.MAX_ITER = steps
    cfg.OUTPUT_DIR = str(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    summary = read_json(args.dataset / "dataset_summary.json")
    write_json(args.output / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "training_ready": summary["training_ready"], "scientific_validation_complete": False,
        "provisional_label_opt_in": args.allow_provisional, "steps": steps,
        "training_tile_index": args.tile_index if args.mode == "smoke" else None,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "dataset_summary_sha256": sha256(args.dataset / "dataset_summary.json"),
        "data_inspection": report, "checkpoint_initialization": str(resume_checkpoint) if resume_checkpoint else args.weights,
        "resume_from": str(resume_checkpoint) if resume_checkpoint else None,
        "starting_step": completed, "additional_steps": steps - completed,
        "evaluation_max_detections": args.max_detections,
        "notes": ["Smoke mode repeats one tile and is not a generalization test.",
                  "Provisional labels contain unresolved regions; generic COCO training does not ignore them.",
                  "Custom detection cap means AP is not directly comparable to standard COCO AP@100.",
                  "Tile counts overlap and must not be summed into an image count."]})
    environment = args.detectron2_root / "barnacle_environment.json"
    if environment.exists():
        shutil.copy2(environment, args.output / "environment.json")

    class LiveMetrics(HookBase):
        def before_train(self):
            self.started = time.monotonic()
            self.start = self.trainer.start_iter

        def after_step(self):
            step = self.trainer.iter + 1
            if step % args.log_period and step != self.trainer.max_iter:
                return
            elapsed = time.monotonic() - self.started
            metrics = {name: float(value) for name, (value, iteration) in self.trainer.storage.latest().items()
                       if iteration == self.trainer.iter and ("loss" in name or name == "lr")
                       and math.isfinite(float(value))}
            metrics.update({"step": step, "total_steps": self.trainer.max_iter,
                            "starting_step": self.start, "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed / max(1, step - self.start) * (self.trainer.max_iter - step)})
            with (args.output / "live_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")

    class Trainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, config):
            augmentations = [] if args.mode == "smoke" else [
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomBrightness(0.85, 1.15), T.RandomContrast(0.85, 1.15)]
            mapper = DatasetMapper(config, is_train=True, augmentations=augmentations)
            return build_detection_train_loader(config, mapper=mapper)

        @classmethod
        def build_test_loader(cls, config, dataset_name):
            return build_detection_test_loader(config, dataset_name,
                mapper=DatasetMapper(config, is_train=False, augmentations=[]))

        @classmethod
        def build_evaluator(cls, config, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, tasks=("bbox", "segm"), distributed=False,
                                 output_dir=output_folder or str(args.output / "evaluation"),
                                 max_dets_per_image=args.max_detections, use_fast_impl=False)

        def build_hooks(self):
            result = super().build_hooks()
            if args.mode == "finetune":
                # After evaluation, preserve the checkpoint with best validation mask AP.
                result.insert(-1, hooks.BestCheckpointer(
                    cfg.TEST.EVAL_PERIOD, self.checkpointer, "segm/AP", mode="max"))
            result.append(LiveMetrics())
            return result

    trainer = Trainer(cfg)
    if resume_checkpoint:
        # A new run directory references the old checkpoint; original files stay intact.
        (args.output / "last_checkpoint").write_text(str(resume_checkpoint), encoding="utf-8")
    trainer.resume_or_load(resume=bool(resume_checkpoint))
    if trainer.start_iter != completed:
        raise ValueError(f"Unexpected restored iteration: {trainer.start_iter}; expected {completed}")
    trainer.train()
    if args.mode == "finetune":
        results = getattr(trainer, "_last_eval_results", None)
        if results is None:
            results = Trainer.test(cfg, trainer.model)

        def finite(value):
            if isinstance(value, dict):
                return {key: finite(v) for key, v in value.items()}
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            return value

        write_json(args.output / "validation_metrics.json", finite(results))
    # Demonstrate actual model predictions separately from the supplied labels.
    trainer.model.eval()
    preview_records = [("train", train_records[args.tile_index]),
                       ("val", DatasetCatalog.get(names["val"])[0])]
    for split, record in preview_records:
        image = read_image(record["file_name"])
        tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        with torch.no_grad():
            instances = trainer.model([{"image": tensor, "height": image.shape[0],
                                         "width": image.shape[1]}])[0]["instances"].to("cpu")
        instances = instances[instances.scores >= args.preview_threshold]
        metadata = MetadataCatalog.get(names[split])
        truth = Visualizer(image[:, :, ::-1], metadata=metadata).draw_dataset_dict(record).get_image()
        prediction = Visualizer(image[:, :, ::-1], metadata=metadata).draw_instance_predictions(instances).get_image()
        # Labels on a separate banner keep the underlying native image coordinates intact.
        panel = np.concatenate([truth, prediction], axis=1)[:, :, ::-1].copy()
        banner = np.full((40, panel.shape[1], 3), 25, np.uint8)
        cv2.putText(banner, f"SUPPLIED LABELS (left) | MODEL PREDICTIONS (right), score >= {args.preview_threshold}",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        write_image(args.output / "previews" / f"{split}_comparison.jpg", np.concatenate([banner, panel]))
        write_json(args.output / "previews" / f"{split}_predictions.json", {
            "annotation_origin": "trained_model_predictions", "image_file": record["file_name"],
            "score_threshold": args.preview_threshold, "count": len(instances),
            "coordinate_system": "native_tile_pixels", "provisional_training_labels": not summary["training_ready"],
            "instances": [{"score": float(instances.scores[i]),
                           "bbox_xyxy": instances.pred_boxes.tensor[i].tolist(),
                           "segmentation": encode_rle(instances.pred_masks[i].numpy())}
                          for i in range(len(instances))]})
    print(f"Run complete: {args.output}")
    print("Smoke predictions can be poor after 50 steps. Inspect losses and overlays before extending training.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.dataset = args.dataset.resolve()
    report = inspect_data(args.dataset)
    print(json.dumps(report, indent=2), flush=True)
    if args.inspect_data:
        return
    if not report["validation"]["training_ready"]:
        if not args.allow_provisional:
            raise ValueError("Labels are provisional. Review them first, or explicitly use --allow-provisional "
                             "for an exploratory diagnostic with unreliable evaluation labels.")
        print("EXPLORATORY RUN: labels and resulting metrics are provisional. "
              "Unresolved annotation areas are not automatically ignored by the model.", flush=True)
    train(args, report)


if __name__ == "__main__":
    main()
