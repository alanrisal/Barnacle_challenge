"""Generate real-model reports for unseen photographs; never read ground-truth masks."""

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from barnacle_challenge.inference import (prediction_records, raw_prediction_records, reconcile_predictions,
                                         render_predictions, summarize_predictions, tiled_predict)
from barnacle_challenge.io import read_image, read_json, sha256, write_image, write_json
from barnacle_challenge.regions import detect_region, region_mask, region_overlay, whole_image_region


def load_predictor(run_dir, checkpoint_name, minimum_score, device):
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.point_rend import add_pointrend_config

    checkpoint = run_dir / checkpoint_name
    if not checkpoint.is_file() or not (run_dir / "config.yaml").is_file():
        raise ValueError("Selected run must contain the chosen checkpoint and config.yaml")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Reconnect to the Colab T4 GPU runtime.")
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(str(run_dir / "config.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = max(0, minimum_score - 0.001)
    model = build_model(cfg)
    state = DetectionCheckpointer(model).load(str(checkpoint))
    iteration = state.get("iteration")
    del state
    model.eval()
    output_counts = []

    def predict(tile):
        tensor = torch.as_tensor(np.ascontiguousarray(tile.transpose(2, 0, 1))).float()
        with torch.inference_mode():
            instances = model([{"image": tensor, "height": tile.shape[0],
                                 "width": tile.shape[1]}])[0]["instances"]
        output_counts.append(len(instances))
        instances = instances[instances.scores >= minimum_score].to("cpu")
        for score, mask in zip(instances.scores.tolist(), instances.pred_masks.numpy()):
            yield score, mask

    provenance = {"run_directory": str(run_dir), "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(run_dir / "config.yaml"),
                  "checkpoint_completed_steps": int(iteration) + 1 if iteration is not None else None,
                  "device": device, "torch": torch.__version__,
                  "max_detections_per_tile": cfg.TEST.DETECTIONS_PER_IMAGE,
                  "annotation_origin": "trained_model_predictions"}
    return predict, provenance, output_counts


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_report(image, region, crop, overlay, predictions, comparison, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [(region_overlay(image, region), "Selected region in source photograph"),
              (crop, "Native-resolution crop"), (overlay, "Predicted masks (no boxes)")]
    for axis, (panel, title) in zip(axes[0], panels):
        axis.imshow(panel[:, :, ::-1])
        axis.set_title(title)
        axis.axis("off")
    areas = [p.area for p in predictions if not (p.roi_boundary or p.internal_tile_edge)]
    axes[1, 0].hist(areas, bins=20)
    axes[1, 0].set_xlabel("Complete predicted mask area (pixels²)")
    axes[1, 0].set_ylabel("Instances")
    axes[1, 0].set_title("Size distribution — learned annotation target")
    axes[1, 1].hist([p.score for p in predictions], bins=np.linspace(0, 1, 21))
    axes[1, 1].set_xlabel("Model confidence")
    axes[1, 1].set_ylabel("Instances")
    axes[1, 1].set_title("Retained prediction scores")
    axes[1, 2].bar([str(item["score_threshold"]) for item in comparison],
                   [item["deduplicated_count"] for item in comparison])
    axes[1, 2].set_xlabel("Score threshold")
    axes[1, 2].set_ylabel("Deduplicated predicted instances")
    axes[1, 2].set_title("Threshold sensitivity (not accuracy)")
    figure.suptitle(f"{summary['image_id']} — {summary['predicted_count']} predicted instances "
                    f"at score ≥ {summary['score_threshold']}\n"
                    "Unlabeled image: accuracy unmeasured. No physical calibration; target is not confirmed whole-shell area.")
    figure.savefig(output, dpi=140)
    plt.close(figure)


def generate_reports(args):
    manifest = read_json(args.inputs)
    scenes = manifest["images"]
    ids = [scene["id"] for scene in scenes]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("Image IDs must be unique safe directory names")
    thresholds = sorted(set([args.score_threshold, *args.compare_thresholds]))
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Score thresholds must lie in [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Report directory is not empty; choose a new name")
    minimum = min(thresholds)
    if not args.prepare_only:
        if not args.run_dir:
            raise ValueError("Choose --run-dir explicitly so the intended checkpoint is used")
        predictor, provenance, output_counts = load_predictor(args.run_dir, args.checkpoint, minimum, args.device)
    else:
        provenance = {"annotation_origin": "region_preparation_only_no_model_predictions"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "model_provenance.json", provenance)
    summaries = []
    for scene in scenes:
        print(f"Processing {scene['id']}...", flush=True)
        image_path = (args.inputs.parent / scene["path"]).resolve()
        image = read_image(image_path)
        mode = scene["region_mode"]
        if mode == "auto":
            region = detect_region(image)
        elif mode == "whole_image":
            region = whole_image_region(image)
        else:
            raise ValueError("Demo region_mode must be auto or whole_image")
        region["source_sha256"] = sha256(image_path)
        region["source_image"] = str(image_path)
        valid = region_mask(region)
        x, y, w, h = region["crop_xywh"]
        valid_crop = valid[y:y + h, x:x + w]
        crop = image[y:y + h, x:x + w].copy()
        crop[valid_crop == 0] = 0
        base = args.output / scene["id"]
        write_json(base / "region.json", region)
        write_image(base / "region_preview.jpg", region_overlay(image, region))
        write_image(base / "crop.png", crop)
        write_image(base / "valid_region.png", valid_crop)
        if args.prepare_only:
            summaries.append({"image_id": scene["id"], "status": "cropped_not_predicted",
                              "crop_shape_hw": [h, w], "source_sha256": region["source_sha256"]})
            continue
        before = len(output_counts)
        started = time.monotonic()
        raw, audit = tiled_predict(crop, valid_crop, predictor, tile_size=args.tile_size,
            overlap=args.overlap, min_score=minimum,
            progress=lambda step, total: print(f"  Tile {step}/{total}", flush=True))
        predictions, decisions = reconcile_predictions(raw, score_threshold=args.score_threshold,
                                                       iou_threshold=args.duplicate_iou)
        overlay, labels, union, overlap = render_predictions(crop, predictions)
        summary = summarize_predictions(predictions, valid_crop, union)
        comparison = []
        for threshold in thresholds:
            kept, removed = reconcile_predictions(raw, score_threshold=threshold, iou_threshold=args.duplicate_iou)
            comparison.append({"score_threshold": threshold, "raw_count": sum(p.score >= threshold for p in raw),
                               "deduplicated_count": len(kept), "duplicates_suppressed": len(removed)})
        summary.update({"image_id": scene["id"], "source_image": str(image_path),
                        "source_sha256": region["source_sha256"], "score_threshold": args.score_threshold,
                        "threshold_comparison": comparison, "tile_size": args.tile_size, "overlap": args.overlap,
                        "tile_count": len(audit["tiles"]), "raw_candidates_at_minimum_score": len(raw),
                        "duplicates_suppressed": len(decisions), "mask_duplicate_iou": args.duplicate_iou,
                        "overlapping_prediction_pixels": int(overlap.sum()),
                        "region_reviewed": region["reviewed"], "region_mode": mode,
                        "checkpoint_sha256": provenance["checkpoint_sha256"],
                        "model_output_capped_tiles": sum(n >= provenance["max_detections_per_tile"] for n in output_counts[before:]),
                        "inference_and_reconciliation_seconds": time.monotonic() - started,
                        "annotation_target": "supplied_outline_region_anatomy_unconfirmed"})
        records = prediction_records(predictions, (x, y), image.shape[:2])
        write_json(base / "summary.json", summary)
        write_json(base / "instances.json", {"coordinate_system": "original_image_pixels_xy",
                   "source_shape_hw": list(image.shape[:2]), "crop_origin_xy": [x, y],
                   "checkpoint_sha256": provenance["checkpoint_sha256"], "instances": records})
        write_json(base / "raw_tile_predictions.json", {"coordinate_system": "crop_pixels_xy",
                   "minimum_score": minimum, "audit": audit, "predictions": raw_prediction_records(raw)})
        write_json(base / "duplicate_decisions.json", decisions)
        write_csv(base / "instances.csv", [{**r, "centroid_x": r["centroid_xy"][0], "centroid_y": r["centroid_xy"][1]}
                                           for r in records],
                  ["id", "raw_prediction_id", "score", "area_px", "centroid_x", "centroid_y",
                   "touches_roi_boundary", "touches_internal_tile_edge", "eligible_for_complete_size_summary"])
        write_image(base / "masks_overlay.png", overlay)
        write_image(base / "instance_id_map.png", labels)
        write_image(base / "overlapping_pixels.png", overlap.astype(np.uint8) * 255)
        full_overlay = image.copy()
        full_overlay[y:y + h, x:x + w][valid_crop > 0] = overlay[valid_crop > 0]
        write_image(base / "source_masks_overlay.png", full_overlay)
        plot_report(image, region, crop, overlay, predictions, comparison, summary, base / "report.png")
        summaries.append(summary)
    combined = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": provenance,
                "reports": summaries, "prepare_only": args.prepare_only,
                "note": "Unseen inputs are unlabeled and may be related crops; no independent accuracy is claimed."}
    write_json(args.output / "demo_summary.json", combined)
    if not args.prepare_only:
        sections = []
        for item in summaries:
            name = html.escape(item["image_id"])
            sections.append(f"<h2>{name}</h2><p>Predicted count: {item['predicted_count']} "
                            f"at score ≥ {item['score_threshold']}. Complete-mask mean area: "
                            f"{item['mean_complete_mask_area_px']} pixels².</p>"
                            f"<img style='max-width:100%' src='{name}/report.png'>"
                            f"<p><a href='{name}/instances.csv'>Instance table</a> | "
                            f"<a href='{name}/summary.json'>Summary</a></p>")
        (args.output / "report.html").write_text("<!doctype html><meta charset='utf-8'>"
            "<title>Barnacle unseen-image report</title><body style='font-family:sans-serif;max-width:1400px;margin:auto'>"
            "<h1>Unseen-image predictions</h1><p>No ground truth: accuracy is unmeasured. Areas are in pixels, "
            "for the learned annotation target; no whole-shell or physical-area claim is made.</p>"
            + "".join(sections) + "</body>", encoding="utf-8")
    print(json.dumps(combined, indent=2))
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", choices=("model_final.pth", "model_best.pth"), default="model_final.pth")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--compare-thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="CPU crop check; no predictions are generated")
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    if args.run_dir:
        args.run_dir = args.run_dir.resolve()
    generate_reports(args)


if __name__ == "__main__":
    main()
