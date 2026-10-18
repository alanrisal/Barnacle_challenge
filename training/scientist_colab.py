"""A scientist's workbench inside a Colab output; no HTTP server or public tunnel.

The browser calls a narrowly scoped Colab callback. One background worker owns
the predictor, so status callbacks remain responsive while GPU tiles run.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import sys
import threading
import uuid

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from barnacle_challenge.coco import encode_rle
from barnacle_challenge.inference import (prediction_records, raw_prediction_records,
    reconcile_predictions, render_predictions, summarize_predictions, tiled_predict)
from barnacle_challenge.io import write_image, write_json
from barnacle_challenge.regions import detect_region, region_mask, whole_image_region
from predict_unseen import load_predictor, write_csv

MAX_BYTES = 40 * 1024**2
MAX_PIXELS = 24_000_000
CHUNK_SIZE = 512 * 1024


def decode_upload(data):
    """Check image dimensions before allocating; normalize EXIF orientation once."""
    from PIL import Image, ImageOps
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError("Choose a PNG or JPEG smaller than 40 MiB.")
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in ("PNG", "JPEG"):
            raise ValueError("Please use a PNG or JPEG photograph.")
        w, h = source.size
        if w * h > MAX_PIXELS or max(w, h) > 16000 or min(w, h) < 32:
            raise ValueError("Use an image between 32 pixels per side and 24 megapixels (maximum side 16000).")
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_html(config=None, result=None):
    template = Path(__file__).with_name("scientist_ui.html").read_text(encoding="utf-8")
    # Base64 prevents filenames and metadata from becoming executable HTML.
    boot = base64.b64encode(json.dumps({"config": config or {}, "result": result},
                                     allow_nan=False).encode()).decode()
    return template.replace("__BARNACLE_BOOT__", boot)


def analyze(image, predictor, provenance, *, threshold=0.3, mode="auto", progress=None):
    if not np.isfinite(threshold) or not 0.2 <= threshold <= 0.95:
        raise ValueError("Confidence must be between 0.20 and 0.95.")
    if mode not in ("auto", "whole_image"):
        raise ValueError("Choose green-frame detection or the whole image.")
    region = detect_region(image) if mode == "auto" else whole_image_region(image)
    valid = region_mask(region)
    x, y, w, h = region["crop_xywh"]
    valid_crop = valid[y:y+h, x:x+w]
    crop = image[y:y+h, x:x+w].copy()
    crop[valid_crop == 0] = 0
    raw, audit = tiled_predict(crop, valid_crop, predictor, tile_size=640, overlap=0.25,
                              min_score=0.2, progress=progress)
    kept, decisions = reconcile_predictions(raw, score_threshold=threshold)
    overlay, labels, union, overlaps = render_predictions(crop, kept)
    summary = summarize_predictions(kept, valid_crop, union)
    areas = [p.area for p in kept if not (p.roi_boundary or p.internal_tile_edge)]
    summary.update({"score_threshold": threshold, "duplicates_suppressed": len(decisions),
        "tile_count": len(audit["tiles"]), "region_reviewed": region["reviewed"],
        "area_std_px": float(np.std(areas)) if areas else None,
        "area_min_px": min(areas) if areas else None, "area_max_px": max(areas) if areas else None,
        "mean_confidence": float(np.mean([p.score for p in kept])) if kept else None,
        "instances_per_megapixel_of_roi": len(kept) / summary["observable_region_area_px"] * 1e6,
        "overlapping_prediction_pixels": int(overlaps.sum()),
        "threshold_comparison": [{"score_threshold": t,
            "predicted_count": len(reconcile_predictions(raw, score_threshold=t)[0])}
            for t in sorted({0.2, 0.3, 0.5, threshold})]})
    records = prediction_records(kept, (x, y), image.shape[:2])
    for record, p in zip(records, kept):
        record["segmentation_local"] = encode_rle(p.mask)
        record["equivalent_diameter_px"] = float(np.sqrt(4 * p.area / np.pi))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode the display image")
    result = {"schema_version": 1, "coordinate_system": "EXIF_normalized_original_image_pixels_xy",
        "source_shape_hw": list(image.shape[:2]), "image_data_url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
        "region": region, "summary": summary, "instances": records,
        "model": provenance, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_origin": "trained_model_predictions"}
    return result, {"crop": crop, "overlay": overlay, "labels": labels,
                    "audit": {"tiling": audit, "duplicate_decisions": decisions,
                              "raw_predictions": raw_prediction_records(raw)}}


def save_analysis(directory, image, result, extra):
    directory.mkdir(parents=True, exist_ok=False)
    write_image(directory / "source.png", image)
    write_image(directory / "crop.png", extra["crop"])
    write_image(directory / "masks_overlay.png", extra["overlay"])
    write_image(directory / "instance_id_map.png", extra["labels"])
    write_json(directory / "analysis.json", {k: v for k, v in result.items() if k != "image_data_url"})
    write_json(directory / "audit.json", extra["audit"])
    rows = [{**r, "centroid_x": r["centroid_xy"][0], "centroid_y": r["centroid_xy"][1],
             "bbox_x": r["bbox_xywh"][0], "bbox_y": r["bbox_xywh"][1],
             "bbox_width": r["bbox_xywh"][2], "bbox_height": r["bbox_xywh"][3]}
            for r in result["instances"]]
    write_csv(directory / "instances.csv", rows, ["id", "score", "area_px", "equivalent_diameter_px",
        "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "touches_roi_boundary", "touches_internal_tile_edge", "eligible_for_complete_size_summary"])
    (directory / "report.html").write_text(make_html(result=result), encoding="utf-8")
    shutil.make_archive(str(directory), "zip", directory)


class ScientistWorkbench:
    def __init__(self, workspace, trained_run, *, checkpoint="model_final.pth", save_root=None):
        self.workspace, self.run = Path(workspace).resolve(), Path(trained_run).resolve()
        if checkpoint not in ("model_final.pth", "model_best.pth"):
            raise ValueError("Select model_final.pth or model_best.pth explicitly.")
        for name in (checkpoint, "config.yaml"):
            if not (self.run / name).is_file():
                raise ValueError(f"Missing selected model file: {self.run / name}")
        self.checkpoint = checkpoint
        self.save_root = Path(save_root) if save_root else self.run.parent / "scientist_analyses"
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="barnacle")
        self.lock = threading.Lock()
        self.state = {"status": "idle"}
        self.upload = None
        self.predictor = None
        self.result_text = None
        self.last_output = None
        self.callback = "barnacle.workbench." + uuid.uuid4().hex

    def _set(self, **values):
        with self.lock:
            self.state.update(values)

    def _work(self, data, filename, threshold, mode):
        try:
            self._set(status="running", message="Reading original image", progress=0.03)
            image = decode_upload(data)
            if self.predictor is None:
                self._set(message="Loading your selected PointRend checkpoint", progress=0.05)
                self.predictor, self.provenance, self.output_counts = load_predictor(
                    self.run, self.checkpoint, 0.2, "cuda")
            self.output_counts.clear()
            self._set(message="Selecting the analysis region", progress=0.08)
            result, extra = analyze(image, self.predictor, self.provenance,
                threshold=threshold, mode=mode,
                progress=lambda step, total: self._set(message=f"Segmenting tile {step} of {total}",
                    progress=0.1 + 0.75 * step / total))
            result["filename"] = filename
            result["summary"]["model_output_capped_tiles"] = sum(
                n >= self.provenance["max_detections_per_tile"] for n in self.output_counts)
            name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            destination = self.save_root / name
            result["saved_to"] = str(destination)
            self._set(message="Saving masks, measurements and interactive report", progress=0.9)
            save_analysis(destination, image, result, extra)
            self.result_text = json.dumps(result, allow_nan=False)
            self.last_output = destination
            self._set(status="complete", message="Analysis ready", progress=1,
                      result_length=len(self.result_text), saved_to=str(destination))
        except Exception as error:
            self._set(status="error", message=str(error), progress=0)

    def dispatch(self, action, args=None):
        args = args or {}
        try:
            if action == "status":
                with self.lock:
                    return dict(self.state)
            if action == "result_chunk":
                if self.state["status"] != "complete":
                    raise ValueError("No completed result is available.")
                start = int(args.get("offset", 0))
                if not 0 <= start < len(self.result_text):
                    raise ValueError("Invalid result offset")
                return {"text": self.result_text[start:start + CHUNK_SIZE]}
            if self.state["status"] == "running":
                raise ValueError("An image is already being processed. Wait for it to finish.")
            if action == "begin":
                size = int(args["size"])
                if not 0 < size <= MAX_BYTES:
                    raise ValueError("Choose a PNG or JPEG smaller than 40 MiB.")
                self.upload = {"id": uuid.uuid4().hex, "size": size, "data": bytearray(),
                               "name": str(args.get("name", "photograph"))[:200]}
                return {"id": self.upload["id"]}
            if action == "chunk":
                upload = self.upload
                if not upload or args["id"] != upload["id"] or int(args["offset"]) != len(upload["data"]):
                    raise ValueError("Upload interrupted. Please select the image again.")
                if len(args["data"]) > CHUNK_SIZE * 2:
                    raise ValueError("Upload chunk is too large")
                chunk = base64.b64decode(args["data"], validate=True)
                if len(upload["data"]) + len(chunk) > upload["size"]:
                    raise ValueError("Upload exceeds declared image size")
                upload["data"].extend(chunk)
                return {"received": len(upload["data"])}
            if action in ("run", "sample"):
                threshold, mode = float(args.get("threshold", 0.3)), args.get("mode", "auto")
                if not np.isfinite(threshold) or not 0.2 <= threshold <= 0.95 or mode not in ("auto", "whole_image"):
                    raise ValueError("Invalid confidence or region setting")
                if action == "sample":
                    sample = args.get("id")
                    if sample not in ("unseen_img1", "unseen_img2"):
                        raise ValueError("Unknown sample")
                    path = self.workspace / "demo/inputs" / f"{sample}.png"
                    if not path.is_file():
                        raise ValueError("Sample missing; upload your own photograph or use the full Colab bundle.")
                    data, filename = path.read_bytes(), path.name
                else:
                    upload = self.upload
                    if not upload or args.get("id") != upload["id"] or len(upload["data"]) != upload["size"]:
                        raise ValueError("The image upload is incomplete.")
                    data, filename = bytes(upload["data"]), upload["name"]
                self.upload = None
                self.result_text = None
                self._set(status="running", message="Preparing analysis", progress=0)
                self.pool.submit(self._work, data, filename, threshold, mode)
                return {"status": "running"}
            raise ValueError("Unknown workbench action")
        except Exception as error:
            return {"error": str(error)}

    def show(self):
        from google.colab import output
        from IPython.display import HTML, JSON, display
        output.register_callback(self.callback, lambda action, args=None: JSON(self.dispatch(action, args)))
        display(HTML(make_html(config={"callback": self.callback,
            "model_name": self.run.name + " / " + self.checkpoint,
            "samples": [(self.workspace / "demo/inputs" / f"unseen_img{i}.png").is_file() for i in (1, 2)]})))
        return self


def launch_scientist_demo(workspace, trained_run, **kwargs):
    """Run after Detectron2 setup. Keep the returned object to download last_output."""
    return ScientistWorkbench(workspace, trained_run, **kwargs).show()
