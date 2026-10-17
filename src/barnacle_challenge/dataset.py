"""Native-resolution dataset preparation and a framework-independent reader."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re

import cv2
import numpy as np

from .annotations import ExtractionConfig, annotation_overlay, blue_strokes, extract_instances
from .coco import decode_rle, encode_rle, mask_geometry
from .io import fit_preview, read_image, read_json, sha256, write_image, write_json
from .regions import (RegionConfig, detect_region, manual_region, region_mask,
                      region_overlay, whole_image_region)


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size < 32 or not 0 <= overlap < 1:
        raise ValueError("Tile size must be >=32 and overlap must lie in [0, 1)")
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _scene_region(image: np.ndarray, scene: dict, root: Path, config: RegionConfig) -> dict:
    mode = scene.get("region_mode", "auto")
    if mode == "whole_image":
        return whole_image_region(image)
    if mode == "corners":
        return manual_region(image, scene["polygon_xy"], reviewed=scene.get("region_reviewed", False))
    if mode == "file":
        saved = read_json(_resolve(root, scene["region_file"]))
        if saved.get("source_sha256") != sha256(_resolve(root, scene["image"])):
            raise ValueError(f"{scene['id']}: saved region belongs to a different image")
        selected = manual_region(image, saved["polygon_xy"], reviewed=bool(saved.get("reviewed")))
        selected["method"] = saved.get("method", "saved_polygon")
        return selected
    if mode != "auto":
        raise ValueError(f"Unsupported region_mode: {mode}")
    return detect_region(image, config)


def _validate_manifest(manifest: dict, root: Path) -> None:
    scenes = manifest.get("scenes", [])
    if not scenes:
        raise ValueError("Manifest contains no scenes")
    ids, groups, hashes, group_targets = set(), {}, {}, {}
    targets = set()
    for scene in scenes:
        scene_id = scene.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id) or scene_id in ids:
            raise ValueError("Scene IDs must be unique and contain only letters, digits, '_' or '-'")
        ids.add(scene_id)
        split = scene.get("split")
        if split not in ("train", "val", "test", "inference"):
            raise ValueError(f"{scene_id}: invalid split")
        group = scene.get("group_id")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{scene_id}: group_id is required; related crops must share it")
        if group in groups and groups[group] != split:
            raise ValueError(f"Scene group {group} leaks across dataset splits")
        groups[group] = split
        target = scene.get("annotation_target", "supplied_outline_region_anatomy_unconfirmed")
        if group in group_targets and group_targets[group] != target:
            raise ValueError(f"Group {group} has inconsistent annotation targets")
        group_targets[group] = target
        path = _resolve(root, scene["image"])
        if path.name.lower().startswith("masked_"):
            raise ValueError(f"{scene_id}: marked-up overlays cannot be training inputs")
        digest = sha256(path)
        if digest in hashes and hashes[digest] != split:
            raise ValueError("An identical input image occurs in different splits")
        hashes[digest] = split
        annotation_keys = [key for key in ("outline_mask", "instance_labels") if scene.get(key)]
        if split != "inference" and len(annotation_keys) != 1:
            raise ValueError(f"{scene_id}: provide exactly one outline_mask or instance_labels")
        if split != "inference":
            targets.add(target)
        for key in annotation_keys:
            if not _resolve(root, scene[key]).is_file():
                raise ValueError(f"{scene_id}: missing {key}")
    if len(targets) > 1:
        raise ValueError("A single-class dataset cannot mix different anatomical targets")


def _write_issue_panels(image, labels, scene, root, issues, base, blue_margin):
    """Save enlarged clean/outlined views so ambiguous strokes are easy to inspect."""
    if not issues:
        return
    strokes = None
    if scene.get("outline_mask"):
        strokes = blue_strokes(read_image(_resolve(root, scene["outline_mask"]), unchanged=True), blue_margin)
    cells = []
    for index, issue in enumerate(issues, start=1):
        x, y, w, h = issue["bbox_xywh"]
        x0, y0 = max(0, x - 12), max(0, y - 12)
        x1, y1 = min(image.shape[1], x + w + 12), min(image.shape[0], y + h + 12)
        clean = image[y0:y1, x0:x1].copy()
        outlined = clean.copy()
        if strokes is not None:
            outlined[strokes[y0:y1, x0:x1] > 0] = (255, 80, 30)
        else:
            outlined = annotation_overlay(clean, labels[y0:y1, x0:x1], [])
        scale = min(140 / clean.shape[1], 140 / clean.shape[0])
        size = (max(1, round(clean.shape[1] * scale)), max(1, round(clean.shape[0] * scale)))
        cell = np.full((190, 300, 3), 25, np.uint8)
        for side, patch in enumerate((clean, outlined)):
            resized = cv2.resize(patch, size, interpolation=cv2.INTER_NEAREST)
            cell[40:40 + size[1], side * 150:side * 150 + size[0]] = resized
        cv2.putText(cell, f"#{index} component {issue['component_id']}  x={x} y={y}", (3, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1)
        cv2.putText(cell, issue["kind"], (3, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        write_image(base / "review" / f"issue_{index:03d}.png", cell)
        issue["review_panel"] = f"review/issue_{index:03d}.png"
        cells.append(cell)
    for offset in range(0, len(cells), 20):
        page = cells[offset:offset + 20]
        sheet = np.full((((len(page) + 3) // 4) * 190, 1200, 3), 25, np.uint8)
        for index, cell in enumerate(page):
            row, col = divmod(index, 4)
            sheet[row * 190:(row + 1) * 190, col * 300:(col + 1) * 300] = cell
        write_image(base / "review" / f"contact_sheet_{offset // 20 + 1:02d}.jpg", sheet)


def _load_labels(scene: dict, root: Path, image: np.ndarray, config: ExtractionConfig):
    if scene.get("outline_mask"):
        path = _resolve(root, scene["outline_mask"])
        annotation = read_image(path, unchanged=True)
        if annotation.shape[:2] != image.shape[:2]:
            raise ValueError(f"{scene['id']}: original and annotation canvases must match exactly")
        labels, ignore, audit = extract_instances(annotation, config)
    else:
        path = _resolve(root, scene["instance_labels"])
        labels = read_image(path, unchanged=True)
        if labels.ndim != 2 or labels.dtype not in (np.uint8, np.uint16):
            raise ValueError("Corrected instance_labels must be a single-channel integer PNG")
        if labels.shape != image.shape[:2]:
            raise ValueError("Corrected label canvas must match the original photograph")
        labels = labels.astype(np.uint16)
        records = []
        yy, xx = np.nonzero(labels)
        pixel_ids = labels[yy, xx]
        max_id = int(labels.max())
        counts = np.bincount(pixel_ids, minlength=max_id + 1)
        xmin = np.full(max_id + 1, labels.shape[1], dtype=np.int32)
        ymin = np.full(max_id + 1, labels.shape[0], dtype=np.int32)
        xmax = np.zeros(max_id + 1, dtype=np.int32)
        ymax = np.zeros(max_id + 1, dtype=np.int32)
        np.minimum.at(xmin, pixel_ids, xx)
        np.minimum.at(ymin, pixel_ids, yy)
        np.maximum.at(xmax, pixel_ids, xx)
        np.maximum.at(ymax, pixel_ids, yy)
        sum_x = np.bincount(pixel_ids, weights=xx, minlength=max_id + 1)
        sum_y = np.bincount(pixel_ids, weights=yy, minlength=max_id + 1)
        for instance_id in np.flatnonzero(counts):
            if instance_id == 0:
                continue
            x0, y0 = int(xmin[instance_id]), int(ymin[instance_id])
            x1, y1 = int(xmax[instance_id]) + 1, int(ymax[instance_id]) + 1
            mask = labels[y0:y1, x0:x1] == instance_id
            bbox, area = mask_geometry(mask)
            bbox[0] += x0
            bbox[1] += y0
            count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            if count != 2:
                raise ValueError(f"Instance {instance_id} has disconnected pieces; review its identity")
            records.append({"id": int(instance_id), "area_px": area, "bbox_xywh": bbox,
                            "centroid_xy": [float(sum_x[instance_id] / counts[instance_id]),
                                            float(sum_y[instance_id] / counts[instance_id])]})
        audit = {"instances": records, "issues": [], "candidate_instance_count": len(records),
                 "boundary_method": "provided_integer_instance_labels", "automatic_gap_repair": False}
        ignore = np.zeros(labels.shape, np.uint8)
    if not np.any(labels):
        raise ValueError(f"{scene['id']}: no recoverable instances")
    audit["source_annotation_sha256"] = sha256(path)
    audit["source_annotation_path"] = str(path)
    return labels, ignore, audit


def prepare_dataset(manifest_path: str | Path, output: str | Path, *,
                    tile_size: int = 640, overlap: float = 0.25) -> dict:
    tile_starts(1, tile_size, overlap)  # Validate arguments before writing artifacts.
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    root = _resolve(manifest_path.parent, manifest.get("root", "."))
    _validate_manifest(manifest, root)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    region_config = RegionConfig(**manifest.get("region_config", {}))
    extraction_config = ExtractionConfig(**manifest.get("extraction_config", {}))
    categories = [{"id": 1, "name": "barnacle_annotated_region", "supercategory": "barnacle"}]
    exports = {split: {"info": {"description": "Provisional barnacle instance masks",
                                "training_ready": False}, "images": [], "annotations": [],
                       "categories": categories} for split in ("train", "val", "test")}
    image_id = annotation_id = 1
    scene_summaries = []
    annotated_targets = set()
    for scene in manifest["scenes"]:
        scene_id, split = scene["id"], scene["split"]
        print(f"Preparing {scene_id} ({split})...", flush=True)
        image_path = _resolve(root, scene["image"])
        image = read_image(image_path)
        region = _scene_region(image, scene, root, region_config)
        region["source_sha256"] = sha256(image_path)
        region["source_image"] = str(image_path)
        valid = region_mask(region)
        x, y, width, height = region["crop_xywh"]
        crop = image[y:y + height, x:x + width].copy()
        valid_crop = valid[y:y + height, x:x + width]
        crop[valid_crop == 0] = 0
        base = output / "scenes" / scene_id
        write_json(base / "region.json", region)
        write_image(base / "region_preview.jpg", region_overlay(image, region))
        write_image(base / "image.png", crop)
        write_image(base / "valid_region.png", valid_crop)
        summary = {"id": scene_id, "group_id": scene["group_id"], "split": split,
                   "source_image_sha256": region["source_sha256"], "source_image": str(image_path),
                   "source_shape": list(image.shape[:2]), "crop_origin_xy": [x, y],
                   "crop_shape": [height, width], "region_reviewed": region["reviewed"],
                   "annotation_target": scene.get("annotation_target", "supplied_outline_region_anatomy_unconfirmed"),
                   "calibration": None, "training_ready": False, "tile_count": 0}
        if split == "inference":
            summary["candidate_instance_count"] = None
            scene_summaries.append(summary)
            write_json(base / "metadata.json", summary)
            continue
        labels, ignore, audit = _load_labels(scene, root, image, extraction_config)
        write_image(base / "instances_full_canvas.png", labels)
        write_image(base / "ignore_full_canvas.png", ignore)
        annotated_targets.add(summary["annotation_target"])
        retained = []
        excluded = []
        for record in audit["instances"]:
            cx, cy = np.rint(record["centroid_xy"]).astype(int)
            if valid[cy, cx] == 0:
                excluded.append(record["id"])
                continue
            retained.append(record["id"])
        # Count by centroid, but preserve all labels in the ROI as training context.
        # Instances whose centers fall outside are not counted; visible fragments remain labeled.
        labels_crop = labels[y:y + height, x:x + width].copy()
        labels_crop[valid_crop == 0] = 0
        ignore_crop = ignore[y:y + height, x:x + width].copy()
        ignore_crop[valid_crop == 0] = 0
        audit["centroid_in_region_ids"] = retained
        audit["centroid_outside_region_ids"] = excluded
        audit["region_clipped_instance_ids"] = []
        full_areas = {record["id"]: record["area_px"] for record in audit["instances"]}
        crop_ids, crop_counts = np.unique(labels_crop[labels_crop > 0], return_counts=True)
        cropped_areas = dict(zip(map(int, crop_ids), map(int, crop_counts)))
        for instance_id, area in cropped_areas.items():
            if area != full_areas[instance_id]:
                audit["region_clipped_instance_ids"].append(instance_id)
        in_region_issues = []
        for issue in audit["issues"]:
            ix, iy, iw, ih = issue["bbox_xywh"]
            if valid[iy:iy + ih, ix:ix + iw].any():
                in_region_issues.append(issue)
        audit["issues_intersecting_region"] = len(in_region_issues)
        # An explicit corrected label map is the path to resolving ambiguous outlines.
        reviewed = bool(scene.get("annotations_reviewed", False))
        complete = bool(scene.get("annotation_coverage_verified", False))
        ready = bool(region["reviewed"] and reviewed and complete and not in_region_issues
                     and summary["annotation_target"] != "supplied_outline_region_anatomy_unconfirmed")
        summary.update({"candidate_instance_count": len(retained),
                        "visible_instance_fragments": len(crop_ids), "annotations_reviewed": reviewed,
                        "annotation_coverage_verified": complete,
                        "issue_count": len(in_region_issues),
                        "issue_counts": dict(Counter(issue["kind"] for issue in in_region_issues)),
                        "region_clipped_instances": len(audit["region_clipped_instance_ids"]),
                        "training_ready": ready})
        write_image(base / "instances.png", labels_crop)
        write_image(base / "ignore_regions.png", ignore_crop)
        _write_issue_panels(image, labels, scene, root, in_region_issues, base, extraction_config.blue_margin)
        write_json(base / "annotation_audit.json", audit)
        overlay = annotation_overlay(crop, labels_crop, in_region_issues, origin=(x, y))
        write_image(base / "annotation_overlay.png", overlay)
        write_image(base / "annotation_preview.jpg", fit_preview(overlay))
        coco = exports[split]
        for ty in tile_starts(height, tile_size, overlap):
            for tx in tile_starts(width, tile_size, overlap):
                tw, th = min(tile_size, width - tx), min(tile_size, height - ty)
                tile_labels = labels_crop[ty:ty + th, tx:tx + tw]
                tile_valid = valid_crop[ty:ty + th, tx:tx + tw]
                if not tile_valid.any():
                    continue
                tile_ignore = ignore_crop[ty:ty + th, tx:tx + tw]
                stem = f"{scene_id}_x{tx}_y{ty}"
                image_file = f"tiles/{split}/images/{stem}.png"
                labels_file = f"tiles/{split}/labels/{stem}.png"
                valid_file = f"tiles/{split}/valid/{stem}.png"
                ignore_file = f"tiles/{split}/ignore/{stem}.png"
                write_image(output / image_file, crop[ty:ty + th, tx:tx + tw])
                write_image(output / labels_file, tile_labels)
                write_image(output / valid_file, tile_valid)
                write_image(output / ignore_file, tile_ignore)
                coco["images"].append({"id": image_id, "file_name": image_file,
                                       "width": tw, "height": th, "scene_id": scene_id,
                                       "group_id": scene["group_id"], "source_sha256": region["source_sha256"],
                                       "source_origin_xy": [x + tx, y + ty],
                                       "labels_file": labels_file, "valid_region_file": valid_file,
                                       "ignore_regions_file": ignore_file, "training_ready": ready,
                                       "has_unresolved_regions": bool(tile_ignore.any())})
                for instance_id in np.unique(tile_labels[tile_labels > 0]):
                    mask = tile_labels == instance_id
                    bbox, area = mask_geometry(mask)
                    coco["annotations"].append({"id": annotation_id, "image_id": image_id,
                                                "category_id": 1, "iscrowd": 0,
                                                "segmentation": encode_rle(mask), "bbox": bbox,
                                                "area": area, "source_instance_id": int(instance_id),
                                                "truncated_by_tile": area != cropped_areas[int(instance_id)],
                                                "truncated_by_region": int(instance_id) in audit["region_clipped_instance_ids"],
                                                "review_status": "reviewed" if ready else "unreviewed"})
                    annotation_id += 1
                summary["tile_count"] += 1
                image_id += 1
        scene_summaries.append(summary)
        write_json(base / "metadata.json", summary)
    if len(annotated_targets) > 1:
        raise ValueError("A single-class dataset cannot mix different anatomical targets")
    annotated = [scene for scene in scene_summaries if scene["split"] != "inference"]
    has_train_val = {"train", "val"}.issubset({scene["split"] for scene in annotated})
    training_ready = bool(has_train_val and all(scene["training_ready"] for scene in annotated))
    summary = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "manifest_sha256": sha256(manifest_path), "tile_size": tile_size, "overlap": overlap,
               "region_config": asdict(region_config), "extraction_config": asdict(extraction_config),
               "training_ready": training_ready, "scenes": scene_summaries,
               "notes": ["Masks are provisional until annotation anatomy, coverage and boundaries are reviewed.",
                         "Counts refer to recovered labels, not model predictions or verified biological counts.",
                         "Related photos/crops must share group_id before splitting.",
                         "Overlapping tiles repeat instances; never sum tile counts.",
                         "No physical scale or biological area is inferred during preparation."]}
    for split, coco in exports.items():
        coco["info"]["training_ready"] = training_ready
        coco["info"]["dataset_root"] = ".."
        coco["info"]["annotation_target"] = next(iter(annotated_targets), None)
        write_json(output / "annotations" / f"{split}.json", coco)
    write_json(output / "dataset_summary.json", summary)
    write_json(output / "input_manifest.json", manifest)
    return summary


class InstanceDataset:
    """Load RGB arrays and instance targets; callers can convert these to tensors.

    Provisional data requires an explicit opt-in. This prevents a future trainer
    from silently treating unresolved outline regions as reliable background.
    """

    def __init__(self, root: str | Path, split: str = "train", *, allow_unreviewed: bool = False):
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be train, val or test")
        self.root = Path(root)
        self.coco = read_json(self.root / "annotations" / f"{split}.json")
        summary = read_json(self.root / "dataset_summary.json")
        if not allow_unreviewed and not (self.coco["info"].get("training_ready") and summary.get("training_ready")):
            raise ValueError("Dataset is provisional. Review anatomy, coverage, regions and outline issues first.")
        self.images = self.coco["images"]
        self.annotations = {}
        for annotation in self.coco["annotations"]:
            self.annotations.setdefault(annotation["image_id"], []).append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        image = cv2.cvtColor(read_image(self.root / record["file_name"]), cv2.COLOR_BGR2RGB)
        annotations = self.annotations.get(record["id"], [])
        h, w = image.shape[:2]
        masks = np.stack([decode_rle(a["segmentation"]) for a in annotations]) if annotations else np.zeros((0, h, w), np.uint8)
        boxes = np.array([a["bbox"] for a in annotations], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        target = {"image_id": record["id"], "boxes": boxes,
                  "labels": np.ones(len(annotations), np.int64), "masks": masks,
                  "area": np.array([a["area"] for a in annotations], np.float32),
                  "iscrowd": np.zeros(len(annotations), np.int64),
                  "source_instance_ids": np.array([a["source_instance_id"] for a in annotations], np.int64),
                  "valid_region": read_image(self.root / record["valid_region_file"], unchanged=True) > 0,
                  "ignore_regions": read_image(self.root / record["ignore_regions_file"], unchanged=True) > 0,
                  "source_origin_xy": record["source_origin_xy"]}
        return image, target
