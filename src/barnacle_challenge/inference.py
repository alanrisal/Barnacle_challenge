"""Native-resolution tiled inference and auditable mask-based deduplication.

The predictor callback accepts a BGR tile and returns (score, binary mask) pairs.
This module has no dependency on PyTorch or Detectron2.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .coco import encode_rle
from .dataset import tile_starts


@dataclass
class MaskPrediction:
    id: int
    tile_id: int
    score: float
    x: int
    y: int
    mask: np.ndarray
    internal_tile_edge: bool = False
    roi_boundary: bool = False

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2 or not self.mask.any():
            raise ValueError("A mask prediction must contain a nonempty 2D mask")
        if self.x < 0 or self.y < 0 or not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Invalid mask origin or confidence")

    @property
    def area(self):
        return int(np.count_nonzero(self.mask))

    @property
    def bbox(self):
        return [self.x, self.y, self.mask.shape[1], self.mask.shape[0]]

    @property
    def centroid(self):
        yy, xx = np.nonzero(self.mask)
        return [float(self.x + xx.mean()), float(self.y + yy.mean())]


def mask_overlap(a: MaskPrediction, b: MaskPrediction):
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1 = min(a.x + a.mask.shape[1], b.x + b.mask.shape[1])
    y1 = min(a.y + a.mask.shape[0], b.y + b.mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    left = a.mask[y0 - a.y:y1 - a.y, x0 - a.x:x1 - a.x]
    right = b.mask[y0 - b.y:y1 - b.y, x0 - b.x:x1 - b.x]
    intersection = int(np.count_nonzero(left & right))
    return (intersection / (a.area + b.area - intersection),
            intersection / min(a.area, b.area))


def reconcile_predictions(predictions, *, score_threshold=0.3, iou_threshold=0.5,
                          fragment_overlap=0.85):
    """Keep intact-context masks first, then confidence; do not average boundaries.

    Partial masks on internal tile borders may match by intersection over smaller
    mask if centers are close. This is a heuristic and its decisions are exported.
    Same-tile duplicates require stricter IoU; nearby nonoverlapping objects stay.
    """
    if not 0 <= score_threshold <= 1 or not 0 < iou_threshold <= 1 or not 0 < fragment_overlap <= 1:
        raise ValueError("Invalid confidence or duplicate matching threshold")
    selected = [p for p in predictions if p.score >= score_threshold]
    selected.sort(key=lambda p: (p.internal_tile_edge, -p.score, p.id))
    retained, decisions, grid = [], [], {}

    def cells(p):
        x, y, w, h = p.bbox
        return [(gx, gy) for gx in range(x // 64, (x + w - 1) // 64 + 1)
                for gy in range(y // 64, (y + h - 1) // 64 + 1)]

    for candidate in selected:
        nearby = sorted({i for cell in cells(candidate) for i in grid.get(cell, [])})
        duplicate = None
        for index in nearby:
            existing = retained[index]
            iou, ios = mask_overlap(candidate, existing)
            same_tile = candidate.tile_id == existing.tile_id
            reason = None
            if iou >= (max(0.75, iou_threshold) if same_tile else iou_threshold):
                reason = "mask_iou"
            elif not same_tile and ios >= fragment_overlap:
                smaller = candidate if candidate.area <= existing.area else existing
                distance = np.linalg.norm(np.array(candidate.centroid) - existing.centroid)
                if smaller.internal_tile_edge and distance <= 0.5 * np.sqrt(max(candidate.area, existing.area)):
                    reason = "tile_edge_fragment"
            if reason:
                duplicate = {"suppressed_raw_id": candidate.id, "retained_raw_id": existing.id,
                             "mask_iou": iou, "intersection_over_smaller": ios, "reason": reason}
                break
        if duplicate:
            decisions.append(duplicate)
        else:
            index = len(retained)
            retained.append(candidate)
            for cell in cells(candidate):
                grid.setdefault(cell, []).append(index)
    return retained, decisions


def tiled_predict(image, valid_region, predictor, *, tile_size=640, overlap=0.25,
                  min_score=0.2, progress=None):
    """Project sparse tile masks into ROI coordinates, retaining pixel geometry."""
    h, w = image.shape[:2]
    if valid_region.shape != (h, w) or not np.any(valid_region):
        raise ValueError("Valid region must match the image and contain pixels")
    valid_region = valid_region > 0
    boundary = valid_region & (cv2.erode(valid_region.astype(np.uint8), np.ones((3, 3), np.uint8),
                                        borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
    positions = [(x, y) for y in tile_starts(h, tile_size, overlap)
                 for x in tile_starts(w, tile_size, overlap)]
    predictions, audit = [], []
    outside_centroids = empty_masks = 0
    for tile_id, (tx, ty) in enumerate(positions):
        tile = image[ty:ty + tile_size, tx:tx + tile_size].copy()
        th, tw = tile.shape[:2]
        tile_valid = valid_region[ty:ty + th, tx:tx + tw]
        tile[~tile_valid] = 0
        tile_count = 0
        if tile_valid.any():
            for score, predicted in predictor(tile):
                score = float(score)
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Predictor returned an invalid confidence")
                mask = np.asarray(predicted, dtype=bool)
                if mask.shape != (th, tw):
                    raise ValueError("Predictor mask canvas must match the native tile")
                if score < min_score:
                    continue
                yy, xx = np.nonzero(mask)
                if not len(xx):
                    empty_masks += 1
                    continue
                cx, cy = int(round(float(xx.mean()))), int(round(float(yy.mean())))
                if not tile_valid[cy, cx]:
                    outside_centroids += 1
                    continue
                internal = bool((tx > 0 and mask[:, 0].any()) or (ty > 0 and mask[0].any())
                                or (tx + tw < w and mask[:, -1].any())
                                or (ty + th < h and mask[-1].any()))
                was_clipped = bool(np.any(mask & ~tile_valid))
                mask = mask & tile_valid
                yy, xx = np.nonzero(mask)
                if not len(xx):
                    empty_masks += 1
                    continue
                x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
                roi_edge = was_clipped or bool(np.any(mask & boundary[ty:ty + th, tx:tx + tw]))
                predictions.append(MaskPrediction(len(predictions) + 1, tile_id, score,
                    tx + x0, ty + y0, mask[y0:y1, x0:x1].copy(), internal, roi_edge))
                tile_count += 1
        audit.append({"tile_id": tile_id, "crop_origin_xy": [tx, ty],
                      "shape_hw": [th, tw], "raw_prediction_count": tile_count})
        if progress:
            progress(tile_id + 1, len(positions))
    return predictions, {"tiles": audit, "outside_roi_centroids_rejected": outside_centroids,
                         "empty_masks_rejected": empty_masks}


def encode_positioned_rle(mask, x, y, shape):
    """Encode a sparse local mask on a full COCO canvas without allocating it."""
    h, w = map(int, shape)
    mask = np.asarray(mask, dtype=bool)
    if x < 0 or y < 0 or x + mask.shape[1] > w or y + mask.shape[0] > h:
        raise ValueError("Positioned mask falls outside its canvas")
    runs = []
    for column in range(mask.shape[1]):
        values = np.pad(mask[:, column].astype(np.int8), 1)
        changes = np.diff(values)
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            begin = (x + column) * h + y + int(start)
            finish = (x + column) * h + y + int(end)
            if runs and runs[-1][1] == begin:
                runs[-1] = (runs[-1][0], finish)
            else:
                runs.append((begin, finish))
    counts, cursor = [], 0
    for start, end in runs:
        counts.extend([start - cursor, end - start])
        cursor = end
    counts.append(h * w - cursor)
    return {"size": [h, w], "counts": counts}


def prediction_records(predictions, origin, source_shape):
    records = []
    for instance_id, p in enumerate(predictions, start=1):
        x, y = p.x + origin[0], p.y + origin[1]
        centroid = p.centroid
        records.append({"id": instance_id, "raw_prediction_id": p.id, "score": p.score,
                        "area_px": p.area, "bbox_xywh": [x, y, p.mask.shape[1], p.mask.shape[0]],
                        "centroid_xy": [centroid[0] + origin[0], centroid[1] + origin[1]],
                        "segmentation": encode_positioned_rle(p.mask, x, y, source_shape),
                        "touches_roi_boundary": p.roi_boundary,
                        "touches_internal_tile_edge": p.internal_tile_edge,
                        "eligible_for_complete_size_summary": not (p.roi_boundary or p.internal_tile_edge)})
    return records


def raw_prediction_records(predictions):
    return [{"id": p.id, "tile_id": p.tile_id, "score": p.score,
             "bbox_xywh_crop": p.bbox, "segmentation_local": encode_rle(p.mask),
             "internal_tile_edge": p.internal_tile_edge, "roi_boundary": p.roi_boundary}
            for p in predictions]


def render_predictions(image, predictions):
    """Mask-only visualization. JSON masks retain any overlapping memberships."""
    labels = np.zeros(image.shape[:2], np.uint16)
    union = np.zeros(image.shape[:2], bool)
    overlap = np.zeros(image.shape[:2], bool)
    if len(predictions) > 65535:
        raise ValueError("Too many instances for display label PNG")
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i].score):
        p = predictions[index]
        ys, xs = slice(p.y, p.y + p.mask.shape[0]), slice(p.x, p.x + p.mask.shape[1])
        overlap[ys, xs] |= union[ys, xs] & p.mask
        union[ys, xs] |= p.mask
        target = labels[ys, xs]
        target[p.mask & (target == 0)] = index + 1
    from .annotations import annotation_overlay
    return annotation_overlay(image, labels, []), labels, union, overlap


def summarize_predictions(predictions, valid_region, union):
    areas = [p.area for p in predictions if not (p.roi_boundary or p.internal_tile_edge)]
    denominator = int(np.count_nonzero(valid_region))
    return {"predicted_count": len(predictions), "observable_region_area_px": denominator,
            "predicted_annotated_region_coverage_percent": float(100 * union.sum() / denominator),
            "complete_size_summary_count": len(areas),
            "excluded_from_complete_size_summary": len(predictions) - len(areas),
            "mean_complete_mask_area_px": float(np.mean(areas)) if areas else None,
            "median_complete_mask_area_px": float(np.median(areas)) if areas else None,
            "physical_calibration": None, "ground_truth_available": False,
            "accuracy_metrics": None,
            "notes": ["Predictions require review; confidence is not a probability of measurement accuracy.",
                      "Area is in pixels for the learned annotation target, not calibrated shell/basal area.",
                      "Complete-size eligibility only excludes detected ROI/tile-edge contact; hidden occlusion is not assessed.",
                      "Coverage uses the union of masks, so overlaps are counted once.",
                      "Display label PNG resolves overlaps by confidence; JSON masks are authoritative."]}
