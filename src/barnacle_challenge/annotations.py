"""Recover provisional instances from transparent blue outline annotations.

An enclosed background component defines the inside of an outline. Stroke pixels
are split at equal distance from inside and outside, approximating the stroke's
center rather than inflating masks to its outer edge. No gaps are auto-closed.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractionConfig:
    blue_margin: int = 35
    min_hole_area_px: int = 12
    min_stroke_component_px: int = 8
    max_instance_area_px: int = 10000

    def __post_init__(self):
        if not 0 <= self.blue_margin <= 255:
            raise ValueError("Blue margin must be in 0..255")
        if min(self.min_hole_area_px, self.min_stroke_component_px) < 1:
            raise ValueError("Minimum annotation sizes must be positive")
        if self.max_instance_area_px <= self.min_hole_area_px:
            raise ValueError("Maximum instance size must exceed minimum hole size")


def blue_strokes(annotation: np.ndarray, margin: int = 35) -> np.ndarray:
    if annotation.ndim != 3 or annotation.shape[2] not in (3, 4):
        raise ValueError("Outline annotation must be a BGR or BGRA color image")
    b, g, r = cv2.split(annotation[:, :, :3].astype(np.int16))
    selected = (b > g + margin) & (b > r + margin)
    if annotation.shape[2] == 4:
        selected &= annotation[:, :, 3] > 0
    return selected.astype(np.uint8)


def extract_instances(annotation: np.ndarray, config: ExtractionConfig | None = None):
    config = config or ExtractionConfig()
    stroke = blue_strokes(annotation, config.blue_margin)
    if not stroke.any():
        raise ValueError("No blue annotation strokes found; check the mask and color settings")
    count, components, stats, _ = cv2.connectedComponentsWithStats(stroke, connectivity=8)
    labels = np.zeros(stroke.shape, np.uint16)
    ignore = np.zeros(stroke.shape, np.uint8)
    issues, records = [], []
    next_id = 1
    noise = 0
    for component_id in range(1, count):
        x, y, w, h, size = map(int, stats[component_id])
        if size < config.min_stroke_component_px:
            noise += 1
            continue
        local = np.pad((components[y:y + h, x:x + w] == component_id).astype(np.uint8), 1)
        # Complementary connectivity prevents diagonal stroke contacts leaking holes.
        n, background, hole_stats, _ = cv2.connectedComponentsWithStats(1 - local, connectivity=4)
        exterior = int(background[0, 0])
        holes = [i for i in range(1, n)
                 if i != exterior and hole_stats[i, cv2.CC_STAT_AREA] >= config.min_hole_area_px]
        bbox = [x, y, w, h]
        if not holes:
            issue = {"kind": "open_or_too_small_outline", "component_id": component_id,
                     "bbox_xywh": bbox, "stroke_area_px": size}
            issues.append(issue)
            ignore[y:y + h, x:x + w] = 255
            continue
        if len(holes) > 1:
            issues.append({"kind": "touching_or_multiple_outlines", "component_id": component_id,
                           "bbox_xywh": bbox, "candidate_count": len(holes)})
            ignore[y:y + h, x:x + w] = 255
        exterior_distance = cv2.distanceTransform((background != exterior).astype(np.uint8),
                                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distances = np.stack([cv2.distanceTransform((background != i).astype(np.uint8),
                                                    cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for i in holes])
        nearest = np.argmin(distances, axis=0)
        within_midline = np.min(distances, axis=0) <= exterior_distance
        for index, hole_id in enumerate(holes):
            interior = background == hole_id
            mask = (interior | ((local > 0) & (nearest == index) & within_midline))[1:-1, 1:-1]
            area = int(mask.sum())
            if area > config.max_instance_area_px:
                issues.append({"kind": "oversized_candidate", "component_id": component_id,
                               "bbox_xywh": bbox, "area_px": area})
                ignore[y:y + h, x:x + w] = 255
                continue
            if next_id > np.iinfo(np.uint16).max:
                raise ValueError("Too many instances for uint16 labels; split this image")
            target = labels[y:y + h, x:x + w]
            if np.any(target[mask]):
                raise ValueError("Recovered masks overlap unexpectedly")
            target[mask] = next_id
            yy, xx = np.nonzero(mask)
            records.append({"id": next_id, "component_id": component_id,
                            "bbox_xywh": [x + int(xx.min()), y + int(yy.min()),
                                          int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)],
                            "centroid_xy": [float(x + xx.mean()), float(y + yy.mean())],
                            "area_px": area, "review_status": "unreviewed",
                            "touching_component": len(holes) > 1})
            next_id += 1
    audit = {"schema_version": 1, "candidate_instance_count": len(records),
             "stroke_component_count": count - 1, "discarded_tiny_stroke_components": noise,
             "issues": issues, "instances": records,
             "boundary_method": "equal inside/outside distance within blue stroke",
             "automatic_gap_repair": False,
             "annotation_target": "supplied_outline_region_anatomy_unconfirmed",
             "review_status": "unreviewed"}
    return labels, ignore, audit


def annotation_overlay(image: np.ndarray, labels: np.ndarray, issues: list[dict], *, origin=(0, 0)):
    result = image.copy()
    colored = np.zeros_like(image)
    yy, xx = np.nonzero(labels)
    ids = labels[yy, xx].astype(np.uint32)
    colored[yy, xx, 0] = 60 + (ids * 71 % 195)
    colored[yy, xx, 1] = 60 + (ids * 131 % 195)
    colored[yy, xx, 2] = 60 + (ids * 193 % 195)
    result[yy, xx] = (0.60 * image[yy, xx] + 0.40 * colored[yy, xx]).astype(np.uint8)
    # Distinct label boundaries, including where two instances touch.
    edges = np.zeros(labels.shape, bool)
    edges[1:] |= labels[1:] != labels[:-1]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    result[edges & (labels > 0)] = (0, 255, 255)
    for issue in issues:
        x, y, w, h = issue["bbox_xywh"]
        x, y = x - origin[0], y - origin[1]
        cv2.rectangle(result, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)
    return result
