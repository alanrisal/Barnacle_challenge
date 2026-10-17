"""Find the interior of a green quadrat without resampling scientific pixels."""

from dataclasses import dataclass

import cv2
import numpy as np

from .io import fit_preview


class RegionNotFound(ValueError):
    pass


@dataclass(frozen=True)
class RegionConfig:
    lower_hsv: tuple[int, int, int] = (75, 55, 20)
    upper_hsv: tuple[int, int, int] = (110, 255, 255)
    analysis_max_size: int = 1600
    close_kernel: int = 9
    min_area_fraction: float = 0.06
    min_rectangularity: float = 0.80
    inset_px: float = 2.0

    def __post_init__(self):
        if len(self.lower_hsv) != 3 or len(self.upper_hsv) != 3:
            raise ValueError("HSV bounds require three values")
        if any(not 0 <= lo <= hi <= limit for lo, hi, limit in
               zip(self.lower_hsv, self.upper_hsv, (179, 255, 255))):
            raise ValueError("HSV bounds must be ordered within H=0..179, S/V=0..255")
        if self.analysis_max_size < 32 or self.close_kernel < 1 or self.close_kernel % 2 == 0:
            raise ValueError("Analysis size must be >=32; closing kernel must be positive and odd")
        if not 0 < self.min_area_fraction < 1 or not 0 < self.min_rectangularity <= 1:
            raise ValueError("Invalid region area or rectangularity threshold")
        if self.inset_px < 0:
            raise ValueError("Inset must be nonnegative")


def ordered_corners(points, shape: tuple[int, ...]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Provide four finite [x, y] points in original-image coordinates")
    h, w = shape[:2]
    if np.any(points < 0) or np.any(points[:, 0] > w - 1) or np.any(points[:, 1] > h - 1):
        raise ValueError("Region corners must lie inside the image")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 16:
        raise ValueError("Region must be a convex quadrilateral with nonzero area")
    return points


def inset_polygon(points: np.ndarray, distance: float) -> np.ndarray:
    """Offset each edge inward without warping or resampling source pixels."""
    if not distance:
        return points.copy()
    normals, constants = [], []
    center = points.mean(axis=0)
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        delta = end - start
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        if np.dot(center - start, normal) < 0:
            normal = -normal
        normals.append(normal)
        constants.append(float(np.dot(normal, start) + distance))
    result = []
    for i in range(4):
        result.append(np.linalg.solve(np.array([normals[i - 1], normals[i]]),
                                      np.array([constants[i - 1], constants[i]])))
    result = np.asarray(result)
    if any(np.dot(normal, point) < constant - 1e-5
           for point in result for normal, constant in zip(normals, constants)):
        raise ValueError("Inset is too large for this polygon")
    return result


def region_mask(region: dict) -> np.ndarray:
    h, w = region["image_shape"][:2]
    points = ordered_corners(region["polygon_xy"], (h, w))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 255)
    return mask


def manual_region(image: np.ndarray, points, *, reviewed: bool = True) -> dict:
    return _region(ordered_corners(points, image.shape), image.shape, "manual", reviewed, {})


def whole_image_region(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    return _region(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
                   image.shape, "whole_image", True, {})


def _region(points, shape, method, reviewed, diagnostics) -> dict:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 255)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    return {"schema_version": 1, "image_shape": list(shape[:2]),
            "polygon_xy": np.asarray(points).tolist(), "crop_xywh": [x, y, w, h],
            "method": method, "reviewed": reviewed,
            "coordinate_system": "original_image_pixels_xy",
            "perspective_rectified": False, "diagnostics": diagnostics}


def detect_region(image: np.ndarray, config: RegionConfig | None = None) -> dict:
    config = config or RegionConfig()
    h, w = image.shape[:2]
    scale = min(1.0, config.analysis_max_size / max(h, w))
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(config.lower_hsv), np.array(config.upper_hsv))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                            np.ones((config.close_kernel, config.close_kernel), np.uint8))
    contours, hierarchy = cv2.findContours(green, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] < 0:  # Only enclosed holes, not the outside of the frame.
                continue
            area = cv2.contourArea(contour)
            if area < small.shape[0] * small.shape[1] * config.min_area_fraction:
                continue
            approximation = cv2.approxPolyDP(contour, 0.012 * cv2.arcLength(contour, True), True)
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue
            rectangle = cv2.minAreaRect(contour)
            rectangularity = area / max(rectangle[1][0] * rectangle[1][1], 1)
            if rectangularity < config.min_rectangularity:
                continue
            factors = np.array([w / small.shape[1], h / small.shape[0]])
            corners = ordered_corners(approximation[:, 0, :] * factors, image.shape)
            candidates.append((area, rectangularity, corners))
    if not candidates:
        raise RegionNotFound("No enclosed green quadrat found. Use --interactive, --corners, "
                             "or --whole-image for an already cropped image.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    area, rectangularity, corners = candidates[0]
    corners = ordered_corners(inset_polygon(corners, config.inset_px), image.shape)
    return _region(corners, image.shape, "hsv_enclosed_quadrilateral", False,
                   {"candidate_count": len(candidates), "rectangularity": rectangularity,
                    "largest_to_second_area_ratio": area / candidates[1][0] if len(candidates) > 1 else None,
                    "lower_hsv": list(config.lower_hsv), "upper_hsv": list(config.upper_hsv),
                    "inset_px": config.inset_px,
                    "selection_rule": "largest enclosed quadrilateral; requires visual review"})


def region_overlay(image: np.ndarray, region: dict) -> np.ndarray:
    result = image.copy()
    points = np.rint(region["polygon_xy"]).astype(np.int32)
    cv2.polylines(result, [points], True, (0, 220, 255), max(2, round(max(image.shape[:2]) / 800)))
    for i, point in enumerate(points):
        cv2.circle(result, tuple(point), 8, (0, 220, 255), -1)
        cv2.putText(result, str(i + 1), tuple(point + [10, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 255), 2)
    return fit_preview(result)
