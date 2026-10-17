"""Dependency-light helpers for portable image artifacts."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, *, unchanged: bool = False) -> np.ndarray:
    path = Path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_preview(image: np.ndarray, max_size: int = 1400) -> np.ndarray:
    scale = min(1.0, max_size / max(image.shape[:2]))
    if scale == 1:
        return image.copy()
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
