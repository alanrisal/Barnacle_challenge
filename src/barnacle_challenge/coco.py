"""Lossless, uncompressed COCO RLE without a training-framework dependency."""

import numpy as np


def encode_rle(mask: np.ndarray) -> dict:
    flat = (mask > 0).ravel(order="F").astype(np.uint8)
    changes = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, changes, flat.size]).astype(int).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def decode_rle(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0) or int(counts.sum()) != h * w:
        raise ValueError("Invalid RLE lengths")
    values = np.arange(len(counts), dtype=np.uint32) % 2
    return np.repeat(values, counts).reshape((h, w), order="F").astype(np.uint8)


def mask_geometry(mask: np.ndarray) -> tuple[list[int], int]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Empty instance mask")
    return [int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1),
            int(yy.max() - yy.min() + 1)], int(len(xx))
