"""Check pixel masks, COCO geometry, provenance, and scene-level split isolation."""

from collections import Counter
from pathlib import Path

import numpy as np

from .coco import decode_rle, mask_geometry
from .io import read_image, read_json


def validate_dataset(root: str | Path, *, require_reviewed: bool = False) -> dict:
    root = Path(root).resolve()
    errors, warnings = [], []
    counts = Counter()
    groups, sources = {}, {}
    summary = read_json(root / "dataset_summary.json")
    scenes = {scene["id"]: scene for scene in summary["scenes"]}
    annotated = [scene for scene in summary["scenes"] if scene["split"] != "inference"]
    expected_ready = bool({"train", "val"}.issubset({s["split"] for s in annotated}) and all(
        s.get("region_reviewed") and s.get("annotations_reviewed") and s.get("annotation_coverage_verified")
        and not s.get("issue_count") and s.get("annotation_target")
        and s["annotation_target"] != "supplied_outline_region_anatomy_unconfirmed" for s in annotated))
    if bool(summary.get("training_ready")) != expected_ready:
        errors.append("Dataset training readiness contradicts scene review metadata")
    if not summary.get("training_ready"):
        message = "Dataset is provisional: anatomy, coverage, regions and annotation issues need review"
        (errors if require_reviewed else warnings).append(message)
    seen_images, seen_annotations = set(), set()
    scene_artifacts = {}

    def artifact(relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes dataset directory")
        return path

    for split in ("train", "val", "test"):
        coco = read_json(root / "annotations" / f"{split}.json")
        if bool(coco["info"].get("training_ready")) != bool(summary.get("training_ready")):
            errors.append(f"{split}: COCO and dataset training readiness disagree")
        by_image = {}
        for annotation in coco["annotations"]:
            if annotation["id"] in seen_annotations:
                errors.append(f"Duplicate annotation ID {annotation['id']}")
            seen_annotations.add(annotation["id"])
            by_image.setdefault(annotation["image_id"], []).append(annotation)
        split_ids = {record["id"] for record in coco["images"]}
        for missing_id in set(by_image) - split_ids:
            errors.append(f"{split}: annotations reference missing image {missing_id}")
        for record in coco["images"]:
            context = f"{split}/{record['id']}"
            if record["id"] in seen_images:
                errors.append(f"Duplicate image ID {record['id']}")
            seen_images.add(record["id"])
            for key, mapping in (("group_id", groups), ("source_sha256", sources)):
                value = record[key]
                if value in mapping and mapping[value] != split:
                    errors.append(f"{context}: {key} leaks between splits")
                mapping[value] = split
            try:
                image = read_image(artifact(record["file_name"]))
                labels = read_image(artifact(record["labels_file"]), unchanged=True)
                valid = read_image(artifact(record["valid_region_file"]), unchanged=True)
                ignore = read_image(artifact(record["ignore_regions_file"]), unchanged=True)
                expected_shape = (record["height"], record["width"])
                if any(a.shape[:2] != expected_shape for a in (image, labels, valid, ignore)):
                    raise ValueError("image/label/valid/ignore dimensions disagree")
                if labels.ndim != 2 or labels.dtype != np.uint16:
                    raise ValueError("instance labels must be single-channel uint16")
                if valid.ndim != 2 or ignore.ndim != 2:
                    raise ValueError("valid/ignore masks must be single-channel")
                if np.any(labels[valid == 0]):
                    raise ValueError("labels exist outside valid region")
                if np.any(image[valid == 0]):
                    raise ValueError("unannotated exterior image pixels were not masked")
                if bool(ignore.any()) != record["has_unresolved_regions"]:
                    raise ValueError("unresolved-region metadata disagrees with ignore mask")
                scene = scenes[record["scene_id"]]
                if scene["split"] != split or scene["group_id"] != record["group_id"]:
                    raise ValueError("tile split/group differs from its source scene")
                if scene["source_image_sha256"] != record["source_sha256"]:
                    raise ValueError("tile source hash differs from its scene")
                if scene["id"] not in scene_artifacts:
                    base = artifact(f"scenes/{scene['id']}")
                    scene_artifacts[scene["id"]] = (
                        read_image(base / "image.png"), read_image(base / "instances.png", unchanged=True))
                source_image, source_labels = scene_artifacts[scene["id"]]
                sx, sy = record["source_origin_xy"]
                ox, oy = scene["crop_origin_xy"]
                tx, ty = sx - ox, sy - oy
                th, tw = expected_shape
                if tx < 0 or ty < 0 or tx + tw > source_image.shape[1] or ty + th > source_image.shape[0]:
                    raise ValueError("tile source coordinates fall outside the ROI crop")
                if not np.array_equal(source_image[ty:ty + th, tx:tx + tw], image):
                    raise ValueError("tile image differs from its native-resolution source crop")
                if not np.array_equal(source_labels[ty:ty + th, tx:tx + tw], labels):
                    raise ValueError("tile instance IDs differ from source labels")
                if require_reviewed and (not record.get("training_ready") or ignore.any()):
                    raise ValueError("tile still contains unreviewed or uncertain annotations")
                reconstructed = np.zeros(expected_shape, np.uint16)
                local_ids = set()
                for annotation in by_image.get(record["id"], []):
                    if annotation["category_id"] != 1 or annotation["iscrowd"] != 0:
                        raise ValueError("unexpected category or crowd flag")
                    if annotation["segmentation"]["size"] != list(expected_shape):
                        raise ValueError("RLE canvas differs from image dimensions")
                    mask = decode_rle(annotation["segmentation"]) > 0
                    bbox, area = mask_geometry(mask)
                    if bbox != annotation["bbox"] or area != annotation["area"]:
                        raise ValueError("RLE mask area/bbox do not match COCO geometry")
                    instance_id = annotation["source_instance_id"]
                    if not 0 < instance_id <= 65535:
                        raise ValueError("invalid source instance ID")
                    if instance_id in local_ids:
                        raise ValueError("a source instance has duplicate annotations in one tile")
                    local_ids.add(instance_id)
                    if np.any(reconstructed[mask]):
                        raise ValueError("instance masks overlap")
                    reconstructed[mask] = instance_id
                    counts["tile_annotations"] += 1
                if not np.array_equal(reconstructed, labels):
                    raise ValueError("COCO masks do not reproduce the lossless label PNG")
                counts[f"{split}_tiles"] += 1
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{context}: {exc}")
    if not counts["train_tiles"] or not counts["val_tiles"]:
        errors.append("Both train and validation splits need at least one tile")
    for scene in summary["scenes"]:
        if scene["split"] != "inference":
            if scene.get("issue_count"):
                warnings.append(f"{scene['id']}: {scene['issue_count']} ambiguous outline components in region")
            if scene.get("region_clipped_instances"):
                warnings.append(f"{scene['id']}: {scene['region_clipped_instances']} instances clipped by ROI boundary")
    return {"valid": not errors, "training_ready": bool(summary.get("training_ready") and not errors),
            "counts": dict(counts), "errors": errors, "warnings": warnings}
