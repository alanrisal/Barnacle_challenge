from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from barnacle_challenge.annotations import ExtractionConfig, extract_instances
from barnacle_challenge.coco import decode_rle, encode_rle
from barnacle_challenge.dataset import InstanceDataset, prepare_dataset, tile_starts
from barnacle_challenge.io import read_json, write_image, write_json
from barnacle_challenge.regions import (RegionConfig, RegionNotFound, detect_region,
                                        manual_region, region_mask, whole_image_region)
from barnacle_challenge.validation import validate_dataset
from barnacle_challenge.viewer import ViewTransform


class RegionTests(unittest.TestCase):
    def test_selects_enclosed_interior_instead_of_frame_exterior(self):
        image = np.zeros((500, 700, 3), np.uint8)
        color = cv2.cvtColor(np.uint8([[[90, 200, 180]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.rectangle(image, (50, 40), (650, 460), color, 15)
        cv2.line(image, (150, 40), (150, 460), color, 15)
        region = detect_region(image, RegionConfig(analysis_max_size=700, inset_px=2))
        x, y, w, h = region["crop_xywh"]
        self.assertGreater(x, 150)
        self.assertGreater(y, 40)
        self.assertLess(x + w, 650)
        self.assertLess(y + h, 460)
        self.assertFalse(region["reviewed"])
        self.assertEqual(region_mask(region)[250, 400], 255)
        self.assertEqual(region_mask(region)[250, 100], 0)

    def test_no_frame_does_not_silently_select_whole_image(self):
        image = np.zeros((100, 100, 3), np.uint8)
        with self.assertRaises(RegionNotFound):
            detect_region(image)
        self.assertTrue(np.all(region_mask(whole_image_region(image)) == 255))

    def test_manual_points_are_order_independent_and_validated(self):
        image = np.zeros((100, 100, 3), np.uint8)
        region = manual_region(image, [[80, 80], [20, 20], [20, 80], [80, 20]])
        self.assertEqual(region["crop_xywh"], [20, 20, 61, 61])
        for points in ([[0, 0]] * 4, [[-1, 0], [50, 0], [50, 50], [0, 50]],
                       [[0, 0], [90, 0], [90, 90], [40, 20]]):
            with self.assertRaises(ValueError):
                manual_region(image, points)

    def test_display_transform_preserves_source_coordinates_under_zoom(self):
        view = ViewTransform(0.4, np.array([20.0, 60.0]))
        source = np.array([1234.5, 999.25])
        np.testing.assert_allclose(view.to_image(view.to_screen(source)), source)
        anchor = np.array([500.0, 400.0])
        before = view.to_image(anchor)
        view.zoom(anchor, 1.25)
        np.testing.assert_allclose(view.to_image(anchor), before)


class AnnotationTests(unittest.TestCase):
    def test_center_of_thick_stroke_approximates_original_boundary(self):
        annotation = np.zeros((100, 100, 4), np.uint8)
        cv2.circle(annotation, (50, 50), 20, (200, 20, 20, 255), 5)
        labels, ignore, audit = extract_instances(annotation)
        expected = np.zeros((100, 100), np.uint8)
        cv2.circle(expected, (50, 50), 20, 1, -1)
        self.assertEqual(audit["candidate_instance_count"], 1)
        self.assertEqual(len(audit["issues"]), 0)
        self.assertFalse(ignore.any())
        relative_error = abs(np.count_nonzero(labels) - expected.sum()) / expected.sum()
        self.assertLess(relative_error, 0.08)

    def test_transparent_rgb_pixels_are_not_annotations(self):
        annotation = np.zeros((100, 100, 4), np.uint8)
        cv2.circle(annotation, (30, 30), 12, (200, 20, 20, 0), 2)
        cv2.circle(annotation, (70, 70), 12, (200, 20, 20, 255), 2)
        labels, _, audit = extract_instances(annotation)
        self.assertEqual(audit["candidate_instance_count"], 1)
        self.assertEqual(labels[30, 30], 0)

    def test_open_outline_is_flagged_not_invented(self):
        annotation = np.zeros((100, 100, 4), np.uint8)
        cv2.ellipse(annotation, (50, 50), (20, 20), 0, 20, 340, (200, 20, 20, 255), 2)
        labels, ignore, audit = extract_instances(annotation)
        self.assertFalse(labels.any())
        self.assertTrue(ignore[50, 50])
        self.assertEqual(audit["issues"][0]["kind"], "open_or_too_small_outline")

    def test_touching_rings_remain_two_candidates_and_are_flagged(self):
        annotation = np.zeros((100, 100, 4), np.uint8)
        for center in ((30, 50), (70, 50)):
            cv2.circle(annotation, center, 20, (200, 20, 20, 255), 3)
        labels, ignore, audit = extract_instances(annotation)
        self.assertEqual(audit["candidate_instance_count"], 2)
        self.assertTrue(ignore.any())
        self.assertTrue(all(record["touching_component"] for record in audit["instances"]))
        self.assertEqual(len(np.unique(labels)), 3)

    def test_minimum_size_is_explicit(self):
        with self.assertRaises(ValueError):
            ExtractionConfig(min_hole_area_px=0)


class RleAndTilingTests(unittest.TestCase):
    def test_rle_round_trip_including_first_pixel_and_empty_masks(self):
        rng = np.random.default_rng(17)
        for mask in (np.zeros((7, 9), np.uint8), np.ones((7, 9), np.uint8),
                     rng.integers(0, 2, (17, 13), dtype=np.uint8)):
            np.testing.assert_array_equal(decode_rle(encode_rle(mask)), mask)

    def test_invalid_rle_rejected(self):
        with self.assertRaises(ValueError):
            decode_rle({"size": [2, 3], "counts": [7]})

    def test_tiles_cover_far_edge_without_gaps(self):
        for length in (50, 640, 641, 1449, 1701):
            starts = tile_starts(length, 640, 0.25)
            covered = np.zeros(length, bool)
            for start in starts:
                covered[start:start + 640] = True
            self.assertTrue(covered.all())
            self.assertEqual(len(starts), len(set(starts)))


class DatasetTests(unittest.TestCase):
    def make_manifest(self, root, *, reviewed=True):
        scenes = []
        for index, split in enumerate(("train", "val"), start=1):
            image = np.full((120, 160, 3), 80 + index, np.uint8)
            labels = np.zeros((120, 160), np.uint16)
            cv2.circle(labels, (45, 50), 15, 1, -1)
            cv2.circle(labels, (110, 75), 18, 2, -1)
            write_image(root / f"image{index}.png", image)
            write_image(root / f"labels{index}.png", labels)
            scenes.append({"id": f"scene{index}", "group_id": f"photo{index}",
                           "image": f"image{index}.png", "instance_labels": f"labels{index}.png",
                           "split": split, "region_mode": "whole_image",
                           "annotation_target": "opercular_region",
                           "annotations_reviewed": reviewed, "annotation_coverage_verified": reviewed})
        path = root / "manifest.json"
        write_json(path, {"root": ".", "scenes": scenes})
        return path

    def test_end_to_end_export_reconstructs_every_instance_and_reads_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / "dataset"
            summary = prepare_dataset(manifest, output, tile_size=80)
            self.assertTrue(summary["training_ready"])
            report = validate_dataset(output, require_reviewed=True)
            self.assertTrue(report["valid"], report["errors"])
            image, target = InstanceDataset(output)[0]
            self.assertEqual(image.shape[:2], target["masks"].shape[1:])
            self.assertEqual(target["boxes"].shape[1], 4)
            self.assertEqual(target["masks"].dtype, np.uint8)
            self.assertEqual(target["valid_region"].dtype, bool)
            self.assertTrue(all(scene["tile_count"] > 0 for scene in summary["scenes"]))

    def test_provisional_data_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset"
            prepare_dataset(self.make_manifest(root, reviewed=False), output, tile_size=80)
            with self.assertRaises(ValueError):
                InstanceDataset(output)
            self.assertGreater(len(InstanceDataset(output, allow_unreviewed=True)), 0)
            self.assertFalse(validate_dataset(output, require_reviewed=True)["valid"])

    def test_scene_group_leakage_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_manifest(root)
            manifest = read_json(path)
            manifest["scenes"][1]["group_id"] = "photo1"
            write_json(path, manifest)
            output = root / "dataset"
            with self.assertRaisesRegex(ValueError, "leaks"):
                prepare_dataset(path, output)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / "dataset"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_dataset(manifest, output)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_corrupted_coco_geometry_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset"
            prepare_dataset(self.make_manifest(root), output, tile_size=80)
            path = output / "annotations/train.json"
            coco = read_json(path)
            coco["annotations"][0]["area"] += 1
            write_json(path, coco)
            self.assertFalse(validate_dataset(output)["valid"])

    def test_wrong_source_coordinates_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset"
            prepare_dataset(self.make_manifest(root), output, tile_size=80)
            path = output / "annotations/train.json"
            coco = read_json(path)
            coco["images"][0]["source_origin_xy"] = [9999, 9999]
            write_json(path, coco)
            report = validate_dataset(output)
            self.assertFalse(report["valid"])
            self.assertTrue(any("source coordinates" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
