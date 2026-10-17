import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

from barnacle_challenge.coco import decode_rle
from barnacle_challenge.inference import (MaskPrediction, encode_positioned_rle, prediction_records,
                                         reconcile_predictions, render_predictions, tiled_predict)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_checkpoint", ROOT / "training/export_checkpoint.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class InferenceTests(unittest.TestCase):
    def test_sparse_masks_encode_on_original_canvas_without_coordinate_shift(self):
        rng = np.random.default_rng(19)
        for mask, x, y, shape in [(rng.integers(0, 2, (7, 9)), 11, 8, (30, 40)),
                                  (np.ones((7, 3)), 0, 0, (7, 5)),
                                  (np.ones((4, 5)), 0, 0, (4, 5))]:
            expected = np.zeros(shape, np.uint8)
            expected[y:y + mask.shape[0], x:x + mask.shape[1]] = mask
            np.testing.assert_array_equal(decode_rle(encode_positioned_rle(mask, x, y, shape)), expected)

    def test_cross_tile_duplicate_prefers_intact_context_without_averaging(self):
        full = MaskPrediction(1, 0, 0.8, 90, 10, np.ones((8, 8), bool))
        repeated = MaskPrediction(2, 1, 0.95, 90, 10, np.ones((8, 8), bool), internal_tile_edge=True)
        kept, audit = reconcile_predictions([repeated, full])
        self.assertEqual([p.id for p in kept], [1])
        self.assertEqual(audit[0]["suppressed_raw_id"], 2)

    def test_neighboring_nonoverlapping_barnacles_remain_separate(self):
        a = MaskPrediction(1, 0, 0.8, 10, 10, np.ones((8, 8), bool))
        b = MaskPrediction(2, 1, 0.8, 19, 10, np.ones((8, 8), bool))
        kept, _ = reconcile_predictions([a, b])
        self.assertEqual(len(kept), 2)

    def test_small_tile_edge_fragment_is_not_counted_twice(self):
        full = MaskPrediction(1, 0, 0.7, 90, 10, np.ones((8, 8), bool))
        partial = MaskPrediction(2, 1, 0.9, 96, 10, np.ones((8, 2), bool), internal_tile_edge=True)
        kept, audit = reconcile_predictions([partial, full])
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit[0]["reason"], "tile_edge_fragment")
        # Mere containment within one tile must not trigger the fragment rule.
        partial.tile_id = 0
        self.assertEqual(len(reconcile_predictions([partial, full])[0]), 2)

    def test_tiling_restores_coordinates_and_reconciles_synthetic_seam_objects(self):
        image = np.zeros((100, 200, 3), np.uint8)
        for center in ((75, 50), (150, 50)):
            cv2.circle(image, center, 8, (255, 255, 255), -1)

        def predictor(tile):
            count, labels = cv2.connectedComponents((tile[:, :, 0] > 0).astype(np.uint8))
            return [(0.8, labels == i) for i in range(1, count)]

        raw, _ = tiled_predict(image, np.ones((100, 200), np.uint8), predictor, tile_size=100, overlap=0.5)
        kept, _ = reconcile_predictions(raw)
        self.assertGreater(len(raw), 2)
        self.assertEqual(len(kept), 2)
        records = prediction_records(kept, (30, 20), (150, 300))
        union = np.zeros((150, 300), bool)
        for record in records:
            union |= decode_rle(record["segmentation"]) > 0
        expected = np.zeros((150, 300), bool)
        expected[20:120, 30:230] = image[:, :, 0] > 0
        np.testing.assert_array_equal(union, expected)

    def test_bad_predictor_mask_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canvas"):
            tiled_predict(np.zeros((100, 100, 3), np.uint8), np.ones((100, 100)),
                          lambda tile: [(0.9, np.ones((10, 10)))])

    def test_display_overlap_does_not_destroy_authoritative_masks(self):
        a = MaskPrediction(1, 0, 0.9, 10, 10, np.ones((8, 8), bool))
        b = MaskPrediction(2, 0, 0.8, 14, 10, np.ones((8, 8), bool))
        _, labels, union, overlap = render_predictions(np.zeros((30, 30, 3), np.uint8), [a, b])
        self.assertEqual(int(union.sum()), 96)
        self.assertEqual(int(overlap.sum()), 32)
        self.assertEqual(a.area, 64)
        self.assertEqual(b.area, 64)
        self.assertEqual(labels[12, 15], 1)


class CheckpointExportTests(unittest.TestCase):
    def test_checkpoint_and_config_are_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "trained_run"
            run.mkdir()
            (run / "model_final.pth").write_bytes(b"synthetic checkpoint test fixture")
            (run / "config.yaml").write_text("MODEL: fixture")
            (run / "run_manifest.json").write_text(json.dumps({"mode": "finetune", "steps": 400}))
            exported = root / "model_snapshot"
            card = exporter.export_checkpoint(run, exported)
            self.assertEqual(card["checkpoint_sha256"], exporter.file_hash(run / "model_final.pth"))
            with zipfile.ZipFile(exported / "model_bundle.zip") as archive:
                self.assertEqual(archive.read("model_final.pth"), (run / "model_final.pth").read_bytes())
                self.assertIn("config.yaml", archive.namelist())
            self.assertEqual((exported / "last_checkpoint").read_text(), "model_final.pth")
            with self.assertRaises(ValueError):
                exporter.export_checkpoint(run, run)


if __name__ == "__main__":
    unittest.main()
