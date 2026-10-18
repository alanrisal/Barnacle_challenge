import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from scientist_colab import ScientistWorkbench, analyze, decode_upload, make_html, save_analysis
from barnacle_challenge.coco import decode_rle


class ScientistDemoTests(unittest.TestCase):
    def test_scientific_coordinates_and_export_preserve_masks(self):
        image = np.full((100, 120, 3), 80, np.uint8)
        def predictor(tile):
            mask = np.zeros(tile.shape[:2], bool)
            mask[20:30, 40:60] = True
            yield .9, mask
            edge = np.zeros_like(mask)
            edge[0:10, 80:90] = True
            yield .8, edge
        result, extra = analyze(image, predictor, {"checkpoint": "test.pth"}, mode="whole_image")
        self.assertEqual(result["summary"]["predicted_count"], 2)
        self.assertEqual(result["summary"]["mean_complete_mask_area_px"], 200)
        self.assertEqual(result["summary"]["excluded_from_complete_size_summary"], 1)
        record = next(r for r in result["instances"] if r["area_px"] == 200)
        self.assertEqual(record["centroid_xy"], [49.5, 24.5])
        self.assertEqual(record["bbox_xywh"], [40, 20, 20, 10])
        mask = decode_rle(record["segmentation"])
        self.assertEqual(int(mask.sum()), 200)
        self.assertTrue(mask[20:30, 40:60].all())
        self.assertEqual(int(decode_rle(record["segmentation_local"]).sum()), 200)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report"
            save_analysis(path, image, result, extra)
            self.assertTrue(path.with_suffix(".zip").is_file())
            self.assertTrue((path / "report.html").is_file())
            self.assertIn("centroid_x", (path / "instances.csv").read_text())
            self.assertEqual(json.loads((path / "analysis.json").read_text())["instances"], result["instances"])
            with self.assertRaises(FileExistsError):
                save_analysis(path, image, result, extra)

    def test_empty_prediction_is_valid_not_an_error(self):
        result, _ = analyze(np.zeros((64, 64, 3), np.uint8), lambda _: [], {}, mode="whole_image")
        self.assertEqual(result["summary"]["predicted_count"], 0)
        self.assertIsNone(result["summary"]["mean_complete_mask_area_px"])
        self.assertEqual(result["summary"]["predicted_annotated_region_coverage_percent"], 0)
        self.assertEqual(result["instances"], [])

    def test_decode_rejects_other_formats_and_normalizes_exif(self):
        from PIL import Image
        with self.assertRaises(Exception):
            decode_upload(b"not an image")
        data = io.BytesIO()
        Image.new("RGB", (40, 80)).save(data, format="JPEG", exif=Image.Exif())
        self.assertEqual(decode_upload(data.getvalue()).shape, (80, 40, 3))
        data = io.BytesIO()
        exif = Image.Exif(); exif[274] = 6
        Image.new("RGB", (40, 80)).save(data, format="JPEG", exif=exif)
        self.assertEqual(decode_upload(data.getvalue()).shape, (40, 80, 3))

    def test_upload_requires_order_and_complete_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("model_final.pth", "config.yaml"):
                (root / name).write_text("fixture")
            demo = ScientistWorkbench(root, root)
            start = demo.dispatch("begin", {"size": 6, "name": "test.png"})
            self.assertIn("error", demo.dispatch("chunk", {"id": start["id"], "offset": 1, "data": "YWJj"}))
            self.assertEqual(demo.dispatch("chunk", {"id": start["id"], "offset": 0, "data": "YWJj"})["received"], 3)
            self.assertIn("error", demo.dispatch("run", {"id": start["id"]}))
            self.assertIn("error", demo.dispatch("sample", {"id": "../../secret"}))
            self.assertIn("error", demo.dispatch("begin", {"size": 50 * 1024**2}))
            demo.pool.shutdown()

    def test_embedded_result_cannot_inject_script(self):
        payload = {"filename": "</script><script>alert(1)</script>"}
        html = make_html(result=payload)
        self.assertNotIn(payload["filename"], html)
        self.assertNotIn("__BARNACLE_BOOT__", html)

    def test_background_upload_to_report_uses_selected_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("model_final.pth", "config.yaml"):
                (root / name).write_text("fixture")
            demo = ScientistWorkbench(root, root, save_root=root / "reports")
            _, image_bytes = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
            data = image_bytes.tobytes()
            with patch("scientist_colab.load_predictor", return_value=(lambda _: [],
                    {"max_detections_per_tile": 600, "checkpoint": "fixture.pth"}, [])) as loader:
                for _ in range(2):
                    upload = demo.dispatch("begin", {"size": len(data), "name": "image.png"})
                    demo.dispatch("chunk", {"id": upload["id"], "offset": 0,
                        "data": base64.b64encode(data).decode()})
                    self.assertEqual(demo.dispatch("run", {"id": upload["id"],
                        "mode": "whole_image"})["status"], "running")
                    deadline = time.monotonic() + 10
                    while demo.dispatch("status")["status"] == "running" and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertEqual(demo.dispatch("status")["status"], "complete", demo.state)
                    result = json.loads(demo.dispatch("result_chunk", {"offset": 0})["text"])
                    self.assertEqual(result["filename"], "image.png")
                    self.assertTrue(demo.last_output.with_suffix(".zip").is_file())
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(loader.call_args.args[0], root)
            demo.pool.shutdown()

    def test_demo_notebook_is_clean_and_compiles(self):
        notebook = json.loads((ROOT / "notebooks/barnacle_scientist_demo.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
