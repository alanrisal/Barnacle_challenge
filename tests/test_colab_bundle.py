import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_colab", ROOT / "tools/package_colab.py")
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)

setup_spec = importlib.util.spec_from_file_location("colab_setup", ROOT / "training/colab_setup.py")
setup = importlib.util.module_from_spec(setup_spec)
setup_spec.loader.exec_module(setup)

trainer_spec = importlib.util.spec_from_file_location("train_pointrend", ROOT / "training/train_pointrend.py")
trainer = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(trainer)
live_spec = importlib.util.spec_from_file_location("live_colab", ROOT / "training/live_colab.py")
live = importlib.util.module_from_spec(live_spec)
live_spec.loader.exec_module(live)


class ColabBundleTests(unittest.TestCase):
    def test_resume_continues_after_completed_step_and_rejects_weights_only(self):
        checkpoint = {"iteration": 199, "trainer": {"_trainer": {"optimizer": {}}}}
        self.assertEqual(trainer.resume_start(checkpoint, 400), 200)
        with self.assertRaisesRegex(ValueError, "already completed"):
            trainer.resume_start(checkpoint, 200)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            trainer.resume_start({"model": {}}, 400)

    def test_live_log_tolerates_incomplete_write_and_preserves_global_step(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "live_metrics.jsonl"
            path.write_text('{"step":205,"total_steps":400,"total_loss":1.2}\n{"step":')
            rows = live.read_live_metrics(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 205)
            self.assertEqual(rows[0]["total_steps"], 400)
            self.assertEqual(live.read_live_metrics(Path(folder) / "not_written_yet"), [])

    def test_install_uses_wheel_backend_and_preserves_existing_build_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            with patch.object(setup, "run") as run:
                setup.install_detectron2(checkout, {"MAX_JOBS": "2"})
            arguments = run.call_args.args
            self.assertNotIn("-e", arguments)
            self.assertIn("--no-build-isolation", arguments)
            self.assertIn("--verbose", arguments)
            self.assertIn("setuptools.build_meta", (checkout / "pyproject.toml").read_text())
            (checkout / "pyproject.toml").write_text("# preserve existing configuration")
            setup.configure_wheel_build(checkout)
            self.assertEqual((checkout / "pyproject.toml").read_text(), "# preserve existing configuration")

    def test_failed_command_preserves_underlying_output(self):
        import sys
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "build.log"
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                setup.run(sys.executable, "-c", "print('underlying build error'); raise SystemExit(7)", log_path=log)
            self.assertIn("underlying build error", log.read_text())

    def test_notebook_code_cells_compile_and_have_no_saved_outputs(self):
        notebook = json.loads((ROOT / "notebooks/barnacle_training_colab_empty.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"colab-cell-{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertEqual(code.count("files.upload()"), 1)
        self.assertNotIn("training_resume_patch.zip", code)
        self.assertIn('LATEST_OUTPUT = NEW_OUTPUT', code)
        self.assertIn('LATEST_OUTPUT = FINETUNE_OUTPUT', code)
        self.assertIn('LATEST_OUTPUT = SMOKE_OUTPUT', code)

    def test_upload_bundle_contains_resume_data_and_portable_unseen_demo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset_summary.json").write_text("{}")
            repository = root / "repo"
            source = repository / "src/barnacle_challenge"
            source.mkdir(parents=True)
            (source / "io.py").write_text("# test source")
            (source / "original.png").write_bytes(b"do not bundle source photos")
            training = repository / "training"
            training.mkdir()
            (training / "train_pointrend.py").write_text("# test trainer")
            (training / "live_colab.py").write_text("# test live plot")
            configs = repository / "configs"
            configs.mkdir()
            (source / "unseen.png").write_bytes(b"unseen demo image")
            (configs / "unseen_demo.json").write_text(json.dumps({"images": [
                {"id": "unseen_img1", "path": "../src/barnacle_challenge/unseen.png", "region_mode": "auto"}
            ]}))
            (repository / "notebooks").mkdir()
            (repository / "notebooks/barnacle_training_colab_final.ipynb").write_text("{}")
            (repository / "docs").mkdir()
            for name in ("COLAB_TRAINING.md", "UNSEEN_DEMO.md"):
                (repository / "docs" / name).write_text("# test guide")
            archive = root / "upload.zip"
            packaging.package(dataset, archive, repository)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                names = set(bundle.namelist())
                self.assertIn("barnacle_training/dataset/dataset_summary.json", names)
                self.assertIn("barnacle_training/training/train_pointrend.py", names)
                self.assertIn("barnacle_training/src/barnacle_challenge/io.py", names)
                self.assertEqual(bundle.read("barnacle_training/training/live_colab.py"), b"# test live plot")
                demo = json.loads(bundle.read("barnacle_training/demo/demo_inputs.json"))
                for scene in demo["images"]:
                    self.assertEqual(bundle.read("barnacle_training/demo/" + scene["path"]), b"unseen demo image")
                self.assertFalse(json.loads(bundle.read("barnacle_training/bundle_manifest.json"))["model_weights_included"])
                self.assertFalse(any(name.endswith("original.png") for name in names))
            with self.assertRaises(ValueError):
                packaging.package(dataset, archive, repository)


if __name__ == "__main__":
    unittest.main()
