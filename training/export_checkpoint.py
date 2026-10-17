"""Preserve an explicitly selected checkpoint, its config and provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(run, output, checkpoint="model_final.pth"):
    run, output = Path(run).resolve(), Path(output).resolve()
    if checkpoint not in ("model_final.pth", "model_best.pth"):
        raise ValueError("Choose model_final.pth or model_best.pth explicitly")
    required = [checkpoint, "config.yaml", "run_manifest.json"]
    for name in required:
        if not (run / name).is_file():
            raise ValueError(f"Missing required run artifact: {run / name}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Model export directory is not empty; choose a new name")
    if output == run or run.is_relative_to(output) or output.is_relative_to(run):
        raise ValueError("Export into a separate model directory outside the training run")
    output.mkdir(parents=True, exist_ok=True)
    names = required + [name for name in ("environment.json", "validation_metrics.json", "metrics.json",
                                          "live_metrics.jsonl") if (run / name).is_file()]
    for name in names:
        shutil.copy2(run / name, output / name)
    versions = {}
    for name in ("torch", "torchvision", "detectron2", "numpy", "opencv-python", "pycocotools"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    card = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(run),
            "selected_checkpoint": checkpoint, "checkpoint_sha256": file_hash(output / checkpoint),
            "config_sha256": file_hash(output / "config.yaml"), "python": sys.version,
            "export_environment_versions": versions,
            "training_run": json.loads((run / "run_manifest.json").read_text(encoding="utf-8")),
            "scientific_validation_complete": False,
            "notes": ["Full checkpoint is preserved, including any saved optimizer/trainer state.",
                      "Original training files are unchanged.",
                      "No accuracy claim is established by exporting or predicting on unlabeled images."]}
    (output / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    (output / "last_checkpoint").write_text(checkpoint, encoding="utf-8")
    with zipfile.ZipFile(output / "model_bundle.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name in names + ["model_card.json", "last_checkpoint"]:
            archive.write(output / name, name)
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=("model_final.pth", "model_best.pth"), default="model_final.pth")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.run_dir, args.output, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
