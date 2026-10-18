"""Create one Colab upload with data, current training/resume code and unseen demo."""

import argparse
import json
from pathlib import Path
import zipfile


def package(dataset: Path, output: Path, repository: Path) -> int:
    if not (dataset / "dataset_summary.json").is_file():
        raise ValueError("Prepare the dataset first; dataset_summary.json is missing")
    if output.exists():
        raise ValueError(f"Archive already exists: {output}; choose a new filename")
    files = [(p, Path("barnacle_training/dataset") / p.relative_to(dataset))
             for p in dataset.rglob("*") if p.is_file()]
    for folder in ("src/barnacle_challenge", "training"):
        files.extend((p, Path("barnacle_training") / p.relative_to(repository))
                     for p in (repository / folder).rglob("*.py"))
    files.extend((p, Path("barnacle_training") / p.relative_to(repository))
                 for p in (repository / "training").glob("*.html"))
    for extra in ("notebooks/barnacle_scientist_demo.ipynb", "docs/SCIENTIST_DEMO.md"):
        if (repository / extra).is_file():
            files.append((repository / extra, Path("barnacle_training") / extra))
    manifest_path = repository / "configs/unseen_demo.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for scene in manifest["images"]:
        source = (manifest_path.parent / scene["path"]).resolve()
        name = f"inputs/{scene['id']}{source.suffix}"
        scene["path"] = name
        files.append((source, Path("barnacle_training/demo") / name))
    notebook = repository / "notebooks/barnacle_training_colab_final.ipynb"
    if not notebook.is_file():
        notebook = repository / "notebooks/barnacle_training_colab_empty.ipynb"
    files.append((notebook, Path("barnacle_training/notebooks") / notebook.name))
    for name in ("COLAB_TRAINING.md", "UNSEEN_DEMO.md"):
        files.append((repository / "docs" / name, Path("barnacle_training/docs") / name))
    names = [name.as_posix() for _, name in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate paths in upload bundle")
    for source, _ in files:
        if not source.is_file():
            raise ValueError(f"Bundle input is missing: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # PNG/JPEG are already compressed; storing avoids unnecessary packaging time.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, name in files:
            archive.write(path, name.as_posix())
        archive.writestr("barnacle_training/demo/demo_inputs.json", json.dumps(manifest, indent=2))
        archive.writestr("barnacle_training/bundle_manifest.json", json.dumps({
            "format_version": 1,
            "includes": ["prepared_dataset", "training_resume", "live_training_plots",
                         "wheel_installer", "unseen_demo", "clean_notebook"],
            "model_weights_included": False,
            "notebook": "notebooks/" + notebook.name,
        }, indent=2))
    return len(files) + 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/pretraining", type=Path)
    parser.add_argument("--output", default="artifacts/barnacle_colab_final.zip", type=Path)
    args = parser.parse_args()
    count = package(args.dataset.resolve(), args.output.resolve(), Path(__file__).resolve().parents[1])
    print(f"Packaged {count} files: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
