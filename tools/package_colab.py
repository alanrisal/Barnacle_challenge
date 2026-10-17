"""Create a portable upload containing prepared data and training code."""

import argparse
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
    output.parent.mkdir(parents=True, exist_ok=True)
    # PNG/JPEG are already compressed; storing avoids unnecessary packaging time.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, name in files:
            archive.write(path, name.as_posix())
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/pretraining", type=Path)
    parser.add_argument("--output", default="artifacts/barnacle_colab.zip", type=Path)
    args = parser.parse_args()
    count = package(args.dataset.resolve(), args.output.resolve(), Path(__file__).resolve().parents[1])
    print(f"Packaged {count} files: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
