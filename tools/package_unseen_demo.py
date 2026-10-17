"""Bundle unseen originals and demo code for an existing Colab training session."""

import argparse
import json
from pathlib import Path
import zipfile


def package_demo(repository, output):
    repository, output = Path(repository), Path(output)
    if output.exists():
        raise ValueError("Demo archive exists; choose a new filename")
    manifest_path = repository / "configs/unseen_demo.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = []
    for scene in manifest["images"]:
        source = (manifest_path.parent / scene["path"]).resolve()
        if not source.is_file():
            raise ValueError(f"Missing unseen original: {source}")
        name = f"inputs/{scene['id']}.png"
        scene["path"] = name
        assets.append((source, f"demo/{name}"))
    for folder in ("training", "src/barnacle_challenge"):
        assets.extend((path, path.relative_to(repository).as_posix())
                      for path in (repository / folder).rglob("*.py"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, name in assets:
            archive.write(path, name)
        archive.writestr("demo/demo_inputs.json", json.dumps(manifest, indent=2) + "\n")
    return output.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/barnacle_unseen_demo.zip", type=Path)
    args = parser.parse_args()
    size = package_demo(Path(__file__).resolve().parents[1], args.output)
    print(f"Created {args.output}: {size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
