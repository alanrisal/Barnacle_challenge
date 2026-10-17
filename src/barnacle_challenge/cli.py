import argparse
import json
from pathlib import Path
import sys

import cv2

from .dataset import prepare_dataset
from .io import read_image, read_json, sha256, write_image, write_json
from .regions import (RegionConfig, RegionNotFound, detect_region, manual_region,
                      region_mask, region_overlay, whole_image_region)
from .validation import validate_dataset


def _region_command(args):
    image = read_image(args.image)
    if args.whole_image:
        region = whole_image_region(image)
    elif args.corners:
        value = args.corners.strip()
        points = json.loads(value) if value.startswith("[") else read_json(value)
        if isinstance(points, dict):
            if "source_sha256" in points and points["source_sha256"] != sha256(args.image):
                raise ValueError("Saved corners belong to a different source image")
            points = points["polygon_xy"]
        region = manual_region(image, points)
    else:
        config = RegionConfig(lower_hsv=tuple(args.lower_hsv), upper_hsv=tuple(args.upper_hsv),
                              inset_px=args.inset)
        try:
            region = detect_region(image, config)
        except RegionNotFound:
            if not args.interactive:
                raise
            region = None
    if args.interactive:
        from .viewer import select_corners

        region = select_corners(image, region)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new output directory.")
    region["source_image"] = str(Path(args.image).resolve())
    region["source_sha256"] = sha256(args.image)
    mask = region_mask(region)
    x, y, w, h = region["crop_xywh"]
    crop = image[y:y + h, x:x + w].copy()
    crop[mask[y:y + h, x:x + w] == 0] = 0
    write_json(output / "region.json", region)
    write_image(output / "region_mask.png", mask)
    write_image(output / "crop.png", crop)
    write_image(output / "preview.jpg", region_overlay(image, region))
    print(json.dumps({"output": str(output), "crop_xywh": region["crop_xywh"],
                      "reviewed": region["reviewed"]}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Barnacle region selection and segmentation data preparation")
    commands = parser.add_subparsers(dest="command")
    select = commands.add_parser("select-region", help="Find or manually select the central counting region")
    select.add_argument("image")
    select.add_argument("--output", required=True, help="New directory for crop, polygon, mask and preview")
    mode = select.add_mutually_exclusive_group()
    mode.add_argument("--whole-image", action="store_true", help="Explicitly use an already cropped image")
    mode.add_argument("--corners", help="JSON [[x,y],...] or path to region JSON; original-image pixels")
    select.add_argument("--interactive", action="store_true", help="Edit four corners with zoom and pan")
    select.add_argument("--lower-hsv", nargs=3, type=int, default=[75, 55, 20], metavar=("H", "S", "V"))
    select.add_argument("--upper-hsv", nargs=3, type=int, default=[110, 255, 255], metavar=("H", "S", "V"))
    select.add_argument("--inset", type=float, default=2.0, help="Conservative inward offset in source pixels")
    select.set_defaults(func=_region_command)
    prepare = commands.add_parser("prepare-data", help="Extract instance masks and export scene-separated COCO tiles")
    prepare.add_argument("--manifest", default="configs/data_sources.json")
    prepare.add_argument("--output", required=True, help="New dataset directory; existing artifacts are never deleted")
    prepare.add_argument("--tile-size", type=int, default=640)
    prepare.add_argument("--overlap", type=float, default=0.25)
    prepare.set_defaults(func=lambda args: print(json.dumps(prepare_dataset(
        args.manifest, args.output, tile_size=args.tile_size, overlap=args.overlap), indent=2)))
    validate = commands.add_parser("validate-data", help="Check lossless masks, geometry and split isolation")
    validate.add_argument("dataset")
    validate.add_argument("--require-reviewed", action="store_true", help="Fail for provisional scientific labels")
    validate.add_argument("--report", help="Save the validation result as JSON")
    validate.set_defaults(func=_validate_command)
    return parser


def _validate_command(args):
    result = validate_dataset(args.dataset, require_reviewed=args.require_reviewed)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
