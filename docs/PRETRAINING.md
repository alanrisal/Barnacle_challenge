# Region selection and initial segmentation data

The preparation pipeline runs on CPU using OpenCV and NumPy. It does not install
PyTorch, download model weights, or train a model. Run commands from the repository
root. The existing virtual environment already provides the required libraries.

## Prepare the supplied data

```
uv run -m barnacle_challenge prepare-data --output artifacts/pretraining
uv run -m barnacle_challenge validate-data artifacts/pretraining --report artifacts/pretraining/validation_report.json
```

The checked-in [source manifest](../configs/data_sources.json) pairs original
photographs with full-size outline masks. `masked_img*.png` overlays never enter
training inputs. Every build needs a new or empty output directory, so existing
artifacts and manual work cannot be silently replaced.

Default tiles are 640 pixels with 25% overlap, at native resolution. Edge tiles
can be smaller; no pixels are stretched or interpolated. Crop exteriors outside
the polygon are blacked out and accompanied by valid-region masks. Grayscale
label PNGs contain **uint16 instance IDs**, not colors or binary foreground.
Overlapping tiles repeat individuals; do not sum tile counts.

The initial split is img1 for training and img2 for validation. The two unseen
images are inference-only, without invented labels. They share a conservative
group ID because their possible crop relationship has not been established.
Related crops/photos must share `group_id`; preparation rejects groups and exact
source-file hashes occurring in multiple splits. Different encodings or views
of the same photograph still require correct manual grouping. Two annotated
scenes provide limited evidence of generalization; this split is an initial
experiment, not a claim of representative scientific validation.

## Select or correct a region

```powershell
uv run -m barnacle_challenge select-region src/barnacle_challenge/img2.png --output artifacts/region_img2
uv run -m barnacle_challenge select-region src/barnacle_challenge/img2.png --interactive --output artifacts/region_img2_reviewed
uv run -m barnacle_challenge select-region src/barnacle_challenge/unseen_img2.png --whole-image --output artifacts/region_crop
```

The detector uses an HSV mask, morphological closing and the largest enclosed
quadrilateral. It finds an opening rather than the outside of the green frame.
Default HSV bounds are H=75..110, S=55..255, V=20..255. Override with
`--lower-hsv H S V`, `--upper-hsv H S V`, and `--inset PIXELS`, or edit manifest
`region_config`. The inset is measured in original pixels. Automatic selection
always remains unreviewed. Unframed images require explicit whole-image or manual
selection; automatic failure does not silently include the entire photograph.

The interactive editor fits a fixed 1100 x 760 window:

- Left-click four corners, or drag existing corner handles.
- Mouse wheel zooms around the cursor; right-drag pans.
- `F` fits the image; `C` clears corners.
- Enter validates and saves; Escape cancels without saving.

Clicks are transformed to original-image coordinates. Display scaling never
changes crop pixels. Four corners can also be supplied without opening a window:

```powershell
uv run -m barnacle_challenge select-region src/barnacle_challenge/img2.png --corners '[[960,240],[2580,240],[2540,1850],[975,1850]]' --output artifacts/manual_img2
```

`--corners` also accepts a saved `region.json` path. Saved regions are checked
against the source image hash. To use a reviewed selection in preparation, change
the matching manifest entry:

```json
{
  "region_mode": "file",
  "region_file": "artifacts/region_img2_reviewed/region.json"
}
```

Paths inside the manifest are resolved relative to its `root` (the repository
root in the supplied manifest), independent of shell working directory. An inline
manifest region uses `region_mode: "corners"`, `polygon_xy: [[x,y], ...]`, and
`region_reviewed: true` after review. No perspective warp or ruler calibration is
performed here.

## What is generated

```text
artifacts/pretraining/
  dataset_summary.json             counts, provenance, split and review state
  input_manifest.json              exact input configuration
  annotations/{train,val,test}.json
  scenes/<scene>/
    region.json                   original coordinates, bounding crop, source hash
    region_preview.jpg            fit-to-screen original with selected polygon
    image.png                     clean native-resolution ROI crop
    valid_region.png              pixels inside the polygon
    instances.png                 native crop, lossless uint16 instance IDs
    instances_full_canvas.png     labels on original full-image canvas, for edits
    ignore_regions.png            unresolved annotation areas within crop
    ignore_full_canvas.png        same uncertainty on original canvas
    annotation_audit.json         instances and ambiguous outline components
    annotation_overlay.png        full-resolution visual audit
    annotation_preview.jpg        smaller audit preview
    review/                       enlarged clean/outlined issue crops and contact sheets
    metadata.json                 source/crop coordinates and review state
  tiles/<split>/{images,labels,valid,ignore}/
```

Inference-only scenes have clean images and region artifacts but no labels or
training tiles. COCO segmentations use lossless **uncompressed column-major RLE**.
Area and bounding boxes are computed from actual mask pixels, not polygons.
Tile `source_origin_xy` maps tile coordinates back to the original photograph.
`source_instance_id` is stable within a scene across overlapping tiles.

## Annotation extraction and review

The transparent mask canvas must match the original photograph dimensions.
Blue pixels are selected by channel differences, with transparent pixels excluded.
Enclosed interiors become candidate instances. Stroke pixels are assigned up to
the approximately equidistant boundary between the inside and outside of the
stroke, avoiding the systematic inflation of filling its outermost edge.
This is a reproducible reconstruction, not recovery of an unknowable exact
hand-drawn boundary.

Open outlines, small non-enclosing strokes, touching/multiple outlines and
oversized candidates are reported. No gaps are silently repaired. The default
minimum enclosed interior is 12 pixels; stroke components smaller than 8 pixels
are counted as discarded noise. These configurable thresholds are annotation
extraction settings, **not biological size cutoffs**. Tiny real individuals can
be missed and must be checked during review.

Red rectangles in audit overlays identify issues. Enlarged review panels show the
clean photo on the left and supplied strokes on the right. A component with
multiple enclosed interiors can produce several provisional masks and remains
flagged. An open outline produces no guessed mask; its bounding rectangle is
marked uncertain. Centroid-inside-ROI IDs define the provisional count. Visible
mask fragments whose centroid is outside remain labeled as training context,
but are not counted. ROI-clipped instances are flagged for later exclusion from
complete-individual morphometrics.

The supplied outlines are called `supplied_outline_region_anatomy_unconfirmed`
until their anatomical target is confirmed. Do not call their area basal area
or whole-shell cover. Calibration is explicitly absent; no mm or mm² estimates
are fabricated.

To finalize labels, edit a copy of `instances_full_canvas.png` using an instance
label editor (for example, an image-label editor that preserves uint16 IDs). Keep
the original photograph dimensions, background 0, and one unique positive ID per
individual. Do not paint display colors into the label image. The manifest entry
then switches from `outline_mask` to:

```json
{
  "instance_labels": "reviewed/img1_instances.png",
  "annotation_target": "opercular_region",
  "annotations_reviewed": true,
  "annotation_coverage_verified": true
}
```

Use the anatomical name established by the annotation protocol; the example name
is not an anatomical determination. Remove `outline_mask` when adding
`instance_labels`. Both annotated scenes must use the same target in a
single-class dataset. Review the complete ROI for missing individuals as well as
the flagged strokes. Review each region polygon. Rebuild into a fresh directory.

## Verify and consume the data

```powershell
uv run -m barnacle_challenge validate-data artifacts/pretraining
uv run -m barnacle_challenge validate-data artifacts/pretraining --require-reviewed
uv run -m unittest discover -s tests -v
```

Ordinary validation checks image dimensions, label types, lossless COCO/PNG
agreement, areas, boxes, label overlaps, valid-region exclusion, native source
coordinates and split isolation. It reports annotation issues without pretending to judge biological
correctness. The strict command deliberately fails on the initial provisional
data. Training readiness requires reviewed regions, reviewed annotations,
verified annotation coverage, a confirmed target, no unresolved outline issues,
and both train and validation scenes.

```python
from barnacle_challenge.dataset import InstanceDataset

# Explicit opt-in is for inspection of the current provisional data.
dataset = InstanceDataset("artifacts/pretraining", "train", allow_unreviewed=True)
image_rgb, target = dataset[0]
# target: boxes (xyxy), masks (N,H,W), foreground labels (1), area,
# source IDs/origin, valid_region, ignore_regions, image_id, iscrowd.
```

The default reader rejects provisional data. The current ignore maps are review
artifacts; a generic COCO trainer does not automatically honor them. Do not train
on provisional exports by bypassing review. Later training should use the strict
validation gate. The data recipe in [model_data.json](../configs/model_data.json)
documents the Mask R-CNN/PointRend input contract. The separate
[Colab training guide](COLAB_TRAINING.md) describes the executable trainer and its
explicit provisional-data diagnostic mode. GPU training is not locally verified.

COCO IDs are foreground category 1. Detectron2 will map that to contiguous class 0;
torchvision targets use foreground class 1. Framework preprocessing must retain
instance masks, nearest-neighbor label transforms, and image/mask alignment.
