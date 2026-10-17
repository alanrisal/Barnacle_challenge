# Unseen-image notebook demo

Use the existing Colab T4 session where Detectron2/PointRend already works. No
retraining or reinstall is needed. This runs the trained model on the two actual
unseen originals; it does not import supplied annotation masks as predictions.

## Use the combined final bundle

For a fresh session, open `notebooks/barnacle_training_colab_final.ipynb` and upload
`artifacts/barnacle_colab_final.zip` in the first upload cell. It already contains
the two unseen images, demo code, resume support and live plots. Enable the optional
unseen-demo cell when ready; no second upload is needed.

## Add the demo to an older session

Download `artifacts/barnacle_unseen_demo.zip` from the workspace. This small bundle
includes the two unseen original photographs, portable input metadata, and updated
Python code. It excludes training data, virtual environments and model weights.
Upload/extract it using this cell:

```python
from google.colab import files
from pathlib import Path
import io, zipfile

uploaded = files.upload()  # Select barnacle_unseen_demo.zip
name = next(name for name in uploaded if name.endswith(".zip"))
with zipfile.ZipFile(io.BytesIO(uploaded[name])) as archive:
    for member in archive.infolist():
        relative = Path(member.filename)
        assert relative.parts[0] in {"training", "src", "demo"}
        destination = (WORKSPACE / relative).resolve()
        assert destination.is_relative_to(WORKSPACE.resolve())
        assert (member.external_attr >> 16) & 0o170000 != 0o120000
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
```

Then explicitly choose the trained run you want to preserve and evaluate:

```python
import sys
sys.path.insert(0, str(WORKSPACE / "training"))
from demo_colab import run_unseen_demo

TRAINED_RUN = LATEST_OUTPUT  # Or Path("/content/drive/MyDrive/.../your_actual_run")
CHECKPOINT_NAME = "model_final.pth"  # Latest/final state in that run
SCORE_THRESHOLD = 0.3

DEMO_ROOT = run_unseen_demo(
    WORKSPACE, TRAINED_RUN,
    checkpoint_name=CHECKPOINT_NAME,
    score_threshold=SCORE_THRESHOLD,
)
```

The exact checkpoint path is printed. `model_best.pth` is an explicit alternative
if you want the best validation mask-AP checkpoint rather than the final training
state. A frozen model snapshot is exported **before** inference, and reports use
that snapshot. If the run lives on Drive, its sibling demo folder lives on Drive
as well. With local `/content` storage, download the archives before disconnecting.

The updated main notebook includes these optional demo cells. Set
`RUN_UNSEEN_DEMO=True` to use them, after choosing the desired trained run.

## Image handling

- `unseen_img1.png`: HSV/geometry selection of the central opening, native crop,
  original-coordinate polygon and source-image preview. Auto selection remains
  visually reviewable; it is not silently marked reviewed.
- `unseen_img2.png`: the entire 500 x 500 image, explicitly treated as pre-cropped.
- Inference: native-resolution 640-pixel tiles with 25% overlap; no forced upscaling
  or perspective warp. Model/config match the selected trained checkpoint.
- Thresholds: candidate inference at the minimum requested score, then separate
  reconciliation at 0.2, 0.3 and 0.5 (plus any selected primary threshold).

## Mask reconciliation and counting

Tile masks are projected into crop coordinates. Candidates are restricted to the
ROI by centroid and mask clipping. Duplicate matching uses mask intersection, not
just overlapping boxes. The default cross-tile IoU threshold is 0.5. Tile-edge
fragments can match by intersection over smaller mask (0.85) with a center-distance
check; same-tile duplicates require stricter IoU. Intact-context predictions take
priority over tile-edge fragments, then confidence. Selected mask boundaries are
preserved, rather than averaged or smoothed. Neighboring nonoverlapping objects
remain separate. These are heuristics, not a guarantee of perfect deduplication.

All suppression decisions and raw tile masks are saved for inspection and later
threshold experiments. Partial ROI/tile-edge predictions are flagged; they remain
in the predicted count but are excluded from complete-size summaries. Occlusion
and biological completeness are not otherwise established automatically.

## Outputs

```text
unseen_demo_<timestamp>/
  model_snapshot/
    model_final.pth (or explicitly selected model_best.pth)
    config.yaml, run_manifest.json, model_card.json
    model_bundle.zip
    optional environment/metrics files
  model_export.log
  inference.log
  reports/
    model_provenance.json           checkpoint/config hashes and actual saved iteration
    demo_summary.json
    report.html                    standalone offline report; keep images beside it
    unseen_img1/ and unseen_img2/
      region.json, region_preview.jpg, crop.png, valid_region.png
      masks_overlay.png            native crop with masks, no boxes or repeated labels
      source_masks_overlay.png     masks placed on original full photograph
      report.png                   crop, prediction overlay and distributions
      summary.json                 primary count, threshold sensitivity, limitations
      instances.json               lossless per-instance masks on ORIGINAL image canvas
      instances.csv                scores, pixel areas, centroids and boundary flags
      instance_id_map.png           display label map; overlapping pixels use highest score
      overlapping_pixels.png       where retained instance masks overlap
      raw_tile_predictions.json    sparse raw masks, scores, tile coordinates and audit
      duplicate_decisions.json     suppressed IDs and matching measurements
  unseen_reports.zip
```

The authoritative per-instance masks are in `instances.json` as full-canvas COCO
RLE; separate masks preserve any overlaps. The display label PNG can only assign
one ID per pixel and is not the measurement source of truth. Coverage uses the
union of masks to avoid double-counting shared pixels.

Reports provide predicted counts, union coverage, eligible pixel-area summaries,
confidence distribution, and threshold sensitivity. These are **predictions on
unlabeled inputs**: no AP, count accuracy, species, alive/dead classification, or
physical area is inferred. Learned masks follow the supplied training annotation
target; do not call them calibrated whole-shell/basal areas. The two unseen images
may be related crops and are not claimed to be independent validation scenes.
An output-cap counter warns if the model returns its maximum allowed detections
for any tile. Native overlays and saved duplicate decisions should be inspected.

## Rebuild the demo bundle locally

```powershell
.\.venv\Scripts\python.exe tools/package_unseen_demo.py --output artifacts/barnacle_unseen_demo_updated.zip
```

CPU-only crop verification (no model predictions):

```powershell
.\.venv\Scripts\python.exe training/predict_unseen.py --inputs configs/unseen_demo.json --output artifacts/unseen_crop_check_new --prepare-only
```

Model GPU inference must be exercised in Colab with your actual checkpoint.
Local tests cover mask projection, seams/duplicates, overlap preservation and
byte-for-byte checkpoint export, not the trained model's scientific accuracy.
