# Training Set up 

Open [barnacle_training_colab_final.ipynb](../notebooks/barnacle_training_colab_final.ipynb)
in Colab: File --> upload then select runtime (T4)

## Local files to upload

Init bundle is `artifacts/barnacle_colab_final.zip`. Notebook cell will prmpt the uplaod. You do not need to upload the entire repository, install
the local Python package in Colab, or move your Windows virtual environment.
The ZIP contains prepared data, preparation/validation modules, current training
and resume scripts, live loss plots, the fixed wheel installer, and both unseen
images with demo/report code. No separate resume or demo patch uploads are needed.
It also includes the clean final notebook and guides. Your original notebook with
saved training results is preserved. Model weights remain in your selected Drive
run; they are not included in this upload. The package's Python >=3.13 install metadata is not used there.
The scripts use Python 3.10+ syntax and must pass their runtime checks in Colab.

Regenerate a bundle after changing labels or training scripts:

```powershell
.\.venv\Scripts\python.exe tools/package_colab.py --dataset artifacts/pretraining --output artifacts/barnacle_colab_updated.zip
```

Archives and run outputs refuse to overwrite existing files/directories. Use a new
name when rebuilding or rerunning a completed cell. Re-running the upload cell
needs a fresh `EXTRACT_ROOT`; re-running training needs a fresh run output path.

## Setup and first diagnostic

1. Verify the notebook sees an NVIDIA GPU and CUDA-enabled PyTorch.
2. Upload/extract the bundle and validate its masks and scene separation.
3. Run the installation cell. It checks the existing torch/torchvision CUDA ops,
   then compiles Detectron2 with PointRend for the runtime GPU. It does not blindly
   replace Colab's PyTorch/CUDA installation. Major CUDA-version mismatches stop
   with an explanation; minor mismatches are left to the extension build checks.
   Installation uses a regular setuptools wheel build, not legacy `setup.py develop`.
   Verbose compilation output is saved to `/content/detectron2_build.log`.
4. Mount Drive for persistent checkpoint storage (default), or choose local runtime
   storage and download checkpoints before disconnecting.
5. Run the 50-step single-tile diagnostic. Check finite losses, a saved checkpoint,
   and the labeled ground-truth/model-prediction comparison panels.

The first run deliberately uses `ALLOW_PROVISIONAL=True` in the notebook. This
permits an exploratory diagnostic with the current unreviewed labels. Unresolved
outline regions are **not automatically ignored** by the generic Detectron2
trainer. Their missing/ambiguous annotations can affect learning. This is not a
scientifically validated experiment. Set the flag to False after replacing the
dataset with a reviewed build; the trainer then enforces dataset readiness.
No review flags are automatically changed or declared complete by training.

The smoke run repeats one tile and disables augmentation. It tests the training
plumbing, not generalization. Fifty steps may be insufficient even to fit that
tile well. Inspect loss and predictions before increasing the run length.

## Model and settings

- Model: Mask R-CNN R50-FPN + PointRend initialized from the official COCO-trained
  PointRend checkpoint, with one foreground class in both detection and point heads.
- Batch: one image per iteration, mixed precision; zero data-loader workers.
- Preserve native tile pixels. Training flips and modest brightness/contrast
  augmentations are enabled only for the multi-tile fine-tuning run.
- Anchor sizes: 8, 16, 32, 64, 128 pixels across FPN levels, as an initial setting
  for the observed small annotations; these settings still need empirical tuning.
- Proposal limits: 3,000 after RPN NMS in training, 2,000 at inference.
- Final detection cap: 600, exceeding the observed maximum 396 ground-truth
  labels/tile. Evaluation uses the same custom cap and is not standard COCO AP@100.
- Initial learning rate: 0.00025. Smoke: 50 iterations. Optional fine-tuning:
  600 iterations, validation every 100. These are starting experiment settings,
  not a claim of convergence or an optimized schedule.

Changing 80 COCO classes to one causes incompatible class-specific output weights
to be reinitialized. Such checkpoint warnings are expected; widespread missing
backbone weights or CUDA errors are not. The installation uses a Detectron2 source
checkout and captures its commit plus torch/CUDA versions in `environment.json`.
Use that commit as `DETECTRON2_REVISION` to reproduce a successful environment.

## Longer run and outputs

After inspecting the smoke run, change `RUN_FINETUNE=False` to True. This creates
a separate run initialized from COCO weights and trains on all nine training
tiles, validating on the 16 validation tiles. It does not resume the one-tile
diagnostic. Training and validation still contain only one source photograph each.

Each run records:

- `config.yaml`, `run_manifest.json`, `environment.json`: settings, data status,
  initialization, runtime and source version.
- `metrics.json`: logged losses and periodic validation metrics.
- `model_final.pth`: final checkpoint; fine-tuning also saves `model_best.pth`
  according to validation mask AP.
- `validation_metrics.json`: final fine-tuning metrics (not generated for smoke).
- `previews/*_comparison.jpg`: supplied labels on the left, actual trained-model
  predictions on the right; clearly marked on a separate banner.
- `previews/*_predictions.json`: scores and lossless masks in native tile
  coordinates, at the explicit preview threshold (default 0.5).

Evaluation uses a lower prediction-score threshold (0.05) to compute AP. A blank
preview at 0.5 does not establish that the model produced no lower-score candidates.
Do not sum counts across overlapping tiles. Whole-photograph inference, duplicate
reconciliation, calibrated scientific measurements and independent validation
remain separate work after the training path runs successfully.

## Continue an existing checkpoint with a live plot

The trainer now accepts `--resume-from /path/to/model_final.pth`. It restores model,
optimizer, AMP trainer state, iteration counter, and available scheduler state.
The original run's `config.yaml` and `run_manifest.json` must sit beside the
checkpoint. Resume keeps the same dataset, mode and diagnostic tile; it writes
to a **new output directory** and preserves the original checkpoint and logs.
It does not promise bit-identical data-order/RNG replay.

`--steps` is the final total, not the number of additional steps. A checkpoint
with iteration 199 has completed 200 steps. `--steps 400` adds 200 steps;
`--steps 600` adds 400. The trainer rejects totals already reached.

The combined final bundle already includes these scripts. For an older Colab
session using the original bundle, upload the updated `training/train_pointrend.py`
and `training/live_colab.py` into `WORKSPACE / "training"`. No dataset re-upload
or Detectron2 reinstall is needed. The updated notebook has a resume cell and
uses the live plot for fresh training too.

`live_colab.run_with_live_plot(command, output)` launches the trainer as a
subprocess while plotting `live_metrics.jsonl`, updated every five steps by
default. It shows absolute step count, progress, ETA, total loss and individual
losses. Full console output is saved beside the run as `<run>.console.log`.
The graph starts at the resumed step; the original run's history stays in its
original directory. Checkpoints still record optimizer/trainer state.

To move from the single-tile diagnostic to all training tiles, use a new
`--mode finetune` run with `--weights /path/to/200_step/model_final.pth`, rather
than `--resume-from`. That retains learned model weights but starts a fresh
optimizer and step counter for the new training procedure. For example,
`--steps 600` then means 600 additional fine-tuning steps on all training tiles.
The updated notebook sets `LATEST_OUTPUT` after each successful run; use it for
subsequent prediction comparisons to avoid accidentally loading an older model.

## Troubleshooting

- No CUDA GPU: select/reconnect to a GPU runtime before the installation cell.
- Build failure: keep the first relevant compiler error and the printed torch,
  torchvision, nvcc and GPU versions. Colab runtime images change; send these logs
  before changing package versions at random.
- Out of memory: keep batch size 1, stop other GPU workloads, and report the
  failing stage. Reducing sampled ROIs or PointRend inference work is preferable
  to silently shrinking the source images; the detection cap must stay above the
  true instance count for the evaluated tiles.
- Dataset readiness failure: either use reviewed/corrected labels, or retain the
  explicit provisional diagnostic opt-in and treat results accordingly.
- Stale output directory: select a new run name; checkpoints are never deleted by
  the scripts.

Primary references: [PointRend implementation and pretrained models](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend),
[Detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
