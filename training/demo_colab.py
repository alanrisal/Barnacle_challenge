"""Notebook entry point: snapshot a chosen model, then report on unseen images."""

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys


def streamed_run(command, log_path):
    with Path(log_path).open("x", encoding="utf-8") as log:
        with subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Demo command failed with exit code {return_code}; full output: {log_path}")


def run_unseen_demo(workspace, trained_run, *, checkpoint_name="model_final.pth",
                    score_threshold=0.3, save_root=None):
    workspace, trained_run = Path(workspace).resolve(), Path(trained_run).resolve()
    if not (trained_run / checkpoint_name).is_file():
        raise ValueError(f"Selected checkpoint does not exist: {trained_run / checkpoint_name}")
    inputs = workspace / "demo/demo_inputs.json"
    if not inputs.is_file():
        raise ValueError("Upload/extract barnacle_unseen_demo.zip into WORKSPACE first")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root = Path(save_root) if save_root else trained_run.parent / f"unseen_demo_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    model_snapshot, reports = root / "model_snapshot", root / "reports"
    print("Selected trained checkpoint:", trained_run / checkpoint_name, flush=True)
    print("Persistent demo outputs:", root, flush=True)
    # Preserve the trained model before starting inference.
    streamed_run([sys.executable, workspace / "training/export_checkpoint.py",
                  "--run-dir", trained_run, "--checkpoint", checkpoint_name,
                  "--output", model_snapshot], root / "model_export.log")
    streamed_run([sys.executable, workspace / "training/predict_unseen.py",
                  "--run-dir", model_snapshot, "--checkpoint", checkpoint_name,
                  "--inputs", inputs, "--output", reports,
                  "--score-threshold", str(score_threshold), "--compare-thresholds", "0.2", "0.3", "0.5"],
                 root / "inference.log")
    shutil.make_archive(str(root / "unseen_reports"), "zip", reports)
    summary = json.loads((reports / "demo_summary.json").read_text(encoding="utf-8"))
    from IPython.display import HTML, Image, display
    for item in summary["reports"]:
        rows = "".join(f"<tr><td>{row['score_threshold']}</td><td>{row['deduplicated_count']}</td>"
                       f"<td>{row['duplicates_suppressed']}</td></tr>" for row in item["threshold_comparison"])
        display(HTML(f"<h3>{html.escape(item['image_id'])}: {item['predicted_count']} predicted instances "
                     f"at score ≥ {item['score_threshold']}</h3><table><tr><th>Score ≥</th>"
                     f"<th>Predicted count</th><th>Duplicates suppressed</th></tr>{rows}</table>"))
        display(Image(filename=str(reports / item["image_id"] / "report.png"), width=1200))
    print("Model archive:", model_snapshot / "model_bundle.zip")
    print("Report archive:", root / "unseen_reports.zip")
    print("Offline report:", reports / "report.html")
    return root
