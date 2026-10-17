"""Launch the training subprocess with an updating notebook plot and progress bar."""

import html
import json
from pathlib import Path
import signal
import subprocess
import time


def read_live_metrics(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A writer may still be finishing the last line.
        if isinstance(row, dict) and "step" in row and "total_steps" in row:
            rows.append(row)
    return rows


def run_with_live_plot(command, output, *, refresh_seconds=3):
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output folder to preserve the existing run")
    log_path = output.parent / f"{output.name}.console.log"
    status = display(HTML("Starting model/data loading…"), display_id=True)
    plot = display(HTML("Waiting for the first live loss measurements…"), display_id=True)
    history = []
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                rows = read_live_metrics(output / "live_metrics.jsonl")
                if rows and (not history or rows[-1]["step"] != history[-1]["step"]):
                    history = rows
                    latest = rows[-1]
                    step, total = latest["step"], latest["total_steps"]
                    loss = latest.get("total_loss")
                    loss_text = f"{loss:.4f}" if loss is not None else "pending"
                    eta = latest.get("eta_seconds", 0)
                    phase = "Finishing evaluation/previews" if step >= total else "Training"
                    status.update(HTML(f"<b>{phase}: {step}/{total} total steps</b> "
                        f"(started at {latest.get('starting_step', 0)}) | Loss {loss_text} | ETA ~{eta / 60:.1f} min"
                        f"<br><progress value='{step}' max='{total}' style='width:95%'></progress>"))
                    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
                    xs = [row["step"] for row in rows]
                    axes[0].plot(xs, [row.get("total_loss", float("nan")) for row in rows])
                    axes[0].set_title("Total loss (raw, every few steps)")
                    keys = sorted({key for row in rows for key in row if key.startswith("loss")})
                    for key in keys:
                        axes[1].plot(xs, [row.get(key, float("nan")) for row in rows], label=key)
                    if keys:
                        axes[1].legend(fontsize=8)
                    axes[1].set_title("Individual loss components")
                    for axis in axes:
                        axis.set_xlabel("Total training step")
                        axis.set_ylabel("Loss")
                        axis.grid(alpha=0.25)
                    fig.tight_layout()
                    plot.update(fig)
                    plt.close(fig)
                if process.poll() is not None:
                    break
                time.sleep(max(0.2, refresh_seconds))
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            raise
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
    if process.returncode:
        status.update(HTML(f"<b>Training stopped with exit code {process.returncode}</b><pre>{html.escape(tail)}</pre>"))
        raise RuntimeError(f"Training failed; complete log: {log_path}")
    status.update(HTML(f"<b>Finished.</b> Outputs: {html.escape(str(output))}"))
    print(tail)
    return history
