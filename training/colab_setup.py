"""Install PointRend against the CUDA-enabled PyTorch already in Colab.

Run this as a separate Python process so newly installed modules are imported
fresh. Nothing in this script is needed by the local preparation environment.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def run(*args, log_path=None, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    command = list(map(str, args))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs)
    log = Path(log_path).open("w", encoding="utf-8") if log_path else None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if log:
                log.write(line)
        return_code = process.wait()
    finally:
        process.stdout.close()
        if log:
            log.close()
    if return_code:
        detail = f" Full output: {log_path}" if log_path else ""
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}.{detail}")
    return return_code


def configure_wheel_build(checkout: Path) -> None:
    """Use setuptools' wheel backend instead of the legacy editable/develop path."""
    pyproject = checkout / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(
            '[build-system]\nrequires = ["setuptools>=64", "wheel"]\n'
            'build-backend = "setuptools.build_meta"\n', encoding="utf-8")


def install_detectron2(checkout: Path, environment: dict) -> None:
    configure_wheel_build(checkout)
    run(sys.executable, "-m", "pip", "install", "--no-build-isolation", "--verbose",
        checkout, env=environment, log_path=checkout.parent / "detectron2_build.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", default="/content/detectron2")
    parser.add_argument("--revision", default="main", help="Use a saved commit hash to reproduce an environment")
    args = parser.parse_args()
    import torch
    import torchvision
    from torch.utils.cpp_extension import CUDA_HOME

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable. In Colab select Runtime > Change runtime type > T4 GPU.")
    if not CUDA_HOME or not (Path(CUDA_HOME) / "bin/nvcc").exists():
        raise RuntimeError("CUDA compiler nvcc is missing. Use a standard Colab GPU runtime with a CUDA toolkit.")
    compiler = subprocess.check_output([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", compiler)
    if not match or int(match[1]) != int(torch.version.cuda.split(".")[0]):
        raise RuntimeError(f"PyTorch CUDA {torch.version.cuda} and CUDA compiler disagree:\n{compiler}\n"
                           "Use matching PyTorch/toolkit versions before compiling Detectron2.")
    if f"{match[1]}.{match[2]}" != torch.version.cuda:
        print(f"CUDA minor versions differ: compiler {match[1]}.{match[2]}, torch {torch.version.cuda}. "
              "The extension build will check compatibility.", flush=True)
    # Exercise torchvision's compiled CUDA ops before building another extension.
    boxes = torch.tensor([[0., 0., 10., 10.]], device="cuda")
    torchvision.ops.nms(boxes, torch.ones(1, device="cuda"), 0.5)
    run(sys.executable, "-m", "pip", "install", "ninja", "pycocotools", "wheel", "setuptools<81")
    checkout = Path(args.checkout)
    if not checkout.exists():
        run("git", "clone", "https://github.com/facebookresearch/detectron2.git", checkout)
    if not (checkout / ".git").is_dir():
        raise RuntimeError(f"Checkout is not a git repository: {checkout}")
    run("git", "-C", checkout, "checkout", args.revision)
    environment = os.environ.copy()
    environment["MAX_JOBS"] = "2"
    major, minor = torch.cuda.get_device_capability()
    environment["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    install_detectron2(checkout, environment)
    # Import/compiled-extension checks must happen in a new process.
    run(sys.executable, "-c",
        "import detectron2; from detectron2 import _C; "
        "from detectron2.projects.point_rend import add_pointrend_config; "
        "print('Detectron2, compiled extension, and PointRend imports passed')")
    provenance = {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "torch_cuda": torch.version.cuda, "cuda_compiler": compiler,
                  "gpu": torch.cuda.get_device_name(), "device_capability": [major, minor],
                  "detectron2_commit": subprocess.check_output(
                      ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()}
    (checkout / "barnacle_environment.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
