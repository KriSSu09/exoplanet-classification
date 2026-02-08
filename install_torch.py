from __future__ import annotations

import re
import subprocess
import sys


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def has_nvidia() -> bool:
    try:
        run(["nvidia-smi"])
        return True
    except Exception:
        return False


def detect_cuda_major_minor() -> tuple[int, int] | None:
    """
    Parse CUDA version from `nvidia-smi` output, e.g. "CUDA Version: 12.2"
    Returns (12, 2) or None.
    """
    try:
        out = run(["nvidia-smi"])
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def choose_torch_index(cuda_mm: tuple[int, int] | None) -> str | None:
    """
    Choose a PyTorch wheel index URL based on detected CUDA version.
    We map to the nearest supported "cuXYZ" index.
    Adjust mappings as PyTorch releases change.
    """
    if cuda_mm is None:
        return None

    major, minor = cuda_mm

    # Conservative mapping: prefer cu121 for CUDA 12.x, cu118 for CUDA 11.x
    # (PyTorch wheels are bundled with CUDA runtime; driver compatibility is the key).
    if major >= 12:
        return "https://download.pytorch.org/whl/cu121"
    if major == 11:
        return "https://download.pytorch.org/whl/cu118"
    return None


def install_torch():
    if not has_nvidia():
        print("No NVIDIA GPU detected -> installing CPU-only PyTorch")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        return

    cuda_mm = detect_cuda_major_minor()
    index_url = choose_torch_index(cuda_mm)

    if index_url is None:
        print("NVIDIA detected but CUDA version unknown/unsupported -> installing CPU-only PyTorch")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        return

    print(f"NVIDIA detected (CUDA {cuda_mm[0]}.{cuda_mm[1]}) -> installing PyTorch from: {index_url}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", index_url
    ])


if __name__ == "__main__":
    install_torch()
