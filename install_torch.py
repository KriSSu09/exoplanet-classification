from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Optional, Tuple, List


# -----------------------------
# Helpers
# -----------------------------
def run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def pip_install(args: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip"] + args
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def pip_uninstall(pkgs: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + pkgs
    print("Running:", " ".join(cmd))
    # Don't fail if packages are not installed
    subprocess.call(cmd)


def has_nvidia() -> bool:
    try:
        run(["nvidia-smi"])
        return True
    except Exception:
        return False


def detect_compute_capability() -> Optional[Tuple[int, int]]:
    """
    Returns (major, minor) compute capability, e.g. (6, 1) for GTX 1070.
    """
    try:
        out = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
        line = out.splitlines()[0].strip()
        m = re.match(r"(\d+)\.(\d+)", line)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def cuda_smoke_test() -> bool:
    """
    Returns True if torch can actually run a CUDA kernel on this GPU.
    Catches the classic 'no kernel image is available for execution on the device'.
    """
    test_code = r"""
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA_NOT_AVAILABLE")
    sys.exit(2)

try:
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("CUDA_OK", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__)
    sys.exit(0)
except Exception as e:
    print("CUDA_FAIL", type(e).__name__, str(e))
    sys.exit(1)
"""
    p = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)
    print(p.stdout.strip())
    if p.returncode == 0:
        return True
    return False


def install_from_index(index_url: str) -> bool:
    """
    Try installing torch/vision/audio from a given PyTorch wheel index URL.
    Return True only if install succeeds AND CUDA smoke test passes.
    """
    # Remove any existing installs first
    pip_uninstall(["torch", "torchvision", "torchaudio"])

    # Install from the specified index
    try:
        pip_install(["install", "--upgrade", "torch", "torchvision", "torchaudio", "--index-url", index_url])
    except subprocess.CalledProcessError:
        print(f"Install failed from {index_url}")
        return False

    # Verify it actually runs on CUDA on this GPU
    ok = cuda_smoke_test()
    if not ok:
        print(f"Installed from {index_url}, but CUDA smoke test FAILED on this GPU. Trying another index...")
        return False

    print(f"✅ Working CUDA PyTorch found from: {index_url}")
    return True


def install_cpu_torch() -> None:
    pip_uninstall(["torch", "torchvision", "torchaudio"])
    pip_install(["install", "--upgrade", "torch", "torchvision", "torchaudio"])
    print("✅ Installed CPU-only PyTorch")


# -----------------------------
# Main logic
# -----------------------------
def main() -> None:
    # upgrade pip first (helps with wheel resolution)
    pip_install(["install", "--upgrade", "pip"])

    if not has_nvidia():
        print("No NVIDIA GPU detected -> installing CPU-only PyTorch")
        install_cpu_torch()
        return

    cc = detect_compute_capability()
    if cc is None:
        print("Could not detect compute capability -> installing CPU-only PyTorch")
        install_cpu_torch()
        return

    print(f"Detected GPU compute capability: {cc[0]}.{cc[1]}")

    # If GPU is Pascal (sm_61), newest CUDA wheels may drop support → probe indexes.
    # Your error shows your installed build supported only sm_75+.
    # Probing is the only reliable way on Windows + Python 3.13.
    # (PyTorch wheels are organized by CUDA runtime index URLs.)  :contentReference[oaicite:1]{index=1}
    candidate_cuda_indexes = [
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu124",
        "https://download.pytorch.org/whl/cu121",
        "https://download.pytorch.org/whl/cu118",
    ]

    # For modern GPUs (>= sm_75) we can just take the newest CUDA index; probing is still safe though.
    # For sm_61 we probe in descending order of “newness”.
    for idx in candidate_cuda_indexes:
        print(f"\n=== Trying PyTorch index: {idx} ===")
        if install_from_index(idx):
            return

    # If we got here, no CUDA wheel worked on this Python (likely due to wheel availability or arch support).
    print("\n❌ No compatible CUDA-enabled PyTorch wheel worked on this system with current Python.")
    print("Next step:")
    print("  - Downgrade Python to 3.12 (or 3.11), recreate the venv, and run this script again.")
    print("Fallback (always works):")
    print("  - Installing CPU-only PyTorch now so you can proceed with development.")
    install_cpu_torch()


if __name__ == "__main__":
    main()
