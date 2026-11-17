"""One-command installer for the training dependencies (used by `chattrainer install-train` and the app)."""
from __future__ import annotations

import platform
import subprocess
import sys

CUDA_BUILDS = ("cu128", "cu126", "cu124")  # tried in order on Windows, where PyPI torch is CPU-only
REST = ["transformers>=4.41", "peft>=0.10", "bitsandbytes>=0.43", "accelerate>=0.30"]


def _pip(*args: str) -> int:
    return subprocess.call([sys.executable, "-m", "pip", "install", *args])


def main() -> int:
    system = platform.system()
    if system == "Darwin":
        print("Training needs an NVIDIA GPU (CUDA), which Macs don't have. Use a Windows/Linux PC or Google Colab.")
        return 1
    print("Installing PyTorch (this downloads a few GB)...")
    if system == "Windows":
        for tag in CUDA_BUILDS:
            print(f"Trying PyTorch with CUDA build {tag}...")
            if _pip("torch", "--index-url", f"https://download.pytorch.org/whl/{tag}") == 0:
                break
        else:
            print("Could not install a CUDA build of PyTorch. See https://pytorch.org/get-started/locally/")
            return 1
    elif _pip("torch") != 0:
        return 1
    print("Installing transformers, peft, bitsandbytes, accelerate...")
    if _pip(*REST) != 0:
        return 1
    ok = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                        capture_output=True, text=True).stdout.strip()
    if ok == "True":
        print("\nDone. CUDA GPU detected, you can train now.")
        return 0
    print("\nInstalled, but PyTorch cannot see an NVIDIA GPU. Update your NVIDIA driver and run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
