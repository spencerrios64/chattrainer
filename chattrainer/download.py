"""Download a model from Hugging Face ahead of training (used by `chattrainer download` and the app's Models tab)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Skip duplicate weight formats and other files training never reads.
IGNORE = ["*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "original/*", "consolidated*", "*.pth"]


def cache_root() -> Path:
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def is_downloaded(model_id: str) -> bool:
    """True if model_id is a local folder or already sits in the Hugging Face cache."""
    if Path(model_id).is_dir():
        return True
    snaps = cache_root() / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any(snaps.glob("*/config.json"))


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub (small)...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        import huggingface_hub
    return huggingface_hub


def login() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("Paste your Hugging Face token first.")
        return 1
    try:
        _hub().login(token=token, add_to_git_credential=False)
    except Exception as e:  # noqa: BLE001 - show any failure plainly
        print(f"Login failed: {e}")
        return 1
    print("Logged in to Hugging Face. Gated models (Llama, Gemma) can now be downloaded once you accept their licence.")
    return 0


def main(model_id: str) -> int:
    model_id = model_id.strip()
    if not model_id:
        print("Choose a model first.")
        return 1
    if Path(model_id).is_dir():
        print(f"{model_id} is a local folder, nothing to download.")
        return 0
    hub = _hub()
    print(f"Downloading {model_id} (several GB, this can take a while)...", flush=True)
    try:
        path = hub.snapshot_download(repo_id=model_id, ignore_patterns=IGNORE)
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if name in ("GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError") or "401" in str(e) or "403" in str(e):
            print(f"Could not download {model_id}: {name}.\n"
                  "If this is a Llama or Gemma model: open its page on huggingface.co, accept the licence, "
                  "then log in with a token (Models tab) and try again. Otherwise check the model id.")
        else:
            print(f"Download failed: {name}: {e}")
        return 1
    print(f"\nDone. Saved to {path}")
    return 0
