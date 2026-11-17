"""Model presets grouped by PC tier, plus GPU detection. Any Hugging Face id or local folder also works."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

TIERS = ("low", "mid", "high")
TIER_LABELS = {
    "low": "Low end (about 4-6 GB VRAM, older or laptop GPUs)",
    "mid": "Mid range (about 8-12 GB VRAM, e.g. RTX 3060)",
    "high": "High end (16 GB VRAM and up)",
}


@dataclass(frozen=True)
class Preset:
    tier: str
    id: str
    params: str
    vram: int  # rough minimum GB for 4-bit QLoRA at ~1024 tokens
    note: str = ""
    default: bool = False


PRESETS: list[Preset] = [
    Preset("low", "Qwen/Qwen2.5-1.5B-Instruct", "1.5B", 4, "smallest, fastest"),
    Preset("low", "meta-llama/Llama-3.2-1B-Instruct", "1B", 4, "gated: accept licence"),
    Preset("low", "google/gemma-3-1b-it", "1B", 4, "gated: accept licence"),
    Preset("low", "Qwen/Qwen2.5-3B-Instruct", "3B", 6, "best quality for low end", True),
    Preset("low", "meta-llama/Llama-3.2-3B-Instruct", "3B", 6, "gated: accept licence"),
    Preset("mid", "Qwen/Qwen3-4B-Instruct-2507", "4B", 8, "light and capable"),
    Preset("mid", "Qwen/Qwen2.5-7B-Instruct", "7B", 10, "good multilingual"),
    Preset("mid", "meta-llama/Llama-3.1-8B-Instruct", "8B", 12, "gated: accept licence", True),
    Preset("mid", "mistralai/Mistral-7B-Instruct-v0.3", "7B", 10, "gated: accept licence"),
    Preset("mid", "google/gemma-3-12b-it", "12B", 12, "tight on 12 GB, gated"),
    Preset("mid", "skilledu/qwen3-4b-heretic", "4B", 8, "Heretic (uncensored), community"),
    Preset("mid", "LeadFootThrottleCock/Qwen2.5-7B-Instruct-heretic", "7B", 10, "Heretic (uncensored), community"),
    Preset("mid", "DreamFast/qwen3-8b-heretic", "8B", 12, "Heretic (uncensored), community"),
    Preset("mid", "p-e-w/gemma-3-12b-it-heretic", "12B", 12, "Heretic (uncensored), tight on 12 GB"),
    Preset("high", "Qwen/Qwen2.5-14B-Instruct", "14B", 16, "great balance", True),
    Preset("high", "google/gemma-3-27b-it", "27B", 24, "tight on 24 GB, gated"),
    Preset("high", "Qwen/Qwen2.5-32B-Instruct", "32B", 24, "very tight on 24 GB"),
    Preset("high", "meta-llama/Llama-3.3-70B-Instruct", "70B", 48, "needs 48 GB or several GPUs, gated"),
    Preset("high", "krthkyadtya/Qwen2.5-72B-Instruct-heretic", "72B", 48, "Heretic (uncensored), needs 48 GB or several GPUs"),
]


def is_heretic(model_id: str) -> bool:
    """True for Heretic / abliterated models, by name (works for any custom Hugging Face id or folder)."""
    m = model_id.lower()
    return "heretic" in m or "abliterat" in m


def models_for(tier: str, heretic: bool = False) -> list[Preset]:
    """Presets for a tier. Heretic (uncensored) models are only listed when heretic=True."""
    ps = [p for p in PRESETS if p.tier == tier] if tier in TIERS else list(PRESETS)
    return ps if heretic else [p for p in ps if not is_heretic(p.id)]


def find(model_id: str) -> Preset | None:
    return next((p for p in PRESETS if p.id == model_id), None)


def default_for(tier: str) -> Preset:
    return next(p for p in PRESETS if p.tier == tier and p.default)


def detect_vram_gb() -> float | None:
    """Total VRAM of the biggest NVIDIA GPU, or None if nvidia-smi is unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return max(float(x) for x in out.split()) / 1024
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def tier_for(vram_gb: float) -> str:
    return "low" if vram_gb < 7.5 else "mid" if vram_gb < 15 else "high"


def describe(p: Preset) -> str:
    return f"{p.params}, needs about {p.vram} GB VRAM (4-bit){', ' + p.note if p.note else ''}"
