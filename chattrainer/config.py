"""Config loading (TOML, stdlib only)."""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # identity
    name: str = "me"
    names: list[str] = field(default_factory=list)  # display names/usernames that are YOU
    platform_names: dict[str, list[str]] = field(default_factory=dict)
    # data
    data_dir: str = "data"
    output_dir: str = "output"
    # privacy
    scrub_pii: bool = True
    scrub_urls: bool = True
    anonymize_names: bool = False
    skip_groups: bool = False
    # dataset
    mode: str = "both"  # pairs | windows | both
    context_turns: int = 6
    window_turns: int = 12
    window_stride: int = 6
    merge_gap_minutes: float = 3.0
    session_gap_hours: float = 8.0
    min_reply_chars: int = 2
    max_turn_chars: int = 1200
    include_solo: bool = True
    val_fraction: float = 0.03
    system_prompt: str = ""
    # training
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_len: int = 1024
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    seed: int = 42
    # export
    llama_cpp_dir: str = ""  # path to a llama.cpp checkout for GGUF export
    gguf_quant: str = "Q4_K_M"

    def system(self) -> str:
        return self.system_prompt or (
            f"You are {self.name}. Reply exactly the way {self.name} texts: same tone, "
            "slang, language, humor and message length.")

    def my_names(self, platform: str) -> set[str]:
        return {n.lower() for n in self.names + self.platform_names.get(platform, [])}


def load(path: str | Path = "config.toml") -> Config:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found. Run `chattrainer init` first.")
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict) and k != "platform_names":
            flat.update(v)  # sections are cosmetic; keys are flat
        else:
            flat[k] = v
    known = Config.__dataclass_fields__
    unknown = set(flat) - set(known)
    if unknown:
        raise SystemExit(f"Unknown config keys: {sorted(unknown)}")
    return Config(**flat)


SECTIONS = {
    "identity": ["name", "names"],
    "paths": ["data_dir", "output_dir"],
    "privacy": ["scrub_pii", "scrub_urls", "anonymize_names", "skip_groups"],
    "dataset": ["mode", "context_turns", "window_turns", "window_stride", "merge_gap_minutes",
                "session_gap_hours", "min_reply_chars", "max_turn_chars", "include_solo",
                "val_fraction", "system_prompt"],
    "training": ["base_model", "max_seq_len", "epochs", "learning_rate", "batch_size", "grad_accum",
                 "lora_r", "lora_alpha", "lora_dropout", "load_in_4bit", "seed"],
    "export": ["llama_cpp_dir", "gguf_quant"],
}


def save(cfg: Config, path: str | Path = "config.toml") -> None:
    """Write a config.toml that load() reads back (used by the desktop app)."""
    lines = ["# Written by chattrainer. Safe to edit by hand."]
    for section, keys in SECTIONS.items():
        lines += ["", f"[{section}]"]
        lines += [f"{k} = {json.dumps(getattr(cfg, k))}" for k in keys]
    if cfg.platform_names:
        lines += ["", "[platform_names]"]
        lines += [f"{k} = {json.dumps(v)}" for k, v in cfg.platform_names.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
