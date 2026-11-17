"""Shared types and helpers for export parsers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class Message:
    platform: str
    chat_id: str
    sender: str
    text: str
    ts: float = 0.0  # unix seconds; 0 means unknown (file order is used)
    is_me: Optional[bool] = None  # None -> decided later from config names


def fix_mojibake(s: str) -> str:
    """Instagram/Facebook exports store UTF-8 bytes as latin-1 code points."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def iter_files(root: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    if root.is_file():
        if root.suffix.lower() in suffixes:
            yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")
