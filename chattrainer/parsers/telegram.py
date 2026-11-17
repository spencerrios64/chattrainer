"""Telegram Desktop "Export Telegram data" (result.json, JSON format)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import Message, iter_files, read_text


def _flat(text) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in text)
    return ""


def _chat(chat: dict, fallback: str) -> list[Message]:
    chat_id = f"telegram:{chat.get('name') or chat.get('id') or fallback}"
    out = []
    for m in chat.get("messages", []):
        if m.get("type") != "message":
            continue
        text = _flat(m.get("text")).strip()
        if not text:
            continue
        try:
            ts = datetime.fromisoformat(m["date"]).timestamp()
        except (KeyError, ValueError):
            ts = 0.0
        out.append(Message("telegram", chat_id, str(m.get("from") or "?"), text, ts))
    return out


def parse(root: Path) -> Iterator[Message]:
    for f in iter_files(root, (".json",)):
        try:
            data = json.loads(read_text(f))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if "chats" in data and isinstance(data["chats"], dict):
            for i, c in enumerate(data["chats"].get("list", [])):
                yield from _chat(c, str(i))
        elif "messages" in data:
            yield from _chat(data, f.stem)
