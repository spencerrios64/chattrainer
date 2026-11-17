"""TikTok data download parser (user_data_tiktok.json or the .txt variant)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .base import Message, iter_files, read_text


def _ts(s: str) -> float:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return 0.0


def _walk(node: Any) -> Iterator[tuple[str, list]]:
    """Find every 'Chat History with <name>:' list anywhere in the JSON tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.lower().startswith("chat history with") and isinstance(v, list):
                yield k, v
            else:
                yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _parse_json(path: Path) -> list[Message]:
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError:
        return []
    out = []
    for key, items in _walk(data):
        name = re.sub(r"^chat history with\s*", "", key, flags=re.I).rstrip(": ").strip()
        chat_id = f"tiktok:{name}"
        chat = [Message("tiktok", chat_id, str(i.get("From", "?")), str(i.get("Content", "")),
                        _ts(str(i.get("Date", "")))) for i in items if isinstance(i, dict)]
        chat = [m for m in chat if m.text.strip() and not m.text.startswith("http")]
        chat.sort(key=lambda m: m.ts)
        out.extend(chat)
    return out


def _parse_txt(path: Path) -> list[Message]:
    out, chat_id, cur = [], "tiktok:?", {}
    def flush():
        if cur.get("Content", "").strip():
            out.append(Message("tiktok", chat_id, cur.get("From", "?"), cur["Content"],
                               _ts(cur.get("Date", ""))))
        cur.clear()
    for line in read_text(path).splitlines():
        h = re.match(r"^Chat History with (.+?):?\s*$", line, re.I)
        if h:
            flush(); chat_id = f"tiktok:{h.group(1)}"; continue
        kv = re.match(r"^(Date|From|Content):\s?(.*)$", line)
        if kv:
            if kv.group(1) == "Date":
                flush()
            cur[kv.group(1)] = kv.group(2)
        elif "Content" in cur and line.strip():
            cur["Content"] += "\n" + line
    flush()
    return out


def parse(root: Path) -> Iterator[Message]:
    for f in iter_files(root, (".json", ".txt")):
        yield from (_parse_json(f) if f.suffix.lower() == ".json" else _parse_txt(f))
