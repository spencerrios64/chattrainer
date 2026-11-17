"""Instagram "Download your information" (Meta) DM parser: JSON (preferred) and HTML."""
from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path
from typing import Iterator

from .base import Message, fix_mojibake, iter_files, read_text


def _parse_json(path: Path) -> list[Message]:
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict) or "messages" not in data:
        return []
    title = fix_mojibake(str(data.get("title") or path.parent.name))
    chat_id = f"instagram:{path.parent.name}"
    out = []
    for m in data["messages"]:
        text = m.get("content")
        if not text or not isinstance(text, str):
            continue
        text = fix_mojibake(text)
        if text.endswith("to your message ") or " sent an attachment" in text:
            continue
        out.append(Message("instagram", chat_id, fix_mojibake(m.get("sender_name", "?")),
                           text, m.get("timestamp_ms", 0) / 1000))
    out.sort(key=lambda x: x.ts)  # Meta exports are newest-first
    _ = title
    return out


BLOCK = re.compile(r'<div class="pam[^"]*_a6-g[^"]*">(.*?)</div></div></div>', re.S)
SENDER = re.compile(r'<div class="[^"]*_a6-h[^"]*">(.*?)</div>', re.S)
BODY = re.compile(r'<div class="[^"]*_a6-p[^"]*">(.*?)</div>\s*<div class="[^"]*_a6-o', re.S)


def _parse_html(path: Path) -> list[Message]:
    """Best effort: Meta's HTML class names change; JSON export is more reliable."""
    src = read_text(path)
    chat_id = f"instagram:{path.parent.name}"
    out = []
    for blk in BLOCK.findall(src):
        s, b = SENDER.search(blk), BODY.search(blk)
        if not (s and b):
            continue
        text = htmllib.unescape(re.sub(r"<[^>]+>", " ", b.group(1))).strip()
        if text:
            out.append(Message("instagram", chat_id, htmllib.unescape(s.group(1)).strip(), text))
    out.reverse()
    return out


def parse(root: Path) -> Iterator[Message]:
    for f in iter_files(root, (".json", ".html")):
        if f.suffix.lower() == ".json" and f.name.startswith("message_"):
            yield from _parse_json(f)
        elif f.suffix.lower() == ".html" and f.name.startswith("message_"):
            yield from _parse_html(f)
