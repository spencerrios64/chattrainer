"""Discord parsers.

* Official data package (messages/c<id>/messages.json or .csv): contains ONLY your own
  messages, so they are marked is_me=True and used as standalone style samples.
* DiscordChatExporter JSON ({"messages":[{"author":{"name"},"content","timestamp"}]}):
  full conversations.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import Message, iter_files, read_text


def _iso(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").split(".")[0]).timestamp()
    except ValueError:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            return 0.0


def parse(root: Path) -> Iterator[Message]:
    for f in iter_files(root, (".json", ".csv")):
        chat_id = f"discord:{f.parent.name}"
        if f.suffix.lower() == ".csv":
            with f.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
                for row in csv.DictReader(fh):
                    text = (row.get("Contents") or "").strip()
                    if text:
                        yield Message("discord", chat_id, "__me__", text, _iso(row.get("Timestamp", "")), True)
            continue
        try:
            data = json.loads(read_text(f))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):  # official package messages.json
            for r in data:
                if isinstance(r, dict) and str(r.get("Contents", "")).strip():
                    yield Message("discord", chat_id, "__me__", r["Contents"], _iso(str(r.get("Timestamp", ""))), True)
        elif isinstance(data, dict) and "messages" in data:  # DiscordChatExporter
            cid = f"discord:{(data.get('channel') or {}).get('name') or f.stem}"
            for m in data["messages"]:
                text = str(m.get("content", "")).strip()
                if text:
                    a = m.get("author") or {}
                    yield Message("discord", cid, str(a.get("name") or a.get("nickname") or "?"),
                                  text, _iso(str(m.get("timestamp", ""))))
