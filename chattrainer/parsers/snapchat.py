"""Snapchat "My Data" export: json/chat_history.json."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import Message, iter_files, read_text


def _ts(s: str) -> float:
    try:
        return datetime.strptime(s.replace(" UTC", "").strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return 0.0


def parse(root: Path) -> Iterator[Message]:
    for f in iter_files(root, (".json",)):
        try:
            data = json.loads(read_text(f))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        msgs = []
        for key, items in data.items():
            if "chat history" not in key.lower() or not isinstance(items, list):
                continue
            for r in items:
                text = str(r.get("Content") or "").strip()
                if not text or str(r.get("Media Type", "TEXT")).upper() != "TEXT":
                    continue
                other = r.get("Conversation Title") or r.get("From") or "?"
                me = r.get("IsSender")
                msgs.append(Message("snapchat", f"snapchat:{other}",
                                    "__me__" if me else str(r.get("From", "?")), text,
                                    _ts(str(r.get("Created", ""))), bool(me)))
        msgs.sort(key=lambda m: (m.chat_id, m.ts))
        yield from msgs
