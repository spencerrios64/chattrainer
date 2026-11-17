"""WhatsApp "Export chat" (.txt) parser. Handles Android and iOS layouts."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import Message, iter_files, read_text

LINE = re.compile(
    r"^[‎‏]?\[?(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\]?\s*(?:-\s*)?"
    r"(?P<sender>[^:]{1,80}?):\s(?P<text>.*)$"
)
SYSTEM_HINTS = ("<media omitted>", "<attached:", "image omitted", "video omitted",
                "sticker omitted", "audio omitted", "this message was deleted",
                "you deleted this message", "<this message was edited>")


def _ts(date: str, time: str) -> float:
    d = re.split(r"[./-]", date)
    t = time.replace(".", ":").replace(" ", "").upper().replace("A.M.", "AM").replace("P.M.", "PM")
    for dfmt in ("%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y", "%Y/%m/%d"):
        for tfmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S%p", "%I:%M%p"):
            try:
                # Ambiguous d/m vs m/d: pick first that parses (only used for ordering/gaps).
                return datetime.strptime("/".join(d) + " " + t, dfmt + " " + tfmt).timestamp()
            except ValueError:
                continue
    return 0.0


def parse_file(path: Path) -> list[Message]:
    chat_id = f"whatsapp:{path.stem}"
    msgs: list[Message] = []
    for raw in read_text(path).splitlines():
        m = LINE.match(raw)
        if m:
            msgs.append(Message("whatsapp", chat_id, m["sender"].strip(), m["text"],
                                _ts(m["date"], m["time"])))
        elif msgs and raw.strip():
            msgs[-1].text += "\n" + raw  # continuation of a multi-line message
    return [x for x in msgs if not any(h in x.text.lower() for h in SYSTEM_HINTS)]


def parse(root: Path) -> Iterator[Message]:
    for f in iter_files(root, (".txt",)):
        yield from parse_file(f)
