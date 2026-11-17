"""PII scrubbing applied to training text."""
from __future__ import annotations

import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE = re.compile(r"(?<!\w)(?:\+|00)?\d[\d\s().-]{7,}\d(?!\w)")


def scrub(text: str, urls: bool = True, names: list[str] | None = None) -> str:
    text = EMAIL.sub("<EMAIL>", text)
    if urls:
        text = URL.sub("<URL>", text)
    text = IBAN.sub("<IBAN>", text)
    text = CARD.sub("<NUMBER>", text)
    text = PHONE.sub("<PHONE>", text)
    for n in names or []:
        if len(n) >= 3:
            text = re.sub(re.escape(n), "<NAME>", text, flags=re.I)
    return text
