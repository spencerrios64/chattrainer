"""Build a fine-tuning set from your Claude / ChatGPT exports (conversations.json or the export .zip).

Only chats you pick are used: filter by minimum length, drop short/low-effort replies, scrub private details,
and keep whole conversations on one side of the train/val split.
"""
from __future__ import annotations

import hashlib
import json
import random
import zipfile
from pathlib import Path

from .scrub import scrub


def _claude(c: dict) -> list[tuple[str, str]]:
    out = []
    for m in c.get("chat_messages", []):
        text = m.get("text") or "\n".join(p.get("text", "") for p in m.get("content", []) if isinstance(p, dict))
        if text.strip():
            out.append(("user" if m.get("sender") == "human" else "assistant", text.strip()))
    return out


def _chatgpt(c: dict) -> list[tuple[str, str]]:
    msgs = [n["message"] for n in c.get("mapping", {}).values() if n.get("message")]
    msgs.sort(key=lambda m: m.get("create_time") or 0)
    out = []
    for m in msgs:
        role = (m.get("author") or {}).get("role")
        if role not in ("user", "assistant"):
            continue
        text = "\n".join(p for p in (m.get("content") or {}).get("parts", []) if isinstance(p, str)).strip()
        if text:
            out.append((role, text))
    return out


def load_chats(path: str | Path) -> list[dict]:
    """Return [{id, title, ai, turns:[(role,text)]}] from a conversations.json, a folder of them, or a .zip."""
    p = Path(path)
    blobs: list[tuple[str, str]] = []
    if p.is_dir():
        blobs = [(f.name, f.read_text(encoding="utf-8")) for f in p.rglob("conversations*.json")]
    elif p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            blobs = [(n, z.read(n).decode("utf-8")) for n in z.namelist() if n.endswith(".json") and "conversation" in n.lower()]
    else:
        blobs = [(p.name, p.read_text(encoding="utf-8"))]
    chats = []
    for name, raw in blobs:
        data = json.loads(raw)
        for i, c in enumerate(data if isinstance(data, list) else [data]):
            if "chat_messages" in c:
                ai, turns = "Claude", _claude(c)
            elif "mapping" in c:
                ai, turns = "ChatGPT", _chatgpt(c)
            else:
                continue
            chats.append({"id": str(c.get("uuid") or c.get("conversation_id") or f"{name}:{i}"),
                          "title": c.get("name") or c.get("title") or "", "ai": ai, "turns": turns})
    return chats


def build_ai_dataset(path: str | Path, out_dir: str | Path = "output", *, keep_titles: list[str] | None = None,
                     min_turns: int = 4, min_reply_chars: int = 200, scrub_text: bool = True,
                     system: str = "You are a helpful assistant.", val_fraction: float = 0.05, seed: int = 1,
                     train_on: str = "assistant") -> dict:
    """train_on='assistant' learns the AI's good answers; train_on='user' learns YOUR side of the chat."""
    chats = load_chats(path)
    want = [t.lower() for t in keep_titles or []]
    train, val, used = [], [], 0
    for c in chats:
        if want and not any(w in c["title"].lower() for w in want):
            continue
        turns = c["turns"]
        if len(turns) < min_turns:
            continue
        if scrub_text:
            turns = [(r, scrub(t)) for r, t in turns]
        msgs = [{"role": "system", "content": system}]
        target = train_on
        for r, t in turns:
            msgs.append({"role": r, "content": t})
        # drop trailing turn that is not the target role, drop chats where target replies are all too short
        while len(msgs) > 1 and msgs[-1]["role"] != target:
            msgs.pop()
        if not any(m["role"] == target and len(m["content"]) >= min_reply_chars for m in msgs):
            continue
        if len(msgs) < 3:
            continue
        used += 1
        h = int(hashlib.md5(c["id"].encode()).hexdigest(), 16) % 1000 / 1000
        (val if h < val_fraction else train).append({"messages": msgs})
    random.Random(seed).shuffle(train)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ai_train.jsonl", train), ("ai_val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"chats_found": len(chats), "chats_used": used, "train_examples": len(train), "val_examples": len(val)}
