"""Turn parsed messages into chat-format training examples (JSONL)."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .config import Config
from .parsers import PARSERS
from .parsers.base import Message
from .scrub import scrub


def load_messages(cfg: Config) -> dict[str, list[Message]]:
    """Parse every data/<platform>/ folder into messages grouped by chat."""
    chats: dict[str, list[Message]] = defaultdict(list)
    root = Path(cfg.data_dir)
    for platform, parse in PARSERS.items():
        d = root / platform
        if d.exists():
            for m in parse(d):
                chats[m.chat_id].append(m)
    return chats


def _is_me(m: Message, cfg: Config) -> bool:
    if m.is_me is not None:
        return m.is_me
    return m.sender.lower() in cfg.my_names(m.platform)


def _turns(msgs: list[Message], cfg: Config) -> list[list]:
    """Merge consecutive same-speaker messages; split on long gaps. Returns sessions of (role, text)."""
    if all(m.ts > 0 for m in msgs):
        msgs = sorted(msgs, key=lambda m: m.ts)
    group = len({m.sender for m in msgs if not _is_me(m, cfg)}) > 1
    names = sorted({m.sender for m in msgs if m.sender != "__me__"}) if cfg.anonymize_names else []
    sessions: list[list] = []
    cur: list = []
    last: Message | None = None
    for m in msgs:
        text = m.text.strip()[: cfg.max_turn_chars]
        if cfg.scrub_pii:
            text = scrub(text, cfg.scrub_urls, names)
        if not text:
            continue
        me = _is_me(m, cfg)
        role = "assistant" if me else "user"
        if not me and group:
            text = f"{scrub(m.sender, False) if cfg.scrub_pii else m.sender}: {text}"
            if cfg.anonymize_names:
                text = "<NAME>: " + text.split(": ", 1)[1]
        gap = (m.ts - last.ts) if (last and m.ts and last.ts) else 0
        if last and gap > cfg.session_gap_hours * 3600:
            if cur:
                sessions.append(cur)
            cur = []
        if cur and cur[-1][0] == role and (not gap or gap <= cfg.merge_gap_minutes * 60):
            cur[-1][1] += "\n" + text
        else:
            cur.append([role, text])
        last = m
    if cur:
        sessions.append(cur)
    return sessions


def _examples(session: list, cfg: Config) -> list[list[dict]]:
    sysmsg = {"role": "system", "content": cfg.system()}
    out = []
    if cfg.mode in ("pairs", "both"):
        for i, (role, text) in enumerate(session):
            if role != "assistant" or len(text) < cfg.min_reply_chars or i == 0:
                continue
            ctx = session[max(0, i - cfg.context_turns): i]
            while ctx and ctx[0][0] != "user":
                ctx = ctx[1:]
            if ctx:
                out.append([sysmsg] + [{"role": r, "content": t} for r, t in ctx]
                           + [{"role": "assistant", "content": text}])
    if cfg.mode in ("windows", "both") and len(session) >= 2:
        for s in range(0, len(session), max(1, cfg.window_stride)):
            w = session[s: s + cfg.window_turns]
            while w and w[0][0] != "user":
                w = w[1:]
            while w and w[-1][0] != "assistant":
                w = w[:-1]
            if len(w) >= 2:
                out.append([sysmsg] + [{"role": r, "content": t} for r, t in w])
    return out


def _solo(session: list, cfg: Config) -> list[list[dict]]:
    sysmsg = {"role": "system", "content": cfg.system()}
    return [[sysmsg, {"role": "user", "content": "(keep chatting)"}, {"role": "assistant", "content": t}]
            for r, t in session if r == "assistant" and len(t) >= cfg.min_reply_chars]


def build(cfg: Config) -> dict:
    chats = load_messages(cfg)
    train, val, stats = [], [], defaultdict(int)
    for chat_id, msgs in chats.items():
        platform = msgs[0].platform
        senders = {m.sender for m in msgs if not _is_me(m, cfg)}
        if cfg.skip_groups and len(senders) > 1:
            stats["groups_skipped"] += 1
            continue
        mine = sum(_is_me(m, cfg) for m in msgs)
        stats[f"{platform}_messages"] += len(msgs)
        stats[f"{platform}_my_messages"] += mine
        if mine == 0:
            stats["chats_without_me"] += 1
            continue
        sessions = _turns(msgs, cfg)
        solo = not senders
        exs = []
        for s in sessions:
            exs.extend(_solo(s, cfg) if solo else _examples(s, cfg))
        if solo and not cfg.include_solo:
            continue
        # keep every chat entirely in train or val so conversations never leak across the split
        h = int(hashlib.md5(chat_id.encode()).hexdigest(), 16) % 1000 / 1000
        (val if h < cfg.val_fraction else train).extend(exs)
    random.Random(cfg.seed).shuffle(train)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")
    stats.update(chats=len(chats), train_examples=len(train), val_examples=len(val))
    return dict(stats)
