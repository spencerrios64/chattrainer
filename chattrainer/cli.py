"""chattrainer command line."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config as config_mod
from .parsers import PARSERS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="chattrainer",
                                 description="Fine-tune a local LLM on your own chat exports.")
    ap.add_argument("-c", "--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in [
        ("init", "create config.toml and data/<platform>/ folders"),
        ("stats", "parse exports and show message counts per platform (checks your names)"),
        ("prepare", "build output/train.jsonl and val.jsonl"),
        ("train", "QLoRA fine-tune (needs `pip install .[train]` and a CUDA GPU)"),
        ("export", "merge adapter into the base model, optional GGUF"),
        ("chat", "talk to the trained model in the terminal"),
    ]:
        sub.add_parser(name, help=help_)
    args = ap.parse_args(argv)

    if args.cmd == "init":
        dst = Path(args.config)
        if dst.exists():
            print(f"{dst} already exists, leaving it alone")
        else:
            shutil.copy(Path(__file__).with_name("config.example.toml"), dst)
            print(f"Wrote {dst}. Edit your names, then drop exports into data/<platform>/")
        for p in PARSERS:
            Path("data", p).mkdir(parents=True, exist_ok=True)
        return

    cfg = config_mod.load(args.config)
    if args.cmd == "stats":
        from .dataset import _is_me, load_messages
        chats = load_messages(cfg)
        per: dict[str, list[int]] = {}
        senders: dict[str, dict[str, int]] = {}
        for msgs in chats.values():
            for m in msgs:
                c = per.setdefault(m.platform, [0, 0])
                c[0] += 1
                c[1] += _is_me(m, cfg)
                senders.setdefault(m.platform, {}).setdefault(m.sender, 0)
                senders[m.platform][m.sender] += 1
        if not per:
            print("No messages found. Put exports in data/<platform>/ (see README).")
        for p, (total, mine) in per.items():
            top = sorted(senders[p].items(), key=lambda kv: -kv[1])[:5]
            print(f"{p}: {total} messages, {mine} identified as you. Top senders: {top}")
            if mine == 0:
                print(f"  -> none matched your names. Add one of the senders above to `names` in {args.config}.")
    elif args.cmd == "prepare":
        from .dataset import build
        for k, v in build(cfg).items():
            print(f"{k}: {v}")
    elif args.cmd == "train":
        from .train import run
        run(cfg)
    elif args.cmd == "export":
        from .export import merge
        merge(cfg)
    elif args.cmd == "chat":
        from .export import chat
        chat(cfg)


if __name__ == "__main__":
    main()
