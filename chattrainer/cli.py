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
        ("gui", "open the desktop app"),
        ("install-train", "install PyTorch (CUDA) and the other training dependencies"),
    ]:
        sub.add_parser(name, help=help_)
    mp = sub.add_parser("models", help="list model presets by PC tier and detect your GPU")
    mp.add_argument("--tier", choices=["low", "mid", "high"])
    dp = sub.add_parser("download", help="download a model from Hugging Face before training")
    dp.add_argument("model")
    sub.add_parser("login", help="log in to Hugging Face using the HF_TOKEN environment variable")
    up = sub.add_parser("use", help="set the base model: a number from `chattrainer models`, a preset id, or any HF id/folder")
    up.add_argument("model")
    args = ap.parse_args(argv)

    if args.cmd == "install-train":
        from .setup_train import main as install_train
        raise SystemExit(install_train())

    if args.cmd == "download":
        from .download import main as download
        raise SystemExit(download(args.model))

    if args.cmd == "login":
        from .download import login
        raise SystemExit(login())

    if args.cmd == "models":
        from .models import PRESETS, TIER_LABELS, TIERS, describe, detect_vram_gb, tier_for
        vram = detect_vram_gb()
        mine = tier_for(vram) if vram else None
        print(f"Detected GPU memory: {vram:.0f} GB -> recommended tier: {mine}" if vram
              else "No NVIDIA GPU detected (nvidia-smi). Pick a tier yourself.")
        for t in ([args.tier] if args.tier else TIERS):
            print(f"\n{TIER_LABELS[t]}" + ("   <- your PC" if t == mine else ""))
            for i, p in enumerate(PRESETS, 1):
                if p.tier == t:
                    print(f"  {i:>2}. {p.id}{'  [default]' if p.default else ''}\n      {describe(p)}")
        print("\nChoose one with: chattrainer use <number or id>")
        return

    if args.cmd == "use":
        import dataclasses
        from .models import PRESETS
        pick = args.model
        if pick.isdigit() and 1 <= int(pick) <= len(PRESETS):
            pick = PRESETS[int(pick) - 1].id
        cfg = config_mod.load(args.config)
        config_mod.save(dataclasses.replace(cfg, base_model=pick), args.config)
        print(f"base_model set to {pick} in {args.config}")
        return

    if args.cmd == "gui":
        from .gui import run
        run()
        return

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
