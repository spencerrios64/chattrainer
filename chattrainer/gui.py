"""Desktop app (Tkinter, no extra dependencies). Launch with `chattrainer gui` or `chattrainer-gui`.

Works in the current folder: config.toml, data/ and output/ live there.
"""
from __future__ import annotations

import dataclasses
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from . import config as config_mod
from .config import Config
from .parsers import PARSERS

CONFIG_PATH = Path("config.toml")
PRESETS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-12b-it",
    "p-e-w/gemma-3-12b-it-heretic",
]
HINTS = {
    "whatsapp": ".txt files from Export chat",
    "instagram": "folder with message_*.json (choose JSON export)",
    "tiktok": "user_data_tiktok.json or .txt",
    "telegram": "result.json from Telegram Desktop",
    "discord": "messages folder from the data package, or DiscordChatExporter JSON",
    "snapchat": "json/chat_history.json",
}


class Log(scrolledtext.ScrolledText):
    def __init__(self, master, **kw):
        super().__init__(master, height=14, state="disabled", wrap="word", font=("Consolas", 10), **kw)

    def write(self, text: str) -> None:
        self.configure(state="normal")
        self.insert("end", text)
        self.see("end")
        self.configure(state="disabled")

    def clear(self) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("chattrainer")
        self.geometry("860x640")
        if not CONFIG_PATH.exists():
            config_mod.save(Config(), CONFIG_PATH)
        self.cfg = config_mod.load(CONFIG_PATH)
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.session = None  # ChatSession once loaded

        ttk.Label(self, text=f"Project folder: {Path.cwd()}", foreground="#555").pack(anchor="w", padx=10, pady=(6, 0))
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        for title, build in [("1. Data", self._data_tab), ("2. Settings", self._settings_tab),
                             ("3. Prepare", self._prepare_tab), ("4. Train", self._train_tab),
                             ("5. Chat", self._chat_tab)]:
            frame = ttk.Frame(nb, padding=10)
            nb.add(frame, text=title)
            build(frame)
        self.after(100, self._pump)

    # ---------------------------------------------------------------- data
    def _data_tab(self, f: ttk.Frame) -> None:
        ttk.Label(f, text="Add your chat exports. Files are copied into data/<platform>/ in this project folder.",
                  wraplength=780).pack(anchor="w", pady=(0, 8))
        self.counts: dict[str, tk.StringVar] = {}
        for p in PARSERS:
            row = ttk.Frame(f)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=p.capitalize(), width=11).pack(side="left")
            ttk.Button(row, text="Add files…", command=lambda p=p: self._add(p, False)).pack(side="left")
            ttk.Button(row, text="Add folder…", command=lambda p=p: self._add(p, True)).pack(side="left", padx=4)
            self.counts[p] = tk.StringVar()
            ttk.Label(row, textvariable=self.counts[p], width=10).pack(side="left", padx=6)
            ttk.Label(row, text=HINTS[p], foreground="#666").pack(side="left")
        ttk.Label(f, text="Your exports stay on this computer. data/ and output/ are git-ignored.",
                  foreground="#666").pack(anchor="w", pady=(14, 0))
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        for p, var in self.counts.items():
            d = Path(self.cfg.data_dir) / p
            n = sum(1 for x in d.rglob("*") if x.is_file()) if d.exists() else 0
            var.set(f"{n} file(s)")

    def _add(self, platform: str, folder: bool) -> None:
        picked = [filedialog.askdirectory(title=f"{platform} export folder")] if folder else \
            list(filedialog.askopenfilenames(title=f"{platform} export files"))
        dest = Path(self.cfg.data_dir) / platform
        dest.mkdir(parents=True, exist_ok=True)
        for src in filter(None, picked):
            s = Path(src)
            if s.is_dir():
                shutil.copytree(s, dest / s.name, dirs_exist_ok=True)
            else:
                shutil.copy2(s, dest / s.name)
        self._refresh_counts()

    # ------------------------------------------------------------ settings
    def _settings_tab(self, f: ttk.Frame) -> None:
        c = self.cfg
        self.v: dict[str, tk.Variable] = {}

        def entry(label: str, key: str, value, width=50):
            r = ttk.Frame(f)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=label, width=24).pack(side="left")
            var = tk.StringVar(value=str(value))
            ttk.Entry(r, textvariable=var, width=width).pack(side="left")
            self.v[key] = var

        def check(label: str, key: str, value: bool):
            var = tk.BooleanVar(value=value)
            ttk.Checkbutton(f, text=label, variable=var).pack(anchor="w")
            self.v[key] = var

        entry("Your name", "name", c.name)
        entry("Names that are YOU", "names", ", ".join(c.names))
        ttk.Label(f, text="Comma separated display names / usernames used in your exports.",
                  foreground="#666").pack(anchor="w", padx=(0, 0))
        ttk.Separator(f).pack(fill="x", pady=8)
        check("Scrub emails, phone numbers, IBANs, card-like numbers", "scrub_pii", c.scrub_pii)
        check("Replace URLs", "scrub_urls", c.scrub_urls)
        check("Anonymize other people's names", "anonymize_names", c.anonymize_names)
        check("Only 1-on-1 chats (skip groups)", "skip_groups", c.skip_groups)
        ttk.Separator(f).pack(fill="x", pady=8)
        r = ttk.Frame(f)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="Dataset mode", width=24).pack(side="left")
        mode = tk.StringVar(value=c.mode)
        ttk.Combobox(r, textvariable=mode, values=["pairs", "windows", "both"], state="readonly", width=12).pack(side="left")
        self.v["mode"] = mode
        r = ttk.Frame(f)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="Base model", width=24).pack(side="left")
        base = tk.StringVar(value=c.base_model)
        ttk.Combobox(r, textvariable=base, width=48, values=PRESETS).pack(side="left")
        self.v["base_model"] = base
        ttk.Label(f, text="Paste any Hugging Face model id (for example a Heretic / abliterated model) or a local folder.",
                  foreground="#666").pack(anchor="w")
        entry("Epochs", "epochs", c.epochs, 10)
        entry("Max sequence length", "max_seq_len", c.max_seq_len, 10)
        entry("Gradient accumulation", "grad_accum", c.grad_accum, 10)
        entry("llama.cpp folder (GGUF)", "llama_cpp_dir", c.llama_cpp_dir)
        ttk.Button(f, text="Save settings", command=self._save).pack(anchor="w", pady=12)

    def _save(self) -> bool:
        v = {k: x.get() for k, x in self.v.items()}
        try:
            self.cfg = dataclasses.replace(
                self.cfg, name=v["name"].strip() or "me",
                names=[n.strip() for n in v["names"].split(",") if n.strip()],
                scrub_pii=v["scrub_pii"], scrub_urls=v["scrub_urls"],
                anonymize_names=v["anonymize_names"], skip_groups=v["skip_groups"],
                mode=v["mode"], base_model=v["base_model"].strip(),
                epochs=float(v["epochs"]), max_seq_len=int(v["max_seq_len"]),
                grad_accum=int(v["grad_accum"]), llama_cpp_dir=v["llama_cpp_dir"].strip())
        except ValueError:
            messagebox.showerror("chattrainer", "Epochs, sequence length and accumulation must be numbers.")
            return False
        config_mod.save(self.cfg, CONFIG_PATH)
        self.session = None  # settings changed: reload model on next chat
        return True

    # ---------------------------------------------------- subprocess runner
    def _run(self, args: list[str], log: Log, buttons: list[ttk.Button]) -> None:
        if self.proc or not self._save():
            return
        log.clear()
        for b in buttons:
            b.state(["disabled"])
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen([sys.executable, "-m", "chattrainer", *args], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", env=env, creationflags=flags)

        def reader(p=self.proc):
            for line in p.stdout:
                self.q.put((log, line))
            code = p.wait()
            self.q.put((log, f"\n[finished with exit code {code}]\n"))
            self.q.put((None, buttons))

        threading.Thread(target=reader, daemon=True).start()

    def _pump(self) -> None:
        try:
            while True:
                target, payload = self.q.get_nowait()
                if target is None:
                    self.proc = None
                    for b in payload:
                        b.state(["!disabled"])
                    self._refresh_counts()
                elif isinstance(target, Log):
                    target.write(payload)
                else:
                    target(payload)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    # ------------------------------------------------------------- prepare
    def _prepare_tab(self, f: ttk.Frame) -> None:
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        log = Log(f)
        b1 = ttk.Button(bar, text="Check my names")
        b2 = ttk.Button(bar, text="Build dataset")
        b1.configure(command=lambda: self._run(["stats"], log, [b1, b2]))
        b2.configure(command=lambda: self._run(["prepare"], log, [b1, b2]))
        b1.pack(side="left")
        b2.pack(side="left", padx=6)
        ttk.Label(f, text="'Check my names' shows how many messages were recognised as yours. "
                          "'Build dataset' writes output/train.jsonl.", wraplength=780,
                  foreground="#666").pack(anchor="w", pady=6)
        log.pack(fill="both", expand=True)

    # --------------------------------------------------------------- train
    def _train_tab(self, f: ttk.Frame) -> None:
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        log = Log(f)
        b1 = ttk.Button(bar, text="Start training")
        b2 = ttk.Button(bar, text="Export model (merge / GGUF)")
        stop = ttk.Button(bar, text="Stop", command=lambda: self.proc and self.proc.terminate())
        b1.configure(command=lambda: self._run(["train"], log, [b1, b2]))
        b2.configure(command=lambda: self._run(["export"], log, [b1, b2]))
        for b in (b1, b2, stop):
            b.pack(side="left", padx=(0, 6))
        ttk.Label(f, text="Needs an NVIDIA GPU and: pip install \"chattrainer[train]\". Llama 3.1 also needs a "
                          "Hugging Face login (accepted licence). Run 'Build dataset' first.",
                  wraplength=780, foreground="#666").pack(anchor="w", pady=6)
        log.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- chat
    def _chat_tab(self, f: ttk.Frame) -> None:
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        self.status = tk.StringVar(value="Model not loaded.")
        self.load_btn = ttk.Button(bar, text="Load model", command=self._load_model)
        self.load_btn.pack(side="left")
        ttk.Button(bar, text="New conversation", command=self._reset_chat).pack(side="left", padx=6)
        ttk.Label(bar, textvariable=self.status, foreground="#666").pack(side="left", padx=8)
        self.chat_log = Log(f)
        self.chat_log.pack(fill="both", expand=True, pady=6)
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.msg = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.msg)
        e.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda _e: self._send())
        self.send_btn = ttk.Button(row, text="Send", command=self._send)
        self.send_btn.pack(side="left", padx=6)

    def _load_model(self) -> None:
        if not self._save():
            return
        self.load_btn.state(["disabled"])
        self.status.set("Loading model (can take a minute)…")

        def work():
            try:
                from .export import ChatSession
                self.session = ChatSession(self.cfg)
                self.q.put((lambda _: self.status.set("Model ready."), None))
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.q.put((lambda m: self.status.set(f"Load failed: {m}"), str(e)))
            self.q.put((lambda _: self.load_btn.state(["!disabled"]), None))

        threading.Thread(target=work, daemon=True).start()

    def _reset_chat(self) -> None:
        if self.session:
            self.session.reset()
        self.chat_log.clear()

    def _send(self) -> None:
        text = self.msg.get().strip()
        if not text:
            return
        if not self.session:
            self.status.set("Load the model first.")
            return
        self.msg.set("")
        self.chat_log.write(f"you: {text}\n")
        self.send_btn.state(["disabled"])

        def work():
            try:
                out = self.session.reply(text)
            except Exception as e:  # noqa: BLE001
                out = f"[error: {e}]"
            self.q.put((self.chat_log, f"{self.cfg.name}: {out}\n\n"))
            self.q.put((lambda _: self.send_btn.state(["!disabled"]), None))

        threading.Thread(target=work, daemon=True).start()


def run() -> None:
    App().mainloop()


if __name__ == "__main__":
    run()
