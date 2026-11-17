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
from .models import TIER_LABELS, TIERS, default_for, describe, detect_vram_gb, best_heretic, find, heretic_for, is_heretic, models_for, tier_for, TIER_VRAM
from .parsers import PARSERS

CONFIG_PATH = Path("config.toml")
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
                             ("3. Models", self._models_tab), ("4. Prepare", self._prepare_tab),
                             ("5. Train", self._train_tab), ("6. Chat", self._chat_tab)]:
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
        tr = ttk.Frame(f)
        tr.pack(fill="x", pady=2)
        ttk.Label(tr, text="My PC", width=24).pack(side="left")
        cur = find(c.base_model)
        self.tier = tk.StringVar(value=cur.tier if cur else "custom")
        tcb = ttk.Combobox(tr, textvariable=self.tier, values=[*TIERS, "custom"], state="readonly", width=10)
        tcb.pack(side="left")
        ttk.Button(tr, text="Detect my GPU", command=self._detect).pack(side="left", padx=6)
        self.gpu_info = tk.StringVar(value="low = 4-6 GB VRAM, mid = 8-12 GB, high = 16 GB+")
        ttk.Label(tr, textvariable=self.gpu_info, foreground="#666").pack(side="left")
        r = ttk.Frame(f)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="Base model", width=24).pack(side="left")
        self.base = tk.StringVar(value=c.base_model)
        self.heretic = tk.BooleanVar(value=is_heretic(c.base_model))
        self.model_cb = ttk.Combobox(r, textvariable=self.base, width=48,
                                     values=[p.id for p in models_for(self.tier.get(), self.heretic.get())])
        self.model_cb.pack(side="left")
        ttk.Checkbutton(r, text="Heretic: pick uncensored models for my PC", variable=self.heretic,
                        command=self._tier_changed).pack(side="left", padx=8)
        self.v["base_model"] = self.base
        self.model_info = tk.StringVar()
        ttk.Label(f, textvariable=self.model_info, foreground="#666").pack(anchor="w")
        tcb.bind("<<ComboboxSelected>>", lambda _e: self._tier_changed())
        self.model_cb.bind("<<ComboboxSelected>>", lambda _e: self._show_model())
        self.model_cb.bind("<KeyRelease>", lambda _e: self._show_model())
        self._show_model()
        entry("Epochs", "epochs", c.epochs, 10)
        entry("Max sequence length", "max_seq_len", c.max_seq_len, 10)
        entry("Gradient accumulation", "grad_accum", c.grad_accum, 10)
        entry("llama.cpp folder (GGUF)", "llama_cpp_dir", c.llama_cpp_dir)
        ttk.Button(f, text="Save settings", command=self._save).pack(anchor="w", pady=12)

    def _pc_vram(self) -> float:
        v = detect_vram_gb()
        if v is not None:
            return v
        return TIER_VRAM.get(self.tier.get(), 12.0)

    def _refresh_list(self) -> None:
        if self.heretic.get():
            self.model_cb["values"] = [p.id for p in heretic_for(self._pc_vram())]
        else:
            self.model_cb["values"] = [p.id for p in models_for(self.tier.get())]

    def _tier_changed(self) -> None:
        if self.heretic.get():
            vram = self._pc_vram()
            best = best_heretic(vram)
            self._refresh_list()
            if best is None:
                self.gpu_info.set(f"No Heretic preset fits {vram:.0f} GB. Type any Heretic model id instead.")
            else:
                self.base.set(best.id)
                self.gpu_info.set(f"Heretic models that fit about {vram:.0f} GB VRAM; picked the biggest.")
            self._show_model()
            return
        t = self.tier.get()
        ids = [p.id for p in models_for(t)]
        self.model_cb["values"] = ids
        if t in TIERS and self.base.get() not in ids:
            self.base.set(default_for(t).id)
        self._show_model()

    def _show_model(self) -> None:
        mid = self.base.get()
        p = find(mid)
        if is_heretic(mid) and not self.heretic.get():
            self.heretic.set(True)  # a Heretic model was picked or typed
            self._refresh_list()
        self.model_info.set(describe(p) if p else
                            ("Heretic model detected. " if is_heretic(mid) else "") +
                            "Custom model: any Hugging Face id or a local folder works.")

    def _detect(self) -> None:
        vram = detect_vram_gb()
        if vram is None:
            self.gpu_info.set("No NVIDIA GPU found. Pick a tier by hand.")
            return
        self.tier.set(tier_for(vram))
        self.gpu_info.set(f"Found about {vram:.0f} GB VRAM -> {TIER_LABELS[self.tier.get()]}")
        self._tier_changed()

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
    def _run(self, args: list[str], log: Log, buttons: list[ttk.Button], extra_env: dict | None = None) -> None:
        if self.proc or not self._save():
            return
        log.clear()
        for b in buttons:
            b.state(["disabled"])
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(extra_env or {})}
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
                    if hasattr(self, "model_tree"):
                        self._refresh_models()
                elif isinstance(target, Log):
                    target.write(payload)
                else:
                    target(payload)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    # -------------------------------------------------------------- models
    def _models_tab(self, f: ttk.Frame) -> None:
        from .download import is_downloaded  # noqa: F401 (also used in _refresh_models)
        ttk.Label(f, text="Download a model before training so training starts right away. "
                          "Pick one, press Download, then 'Use for training'.", wraplength=780).pack(anchor="w")
        cols = ("model", "size", "fits", "status", "pick")
        self.model_tree = ttk.Treeview(f, columns=cols, show="headings", height=11, selectmode="browse")
        for c, t, w in (("model", "Model", 300), ("size", "Size / VRAM", 140), ("fits", "Fits my GPU", 80),
                        ("status", "Downloaded", 80), ("pick", "Recommended", 120)):
            self.model_tree.heading(c, text=t)
            self.model_tree.column(c, width=w, anchor="w")
        self.model_tree.pack(fill="x", pady=6)
        self.model_vram = detect_vram_gb()
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        log = Log(f)
        log.configure(height=8)
        rec = ttk.Frame(f)
        rec.pack(fill="x", before=bar)
        r1 = ttk.Button(rec, text="Download recommended (normal)",
                        command=lambda: self._download_rec(False, log))
        r2 = ttk.Button(rec, text="Download recommended (Heretic)",
                        command=lambda: self._download_rec(True, log))
        r1.pack(side="left", padx=(0, 6))
        r2.pack(side="left")
        b1 = ttk.Button(bar, text="Download selected")
        b2 = ttk.Button(bar, text="Use for training", command=self._use_selected)
        b3 = ttk.Button(bar, text="Refresh", command=self._refresh_models)
        self.dl_btns = [b1, r1, r2]
        b1.configure(command=lambda: self._download(self._selected_model(), log))
        for b in (b1, b2, b3):
            b.pack(side="left", padx=(0, 6))
        self.models_msg = tk.StringVar(value="Heretic (uncensored) models are marked in the Size column.")
        ttk.Label(bar, textvariable=self.models_msg, foreground="#666").pack(side="left", padx=6)
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(8, 2))
        ttk.Label(r, text="Other Hugging Face id", width=22).pack(side="left")
        self.custom_id = tk.StringVar()
        ttk.Entry(r, textvariable=self.custom_id, width=44).pack(side="left")
        b4 = ttk.Button(r, text="Download", command=lambda: self._download(self.custom_id.get(), log))
        b4.pack(side="left", padx=6)
        self.dl_btns.append(b4)
        t = ttk.Frame(f)
        t.pack(fill="x", pady=2)
        ttk.Label(t, text="Hugging Face token", width=22).pack(side="left")
        self.hf_token = tk.StringVar()
        ttk.Entry(t, textvariable=self.hf_token, width=44, show="*").pack(side="left")
        b5 = ttk.Button(t, text="Log in", command=lambda: self._hf_login(log))
        b5.pack(side="left", padx=6)
        self.dl_btns.append(b5)
        ttk.Label(f, text="Llama and Gemma are gated: accept the licence on the model's huggingface.co page, then make a "
                          "'Read' token at huggingface.co/settings/tokens and log in once. Qwen and Mistral-style models "
                          "mostly need no login.", wraplength=780, foreground="#666").pack(anchor="w", pady=4)
        log.pack(fill="both", expand=True)
        self._refresh_models()

    def _refresh_models(self) -> None:
        from .download import is_downloaded
        from .models import PRESETS
        vram = self.model_vram
        rec_n, rec_h = self._recommended()
        sel = self._selected_model()
        self.model_tree.delete(*self.model_tree.get_children())
        rows = sorted(PRESETS, key=lambda p: (vram is not None and p.vram > vram + 0.5, p.vram))
        for p in rows:
            fits = "?" if vram is None else ("yes" if p.vram <= vram + 0.5 else "no")
            tag = " [Heretic]" if is_heretic(p.id) else ""
            self.model_tree.insert("", "end", iid=p.id, values=(
                p.id, f"{p.params}, ~{p.vram} GB{tag}", fits, "yes" if is_downloaded(p.id) else "no",
                "* best normal" if p.id == rec_n else "* best Heretic" if p.id == rec_h else ""))
        if sel and self.model_tree.exists(sel):
            self.model_tree.selection_set(sel)

    def _recommended(self) -> tuple[str, str]:
        """Ids of the recommended normal and Heretic model for this PC."""
        vram = self.model_vram
        tier = tier_for(vram) if vram is not None else "mid"
        normal = default_for(tier).id
        h = best_heretic(vram if vram is not None else TIER_VRAM[tier])
        return normal, (h.id if h else "")

    def _download_rec(self, heretic: bool, log: Log) -> None:
        mid = self._recommended()[1 if heretic else 0]
        if not mid:
            messagebox.showinfo("chattrainer", "No Heretic preset fits this GPU. Type a Heretic model id below instead.")
            return
        if self.model_tree.exists(mid):
            self.model_tree.selection_set(mid)
        self._download(mid, log)

    def _selected_model(self) -> str:
        s = self.model_tree.selection()
        return s[0] if s else ""

    def _download(self, model_id: str, log: Log) -> None:
        if not model_id.strip():
            messagebox.showinfo("chattrainer", "Select a model in the list (or type a Hugging Face id) first.")
            return
        self._run(["download", model_id.strip()], log, self.dl_btns)

    def _hf_login(self, log: Log) -> None:
        if not self.hf_token.get().strip():
            messagebox.showinfo("chattrainer", "Paste your Hugging Face token first.")
            return
        self._run(["login"], log, self.dl_btns, extra_env={"HF_TOKEN": self.hf_token.get().strip()})
        self.hf_token.set("")

    def _use_selected(self) -> None:
        mid = self._selected_model()
        if not mid:
            messagebox.showinfo("chattrainer", "Select a model in the list first.")
            return
        self.base.set(mid)
        self._show_model()
        if self._save():
            self.models_msg.set(f"Training will use {mid}.")

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
        b3 = ttk.Button(bar, text="Install training components")
        stop = ttk.Button(bar, text="Stop", command=lambda: self.proc and self.proc.terminate())
        b1.configure(command=lambda: self._run(["train"], log, [b1, b2, b3]))
        b2.configure(command=lambda: self._run(["export"], log, [b1, b2, b3]))
        b3.configure(command=lambda: self._run(["install-train"], log, [b1, b2, b3]))
        b4 = ttk.Button(bar, text="Phone remote")
        b4.configure(command=lambda: self._run(["serve"], log, [b4]))
        for b in (b3, b1, b2, b4, stop):
            b.pack(side="left", padx=(0, 6))
        ttk.Label(f, text="First time: press 'Install training components' (NVIDIA GPU needed). Gated models such as "
                          "Llama also need a Hugging Face login. Run 'Build dataset' first.",
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
