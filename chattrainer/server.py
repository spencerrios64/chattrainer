"""Tiny HTTP server so the Android app can drive training on this PC over your Wi-Fi.

Run:  chattrainer serve        (or press "Phone remote" in the desktop app)
The phone needs the address and the pairing code this prints. Nothing leaves your network.
"""
from __future__ import annotations

import dataclasses
import io
import json
import secrets
import socket
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config as config_mod

MAX_UPLOAD = 200 * 1024 * 1024


def lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips or ["127.0.0.1"]


class State:
    def __init__(self, config_path: str, token: str):
        self.config_path = config_path
        self.token = token
        self.proc: subprocess.Popen | None = None
        self.log = bytearray()
        self.lock = threading.Lock()

    def cfg(self):
        return config_mod.load(self.config_path)

    def out_dir(self) -> Path:
        d = Path(self.cfg().output_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(512), b""):
            with self.lock:
                self.log += chunk
        proc.wait()
        with self.lock:
            self.log += f"\n[finished with code {proc.returncode}]\n".encode()

    def start(self, model: str | None) -> str | None:
        if self.busy():
            return "Training is already running."
        if not (self.out_dir() / "train.jsonl").exists():
            return "No dataset uploaded yet."
        if model:
            cfg = self.cfg()
            config_mod.save(dataclasses.replace(cfg, base_model=model), self.config_path)
        with self.lock:
            self.log = bytearray()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "chattrainer", "-c", self.config_path, "train"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        return None

    def stop(self) -> None:
        if self.busy():
            self.proc.terminate()

    def status(self) -> dict:
        cfg = self.cfg()
        out = Path(cfg.output_dir)
        try:
            from .models import detect_vram_gb
            vram = detect_vram_gb()
        except Exception:
            vram = None
        with self.lock:
            log_len = len(self.log)
        return {
            "busy": self.busy(),
            "model": cfg.base_model,
            "vram_gb": vram,
            "has_dataset": (out / "train.jsonl").exists(),
            "has_adapter": (out / "adapter" / "adapter_config.json").exists(),
            "log_len": log_len,
            "exit_code": None if self.proc is None or self.busy() else self.proc.returncode,
        }


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        server_version = "chattrainer"

        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode())

        def _authed(self) -> bool:
            tok = self.headers.get("X-Token", "")
            if secrets.compare_digest(tok, state.token):
                return True
            self._json(401, {"error": "wrong pairing code"})
            return False

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/ping":  # no auth: lets the phone check the address
                return self._json(200, {"app": "chattrainer"})
            if not self._authed():
                return
            q = parse_qs(u.query)
            if u.path == "/api/status":
                return self._json(200, state.status())
            if u.path == "/api/log":
                start = int(q.get("from", ["0"])[0])
                with state.lock:
                    data = bytes(state.log[start:])
                return self._send(200, data, "text/plain; charset=utf-8")
            if u.path == "/api/adapter":
                ad = Path(state.cfg().output_dir) / "adapter"
                if not ad.is_dir():
                    return self._json(404, {"error": "no adapter yet"})
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in ad.rglob("*"):
                        if f.is_file():
                            z.write(f, f.relative_to(ad).as_posix())
                return self._send(200, buf.getvalue(), "application/zip")
            self._json(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if not self._authed():
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                return self._json(413, {"error": "too large"})
            body = self.rfile.read(n) if n else b""
            q = parse_qs(u.query)
            if u.path == "/api/dataset":
                name = q.get("name", [""])[0]
                if name not in ("train", "val"):
                    return self._json(400, {"error": "name must be train or val"})
                if state.busy():
                    return self._json(409, {"error": "training is running"})
                (state.out_dir() / f"{name}.jsonl").write_bytes(body)
                return self._json(200, {"ok": True, "bytes": len(body)})
            if u.path == "/api/train":
                try:
                    model = (json.loads(body or b"{}").get("model") or "").strip() or None
                except ValueError:
                    model = None
                err = state.start(model)
                return self._json(409 if err else 200, {"error": err} if err else {"ok": True})
            if u.path == "/api/stop":
                state.stop()
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

    return H


def serve(config_path: str = "config.toml", port: int = 8765) -> None:
    out = Path(config_mod.load(config_path).output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_file = out / ".remote_token"
    token = tok_file.read_text().strip() if tok_file.exists() else ""
    if not token:
        token = f"{secrets.randbelow(10**6):06d}"
        tok_file.write_text(token)
    state = State(config_path, token)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(state))
    print("Phone remote is running. In the Android app, Train tab > PC, enter:")
    for ip in lan_ips():
        print(f"  address: {ip}:{port}")
    print(f"  pairing code: {token}")
    print("Keep this window open. Phone and PC must be on the same Wi-Fi. Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
