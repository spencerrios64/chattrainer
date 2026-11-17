"""Merge the LoRA adapter into the base model, optionally convert to GGUF, and chat test."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import Config


def merge(cfg: Config) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(cfg.output_dir)
    adapter = out / "adapter"
    if not adapter.exists():
        raise SystemExit("output/adapter missing. Run `chattrainer train` first.")
    print("Merging on CPU (needs ~16 GB free RAM for an 8B model)...")
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16,
                                                device_map="cpu", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged = out / "merged"
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    print(f"Merged model: {merged}")

    if cfg.llama_cpp_dir:
        lc = Path(cfg.llama_cpp_dir)
        f16 = out / "model-f16.gguf"
        subprocess.run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(merged),
                        "--outfile", str(f16), "--outtype", "f16"], check=True)
        quant = next((p for p in (lc / "build/bin/llama-quantize", lc / "llama-quantize") if p.exists()), None)
        if quant:
            q = out / f"model-{cfg.gguf_quant}.gguf"
            subprocess.run([str(quant), str(f16), str(q), cfg.gguf_quant], check=True)
            print(f"GGUF ready: {q}  (load it in LM Studio)")
        else:
            print(f"f16 GGUF at {f16}; build llama.cpp's llama-quantize to shrink it.")
    return merged


class ChatSession:
    """Loads base model + adapter once; used by the CLI chat and the desktop app."""

    def __init__(self, cfg: Config):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg, self.torch = cfg, torch
        adapter = Path(cfg.output_dir) / "adapter"
        if not adapter.exists():
            raise RuntimeError("output/adapter missing. Train a model first.")
        from .train import prepare_tokenizer
        self.tok = prepare_tokenizer(AutoTokenizer.from_pretrained(adapter))
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=quant, device_map="auto")
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.reset()

    def reset(self) -> None:
        self.history = [{"role": "system", "content": self.cfg.system()}]

    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        from .train import render
        ids = render(self.tok, self.history, add_generation_prompt=True,
                     return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.9)
        out = self.tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        self.history.append({"role": "assistant", "content": out})
        return out


def chat(cfg: Config) -> None:
    session = ChatSession(cfg)
    print("Chat with your clone. Ctrl+C to quit.")
    while True:
        try:
            line = input("you> ")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        print(f"{cfg.name}> {session.reply(line)}")
