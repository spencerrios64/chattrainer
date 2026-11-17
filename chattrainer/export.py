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


def chat(cfg: Config) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter = Path(cfg.output_dir) / "adapter"
    tok = AutoTokenizer.from_pretrained(adapter)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter).eval()
    history = [{"role": "system", "content": cfg.system()}]
    print("Chat with your clone. Ctrl+C to quit.")
    while True:
        try:
            history.append({"role": "user", "content": input("you> ")})
        except (KeyboardInterrupt, EOFError):
            print()
            return
        ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.9)
        reply = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"{cfg.name}> {reply}")
