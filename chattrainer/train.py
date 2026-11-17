"""QLoRA fine-tuning with loss only on YOUR replies. Needs the [train] extra."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def prepare_tokenizer(tok):
    """Make odd checkpoints (merges, abliterated/Heretic variants, base models) usable."""
    tok.pad_token = tok.pad_token or tok.eos_token
    if not getattr(tok, "chat_template", None):  # fall back to ChatML
        tok.chat_template = (
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return tok


def render(tok, msgs: list[dict], **kw):
    """apply_chat_template, folding the system prompt into the first user turn for
    templates that reject the system role (e.g. Gemma 2)."""
    try:
        return tok.apply_chat_template(msgs, **kw)
    except Exception:
        if msgs and msgs[0]["role"] == "system" and len(msgs) > 1:
            first = {"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}
            return tok.apply_chat_template([first] + msgs[2:], **kw)
        raise


def encode(tok, messages: list[dict], max_len: int):
    """Tokenize a chat; label only the assistant turns (everything else is -100).
    Returns None when the model's template does not render turns as a clean prefix chain."""
    msgs = list(messages)
    while True:
        full = render(tok, msgs, tokenize=False)
        spans = []
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            pre = render(tok, msgs[:i], tokenize=False, add_generation_prompt=True)
            upto = render(tok, msgs[: i + 1], tokenize=False)
            if not (full.startswith(upto) and upto.startswith(pre)):
                return None
            spans.append((len(pre), len(upto)))
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= max_len or len(msgs) <= 3:
            break
        msgs = [msgs[0]] + msgs[3:]  # drop oldest user+assistant pair, keep the system prompt
    ids = enc["input_ids"][:max_len]
    labels = [
        tid if any(s <= off[0] < e for s, e in spans) else -100
        for tid, off in zip(ids, enc["offset_mapping"][:max_len])
    ]
    return {"input_ids": ids, "labels": labels}


def _load(tok, path: Path, max_len: int):
    rows, skipped = [], 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            ex = encode(tok, json.loads(line)["messages"], max_len)
            if ex is None:
                skipped += 1
            elif any(x != -100 for x in ex["labels"]):
                rows.append(ex)
    if skipped:
        print(f"Skipped {skipped} examples this model's chat template cannot mask cleanly")
    return rows


def run(cfg: Config) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    out = Path(cfg.output_dir)
    if not (out / "train.jsonl").exists():
        raise SystemExit("output/train.jsonl missing. Run `chattrainer prepare` first.")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Train on a GPU machine or a Colab runtime.")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    prepare_tokenizer(tok)
    train_rows = _load(tok, out / "train.jsonl", cfg.max_seq_len)
    val_rows = _load(tok, out / "val.jsonl", cfg.max_seq_len)
    if not train_rows:
        raise SystemExit("No usable training examples. Check names in config.toml (`chattrainer stats`).")
    print(f"{len(train_rows)} train / {len(val_rows)} val examples")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16) if cfg.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto")
    if quant:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, task_type="CAUSAL_LM",
        target_modules="all-linear"))
    model.print_trainable_parameters()

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        }

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=10,
        save_strategy="epoch", save_total_limit=2, eval_strategy="epoch" if val_rows else "no",
        gradient_checkpointing=True, optim="paged_adamw_8bit" if quant else "adamw_torch",
        report_to="none", seed=cfg.seed, remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=val_rows or None,
            data_collator=collate).train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    print(f"Adapter saved to {out / 'adapter'}")
