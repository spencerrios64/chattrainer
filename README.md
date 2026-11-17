# chattrainer

Fine-tune a local LLM on **your own** chat exports so it texts like you. Parses WhatsApp, Instagram, TikTok, Telegram, Discord and Snapchat exports, scrubs personal data, builds a chat-format dataset, trains a QLoRA adapter (loss only on your replies) and exports to GGUF for LM Studio.

Your chats never leave your machine and `data/`, `output/` and `config.toml` are git-ignored.

## Quick start

```bash
git clone https://github.com/spencerrios64/chattrainer && cd chattrainer
pip install -e .            # parsing/dataset only, no dependencies
chattrainer init            # creates config.toml and data/<platform>/ folders
# 1. put your exports in data/whatsapp, data/instagram, ...
# 2. edit config.toml: your names
chattrainer stats           # shows how many messages were recognised as yours
chattrainer prepare         # writes output/train.jsonl + val.jsonl

pip install -e ".[train]"   # torch, transformers, peft, bitsandbytes
huggingface-cli login       # Llama 3.1 is gated: accept the licence first
chattrainer train           # QLoRA, fits 12 GB VRAM (RTX 3060)
chattrainer chat            # test your clone in the terminal
chattrainer export          # merge adapter (+ GGUF if llama_cpp_dir is set)
```

Requires Python 3.11+. Training needs an NVIDIA GPU (on Windows, install a CUDA build of PyTorch first; `bitsandbytes` 4-bit support on Windows needs a recent version). Merging an 8B model on CPU needs about 16 GB of free RAM, so close other apps or run `export` on Colab.

## Where to put each export

| Folder | Export | How to get it |
|---|---|---|
| `data/whatsapp/` | `.txt` chat files | Chat → ⋮ → More → Export chat → Without media |
| `data/instagram/` | `message_*.json` (HTML best effort) | Accounts Center → Download your information → Messages → **JSON** |
| `data/tiktok/` | `user_data_tiktok.json` or `.txt` | Settings → Account → Download your data |
| `data/telegram/` | `result.json` | Telegram Desktop → Settings → Advanced → Export Telegram data → **JSON** |
| `data/discord/` | official package `messages/c*/messages.json` or `.csv`, or DiscordChatExporter JSON | Settings → Privacy → Request all of my data |
| `data/snapchat/` | `json/chat_history.json` | accounts.snapchat.com → My Data |

Notes:
- **Discord's official package and Snapchat's saved history can contain only your side.** Those become standalone style samples (`include_solo`) rather than reply pairs.
- Instagram HTML parsing depends on Meta's CSS class names and may break; choose JSON.
- Voice notes, images, stickers and deleted messages are skipped.

## How your messages are identified

Set every display name or username that is you in `config.toml`:

```toml
[identity]
names = ["Your Name", "your_username"]
```

`chattrainer stats` lists the top senders per platform and warns when none match.

## Dataset

- Consecutive messages from one person within `merge_gap_minutes` become one turn; a gap over `session_gap_hours` starts a new conversation.
- `mode = "pairs"`: the last `context_turns` turns are the prompt and your next reply is the target.
- `mode = "windows"`: longer multi-turn chunks, trained on all your replies in them.
- `mode = "both"` (default) builds both.
- Chats are split into train/val by chat, so a conversation never appears on both sides.

## Privacy

With `scrub_pii = true` emails, phone numbers, IBANs, card-like numbers and URLs become placeholders. Options: `anonymize_names`, `skip_groups`. Scrubbing is regex-based and will miss things such as street addresses or nicknames: review `output/train.jsonl` before sharing the dataset or any model trained on it. A model trained on private chats can memorise and repeat them, so do not publish the adapter or GGUF without thinking about who is in the data.

## Status

Parsers, scrubbing and dataset building are covered by tests (`python -m unittest discover -s tests -t .`). The training, export and chat code follows the standard transformers/peft QLoRA flow but has not been run end to end yet; if something breaks on your setup, open an issue.

## License

MIT
