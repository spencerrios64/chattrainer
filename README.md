# chattrainer

Fine-tune a local LLM on **your own** chat exports so it texts like you. Parses WhatsApp, Instagram, TikTok, Telegram, Discord and Snapchat exports, scrubs personal data, builds a chat-format dataset, trains a QLoRA adapter (loss only on your replies) and exports to GGUF for LM Studio.

Your chats never leave your machine and `data/`, `output/` and `config.toml` are git-ignored.

## Install (easiest)

**Windows:** on the repo page click Code, Download ZIP, unzip it, then double-click `install.bat`. It creates a private environment, installs chattrainer and puts a shortcut on your desktop. Start the app from the shortcut (or `run.bat`).

**macOS / Linux:** download or clone the repo, then run `bash install.sh` and start the app with `./run.sh`.

Both need Python 3.11+ (Windows: `winget install Python.Python.3.12`). Then, in the app's Train tab, press **Install training components** once. It installs PyTorch with CUDA and the rest (needs an NVIDIA GPU; several GB download). From a terminal the same thing is `chattrainer install-train`.

Prefer pip? `pip install "chattrainer @ git+https://github.com/spencerrios64/chattrainer"` then `chattrainer gui`.

## Quick start (command line)

```bash
git clone https://github.com/spencerrios64/chattrainer && cd chattrainer
pip install -e .            # parsing/dataset only, no dependencies
chattrainer init            # creates config.toml and data/<platform>/ folders
# 1. put your exports in data/whatsapp, data/instagram, ...
# 2. edit config.toml: your names
chattrainer stats           # shows how many messages were recognised as yours
chattrainer prepare         # writes output/train.jsonl + val.jsonl

chattrainer install-train   # PyTorch (CUDA), transformers, peft, bitsandbytes
huggingface-cli login       # Llama 3.1 is gated: accept the licence first
chattrainer train           # QLoRA, fits 12 GB VRAM (RTX 3060)
chattrainer chat            # test your clone in the terminal
chattrainer export          # merge adapter (+ GGUF if llama_cpp_dir is set)
```

Requires Python 3.11+. Training needs an NVIDIA GPU (on Windows, install a CUDA build of PyTorch first; `bitsandbytes` 4-bit support on Windows needs a recent version). Merging an 8B model on CPU needs about 16 GB of free RAM, so close other apps or run `export` on Colab.

## Desktop app

```bash
pip install -e .
chattrainer gui        # or: chattrainer-gui
```

A Tkinter window (no extra dependencies; included with Python on Windows and macOS, on Linux install `python3-tk`) with five tabs: add your exports, settings, build the dataset, train/export with live logs, and chat with your clone. It works in the folder you launch it from (config.toml, data/ and output/ live there). Training still needs the `[train]` extra and an NVIDIA GPU.

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

## Choosing a model for your PC

`chattrainer models` lists presets for three tiers and detects your GPU; `chattrainer use <number or id>` sets the one you pick. The desktop app has the same thing in Settings ("My PC" tier, "Detect my GPU", model dropdown). VRAM figures are rough minimums for 4-bit QLoRA at 1024 tokens.

| Tier | VRAM | Presets |
|---|---|---|
| Low | 4-6 GB | Qwen2.5 1.5B / 3B, Llama 3.2 1B / 3B, Gemma 3 1B |
| Mid | 8-12 GB | Qwen3 4B, Qwen2.5 7B, Llama 3.1 8B (default), Mistral 7B, Gemma 3 12B, Gemma 3 12B Heretic |
| High | 16 GB+ | Qwen2.5 14B (default), Gemma 3 27B, Qwen2.5 32B, Llama 3.3 70B (48 GB or multi-GPU) |

Any other Hugging Face id or local folder works too.

## Heretic and other uncensored models

Heretic models are ordinary Hugging Face checkpoints with the refusal direction removed, so chattrainer trains them like any other model. Set `base_model` in `config.toml` (or in the app's Settings tab) to the repo id of the model you want, for example `p-e-w/gemma-3-12b-it-heretic`, or to a local folder if you made one with `heretic` yourself. Most Heretic repos are not gated, so no Hugging Face login is needed.

- Pick a size that fits your GPU: with 4-bit QLoRA an 8B model fits 12 GB comfortably, a 12B model is tight, 4B is easy.
- Chat templates differ between families. chattrainer falls back to ChatML when a model has none, folds the system prompt into the first user turn for templates that reject it, and skips examples a template cannot mask cleanly (it prints how many).
- LoRA targets all linear layers, so it works across Llama, Qwen, Gemma and similar architectures.
- Heretic only removes refusals; your model will still copy whatever is in the chats. Review the training data first.

## Dataset

- Consecutive messages from one person within `merge_gap_minutes` become one turn; a gap over `session_gap_hours` starts a new conversation.
- `mode = "pairs"`: the last `context_turns` turns are the prompt and your next reply is the target.
- `mode = "windows"`: longer multi-turn chunks, trained on all your replies in them.
- `mode = "both"` (default) builds both.
- Chats are split into train/val by chat, so a conversation never appears on both sides.

## Train on your AI chats (Claude / ChatGPT)

Build a clean fine-tuning set from your Claude or ChatGPT export (`conversations.json`, a folder, or the export `.zip`):

```
chattrainer ai-dataset path/to/conversations.json --title docker --min-turns 4
```

It scrubs emails, phone numbers and similar details, drops short or low-effort chats, keeps each conversation entirely in train or validation, and writes `output/ai_train.jsonl` and `output/ai_val.jsonl`. Use `--train-on user` to learn your side of the conversation instead of the AI's answers.

## Privacy

With `scrub_pii = true` emails, phone numbers, IBANs, card-like numbers and URLs become placeholders. Options: `anonymize_names`, `skip_groups`. Scrubbing is regex-based and will miss things such as street addresses or nicknames: review `output/train.jsonl` before sharing the dataset or any model trained on it. A model trained on private chats can memorise and repeat them, so do not publish the adapter or GGUF without thinking about who is in the data.

## Status

Parsers, scrubbing and dataset building are covered by tests (`python -m unittest discover -s tests -t .`). The training, export and chat code follows the standard transformers/peft QLoRA flow but has not been run end to end yet; if something breaks on your setup, open an issue.

## License

MIT
