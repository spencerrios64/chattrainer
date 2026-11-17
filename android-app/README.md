# chattrainer for Android

Android port of the data side of chattrainer. It reads your chat exports on the phone
(WhatsApp .txt, Telegram result.json, Instagram message_*.json), scrubs private details,
and builds `train.jsonl` and `val.jsonl` in the same format as the desktop app.

Fine-tuning an 8B model needs an NVIDIA GPU, so it does not run on a phone. Send the two
files to the PC app, put them in its `output` folder and press **Start training** on the Train tab.

Open this folder in Android Studio and press Run. Needs Android 8.0+ (API 26).
