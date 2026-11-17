import json
import tempfile
import unittest
from pathlib import Path

from chattrainer.config import Config
from chattrainer.dataset import build
from chattrainer.scrub import scrub

WA = """12/31/23, 10:15 PM - Messages and calls are end-to-end encrypted.
12/31/23, 10:15 PM - Alex: yo what's up
12/31/23, 10:16 PM - Me: not much, call me on +30 691 234 5678
12/31/23, 10:16 PM - Me: or mail me@example.com
second line here
12/31/23, 10:17 PM - Alex: <Media omitted>
12/31/23, 10:18 PM - Alex: ok lol
12/31/23, 10:19 PM - Me: haha
"""


class T(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        for p in ("whatsapp", "instagram", "tiktok", "telegram", "discord", "snapchat"):
            (self.d / "data" / p).mkdir(parents=True)
        self.cfg = Config(names=["Me"], data_dir=str(self.d / "data"),
                          output_dir=str(self.d / "out"), val_fraction=0.0)

    def rows(self):
        stats = build(self.cfg)
        lines = (self.d / "out" / "train.jsonl").read_text().splitlines()
        return stats, [json.loads(x)["messages"] for x in lines]

    def test_scrub(self):
        s = scrub("mail a@b.co or +30 691 234 5678 see https://x.io/a")
        self.assertEqual(s, "mail <EMAIL> or <PHONE> see <URL>")

    def test_whatsapp(self):
        (self.d / "data/whatsapp/chat.txt").write_text(WA)
        stats, rows = self.rows()
        self.assertGreater(len(rows), 0)
        blob = json.dumps(rows)
        self.assertIn("<PHONE>", blob)
        self.assertIn("<EMAIL>", blob)
        self.assertNotIn("Media omitted", blob)
        self.assertIn("mail <EMAIL>\\nsecond line here", blob)
        for r in rows:
            self.assertEqual(r[-1]["role"], "assistant")
            self.assertEqual(r[1]["role"], "user")

    def test_instagram_mojibake(self):
        msgs = [{"sender_name": "Me", "timestamp_ms": 2000, "content": "cafÃ©"},
                {"sender_name": "Bob", "timestamp_ms": 1000, "content": "hey"}]
        (self.d / "data/instagram/message_1.json").write_text(json.dumps({"messages": msgs}))
        _, rows = self.rows()
        self.assertIn("café", json.dumps(rows, ensure_ascii=False))

    def test_tiktok_telegram_snap_discord(self):
        tt = {"Direct Message": {"Direct Messages": {"ChatHistory": {"Chat History with bob:": [
            {"Date": "2024-01-01 10:00:00", "From": "bob", "Content": "hi"},
            {"Date": "2024-01-01 10:01:00", "From": "Me", "Content": "hello"}]}}}}
        (self.d / "data/tiktok/user_data.json").write_text(json.dumps(tt))
        tg = {"name": "Sam", "messages": [
            {"type": "message", "date": "2024-01-01T10:00:00", "from": "Sam", "text": "yo"},
            {"type": "message", "date": "2024-01-01T10:01:00", "from": "Me", "text": ["sup ", {"type": "bold", "text": "man"}]}]}
        (self.d / "data/telegram/result.json").write_text(json.dumps(tg))
        sc = {"Received Saved Chat History": [{"From": "kim", "Media Type": "TEXT", "Created": "2024-01-01 10:00:00 UTC", "Content": "hey", "IsSender": False}],
              "Sent Saved Chat History": [{"From": "me", "Media Type": "TEXT", "Created": "2024-01-01 10:01:00 UTC", "Content": "hi kim", "Conversation Title": "kim", "IsSender": True}]}
        (self.d / "data/snapchat/chat_history.json").write_text(json.dumps(sc))
        (self.d / "data/discord/c1").mkdir()
        (self.d / "data/discord/c1/messages.json").write_text(json.dumps(
            [{"ID": 1, "Timestamp": "2024-01-01 10:00:00", "Contents": "solo message", "Attachments": ""}]))
        stats, rows = self.rows()
        blob = json.dumps(rows)
        for needle in ("hello", "sup man", "hi kim", "solo message"):
            self.assertIn(needle, blob)
        self.assertEqual(stats["chats"], 4)


if __name__ == "__main__":
    unittest.main()
