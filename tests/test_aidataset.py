import json
from chattrainer.aidataset import build_ai_dataset, load_chats


def _write(tmp_path):
    claude = [{"uuid": "c1", "name": "Docker help", "chat_messages": [
        {"sender": "human", "text": "mail me at a@b.com about docker"},
        {"sender": "assistant", "text": "x" * 250},
        {"sender": "human", "text": "thanks, and compose?"},
        {"sender": "assistant", "text": "y" * 250}]}]
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps(claude), encoding="utf-8")
    return f


def test_load_and_build(tmp_path):
    f = _write(tmp_path)
    assert load_chats(f)[0]["ai"] == "Claude"
    r = build_ai_dataset(f, tmp_path / "out", val_fraction=0)
    assert r["chats_used"] == 1 and r["train_examples"] == 1
    row = json.loads((tmp_path / "out" / "ai_train.jsonl").read_text().splitlines()[0])
    assert "<EMAIL>" in row["messages"][1]["content"]
    assert row["messages"][-1]["role"] == "assistant"
