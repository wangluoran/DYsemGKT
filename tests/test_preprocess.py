from __future__ import annotations

import json

import numpy as np

from dysemkt.preprocess import preprocess_moocradar
from dysemkt.text import HashTextEncoder


def test_preprocess_builds_answer_free_chronological_contract(raw_moocradar, tmp_path):
    output = tmp_path / "processed"
    metadata = preprocess_moocradar(raw_moocradar, output, HashTextEncoder(32), seed=7)

    assert metadata["num_events"] == 72
    assert metadata["num_users"] == 6
    assert metadata["num_items"] == 8
    assert metadata["semantic_dim"] == 32
    assert metadata["rejected_questions"][0]["problem_id"] == "P_bad"

    with np.load(output / "events.npz") as events:
        assert np.all(np.diff(events["timestamp"]) >= 0)
        for split_name in ("temporal_split", "cold_split"):
            assert set(events[split_name].tolist()) == {0, 1, 2}
        cold = events["cold_split"]
        item = events["item"]
        sets = [set(item[cold == code].tolist()) for code in range(3)]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])

    texts = [json.loads(line)["text"] for line in (output / "question_text.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("测试题目" in text for text in texts)
    assert all("正确选项" in text for text in texts)
    assert all('答案' not in text and '["B"]' not in text for text in texts)


def test_hash_encoder_is_deterministic_and_normalized():
    encoder = HashTextEncoder(24)
    first = encoder.encode(["相同文本", "不同文本"])
    second = encoder.encode(["相同文本", "不同文本"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)

