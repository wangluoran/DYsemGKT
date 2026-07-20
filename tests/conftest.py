from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def raw_moocradar(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    problems = []
    for item in range(8):
        detail = {
            "title": f"单元 {item // 2}",
            "content": f"测试题目 {item} 的正文",
            "option": {"A": "错误选项", "B": "正确选项"},
            "answer": '["B"]',
            "typetext": "单选题",
            "language": "Chinese",
        }
        problems.append({
            "problem_id": f"P_{item}",
            "exercise_id": f"E_{item // 2}",
            "course_id": f"C_{item // 4}",
            "detail": repr(detail),
            "concepts": [f"概念{item % 3}"],
            "knowledge_type": 2,
            "cognitive_dimension": 2,
        })
    problems.append({
        "problem_id": "P_bad", "exercise_id": "E_bad", "course_id": "C_bad",
        "detail": "{'content': 'broken'", "concepts": ["坏数据"],
    })
    with (raw / "problem.json").open("w", encoding="utf-8") as handle:
        for record in problems:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    base = datetime(2024, 1, 1)
    sequences = []
    for user in range(6):
        seq = []
        for step in range(12):
            moment = base + timedelta(minutes=user * 100 + step)
            seq.append({
                "log_id": f"L_{user}_{step}",
                "problem_id": f"P_{(user + step) % 8}",
                "user_id": f"U_{user}",
                "is_correct": (user + step) % 2,
                "attempts": 1,
                "score": float((user + step) % 2),
                "submit_time": moment.strftime("%Y-%m-%d %H:%M:%S"),
            })
        sequences.append({"seq": list(reversed(seq)) if user % 2 else seq})
    with (raw / "student-problem-fine.json").open("w", encoding="utf-8") as handle:
        json.dump(sequences, handle, ensure_ascii=False)
    return raw

