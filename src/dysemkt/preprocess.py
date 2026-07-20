from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import sha256_file, write_json
from .text import TextEncoder


@dataclass(frozen=True)
class Question:
    problem_id: str
    course_id: str
    exercise_id: str
    text: str
    concepts: tuple[str, ...]
    knowledge_type: int | None
    cognitive_dimension: int | None


def _answer_free_text(detail: dict, concepts: Iterable[str]) -> str:
    parts = []
    title = str(detail.get("title") or "").strip()
    content = str(detail.get("content") or "").strip()
    if title:
        parts.append(f"题目组：{title}")
    if content:
        parts.append(f"题目：{content}")
    options = detail.get("option")
    if isinstance(options, dict):
        rendered = " ".join(f"{key}. {value}" for key, value in sorted(options.items()))
        if rendered.strip():
            parts.append(f"选项：{rendered}")
    clean_concepts = [str(value).strip() for value in concepts if str(value).strip()]
    if clean_concepts:
        parts.append("概念：" + "；".join(clean_concepts))
    return "\n".join(parts)


def load_questions(path: Path) -> tuple[dict[str, Question], list[dict]]:
    questions: dict[str, Question] = {}
    rejected: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = str(record.get("problem_id") or "")
            try:
                detail = ast.literal_eval(record.get("detail") or "{}")
                if not isinstance(detail, dict):
                    raise ValueError("detail is not a dictionary")
            except (SyntaxError, ValueError) as exc:
                rejected.append({"line": line_number, "problem_id": problem_id, "reason": str(exc)})
                continue
            concepts = tuple(str(value) for value in (record.get("concepts") or []))
            text = _answer_free_text(detail, concepts)
            if not problem_id or not text:
                rejected.append({"line": line_number, "problem_id": problem_id, "reason": "missing id or text"})
                continue
            questions[problem_id] = Question(
                problem_id=problem_id,
                course_id=str(record.get("course_id") or ""),
                exercise_id=str(record.get("exercise_id") or ""),
                text=text,
                concepts=concepts,
                knowledge_type=record.get("knowledge_type"),
                cognitive_dimension=record.get("cognitive_dimension"),
            )
    return questions, rejected


def load_interactions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    interactions: list[dict] = []
    seen_logs: set[str] = set()
    for sequence in sequences:
        for record in sequence.get("seq", []):
            log_id = str(record.get("log_id") or "")
            if log_id and log_id in seen_logs:
                continue
            if log_id:
                seen_logs.add(log_id)
            interactions.append(record)
    return interactions


def _unix_time(value: str) -> int:
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _temporal_split(size: int, train_ratio: float, val_ratio: float) -> np.ndarray:
    train_end = max(1, int(size * train_ratio))
    val_end = max(train_end + 1, int(size * (train_ratio + val_ratio)))
    val_end = min(val_end, size)
    split = np.full(size, 2, dtype=np.int8)
    split[:train_end] = 0
    split[train_end:val_end] = 1
    return split


def _cold_split(item: np.ndarray, num_items: int, val_ratio: float, test_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    n_val = max(1, round(num_items * val_ratio))
    n_test = max(1, round(num_items * test_ratio))
    item_split = np.zeros(num_items, dtype=np.int8)
    item_split[order[:n_val]] = 1
    item_split[order[n_val : n_val + n_test]] = 2
    return item_split[item]


def preprocess_moocradar(
    raw_dir: Path,
    output_dir: Path,
    encoder: TextEncoder,
    interaction_file: str = "student-problem-fine.json",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_user_interactions: int = 2,
    seed: int = 42,
) -> dict:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    problem_path = raw_dir / "problem.json"
    interaction_path = raw_dir / interaction_file
    if not problem_path.exists() or not interaction_path.exists():
        raise FileNotFoundError("raw_dir must contain problem.json and the selected interaction file")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must be positive and sum to less than one")

    questions, rejected_questions = load_questions(problem_path)
    raw_interactions = load_interactions(interaction_path)
    user_counts: dict[str, int] = {}
    for record in raw_interactions:
        user_id = str(record.get("user_id") or "")
        if str(record.get("problem_id") or "") in questions and record.get("is_correct") in (0, 1):
            user_counts[user_id] = user_counts.get(user_id, 0) + 1

    rows = []
    rejected_interactions = 0
    for order, record in enumerate(raw_interactions):
        user_id = str(record.get("user_id") or "")
        problem_id = str(record.get("problem_id") or "")
        if user_counts.get(user_id, 0) < min_user_interactions or problem_id not in questions:
            rejected_interactions += 1
            continue
        try:
            timestamp = _unix_time(str(record["submit_time"]))
            label = int(record["is_correct"])
        except (KeyError, TypeError, ValueError):
            rejected_interactions += 1
            continue
        if label not in (0, 1):
            rejected_interactions += 1
            continue
        rows.append((timestamp, order, user_id, problem_id, label))
    rows.sort(key=lambda value: (value[0], value[1]))
    if len(rows) < 3:
        raise ValueError("not enough valid interactions")

    users = sorted({row[2] for row in rows})
    items = sorted({row[3] for row in rows})
    user_to_idx = {value: idx for idx, value in enumerate(users)}
    item_to_idx = {value: idx for idx, value in enumerate(items)}
    user = np.asarray([user_to_idx[row[2]] for row in rows], dtype=np.int64)
    item = np.asarray([item_to_idx[row[3]] for row in rows], dtype=np.int64)
    timestamp = np.asarray([row[0] for row in rows], dtype=np.int64)
    label = np.asarray([row[4] for row in rows], dtype=np.float32)
    temporal_split = _temporal_split(len(rows), train_ratio, val_ratio)
    cold_split = _cold_split(item, len(items), val_ratio, 1 - train_ratio - val_ratio, seed)

    texts = [questions[problem_id].text for problem_id in items]
    features = np.asarray(encoder.encode(texts), dtype=np.float32)
    if features.shape != (len(items), encoder.dimension):
        raise ValueError("text encoder returned an invalid shape")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "events.npz", user=user, item=item, timestamp=timestamp, label=label,
        temporal_split=temporal_split, cold_split=cold_split,
    )
    np.save(output_dir / "question_features.npy", features)
    with (output_dir / "question_text.jsonl").open("w", encoding="utf-8") as handle:
        for problem_id, text in zip(items, texts):
            handle.write(json.dumps({"problem_id": problem_id, "text": text}, ensure_ascii=False) + "\n")
    write_json(output_dir / "question_context.json", [
        {
            "problem_id": problem_id,
            "course_id": questions[problem_id].course_id,
            "exercise_id": questions[problem_id].exercise_id,
            "concepts": list(questions[problem_id].concepts),
            "knowledge_type": questions[problem_id].knowledge_type,
            "cognitive_dimension": questions[problem_id].cognitive_dimension,
        }
        for problem_id in items
    ])
    write_json(output_dir / "mappings.json", {"users": users, "items": items})
    metadata = {
        "format_version": 2,
        "source": "MOOCRadar",
        "interaction_file": interaction_file,
        "source_sha256": {"problem": sha256_file(problem_path), "interactions": sha256_file(interaction_path)},
        "num_events": len(rows), "num_users": len(users), "num_items": len(items),
        "semantic_dim": int(features.shape[1]), "positive_rate": float(label.mean()),
        "text_encoder": getattr(encoder, "identifier", type(encoder).__name__),
        "rejected_questions": rejected_questions,
        "rejected_interactions": rejected_interactions,
        "min_user_interactions": min_user_interactions, "seed": seed,
        "split_counts": {
            "temporal": np.bincount(temporal_split, minlength=3).tolist(),
            "cold": np.bincount(cold_split, minlength=3).tolist(),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata
