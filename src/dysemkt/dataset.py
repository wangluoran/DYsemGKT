from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_json


def compute_global_stats(data: ProcessedData, train_mask: np.ndarray) -> np.ndarray:
    """Compute per-question global statistics from training data only.

    Returns (num_items, 3): [avg_correctness, log(1+attempts), avg_log_time]
    """
    num_items = len(data.question_features)
    correct = np.zeros(num_items, dtype=np.float64)
    counts = np.zeros(num_items, dtype=np.float64)
    time_sum = np.zeros(num_items, dtype=np.float64)

    train_idx = np.flatnonzero(train_mask)
    items = data.item[train_idx]
    labels = data.label[train_idx]
    timestamps = data.timestamp[train_idx]

    for i, it in enumerate(items):
        correct[it] += labels[i]
        counts[it] += 1.0
        # Average inter-event time as a proxy for time spent
        if i > 0 and train_idx[i] == train_idx[i - 1] + 1:
            pass  # not computing per-item time for now, use 0
    # Approximate: use overall timestamp span per item
    for it in range(num_items):
        item_events = np.flatnonzero((items == it))
        if len(item_events) > 1:
            time_sum[it] = float(timestamps[item_events[-1]] - timestamps[item_events[0]])

    avg_correct = np.divide(correct, counts, where=counts > 0, out=np.full_like(correct, 0.5))
    log_attempts = np.log1p(counts) / 10.0
    avg_log_time = np.log1p(np.divide(time_sum, counts, where=counts > 0, out=np.zeros_like(time_sum))) / 16.0

    return np.stack([avg_correct, log_attempts, avg_log_time], axis=-1).astype(np.float32)


class ProcessedData:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        with np.load(self.directory / "events.npz") as values:
            for name in values.files:
                setattr(self, name, values[name])
        self.question_features = np.load(self.directory / "question_features.npy").astype(np.float32)
        self.metadata = read_json(self.directory / "metadata.json")
        self.mappings = read_json(self.directory / "mappings.json")
        self.question_context = read_json(self.directory / "question_context.json")
        size = len(self.user)
        if any(len(getattr(self, name)) != size for name in ("item", "timestamp", "label", "temporal_split", "cold_split")):
            raise ValueError("event arrays have inconsistent lengths")
        if np.any(np.diff(self.timestamp) < 0):
            raise ValueError("events must be globally chronological")
        if len(self.question_context) != len(self.question_features):
            raise ValueError("question context and feature rows are not aligned")
        if [row["problem_id"] for row in self.question_context] != self.mappings["items"]:
            raise ValueError("question context does not follow the dense item mapping")
        exercise_values = sorted({str(row.get("exercise_id") or "") for row in self.question_context})
        exercise_to_idx = {value: idx for idx, value in enumerate(exercise_values) if value}
        self.item_exercise = np.asarray([
            exercise_to_idx.get(str(row.get("exercise_id") or ""), -1)
            for row in self.question_context
        ], dtype=np.int64)
        self.item_concepts = [
            frozenset(str(value) for value in (row.get("concepts") or []) if str(value).strip())
            for row in self.question_context
        ]


def build_history_cache(
    data: ProcessedData,
    allowed_mask: np.ndarray,
    history_length: int = 40,
    cache_path: Path | None = None,
    retrieval: str = "hybrid",
) -> np.ndarray:
    """Precompute retrieval results for all events.

    retrieval="hybrid": forced recent 16 + structure-scored top 24 (N=40)
    retrieval="recent": simple last N (pure recency baseline)

    Returns (num_events, history_length) int64 array of event indices, with -1 for padding.
    """
    # Build user_events from allowed_mask (same as TemporalHistoryDataset)
    user_events: dict[int, list[int]] = {}
    for event in np.flatnonzero(allowed_mask):
        user_events.setdefault(int(data.user[event]), []).append(int(event))

    n_total = len(data.user)
    cache = np.full((n_total, history_length), -1, dtype=np.int64)

    if retrieval == "recent":
        # Pure recent-N: just take the last history_length items
        for event in range(n_total):
            user = int(data.user[event])
            values = user_events.get(user, [])
            end = bisect_left(values, event)
            all_history = values[:end]
            if all_history:
                selected = all_history[-history_length:]
                cache[event, -len(selected):] = selected
            if (event + 1) % 100_000 == 0:
                print(f"  recent cache: {event+1}/{n_total} events processed", flush=True)
        print(f"  recent cache: {n_total}/{n_total} events done", flush=True)
        if cache_path is not None:
            np.save(cache_path, cache)
            print(f"  recent cache saved to {cache_path}", flush=True)
        return cache

    n_forced = min(16, history_length)
    n_scored = history_length - n_forced
    candidate_pool = 256

    for event in range(n_total):
        user = int(data.user[event])
        item = int(data.item[event])
        values = user_events.get(user, [])
        end = bisect_left(values, event)
        all_history = values[:end]

        if len(all_history) <= n_forced:
            selected = all_history
        else:
            recent = all_history[-n_forced:]
            remaining = all_history[:-n_forced]
            if len(remaining) > candidate_pool:
                remaining = remaining[-candidate_pool:]

            scores: list[tuple[int, float]] = []
            current_exercise = int(data.item_exercise[item])
            current_concepts = data.item_concepts[item]

            for e in remaining:
                h_item = int(data.item[e])
                if h_item == item:                     # exclude same question
                    continue

                se = 0.0
                if current_exercise >= 0:
                    h_exercise = int(data.item_exercise[h_item])
                    if h_exercise >= 0 and h_exercise == current_exercise:
                        se = 1.0

                co = 0.0
                h_concepts = data.item_concepts[h_item]
                union = current_concepts | h_concepts
                if union:
                    co = len(current_concepts & h_concepts) / len(union)

                score = 1.5 * 0 + 1.0 * se + 0.8 * co  # I_same_q(=0 excluded)+I_same_ex+concept_overlap
                if score > 0:
                    scores.append((e, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_scored = [e for e, _ in scores[:n_scored]]
            selected = sorted(set(recent + top_scored))

        if selected:
            cache[event, -len(selected):] = selected

        if (event + 1) % 100_000 == 0:
            print(f"  history cache: {event+1}/{n_total} events processed", flush=True)

    print(f"  history cache: {n_total}/{n_total} events done", flush=True)

    if cache_path is not None:
        np.save(cache_path, cache)
        print(f"  history cache saved to {cache_path}", flush=True)

    return cache


class TemporalHistoryDataset(Dataset):
    """Creates leak-safe two-sided histories for selected prediction events."""

    def __init__(
        self,
        data: ProcessedData,
        indices: np.ndarray,
        history_length: int = 40,
        allowed_history: np.ndarray | None = None,
        global_stats: np.ndarray | None = None,
        history_cache: np.ndarray | None = None,
        retrieval: str = "hybrid",
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.history_length = history_length
        self.retrieval = retrieval
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if allowed_history is None:
            allowed_history = np.ones(len(data.user), dtype=bool)
        allowed = np.asarray(allowed_history, dtype=bool)
        self.user_events: dict[int, list[int]] = {}
        self.item_events: dict[int, list[int]] = {}
        for event in np.flatnonzero(allowed):
            self.user_events.setdefault(int(data.user[event]), []).append(int(event))
            self.item_events.setdefault(int(data.item[event]), []).append(int(event))

        # Precomputed history cache: (num_events, history_length), -1 = padding
        self.history_cache = history_cache

        # Per-question global stats: (num_items, 3) — [avg_correctness, log_attempts, avg_log_time]
        if global_stats is not None:
            self.global_stats = np.asarray(global_stats, dtype=np.float32)
        else:
            self.global_stats = np.zeros((len(data.question_features), 3), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def _history(self, mapping: dict[int, list[int]], key: int, event: int) -> list[int]:
        values = mapping.get(key, [])
        end = bisect_left(values, event)
        return values[max(0, end - self.history_length) : end]

    def _hybrid_retrieval(self, user: int, event: int, current_item: int) -> list[int]:
        """Retrieve history for one event.

        retrieval="hybrid": forced recent 16 + structure-scored top 24 = max 40 items
        retrieval="recent": pure last-N (no structure scoring)
        """
        if self.retrieval == "recent":
            values = self.user_events.get(user, [])
            end = bisect_left(values, event)
            return values[max(0, end - self.history_length):end]

        """Hybrid retrieval: forced recent 16 + structure-scored top 24 = max 40 items.

        Score = 1.0*same_exercise + 0.8*concept_overlap (same-question excluded from
        scored pool for label-leak prevention, but allowed in forced-recent window).

        Candidate pool is capped at 256 for performance; items with score==0 are
        excluded entirely.
        """
        values = self.user_events.get(user, [])
        end = bisect_left(values, event)
        all_history = values[:end]

        n_forced = min(16, self.history_length)
        n_scored = self.history_length - n_forced
        candidate_pool = 256

        if len(all_history) <= n_forced:
            return all_history

        # Force most recent N
        recent = all_history[-n_forced:]
        remaining = all_history[:-n_forced]

        # Cap candidate pool for performance
        if len(remaining) > candidate_pool:
            remaining = remaining[-candidate_pool:]

        # Score remaining candidates
        scores: list[tuple[int, float]] = []
        current_exercise = int(self.data.item_exercise[current_item])
        current_concepts = self.data.item_concepts[current_item]

        for e in remaining:
            h_item = int(self.data.item[e])
            # Exclude same question from scored pool (label-leak prevention)
            if h_item == current_item:
                continue

            # same_exercise
            se = 0.0
            if current_exercise >= 0:
                h_exercise = int(self.data.item_exercise[h_item])
                if h_exercise >= 0 and h_exercise == current_exercise:
                    se = 1.0

            # concept_overlap
            co = 0.0
            h_concepts = self.data.item_concepts[h_item]
            union = current_concepts | h_concepts
            if union:
                co = len(current_concepts & h_concepts) / len(union)

            score = 1.5 * 0 + 1.0 * se + 0.8 * co  # I_same_q(=0 excluded)+I_same_ex+concept_overlap
            if score > 0:
                scores.append((e, score))

        # Sort by score descending, take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scored = [e for e, _ in scores[:n_scored]]

        # Merge and sort by event index (chronological order)
        merged = sorted(set(recent + top_scored))

        return merged

    def _padded(self, events: list[int], current_time: int, include_items: bool) -> dict[str, torch.Tensor]:
        length = self.history_length
        valid = len(events)
        labels = np.zeros(length, dtype=np.int64)
        deltas = np.zeros(length, dtype=np.float32)
        mask = np.zeros(length, dtype=bool)
        items = np.zeros(length, dtype=np.int64)
        if valid:
            start = length - valid
            event_array = np.asarray(events, dtype=np.int64)
            labels[start:] = self.data.label[event_array].astype(np.int64)
            deltas[start:] = np.maximum(0, current_time - self.data.timestamp[event_array]).astype(np.float32)
            mask[start:] = True
            if include_items:
                items[start:] = self.data.item[event_array]
        result = {
            "response": torch.from_numpy(labels),
            "delta": torch.from_numpy(deltas),
            "mask": torch.from_numpy(mask),
        }
        if include_items:
            result["item"] = torch.from_numpy(items)
        return result

    def _student_structure(self, events: list[int], current_item: int, current_time: int) -> dict[str, torch.Tensor]:
        length = self.history_length
        same_question = np.zeros(length, dtype=np.float32)
        same_exercise = np.zeros(length, dtype=np.float32)
        concept_overlap = np.zeros(length, dtype=np.float32)
        if events:
            start = length - len(events)
            history_items = self.data.item[np.asarray(events, dtype=np.int64)]
            same_question[start:] = history_items == current_item
            current_exercise = int(self.data.item_exercise[current_item])
            if current_exercise >= 0:
                same_exercise[start:] = self.data.item_exercise[history_items] == current_exercise
            current_concepts = self.data.item_concepts[current_item]
            for offset, history_item in enumerate(history_items, start=start):
                history_concepts = self.data.item_concepts[int(history_item)]
                union = current_concepts | history_concepts
                concept_overlap[offset] = len(current_concepts & history_concepts) / len(union) if union else 0.0

        repeated_positions = np.flatnonzero(same_question)
        has_repeat = float(len(repeated_positions) > 0)
        repeat_count = float(len(repeated_positions))
        last_correct = 0.0
        last_delta = 0.0
        if repeated_positions.size:
            history_position = int(repeated_positions[-1] - (length - len(events)))
            last_event = events[history_position]
            last_correct = float(self.data.label[last_event])
            last_delta = float(max(0, current_time - int(self.data.timestamp[last_event])))
        return {
            "same_question": torch.from_numpy(same_question),
            "same_exercise": torch.from_numpy(same_exercise),
            "concept_overlap": torch.from_numpy(concept_overlap),
            "has_repeat": torch.tensor(has_repeat, dtype=torch.float32),
            "repeat_count": torch.tensor(repeat_count, dtype=torch.float32),
            "last_same_correct": torch.tensor(last_correct, dtype=torch.float32),
            "last_same_delta": torch.tensor(last_delta, dtype=torch.float32),
        }

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        event = int(self.indices[position])
        user = int(self.data.user[event])
        item = int(self.data.item[event])
        current_time = int(self.data.timestamp[event])
        if self.history_cache is not None:
            row = self.history_cache[event]
            student_events = [int(e) for e in row[row >= 0].tolist()]
        else:
            student_events = self._hybrid_retrieval(user, event, item)
        student = self._padded(student_events, current_time, True)
        structure = self._student_structure(student_events, item, current_time)
        question = self._padded(self._history(self.item_events, item, event), current_time, False)

        # Self-history: student's own past interactions with THIS specific question
        self_events = [e for e in student_events if int(self.data.item[e]) == item]
        self_history = self._padded(self_events, current_time, False)

        return {
            "event": torch.tensor(event),
            "item": torch.tensor(item),
            "label": torch.tensor(self.data.label[event], dtype=torch.float32),
            "student_item": student["item"],
            "student_response": student["response"],
            "student_delta": student["delta"],
            "student_mask": student["mask"],
            "same_question": structure["same_question"],
            "same_exercise": structure["same_exercise"],
            "concept_overlap": structure["concept_overlap"],
            "has_repeat": structure["has_repeat"],
            "repeat_count": structure["repeat_count"],
            "last_same_correct": structure["last_same_correct"],
            "last_same_delta": structure["last_same_delta"],
            "question_response": question["response"],
            "question_delta": question["delta"],
            "question_mask": question["mask"],
            "self_response": self_history["response"],
            "self_delta": self_history["delta"],
            "self_mask": self_history["mask"],
            "global_stats": torch.from_numpy(self.global_stats[item]),
        }
