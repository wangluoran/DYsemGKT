from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_json


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


class TemporalHistoryDataset(Dataset):
    """Creates leak-safe two-sided histories for selected prediction events."""

    def __init__(
        self,
        data: ProcessedData,
        indices: np.ndarray,
        history_length: int = 50,
        allowed_history: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.history_length = history_length
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

    def __len__(self) -> int:
        return len(self.indices)

    def _history(self, mapping: dict[int, list[int]], key: int, event: int) -> list[int]:
        values = mapping.get(key, [])
        end = bisect_left(values, event)
        return values[max(0, end - self.history_length) : end]

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
        student_events = self._history(self.user_events, user, event)
        student = self._padded(student_events, current_time, True)
        structure = self._student_structure(student_events, item, current_time)
        question = self._padded(self._history(self.item_events, item, event), current_time, False)
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
        }
