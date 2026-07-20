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
        size = len(self.user)
        if any(len(getattr(self, name)) != size for name in ("item", "timestamp", "label", "temporal_split", "cold_split")):
            raise ValueError("event arrays have inconsistent lengths")
        if np.any(np.diff(self.timestamp) < 0):
            raise ValueError("events must be globally chronological")


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

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        event = int(self.indices[position])
        user = int(self.data.user[event])
        item = int(self.data.item[event])
        current_time = int(self.data.timestamp[event])
        student = self._padded(self._history(self.user_events, user, event), current_time, True)
        question = self._padded(self._history(self.item_events, item, event), current_time, False)
        return {
            "event": torch.tensor(event),
            "item": torch.tensor(item),
            "label": torch.tensor(self.data.label[event], dtype=torch.float32),
            "student_item": student["item"],
            "student_response": student["response"],
            "student_delta": student["delta"],
            "student_mask": student["mask"],
            "question_response": question["response"],
            "question_delta": question["delta"],
            "question_mask": question["mask"],
        }

