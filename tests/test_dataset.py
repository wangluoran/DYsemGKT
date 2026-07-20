from __future__ import annotations

import numpy as np

from dysemkt.dataset import ProcessedData, TemporalHistoryDataset
from dysemkt.preprocess import preprocess_moocradar
from dysemkt.text import HashTextEncoder


def test_histories_are_strictly_prior_and_train_only(raw_moocradar, tmp_path):
    output = tmp_path / "processed"
    preprocess_moocradar(raw_moocradar, output, HashTextEncoder(16))
    data = ProcessedData(output)
    train_mask = data.temporal_split == 0
    validation_indices = np.flatnonzero(data.temporal_split == 1)
    dataset = TemporalHistoryDataset(data, validation_indices, history_length=5, allowed_history=train_mask)

    sample = dataset[0]
    event = int(sample["event"])
    user = int(data.user[event])
    expected = [
        idx for idx in np.flatnonzero(train_mask)
        if idx < event and int(data.user[idx]) == user
    ][-5:]
    actual_items = sample["student_item"][sample["student_mask"]].numpy().tolist()
    assert actual_items == data.item[expected].tolist()
    assert all(idx < event for idx in expected)


def test_empty_history_has_stable_shapes(raw_moocradar, tmp_path):
    output = tmp_path / "processed"
    preprocess_moocradar(raw_moocradar, output, HashTextEncoder(16))
    data = ProcessedData(output)
    dataset = TemporalHistoryDataset(data, np.array([0]), history_length=4)
    sample = dataset[0]
    assert sample["student_item"].shape == (4,)
    assert sample["question_response"].shape == (4,)
    assert not sample["student_mask"].any()
    assert not sample["question_mask"].any()
    assert not sample["same_question"].any()
    assert sample["has_repeat"].item() == 0


def test_explicit_repeat_and_hierarchy_features(raw_moocradar, tmp_path):
    output = tmp_path / "processed"
    preprocess_moocradar(raw_moocradar, output, HashTextEncoder(16))
    data = ProcessedData(output)
    target = None
    for event in range(len(data.user)):
        prior = np.flatnonzero(
            (data.user[:event] == data.user[event]) & (data.item[:event] == data.item[event])
        )
        if prior.size:
            target = event
            break
    assert target is not None
    dataset = TemporalHistoryDataset(data, np.array([target]), history_length=12)
    sample = dataset[0]
    repeated = sample["same_question"].bool()
    assert repeated.any()
    assert sample["same_exercise"][repeated].eq(1).all()
    assert sample["concept_overlap"][repeated].eq(1).all()
    assert sample["has_repeat"].item() == 1
    assert sample["repeat_count"].item() == repeated.sum().item()
    assert sample["last_same_delta"].item() > 0
