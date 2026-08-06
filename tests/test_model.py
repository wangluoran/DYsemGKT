from __future__ import annotations

import torch

from dysemkt.model import DySemKT


def _batch(batch_size=3, history=5):
    return {
        "item": torch.tensor([0, 1, 2]),
        "student_item": torch.randint(0, 6, (batch_size, history)),
        "student_response": torch.randint(0, 2, (batch_size, history)),
        "student_delta": torch.rand(batch_size, history) * 1000,
        "student_mask": torch.tensor([[False] * history, [False, True, True, True, True], [True] * history]),
        "same_question": torch.tensor([[0.0] * history, [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]]),
        "same_exercise": torch.tensor([[0.0] * history, [0.0, 1.0, 0.0, 1.0, 0.0], [1.0] * history]),
        "concept_overlap": torch.rand(batch_size, history),
        "has_repeat": torch.tensor([0.0, 1.0, 1.0]),
        "repeat_count": torch.tensor([0.0, 1.0, 2.0]),
        "last_same_correct": torch.tensor([0.0, 1.0, 0.0]),
        "last_same_delta": torch.tensor([0.0, 120.0, 240.0]),
        "question_response": torch.randint(0, 2, (batch_size, history)),
        "question_delta": torch.rand(batch_size, history) * 1000,
        "question_mask": torch.tensor([[False] * history, [False, False, True, True, True], [True] * history]),
        "self_response": torch.randint(0, 2, (batch_size, history)),
        "self_delta": torch.rand(batch_size, history) * 1000,
        "self_mask": torch.tensor([[False] * history, [False, False, True, False, False], [True, True, False, False, False]]),
        "global_stats": torch.rand(batch_size, 3),
    }


def test_all_feature_modes_forward_and_backward():
    features = torch.randn(6, 20)
    for mode in ("semantic", "id", "hybrid"):
        model = DySemKT(features, d_model=16, max_history=5, feature_mode=mode)
        logits = model(_batch())
        assert logits.shape == (3,)
        assert torch.isfinite(logits).all()
        logits.sum().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
