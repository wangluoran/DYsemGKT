from __future__ import annotations

import torch
from torch import nn


class TimeProjection(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        days = torch.log1p(seconds.clamp_min(0)) / 16.0
        return self.network(days.unsqueeze(-1))


class HistoryEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float, max_length: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim), enable_nested_tensor=False,
        )
        self.position = nn.Embedding(max_length + 1, hidden_dim)
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.token, std=0.02)

    def forward(self, sequence: torch.Tensor, valid: torch.Tensor, token_seed: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = sequence.shape
        positions = torch.arange(length + 1, device=sequence.device)
        token = self.token.expand(batch, -1, -1)
        if token_seed is not None:
            token = token + token_seed.unsqueeze(1)
        values = torch.cat([token, sequence], dim=1) + self.position(positions).unsqueeze(0)
        token_valid = torch.ones((batch, 1), dtype=torch.bool, device=sequence.device)
        padding_mask = ~torch.cat([token_valid, valid], dim=1)
        return self.encoder(values, src_key_padding_mask=padding_mask)[:, 0]


class DySemKT(nn.Module):
    def __init__(
        self,
        question_features: torch.Tensor,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_history: int = 50,
        feature_mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if feature_mode not in {"semantic", "id", "hybrid"}:
            raise ValueError("feature_mode must be semantic, id, or hybrid")
        features = torch.as_tensor(question_features, dtype=torch.float32)
        self.register_buffer("question_features", features)
        self.feature_mode = feature_mode
        self.semantic = nn.Sequential(nn.Linear(features.shape[1], hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.item_id = nn.Embedding(features.shape[0], hidden_dim)
        self.response = nn.Embedding(2, hidden_dim)
        self.time = TimeProjection(hidden_dim)
        self.structure = nn.Linear(3, hidden_dim, bias=False)
        self.repeat_summary = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.student_encoder = HistoryEncoder(hidden_dim, num_heads, num_layers, dropout, max_history)
        self.question_encoder = HistoryEncoder(hidden_dim, num_heads, max(1, num_layers // 2), dropout, max_history)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def question_embedding(self, item: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic(self.question_features[item])
        identity = self.item_id(item)
        if self.feature_mode == "semantic":
            return semantic
        if self.feature_mode == "id":
            return identity
        return semantic + identity

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        current = self.question_embedding(batch["item"])
        history_questions = self.question_embedding(batch["student_item"])
        structure_features = torch.stack([
            batch["same_question"], batch["same_exercise"], batch["concept_overlap"],
        ], dim=-1)
        student_sequence = (
            history_questions + self.response(batch["student_response"])
            + self.time(batch["student_delta"]) + self.structure(structure_features)
        )
        question_sequence = self.response(batch["question_response"]) + self.time(batch["question_delta"])
        repeat_features = torch.stack([
            batch["has_repeat"],
            torch.log1p(batch["repeat_count"]) / 4.0,
            batch["last_same_correct"],
            torch.log1p(batch["last_same_delta"].clamp_min(0)) / 16.0,
        ], dim=-1)
        student_state = self.student_encoder(student_sequence, batch["student_mask"], current)
        student_state = student_state + self.repeat_summary(repeat_features)
        question_state = self.question_encoder(question_sequence, batch["question_mask"], current)
        gate = self.gate(torch.cat([student_state, question_state, current], dim=-1))
        dynamic = gate * student_state + (1.0 - gate) * question_state
        return self.predictor(torch.cat([dynamic, current, dynamic * current], dim=-1)).squeeze(-1)
