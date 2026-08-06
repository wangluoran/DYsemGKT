from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DySemKT(nn.Module):
    """Dual-tower relation-aware KT with semantic residual.

    Student tower — long-term knowledge state from related history (excludes same-question).
    Question tower — difficulty + recency from self-history + global stats.
    Final gate fuses both towers, with raw BGE residual pass-through.
    """

    def __init__(
        self,
        question_features: torch.Tensor,
        d_model: int = 128,
        max_history: int = 40,
        dropout: float = 0.1,
        feature_mode: str = "semantic",
    ) -> None:
        super().__init__()
        if feature_mode not in {"semantic", "id", "hybrid"}:
            raise ValueError("feature_mode must be semantic, id, or hybrid")

        features = torch.as_tensor(question_features, dtype=torch.float32)
        self.register_buffer("question_features", features)
        self.feature_mode = feature_mode
        self.d_model = d_model
        self.max_history = max_history

        feat_dim = int(features.shape[1])    # BGE raw, e.g. 1024
        sem_dim = min(feat_dim, 512)         # internal semantic dim
        id_dim = d_model

        # ── Semantic projection ──
        self.semantic = nn.Sequential(
            nn.Linear(feat_dim, sem_dim), nn.LayerNorm(sem_dim), nn.ReLU(),
            nn.Linear(sem_dim, sem_dim),
        )
        self.item_id = nn.Embedding(features.shape[0], id_dim)

        # Q dim varies by feature_mode
        if feature_mode == "semantic":
            q_dim = sem_dim  # go through deep semantic, same space as K
        elif feature_mode == "id":
            q_dim = id_dim
        else:
            q_dim = sem_dim + id_dim  # sem+id, both deep-projected

        # ── Feature alignment ──
        self.proj_q = nn.Linear(q_dim, d_model)          # current question → Q
        self.proj_k = nn.Linear(sem_dim, d_model)         # history semantic → K

        # ── Raw low-dim encodings (keep small, don't blow up to d_model yet) ──
        self.raw_resp = nn.Embedding(2, 4)               # binary 0/1 → 4-dim
        self.raw_time = nn.Sequential(
            nn.Linear(1, 8), nn.GELU(),                  # scalar Δ → 8-dim
        )
        self.raw_struct = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(),                 # [同题,同练习,重叠] → 16-dim
        )

        # ── Project to d_model (for student tower V values) ──
        self.proj_resp = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.proj_time = nn.Sequential(
            nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.proj_struct = nn.Sequential(
            nn.Linear(16, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Structure bias: 3-dim raw → scalar, hijacks attention logits
        self.struct_bias = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1),
        )

        # ── Student tower: 4-modality gate ──
        self.student_gate = nn.Sequential(
            nn.Linear(d_model * 4, 128), nn.ReLU(), nn.Linear(128, 4), nn.Softmax(dim=-1),
        )

        # ── Question tower: raw features (4+8+3=15) projected together ──
        self.proj_qk = nn.Sequential(
            nn.Linear(15, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.q_to_question = nn.Linear(d_model, d_model, bias=False)  # align Q into K_q space
        self.tau = nn.Parameter(torch.tensor(2.0))   # time decay half-life (hours)
        self.empty_question = nn.Parameter(torch.zeros(1, d_model))

        # ── Final tower gate ──
        self.tower_gate = nn.Sequential(
            nn.Linear(d_model * 3, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=-1),
        )

        # ── Residual: raw BGE → d_model ──
        self.residual_proj = nn.Linear(feat_dim, d_model)

        # ── Predictor ──
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, 128),  # 384 → 128
            nn.ReLU(),                     # ReLU, not GELU
            nn.Dropout(dropout),           # Dropout after activation, before compression
            nn.Linear(128, 1),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _current_query(self, item: torch.Tensor) -> torch.Tensor:
        raw = self.question_features[item]
        if self.feature_mode == "semantic":
            return self.proj_q(self.semantic(raw))  # deep semantic, same space as K
        if self.feature_mode == "id":
            return self.proj_q(self.item_id(item))
        return self.proj_q(torch.cat([self.semantic(raw), self.item_id(item)], dim=-1))

    def _history_keys(self, history_items: torch.Tensor) -> torch.Tensor:
        return self.proj_k(self.semantic(self.question_features[history_items]))

    @staticmethod
    def _time_scalar(delta_seconds: torch.Tensor) -> torch.Tensor:
        """Normalize delta seconds to a single scalar in [0, ~1]."""
        return torch.log1p(delta_seconds.clamp_min(0)) / 16.0

    def _student_attention(
        self, Q: torch.Tensor, K: torch.Tensor, bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scaled dot-product attention with structure bias and same-question mask."""
        scale = self.d_model ** 0.5
        logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
        logits = logits + bias.transpose(-2, -1) * scale  # amplify bias to match logit magnitude
        logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        return F.softmax(logits, dim=-1)

    # ── Towers ─────────────────────────────────────────────────────────────

    def _student_tower(
        self, Q: torch.Tensor, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Student tower: Q attends over related student history (no same-question)."""
        K = self._history_keys(batch["student_item"])

        # ── Raw low-dim encodings ──
        raw_resp  = self.raw_resp(batch["student_response"].long())          # (B,N,4)
        t_scalar  = self._time_scalar(batch["student_delta"])
        raw_time  = self.raw_time(t_scalar.unsqueeze(-1))                   # (B,N,8)
        struct_raw = torch.stack([
            batch["same_question"], batch["same_exercise"], batch["concept_overlap"],
        ], dim=-1)                                                          # (B,N,3)
        raw_struct = self.raw_struct(struct_raw)                            # (B,N,16)

        # ── Project to d_model ──
        V_resp   = self.proj_resp(raw_resp)                                 # (B,N,128)
        V_time   = self.proj_time(raw_time)                                 # (B,N,128)
        V_struct = self.proj_struct(raw_struct)                             # (B,N,128)
        bias     = self.struct_bias(struct_raw)                             # (B,N,1)

        # Mask: exclude same-question (label leak) + padding only
        # Unrelated history is NOT masked — struct_bias from [0,0,0]
        # naturally produces low weight (~-16), preserving gradient flow
        same_q = (batch["same_question"] == 0).float()            # exclude same-q
        valid = batch.get("student_mask", torch.ones_like(same_q))
        attn_mask = same_q * valid.float()

        attn = self._student_attention(Q, K, bias, attn_mask)

        ctx_sem    = (attn @ K).squeeze(1)
        ctx_resp   = (attn @ V_resp).squeeze(1)
        ctx_time   = (attn @ V_time).squeeze(1)
        ctx_struct = (attn @ V_struct).squeeze(1)

        gate_in = torch.cat([ctx_sem, ctx_resp, ctx_time, ctx_struct], dim=-1)
        gates = self.student_gate(gate_in)
        return (gates[:, 0:1] * ctx_sem + gates[:, 1:2] * ctx_resp +
                gates[:, 2:3] * ctx_time + gates[:, 3:4] * ctx_struct)

    def _question_tower(
        self, Q: torch.Tensor, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Question tower: self-history (same question) + global stats + time decay."""
        if Q.dim() == 3:
            Q = Q.squeeze(1)  # (B, 1, d) → (B, d)
        B, d = Q.shape
        N = batch["self_response"].shape[1]

        # ── Raw low-dim encodings (shared with student tower) ──
        raw_resp = self.raw_resp(batch["self_response"].long())               # (B,N,4)
        t_scalar = self._time_scalar(batch["self_delta"])
        raw_time = self.raw_time(t_scalar.unsqueeze(-1))                      # (B,N,8)

        # Global stats: (B, 3) → expand to (B, N, 3)
        g_stats = batch.get("global_stats", torch.zeros(B, 3, device=Q.device))
        # During training, randomly zero global_stats per sample (30%)
        # to force the model to handle cold-start (missing global stats gracefully)
        if self.training:
            keep_mask = torch.rand(B, 1, device=Q.device) > 0.3  # keep 70%, zero 30%
            g_stats = g_stats * keep_mask.float()
        g_expand = g_stats.unsqueeze(1).expand(-1, N, -1)

        # Concatenate raw features: 4 + 8 + 3 = 15 → d_model
        K_q = self.proj_qk(torch.cat([raw_resp, raw_time, g_expand], dim=-1))  # (B,N,d)

        scale = d ** 0.5
        Q_proj = self.q_to_question(Q)  # map Q into K_q space for meaningful dot-product
        logits = torch.matmul(Q_proj.unsqueeze(1), K_q.transpose(-2, -1)) / scale   # (B,1,N)

        # Time decay: exp(-delta_hours / tau)
        delta_hours = batch["self_delta"] / 3600.0                              # (B,N)
        decay = torch.exp(-delta_hours / (self.tau.abs() + 1e-6)).unsqueeze(-1) # (B,N,1)
        log_decay = torch.log(decay + 1e-6).clamp(min=-10.0)  # clamp to avoid NaN/gradient vanishing
        logits = logits + log_decay.transpose(-2, -1)

        # Mask
        s_mask = batch.get("self_mask", torch.ones(B, N, device=Q.device))
        logits = logits.masked_fill(s_mask.bool().unsqueeze(1) == False, -1e9)

        # Cold-start: if no self-history, use learnable empty embedding
        has_history = s_mask.any(dim=-1)                                       # (B,)
        attn = F.softmax(logits, dim=-1)
        q_out = (attn @ K_q).squeeze(1)                                        # (B,d)
        q_out = torch.where(
            has_history.unsqueeze(-1), q_out, self.empty_question.expand(B, -1),
        )
        return q_out

    def summarize(self) -> None:
        """Print a human-readable architecture summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines = [
            "=" * 64,
            f"  DySemKT  Architecture Summary",
            "=" * 64,
            f"  feature_mode : {self.feature_mode}",
            f"  d_model      : {self.d_model}",
            f"  max_history  : {self.max_history}",
            f"  parameters   : {trainable:,} trainable / {total:,} total",
            "",
            "  ── Semantic ─────────────────────────────────────",
            f"  semantic     : Linear({self.semantic[0].in_features}→{self.semantic[0].out_features})"
            f" → LN → ReLU → Linear({self.semantic[3].in_features}→{self.semantic[3].out_features})",
            f"  proj_q       : Linear({self.proj_q.in_features}→{self.proj_q.out_features})",
            f"  proj_k       : Linear({self.proj_k.in_features}→{self.proj_k.out_features})",
            "",
            "  ── Raw Low-Dim Encodings ───────────────────────",
            f"  raw_resp     : Embedding(2, {self.raw_resp.embedding_dim})    # 0/1 → 4-dim",
            f"  raw_time     : Linear({self.raw_time[0].in_features}→{self.raw_time[0].out_features}) → GELU   # scalar Δ → 8-dim",
            f"  raw_struct   : Linear({self.raw_struct[0].in_features}→{self.raw_struct[0].out_features}) → GELU   # 3-dim → 16-dim",
            "",
            "  ── Student Tower Projections ───────────────────",
            f"  proj_resp    : Linear(4→{self.proj_resp[-1].out_features}) → GELU → Linear(→{self.proj_resp[-1].out_features})",
            f"  proj_time    : Linear(8→{self.proj_time[-1].out_features}) → GELU → Linear(→{self.proj_time[-1].out_features})",
            f"  proj_struct  : Linear(16→{self.proj_struct[-1].out_features}) → GELU → Linear(→{self.proj_struct[-1].out_features})",
            f"  struct_bias  : Linear(3→{self.struct_bias[0].out_features}) → ReLU → Linear(→1)  # scalar bias",
            f"  student_gate : Linear({self.student_gate[0].in_features}→128) → ReLU → Linear(→4) → Softmax",
            "",
            "  ── Question Tower ──────────────────────────────",
            f"  proj_qk      : Linear(15→{self.proj_qk[-1].out_features}) → GELU → Linear(→{self.proj_qk[-1].out_features})",
            f"  q_to_question: Linear({self.q_to_question.in_features}→{self.q_to_question.out_features}, bias=False)  # Q→K_q space align",
            f"  tau (learnable): {float(self.tau.detach()):.2f}",
            "",
            "  ── Fusion & Output ────────────────────────────",
            f"  tower_gate   : Linear({self.tower_gate[0].in_features}→128) → ReLU → Linear(→2) → Softmax",
            f"  residual     : Linear({self.residual_proj.in_features}→{self.residual_proj.out_features})",
            f"  predictor    : LN → Linear({self.predictor[1].in_features}→{self.predictor[1].out_features}) → ReLU → Dropout → Linear(→1)",
            "=" * 64,
        ]
        print("\n".join(lines), flush=True)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        Q = self._current_query(batch["item"]).unsqueeze(1)         # (B,1,d)

        student_out  = self._student_tower(Q, batch)                # (B,d)
        question_out = self._question_tower(Q, batch)               # (B,d)

        # Final gate: student vs question
        gate_in = torch.cat([student_out, question_out, Q.squeeze(1)], dim=-1)
        gates = self.tower_gate(gate_in)
        fused = gates[:, 0:1] * student_out + gates[:, 1:2] * question_out

        # Residual from raw semantic
        residual = self.residual_proj(self.question_features[batch["item"]])
        output = fused + residual

        # FM second-order interaction: [output, current_Q, output * current_Q]
        Q_sq = Q.squeeze(1)  # (B, d)
        final = torch.cat([output, Q_sq, output * Q_sq], dim=-1)  # (B, 3d)
        return self.predictor(final).squeeze(-1)