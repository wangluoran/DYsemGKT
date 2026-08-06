from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dygkt_data import NeighborSampler


# ── Time Encoders ────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    """Cosine-based time encoder with trainable linear projection."""

    def __init__(self, time_dim: int = 16) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        weight = torch.from_numpy(
            1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
        ).reshape(time_dim, -1)
        self.w.weight = nn.Parameter(weight)
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.w(timestamps.unsqueeze(dim=-1)))


class TimeDecayEncoder(nn.Module):
    """Exponential decay time encoder."""

    def __init__(self, time_dim: int = 16) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        weight = torch.from_numpy(
            1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
        ).reshape(time_dim, -1)
        self.w.weight = nn.Parameter(weight)
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.relu(self.w(timestamps.unsqueeze(dim=-1))))


class TimeDualDecayEncoder(nn.Module):
    """Dual-scale time encoder: short-term (<24h) and long-term (>24h) decays."""

    def __init__(self, time_dim: int = 16) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.w_short = nn.Linear(1, time_dim)
        self.w_long = nn.Linear(1, time_dim)
        weight = 1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
        self.w_short.weight = nn.Parameter(torch.from_numpy(weight).reshape(time_dim, -1))
        self.w_short.bias = nn.Parameter(torch.zeros(time_dim))
        self.w_long.weight = nn.Parameter(torch.from_numpy(weight).reshape(time_dim, -1))
        self.w_long.bias = nn.Parameter(torch.zeros(time_dim))
        self.w_o = nn.Linear(time_dim, time_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        # timestamps: (batch, seq_len)
        timestamps = timestamps.unsqueeze(dim=-1)
        timestamps_right = timestamps.clone()
        timestamps_right = torch.cat(
            [timestamps_right[:, 1:, :], timestamps_right[:, -1:, :]], dim=1,
        )
        timestamps_diff = timestamps_right - timestamps
        # Short-term mask: diff <= 24h (86400 seconds)
        short_mask = (timestamps_diff <= 86400).float()
        long_mask = 1.0 - short_mask
        ts_short = torch.relu(self.w_short(timestamps_diff * short_mask))
        ts_long = torch.relu(self.w_long(timestamps_diff * long_mask))
        return self.w_o(ts_short + ts_long)


# ── GRU Sequence Encoder ─────────────────────────────────────────────────────

class DyKT_Seq(nn.Module):
    """GRU-based sequence encoder for student and question history."""

    def __init__(self, edge_dim: int, node_dim: int) -> None:
        super().__init__()
        self.hid_node_updater = nn.GRU(
            input_size=edge_dim, hidden_size=node_dim, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.hid_node_updater(x)
        return hidden.squeeze(0)


# ── Merge Layer ──────────────────────────────────────────────────────────────

class MergeLayer(nn.Module):
    """Two-layer MLP merging src + dst embeddings into a scalar prediction."""

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(torch.cat([x1, x2], dim=-1))))


# ── DyGKT Model ──────────────────────────────────────────────────────────────

class DyGKTModel(nn.Module):
    """DyGKT: Dynamic Graph Knowledge Tracing with dual GRU encoders.

    Adapted from the original DyGKT to work with ProcessedData text features.
    """

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        time_dim: int = 16,
        num_neighbors: int = 50,
        node_dim: int = 64,
        edge_dim: int = 64,
        dropout: float = 0.5,
        ablation: str = "-1",
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.ablation = ablation
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))

        self.num_nodes = int(self.node_raw_features.shape[0])
        input_feat_dim = int(self.node_raw_features.shape[1])

        self.projection_layer = nn.ModuleDict({
            'feature_Linear': nn.Linear(input_feat_dim, node_dim, bias=True),
            'edge': nn.Linear(1, node_dim, bias=True),
            'time': nn.Linear(time_dim, node_dim, bias=True),
            'struct': nn.Linear(1, node_dim, bias=True),
        })

        self.output_layer = nn.Linear(node_dim, node_dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.src_node_updater = DyKT_Seq(edge_dim=edge_dim, node_dim=node_dim)
        self.dst_node_updater = DyKT_Seq(edge_dim=edge_dim, node_dim=node_dim)

        if ablation == 'dual':
            self.time_encoder: nn.Module = TimeEncoder(time_dim=time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=time_dim)

        self._neighbor_sampler: NeighborSampler | None = None

    def set_neighbor_sampler(self, sampler: NeighborSampler) -> None:
        self._neighbor_sampler = sampler

    @property
    def neighbor_sampler(self) -> NeighborSampler:
        if self._neighbor_sampler is None:
            raise RuntimeError("NeighborSampler not set — call set_neighbor_sampler first")
        return self._neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        edge_ids: np.ndarray,
        node_interact_times: np.ndarray,
        dst_node_ids: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        batch_size = len(src_node_ids)

        # ── sample neighbors ──
        src_nbr_ids, src_nbr_edge_ids, src_nbr_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors,
        )
        dst_nbr_ids, dst_nbr_edge_ids, dst_nbr_times = self.neighbor_sampler.get_historical_neighbors(
            dst_node_ids, node_interact_times, self.num_neighbors,
        )

        # append current node at the end
        src_nbr_ids = np.concatenate([src_nbr_ids, src_node_ids[:, np.newaxis]], axis=1)
        src_nbr_edge_ids = np.concatenate([src_nbr_edge_ids, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        src_nbr_times = np.concatenate([src_nbr_times, node_interact_times[:, np.newaxis]], axis=1)

        dst_nbr_ids = np.concatenate([dst_nbr_ids, dst_node_ids[:, np.newaxis]], axis=1)
        dst_nbr_edge_ids = np.concatenate([dst_nbr_edge_ids, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        dst_nbr_times = np.concatenate([dst_nbr_times, node_interact_times[:, np.newaxis]], axis=1)

        # ── structural features ──
        # co-occurrence: neighbor node == current opposite node
        src_cooccur = (
            torch.from_numpy(src_nbr_ids[:, :-1]) == torch.from_numpy(dst_node_ids).unsqueeze(1)
        ).unsqueeze(-1).float().to(device)
        dst_cooccur = (
            torch.from_numpy(dst_nbr_ids[:, :-1]) == torch.from_numpy(src_node_ids).unsqueeze(1)
        ).unsqueeze(-1).float().to(device)

        # skill matching: neighbor question skill == current question skill
        src_skill = self.node_raw_features[torch.from_numpy(src_nbr_ids).long()][:, :-1, 0].long().to(device)
        dst_skill = self.node_raw_features[torch.from_numpy(dst_node_ids).long()][:, 0].long().to(device).unsqueeze(1)
        src_skill_match = (src_skill == dst_skill).unsqueeze(-1).float()

        # ablation: counter removes co-occurrence
        coef = 0.0 if self.ablation == 'counter' else 1.0

        src_struct = self.projection_layer['struct'](coef * src_cooccur)
        dst_struct = self.projection_layer['struct'](coef * dst_cooccur)
        src_skill_struct = self.projection_layer['struct'](coef * src_skill_match)

        # ── features ──
        src_node_feat, src_edge_feat, src_time_feat = self._get_features(
            node_interact_times, src_nbr_edge_ids, src_nbr_ids, src_nbr_times,
        )
        dst_node_feat, dst_edge_feat, dst_time_feat = self._get_features(
            node_interact_times, dst_nbr_edge_ids, dst_nbr_ids, dst_nbr_times,
        )

        src_features = src_node_feat + src_edge_feat + src_time_feat
        dst_features = dst_node_feat + dst_edge_feat + dst_time_feat

        # ── encode ──
        src_emb = self.src_node_updater(
            src_features[:, :-1, :] + src_skill_struct + src_struct,
        ) + (src_edge_feat + src_time_feat)[:, -1, :]

        if self.ablation in ('q_qid', 'q_kid'):
            dst_emb = dst_node_feat[:, -1, :]
        else:
            dst_emb = self.dst_node_updater(
                (dst_edge_feat + dst_time_feat)[:, :-1, :] + dst_struct,
            ) + dst_features[:, -1, :]

        src_emb = self.output_layer(self.dropout(src_emb))
        dst_emb = self.output_layer(self.dropout(dst_emb))

        return src_emb, dst_emb

    def _get_features(
        self,
        node_interact_times: np.ndarray,
        nodes_edge_ids: np.ndarray,
        nodes_neighbor_ids: np.ndarray,
        nodes_neighbor_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        nid_tensor = torch.from_numpy(nodes_neighbor_ids).long().to(device)
        if self.ablation in ('embed', 'q_kid'):
            node_feat = self.projection_layer['feature_Linear'](
                self.node_raw_features[nid_tensor.cpu()].to(device),
            )
        elif self.ablation == 'q_qid':
            # Use pure node ID embedding — created on-the-fly via linear on one-hot-like IDs
            node_feat = self.projection_layer['feature_Linear'](
                self.node_raw_features[nid_tensor.cpu()].to(device),
            )
        else:
            node_feat = self.projection_layer['feature_Linear'](
                self.node_raw_features[nid_tensor.cpu()].to(device),
            )

        if self.ablation == 'dual':
            time_feat = self.time_encoder(
                torch.from_numpy(
                    node_interact_times[:, np.newaxis] - nodes_neighbor_times,
                ).float().to(device),
            )
        else:
            time_feat = self.time_encoder(
                torch.from_numpy(nodes_neighbor_times).float().to(device),
            )
        time_feat = self.projection_layer['time'](time_feat)

        eid_tensor = torch.from_numpy(nodes_edge_ids).long().to(device)
        edge_feat = self.projection_layer['edge'](
            self.edge_raw_features[eid_tensor.cpu()].to(device)[:, :, :1],
        )

        if self.ablation == 'time':
            time_feat = time_feat * 0
        elif self.ablation == 'skill':
            node_feat = node_feat * 0

        return node_feat, edge_feat, time_feat
