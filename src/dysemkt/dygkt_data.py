from __future__ import annotations

import numpy as np

from .dataset import ProcessedData


class DyGKTData:
    """Minimal edge-list data container matching DyGKT's Data format."""

    __slots__ = ("src_node_ids", "dst_node_ids", "node_interact_times", "edge_ids", "labels",
                 "num_interactions", "unique_node_ids", "num_unique_nodes")

    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.src_node_ids = np.asarray(src_node_ids, dtype=np.int64)
        self.dst_node_ids = np.asarray(dst_node_ids, dtype=np.int64)
        self.node_interact_times = np.asarray(node_interact_times, dtype=np.float64)
        self.edge_ids = np.asarray(edge_ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(self.src_node_ids) | set(self.dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


class NeighborSampler:
    """Builds per-node sorted adjacency lists for temporal neighbor sampling."""

    def __init__(self, data: DyGKTData, seed: int | None = None) -> None:
        max_node_id = max(int(data.src_node_ids.max()), int(data.dst_node_ids.max()))
        adj_list: list[list[tuple[int, int, float]]] = [[] for _ in range(max_node_id + 1)]
        for src, dst, eid, t in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
            adj_list[int(src)].append((int(dst), int(eid), float(t)))
            adj_list[int(dst)].append((int(src), int(eid), float(t)))

        self.nodes_neighbor_ids: list[np.ndarray] = []
        self.nodes_edge_ids: list[np.ndarray] = []
        self.nodes_neighbor_times: list[np.ndarray] = []

        for per_node in adj_list:
            sorted_neighbors = sorted(per_node, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_neighbors], dtype=np.int64))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_neighbors], dtype=np.int64))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_neighbors], dtype=np.float64))

        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def find_neighbors_before(self, node_id: int, interact_time: float):
        # Guard against nodes never seen in training (cold-split test questions, etc.)
        if node_id >= len(self.nodes_neighbor_times) or node_id < 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty, np.array([], dtype=np.float64)
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)
        return (
            self.nodes_neighbor_ids[node_id][:i],
            self.nodes_edge_ids[node_id][:i],
            self.nodes_neighbor_times[node_id][:i],
        )

    def get_historical_neighbors(
        self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = len(node_ids)
        neighbor_ids = np.zeros((batch_size, num_neighbors), dtype=np.int64)
        neighbor_edges = np.zeros((batch_size, num_neighbors), dtype=np.int64)
        neighbor_times = np.zeros((batch_size, num_neighbors), dtype=np.float64)

        for idx, (node_id, t) in enumerate(zip(node_ids, node_interact_times)):
            nids, eids, ntimes = self.find_neighbors_before(int(node_id), float(t))
            if len(nids) == 0:
                continue
            if len(nids) <= num_neighbors:
                neighbor_ids[idx, num_neighbors - len(nids):] = nids
                neighbor_edges[idx, num_neighbors - len(eids):] = eids
                neighbor_times[idx, num_neighbors - len(ntimes):] = ntimes
            else:
                sample_idx = self._rng.choice(len(nids), size=num_neighbors, replace=False)
                sample_idx.sort()
                neighbor_ids[idx] = nids[sample_idx]
                neighbor_edges[idx] = eids[sample_idx]
                neighbor_times[idx] = ntimes[sample_idx]

        return neighbor_ids, neighbor_edges, neighbor_times


def build_dygkt_data(data: ProcessedData, split: str = "temporal") -> tuple[np.ndarray, np.ndarray, DyGKTData, DyGKTData, DyGKTData]:
    """Convert ProcessedData into DyGKT's graph format.

    Returns:
        node_raw_features:  (num_users + num_items, 1 + semantic_dim)
        edge_raw_features:  (num_events, 1)
        train_data, val_data, test_data: DyGKTData objects
    """
    if split not in ("temporal", "cold"):
        raise ValueError("split must be 'temporal' or 'cold'")

    num_users = int(data.user.max()) + 1
    num_items = int(data.item.max()) + 1
    total_nodes = num_users + num_items
    num_events = len(data.user)

    # --- node_raw_features ---
    # Column 0: skill/exercise index
    # Remaining columns: question text features (for question nodes), zeros (for student nodes)
    semantic_dim = int(data.question_features.shape[1])
    feature_dim = 1 + semantic_dim

    node_raw_features = np.zeros((total_nodes, feature_dim), dtype=np.float32)

    # Student nodes (0 .. num_users-1): skill = 0, features all zero
    # (student identity is encoded through interaction history)

    # Question nodes (num_users .. total_nodes-1): skill = exercise_id, features = text embedding
    question_start = num_users
    for item_idx in range(num_items):
        node_id = question_start + item_idx
        exercise_id = int(data.item_exercise[item_idx])
        node_raw_features[node_id, 0] = float(max(exercise_id, item_idx))
        node_raw_features[node_id, 1:] = data.question_features[item_idx]

    # --- edge_raw_features ---
    edge_raw_features = data.label[:, np.newaxis].astype(np.float32)

    # --- graph edges ---
    split_mask = data.temporal_split if split == "temporal" else data.cold_split
    # offset question node IDs
    dst_nodes = data.item.astype(np.int64) + num_users
    src_nodes = data.user.astype(np.int64)
    timestamps = data.timestamp.astype(np.float64)
    edge_ids = np.arange(num_events, dtype=np.int64)
    labels = data.label.astype(np.float32)

    def _make_data(mask: np.ndarray) -> DyGKTData:
        indices = np.flatnonzero(mask)
        return DyGKTData(
            src_node_ids=src_nodes[indices],
            dst_node_ids=dst_nodes[indices],
            node_interact_times=timestamps[indices],
            edge_ids=indices.astype(np.int64),  # preserve original edge ids for feature lookup
            labels=labels[indices],
        )

    train_data = _make_data(split_mask == 0)
    val_data = _make_data(split_mask == 1)
    test_data = _make_data(split_mask == 2)

    return node_raw_features, edge_raw_features, train_data, val_data, test_data


def build_neighbor_sampler(data: DyGKTData, seed: int = 0) -> NeighborSampler:
    """Build a NeighborSampler from DyGKTData."""
    return NeighborSampler(data, seed=seed)
