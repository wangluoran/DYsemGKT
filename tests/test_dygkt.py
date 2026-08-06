from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dysemkt.dataset import ProcessedData
from dysemkt.dygkt_data import DyGKTData, NeighborSampler, build_dygkt_data
from dysemkt.dygkt_engine import DyGKTTrainConfig, evaluate_dygkt, train_dygkt
from dysemkt.dygkt_model import DyGKTModel, MergeLayer
from dysemkt.preprocess import preprocess_moocradar
from dysemkt.text import HashTextEncoder


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dygkt_processed(raw_moocradar, tmp_path):
    """Run full preprocess pipeline and return ProcessedData + DyGKT data."""
    data_dir = tmp_path / "processed"
    preprocess_moocradar(raw_moocradar, data_dir, HashTextEncoder(16))
    proc = ProcessedData(data_dir)
    node_feat, edge_feat, train_d, val_d, test_d = build_dygkt_data(proc, split="temporal")
    return proc, node_feat, edge_feat, train_d, val_d, test_d


# ── Data Adapter Tests ───────────────────────────────────────────────────────

def test_build_dygkt_data_shapes(dygkt_processed):
    proc, node_feat, edge_feat, train_d, val_d, test_d = dygkt_processed
    num_users = int(proc.user.max()) + 1
    num_items = int(proc.item.max()) + 1
    num_events = len(proc.user)

    # node features: all nodes, 1 + semantic_dim columns
    assert node_feat.shape == (num_users + num_items, 1 + 16)
    # edge features: one row per event
    assert edge_feat.shape == (num_events, 1)
    # split counts
    total = train_d.num_interactions + val_d.num_interactions + test_d.num_interactions
    assert total == num_events


def test_build_dygkt_data_user_item_ranges(dygkt_processed):
    proc, node_feat, edge_feat, train_d, val_d, test_d = dygkt_processed
    num_users = int(proc.user.max()) + 1
    num_items = int(proc.item.max()) + 1
    total_nodes = num_users + num_items

    for d in (train_d, val_d, test_d):
        if d.num_interactions == 0:
            continue
        assert d.src_node_ids.min() >= 0
        assert d.src_node_ids.max() < num_users
        assert d.dst_node_ids.min() >= num_users
        assert d.dst_node_ids.max() < total_nodes


def test_build_dygkt_data_student_features_zero(dygkt_processed):
    proc, node_feat, edge_feat, train_d, val_d, test_d = dygkt_processed
    num_users = int(proc.user.max()) + 1
    # Student node features should be all zeros
    assert np.allclose(node_feat[:num_users], 0.0)


def test_build_dygkt_data_question_features_present(dygkt_processed):
    proc, node_feat, edge_feat, train_d, val_d, test_d = dygkt_processed
    num_users = int(proc.user.max()) + 1
    num_items = int(proc.item.max()) + 1
    # Question nodes should have non-zero features (from hash encoder)
    question_features = node_feat[num_users:num_users + num_items, 1:]
    assert not np.allclose(question_features, 0.0)


# ── NeighborSampler Tests ────────────────────────────────────────────────────

def test_neighbor_sampler_basic(dygkt_processed):
    _, _, _, train_d, _, _ = dygkt_processed
    if train_d.num_interactions < 5:
        pytest.skip("not enough training interactions")
    sampler = NeighborSampler(train_d, seed=0)
    nids = np.array([train_d.src_node_ids[0]])
    times = np.array([train_d.node_interact_times[0]])
    nbr_ids, nbr_eids, nbr_times = sampler.get_historical_neighbors(nids, times, num_neighbors=10)
    assert nbr_ids.shape == (1, 10)
    assert nbr_eids.shape == (1, 10)
    assert nbr_times.shape == (1, 10)
    # All timestamps should be strictly before the query time
    for t in nbr_times[0]:
        if t > 0:
            assert t < times[0]


# ── Model Tests ──────────────────────────────────────────────────────────────

def test_dygkt_model_forward(dygkt_processed):
    _, node_feat, edge_feat, train_d, _, _ = dygkt_processed
    if train_d.num_interactions < 4:
        pytest.skip("not enough training interactions")
    sampler = NeighborSampler(train_d, seed=0)

    model = DyGKTModel(
        node_raw_features=node_feat,
        edge_raw_features=edge_feat,
        time_dim=8,
        num_neighbors=4,
        node_dim=16,
        edge_dim=16,
        dropout=0.0,
        ablation="-1",
    )
    model.set_neighbor_sampler(sampler)

    batch_size = 3
    src = train_d.src_node_ids[:batch_size]
    dst = train_d.dst_node_ids[:batch_size]
    times = train_d.node_interact_times[:batch_size]
    eids = train_d.edge_ids[:batch_size]

    src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(src, eids, times, dst)
    assert src_emb.shape == (batch_size, 16)
    assert dst_emb.shape == (batch_size, 16)


def test_dygkt_model_backward(dygkt_processed):
    _, node_feat, edge_feat, train_d, _, _ = dygkt_processed
    if train_d.num_interactions < 4:
        pytest.skip("not enough training interactions")
    sampler = NeighborSampler(train_d, seed=0)

    model = DyGKTModel(
        node_raw_features=node_feat, edge_raw_features=edge_feat,
        time_dim=8, num_neighbors=4, node_dim=16, edge_dim=16,
        dropout=0.0, ablation="-1",
    )
    merge = MergeLayer(16, 16, 16, output_dim=1)
    model.set_neighbor_sampler(sampler)

    batch_size = 3
    src = train_d.src_node_ids[:batch_size]
    dst = train_d.dst_node_ids[:batch_size]
    times = train_d.node_interact_times[:batch_size]
    eids = train_d.edge_ids[:batch_size]

    src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(src, eids, times, dst)
    logits = merge(src_emb, dst_emb).squeeze(-1)
    loss = logits.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


# ── Ablation Tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("ablation", ["-1", "dual", "time", "skill", "counter"])
def test_dygkt_ablation_modes(ablation, dygkt_processed):
    _, node_feat, edge_feat, train_d, _, _ = dygkt_processed
    if train_d.num_interactions < 3:
        pytest.skip("not enough training interactions")
    sampler = NeighborSampler(train_d, seed=0)

    model = DyGKTModel(
        node_raw_features=node_feat, edge_raw_features=edge_feat,
        time_dim=8, num_neighbors=4, node_dim=16, edge_dim=16,
        dropout=0.0, ablation=ablation,
    )
    model.set_neighbor_sampler(sampler)

    src = train_d.src_node_ids[:3]
    dst = train_d.dst_node_ids[:3]
    times = train_d.node_interact_times[:3]
    eids = train_d.edge_ids[:3]

    src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(src, eids, times, dst)
    assert torch.isfinite(src_emb).all()
    assert torch.isfinite(dst_emb).all()


# ── End-to-End Training Test ─────────────────────────────────────────────────

def test_dygkt_end_to_end_training(raw_moocradar, tmp_path):
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "run"
    preprocess_moocradar(raw_moocradar, data_dir, HashTextEncoder(16))
    config = DyGKTTrainConfig(
        num_neighbors=4, time_dim=4, node_dim=16,
        dropout=0.0, batch_size=16, epochs=1, patience=1,
    )
    result = train_dygkt(data_dir, output_dir, config)
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert 0.0 <= result["test"]["accuracy"] <= 1.0
    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["split_counts"]["train"] > 0


# ── Split Tests ──────────────────────────────────────────────────────────────

def test_dygkt_cold_split(dygkt_processed):
    proc, _, _, _, _, _ = dygkt_processed
    _, _, train_d, val_d, test_d = build_dygkt_data(proc, split="cold")
    # Cold split should produce non-empty splits for the test data
    total = train_d.num_interactions + val_d.num_interactions + test_d.num_interactions
    assert total == len(proc.user)


def test_dygkt_invalid_split(dygkt_processed):
    proc, _, _, _, _, _ = dygkt_processed
    with pytest.raises(ValueError, match="split must be"):
        build_dygkt_data(proc, split="invalid")
