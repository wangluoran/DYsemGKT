from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Workaround: Flash/Mem-Efficient SDPA can cause illegal memory access on
# CUDA 13.0 + Turing GPUs (e.g. RTX 2080 Ti). Force math backend.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from .dataset import ProcessedData
from .dygkt_data import DyGKTData, NeighborSampler, build_dygkt_data
from .dygkt_model import DyGKTModel, MergeLayer
from .io import write_json
from .metrics import binary_metrics


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class DyGKTTrainConfig:
    seed: int = 42
    split: str = "cold"
    num_neighbors: int = 50
    time_dim: int = 16
    node_dim: int = 64
    dropout: float = 0.5
    ablation: str = "-1"
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5
    device: str = "auto"
    num_workers: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).to(device)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class _IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dygkt(
    model: nn.Module,
    data: DyGKTData,
    loader: DataLoader,
    neighbor_sampler: NeighborSampler,
    device: torch.device,
) -> dict[str, float]:
    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    for batch_indices in loader:
        indices = batch_indices.numpy()
        src = data.src_node_ids[indices]
        dst = data.dst_node_ids[indices]
        times = data.node_interact_times[indices]
        eids = data.edge_ids[indices]
        labels = data.labels[indices]

        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(src, eids, times, dst)
        logits = model[1](src_emb, dst_emb).squeeze(-1)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels)

    if not all_labels:
        raise ValueError("evaluation split is empty")
    return binary_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


# ── Training Loop ────────────────────────────────────────────────────────────

def train_dygkt(data_dir: Path, output_dir: Path, config: DyGKTTrainConfig) -> dict:
    if config.split not in ("temporal", "cold"):
        raise ValueError("split must be 'temporal' or 'cold'")
    set_seed(config.seed)

    # load and convert data
    proc = ProcessedData(data_dir)
    node_raw_features, edge_raw_features, train_data, val_data, test_data = build_dygkt_data(
        proc, split=config.split,
    )

    # build neighbor samplers (train-data only for leak safety)
    train_sampler = NeighborSampler(train_data, seed=0)
    eval_sampler = NeighborSampler(train_data, seed=1)

    # data loaders
    split_loaders = {
        "train": DataLoader(
            _IndexDataset(train_data.num_interactions),
            batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
        ),
        "validation": DataLoader(
            _IndexDataset(val_data.num_interactions),
            batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        ),
        "test": DataLoader(
            _IndexDataset(test_data.num_interactions),
            batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        ),
    }

    # device
    device_name = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    print(
        f"dygkt config: device={device} split={config.split} "
        f"epochs={config.epochs} patience={config.patience} batch_size={config.batch_size} "
        f"learning_rate={config.learning_rate:g} node_dim={config.node_dim} "
        f"num_neighbors={config.num_neighbors} time_dim={config.time_dim} "
        f"dropout={config.dropout:g} ablation={config.ablation or 'none'} "
        f"num_workers={config.num_workers}",
        flush=True,
    )
    print(
        f"dataset sizes: train={train_data.num_interactions} "
        f"validation={val_data.num_interactions} test={test_data.num_interactions}",
        flush=True,
    )

    # model
    dygkt = DyGKTModel(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        time_dim=config.time_dim,
        num_neighbors=config.num_neighbors,
        node_dim=config.node_dim,
        edge_dim=config.node_dim,
        dropout=config.dropout,
        ablation=config.ablation,
    ).to(device)
    merge = MergeLayer(config.node_dim, config.node_dim, config.node_dim, output_dim=1).to(device)
    model = nn.Sequential(dygkt, merge)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 1e-2,
    )
    criterion = nn.BCEWithLogitsLoss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save config
    write_json(output_dir / "config.json", asdict(config))

    best_auc = -float("inf")
    stale = 0
    history: list[dict] = []
    epoch_times: list[float] = []
    run_start = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        # ── train ──
        model.train()
        model[0].set_neighbor_sampler(train_sampler)
        losses: list[float] = []
        epoch_start = time.monotonic()
        total_batches = len(split_loaders["train"])
        progress_every = max(1, min(50, total_batches // 10))
        print(f"epoch {epoch}/{config.epochs} start, train batches={total_batches}", flush=True)

        for batch_idx, batch_indices in enumerate(split_loaders["train"], start=1):
            indices = batch_indices.numpy()
            src = train_data.src_node_ids[indices]
            dst = train_data.dst_node_ids[indices]
            times = train_data.node_interact_times[indices]
            eids = train_data.edge_ids[indices]
            labels = train_data.labels[indices]

            src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(src, eids, times, dst)
            logits = model[1](src_emb, dst_emb).squeeze(-1)

            loss = criterion(logits, _to_tensor(labels, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0:
                elapsed = time.monotonic() - epoch_start
                bps = batch_idx / max(elapsed, 1e-9)
                eta = (total_batches - batch_idx) / max(bps, 1e-9)
                print(
                    f"  train progress: batch {batch_idx}/{total_batches} "
                    f"loss={losses[-1]:.4f} avg_loss={float(np.mean(losses)):.4f} "
                    f"speed={bps:.2f} batch/s eta_epoch={_format_duration(eta)}",
                    flush=True,
                )

        epoch_times.append(time.monotonic() - epoch_start)

        # ── validate ──
        print(f"epoch {epoch}/{config.epochs} validation start", flush=True)
        val_metrics = evaluate_dygkt(
            model, val_data, split_loaders["validation"], eval_sampler, device,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_metrics}
        history.append(row)

        elapsed_run = time.monotonic() - run_start
        average_epoch = elapsed_run / epoch
        eta_run = average_epoch * (config.epochs - epoch)
        print(
            f"epoch {epoch}/{config.epochs} done in {_format_duration(epoch_times[-1])} "
            f"loss={row['train_loss']:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_ap={val_metrics['average_precision']:.4f} "
            f"elapsed={_format_duration(elapsed_run)} eta_total={_format_duration(eta_run)}",
            flush=True,
        )

        # ── checkpoint ──
        score = val_metrics["roc_auc"]
        if np.isnan(score):
            score = -val_metrics["log_loss"]
        if score > best_auc:
            best_auc = score
            stale = 0
            torch.save(
                {"model": model.state_dict(), "config": asdict(config)},
                output_dir / "best.pt",
            )
            print(f"  checkpoint saved: {output_dir / 'best.pt'}", flush=True)
        else:
            stale += 1
            print(f"  no validation improvement, stale={stale}/{config.patience}", flush=True)
            if stale >= config.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
        scheduler.step()

    # ── final evaluation ──
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model[0].set_neighbor_sampler(eval_sampler)
    print("final validation and test evaluation start", flush=True)
    result = {
        "best_validation": evaluate_dygkt(
            model, val_data, split_loaders["validation"], eval_sampler, device,
        ),
        "test": evaluate_dygkt(
            model, test_data, split_loaders["test"], eval_sampler, device,
        ),
        "epochs": history,
        "split_counts": {
            "train": train_data.num_interactions,
            "validation": val_data.num_interactions,
            "test": test_data.num_interactions,
        },
    }
    write_json(output_dir / "metrics.json", result)
    print(f"metrics saved: {output_dir / 'metrics.json'}", flush=True)
    return result
