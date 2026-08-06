from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Workaround: Flash/Mem-Efficient SDPA can cause illegal memory access on
# CUDA 13.0 + Turing GPUs (e.g. RTX 2080 Ti). Force math backend.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from .dataset import ProcessedData, TemporalHistoryDataset, build_history_cache, compute_global_stats
from .io import write_json
from .metrics import binary_metrics
from .model import DySemKT


@dataclass
class TrainConfig:
    seed: int = 42
    split: str = "cold"
    feature_mode: str = "hybrid"
    d_model: int = 128
    history_length: int = 40
    retrieval: str = "hybrid"
    dropout: float = 0.1
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5
    device: str = "auto"
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    if not labels:
        raise ValueError("evaluation split is empty")
    return binary_metrics(np.concatenate(labels), np.concatenate(probabilities))


def train(data_dir: Path, output_dir: Path, config: TrainConfig) -> dict:
    if config.split not in {"temporal", "cold"}:
        raise ValueError("split must be temporal or cold")
    set_seed(config.seed)
    data = ProcessedData(data_dir)
    split = data.temporal_split if config.split == "temporal" else data.cold_split
    train_mask = split == 0
    allowed_history = train_mask.copy()

    # ── Load or build history cache ──
    cache_name = "history_cache.npy" if config.retrieval == "hybrid" else "history_cache_recent.npy"
    cache_path = data_dir / cache_name
    if cache_path.exists():
        history_cache = np.load(cache_path)
        print(f"history cache loaded: {cache_path}  shape={history_cache.shape}  "
              f"dtype={history_cache.dtype}  retrieval={config.retrieval}  ⚡ fast path", flush=True)
    else:
        print(f"history cache not found — building now (retrieval={config.retrieval})...", flush=True)
        history_cache = build_history_cache(
            data, allowed_history, history_length=config.history_length,
            cache_path=cache_path, retrieval=config.retrieval,
        )
        print(f"history cache built and saved: {cache_path}  shape={history_cache.shape}", flush=True)

    # ── Compute global question stats from training data ──
    global_stats = compute_global_stats(data, train_mask)
    print(f"global stats computed: {global_stats.shape}", flush=True)

    datasets = {
        name: TemporalHistoryDataset(
            data, np.flatnonzero(split == code), config.history_length, allowed_history,
            global_stats=global_stats, history_cache=history_cache,
            retrieval=config.retrieval,
        )
        for name, code in (("train", 0), ("validation", 1), ("test", 2))
    }
    loaders = {
        name: DataLoader(
            dataset, batch_size=config.batch_size, shuffle=name == "train",
            num_workers=config.num_workers,
        )
        for name, dataset in datasets.items()
    }
    device_name = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    print(
        f"train config: device={device} split={config.split} feature_mode={config.feature_mode} "
        f"epochs={config.epochs} patience={config.patience} batch_size={config.batch_size} "
        f"learning_rate={config.learning_rate:g} d_model={config.d_model} "
        f"history_length={config.history_length} retrieval={config.retrieval} "
        f"dropout={config.dropout:g} num_workers={config.num_workers}",
        flush=True,
    )
    print(
        f"dataset sizes: train={len(datasets['train'])} validation={len(datasets['validation'])} "
        f"test={len(datasets['test'])}",
        flush=True,
    )
    model = DySemKT(
        torch.from_numpy(data.question_features), d_model=config.d_model,
        dropout=config.dropout, max_history=config.history_length,
        feature_mode=config.feature_mode,
    ).to(device)
    model.summarize()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01,
    )
    criterion = nn.BCEWithLogitsLoss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(config))
    best_auc = -float("inf")
    stale = 0
    history = []
    run_start = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        epoch_start = time.monotonic()
        total_batches = len(loaders["train"])
        progress_every = max(1, min(50, total_batches // 10))
        print(f"epoch {epoch}/{config.epochs} start, train batches={total_batches}", flush=True)
        for batch_index, batch in enumerate(loaders["train"], start=1):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index == 1 or batch_index == total_batches or batch_index % progress_every == 0:
                elapsed = time.monotonic() - epoch_start
                batches_per_second = batch_index / max(elapsed, 1e-9)
                remaining_batches = total_batches - batch_index
                eta_epoch = remaining_batches / max(batches_per_second, 1e-9)
                average_loss = float(np.mean(losses))
                print(
                    f"  train progress: batch {batch_index}/{total_batches} "
                    f"loss={losses[-1]:.4f} avg_loss={average_loss:.4f} "
                    f"speed={batches_per_second:.2f} batch/s eta_epoch={_format_duration(eta_epoch)}",
                    flush=True,
                )
        print(f"epoch {epoch}/{config.epochs} validation start", flush=True)
        validation = evaluate(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_seconds = time.monotonic() - epoch_start
        elapsed_run = time.monotonic() - run_start
        average_epoch = elapsed_run / epoch
        eta_run = average_epoch * (config.epochs - epoch)
        print(
            f"epoch {epoch}/{config.epochs} done in {_format_duration(epoch_seconds)} "
            f"loss={row['train_loss']:.4f} val_auc={validation['roc_auc']:.4f} "
            f"val_ap={validation['average_precision']:.4f} lr={current_lr:.2g} "
            f"elapsed={_format_duration(elapsed_run)} eta_total={_format_duration(eta_run)}",
            flush=True,
        )
        score = validation["roc_auc"]
        if np.isnan(score):
            score = -validation["log_loss"]
        if score > best_auc:
            best_auc = score
            stale = 0
            torch.save({"model": model.state_dict(), "config": asdict(config)}, output_dir / "best.pt")
            print(f"  checkpoint saved: {output_dir / 'best.pt'}", flush=True)
        else:
            stale += 1
            print(f"  no validation improvement, stale={stale}/{config.patience}", flush=True)
            if stale >= config.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print("final validation and test evaluation start", flush=True)
    result = {
        "best_validation": evaluate(model, loaders["validation"], device),
        "test": evaluate(model, loaders["test"], device),
        "epochs": history,
        "split_counts": {name: len(dataset) for name, dataset in datasets.items()},
    }
    write_json(output_dir / "metrics.json", result)
    print(f"metrics saved: {output_dir / 'metrics.json'}", flush=True)
    return result
