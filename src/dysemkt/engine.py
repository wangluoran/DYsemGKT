from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ProcessedData, TemporalHistoryDataset
from .io import write_json
from .metrics import binary_metrics
from .model import DySemKT


@dataclass
class TrainConfig:
    seed: int = 42
    split: str = "temporal"
    feature_mode: str = "hybrid"
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    history_length: int = 50
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 5e-4
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
    datasets = {
        name: TemporalHistoryDataset(
            data, np.flatnonzero(split == code), config.history_length, allowed_history,
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
    model = DySemKT(
        torch.from_numpy(data.question_features), hidden_dim=config.hidden_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        dropout=config.dropout, max_history=config.history_length,
        feature_mode=config.feature_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(config))
    best_auc = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_auc={validation['roc_auc']:.4f} val_ap={validation['average_precision']:.4f}"
        )
        score = validation["roc_auc"]
        if np.isnan(score):
            score = -validation["log_loss"]
        if score > best_auc:
            best_auc = score
            stale = 0
            torch.save({"model": model.state_dict(), "config": asdict(config)}, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= config.patience:
                break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    result = {
        "best_validation": evaluate(model, loaders["validation"], device),
        "test": evaluate(model, loaders["test"], device),
        "epochs": history,
        "split_counts": {name: len(dataset) for name, dataset in datasets.items()},
    }
    write_json(output_dir / "metrics.json", result)
    return result
