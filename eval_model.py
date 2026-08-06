#!/usr/bin/env python3
"""Quick evaluation of a trained DySemKT checkpoint."""
import sys, json, time, os, argparse
sys.path.insert(0, "/root/DYsemGKT/src")
import numpy as np
import torch
from torch.utils.data import DataLoader
from dysemkt.dataset import ProcessedData, TemporalHistoryDataset, compute_global_stats, build_history_cache
from dysemkt.model import DySemKT
from dysemkt.metrics import binary_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="/root/DYsemGKT/data/processed/moocradar_api")
parser.add_argument("--checkpoint", default="/root/DYsemGKT/outputs/semantic_cold/best.pt")
args = parser.parse_args()

os.chdir("/root/DYsemGKT")
device = torch.device("cuda")

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
config = ckpt["config"]
print(f"Checkpoint: d_model={config.get('d_model')}, feature_mode={config.get('feature_mode')}, split={config.get('split')}")

data = ProcessedData(args.data_dir)
split_mask = data.cold_split if config.get("split", "cold") == "cold" else data.temporal_split
train_mask = split_mask == 0

global_stats = compute_global_stats(data, train_mask)
hl = config.get("history_length", 40)
cache_path = f"{args.data_dir}/history_cache.npy"
history_cache = np.load(cache_path) if os.path.exists(cache_path) else build_history_cache(data, train_mask, hl, cache_path)

def make_loader(mask):
    ds = TemporalHistoryDataset(data, np.flatnonzero(mask), hl,
                                allowed_history=train_mask, global_stats=global_stats, history_cache=history_cache)
    return DataLoader(ds, batch_size=min(config.get("batch_size", 256), 4096), shuffle=False)

model = DySemKT(data.question_features, d_model=config.get("d_model", 128),
                max_history=hl, dropout=config.get("dropout", 0.1),
                feature_mode=config.get("feature_mode", "semantic")).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

@torch.no_grad()
def evaluate(loader):
    labels, probs = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return binary_metrics(np.concatenate(labels), np.concatenate(probs))

for name, mask in [("Validation", split_mask == 1), ("Test", split_mask == 2)]:
    t0 = time.time()
    m = evaluate(make_loader(mask))
    print(f"{name} ({time.time()-t0:.1f}s): auc={m['roc_auc']:.4f}  ap={m['average_precision']:.4f}  loss={m['log_loss']:.4f}  acc={m['accuracy']:.4f}")
