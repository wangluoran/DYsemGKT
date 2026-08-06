"""Verify hybrid retrieval is working on server."""
import sys
sys.path.insert(0, "src")
from dysemkt.dataset import ProcessedData, TemporalHistoryDataset, compute_global_stats
from pathlib import Path
import numpy as np
import bisect

data = ProcessedData(Path("data/processed/moocradar_api"))
split = data.cold_split
train_mask = split == 0
global_stats = compute_global_stats(data, train_mask)
ds = TemporalHistoryDataset(
    data, np.flatnonzero(split == 0), history_length=40,
    allowed_history=train_mask, global_stats=global_stats,
)

# Count hybrid vs simple recent items
n_samples = 200
total_hybrid = 0
total_recent = 0
hybrid_counts = []
recent_counts = []

for i in range(n_samples):
    batch = ds[i]
    hybrid_count = int(batch["student_mask"].sum().item())
    user = int(data.user[int(ds.indices[i])])
    event = int(ds.indices[i])
    values = ds.user_events.get(user, [])
    end = bisect.bisect_left(values, event)
    recent = values[max(0, end - 40):end]
    total_hybrid += hybrid_count
    total_recent += len(recent)
    hybrid_counts.append(hybrid_count)
    recent_counts.append(len(recent))

# How often does hybrid select DIFFERENT items than pure recent?
diff_count = sum(1 for h, r in zip(hybrid_counts, recent_counts) if h != r)

print(f"Samples checked: {n_samples}")
print(f"Avg hybrid items: {total_hybrid/n_samples:.1f}")
print(f"Avg recent items: {total_recent/n_samples:.1f}")
print(f"Cases where hybrid != recent count: {diff_count}/{n_samples}")
print()

# Show distribution
from collections import Counter
print("Hybrid count distribution:")
for k, v in sorted(Counter(hybrid_counts).items()):
    print(f"  {k} items: {v} samples ({100*v/n_samples:.0f}%)")
print()

# Show a sample where counts differ
for i in range(n_samples):
    if hybrid_counts[i] != recent_counts[i]:
        b = ds[i]
        user = int(data.user[int(ds.indices[i])])
        event = int(ds.indices[i])
        item = int(b["item"])
        print(f"--- Sample {i} (user={user}, item={item}) ---")
        print(f"  hybrid count: {hybrid_counts[i]}, recent count: {recent_counts[i]}")
        print(f"  student_items: {b['student_item'][-min(10,hybrid_counts[i]):].tolist()}")
        print(f"  same_question: {b['same_question'][-min(10,hybrid_counts[i]):].tolist()}")
        print(f"  same_exercise: {b['same_exercise'][-min(10,hybrid_counts[i]):].tolist()}")
        print(f"  concept_overlap: {[round(x,2) for x in b['concept_overlap'][-min(10,hybrid_counts[i]):].tolist()]}")
        print(f"  mask (last 10): {b['student_mask'][-min(10,hybrid_counts[i]):].tolist()}")
        break
