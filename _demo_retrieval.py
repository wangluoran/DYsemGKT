"""Demo: pick a question, retrieve most similar ones by BGE-M3 cosine similarity."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

data_dir = Path("data/processed/moocradar_api")

# Load
features = np.load(data_dir / "question_features.npy").astype(np.float32)
texts = []
with (data_dir / "question_text.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

# Pick a query question
query_id = 1000  # change this to try different questions
query_vec = features[query_id]
print(f"=== 查询题目 ID={query_id} ===")
print(texts[query_id][:500])
print()

# Cosine similarity with all questions
query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
norms = np.linalg.norm(features, axis=1) + 1e-8
all_sim = np.dot(features, query_norm) / norms  # cosine sim

# Top-K (exclude self)
k = 10
top_idx = np.argsort(all_sim)[::-1]
top_idx = [i for i in top_idx if i != query_id][:k]

print(f"=== Top-{k} 最相似题目（排除自身）===")
for rank, idx in enumerate(top_idx, 1):
    print(f"\n--- #{rank}  ID={idx}  cosine_sim={all_sim[idx]:.4f} ---")
    print(texts[idx][:400])
