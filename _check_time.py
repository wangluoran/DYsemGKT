import numpy as np
data = np.load("data/processed/moocradar_api/events.npz")
ts = data["timestamp"]

print("timestamp range:", ts.min(), "-", ts.max())
print("total span (days):", (ts.max() - ts.min()) / 86400)

deltas = np.diff(ts)
print("\ndeltas between adjacent events (seconds):")
for p in [1, 5, 25, 50, 75, 95, 99]:
    v = np.percentile(deltas, p)
    print(f"  p{p:2d}: {v:8.0f}s = {v/3600:6.1f}h = {v/86400:.2f}d")
print(f"  mean: {deltas.mean():.0f}s, std: {deltas.std():.0f}s, max: {deltas.max()/86400:.0f}d")

# Per-student history span
users = data["user"]
user_ts = {}
for i, (u, t) in enumerate(zip(users, ts)):
    user_ts.setdefault(int(u), []).append(float(t))
spans = []
for u, times in user_ts.items():
    if len(times) > 1:
        spans.append(max(times) - min(times))
spans = np.array(spans)
print("\nper-student history span:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  span p{p:2d}: {np.percentile(spans, p)/86400:.1f}d")
print(f"  span mean: {spans.mean()/86400:.1f}d, max: {spans.max()/86400:.1f}d")

# The actual delta in the dataset (student_delta in batch)
# Let's also check what the 50-history window looks like
print("\n50-history window spans (per-student, last 50 events):")
w50_spans = []
for u, times in user_ts.items():
    if len(times) >= 50:
        last50 = sorted(times)[-50:]
        w50_spans.append(last50[-1] - last50[0])
w50_spans = np.array(w50_spans)
for p in [1, 5, 25, 50, 75, 95, 99]:
    v = np.percentile(w50_spans, p)
    print(f"  span p{p:2d}: {v:.0f}s = {v/3600:.1f}h = {v/86400:.2f}d")
print(f"  mean: {w50_spans.mean()/86400:.1f}d")
