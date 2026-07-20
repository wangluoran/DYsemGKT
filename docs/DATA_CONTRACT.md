# Processed Data Contract

Each processed dataset directory contains:

```text
events.npz
question_features.npy
question_text.jsonl
question_context.json
mappings.json
metadata.json
```

`events.npz` arrays have equal length and are globally sorted by timestamp:

| Array | Type | Meaning |
|---|---|---|
| `user` | int64 | Dense student index `[0, num_users)` |
| `item` | int64 | Dense question index `[0, num_items)` |
| `timestamp` | int64 | Unix timestamp in seconds |
| `label` | float32 | Binary response correctness |
| `temporal_split` | int8 | 0 train, 1 validation, 2 test |
| `cold_split` | int8 | Item-disjoint 0/1/2 split |

`question_features.npy` is float32 with shape `(num_items, semantic_dim)`.
Rows align with the dense item mapping. `question_text.jsonl` contains the exact
answer-free text passed to the encoder for auditability.

`question_context.json` follows the same dense item order and stores each
question's `course_id`, `exercise_id`, concepts, knowledge type and cognitive
dimension. These fields support explicit same-question, same-exercise and
concept-overlap features without parsing text during training.

Split statistics and source SHA-256 fingerprints are recorded in
`metadata.json`. Difficulty and discrimination are deliberately absent from
the base contract; if added later they must be estimated from training labels
only and saved as split-specific features.
