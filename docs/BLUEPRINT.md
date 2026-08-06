# Architecture Blueprint

## Goal

Build a clean successor experiment to DyGKT in which a question is represented
by both its content and its temporal graph behavior.

## Data Flow

```text
                    【输入】当前题 Q_raw (1024维 BGE) + 当前题 ID
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            【混合检索器】                        【全局特征提取器】
    从该生历史中取40条：                    查这道题的全局统计：
    强制最近16条 +                          (平均正确率, 尝试次数,
    结构分数Top24                           平均耗时) → 3维
    Score=1.5×同题+1.0×同练习+0.8×重叠               │
    (cache: history_cache.npy, 自动生成)              │
                    │                                   │
                    ▼                                   ▼
            【特征对齐】统一映射到128维                   │
    语义512→128, 响应4→128,                             │
    时间8→128, 结构16→128                               │
    保留原始3维关系 → 用于注意力偏置                      │
                    │                                   │
        ┌───────────┴───────────┐                       │
        ▼                       ▼                       ▼
   【学生侧塔】            【题目侧塔】◄──────────────────┘
   学习长期知识状态         学习题目绝对难度+近因记忆
   (排除当前题ID,           (自历史: 该生在此题的过往尝试
   不丢弃无关历史,           + 全局统计3维 @30% Dropout
   4模态门控融合 512→128→4)  + Q-K空间对齐 q_to_question
                             + 可学习时间衰减τ, clamp(-10))
        │                       │
        └───────────┬───────────┘
                    ▼
        【自适应门控融合 384→128→2】
        student vs question
                    +
        【残差连接】原始BGE 1024 → Linear → 128
                    ▼
        【预测层】LN → Linear(384→128) → ReLU → Dropout → Linear(128→1)
                    ▼
               P(correct)
```

## DySemKT

For event `(student, question, time)`:

1. **Student tower**: relation-aware attention over the student's past history
   (excluding same-question to prevent label leakage). Unrelated history is
   NOT masked out — struct_bias([0,0,0]) naturally produces low but non-zero
   attention weights, preserving gradient flow critical for cold-start.
   Four independent modalities — question semantics, response correctness,
   time delta, and structural relation — with 4-way gated fusion (512→128→4).

2. **Question tower**: three sources of information about the current question:
   - **Self-history**: the student's own prior attempts on this specific question,
     with a learnable time-decay parameter τ controlling recency weighting.
     Log-decay clamped to -10.0 for numerical stability.
   - **Q-K space alignment** (`q_to_question`): a bias-free linear layer maps
     the semantic-space Q into the mixed-space K_q before computing attention
     logits, ensuring meaningful dot-product compatibility.
   - **Global feature extractor** (`compute_global_stats`): per-question
     statistics computed from training data only — avg_correctness,
     log(1+attempts), avg_log_time — concatenated as a 3-dim vector.
     During training, global stats are randomly zeroed (30% per sample) to
     force the model to handle cold-start questions without statistics.
   Cold-start questions without self-history use a learnable `empty_question` embedding.

3. **Current question**: raw BGE-1024 vector projected through semantic MLP
   (Linear→LN→ReLU→Linear), then projected to d_model=128 as the attention
   query. An optional ID embedding supports feature_mode comparisons
   (`semantic` / `id` / `hybrid`).

4. **Fusion**: a learned gate (384→128→2) weights student-tower vs question-tower
   outputs, then a **residual connection** from the raw 1024-dim BGE vector (via
   `residual_proj`) directly feeds the predictor — guaranteeing semantic
   information always reaches the output. FM second-order interaction
   [output, Q, output⊙Q] feeds the predictor (LN → 384→128 → ReLU → Dropout → 1).

No current response or future event is visible to either tower.

## Hybrid Retriever

History retrieval is precomputed via `build_history_cache()` and saved to
`data/processed/<dataset>/history_cache.npy`. On training start, if the cache
is missing, it is automatically generated (one-time cost, ~2 minutes for
~900k events).

| Pool | Count | Strategy |
|---|---|---|
| Forced recent window | 16 | Most recent N items regardless of relevance |
| Semantic relevance window | 24 | Top-24 by Score = 1.5×same_q + 1.0×same_ex + 0.8×overlap (same-q excluded) |

Score=0 items are physically discarded. The two pools are merged, deduplicated,
and sorted chronologically to a fixed total of N=40.

## Experiment Matrix

| Axis | Values |
|---|---|
| Feature mode | `id`, `semantic`, `hybrid` |
| Split | strict unseen-question (cold, default), global temporal |
| Model | DySemKT, DyGKT (baseline) |
| Encoder | BGE-M3 (frozen), hash (verification only) |

Primary metrics are ROC-AUC, average precision, log loss and accuracy. Because
MOOCRadar is label-imbalanced, accuracy is never reported alone.

## Milestones

1. Data integrity and deterministic preprocessing. ✅
2. DyGKT baseline ported to shared data pipeline. ✅
3. DySemKT architecture: relation-aware attention, dual-tower, semantic residual. ✅
4. Hybrid retriever: forced-recent + structure-scored, precomputed cache. ✅
5. Cold-start robustness: global stats dropout, Q-K alignment, unrelated-history preservation. ✅
6. Offline smoke model, unit tests, and end-to-end tests (27/27 passing). ✅
7. Temporal and unseen-question ablations across three feature modes. ✅
8. Optional extensions: concept graph encoder, negative sampling, multi-scale time encoding.
