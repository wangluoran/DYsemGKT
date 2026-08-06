# DySemKT 模型架构说明

## 1. 整体架构

```
                        当前题目 Q_raw (1024维 BGE) + Q_ID
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        【学生塔】        【题目塔】        【残差】
     long-term state    difficulty+recency  1024→128
              │               │               │
              └───────┬───────┘               │
                      ▼                       │
               【最终门控】                    │
              g_s·stu + g_q·ques              │
                      │                       │
                      ├───────────────────────┘
                      ▼
                [output, Q, output⊙Q]
                      │
                   predictor
                      │
                  P(correct)
```

核心位于 `src/dysemkt/model.py`，d_model=128。

## 2. 题目语义表示

预处理阶段使用 BGE-M3 编码题目文本（题干+选项+概念），得到 1024 维原始向量。

语义投影（带 ReLU 激活，保留稀疏性）：

`s_q → Linear(1024, 512) → LayerNorm → ReLU → Linear(512, 512)`

三种 `feature_mode`：

| 模式 | 当前题 Q | 说明 |
|---|---|---|
| `semantic` | `Linear(512→128)` | 纯 BGE 语义，冷启动默认 |
| `id` | `Linear(128→128)` | 纯 ID embedding，对照基线 |
| `hybrid` | `Linear(512+128→128)` | BGE + ID 拼接 |

## 3. 混合检索器（Hybrid Retriever）

训练前预计算并缓存到 `history_cache.npy`。若缓存不存在，`engine.py` 训练启动时自动生成。

### 3.1 检索策略

从学生全部历史中取 N=40 条：

| 池 | 数量 | 策略 |
|---|---|---|
| **强制近因窗口** | 16条 | 无论相关性，强行取时间最近的16条 |
| **语义相关性窗口** | 24条 | 在剩余历史中按结构分数取 Top-24 |

### 3.2 评分公式

```
Score = 1.5 × I_同题 + 1.0 × I_同练习 + 0.8 × ConceptOverlap
```

语义相关性窗口严格剔除同题 ID（防标签泄漏），因此 I_同题 在此池中恒为 0。
Score=0 的记录被物理丢弃，不进入模型计算。两池合并去重后按时间排序。

## 4. 学生塔

### 4.1 输入

每条历史包含四个模态：

| 模态 | 原始信号 | 投影 | 说明 |
|---|---|---|---|
| 语义 K/V | 512维语义 | proj_k → 128 | 历史题目的语义 |
| 对错 V | 0/1 标量 | Embedding(2,4) → Linear(4,128) | 做对还是做错 |
| 时间 V | 秒数标量 | log(1+Δ)/16 → Linear(1,8) → Linear(8,128) | 距现在多久 |
| 结构 V+bias | [同题?,同练习?,概念重叠] 3维 | Linear(3,16) → Linear(16,128)(V) + MLP→1(bias) | 历史题与当前题的关系 |

### 4.2 关系感知注意力

```
logits = Q·Kᵀ / √128          ← 标准缩放点积
logits += struct_bias × √128   ← 结构偏置直接劫持 attention logits
logits = mask_fill(mask==0, -∞) ← 排除同题（防泄漏）+ 填充位
attn = softmax(logits)
```

**关键设计决策**：不相关的历史（struct_raw 全零）不再被 mask 掉。
`struct_bias([0,0,0])` 自然输出低偏置 → Softmax 后权重极低但非零 → 保留梯度流动。
这在冷启动（新题与历史无结构关联）下至关重要。

### 4.3 四模态门控融合

```python
ctx_sem    = attn @ K           # 语义上下文
ctx_resp   = attn @ V_resp      # 对错上下文
ctx_time   = attn @ V_time      # 时间上下文
ctx_struct = attn @ V_struct    # 结构上下文

gates = Softmax(MLP(512→128→4))  # 4-modal gate
student_out = g₀·ctx_sem + g₁·ctx_resp + g₂·ctx_time + g₃·ctx_struct
```

## 5. 题目塔

### 5.1 输入

| 信号 | 说明 |
|---|---|
| 自历史对错 | 该学生之前做这道题的结果 (0/1) |
| 自历史时间 | 距当前的秒数 |
| 全局统计 | [正确率, log(1+尝试次数), log(1+平均耗时)] 3维，训练集预计算 |

### 5.2 Q-K 空间对齐

```python
Q_proj = q_to_question(Q)    # Linear(128,128, bias=False) — 将 Q 映射到 K_q 空间
logits = Q_proj·K_qᵀ / √128
logits += log(clamp(exp(-Δh/τ), min=-10.0))  ← 可学习时间衰减，clamp 防梯度消失
attn = softmax(logits)
question_out = attn @ K_q
```

`q_to_question` 确保语义空间的 Q 与 响应+时间+统计空间的 K_q 在同一个可训练流形中进行点积。

### 5.3 冷启动处理

两层保护：

1. **无自历史**：使用可学习的 `empty_question` 向量作为题目塔输出。
2. **全局统计缺失**（冷启动测试题）：训练时以 30% 概率随机将 `global_stats` 置零，
   强迫模型学会在缺失全局统计时依赖"对错+时间衰减"做判断。

## 6. 最终融合

```python
# 双塔门控 (隐层扩容至128，避免退化为平均池化)
gates = Softmax(MLP(384→128→2))  # [student_out, question_out, Q]
fused = g_s·student_out + g_q·question_out

# 语义残差（保底）
output = fused + Linear(1024→128)(Q_raw)

# FM二阶交互 + 预测
logit = LN → Linear(384→128) → ReLU → Dropout → Linear(128→1)
```

## 7. 关键保护机制

| 机制 | 位置 | 目的 |
|---|---|---|
| 同题排除 | 学生塔 mask | 防标签泄漏（上次做对→这次也对） |
| 无关历史不丢弃 | 学生塔 mask | 冷启动下保留全部历史，struct_bias 自然降权 |
| 冷启动兜底 | 题目塔 empty_question | 无自历史时不输出噪声 |
| 全局统计 Dropout | 题目塔 (training 30%) | 强迫模型适应冷启动缺失全局统计 |
| 语义残差 | 输出层残差连接 | BGE 原始语义直达预测，不经过 attention 稀释 |
| 结构劫持 | 学生塔 logits+bias | 结构关系不经过投影，直接控制注意力分配 |
| 时间衰减+clamp | 题目塔 logits+log(decay) | 近期交互天然比远期更重要，clamp(-10) 防 NaN |
| Q-K 空间对齐 | 题目塔 q_to_question | Q(语义空间) 与 K_q(混合空间) 可训练对齐 |

## 8. 张量维度

| 张量 | 维度 |
|---|---|
| 原始题目语义 (question_features) | (num_items, 1024) |
| 语义投影后 | (B, N, 512) |
| 所有模态统一维度 (d_model) | 128 |
| 当前题 Q | (B, 128) |
| 学生历史 K/V | (B, 40, 128) |
| 结构偏置 | (B, 40, 1) |
| 注意力权重 | (B, 1, 40) |
| 学生塔输出 | (B, 128) |
| 题目塔输出 | (B, 128) |
| 最终门控权重 | (B, 2) |
| 预测中间层 | (B, 128) |
| 输出 logit | (B,) |

## 9. 训练配置

| 参数 | 默认值 |
|---|---|
| d_model | 128 |
| max_history | 40 |
| dropout | 0.1 |
| 优化器 | AdamW |
| 学习率 | 5e-4 |
| 权重衰减 | 1e-4 |
| 梯度裁剪 | 5.0 |
| 最大轮数 | 30 |
| Early stopping | ROC-AUC, patience=5 |
| 损失函数 | BCEWithLogitsLoss |
| history_cache | 自动生成/加载 |

## 10. 数据泄漏控制

数据集通过二分查找只返回当前事件之前的历史。训练与评估默认设置 `allowed_history = train_mask`，验证和测试预测只能使用训练集标签作为历史。

学生塔排除 `same_question==1` 的位置，防止模型通过"上次做对/错这道题"直接得到答案。

题目塔的自历史仅用于捕捉学生对同一道题的重复作答模式，不参与学生塔的语义关联推理。

混合检索器（Hybrid Retriever）的语义相关性窗口严格剔除同题 ID。强制近因窗口允许同题（天然存在，无法避免）。

## 11. 与原始 DyGKT 的关系

| 原始 DyGKT | DySemKT (当前) |
|---|---|
| GRU 序列编码 | 关系感知单层 Attention |
| 节点特征+边特征+时间+结构 add 混合 | 4模态独立 K/V + 门控融合 |
| 结构特征作为普通特征 | 结构特征劫持 attention logits |
| 无语义残差 | BGE 1024→128 残差连接 |
| 双塔 GRU 对称设计 | 学生塔(排除同题)+题目塔(自历史+全局)非对称 |
| MergeLayer 预测 | FM二阶交互 + ReLU → 128 → Dropout → 1 |

## 12. 已实现特性

- ✅ 混合检索器（强制最近16条 + 结构分数Top24，N=40）
- ✅ history_cache 自动生成与加载
- ✅ 题目塔 Q-K 空间对齐 (q_to_question)
- ✅ 全局统计训练时随机 Dropout (30%)
- ✅ 学生塔不丢弃无关历史
- ✅ 门控隐层扩容 (64→128)
- ✅ predictor 128维 + ReLU
- ✅ 时间衰减 log clamp (-10.0)
- ✅ global_stats 训练集预计算 (compute_global_stats)
- ✅ 冷启动空嵌入 (empty_question)

## 13. 当前边界

当前版本刻意未包含：
- 标准答案 embedding
- 概念图 GCN / 课程层级 embedding
- 负采样训练
- 多尺度时间编码
- IRT 参数化 (difficulty/discrimination)
