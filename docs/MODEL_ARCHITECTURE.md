# DySemKT 模型架构说明

## 1. 模型目标

DySemKT 用于预测学生在时间 `t` 回答题目 `q` 时答对的概率：

$$
P(y_{u,q,t}=1 \mid H_u(t), H_q(t), S_q)
$$

即：根据学生历史、题目历史和当前题目语义，预测本次回答正确的概率。

其中：

- `u`：学生；
- `q`：当前题目；
- `t`：当前交互时间；
- `H_u(t)`：学生在 `t` 之前的答题历史；
- `H_q(t)`：题目在 `t` 之前被其他学生作答的历史；
- `S_q`：题目的静态文本语义；
- `y`：当前回答是否正确。

模型保留 DyGKT 的学生-题目连续时间二部图思想，同时引入真实题干语义和双侧 Transformer 历史编码。

```text
                         当前题目文本
                              |
                      预训练文本编码器
                              |
                       当前题目表示 e_q
                              |
             +----------------+----------------+
             |                                 |
      学生侧历史编码                      题目侧历史编码
             |                                 |
     student_state                      question_state
             |                                 |
             +-------------门控融合------------+
                              |
                  动态状态与当前题目匹配
                              |
                         P(correct)
```

核心实现位于 `src/dysemkt/model.py`。

## 2. 输入数据

每个预测事件包含：

| 字段 | 含义 |
|---|---|
| `item` | 当前题目 ID |
| `label` | 当前回答是否正确，仅作为监督标签 |
| `student_item` | 学生最近回答过的题目 |
| `student_response` | 学生历史回答结果 |
| `student_delta` | 学生历史事件距当前时刻的秒数 |
| `student_mask` | 学生历史有效位置 |
| `question_response` | 当前题目的历史回答结果 |
| `question_delta` | 题目历史事件距当前时刻的秒数 |
| `question_mask` | 题目历史有效位置 |

默认最多取两侧各 50 条历史。数据集通过事件索引二分查找，只返回当前事件之前的交互。

## 3. 题目文本语义

预处理阶段使用以下字段构造文本：

```text
题目组：第一章作业
题目：“逻辑”一词其语源最初来自：
选项：A. 英语 B. 法语 C. 拉丁语 D. 希腊语
概念：希腊语；逻辑
```

模型输入不包含标准答案、当前作答结果、测试集统计量或全数据难度。

文本先由预训练模型编码：

$$
s_q=\operatorname{TextEncoder}(title, content, options, concepts)
$$

其中，$s_q$ 是预训练文本编码器输出的原始题目向量。

使用 BGE-M3 时，原始题目语义通常为 1024 维。文本向量在 KT 训练阶段作为固定特征保存，再投影到模型隐藏维度 `d`：

$$
e_q^{sem}=\operatorname{GELU}\left(\operatorname{LayerNorm}(W_s s_q+b_s)\right)
$$

默认 `d=256`。

## 4. 三种题目表示模式

### 4.1 Semantic

只使用题目文本语义：

$$
e_q=e_q^{sem}
$$

该模式不依赖题目 ID，适合严格未见题目冷启动实验。

### 4.2 ID

只使用可训练题目 ID embedding：

$$
e_q=e_q^{id}
$$

这是传统题目表示对照组。它可以记忆训练题目，但不能表示未见题目。

### 4.3 Hybrid

融合语义和题目 ID：

$$
e_q=e_q^{sem}+e_q^{id}
$$

普通时间预测以该模式为主。严格题目冷启动实验应主要使用 `semantic`，因为冷启动题目的 ID embedding 未经过训练。

## 5. 时间编码

对每条历史事件计算距当前预测时刻的时间差：

$$
\Delta t_k=t-t_k
$$

时间跨度可能从数秒到数月，因此先进行对数压缩：

$$
\widehat{\Delta t_k}=\frac{\log(1+\Delta t_k)}{16}
$$

然后通过两层 MLP 投影：

$$
e_k^{time}=W_2\operatorname{GELU}\left(W_1\widehat{\Delta t_k}+b_1\right)+b_2
$$

该表示允许模型区分近期交互和远期交互，同时避免原始秒数导致数值范围过大。

## 6. 学生侧动态编码

学生侧历史是当前学生最近的 `L` 条交互：

$$
H_u(t)=\{(q_1,r_1,t_1),\ldots,(q_L,r_L,t_L)\}
$$

对历史题目与当前题目计算三种显式结构关系：

$$
c_k=[I(q_k=q),\ I(exercise_k=exercise_q),\ J(concepts_k,concepts_q)]
$$

其中，$I(\cdot)$ 是 0/1 指示函数，$J(\cdot)$ 是两个概念集合的 Jaccard 重叠率：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

因此，精确重复同一道题、同练习单元下的不同题，以及概念相似题不会被混为同一种关系。

每条学生历史交互表示为：

$$
x_k^{student}=e_{q_k}+e_{r_k}+e_{\Delta t_k}+W_c c_k
$$

其中：

- `e_qk`：历史题目的语义、ID 或混合表示；
- `e_rk`：答对/答错 embedding；
- `e_delta`：时间差表示；
- `c_k`：同题、同练习单元和概念重叠结构特征。

模型还汇总当前学生对同一道题的重复行为：

$$
v_{repeat}=[I(repeated),\ \log(1+count),\ last\_correct,\ \log(1+last\_delta)]
$$

它包含是否做过、历史重复次数、上次是否答对和距上次同题作答的时间。

学生编码器的状态 token 同时注入当前题目表示，使历史注意力以当前题目为条件：

$$
STATE_u^{(0)}=learned\_token+e_q
$$

随后添加位置编码并送入 Transformer：

$$
Z_u=\operatorname{Transformer}_u\left(
[STATE_u^{(0)},x_1^{student},\ldots,x_L^{student}]+P
\right)
$$

取第一个 token 的输出，并加入重复行为摘要：

$$
h_u=Z_u[:,0]+\operatorname{MLP}_{repeat}(v_{repeat})
$$

默认学生编码器配置：

- 2 层 Pre-LN Transformer；
- 4 个注意力头；
- 隐藏维度 256；
- FFN 维度 512；
- GELU 激活；
- 最大历史长度 50。

模型没有额外使用 causal mask，因为输入已经严格限制为当前事件之前的历史。Transformer 可以在所有过去事件之间建模，但不能访问当前答案或未来事件。

## 7. 题目侧动态编码

题目侧历史是当前题目最近的 `L` 次作答：

$$
H_q(t)=\{(r_1,t_1),\ldots,(r_L,t_L)\}
$$

每条题目历史交互表示为：

$$
x_k^{question}=e_{r_k}+e_{\Delta t_k}
$$

当前版本不引入历史学生 ID，而是编码该题最近的群体作答结果和时间变化。这降低了学生 ID 记忆和隐私依赖，并使题目动态状态更容易迁移。

题目编码器的状态 token 由当前题目表示初始化：

$$
STATE_q^{(0)}=learned\_token+e_q
$$

随后计算题目动态状态：

$$
h_q=\operatorname{Transformer}_q\left(
[STATE_q^{(0)},x_1^{question},\ldots,x_L^{question}]+P
\right)[:,0]
$$

因此 `h_q` 不是简单的历史正确率，而是当前题目语义条件下的历史群体行为表示。

题目编码器默认使用学生编码器一半的层数，即学生侧 2 层时题目侧使用 1 层。

## 8. 动态门控融合

模型此时得到三个 `d` 维向量：

- `h_u`：学生动态知识状态；
- `h_q`：题目动态群体状态；
- `e_q`：当前题目静态表示。

先根据学生状态、题目状态和当前题目表示计算逐维门控：

$$
g=\sigma\left(W_g[h_u;h_q;e_q]+b_g\right)
$$

门控权重的每个维度均位于 `(0,1)`。随后融合双侧动态状态：

$$
h_{dynamic}=g\odot h_u+(1-g)\odot h_q
$$

这意味着模型可以让部分维度更依赖学生个人历史，另一些维度更依赖题目的群体作答行为。当前题目语义参与门控计算，因此不同题目可以采用不同的融合比例。

## 9. 状态-题目匹配与预测

除了拼接动态状态和题目表示，模型还加入逐维匹配：

$$
m=h_{dynamic}\odot e_q
$$

它表示当前学生动态状态与题目要求之间的匹配关系。最终预测输入为：

$$
z=[h_{dynamic};e_q;m]\in\mathbb{R}^{3d}
$$

预测头计算：

$$
logit=W_2\operatorname{Dropout}\left(
\operatorname{GELU}\left(W_1\operatorname{LayerNorm}(z)+b_1\right)
\right)+b_2
$$

最终正确概率为：

$$
P(correct)=\sigma(logit)
$$

## 10. 张量维度

设批大小为 `B`、历史长度为 `L`、文本维度为 `D_text`、模型维度为 `d`：

| 张量 | 维度 |
|---|---|
| 原始题目语义 | `(num_items, D_text)` |
| 当前题目表示 | `(B, d)` |
| 学生历史题目 | `(B, L, d)` |
| 学生历史序列 | `(B, L, d)` |
| 同题/同练习/概念重叠 | `(B, L, 3)` |
| 重复行为摘要 | `(B, 4)` |
| 题目历史序列 | `(B, L, d)` |
| 学生状态 | `(B, d)` |
| 题目状态 | `(B, d)` |
| 门控向量 | `(B, d)` |
| 预测头输入 | `(B, 3d)` |
| 输出 logit | `(B,)` |

默认配置下：

```text
D_text = 1024  # BGE-M3
d      = 256
L      = 50
```

## 11. 数据泄漏控制

数据层分别建立：

```text
student_id -> 历史事件索引
question_id -> 历史事件索引
```

针对当前事件 `i`，通过二分查找只返回索引 `j < i` 的历史事件。同题重复次数、上次同题结果和重复间隔也只从这些允许历史中计算。

训练与评估默认进一步设置：

```python
allowed_history = train_mask
```

因此验证和测试预测只能使用训练集标签作为历史，验证集和测试集答案不会通过学生侧或题目侧动态图反向泄漏。

这是保守的离线评估方式。真实在线系统可以在一次回答完成后，将其标签加入下一次预测历史；当前默认实验没有启用这种在线更新。

## 12. 训练目标

模型使用二元交叉熵。直观含义是：答对样本的预测概率应接近 1，答错样本的预测概率应接近 0。

$$
\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

其中，$p_i=\sigma(logit_i)$，$y_i\in\{0,1\}$。

实现采用 `BCEWithLogitsLoss`，直接接收模型输出的 logit。

默认训练配置：

- AdamW；
- 学习率 `5e-4`；
- 权重衰减 `1e-4`；
- 梯度裁剪 `5.0`；
- 最多训练 30 轮；
- 验证集 ROC-AUC early stopping；
- patience 为 5。

评估报告 ROC-AUC、Average Precision、Log Loss 和 Accuracy。MOOCRadar 正确率约为 81.55%，因此不能单独使用 Accuracy 判断模型效果。

## 13. 与原始 DyGKT 的关系

DySemKT 保留以下 DyGKT 思想：

- 学生和题目构成二部图；
- 每次作答是带时间戳和正确性标签的动态边；
- 学生侧和题目侧都使用历史邻居；
- 当前预测只能访问过去交互；
- 以连续时间事件预测为核心，而不是人为拆分静态学生序列。

主要变化如下：

| 原始 DyGKT | DySemKT |
|---|---|
| 题目主要依赖 ID 或知识点特征 | 使用真实题干、选项和概念语义 |
| GRU 风格历史聚合 | 双侧 Pre-LN Transformer |
| 结构特征依赖具体数据字段 | 显式、统一的数据契约 |
| 未见题目迁移能力有限 | 支持严格题目语义冷启动 |
| 外部 MergeLayer 分类 | 模型内部完成状态-题目匹配 |
| 数据统计和历史边界不明确 | 训练历史白名单控制泄漏 |
| 模型输出维度随数据变化 | 所有内部状态统一为 `d` 维 |

## 14. 当前架构边界

当前版本刻意没有加入：

- 标准答案 embedding；
- 全数据难度与区分度；
- 概念知识图谱 GCN；
- `knowledge_type` 独立 embedding；
- `cognitive_dimension` 独立 embedding；
- 课程和练习层级 embedding；
- 题目侧历史学生的动态知识状态；
- 显式遗忘曲线或单调注意力；
- 同概念题目的额外结构边。

`knowledge_type` 和 `cognitive_dimension` 已在原始数据解析阶段读取，但当前模型尚未使用。第一版的目标是先建立边界清晰的真实语义基线，再通过受控消融逐步加入概念图、认知维度和多层级结构。

因此，当前架构最准确的定位是：

> 真实题目语义驱动的双侧连续时间动态图知识追踪模型，而不是简单地用文本 embedding 替换题目 ID。
