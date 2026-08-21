# DySemKT — 面向题目冷启动的语义动态图知识追踪

DySemKT 是一个面向知识追踪任务的研究项目，核心目标是让模型直接理解“题目本身”，而不是只依赖预定义知识点或题目 ID 记忆历史表现。项目把题干、选项、概念标签和学生作答序列统一到动态历史建模框架中，重点解决未见题目冷启动、语义迁移和时间顺序防泄漏评估问题。

项目当前版本支持 MOOCRadar 真实数据集，可以完成从原始交互清洗、题目文本编码、冷启动划分、模型训练到评估报告的完整实验流程。DyGKT 在本仓库中作为可复现实验基线存在，用来衡量 DySemKT 在语义表达、历史检索、双塔建模和冷启动泛化上的增量贡献。

## 核心创新

- **题目语义驱动的知识追踪**：将题干、选项和概念文本编码为题目语义向量，使模型能够在未见题目上依靠语义相似性和学生历史进行预测，而不是只能记忆训练集中出现过的题目 ID。
- **严格冷启动评估协议**：提供按题目划分的 cold split，验证集和测试集题目在训练阶段未出现，历史标签也限制为训练集可见事件，避免把未来作答或测试题统计泄漏到模型中。
- **双塔非对称动态建模**：学生塔学习长期知识状态，题目塔学习当前题目的重复作答、全局难度和时间衰减，两者通过自适应门控融合，而不是用同一种序列编码器处理所有关系。
- **关系感知注意力偏置**：同题、同练习、概念重叠三类结构关系不只是拼接进特征，而是直接进入 attention logits，显式控制历史事件对当前预测的影响。
- **混合历史检索器**：每次预测同时保留最近历史和结构相关历史，兼顾学生状态的近因变化与题目语义/结构相关性，避免纯最近窗口漏掉关键相似题。
- **语义残差通路**：原始题目语义向量通过残差投影直达预测层，保证题目文本信息不会在历史注意力或门控融合中被完全稀释。
- **可控消融实验设计**：内置 `semantic`、`id`、`hybrid` 三种题目表示模式，以及 `hybrid` / `recent` 两种检索策略，便于量化语义、ID 记忆和结构检索各自的贡献。

## 项目能力

- 读取、校验并按时序排列 MOOCRadar JSON 数据；
- 清理嘈杂文本并构建每道题的标准化答案文本；
- 支持离线哈希文本编码、SentenceTransformer 语义编码，以及 OpenAI 兼容 embeddings API；
- 提供全局时序划分和严格未见题目（cold）两种划分；
- 实现 DySemKT：双塔关系感知注意力（学生侧 + 题目侧）+ 全局特征 + 语义残差；
- 完整的训练/验证/测试/早停/指标报告流程；
- 覆盖数据处理/历史边界/模型前向传播/端到端训练的全套测试。

## 模型概览

```text
                    【输入】当前题 Q_raw (1024维 BGE) + 当前题 ID
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            【混合检索器】                        【全局特征提取器】
    从该生历史中取40条：                    查这道题的全局统计：
    强制最近16条 +                          (平均正确率, 尝试次数,
    结构分数Top24                           平均耗时) → 3维
    (按[同题,同练习,重叠]评分)                          │
                    │                                   │
                    ▼                                   │
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
    4模态门控融合)            + 全局统计3维 + 可学习时间衰减τ)
        │                       │
        └───────────┬───────────┘
                    ▼
        【自适应门控融合】
        student vs question → 2-way softmax gate
                    +
        【残差连接】原始BGE 1024 → Linear → 128
                    ▼
                【预测层】
               P(correct)
```

详细公式、维度说明和设计边界见[模型架构说明](docs/MODEL_ARCHITECTURE.md)。

## 目录结构

```text
KT/
|-- configs/
|   `-- default.json
|-- docs/
|   |-- BLUEPRINT.md
|   |-- BOUNDARIES.md
|   |-- DATA_CONTRACT.md
|   `-- MODEL_ARCHITECTURE.md
|-- src/dysemkt/
|   |-- cli.py
|   |-- dataset.py
|   |-- dygkt_data.py
|   |-- dygkt_engine.py
|   |-- dygkt_model.py
|   |-- engine.py
|   |-- metrics.py
|   |-- model.py
|   |-- preprocess.py
|   `-- text.py
|-- tests/
|-- .python-version
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

原始数据存放于仓库默认忽略的 `data/raw/`，处理结果写入 `data/processed/`，实验结果写入 `outputs/`。这些目录不被提交到 Git。

## 环境安装

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境和依赖。仓库通过 `.python-version` 默认使用 Python 3.13，通过 `uv.lock` 锁定依赖版本，并在 `pyproject.toml` 中配置清华 PyPI 镜像，提高国内安装速度。项目兼容 Python 3.11 至 3.13。

克隆仓库后在仓库目录内安装核心依赖和开发工具即可：

```console
uv sync --frozen
```

如果你系统已经安装 Python 3.13，并希望阻止 uv 自动下载 Python，请使用：

```console
uv sync --frozen --python 3.13 --no-python-downloads
```

运行测试：

```console
uv run --frozen pytest
```

正式运行实验还需要安装 `semantic` 可选依赖：

```console
uv sync --frozen --extra semantic
```

当前项目测试覆盖包括：

- 题目文本清理与标准化答案构建；
- 原始交互数据完整性校验；
- 时序划分和未见题目划分的防泄漏约束；
- 学生和题目历史严格位于当前事件之前；
- 验证集和测试集标签不超过训练历史；
- 多种题目表示模式下的前向和反向传播；
- 小规模数据上可运行的端到端训练、评估和检查点保存。

## 数据预处理

数据链接：https://bhpan.buaa.edu.cn/anyshare/zh-cn/link/AAE5A458AFE444474EBEC928F98986E5B8?_tb=none&expires_at=2030-10-31T23%3A59%3A46%2B08%3A00&item_type=folder&password_required=false&title=MOOCRadar&type=anonymous

下载 MOOCRadar 后，将四个原始文件放置到仓库根目录下：

```text
data/raw/moocradar/
|-- problem.json
|-- student-problem-coarse.json
|-- student-problem-middle.json
`-- student-problem-fine.json
```

后续命令均在仓库根目录运行，不依赖任何用户或系统的具体路径。

## 命令行使用

以下命令覆盖从数据处理到模型训练的全部操作。正式实验建议在 `tmux` 或 `screen` 会话中运行，避免终端断开导致训练中断。

### 数据预处理与语义编码

默认使用 BGE-M3（SentenceTransformer），首次运行会自动下载模型：

```console
uv run --frozen dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_api --encoder sentence-transformer
```

如需使用哈希替代语义编码（仅用于验证数据管线）：

```console
uv run --frozen dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_api_hash --encoder hash
```

如需使用外部 embedding API（OpenAI 兼容格式）：

```console
uv run --frozen dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_api_openai --encoder api
```

### 模型训练

语义特征 + 严格未见题目划分（semantic + cold）：

```console
uv run --frozen dysemkt train     --data-dir data/processed/moocradar_api     --output-dir outputs/semantic_cold     --split cold     --feature-mode semantic     --batch-size 1024
```

纯 ID 特征 + 时序划分（id + temporal）：

```console
uv run --frozen dysemkt train     --data-dir data/processed/moocradar_api     --output-dir outputs/id_temporal     --split temporal     --feature-mode id
```

混合特征 + 未见题目划分（hybrid + cold）：

```console
uv run --frozen dysemkt train     --data-dir data/processed/moocradar_api     --output-dir outputs/hybrid_cold     --split cold     --feature-mode hybrid
```

### 常用命令行参数

```
--encoder hash|sentence-transformer|api  预处理阶段的题目文本编码器
--split cold|temporal         数据划分方式
--feature-mode semantic|id|hybrid   题目特征模式
--batch-size N                训练批次大小（默认 256）
--epochs N                    最大训练轮数
--patience N                  早停耐心值
--learning-rate LR            学习率
--seed SEED                   随机种子
--d-model N                   模型隐藏维度（默认 128）
--history-length N            每侧历史窗口大小（默认 40）
--retrieval hybrid|recent     历史检索策略（默认 hybrid，recent=纯最近N条）
```

## 对比基线

项目保留 DyGKT 模型（`src/dysemkt/dygkt_model.py`）作为对比基线。这样可以在同一份 MOOCRadar 预处理数据、同一套 cold/temporal 划分和相同评估指标下，直接观察 DySemKT 的创新模块是否带来增益：

```console
# DyGKT 冷启动
uv run --frozen dysemkt train --model dygkt --data-dir data/processed/moocradar_api --output-dir outputs/dygkt_cold --split cold --batch-size 1024 --num-neighbors 40

# DyGKT 时序划分
uv run --frozen dysemkt train --model dygkt \
    --data-dir data/processed/moocradar_api \
    --output-dir outputs/dygkt_temporal \
    --split temporal \
    --batch-size 1024
```

DyGKT 与 DySemKT 的方法差异：

| 维度 | DyGKT | DySemKT |
|------|-------|---------|
| 序列编码 | GRU | 关系感知 Attention |
| 历史选择 | 近邻序列 | 最近历史 + 结构相关历史混合检索 |
| 多模态融合 | 加性混合 | 语义/对错/时间/结构四模态独立建模后门控融合 |
| 结构特征 | 普通输入特征 | 结构关系直接修正 attention logits |
| 题目语义 | 不作为主通路 | 题目文本语义作为当前题表示和残差通路 |
| 题目塔 | 对称序列建模 | 自历史 + 全局统计 + Q-K 对齐的非对称题目塔 |
| 冷启动处理 | 依赖历史邻居 | cold split、全局统计 dropout、空题目嵌入和语义迁移 |

## 默认训练参数

| 参数 | DySemKT | DyGKT |
|------|---------|-------|
| d_model / node_dim | 128 | 64 |
| max_history / num_neighbors | 40 | 50 |
| dropout | 0.1 | 0.5 |
| batch_size | 256 | 256 |
| 学习率 | 3e-4 (余弦退火至 3e-6) | 5e-4 |
| 权重衰减 | 1e-4 | 1e-4 |
| 最大轮数 | 30 | 30 |
| Early stopping | ROC-AUC, patience=5 | ROC-AUC, patience=5 |

## 推荐实验矩阵

| 实验 | 模型 | 划分 | 特征模式 | 目的 |
|------|------|------|----------|------|
| DyGKT Cold | dygkt | cold | — | DyGKT 冷启动基线 |
| DyGKT Temporal | dygkt | temporal | — | DyGKT 时序基线 |
| ID Temporal | dysemkt | temporal | id | 纯 ID 基线（无语义泄漏） |
| **ID Cold + Recent** | dysemkt | cold | id | **真正零语义基线**（纯ID+纯最近40条，无结构检索） |
| Semantic Cold | dysemkt | cold | semantic | 未见题目迁移能力 |
| Hybrid Cold | dysemkt | cold | hybrid | 语义+ID 互补验证 |
| Semantic Temporal | dysemkt | temporal | semantic | 时序下语义贡献验证 |

正式报告应运行多个随机种子（`--seed 42 123 456`），在相同数据划分和参数下比较。

纯 ID + 纯最近检索（真正零语义基线）：
```console
uv run --frozen dysemkt train \
    --data-dir data/processed/moocradar_api \
    --output-dir outputs/id_cold_recent \
    --split cold \
    --feature-mode id \
    --retrieval recent
```

## 补充说明

- **Temporal 划分需要谨慎解读**：label 极度不平衡（~75% 正确），时序划分引入的用户行为漂移会同时影响训练和评估。本项目的默认划分是 conservative 的 cold split。
- **语义模型建议固定**：当前管线以 frozen 方式使用语义编码器，微调编码器本身需要额外实验和验证。
- **向项目贡献**：请在独立的 git 分支上提交修改，并确保所有测试通过（`uv run --frozen pytest`）。
