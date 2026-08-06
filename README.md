# DySemKT — 语义增强的动态图知识追踪

DySemKT 是一个用于知识追踪研究的紧凑项目，旨在探索如何通过题目实现两阶段的 DyGKT。该项目继承 DyGKT 的学生-题目-时间动态图思想，同时引入语义理解、防泄漏数据处理流程和双塔动态历史模型。

项目当前版本支持 MOOCRadar 真实数据集。你可以完成端到端的题目语义学习，而不仅是预定义知识点的概率预测——模型学习题目本身的语义，而非依赖预定义知识点来泛化到未见题目。

## 项目亮点

- 读取、校验并按时序排列 MOOCRadar JSON 数据；
- 清理嘈杂文本并构建每道题的标准化答案文本；
- 支持离线哈希文本编码、SentenceTransformer 语义编码，以及 OpenAI 兼容 embeddings API；
- 提供全局时序划分和严格未见题目（cold）两种划分；
- 实现 DySemKT：双塔关系感知注意力（学生侧 + 题目侧）+ 全局特征 + 语义残差；
- 结构化注意力偏置权重：同题/同练习/概念重叠三个维度直接修改 attention logits；
- 提供 `semantic`、`id`、`hybrid` 三种题目表示模式，方便对比实验；
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
uv run --frozen dysemkt preprocess --data-dir data/raw/moocradar --out data/processed/moocradar_api
```

如需使用哈希替代语义编码（仅用于验证数据管线）：

```console
uv run --frozen dysemkt preprocess --data-dir data/raw/moocradar --out data/processed/moocradar_api_hash --encoder hash
```

如需使用外部 embedding API（OpenAI 兼容格式）：

```console
uv run --frozen dysemkt preprocess --data-dir data/raw/moocradar --out data/processed/moocradar_api_openai --encoder openai:your-model-name
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
--encoder / --no-encoder      是否使用/跳过语义编码
--split cold|temporal         数据划分方式
--feature-mode semantic|id|hybrid   题目特征模式
--batch-size N                训练批次大小（默认 256）
--epochs N                    最大训练轮数
--patience N                  早停耐心值
--lr LR                       学习率
--seed SEED                   随机种子
--d-model N                   模型隐藏维度（默认 128）
--max-history N               每侧历史窗口大小（默认 40）
--retrieval hybrid|recent     历史检索策略（默认 hybrid，recent=纯最近N条）
--history-cache PATH          预计算 history cache 路径（可选，加速训练）
```

## DyGKT 基线

项目完整保留了 DyGKT 模型（`src/dysemkt/dygkt_model.py`），可在相同数据管线中直接对比：

```console
# DyGKT 冷启动
uv run --frozen dysemkt train --model dygkt \
    --data-dir data/processed/moocradar_api \
    --output-dir outputs/dygkt_cold \
    --split cold \
    --batch-size 1024

# DyGKT 时序划分
uv run --frozen dysemkt train --model dygkt \
    --data-dir data/processed/moocradar_api \
    --output-dir outputs/dygkt_temporal \
    --split temporal \
    --batch-size 1024
```

DyGKT 与 DySemKT 的核心区别：

| 维度 | DyGKT | DySemKT |
|------|-------|---------|
| 序列编码 | GRU | 单层关系感知 Attention |
| 多模态融合 | add 混合 | 4模态独立 K/V + 门控融合 |
| 结构特征 | 普通加性特征 | 劫持 attention logits（标量偏置） |
| 语义残差 | 无 | BGE 1024→128 直达预测层 |
| 题目塔 | GRU 对称 | 非对称：自历史 + 全局统计 + Q-K对齐 |

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