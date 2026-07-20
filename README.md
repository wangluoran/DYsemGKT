# DySemKT：语义增强的动态图知识追踪

DySemKT 是一个面向知识追踪研究的精简项目，用于探索如何通过真实题目语义改进 DyGKT。项目保留 DyGKT 的学生-题目连续时间动态图思想，同时重新设计了明确的数据契约、无泄漏的数据处理流程和双侧动态历史模型。

项目当前首先支持 MOOCRadar。该数据集包含真实题干、选项和题目概念，因此模型学习的是题目内容语义，而不是把少量知识点名称复制给大量题目。

## 项目功能

- 读取、校验并按时间重新排序 MOOCRadar JSON 数据；
- 构建可审计且不包含标准答案的题目文本；
- 支持离线哈希文本特征和 SentenceTransformer 预训练语义特征；
- 提供全局时间划分和严格未见题目冷启动划分；
- 实现 DySemKT：学生历史编码器、题目历史编码器和语义门控融合；
- 显式建模同题重复、同练习单元、概念重叠、上次同题结果和重复间隔；
- 提供 `semantic`、`id`、`hybrid` 三种题目表示模式用于消融实验；
- 包含训练、验证、测试、早停、检查点和指标输出流程；
- 包含数据处理、历史边界、模型反向传播和端到端训练测试。

## 模型概览

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

详细公式、张量维度和设计边界见[模型架构说明](docs/MODEL_ARCHITECTURE.md)。

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

原始数据保持只读并默认放在 `data/raw/`。派生数据写入 `data/processed/`，实验输出写入 `outputs/`。这些目录均不会提交到 Git。

## 环境安装

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python、虚拟环境和锁定依赖。仓库通过 `.python-version` 默认使用 Python 3.13，通过 `uv.lock` 固定完整依赖版本，并在 `pyproject.toml` 中配置清华 PyPI 镜像以提高国内安装速度。项目兼容 Python 3.11 至 3.13。

克隆仓库后，在仓库根目录安装锁定的核心依赖和开发依赖：

```console
uv sync --frozen
```

如果本机已经安装 Python 3.13，并希望禁止 uv 下载 Python，可使用：

```console
uv sync --frozen --python 3.13 --no-python-downloads
```

运行测试：

```console
uv run --frozen pytest
```

正式语义实验还需要安装 `semantic` 可选依赖：

```console
uv sync --frozen --extra semantic
```

当前项目测试包括：

- 题目文本不包含标准答案；
- 原始逆序交互被正确排序；
- 时间划分和冷启动题目集合符合约束；
- 学生侧与题目侧历史严格早于当前事件；
- 验证集和测试集标签不会进入允许历史；
- 三种题目表示模式均可完成前向与反向传播；
- 微型数据可以完成端到端训练、评估和检查点保存。

## 数据预处理

下载 MOOCRadar 后，将四个原始文件放入以下仓库相对目录：

```text
data/raw/moocradar/
|-- problem.json
|-- student-problem-coarse.json
|-- student-problem-middle.json
`-- student-problem-fine.json
```

所有后续命令均从仓库根目录运行，不依赖任何用户或操作系统的绝对路径。

## 完整运行流程

以下命令覆盖环境安装、测试、正式 embedding 生成、数据检查和模型训练。第一次执行 BGE-M3 预处理时会下载模型，后续运行复用本地缓存。

```console
# 1. 安装锁定依赖，包括 SentenceTransformer
uv sync --frozen --extra semantic

# 2. 运行项目测试
uv run --frozen pytest

# 3. 生成正式题目 embedding 和处理后交互数据
uv run --frozen --extra semantic dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_bge --encoder sentence-transformer --model-name BAAI/bge-m3 --batch-size 32 --device cuda

# 4. 检查处理结果
uv run --frozen dysemkt inspect --data-dir data/processed/moocradar_bge

# 5. 普通时间划分训练
uv run --frozen dysemkt train --data-dir data/processed/moocradar_bge --output-dir outputs/hybrid_temporal --split temporal --feature-mode hybrid

# 6. 严格未见题目冷启动训练
uv run --frozen dysemkt train --data-dir data/processed/moocradar_bge --output-dir outputs/semantic_cold --split cold --feature-mode semantic
```

没有 CUDA 时，删除预处理命令末尾的 `--device cuda`，模型训练会自动回退到 CPU。BGE-M3 在 CPU 上能够运行，但速度会明显更慢。

### 1. 离线流程验证

哈希编码器不需要下载模型，适合验证数据处理和训练流程：

```console
uv run --frozen dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_hash --encoder hash --embedding-dim 256
```

哈希编码器只是确定性的字符 n-gram 特征，不能用于支持正式的题目语义研究结论。

### 2. 正式语义实验：自动下载模型后本地计算

使用多语言 SentenceTransformer，例如 BGE-M3：

```console
uv run --frozen --extra semantic dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_bge --encoder sentence-transformer --model-name BAAI/bge-m3 --batch-size 32
```

这里的 `BAAI/bge-m3` 是 Hugging Face 模型标识，不是远程 embedding API。执行过程如下：

```text
题干、选项、概念
        |
首次运行时下载 BAAI/bge-m3 到 Hugging Face 本地缓存
        |
SentenceTransformer 在本机 CPU/GPU 上批量推理
        |
生成 question_features.npy
        |
训练阶段直接读取该文件，不再加载或调用文本模型
```

可以通过 `--device cuda` 指定 GPU 编码：

```console
uv run --frozen --extra semantic dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_bge --encoder sentence-transformer --model-name BAAI/bge-m3 --batch-size 32 --device cuda
```

### 3. 使用本地模型目录

如果已经下载并固定了 SentenceTransformer 模型快照，可以放入被 Git 忽略的 `data/models/`：

```text
data/models/bge-m3/
|-- config.json
|-- modules.json
|-- model.safetensors
`-- ...
```

将本地目录直接传给 `--model-name`：

```console
uv run --frozen --extra semantic dysemkt preprocess --raw-dir data/raw/moocradar --output-dir data/processed/moocradar_bge --encoder sentence-transformer --model-name data/models/bge-m3 --batch-size 32 --device cuda
```

本地目录模式不会通过 embedding API 发送题目数据。为了严格复现实验，建议记录模型仓库、具体 revision 或 commit，并保留处理目录中的 `metadata.json`。

### 4. API embedding 支持边界

当前版本不支持 OpenAI、SiliconFlow 或其他远程 embedding API。正式编码只有 SentenceTransformer 本地推理路径，原因是：

- 题目文本不会离开本机；
- 不依赖 API Key、配额或服务可用性；
- 批量实验成本和结果版本更容易控制；
- 生成一次 embedding 后，所有 KT 消融实验复用同一特征文件。

代码中的 `TextEncoder` 是可扩展接口，后续可以增加 API 实现，但在实现请求重试、批次恢复、响应维度校验、模型版本记录和本地缓存前，不应把 API 路径用于正式实验。

### 5. Embedding 如何进入模型

embedding 只在预处理阶段生成一次，数据流为：

```text
TextEncoder.encode(题目文本列表)
        |
question_features.npy，形状为 (题目数, 文本向量维度)
        |
ProcessedData.question_features
        |
DySemKT(question_features=...)
        |
冻结原始文本向量 + 可训练线性投影
        |
256 维题目语义表示
```

原始 embedding 在 KT 训练阶段注册为固定 buffer，不被梯度更新；可训练的是从原始文本维度到模型隐藏维度的投影层。这样不同 KT 消融实验使用完全相同的基础文本特征。

题目编码文本包括：

- 题目组标题；
- 真实题干；
- 选项文本；
- 题目概念。

标准答案不会进入文本编码。损坏题目、无效标签和过短学生序列会被记录在 `metadata.json`，而不是静默忽略。

## 检查处理结果

```console
uv run --frozen dysemkt inspect --data-dir data/processed/moocradar_bge
```

处理目录包含：

```text
events.npz
question_features.npy
question_text.jsonl
question_context.json
mappings.json
metadata.json
```

其中 `metadata.json` 记录数据规模、划分数量、文本编码器、随机种子、拒绝记录和原始文件 SHA-256 指纹。

## 模型训练

### 普通时间预测

默认使用语义与题目 ID 的混合表示：

```console
uv run --frozen dysemkt train --data-dir data/processed/moocradar_bge --output-dir outputs/hybrid_temporal --split temporal --feature-mode hybrid
```

### 严格未见题目冷启动

冷启动实验主要使用纯语义模式，避免未训练的题目 ID embedding 干扰结果：

```console
uv run --frozen dysemkt train --data-dir data/processed/moocradar_bge --output-dir outputs/semantic_cold --split cold --feature-mode semantic
```

### ID 对照实验

```console
uv run --frozen dysemkt train --data-dir data/processed/moocradar_bge --output-dir outputs/id_temporal --split temporal --feature-mode id
```

训练默认自动使用可用的 CUDA 设备，否则使用 CPU。可显式指定：

```console
uv run --frozen dysemkt train --data-dir data/processed/moocradar_bge --output-dir outputs/run --device cuda
```

## 默认训练参数

| 参数 | 默认值 |
|---|---:|
| 隐藏维度 | 256 |
| 学生侧 Transformer 层数 | 2 |
| 题目侧 Transformer 层数 | 1 |
| 注意力头数 | 4 |
| 双侧历史长度 | 50 |
| Dropout | 0.1 |
| Batch size | 256 |
| 学习率 | 0.0005 |
| 权重衰减 | 0.0001 |
| 最大训练轮数 | 30 |
| Early stopping patience | 5 |

完整默认配置见 [configs/default.json](configs/default.json)。

## 训练输出

每个实验目录包含：

```text
config.json
metrics.json
best.pt
```

`metrics.json` 报告：

- ROC-AUC；
- Average Precision；
- Log Loss；
- Accuracy；
- 每轮训练损失和验证指标；
- 训练、验证和测试样本数。

MOOCRadar 的正确率约为 81.55%，类别存在明显不平衡，因此不能只根据 Accuracy 判断模型效果。

## 数据泄漏约束

默认评估只允许训练集交互作为学生侧和题目侧的标签历史：

```python
allowed_history = train_mask
```

因此验证集和测试集答案不会通过动态图历史进入后续预测。该设置比真实在线学习更保守，但更适合作为第一阶段研究基线。

此外：

- 文本输入不包含标准答案；
- 当前标签不参与当前事件编码；
- 当前事件只能访问索引更小的历史事件；
- 同题重复次数、上次结果和重复间隔只从允许历史计算；
- 不使用基于全数据标签计算的难度或区分度；
- 冷启动划分中的题目集合在训练、验证和测试之间严格互斥。

## 推荐实验矩阵

| 实验 | 划分 | 特征模式 | 目的 |
|---|---|---|---|
| ID Temporal | 时间 | `id` | 传统题目 ID 基线 |
| Semantic Temporal | 时间 | `semantic` | 验证真实语义贡献 |
| Hybrid Temporal | 时间 | `hybrid` | 验证语义与 ID 互补性 |
| Semantic Cold | 题目互斥 | `semantic` | 验证未见题目迁移能力 |

正式报告应运行多个随机种子，并在相同数据划分、历史长度和训练参数下比较三种表示模式。

## 项目文档

- [项目边界](docs/BOUNDARIES.md)
- [架构蓝图](docs/BLUEPRINT.md)
- [详细模型架构](docs/MODEL_ARCHITECTURE.md)
- [处理后数据契约](docs/DATA_CONTRACT.md)

## 当前边界

当前版本聚焦于验证真实题目语义和双侧时间历史，不包括所有原始 DyGKT 基线，也暂未将课程层级、练习层级、认知维度或概念图作为独立可训练模块。

本项目能够支持的研究问题是：

> 在固定的连续时间知识追踪框架下，真实题目文本语义能否改善普通预测和未见题目迁移？

它不能直接支持因果学习效果、教学策略有效性或生产推荐效果等结论。
