# LDA-Topic-Modeling | LDA中文主题建模工具

<div align="center">

**Language / 语言选择:**
[English](#english) | [中文](#中文)

---
</div>

<a id="english"></a>

# LDA-Topic-Modeling

A lightweight and efficient Python library for Latent Dirichlet Allocation (LDA) topic modeling on Chinese text. This project employs **adaptive N-gram profiles** and **multi-coefficient voting** to solve common noise issues in large-scale corpora.

## Features

- **Adaptive N-gram Profiles**: Offers three modes (`none`, `auto`, `strict`) to balance between speed and terminology precision.
- **Optimized Bigram Detection**: Prevents "Bigram Explosion" noise by strictly filtering compound words.
- **High Coherence**: Achieves $C_V > 0.63$ on standard news corpora (THUCNews).
- **Full Pipeline Support**: From segmentation to model training, coherence evaluation, and report export (JSON/CSV).
- **CLI Interface**: Simple command-line usage for batch processing.

## Installation

```bash
git clone https://github.com/Mal-Suen/LDA-Topic-Modeling.git
cd LDA-Topic-Modeling
pip install -r requirements.txt
```

**Dependencies:**
- **Operating Systems**: Linux, macOS, Windows
- **Python**: >= 3.8
- **Core Libraries**: `gensim` (4.x), `jieba`, `pyLDAvis`

## Usage

### 1. Command Line Interface (CLI)

The tool supports multiple N-gram strategies to handle different corpus types.

```bash
# General News (Fastest, No Bigram)
python -m topic_model.cli analyze data/news.txt --ngram none -k 10 -o output

# Mixed Reports (Balanced, Recommended)
python -m topic_model.cli analyze data/reports.txt --ngram auto -k 10 -o output

# Specialized Domain (Strict, e.g., Medical/Legal)
python -m topic_model.cli analyze data/medical.txt --ngram strict -k 10 -o output

# Auto-find optimal K
python -m topic_model.cli analyze data/news.txt --ngram none --auto-k --k-min 5 --k-max 20
```

### 2. Python API

```python
from topic_model.lda_model import LDATopicModel

# Initialize with 'auto' ngram mode
model = LDATopicModel(num_topics=10, ngram_mode='auto')

# Run analysis
report = model.run_analysis('data/news.txt', output_dir='output')
print(f"Coherence Score: {report['model_info']['coherence_score']}")
```

## Algorithm Principle

This project addresses the "Bigram Explosion" problem common in LDA implementations. By implementing a threshold-based Bigram filter and a post-processing stopword cleaner, we ensure that topics remain distinct and interpretable.

See `docs/OPTIMIZATION_LOG.md` for a detailed discussion on algorithm optimization and the relevance of LDA in the LLM era.

<a id="中文"></a>

# LDA-Topic-Modeling

一个轻量级、高效的 Python LDA 主题建模工具，专为中文文本优化。本项目通过**自适应 N-gram 预设模式**和**优化的词组检测算法**，解决了大规模语料中常见的噪声干扰问题。

## 特性

- **自适应 N-gram 模式**：提供 `none`, `auto`, `strict` 三种模式，平衡速度与专业术语识别精度。
- **抗噪优化**：解决了传统 LDA 中 Bigram 检测导致的“词汇爆炸”和主题模糊问题。
- **高一致性得分**：在 THUCNews 标准新闻语料上，$C_V$ 一致性得分稳定在 0.63 以上。
- **完整工具链**：支持分词、训练、评估、可视化（pyLDAvis）及报告导出（JSON/CSV）。
- **命令行支持**：提供简单易用的 CLI 接口，适合批量处理。

## 安装

```bash
git clone https://github.com/Mal-Suen/LDA-Topic-Modeling.git
cd LDA-Topic-Modeling
pip install -r requirements.txt
```

**运行环境**:
- **操作系统**: Linux, macOS, Windows
- **Python**: >= 3.8
- **核心依赖**: `gensim` (4.x), `jieba`, `pyLDAvis`

## 用法

### 1. 命令行 (CLI)

工具针对不同类型的语料提供了几种预设模式：

```bash
# 通用新闻（推荐模式，最快且无噪声）
python -m topic_model.cli analyze data/news.txt --ngram none -k 10 -o output

# 混合文档（平衡模式，适合大多数场景）
python -m topic_model.cli analyze data/reports.txt --ngram auto -k 10 -o output

# 专业领域（严格模式，适合医疗、法律等术语密集场景）
python -m topic_model.cli analyze data/medical.txt --ngram strict -k 10 -o output

# 自动搜索最优主题数 K
python -m topic_model.cli analyze data/news.txt --ngram none --auto-k --k-min 5 --k-max 20
```

### 2. Python API

```python
from topic_model.lda_model import LDATopicModel

# 初始化模型，使用自动 N-gram 模式
model = LDATopicModel(num_topics=10, ngram_mode='auto')

# 执行分析
report = model.run_analysis('data/news.txt', output_dir='output')
print(f"一致性得分: {report['model_info']['coherence_score']}")
```

## 算法原理与优化

本项目重点解决了传统 LDA 实现中常见的 **Bigram 爆炸**问题。

通过引入**阈值过滤**和**二次停用词清洗**，我们确保了复合词组（如“人工智能”）能被正确识别，同时过滤掉无意义的噪声组合（如“谢谢_专家”）。

有关详细的算法优化过程以及在 LLM（大语言模型）普及的今天为何仍然需要 LDA 技术，请参阅项目文档：
👉 `docs/OPTIMIZATION_LOG.md`
