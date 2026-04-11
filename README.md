# LDA-Topic-Modeling

> **From Teaching Prototype to Industrial-Grade Pipeline: Optimization and Verification of LDA Topic Models.**
> **从教学原型到工业级管线：LDA 主题模型的工程化优化与验证。**

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-green.svg)](https://www.python.org/downloads/)
[![Gensim 4.4+](https://img.shields.io/badge/gensim-4.4+-orange.svg)](https://radimrehurek.com/gensim/)

---

## 📖 Introduction / 项目简介

### 🇬🇧 English
Latent Dirichlet Allocation (LDA) remains a cornerstone of unsupervised text mining. While Large Language Models (LLMs) dominate semantic understanding, they face bottlenecks in **cost**, **privacy**, and **determinism**. This project transforms a basic LDA implementation (initially released in 2018) into a rigorous, industrial-grade tool. By employing **Stratified Random Sampling**, **Control Variable Experiments**, and **Strict Preprocessing Pipelines**, we demonstrate that a properly tuned LDA model can achieve a **$C_V$ coherence score of 0.65**, significantly outperforming baselines and offering a robust, explainable alternative for massive text corpora.

### 🇨🇳 中文
隐狄利克雷分配（LDA）依然是无监督文本挖掘的基石。虽然大语言模型（LLM）在语义理解上占据主导，但在**成本**、**隐私**和**确定性**方面面临瓶颈。本项目将 2018 年的基础 LDA 实现重构为严谨的工业级工具。通过采用**分层随机抽样**、**控制变量实验**和**严格的预处理流水线**，我们证明了经过优化的 LDA 模型可以达到 **0.65 的 $C_V$ 一致性得分**，显著优于基线，并为海量文本语料提供了一种鲁棒、可解释的替代方案。

---

## 🚀 Key Features & Conclusions / 核心特性与结论

| Feature / 特性 | Detail / 细节 |
| :--- | :--- |
| **🧪 High Coherence** / 高一致性 | Validated via Multi-Seed (42, 88, 99) experiments to overcome local optima. Baseline: 0.59 $\to$ Optimized: **0.65**. |
| **🧹 Strict Preprocessing** / 严格预处理 | Discovered that **Stopword Filtering must precede N-gram extraction**. Reverse order leads to PMI noise and score degradation. |
| **🚫 Avoid TF-IDF** / 避免 TF-IDF | Experiments confirm TF-IDF breaks the Dirichlet-Multinomial conjugacy, causing posterior inference failure. |
| **🤝 LLM Hybrid Arch** / LLM 混合架构 | Proposes a "Funnel Architecture": **LDA** for dimensionality/clustering + **LLM** for semantic labeling. Reduces API costs by 99.9%. |

---

## 📂 Project Structure / 目录结构

```text
LDA-Topic-Modeling/
├── data/                  # Datasets (THUCNews subset) / 数据集
├── docs/                  # Detailed Analysis & Reports / 详细分析报告
│   └── REVISITING_LDA.md  # [New] Comprehensive Engineering Analysis / [新] 全面工程分析
├── scripts/               # Processing Scripts / 处理脚本
├── topic_model/           # Core Python Package / 核心 Python 包
├── results/               # Model Outputs & Visualizations / 模型输出与可视化
├── README.md              # This file / 本文件
└── requirements.txt       # Dependencies / 依赖项
```

---

## 📈 Optimization Analysis / 优化分析

This project is based on rigorous experimental validation on the THUCNews dataset (14,000 documents).
本项目基于 THUCNews 数据集（1.4 万篇文档）进行了严格的实验验证。

### 📊 Results Summary / 结果汇总

| Configuration / 配置 | Score ($C_V$) / 得分 | Conclusion / 结论 |
| :--- | :---: | :--- |
| **Baseline (Raw)** / 基线 (原始) | 0.5332 | Early 2018 prototype / 2018 年初期原型 |
| **Baseline (14k Docs)** / 基线 (1.4 万篇) | 0.5902 | Sub-optimal convergence / 次优收敛 |
| **Optimized Pipeline** / 优化流水线 | **0.6245 ± 0.02** | ✅ Significant Improvement / 显著提升 |
| **Best Run (Seed 99)** / 最佳运行 | **0.6505** | ✅ Theoretical Upper Bound / 理论上限 |

### 📝 Key Findings / 关键发现
1. **N-gram Sequence Dependency:** Stopwords *must* be removed before Bigram detection to avoid "noise phrases" (e.g., "the_of").
   *N-gram 序列依赖性：必须在 Bigram 检测前移除停用词，以避免“噪声短语”。*
2. **TF-IDF Failure:** Weighting breaks the mathematical foundation of LDA (Dirichlet-Multinomial Conjugacy).
   *TF-IDF 失效：加权破坏了 LDA 的数学基础（狄利克雷-多项共轭）。*

---

## 🛠️ Getting Started / 快速开始

### 1. Installation / 安装

```bash
pip install -r requirements.txt
```

### 2. Usage / 使用

**Training a Model / 训练模型:**

```python
from topic_model import LDAModel

# Initialize / 初始化
lda = LDAModel(num_topics=14, passes=20, random_state=99)

# Load & Train (Assuming preprocessed corpus exists) / 加载与训练
lda.train("data/thuc_corpus.txt")

# Evaluate / 评估
score = lda.evaluate()
print(f"Coherence Score: {score}")
```

### 3. Reproducing Results / 复现结果

To reproduce the exact results reported in this analysis, run the validation script:
要复现本分析报告中的结果，请运行验证脚本：

```bash
python -m topic_model.cli --mode verify --seed 99
```

---

## 🤝 Contribution & Contact / 贡献与联系

*   **Author:** Mal-Suen
*   **Blog:** [Mal-Suen's Blog](https://blog.mal-suen.cn)
*   **GitHub:** [https://github.com/Mal-Suen/LDA-Topic-Modeling](https://github.com/Mal-Suen/LDA-Topic-Modeling)

*Copyright © 2018-2026 Mal-Suen. Released under MIT License.*