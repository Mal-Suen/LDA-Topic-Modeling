#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行脚本 - 演示 LDA 主题建模完整流程

输出:
    - results/lda_visualization.html: 交互式可视化
    - results/report.json: 分析报告
    - results/classifications.csv: 文档分类结果
    - results/model/: 保存的模型文件
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from topic_model.lda_model import LDATopicModel
from topic_model import setup_logging

setup_logging(level='INFO')


def main():
    # 加载停用词表
    stopwords_file = Path(__file__).parent / "data" / "stopwords.txt"
    stopwords = LDATopicModel.load_stopwords(str(stopwords_file))

    # 初始化模型 - 示例数据较小，使用3个主题
    model = LDATopicModel(
        num_topics=3,
        passes=15,
        iterations=100,
        random_state=42,
        custom_stopwords=stopwords
    )

    # 运行完整分析流程
    corpus_file = Path(__file__).parent / "data" / "sample_corpus.txt"
    output_dir = Path(__file__).parent / "results"
    results = model.run_analysis(str(corpus_file), output_dir=str(output_dir))

    # 保存模型供后续使用
    model.save_model(str(output_dir / "model"))

    print("\n运行完成！输出文件:")
    print(f"  可视化: {output_dir / 'lda_visualization.html'}")
    print(f"  模型:   {output_dir / 'model'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
