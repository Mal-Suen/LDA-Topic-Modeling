#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行脚本 - 直接执行即可看到分析结果
"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from topic_model.lda_model import LDATopicModel
from topic_model import setup_logging

# 使用统一的日志配置
setup_logging(level='INFO')


def main():
    # 加载停用词
    stopwords_file = Path(__file__).parent / "data" / "stopwords.txt"
    stopwords = LDATopicModel.load_stopwords(str(stopwords_file))

    # 初始化模型
    model = LDATopicModel(
        num_topics=3,
        passes=15,
        iterations=100,
        random_state=42,
        custom_stopwords=stopwords
    )

    # 运行分析
    corpus_file = Path(__file__).parent / "data" / "sample_corpus.txt"
    output_dir = Path(__file__).parent / "results"
    
    results = model.run_analysis(str(corpus_file), output_dir=str(output_dir))

    # 保存模型
    model.save_model(str(output_dir / "model"))
    
    print("\n运行完成！输出文件:")
    print(f"  可视化: {output_dir / 'lda_visualization.html'}")
    print(f"  模型:   {output_dir / 'model'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
