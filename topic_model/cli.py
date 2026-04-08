#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI命令行入口
"""

import argparse
import logging
import sys
from pathlib import Path

from topic_model.lda_model import LDATopicModel

logger = logging.getLogger(__name__)


def cmd_analyze(args):
    """执行主题建模分析"""
    # 加载停用词
    stopwords = []
    if args.stopwords:
        stopwords = LDATopicModel.load_stopwords(args.stopwords)

    # 初始化模型
    model = LDATopicModel(
        num_topics=args.num_topics,
        passes=args.passes,
        iterations=args.iterations,
        random_state=args.seed,
        custom_stopwords=stopwords
    )

    # 运行分析
    output_dir = args.output or "output"
    results = model.run_analysis(args.input, output_dir=output_dir)
    
    # 保存模型（可选）
    if args.save_model:
        model.save_model(args.save_model)
    
    return 0


def cmd_find_topics(args):
    """寻找最优主题数"""
    stopwords = []
    if args.stopwords:
        stopwords = LDATopicModel.load_stopwords(args.stopwords)

    model = LDATopicModel(
        num_topics=2,
        passes=args.passes,
        custom_stopwords=stopwords
    )

    model.load_corpus(args.input)
    topic_range = range(args.min_topics, args.max_topics + 1)
    results = model.find_optimal_topics(topic_range)

    print("\n主题数搜索结果")
    print("-" * 30)
    for k, score in results:
        print(f"  k={k:2d}, C_V={score:.4f}")
    
    best_k = max(results, key=lambda x: x[1])
    print(f"\n最优主题数: {best_k[0]} (C_V={best_k[1]:.4f})")
    return 0


def cmd_tokenize(args):
    """中文分词工具"""
    if not args.input or args.input == "-":
        # 从标准输入读取
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding='utf-8')

    stopwords = []
    if args.stopwords:
        stopwords = LDATopicModel.load_stopwords(args.stopwords)

    model = LDATopicModel(custom_stopwords=stopwords)
    words = model.tokenize(text)
    
    if args.delimiter:
        print(args.delimiter.join(words))
    else:
        print(" ".join(words))
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LDA主题建模工具 - 基于gensim的中文主题分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 执行主题建模分析
  python -m topic_model.cli analyze data/corpus.txt -k 5 -o output

  # 寻找最优主题数
  python -m topic_model.cli find-topics data/corpus.txt --min 2 --max 10

  # 中文分词
  python -m topic_model.cli tokenize -s stopwords.txt input.txt
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细日志')
    parser.add_argument('-s', '--stopwords', type=str,
                       help='停用词文件路径（每行一个词）')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='执行主题建模分析')
    analyze_parser.add_argument('input', help='输入文本文件（每行一篇文档）')
    analyze_parser.add_argument('-k', '--num-topics', type=int, default=3,
                               help='主题数量 (默认: 3)')
    analyze_parser.add_argument('-p', '--passes', type=int, default=15,
                               help='训练轮数 (默认: 15)')
    analyze_parser.add_argument('-i', '--iterations', type=int, default=100,
                               help='每轮迭代次数 (默认: 100)')
    analyze_parser.add_argument('--seed', type=int, default=42,
                               help='随机种子 (默认: 42)')
    analyze_parser.add_argument('-o', '--output', type=str, default='output',
                               help='输出目录 (默认: output)')
    analyze_parser.add_argument('--save-model', type=str,
                               help='保存模型到指定目录')

    # find-topics 命令
    ft_parser = subparsers.add_parser('find-topics', help='寻找最优主题数')
    ft_parser.add_argument('input', help='输入文本文件')
    ft_parser.add_argument('--min', type=int, default=2, dest='min_topics',
                          help='最小主题数 (默认: 2)')
    ft_parser.add_argument('--max', type=int, default=10, dest='max_topics',
                          help='最大主题数 (默认: 10)')
    ft_parser.add_argument('-p', '--passes', type=int, default=15,
                          help='训练轮数 (默认: 15)')

    # tokenize 命令
    tok_parser = subparsers.add_parser('tokenize', help='中文分词')
    tok_parser.add_argument('input', nargs='?', default='-',
                           help='输入文件（默认读取标准输入）')
    tok_parser.add_argument('-d', '--delimiter', type=str, default=' ',
                           help='输出分隔符 (默认: 空格)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 路由到对应命令
    commands = {
        'analyze': cmd_analyze,
        'find-topics': cmd_find_topics,
        'tokenize': cmd_tokenize,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
