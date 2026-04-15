#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI命令行接口

命令:
    analyze      执行主题建模分析
    find-topics  搜索最优主题数量
    verify       验证实验结果（复现论文）
    tokenize     中文分词工具

示例:
    python -m topic_model.cli analyze data/corpus.txt -k 5 -o results
    python -m topic_model.cli find-topics data/corpus.txt --min 2 --max 10
    python -m topic_model.cli verify data/corpus.txt --seed 99
"""
import argparse
import logging
import sys
from pathlib import Path

from topic_model.lda_model import LDATopicModel
from topic_model import setup_logging

logger = logging.getLogger(__name__)


def cmd_analyze(args):
    """执行主题建模分析"""
    stopwords = []
    if args.stopwords:
        stopwords = LDATopicModel.load_stopwords(args.stopwords)

    num_topics = args.num_topics if args.num_topics else 10
    auto_find_k = args.auto_k

    model = LDATopicModel(
        num_topics=num_topics,
        passes=args.passes,
        iterations=args.iterations,
        random_state=args.seed,
        custom_stopwords=stopwords,
        ngram_mode=args.ngram,
    )

    output_dir = args.output or "results"

    if auto_find_k:
        # 自动搜索最优主题数
        results = model.run_analysis(
            args.input,
            output_dir=output_dir,
            auto_find_k=True,
            k_min=args.k_min,
            k_max=args.k_max,
            k_step=args.k_step
        )
    else:
        results = model.run_analysis(args.input, output_dir=output_dir)

    if args.save_model:
        model.save_model(args.save_model)

    return 0


def cmd_find_topics(args):
    """寻找最优主题数（通过C_V一致性分数评估）"""
    stopwords = []
    if args.stopwords:
        stopwords = LDATopicModel.load_stopwords(args.stopwords)

    model = LDATopicModel(
        num_topics=2,  # 临时值，后续会遍历
        passes=args.passes,
        custom_stopwords=stopwords
    )

    model.load_corpus(args.input)
    results = model.find_optimal_topics(
        min_topics=args.min_topics,
        max_topics=args.max_topics,
        step=1
    )

    # 输出搜索结果
    print("\n主题数搜索结果")
    print("-" * 30)
    for k, score in results:
        print(f"  k={k:2d}, C_V={score:.4f}")

    best_k = max(results, key=lambda x: x[1])
    print(f"\n最优主题数: {best_k[0]} (C_V={best_k[1]:.4f})")
    return 0


def cmd_verify(args):
    """
    验证实验结果（复现论文）
    
    使用论文验证的最佳配置:
        - num_topics = 14
        - random_state = 99
        - passes = 20
        - ngram_mode = 'auto'
    
    基线分数:
        - baseline: 0.5902 (基线配置)
        - optimized: 0.6245 (优化后平均)
        - best: 0.6505 (最佳单次运行)
    """
    stopwords = []
    if args.stopwords:
        stopwords = LDATopicModel.load_stopwords(args.stopwords)

    # 使用论文验证的配置（可通过参数覆盖）
    num_topics = args.num_topics or 14
    seed = args.seed or 99
    passes = args.passes or 20

    logger.info(f"验证模式: num_topics={num_topics}, seed={seed}, passes={passes}")

    model = LDATopicModel(
        num_topics=num_topics,
        passes=passes,
        iterations=args.iterations,
        random_state=seed,
        custom_stopwords=stopwords,
        ngram_mode='auto',
    )

    output_dir = args.output or "results"
    results = model.run_analysis(args.input, output_dir=output_dir)

    # 与论文基线对比
    coherence = results['model_info']['coherence_score']
    baseline = 0.5902
    optimized = 0.6245

    logger.info(f"验证完成 - C_V得分: {coherence:.4f}")
    logger.info(f"基线得分: {baseline:.4f}")
    logger.info(f"优化后平均得分: {optimized:.4f}")

    if coherence >= baseline:
        logger.info("✅ 得分达到或超过基线水平")
    else:
        logger.warning("⚠️ 得分低于基线，建议检查数据和预处理流程")

    return 0


def cmd_tokenize(args):
    """中文分词工具（支持管道输入）"""
    # 从文件或标准输入读取
    if not args.input or args.input == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(args.input).read_text(encoding='utf-8')
        except FileNotFoundError:
            logger.error(f"文件不存在: {args.input}")
            return 1
        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误: {e}")
            return 1

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
  python -m topic_model.cli analyze data/corpus.txt -k 5 -o results
  python -m topic_model.cli find-topics data/corpus.txt --min 2 --max 10
  python -m topic_model.cli verify data/corpus.txt --seed 99
  python -m topic_model.cli tokenize -s stopwords.txt input.txt
        """
    )

    # 全局参数
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细日志')
    parser.add_argument('-s', '--stopwords', type=str,
                       help='停用词文件路径（每行一个词）')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # analyze 子命令
    analyze_parser = subparsers.add_parser('analyze', help='执行主题建模分析')
    analyze_parser.add_argument('input', help='输入文本文件（每行一篇文档）')
    analyze_parser.add_argument('-k', '--num-topics', type=int, default=None,
                               help='主题数量 (默认: 10)')
    analyze_parser.add_argument('-p', '--passes', type=int, default=15,
                               help='训练轮数 (默认: 15)')
    analyze_parser.add_argument('-i', '--iterations', type=int, default=100,
                               help='每轮迭代次数 (默认: 100)')
    analyze_parser.add_argument('--seed', type=int, default=42,
                               help='随机种子 (默认: 42)')
    analyze_parser.add_argument('-o', '--output', type=str, default='results',
                               help='输出目录 (默认: results)')
    analyze_parser.add_argument('--save-model', type=str,
                               help='保存模型到指定目录')
    analyze_parser.add_argument('--auto-k', action='store_true',
                               help='自动搜索最优主题数')
    analyze_parser.add_argument('--k-min', type=int, default=5,
                               help='最小主题数 (默认: 5)')
    analyze_parser.add_argument('--k-max', type=int, default=20,
                               help='最大主题数 (默认: 20)')
    analyze_parser.add_argument('--k-step', type=int, default=1,
                               help='搜索步长 (默认: 1)')
    analyze_parser.add_argument('--ngram', type=str, default='auto',
                               choices=['none', 'auto', 'strict'],
                               help='N-gram 模式 (默认: auto)')

    # find-topics 子命令
    ft_parser = subparsers.add_parser('find-topics', help='寻找最优主题数')
    ft_parser.add_argument('input', help='输入文本文件')
    ft_parser.add_argument('--min', type=int, default=2, dest='min_topics',
                          help='最小主题数 (默认: 2)')
    ft_parser.add_argument('--max', type=int, default=10, dest='max_topics',
                          help='最大主题数 (默认: 10)')
    ft_parser.add_argument('-p', '--passes', type=int, default=15,
                          help='训练轮数 (默认: 15)')

    # verify 子命令
    verify_parser = subparsers.add_parser('verify', help='验证实验结果')
    verify_parser.add_argument('input', help='输入文本文件')
    verify_parser.add_argument('-k', '--num-topics', type=int, default=14,
                              help='主题数量 (默认: 14)')
    verify_parser.add_argument('-p', '--passes', type=int, default=20,
                              help='训练轮数 (默认: 20)')
    verify_parser.add_argument('-i', '--iterations', type=int, default=100,
                              help='每轮迭代次数 (默认: 100)')
    verify_parser.add_argument('--seed', type=int, default=99,
                              help='随机种子 (默认: 99)')
    verify_parser.add_argument('-o', '--output', type=str, default='results',
                              help='输出目录 (默认: results)')

    # tokenize 子命令
    tok_parser = subparsers.add_parser('tokenize', help='中文分词')
    tok_parser.add_argument('input', nargs='?', default='-',
                           help='输入文件（默认读取标准输入）')
    tok_parser.add_argument('-d', '--delimiter', type=str, default=' ',
                           help='输出分隔符 (默认: 空格)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 配置日志级别
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)

    # 命令路由
    commands = {
        'analyze': cmd_analyze,
        'find-topics': cmd_find_topics,
        'verify': cmd_verify,
        'tokenize': cmd_tokenize,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
