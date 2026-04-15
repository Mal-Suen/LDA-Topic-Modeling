#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THUCNews 清华新闻语料库处理脚本

功能:
    - 从THUCNews数据集提取文本
    - 执行中文分词和停用词过滤
    - 支持按类别均衡采样

使用前需先申请数据集: http://thuctc.thunlp.org/message

THUCNews目录结构:
    THUCNews/
    ├── 体育/
    │   ├── 0001.txt
    │   └── ...
    ├── 财经/
    ├── 娱乐/
    └── ...

用法:
    python data/process_thucnews.py /path/to/THUCNews -o data/thuc_corpus.txt
    python data/process_thucnews.py --no-limit  # 处理全部文档
"""
import os
import logging
import jieba
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 默认路径配置（脚本在 data/ 目录下）
DEFAULT_THU_DIR = Path(__file__).parent / "THUCNews"
OUTPUT_FILE = Path(__file__).parent / "thuc_corpus.txt"
STOPWORDS_FILE = Path(__file__).parent / "stopwords.txt"


def load_stopwords(file_path: Optional[str] = None) -> set:
    """加载停用词表"""
    stopwords = set()
    path = Path(file_path) if file_path else STOPWORDS_FILE
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            stopwords = {line.strip() for line in f if line.strip()}
        logging.info(f"加载停用词: {len(stopwords)} 个")
    return stopwords


def process_thucnews(
    thuc_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    max_docs_per_class: int = 500,
    max_total_docs: int = 10000
) -> dict:
    """
    处理THUCNews语料库，转换为每行一篇文档的格式
    
    Args:
        thuc_dir: THUCNews解压目录
        output_file: 输出文件路径
        max_docs_per_class: 每个类别最大文档数（均衡采样）
        max_total_docs: 总文档数上限
    
    Returns:
        处理统计信息
    """
    thuc_path = Path(thuc_dir) if thuc_dir else DEFAULT_THU_DIR
    out_path = Path(output_file) if output_file else OUTPUT_FILE

    if not thuc_path.exists():
        logging.error(f"THUCNews 目录不存在: {thuc_path}")
        logging.info("请先申请并下载 THUCNews 数据集")
        logging.info("申请地址: http://thuctc.thunlp.org/message")
        return {"status": "error", "message": "目录不存在"}

    logging.info("=" * 50)
    logging.info("THUCNews 语料库处理工具")
    logging.info("=" * 50)
    logging.info(f"输入目录: {thuc_path}")
    logging.info(f"输出文件: {out_path}")
    logging.info(f"每类上限: {max_docs_per_class} 篇")
    logging.info(f"总量上限: {max_total_docs} 篇")

    stopwords = load_stopwords()

    # 获取所有类别目录
    categories = sorted([
        d.name for d in thuc_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not categories:
        logging.error(f"未找到任何类别目录: {thuc_path}")
        return {"status": "error", "message": "无类别目录"}

    logging.info(f"发现 {len(categories)} 个类别: {', '.join(categories)}")

    docs = []
    doc_count = 0
    stats = {"total": 0, "by_class": {}}

    for cat in categories:
        cat_dir = thuc_path / cat
        if not cat_dir.exists():
            continue

        cat_files = sorted(cat_dir.glob("*.txt"))
        cat_docs = 0

        logging.info(f"\n处理类别: {cat} ({len(cat_files)} 个文件)")

        for txt_file in cat_files:
            # 检查上限
            if doc_count >= max_total_docs:
                logging.info(f"达到总量上限 ({max_total_docs})，停止处理")
                break
            if cat_docs >= max_docs_per_class:
                logging.info(f"  类别上限 ({max_docs_per_class})，跳过剩余文件")
                break

            try:
                content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                if not content:
                    continue

                # 分词并过滤
                words = jieba.lcut(content)
                filtered = [
                    w for w in words
                    if w not in stopwords
                    and len(w) > 1
                    and not w.isdigit()
                    and not all(c.isspace() for c in w)
                ]

                if filtered:
                    docs.append(" ".join(filtered))
                    cat_docs += 1
                    doc_count += 1

            except Exception as e:
                logging.warning(f"处理 {txt_file.name} 失败: {e}")

        stats["by_class"][cat] = cat_docs
        stats["total"] += cat_docs

        if doc_count >= max_total_docs:
            break

    # 保存结果
    if docs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(docs))

        file_size = out_path.stat().st_size / (1024 * 1024)

        logging.info("\n" + "=" * 50)
        logging.info("处理完成！")
        logging.info("=" * 50)
        logging.info(f"总文档数: {stats['total']}")
        logging.info(f"文件大小: {file_size:.2f} MB")
        logging.info(f"\n类别分布:")
        for cat, count in stats["by_class"].items():
            if count > 0:
                pct = count / stats["total"] * 100
                logging.info(f"  {cat:6s}: {count:4d} ({pct:.1f}%)")
        logging.info(f"\n输出文件: {out_path}")
        logging.info(f"\n后续使用:")
        logging.info(f"  python -m topic_model.cli analyze {out_path} -k 10 -o output")

        return {
            "status": "success",
            "total_docs": stats["total"],
            "file_size_mb": file_size,
            "by_class": stats["by_class"],
            "output_file": str(out_path)
        }
    else:
        logging.error("未处理到任何文档")
        return {"status": "error", "message": "无有效文档"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="THUCNews 语料库处理工具")
    parser.add_argument("input_dir", nargs="?", default=None,
                       help="THUCNews 解压目录 (默认: data/THUCNews)")
    parser.add_argument("-o", "--output", default=None,
                       help="输出文件路径 (默认: data/thuc_corpus.txt)")
    parser.add_argument("-n", "--num-per-class", type=int, default=500,
                       help="每类别最大文档数 (默认: 500)")
    parser.add_argument("-m", "--max-total", type=int, default=10000,
                       help="总文档数上限 (默认: 10000)")
    parser.add_argument("--no-limit", action="store_true",
                       help="不限制数量，处理全部文档")

    args = parser.parse_args()

    if args.no_limit:
        args.num_per_class = 999999
        args.max_total = 99999999

    process_thucnews(
        thuc_dir=args.input_dir,
        output_file=args.output,
        max_docs_per_class=args.num_per_class,
        max_total_docs=args.max_total
    )


if __name__ == "__main__":
    main()
