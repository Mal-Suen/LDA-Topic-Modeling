#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA主题建模核心模块 - 增强版
支持：
- N-gram/Bigram自动检测
- 自动优化主题数
- 文档分类
- 报告导出
"""

import logging
import json
import csv
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Generator
from collections import Counter

import jieba
from gensim import corpora, models
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

logger = logging.getLogger(__name__)


class LDATopicModel:
    """
    LDA主题建模类（增强版）
    
    功能：
    - 中文分词 + N-gram检测
    - 自动优化主题数
    - 多系数对投票（已有）
    - 文档分类
    - 报告导出
    """

    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 15,
        iterations: int = 100,
        random_state: int = 42,
        custom_stopwords: Optional[List[str]] = None,
        custom_dict: Optional[List[str]] = None,
        ngram_mode: str = "auto",
    ):
        """
        Args:
            num_topics: 主题数量
            passes: 训练轮数
            iterations: 每轮迭代次数
            random_state: 随机种子
            custom_stopwords: 自定义停用词列表
            custom_dict: 自定义词典（用于jieba）
            ngram_mode: N-gram 模式 ("none", "auto", "strict")
        """
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        self.ngram_mode = ngram_mode
        self.stopwords = set(custom_stopwords or [])
        
        # 加载自定义词典
        if custom_dict:
            for word in custom_dict:
                jieba.add_word(word)
            logger.info(f"加载自定义词典: {len(custom_dict)} 个词")
        
        # 内部状态
        self.bigram_model = None
        self.trigram_model = None
        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[List] = None
        self.lda_model: Optional[models.LdaModel] = None
        self.documents: List[List[str]] = []
        self.original_documents: List[List[str]] = []  # 保存原始分词结果
        self.is_fitted = False

    @staticmethod
    def load_stopwords(file_path: str) -> List[str]:
        """从文件加载停用词表（每行一个词）"""
        stopwords = []
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f if line.strip()]
            logger.info(f"加载停用词: {len(stopwords)} 个")
        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """对中文文本进行分词，过滤停用词和单字"""
        words = jieba.lcut(text)
        filtered = [
            w for w in words
            if w not in self.stopwords
            and len(w) > 1
            and not w.isdigit()
            and not all(c.isspace() for c in w)
        ]
        return filtered

    def build_ngram_models(self, documents: List[List[str]]) -> None:
        """
        构建Bigram模型（增强版）
        - 支持多种预设模式
        - 先检测词组，再过滤停用词
        - 保存原始文档副本以便回溯
        """
        # 保存原始文档副本
        self.original_documents = [doc[:] for doc in documents]
        logger.info(f"已保存原始文档副本: {len(documents)} 篇")

        # 根据模式设定参数
        if self.ngram_mode == "none":
            logger.info("N-gram 模式：无 (跳过 Bigram 检测)")
            return

        elif self.ngram_mode == "auto":
            min_count, threshold = 15, 50.0
            logger.info("N-gram 模式：自动 (适合混合语料)")

        elif self.ngram_mode == "strict":
            min_count, threshold = 8, 100.0
            logger.info("N-gram 模式：严格 (适合专业术语)")

        else:
            raise ValueError(f"未知的 ngram_mode: {self.ngram_mode}")

        logger.info("构建 Bigram 模型...")

        self.bigram_model = Phrases(
            documents,
            min_count=min_count,
            threshold=threshold,
            delimiter='_',
            max_vocab_size=500000
        )
        bigram = Phraser(self.bigram_model)

        # 应用 Bigram 后再次过滤停用词
        ngram_docs = []
        for doc in documents:
            ngram_doc = bigram[doc]
            filtered = [
                w for w in ngram_doc
                if w not in self.stopwords
                and len(w.replace('_', '')) > 1
                and not w.replace('_', '').isdigit()
            ]
            if filtered:
                ngram_docs.append(filtered)

        logger.info(f"Bigram 检测完成，发现 {len(bigram.phrasegrams)} 个有效词组")
        logger.info(f"过滤后有效文档：{len(ngram_docs)} 篇")

        self.documents = ngram_docs
        logger.info("N-gram 处理完成")

    def reset_to_original_documents(self) -> None:
        """
        重置为原始分词结果（不含N-gram）
        可用于尝试不同的N-gram参数
        """
        if not self.original_documents:
            logger.warning("没有可用的原始文档副本")
            return
        
        self.documents = [doc[:] for doc in self.original_documents]
        self.bigram_model = None
        logger.info(f"已重置为原始文档: {len(self.documents)} 篇")

    def load_corpus(self, file_path: str) -> Optional[List[List[str]]]:
        """加载语料数据，自动进行中文分词
        
        Returns:
            分词后的文档列表，失败时返回None
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            return []

        try:
            # 检查文件大小
            file_size = path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"文件较大 ({file_size / (1024*1024):.2f} MB)，建议使用 load_corpus_streaming() 方法")

            with open(path, 'r', encoding='utf-8') as f:
                raw_docs = [line.strip() for line in f if line.strip()]

        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误，请使用UTF-8编码: {e}")
            return []
        except PermissionError:
            logger.error(f"没有读取权限: {file_path}")
            return []
        except Exception as e:
            logger.error(f"加载文件时发生错误: {e}")
            return []

        logger.info(f"加载原始文档: {len(raw_docs)} 篇")

        self.documents = [self.tokenize(doc) for doc in raw_docs]
        self.documents = [doc for doc in self.documents if doc]
        logger.info(f"分词完成，有效文档: {len(self.documents)} 篇")

        # 构建N-gram
        self.build_ngram_models(self.documents)
        return self.documents

    def load_corpus_streaming(self, file_path: str, batch_size: int = 1000) -> Generator[List[str], None, None]:
        """
        流式加载语料数据（适合大文件）
        
        Args:
            file_path: 输入文件路径
            batch_size: 每批处理的文档数量
            
        Yields:
            分批返回处理后的文档
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            return

        try:
            file_size = path.stat().st_size
            logger.info(f"流式加载大文件: {file_size / (1024*1024):.2f} MB")
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return

        batch = []
        doc_count = 0

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 分词并过滤
                    tokens = self.tokenize(line)
                    if tokens:
                        batch.append(tokens)
                        doc_count += 1

                    # 达到批次大小，yield并清空
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                # 返回剩余文档
                if batch:
                    yield batch

        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误: {e}")
        except Exception as e:
            logger.error(f"流式加载失败: {e}")

        logger.info(f"流式加载完成，共处理 {doc_count} 篇文档")

    def load_corpus_from_texts(self, texts: List[str]) -> Optional[List[List[str]]]:
        """从字符串列表加载语料
        
        Returns:
            分词后的文档列表，失败时返回None
        """
        self.documents = [self.tokenize(text) for text in texts]
        self.documents = [doc for doc in self.documents if doc]
        logger.info(f"分词完成，有效文档: {len(self.documents)} 篇")
        self.build_ngram_models(self.documents)
        return self.documents

    def build_dictionary_and_corpus(
        self,
        no_below: int = 5,
        no_above: float = 0.5,
        keep_n: int = 100000
    ) -> Tuple[Optional[corpora.Dictionary], Optional[List]]:
        """
        构建词典和BoW语料库
        
        Returns:
            (dictionary, corpus) 元组，失败时返回 (None, None)
        """
        if not self.documents:
            logger.error("没有可用的文档数据")
            return None, None

        self.dictionary = corpora.Dictionary(self.documents)
        before = len(self.dictionary)
        
        self.dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n
        )
        after = len(self.dictionary)
        
        logger.info(f"词典构建: {before} -> {after} 个词（过滤后）")
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        return self.dictionary, self.corpus

    def train_model(self) -> Optional[models.LdaModel]:
        """
        训练LDA模型
        
        Returns:
            训练好的LDA模型，失败时返回None
        """
        if self.corpus is None:
            self.build_dictionary_and_corpus()

        self.lda_model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            random_state=self.random_state,
            alpha='auto',
            per_word_topics=True
        )
        self.is_fitted = True
        logger.info(f"LDA模型训练完成: {self.num_topics} 个主题, {self.passes} 轮")
        return self.lda_model

    def find_optimal_topics(
        self,
        min_topics: int = 5,
        max_topics: int = 20,
        step: int = 1
    ) -> List[Tuple[int, float]]:
        """
        寻找最优主题数量
        通过C_V一致性分数评估
        
        Args:
            min_topics: 最小主题数
            max_topics: 最大主题数
            step: 搜索步长
            
        Returns:
            List of (k, score) tuples
        """
        topic_range = list(range(min_topics, max_topics + 1, step))
        logger.info(f"搜索最优主题数: {topic_range}")
        results = []
        original_num_topics = self.num_topics

        for k in topic_range:
            self.num_topics = k
            logger.info(f"\n尝试主题数 k={k}...")
            self.build_dictionary_and_corpus()
            self.train_model()
            score = self.evaluate_model()
            results.append((k, score))
            logger.info(f"  k={k:3d}, C_V={score:.4f}")

        self.num_topics = original_num_topics

        best_k = max(results, key=lambda x: x[1])
        logger.info(f"\n最优主题数: {best_k[0]} (C_V={best_k[1]:.4f})")
        return results

    def get_topic_keywords(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """获取每个主题的关键词"""
        if self.lda_model is None:
            self.train_model()
        return {
            tid: self.lda_model.show_topic(tid, topn=top_n)
            for tid in range(self.num_topics)
        }

    def analyze_document_topics(
        self,
        top_n: int = 3
    ) -> Dict[int, List[Tuple[int, float]]]:
        """分析每篇文档的主题分布"""
        if self.lda_model is None:
            self.train_model()
        doc_topics = {}
        for doc_id in range(len(self.documents)):
            doc_bow = self.dictionary.doc2bow(self.documents[doc_id])
            topic_dist = self.lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
            top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:top_n]
            doc_topics[doc_id] = top_topics
        return doc_topics

    def classify_documents(self) -> List[Dict[str, Any]]:
        """
        对所有文档进行分类
        返回每篇文档的主主题和置信度
        """
        if self.lda_model is None:
            self.train_model()
        
        classifications = []
        for doc_id in range(len(self.documents)):
            doc_bow = self.dictionary.doc2bow(self.documents[doc_id])
            topic_dist = self.lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
            top_topic = max(topic_dist, key=lambda x: x[1])
            
            classifications.append({
                'doc_id': doc_id,
                'primary_topic': top_topic[0],
                'confidence': top_topic[1],
                'all_topics': sorted(topic_dist, key=lambda x: x[1], reverse=True)[:3]
            })
        return classifications

    def evaluate_model(self) -> float:
        """评估模型性能（C_V一致性分数）"""
        if self.lda_model is None:
            self.train_model()
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=self.documents,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        logger.info(f"主题一致性得分 (C_V): {score:.4f}")
        return score

    def export_report(self, output_dir: str) -> Dict[str, Any]:
        """
        导出详细分析报告（JSON + CSV）
        """
        if self.lda_model is None:
            self.train_model()
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 主题关键词
        topics = self.get_topic_keywords(top_n=15)
        
        # 2. 文档分类
        classifications = self.classify_documents()
        
        # 3. 模型评估
        coherence = self.evaluate_model()
        
        # 4. 主题文档统计
        topic_doc_counts = Counter(c['primary_topic'] for c in classifications)
        
        # 构建报告
        report = {
            'model_info': {
                'num_topics': self.num_topics,
                'passes': self.passes,
                'iterations': self.iterations,
                'num_documents': len(self.documents),
                'vocab_size': len(self.dictionary),
                'coherence_score': coherence
            },
            'topics': {},
            'topic_distribution': dict(topic_doc_counts)
        }
        
        for tid, keywords in topics.items():
            report['topics'][f'topic_{tid}'] = {
                'keywords': [{'word': w, 'weight': round(float(p), 4)} for w, p in keywords],
                'document_count': topic_doc_counts.get(tid, 0),
                'percentage': round(float(topic_doc_counts.get(tid, 0)) / len(self.documents) * 100, 2)
            }
        
        # 保存JSON
        json_path = out_path / 'report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON报告: {json_path}")
        
        # 保存CSV（每篇文档的分类）
        csv_path = out_path / 'classifications.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['doc_id', 'primary_topic', 'confidence', 'topic_distribution'])
            for c in classifications:
                dist_str = ';'.join([f"t{t}:{p:.3f}" for t, p in c['all_topics']])
                writer.writerow([c['doc_id'], c['primary_topic'], round(c['confidence'], 4), dist_str])
        logger.info(f"CSV分类: {csv_path}")
        
        return report

    def save_visualization(self, output_path: str = "lda_visualization.html") -> None:
        """保存pyLDAvis交互式可视化"""
        if self.lda_model is None:
            self.train_model()
        vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)
        pyLDAvis.save_html(vis_data, output_path)
        logger.info(f"可视化: {output_path}")

    def save_model(self, model_dir: str) -> None:
        """保存模型、词典和N-gram模型"""
        if self.lda_model is None:
            self.train_model()
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.lda_model.save(str(path / "lda_model"))
        self.dictionary.save(str(path / "dictionary.dict"))
        if self.bigram_model:
            self.bigram_model.save(str(path / "bigram_model"))
        logger.info(f"模型: {model_dir}")

    def run_analysis(
        self,
        file_path: str,
        output_dir: str = "results",
        auto_find_k: bool = False,
        k_min: int = 5,
        k_max: int = 20,
        k_step: int = 1
    ) -> Dict[str, Any]:
        """
        执行完整分析流程

        Args:
            file_path: 输入文本文件路径
            output_dir: 输出目录
            auto_find_k: 是否自动搜索最优K
            k_min: 最小主题数
            k_max: 最大主题数
            k_step: 搜索步长
        """
        logger.info("=" * 50)
        logger.info("LDA 主题建模分析开始")
        logger.info("=" * 50)

        # 1. 加载数据
        self.load_corpus(file_path)

        # 2. 自动搜索最优K
        if auto_find_k:
            logger.info(f"\n自动搜索最优主题数: range({k_min}, {k_max}, {k_step})")
            results = self.find_optimal_topics(
                min_topics=k_min,
                max_topics=k_max,
                step=k_step
            )
            best_k = max(results, key=lambda x: x[1])
            logger.info(f"✅ 最优主题数: {best_k[0]} (C_V={best_k[1]:.4f})")
            self.num_topics = best_k[0]
        
        # 3. 构建和训练
        self.build_dictionary_and_corpus()
        self.train_model()

        # 4. 导出报告
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report = self.export_report(str(out_path))
        
        # 5. 可视化
        self.save_visualization(str(out_path / "lda_visualization.html"))
        
        # 6. 保存模型
        self.save_model(str(out_path / "model"))
        
        # 7. 打印摘要
        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict) -> None:
        """打印分析摘要"""
        print("\n" + "=" * 60)
        print("  主题分析报告")
        print("=" * 60)
        
        info = report['model_info']
        print(f"\n  模型配置:")
        print(f"    文档数: {info['num_documents']}")
        print(f"    词典大小: {info['vocab_size']}")
        print(f"    主题数: {info['num_topics']}")
        print(f"    C_V得分: {info['coherence_score']:.4f}")
        
        print(f"\n  主题关键词 (Top 8):")
        print("-" * 60)
        for tid, tdata in report['topics'].items():
            words = ", ".join([k['word'] for k in tdata['keywords'][:8]])
            pct = tdata['percentage']
            print(f"    {tid:8s}: {words} ({pct}%)")
        
        print("=" * 60)
