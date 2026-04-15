# -*- coding: utf-8 -*-
"""
LDA主题建模单元测试

测试覆盖:
    - TestLDATopicModel: 核心功能测试
    - TestStopwords: 停用词加载测试
    - TestErrorHandling: 错误处理测试

运行: pytest tests/test_lda_model.py -v
"""
import os
import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_model.lda_model import LDATopicModel


# ============================================================================
# 测试数据 Fixtures
# ============================================================================

@pytest.fixture
def sample_texts():
    """测试文本（春节/股市两个主题）"""
    return [
        "春节 联欢晚会 除夕 守岁 烟花 新年 团圆",
        "新春 备 年货 拜年 红包 压岁钱 过年",
        "股市 大盘 下跌 散户 亏钱 套牢 亏损",
        "财经 股票 反弹 行情 股民 交易所 指数",
    ]


@pytest.fixture
def model(sample_texts):
    """预训练的LDA模型（2个主题）"""
    m = LDATopicModel(num_topics=2, passes=5)
    m.load_corpus_from_texts(sample_texts)
    m.build_dictionary_and_corpus()
    m.train_model()
    return m


# ============================================================================
# 核心功能测试
# ============================================================================

class TestLDATopicModel:
    """LDA主题建模核心功能测试"""

    # ---- 分词测试 ----

    def test_tokenize(self):
        """测试中文分词，单字应被过滤"""
        model = LDATopicModel()
        words = model.tokenize("机器学习算法模型训练")
        assert len(words) > 0
        for w in words:
            assert len(w) > 1

    def test_tokenize_with_stopwords(self):
        """测试停用词过滤"""
        model = LDATopicModel(custom_stopwords=["算法", "模型"])
        words = model.tokenize("机器学习算法模型训练")
        assert "算法" not in words
        assert "模型" not in words

    # ---- 语料加载测试 ----

    def test_load_corpus(self, tmp_path):
        """测试从文件加载语料"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("春节 年货 红包\n股市 大盘 下跌\n", encoding='utf-8')

        model = LDATopicModel()
        docs = model.load_corpus(str(test_file))
        assert len(docs) == 2

    def test_load_corpus_not_found(self):
        """测试文件不存在时的处理"""
        model = LDATopicModel()
        docs = model.load_corpus("/nonexistent/path/file.txt")
        assert docs == []

    def test_load_corpus_from_texts(self, sample_texts):
        """测试从文本列表加载"""
        model = LDATopicModel()
        docs = model.load_corpus_from_texts(sample_texts)
        assert len(docs) == 4

    # ---- 词典和模型测试 ----

    def test_build_dictionary(self, model):
        """测试词典构建"""
        assert model.dictionary is not None
        assert len(model.dictionary) > 0
        assert len(model.corpus) == 4

    def test_train_model(self, model):
        """测试模型训练"""
        assert model.lda_model is not None
        assert model.is_fitted is True
        assert model.num_topics == 2

    # ---- 主题分析测试 ----

    def test_get_topic_keywords(self, model):
        """测试主题关键词提取"""
        topics = model.get_topic_keywords(top_n=5)
        assert len(topics) == 2
        for topic_id, keywords in topics.items():
            assert len(keywords) == 5

    def test_analyze_document_topics(self, model):
        """测试文档主题分布分析"""
        doc_topics = model.analyze_document_topics(top_n=2)
        assert len(doc_topics) == 4
        for doc_id, topics in doc_topics.items():
            for topic_id, prob in topics:
                assert 0.0 <= prob <= 1.0

    def test_evaluate_model(self, model):
        """测试模型评估（C_V分数应在0-1之间）"""
        score = model.evaluate_model()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # ---- 输出测试 ----

    def test_save_model(self, model, tmp_path):
        """测试模型保存"""
        model_dir = str(tmp_path / "model")
        model.save_model(model_dir)
        assert Path(model_dir, "lda_model").exists()
        assert Path(model_dir, "dictionary.dict").exists()

    def test_export_report(self, model, tmp_path):
        """测试报告导出"""
        output_dir = str(tmp_path / "report")
        report = model.export_report(output_dir)

        assert 'model_info' in report
        assert 'topics' in report
        assert Path(output_dir, "report.json").exists()
        assert Path(output_dir, "classifications.csv").exists()

    def test_run_analysis(self, tmp_path):
        """测试完整分析流程"""
        test_file = tmp_path / "corpus.txt"
        test_file.write_text("春节 年货 红包 团圆\n股市 大盘 下跌 散户\n", encoding='utf-8')
        output_dir = str(tmp_path / "output")

        model = LDATopicModel(num_topics=2, passes=5)
        results = model.run_analysis(str(test_file), output_dir=output_dir)

        assert 'model_info' in results
        assert Path(output_dir, "lda_visualization.html").exists()

    # ---- N-gram测试 ----

    def test_reset_to_original_documents(self, sample_texts):
        """测试重置为原始文档"""
        model = LDATopicModel(num_topics=2, passes=5, ngram_mode='auto')
        model.load_corpus_from_texts(sample_texts)

        assert len(model.original_documents) > 0
        model.reset_to_original_documents()
        assert len(model.documents) > 0
        assert model.bigram_model is None

    # ---- 流式加载测试 ----

    def test_load_corpus_streaming(self, tmp_path):
        """测试流式加载（适合大文件）"""
        test_file = tmp_path / "large_corpus.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            for i in range(100):
                f.write("春节 年货 红包 团圆\n" if i % 2 == 0 else "股市 大盘 下跌 散户\n")

        model = LDATopicModel()
        batches = list(model.load_corpus_streaming(str(test_file), batch_size=10))

        assert len(batches) > 0
        total_docs = sum(len(batch) for batch in batches)
        assert total_docs == 100


# ============================================================================
# 停用词测试
# ============================================================================

class TestStopwords:
    """停用词加载测试"""

    def test_load_stopwords(self, tmp_path):
        """测试从文件加载停用词"""
        sw_file = tmp_path / "stopwords.txt"
        sw_file.write_text("的\n了\n在\n", encoding='utf-8')

        stopwords = LDATopicModel.load_stopwords(str(sw_file))
        assert len(stopwords) == 3
        assert "的" in stopwords

    def test_load_stopwords_not_found(self):
        """测试文件不存在时返回空列表"""
        stopwords = LDATopicModel.load_stopwords("/nonexistent/path")
        assert stopwords == []


# ============================================================================
# 错误处理测试
# ============================================================================

class TestErrorHandling:
    """错误处理测试"""

    def test_empty_documents(self):
        """测试空文档列表"""
        model = LDATopicModel()
        dic, corpus = model.build_dictionary_and_corpus()
        assert dic is None
        assert corpus is None

    def test_load_corpus_empty_file(self, tmp_path):
        """测试加载空文件"""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding='utf-8')

        model = LDATopicModel()
        docs = model.load_corpus(str(test_file))
        assert docs == []

    def test_load_corpus_with_empty_lines(self, tmp_path):
        """测试加载包含空行的文件"""
        test_file = tmp_path / "with_empty.txt"
        test_file.write_text("\n\n春节 年货\n\n股市 下跌\n\n", encoding='utf-8')

        model = LDATopicModel()
        docs = model.load_corpus(str(test_file))
        assert len(docs) == 2  # 空行被过滤

    def test_ngram_mode_invalid(self):
        """测试无效的N-gram模式"""
        model = LDATopicModel(ngram_mode='invalid')
        with pytest.raises(ValueError, match="未知的 ngram_mode"):
            model.build_ngram_models([["test", "words"]])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
