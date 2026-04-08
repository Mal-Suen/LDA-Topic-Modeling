"""
LDA主题建模单元测试
"""

import os
import sys
import pytest
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_model.lda_model import LDATopicModel


@pytest.fixture
def sample_texts():
    """Fixture: 提供测试文本数据"""
    return [
        "春节 联欢晚会 除夕 守岁 烟花 新年 团圆",
        "新春 备 年货 拜年 红包 压岁钱 过年",
        "股市 大盘 下跌 散户 亏钱 套牢 亏损",
        "财经 股票 反弹 行情 股民 交易所 指数",
    ]


@pytest.fixture
def model(sample_texts):
    """Fixture: 提供训练好的LDA模型"""
    m = LDATopicModel(num_topics=2, passes=5)
    m.load_corpus_from_texts(sample_texts)
    m.build_dictionary_and_corpus()
    m.train_model()
    return m


class TestLDATopicModel:
    """LDA主题建模测试类"""

    def test_tokenize(self):
        """测试中文分词"""
        model = LDATopicModel()
        words = model.tokenize("机器学习算法模型训练")
        assert len(words) > 0
        # 应该过滤掉单字
        for w in words:
            assert len(w) > 1

    def test_tokenize_with_stopwords(self):
        """测试带停用词的分词"""
        model = LDATopicModel(custom_stopwords=["算法", "模型"])
        words = model.tokenize("机器学习算法模型训练")
        assert "算法" not in words
        assert "模型" not in words

    def test_load_corpus(self, tmp_path):
        """测试语料加载"""
        # 创建临时文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("春节 年货 红包\n股市 大盘 下跌\n", encoding='utf-8')

        model = LDATopicModel()
        docs = model.load_corpus(str(test_file))
        assert len(docs) == 2
        assert len(model.documents) == 2

    def test_load_corpus_from_texts(self, sample_texts):
        """测试从文本列表加载"""
        model = LDATopicModel()
        docs = model.load_corpus_from_texts(sample_texts)
        assert len(docs) == 4

    def test_build_dictionary(self, model):
        """测试词典构建"""
        assert model.dictionary is not None
        assert len(model.dictionary) > 0
        assert model.corpus is not None
        assert len(model.corpus) == 4

    def test_train_model(self, model):
        """测试模型训练"""
        assert model.lda_model is not None
        assert model.is_fitted is True
        assert model.num_topics == 2

    def test_get_topic_keywords(self, model):
        """测试主题关键词提取"""
        topics = model.get_topic_keywords(top_n=5)
        assert len(topics) == 2
        for topic_id, keywords in topics.items():
            assert len(keywords) == 5
            for word, weight in keywords:
                assert isinstance(word, str)
                assert isinstance(weight, float)

    def test_analyze_document_topics(self, model):
        """测试文档主题分析"""
        doc_topics = model.analyze_document_topics(top_n=2)
        assert len(doc_topics) == 4
        for doc_id, topics in doc_topics.items():
            assert len(topics) <= 2
            for topic_id, prob in topics:
                assert 0.0 <= prob <= 1.0

    def test_evaluate_model(self, model):
        """测试模型评估"""
        score = model.evaluate_model()
        assert isinstance(score, float)
        # C_V分数通常在0-1之间
        assert 0.0 <= score <= 1.0

    def test_save_model(self, model, tmp_path):
        """测试模型保存"""
        model_dir = str(tmp_path / "model")
        model.save_model(model_dir)
        
        assert Path(model_dir, "lda_model").exists()
        assert Path(model_dir, "dictionary.dict").exists()

    def test_run_analysis(self, tmp_path):
        """测试完整分析流程"""
        # 创建测试数据
        test_file = tmp_path / "corpus.txt"
        test_file.write_text(
            "春节 年货 红包 团圆\n股市 大盘 下跌 散户\n",
            encoding='utf-8'
        )
        output_dir = str(tmp_path / "output")

        model = LDATopicModel(num_topics=2, passes=5)
        results = model.run_analysis(str(test_file), output_dir=output_dir)

        assert 'topic_keywords' in results
        assert 'doc_topics' in results
        assert 'coherence_score' in results
        assert Path(output_dir, "lda_visualization.html").exists()


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
        """测试文件不存在时"""
        stopwords = LDATopicModel.load_stopwords("/nonexistent/path")
        assert stopwords == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
