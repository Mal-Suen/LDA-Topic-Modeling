# LDA-Topic-Modeling

基于 Gensim 的中文 LDA 主题建模工具。支持中文分词、模型训练、评估、可视化和命令行操作。

## 特性

- 中文分词支持（jieba）
- LDA 主题建模（gensim）
- C_V 主题一致性评估
- pyLDAvis 交互式可视化
- 命令行界面（CLI）
- 最优主题数搜索

## 安装

```bash
# 克隆项目
git clone https://github.com/Mal-Suen/LDA-Topic-Modeling.git
cd LDA-Topic-Modeling

# 安装依赖
pip install -r requirements.txt

# 或者使用 pip 安装（如果发布到 PyPI）
pip install lda-topic-modeling
```

## 快速开始

### 方式一：直接运行

```bash
python run.py
```

### 方式二：使用 CLI

```bash
# 执行主题建模
python -m topic_model.cli analyze data/sample_corpus.txt -k 3 -o output

# 寻找最优主题数
python -m topic_model.cli find-topics data/sample_corpus.txt --min 2 --max 10

# 中文分词
python -m topic_model.cli tokenize data/sample_corpus.txt -s data/stopwords.txt
```

### 方式三：Python API

```python
from topic_model.lda_model import LDATopicModel

# 初始化模型
model = LDATopicModel(num_topics=3, passes=15)

# 加载数据并运行
results = model.run_analysis('data/sample_corpus.txt', output_dir='output')

# 查看结果
print(f"一致性得分: {results['coherence_score']:.4f}")
```

## CLI 命令说明

### analyze - 执行主题建模

```bash
python -m topic_model.cli analyze <输入文件> [选项]

选项:
  -k, --num-topics N      主题数量 (默认: 3)
  -p, --passes N          训练轮数 (默认: 15)
  -i, --iterations N      每轮迭代次数 (默认: 100)
  --seed N                随机种子 (默认: 42)
  -o, --output DIR        输出目录 (默认: output)
  -s, --stopwords FILE    停用词文件
  --save-model DIR        保存模型到指定目录
  -v, --verbose           显示详细日志
```

### find-topics - 寻找最优主题数

```bash
python -m topic_model.cli find-topics <输入文件> [选项]

选项:
  --min N      最小主题数 (默认: 2)
  --max N      最大主题数 (默认: 10)
  -p, --passes N    训练轮数 (默认: 15)
  -s, --stopwords FILE  停用词文件
```

### tokenize - 中文分词

```bash
python -m topic_model.cli tokenize [输入文件] [选项]

选项:
  -d, --delimiter STR   输出分隔符 (默认: 空格)
  -s, --stopwords FILE  停用词文件
```

## 数据格式

输入文件为纯文本，每行一篇文档。词之间用空格分隔（或依赖内置分词）。

```
春节 联欢晚会 除夕 守岁 烟花 新年
股市 大盘 下跌 散户 亏钱 套牢
机器学习 算法 模型 训练 数据 特征
```

## 项目结构

```
LDA-Topic-Modeling/
├── topic_model/          # 核心包
│   ├── __init__.py
│   ├── __main__.py       # CLI入口
│   ├── cli.py            # 命令行解析
│   └── lda_model.py      # LDA模型实现
├── tests/                # 单元测试
│   └── test_lda_model.py
├── data/                 # 示例数据
│   ├── sample_corpus.txt
│   └── stopwords.txt
├── output/               # 分析输出
├── run.py                # 快速运行脚本
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 测试

```bash
pip install pytest
pytest tests/ -v
```

## 依赖

- Python >= 3.8
- gensim >= 4.3.0
- jieba >= 0.42.1
- pyLDAvis >= 3.4.1
- numpy >= 1.24.0

## License

MIT
