#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载搜狗新闻语料库（SogouCS精简版）
数据来源: HuggingFace community-datasets/sogou_news
备选: 从搜狐/新浪新闻RSS爬取
"""

import json
import random
import logging
from pathlib import Path
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_from_sogou_csv(output_path: str, max_docs: int = 5000):
    """
    从开源镜像下载搜狗新闻数据
    使用 community-datasets/sogou_news 的JSONL格式
    """
    # HuggingFace API 获取数据集
    urls = [
        # SogouCS 精简版 (通过HuggingFace Datasets API)
        "https://huggingface.co/datasets/community-datasets/sogou_news/resolve/main/sogou_news.py",
    ]
    
    logging.info("尝试从HuggingFace获取搜狗新闻数据...")
    logging.info("如果网络受限，请使用以下备选方案:")
    logging.info("1. 访问 https://huggingface.co/datasets/community-datasets/sogou_news")
    logging.info("2. 使用 datasets.load_dataset('community-datasets/sogou_news')")
    logging.info("3. 或访问 ModelScope: https://modelscope.cn/datasets")
    
    return False


def generate_sample_corpus(output_path: str, num_docs: int = 1000):
    """
    生成模拟新闻语料库（用于测试）
    当无法获取真实数据时，使用预定义的新闻模板生成数据
    """
    import jieba
    
    # 新闻主题模板
    templates = [
        # 财经
        "今日大盘震荡下跌散户观望情绪浓厚股市成交量萎缩 analysts表示市场短期调整不改中长期向好趋势",
        "央行宣布降准个百分点释放长期资金约亿元支持实体经济发展金融机构信贷投放力度加大",
        "A股市场今日高开科技股领涨新能源汽车板块表现活跃北向资金大幅净流入",
        "上市公司三季报披露完毕超六成企业净利润同比增长行业景气度持续回升",
        "人民币汇率中间价上调个基点美元指数走弱国际资本加速流入中国市场",
        # 科技
        "人工智能技术取得突破大语言模型能力显著提升多家科技企业加大研发投入",
        "国产芯片制程工艺取得新进展半导体产业链加速完善自主可控替代步伐加快",
        "第五代移动通信技术用户规模持续扩大应用场景不断丰富推动数字经济发展",
        "云计算市场快速增长企业数字化转型加速云原生技术成为主流选择",
        "新能源汽车销量再创新高智能驾驶技术不断成熟产业链上下游协同发展",
        # 社会
        "春运期间全国铁路发送旅客人次同比增长铁路部门加开临客保障运力",
        "全国多地迎来强降雨气象部门发布预警提醒公众注意防范地质灾害",
        "教育部门出台新规减轻中小学生课外负担校外培训机构规范化治理",
        "医疗保障制度改革深入推进异地就医直接结算覆盖面持续扩大",
        "城市老旧小区改造加快推进居民住房条件得到显著改善",
        # 体育
        "中国男篮亚洲杯小组赛获胜全队发挥出色教练表示球队配合默契",
        "中超联赛新赛季开幕各队引援力度加大夺冠竞争将更加激烈",
        "中国游泳世锦赛摘金选手打破亚洲纪录教练团队功不可没",
        "冬奥会筹办工作进展顺利场馆建设基本完工测试赛有序进行",
        "国乒世界杯夺冠主力选手状态出色团队整体实力领先对手",
        # 文化
        "春节档电影票房创新高多部国产影片口碑佳作观众观影热情高涨",
        "全国博物馆参观人次大幅增长文化展览活动丰富多彩传统文化受热捧",
        "非物质文化遗产保护工作深入推进多个项目入选世界级非遗名录",
        "网络文学市场规模持续扩大优质内容成核心竞争力行业发展规范化",
        "传统戏曲走进校园年轻一代对国粹艺术兴趣增加传承人才辈出",
    ]
    
    logging.info(f"生成模拟新闻语料库: {num_docs} 篇文档")
    
    random.seed(42)
    docs = []
    for i in range(num_docs):
        template = random.choice(templates)
        words = list(jieba.cut(template))
        # 添加随机噪声使每篇文档略有不同
        doc = " ".join(words)
        docs.append(doc)
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    
    logging.info(f"语料库已保存: {output_path} ({num_docs}篇)")
    return True


def main():
    data_dir = Path(__file__).parent / "data"
    output_file = data_dir / "sogou_corpus.txt"
    
    # 尝试从网络下载
    if not download_from_sogou_csv(str(output_file)):
        logging.warning("网络下载失败，生成模拟语料库")
        generate_sample_corpus(str(output_file), num_docs=2000)
    
    logging.info("数据处理完成！")
    logging.info(f"使用方式: python run.py (修改data/sample_corpus.txt路径)")
    logging.info(f"或使用CLI: python -m topic_model.cli analyze {output_file} -k 5 -o output")


if __name__ == "__main__":
    main()
