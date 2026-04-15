# -*- coding: utf-8 -*-
"""
LDA主题建模工具包 - 基于gensim的中文主题分析

功能:
    - 中文分词与预处理
    - N-gram词组检测
    - LDA模型训练与评估
    - 交互式可视化
"""
import logging

__version__ = "1.0.0"
__author__ = "malcolmsuen"

# 使用NullHandler避免库级别强制输出日志，由使用者决定输出方式
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def setup_logging(level: str = "INFO", format_string: str = None) -> None:
    """
    配置项目日志系统
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        format_string: 日志格式字符串
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 获取'topic_model'命名空间的根记录器，子模块会继承此配置
    root_logger = logging.getLogger('topic_model')
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(handler)
    root_logger.propagate = True
