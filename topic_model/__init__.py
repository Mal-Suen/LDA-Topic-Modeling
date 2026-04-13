# LDA主题建模工具包
"""
基于gensim的LDA主题建模工具，支持中文分词、模型训练、评估和可视化。
"""

import logging

__version__ = "1.0.0"
__author__ = "malcolmsuen"

# 统一配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    # 添加NullHandler避免"No handler found"警告
    logger.addHandler(logging.NullHandler())


def setup_logging(level: str = "INFO", format_string: str = None) -> None:
    """
    配置项目日志
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        format_string: 日志格式字符串
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置根日志记录器
    root_logger = logging.getLogger('topic_model')
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 设置日志级别
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 添加控制台处理器
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(handler)
    
    # 传播到子记录器
    root_logger.propagate = True
