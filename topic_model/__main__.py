# -*- coding: utf-8 -*-
"""
CLI入口点 - 支持 python -m topic_model 命令执行
"""
import sys
from topic_model.cli import main

if __name__ == '__main__':
    sys.exit(main())
