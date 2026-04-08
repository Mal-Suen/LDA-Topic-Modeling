#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载搜狗新闻语料库 SogouCA
使用 Python ftplib 从搜狗实验室 FTP 下载
"""

import ftplib
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FTP_HOST = "ftp.labs.sogou.com"
FTP_USER = "hebin_hit@foxmail.com"
FTP_PASS = "4FqLSYdNcrDXvNDi"
REMOTE_PATH = "/Data/SogouCA/SogouCA.tar.gz"
LOCAL_FILE = "SogouCA.tar.gz"

def download_sogouca(output_dir: str):
    """
    通过 FTP 下载 SogouCA 语料库
    """
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, LOCAL_FILE)
    
    logging.info(f"连接 FTP: {FTP_HOST}")
    logging.info(f"远程路径: {REMOTE_PATH}")
    logging.info(f"本地保存: {local_path}")
    
    try:
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login(FTP_USER, FTP_PASS)
        logging.info("FTP 登录成功")
        
        # 切换到远程目录
        remote_dir = os.path.dirname(REMOTE_PATH)
        remote_file = os.path.basename(REMOTE_PATH)
        ftp.cwd(remote_dir)
        logging.info(f"已切换到 {remote_dir}")
        
        # 获取文件大小
        size = ftp.size(remote_file)
        if size:
            logging.info(f"文件大小: {size / (1024*1024):.1f} MB")
        
        # 下载文件
        logging.info("开始下载...")
        downloaded = 0
        
        with open(local_path, "wb") as f:
            def write_chunk(data):
                nonlocal downloaded
                f.write(data)
                downloaded += len(data)
                if size and downloaded % (10 * 1024 * 1024) < len(data):
                    pct = downloaded / size * 100
                    logging.info(f"  进度: {downloaded/(1024*1024):.1f} MB / {size/(1024*1024):.1f} MB ({pct:.1f}%)")
            
            ftp.retrbinary(f"RETR {remote_file}", write_chunk)
        
        ftp.quit()
        logging.info(f"下载完成: {local_path}")
        
        # 检查文件
        actual_size = os.path.getsize(local_path)
        logging.info(f"文件保存成功: {actual_size / (1024*1024):.1f} MB")
        
        return True
        
    except ftplib.all_errors as e:
        logging.error(f"FTP 错误: {e}")
        return False
    except Exception as e:
        logging.error(f"未知错误: {e}")
        return False


def extract_tar_gz(tar_path: str, output_dir: str):
    """
    解压 tar.gz 文件
    """
    import tarfile
    
    logging.info(f"解压: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    logging.info(f"解压完成到: {output_dir}")


def convert_xml_to_corpus(xml_dir: str, output_file: str, max_docs: int = 5000):
    """
    将搜狐/搜狗 XML 新闻格式转换为 LDA 工具可用的纯文本格式
    每行一篇文档（仅内容，已去除 HTML 标签）
    """
    import re
    import glob
    from pathlib import Path
    
    logging.info(f"转换 XML 文件为纯文本语料库")
    
    docs = []
    xml_files = list(Path(xml_dir).glob("*.xml"))
    logging.info(f"找到 {len(xml_files)} 个 XML 文件")
    
    for xml_file in xml_files[:max_docs]:
        try:
            content = xml_file.read_text(encoding="utf-8", errors="ignore")
            
            # 提取 <content> 标签内的内容
            match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)
            if match:
                text = match.group(1)
                # 移除 HTML 标签
                text = re.sub(r'<[^>]+>', '', text)
                # 移除多余空白
                text = ' '.join(text.split())
                if len(text) > 20:  # 过滤太短的
                    docs.append(text)
        except Exception as e:
            logging.warning(f"处理 {xml_file} 失败: {e}")
        
        if len(docs) % 500 == 0 and len(docs) > 0:
            logging.info(f"已处理 {len(docs)} 篇文档")
    
    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    
    logging.info(f"语料库已保存: {output_file} ({len(docs)} 篇)")
    return len(docs)


def main():
    data_dir = Path(__file__).parent / "data"
    output_file = data_dir / "sogou_corpus.txt"
    tar_path = data_dir / LOCAL_FILE
    
    # 1. 下载
    if not tar_path.exists():
        logging.info("开始下载 SogouCA 语料库...")
        if not download_sogouca(str(data_dir)):
            logging.error("下载失败")
            return
    else:
        logging.info(f"文件已存在: {tar_path}")
        logging.info("如需重新下载，请删除该文件后重新运行")
    
    # 2. 解压
    extract_dir = data_dir / "SogouCA"
    if not extract_dir.exists():
        logging.info("开始解压...")
        extract_tar_gz(str(tar_path), str(data_dir))
    else:
        logging.info(f"已解压: {extract_dir}")
    
    # 3. 转换为 LDA 可用格式
    logging.info("开始转换为纯文本语料库...")
    # SogouCA 解压后通常是 news_tensite_xml.smarty 目录
    xml_dir = extract_dir
    if not xml_dir.exists():
        # 尝试其他可能的目录
        for d in data_dir.iterdir():
            if d.is_dir() and "xml" in d.name.lower():
                xml_dir = d
                break
    
    count = convert_xml_to_corpus(str(xml_dir), str(output_file), max_docs=10000)
    
    logging.info("=" * 50)
    logging.info("数据处理完成！")
    logging.info(f"使用方式: python -m topic_model.cli analyze {output_file} -k 5 -o output")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
