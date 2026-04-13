# 多阶段构建 - 构建阶段
FROM python:3.11-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt pyproject.toml ./

# 安装Python依赖
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 最终阶段
FROM python:3.11-slim

WORKDIR /app

# 复制构建好的依赖
COPY --from=builder /install /usr/local

# 复制项目代码
COPY . .

# 创建非root用户
RUN useradd -m appuser && \
    mkdir -p /app/results /app/data && \
    chown -R appuser:appuser /app

USER appuser

# 默认命令
ENTRYPOINT ["python", "-m", "topic_model.cli"]

# 显示帮助信息
CMD ["--help"]
