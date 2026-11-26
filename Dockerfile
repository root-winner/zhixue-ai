# ============================================
# 智学AI 后端（云端构建用：GitHub Actions / 华为云）
# 本机不再构建。无 torch，Embedding 走方舟在线 API。
# ============================================

FROM python:3.11-slim AS base

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_DEFAULT_TIMEOUT=120 \
    USE_LOCAL_EMBEDDING=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.docker.txt .
RUN pip install --no-cache-dir --retries 5 -r requirements.docker.txt

COPY . .

RUN mkdir -p /app/data/knowledge_base /app/chroma_db /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
