# ============================================
# 智学AI 前端 Dockerfile (Streamlit)
# ============================================

FROM python:3.11-slim AS base

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 前端只需 streamlit 和 httpx
RUN pip install --no-cache-dir streamlit==1.41.1 httpx==0.28.1

COPY frontend/ ./frontend/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
