"""
智学AI - 智能知识库对话系统

FastAPI 主应用入口

启动方式:
  开发模式: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  生产模式: uvicorn app.main:app --host 0.0.0.0 --port 8000

API 文档: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, knowledge, tools
from app.core.llm import llm_manager
from app.core.memory import memory_manager
from app.core.metrics import metrics
from app.core.rag import rag_engine
from app.models.schemas import HealthResponse
from app.utils.logger import log
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    log.info("=" * 50)
    log.info("智学AI 启动中...")
    log.info(f"LLM 模型: {settings.llm_model_name}")
    log.info(f"Embedding: {'本地' if settings.use_local_embedding else '在线'} - {settings.embedding_model_name}")
    log.info(f"知识库文档块数: {rag_engine._count_chunks()}")
    log.info("=" * 50)

    yield

    # 关闭时
    log.info("智学AI 正在关闭...")
    memory_manager.cleanup_expired()
    log.info("已清理会话, 再见!")


# 创建 FastAPI 应用
app = FastAPI(
    title="智学AI - 智能知识库对话系统",
    description=(
        "基于 RAG 的智能知识库对话系统\n\n"
        "## 核心功能\n"
        "- 📚 RAG 知识库问答: 上传文档, 智能问答\n"
        "- 💬 多轮对话记忆: 上下文感知的流畅对话\n"
        "- 🔧 Function Calling: 天气查询、网络搜索\n"
        "- ⚡ 流式输出: 实时打字机效果\n\n"
        "## 技术栈\n"
        "FastAPI + LangChain + ChromaDB + OpenAI 兼容 LLM"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置 (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    """请求日志 + 指标采集中间件"""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    metrics.record_request(response.status_code, elapsed)
    log.info(
        f"{request.method} {request.url.path} -> "
        f"{response.status_code} ({elapsed * 1000:.0f}ms)"
    )
    return response


# 注册路由
app.include_router(chat.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(tools.router, prefix="/api")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查 / 系统状态"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        llm_model=settings.llm_model_name,
        embedding_model=(
            settings.local_embedding_model
            if settings.use_local_embedding
            else settings.embedding_model_name
        ),
        knowledge_chunks=rag_engine._count_chunks(),
        active_sessions=len(memory_manager.list_sessions()),
    )


@app.get("/api")
async def api_root():
    """API 根路径 - 返回所有可用接口"""
    return {
        "name": "智学AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "对话": {
                "流式对话": "POST /api/chat",
                "同步对话": "POST /api/chat/sync",
                "会话列表": "GET /api/chat/sessions",
                "清空会话": "DELETE /api/chat/session/{session_id}",
            },
            "知识库": {
                "上传文档": "POST /api/knowledge/upload",
                "文档列表": "GET /api/knowledge/documents",
                "统计信息": "GET /api/knowledge/stats",
                "清空知识库": "DELETE /api/knowledge/all",
            },
            "工具": {
                "工具列表": "GET /api/tools",
                "天气查询": "POST /api/tools/weather",
                "网络搜索": "POST /api/tools/search",
            },
            "可观测性": {
                "系统指标": "GET /api/metrics",
            },
        },
    }


@app.get("/api/metrics")
async def get_metrics():
    """系统指标 (Token 用量 / 检索监控 / 请求统计)"""
    return metrics.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
