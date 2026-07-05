"""
pytest 公共夹具

为单元测试提供 mock 环境, 避免连接真实 LLM / 向量库。
测试运行前设置占位环境变量, 防止模块初始化时因缺少 API Key 报错。
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================
# 环境变量预设 (在导入任何 app 模块前设置)
# ============================================================
os.environ.setdefault("LLM_API_KEY", "sk-test-placeholder")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("LLM_MODEL_NAME", "test-model")
os.environ.setdefault("EMBEDDING_API_KEY", "sk-test-placeholder")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "test-embedding")
os.environ.setdefault("USE_LOCAL_EMBEDDING", "false")

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 公共夹具
# ============================================================

@pytest.fixture
def sample_documents():
    """构造测试用 Document 列表"""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="智学AI是一个基于RAG技术的智能知识库对话系统，支持文档上传和智能问答。",
            metadata={"source_file": "intro.pdf", "doc_id": "001"},
        ),
        Document(
            page_content="系统采用FastAPI作为后端框架，使用LangChain进行AI能力编排，ChromaDB存储向量数据。",
            metadata={"source_file": "tech.pdf", "doc_id": "002"},
        ),
        Document(
            page_content="Python是一种广泛使用的高级编程语言，以简洁易读著称。",
            metadata={"source_file": "python.txt", "doc_id": "003"},
        ),
        Document(
            page_content="天气查询工具通过wttr.in免费API获取实时天气信息，无需API Key。",
            metadata={"source_file": "tools.pdf", "doc_id": "004"},
        ),
    ]


@pytest.fixture
def mock_vectorstore():
    """Mock 向量数据库, 避免连接真实 Chroma"""
    store = MagicMock()
    store._collection.count.return_value = 100
    return store


@pytest.fixture
def rag_engine_mock(mock_vectorstore):
    """
    构造不连接真实服务的 RAGEngine 实例
    通过 patch 掉 _init_vectorstore 实现
    """
    from app.core.rag import RAGEngine

    with patch.object(RAGEngine, "_init_vectorstore", lambda self: setattr(self, "_vectorstore", mock_vectorstore)):
        engine = RAGEngine()
        yield engine


# ============================================================
# 集成测试辅助: 检测是否有真实 LLM API Key
# ============================================================
def has_real_llm_key() -> bool:
    """判断是否配置了真实 (非占位) LLM API Key"""
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        return False
    placeholders = ("placeholder", "test", "sk-your", "your-key")
    return not any(p in key.lower() for p in placeholders)


skip_no_llm = pytest.mark.skipif(
    not has_real_llm_key(),
    reason="集成测试需要真实 LLM API Key (在 .env 中配置 LLM_API_KEY)",
)
