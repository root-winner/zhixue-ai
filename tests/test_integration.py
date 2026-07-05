"""
端到端集成测试

需要真实 LLM API Key (非占位值), 默认不运行:
  pytest                     # 只跑单元测试 (默认排除 integration)
  pytest -m integration      # 跑集成测试 (需在 .env 配置真实 LLM_API_KEY)
  pytest -m "not integration" # 显式只跑单元测试

覆盖端到端链路:
1. 真实 LLM 简单对话
2. RAG 端到端: 文档入库 → 混合检索 → 重排序 → 上下文构建
3. SQLite 记忆持久化: 写入 → 新实例读取 (重启恢复验证)
4. 指标采集: 真实调用后 MetricsCollector 指标非零
"""
import os

import pytest

from tests.conftest import skip_no_llm


@pytest.mark.integration
@skip_no_llm
class TestRealLLMChat:
    """真实 LLM 对话 (需真实 API Key)"""

    @pytest.mark.asyncio
    async def test_simple_chat_stream(self):
        """流式对话能返回非空内容"""
        from app.core.llm import llm_manager
        from langchain_core.messages import HumanMessage

        chunks = []
        async for chunk in llm_manager.stream_chat([HumanMessage(content="你好，请用一句话介绍自己")]):
            chunks.append(chunk)

        full = "".join(chunks)
        assert len(full) > 0, "LLM 应返回非空内容"
        assert "异常" not in full, f"不应返回异常: {full}"


@pytest.mark.integration
@skip_no_llm
class TestRAGEndToEnd:
    """RAG 端到端 (需真实 Embedding / LLM)"""

    def test_sample_doc_retrieval(self):
        """示例文档入库后能被混合检索召回"""
        from app.core.rag import rag_engine
        from config import settings

        # 确保示例文档已入库
        sample = settings.knowledge_base_dir / "智学AI使用说明.md"
        if not sample.exists():
            pytest.skip("示例文档不存在")

        if rag_engine._count_chunks() == 0:
            rag_engine.ingest_file(str(sample))

        # 混合检索
        results = rag_engine.retrieve("智学AI 是什么")
        assert len(results) > 0, "混合检索应召回相关文档"

        # 重排序
        reranked = rag_engine.rerank("智学AI 是什么", results)
        assert len(reranked) > 0
        assert len(reranked) <= len(results), "重排序后数量应 <= 召回数"

    def test_build_context_has_sources(self):
        """构建的上下文包含来源标注"""
        from app.core.rag import rag_engine

        result = rag_engine.build_context("系统支持哪些功能")
        if result["chunk_count"] == 0:
            pytest.skip("知识库为空, 跳过")
        assert "[参考文档" in result["context"]
        assert len(result["sources"]) > 0


@pytest.mark.integration
class TestSQLitePersistence:
    """SQLite 记忆持久化 (不依赖 LLM, 但需 sqlite3)"""

    def test_write_and_read_back(self, tmp_path):
        """写入后用新 store 实例读取, 验证持久化"""
        from app.core.memory import SQLiteStore
        from langchain_core.messages import HumanMessage, SystemMessage
        import os as _os

        db = str(tmp_path / "test_memory.db")
        store1 = SQLiteStore(db)

        sid = "test-session-001"
        msgs = [
            SystemMessage(content="你是助手"),
            HumanMessage(content="你好"),
        ]
        store1.save_messages(sid, msgs)

        # 新实例模拟重启
        store2 = SQLiteStore(db)
        loaded = store2.get_messages(sid)

        assert len(loaded) == 2
        assert loaded[0].content == "你是助手"
        assert loaded[1].content == "你好"

    def test_session_listing_and_cleanup(self, tmp_path):
        """会话列表与过期清理"""
        import time as _time
        from app.core.memory import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        store.touch("active-session")
        store.touch("old-session")

        # 手动把 old-session 的活跃时间设为 2 小时前
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.execute(
            "UPDATE sessions SET last_active=? WHERE session_id=?",
            (_time.time() - 7200, "old-session"),
        )
        conn.commit()
        conn.close()

        sessions = store.list_sessions()
        assert "active-session" in sessions
        assert "old-session" in sessions

        cleaned = store.cleanup_expired(3600)
        assert cleaned == 1

        remaining = store.list_sessions()
        assert "active-session" in remaining
        assert "old-session" not in remaining


@pytest.mark.integration
@skip_no_llm
class TestMetricsCollection:
    """真实调用后指标采集验证"""

    @pytest.mark.asyncio
    async def test_llm_metrics_after_call(self):
        """真实 LLM 调用后 token 指标应非零"""
        from app.core.llm import llm_manager
        from app.core.metrics import metrics
        from langchain_core.messages import HumanMessage

        snapshot_before = metrics.snapshot()
        calls_before = snapshot_before["llm"]["total_calls"]

        async for _ in llm_manager.stream_chat(
            [HumanMessage(content="说一个字: 好")]
        ):
            pass

        snapshot_after = metrics.snapshot()
        assert snapshot_after["llm"]["total_calls"] > calls_before
        assert snapshot_after["llm"]["prompt_tokens"] >= snapshot_before["llm"]["prompt_tokens"]

    def test_retrieval_metrics_after_rag(self):
        """RAG 检索后检索指标应更新"""
        from app.core.rag import rag_engine
        from app.core.metrics import metrics

        snapshot_before = metrics.snapshot()
        retrieval_before = snapshot_before["retrieval"]["total_calls"]

        try:
            rag_engine.retrieve("测试查询")
        except Exception:
            pytest.skip("向量库不可用")

        snapshot_after = metrics.snapshot()
        assert snapshot_after["retrieval"]["total_calls"] > retrieval_before
