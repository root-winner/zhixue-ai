"""
API 接口测试

覆盖:
1. 健康检查
2. 知识库管理接口
3. 工具接口
4. 会话管理接口
5. 对话接口 (mock LLM)
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    from app.main import app
    return TestClient(app)


@pytest.mark.unit
class TestHealthCheck:
    """健康检查"""

    def test_health_check(self, client):
        """GET / 返回系统状态"""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm_model" in data
        assert "knowledge_chunks" in data

    def test_api_root(self, client):
        """GET /api 返回接口列表"""
        resp = client.get("/api")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data
        assert "对话" in data["endpoints"]
        assert "知识库" in data["endpoints"]


@pytest.mark.unit
class TestKnowledgeAPI:
    """知识库管理接口"""

    def test_list_documents(self, client):
        """GET /api/knowledge/documents"""
        resp = client.get("/api/knowledge/documents")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_knowledge_stats(self, client):
        """GET /api/knowledge/stats"""
        resp = client.get("/api/knowledge/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_chunks" in data
        assert "total_files" in data
        assert "documents" in data

    def test_upload_unsupported_format(self, client):
        """上传不支持的文件格式返回 400"""
        resp = client.post(
            "/api/knowledge/upload",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "不支持" in resp.json()["detail"] or "unsupported" in resp.json()["detail"].lower()

    def test_upload_txt_file(self, client, tmp_path):
        """上传 TXT 文件成功入库"""
        # 创建测试文件
        test_content = "这是一个测试文档，用于验证知识库上传功能。"
        resp = client.post(
            "/api/knowledge/upload",
            files={"file": ("test_upload.txt", test_content.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file"] == "test_upload.txt"
        assert data["chunks"] > 0


@pytest.mark.unit
class TestToolsAPI:
    """工具接口"""

    def test_list_tools(self, client):
        """GET /api/tools 返回工具列表"""
        resp = client.get("/api/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert len(tools) >= 2
        tool_names = [t["name"] for t in tools]
        assert "weather_tool" in tool_names
        assert "web_search_tool" in tool_names

    @patch("app.tools.weather.weather_tool.invoke")
    def test_call_weather(self, mock_invoke, client):
        """POST /api/tools/weather 调用天气工具"""
        mock_invoke.return_value = "北京 25°C 晴"
        resp = client.post("/api/tools/weather", json={"input": "北京"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "weather"
        assert "25" in data["result"]

    @patch("app.tools.search.web_search_tool.invoke")
    def test_call_search(self, mock_invoke, client):
        """POST /api/tools/search 调用搜索工具"""
        mock_invoke.return_value = "搜索结果: 测试内容"
        resp = client.post("/api/tools/search", json={"input": "测试"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "search"


@pytest.mark.unit
class TestSessionAPI:
    """会话管理接口"""

    def test_list_sessions_empty(self, client):
        """GET /api/chat/sessions 初始为空"""
        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_clear_nonexistent_session(self, client):
        """DELETE 不存在的会话返回 404"""
        resp = client.delete("/api/chat/session/nonexistent-session")
        assert resp.status_code == 404

    def test_cleanup_sessions(self, client):
        """POST /api/chat/sessions/cleanup 清理过期会话"""
        resp = client.post("/api/chat/sessions/cleanup")
        assert resp.status_code == 200
        assert "cleaned" in resp.json()


@pytest.mark.unit
class TestChatAPI:
    """对话接口 (mock LLM)"""

    @patch("app.core.agent.agent_manager.run")
    def test_chat_sync_success(self, mock_run, client):
        """POST /api/chat/sync 同步对话 (mock LLM)"""
        async def mock_async_run(messages, enable_tools=True):
            yield {"type": "token", "content": "你好"}
            yield {"type": "token", "content": "！我是智学AI。"}
            yield {"type": "done", "full_response": "你好！我是智学AI。"}

        mock_run.side_effect = mock_async_run

        resp = client.post("/api/chat/sync", json={
            "message": "你好",
            "session_id": "test-session",
            "use_rag": False,
            "use_tools": False,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-session"
        assert "智学AI" in data["reply"]

    def test_chat_invalid_request(self, client):
        """POST /api/chat/sync 缺少必填字段返回 422"""
        resp = client.post("/api/chat/sync", json={
            "message": "测试",
            # 缺少 session_id
        })
        assert resp.status_code == 422

    def test_chat_empty_message(self, client):
        """POST /api/chat/sync 空消息返回 422"""
        resp = client.post("/api/chat/sync", json={
            "message": "",
            "session_id": "test",
        })
        assert resp.status_code == 422

    @patch("app.core.agent.agent_manager.run")
    def test_chat_sync_with_rag(self, mock_run, client):
        """RAG 模式下对话包含来源"""
        async def mock_async_run(messages, enable_tools=True):
            yield {"type": "done", "full_response": "基于文档的回答"}

        mock_run.side_effect = mock_async_run

        # mock RAG 返回结果
        with patch("app.core.rag.rag_engine.build_context") as mock_context:
            mock_context.return_value = {
                "context": "参考文档内容",
                "sources": ["test.pdf"],
                "chunk_count": 2,
            }

            resp = client.post("/api/chat/sync", json={
                "message": "文档里说了什么",
                "session_id": "rag-test",
                "use_rag": True,
                "use_tools": False,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert "test.pdf" in data["sources"]
            assert data["rag_chunks"] == 2

    @patch("app.core.agent.agent_manager.run")
    def test_chat_stream_sse(self, mock_run, client):
        """POST /api/chat 流式对话返回 SSE"""
        async def mock_async_run(messages, enable_tools=True):
            yield {"type": "token", "content": "你"}
            yield {"type": "token", "content": "好"}
            yield {"type": "done", "full_response": "你好"}

        mock_run.side_effect = mock_async_run

        resp = client.post("/api/chat", json={
            "message": "你好",
            "session_id": "stream-test",
            "use_rag": False,
            "use_tools": False,
        })

        assert resp.status_code == 200
        # SSE 响应应包含 data: 前缀的事件
        text = resp.text
        assert "data:" in text or "event:" in text
