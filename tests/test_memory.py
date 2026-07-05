"""
对话记忆管理单元测试

覆盖:
1. 消息添加与获取
2. 系统提示词设置 (覆盖逻辑)
3. Token 滑动窗口裁剪 (核心逻辑)
4. 会话隔离 (多 session 互不干扰)
5. 会话清空
6. 过期会话清理
7. 会话统计信息
"""
import time
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.memory import MemoryManager


@pytest.fixture
def memory():
    """每个测试用独立的 MemoryManager 实例"""
    return MemoryManager()


@pytest.mark.unit
class TestMessageManagement:
    """消息添加与获取"""

    def test_add_and_get_user_message(self, memory):
        """添加并获取用户消息"""
        memory.add_user_message("session-1", "你好")
        messages = memory.get_messages("session-1")

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "你好"

    def test_add_and_get_ai_message(self, memory):
        """添加并获取 AI 消息"""
        memory.add_ai_message("session-1", "你好！有什么可以帮你的？")
        messages = memory.get_messages("session-1")

        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert "你好" in messages[0].content

    def test_multi_turn_conversation(self, memory):
        """多轮对话历史顺序正确"""
        sid = "multi-turn"
        memory.add_user_message(sid, "第一个问题")
        memory.add_ai_message(sid, "第一个回答")
        memory.add_user_message(sid, "第二个问题")
        memory.add_ai_message(sid, "第二个回答")

        messages = memory.get_messages(sid)
        assert len(messages) == 4
        # 验证顺序: human -> ai -> human -> ai
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)
        assert isinstance(messages[3], AIMessage)
        assert messages[0].content == "第一个问题"
        assert messages[3].content == "第二个回答"


@pytest.mark.unit
class TestSystemPrompt:
    """系统提示词管理"""

    def test_set_system_prompt(self, memory):
        """设置系统提示词"""
        memory.set_system_prompt("s1", "你是智学AI助手")
        messages = memory.get_messages("s1")

        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "你是智学AI助手"

    def test_system_prompt_overwrite(self, memory):
        """重复设置 system prompt 应覆盖, 不累积"""
        memory.set_system_prompt("s1", "第一个提示词")
        memory.set_system_prompt("s1", "第二个提示词")
        messages = memory.get_messages("s1")

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "第二个提示词"

    def test_system_prompt_stays_first(self, memory):
        """system 消息始终在最前面"""
        memory.set_system_prompt("s1", "系统提示")
        memory.add_user_message("s1", "用户消息")
        memory.add_ai_message("s1", "AI回复")

        messages = memory.get_messages("s1")
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)


@pytest.mark.unit
class TestSlidingWindow:
    """Token 滑动窗口裁剪 (核心逻辑)"""

    def test_no_trim_under_limit(self, memory):
        """未超限时不应裁剪"""
        sid = "trim-test"
        memory.add_user_message(sid, "短消息")

        messages = memory.get_messages(sid)
        assert len(messages) == 1  # 不应被裁剪

    def test_trim_removes_oldest(self, memory):
        """超过 Token 限制时应裁剪最早消息, 保留 system"""
        sid = "trim-test"
        memory.set_system_prompt(sid, "系统提示词" * 10)  # 较长的 system

        # 添加大量消息触发裁剪
        # max_history_tokens 默认 2000, 每条中文约几十 token
        for i in range(100):
            memory.add_user_message(sid, f"这是第{i}条用户消息，内容较长用于触发裁剪机制测试。")
            memory.add_ai_message(sid, f"这是第{i}条AI回复，内容同样较长用于触发裁剪机制测试。")

        messages = memory.get_messages(sid)

        # system 消息必须保留
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1, "system 消息应始终保留"

        # 消息总数应远少于添加的数量 (200条)
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        assert len(non_system) < 200, "应已裁剪部分历史消息"

        # 最新的消息应保留 (最后添加的)
        assert "第99条AI回复" in non_system[-1].content

    def test_trim_preserves_recent_context(self, memory):
        """裁剪后最近的对话上下文完整"""
        sid = "context-test"
        memory.add_user_message(sid, "最近的用户问题")
        memory.add_ai_message(sid, "最近的AI回答")

        messages = memory.get_messages(sid)
        contents = [m.content for m in messages]

        assert "最近的用户问题" in contents
        assert "最近的AI回答" in contents


@pytest.mark.unit
class TestSessionIsolation:
    """会话隔离"""

    def test_different_sessions_isolated(self, memory):
        """不同 session 的消息互不干扰"""
        memory.add_user_message("session-a", "A的消息")
        memory.add_user_message("session-b", "B的消息")

        msg_a = memory.get_messages("session-a")
        msg_b = memory.get_messages("session-b")

        assert len(msg_a) == 1
        assert len(msg_b) == 1
        assert msg_a[0].content == "A的消息"
        assert msg_b[0].content == "B的消息"

    def test_list_sessions(self, memory):
        """列出所有活跃会话"""
        memory.add_user_message("s1", "msg")
        memory.add_user_message("s2", "msg")
        memory.add_user_message("s3", "msg")

        sessions = memory.list_sessions()
        assert len(sessions) == 3
        assert "s1" in sessions
        assert "s2" in sessions
        assert "s3" in sessions


@pytest.mark.unit
class TestSessionCleanup:
    """会话清理"""

    def test_clear_session(self, memory):
        """清空指定会话"""
        memory.add_user_message("s1", "消息1")
        memory.add_user_message("s1", "消息2")
        assert len(memory.get_messages("s1")) == 2

        memory.clear_session("s1")

        # 清空后获取应返回空列表 (新会话)
        messages = memory.get_messages("s1")
        assert len(messages) == 0

    def test_clear_nonexistent_session(self, memory):
        """清空不存在的会话不报错"""
        memory.clear_session("nonexistent")  # 不应抛异常

    def test_cleanup_expired_sessions(self, memory):
        """清理过期会话"""
        memory.add_user_message("active", "活跃消息")
        memory.add_user_message("expired", "过期消息")

        # 手动修改 expired 会话的最后活跃时间为很久以前
        memory._sessions["expired"]["last_active"] = time.time() - 999999

        cleaned = memory.cleanup_expired()
        assert cleaned == 1
        assert "active" in memory.list_sessions()
        assert "expired" not in memory.list_sessions()


@pytest.mark.unit
class TestSessionInfo:
    """会话统计信息"""

    def test_get_session_info_existing(self, memory):
        """获取已存在会话的信息"""
        memory.add_user_message("s1", "测试消息")

        info = memory.get_session_info("s1")
        assert info["exists"] is True
        assert info["message_count"] == 1
        assert info["tokens"] > 0
        assert info["last_active"] > 0

    def test_get_session_info_nonexistent(self, memory):
        """获取不存在会话的信息"""
        info = memory.get_session_info("nonexistent")
        assert info["exists"] is False
        assert info["message_count"] == 0
        assert info["tokens"] == 0
