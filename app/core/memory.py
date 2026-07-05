"""
对话记忆管理模块

实现多会话隔离的对话历史管理:
1. 会话隔离: 每个 session_id 独立维护历史
2. 滑动窗口: 按 token 数限制历史长度, 避免超出上下文窗口
3. 自动过期: 长时间不活跃的会话自动清理
4. 角色管理: 区分 system / human / ai 消息
5. 持久化: 支持 SQLite 存储, 重启不丢失 (生产环境可换 Redis)

存储层抽象 (Strategy 模式):
  MemoryStore (抽象基类)
    ├── InMemoryStore  内存字典 (默认轻量)
    └── SQLiteStore    SQLite 持久化 (重启恢复)
  通过 settings.memory_backend 切换, 上层 MemoryManager 无感知。
"""
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.utils.logger import log
from config import settings

# 角色类型 <-> LangChain 消息类 映射
_ROLE_TO_CLASS = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
}
_CLASS_TO_ROLE = {v: k for k, v in _ROLE_TO_CLASS.items()}


def _serialize_message(msg: BaseMessage) -> dict:
    """LangChain 消息 -> 可存储的 dict"""
    role = _CLASS_TO_ROLE.get(type(msg), "human")
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    extra = {}
    if isinstance(msg, ToolMessage):
        extra["tool_call_id"] = msg.tool_call_id
    return {"role": role, "content": content, "extra": extra}


def _deserialize_message(data: dict) -> BaseMessage:
    """可存储的 dict -> LangChain 消息"""
    cls = _ROLE_TO_CLASS.get(data["role"], HumanMessage)
    if cls is ToolMessage:
        return ToolMessage(
            content=data["content"],
            tool_call_id=data.get("extra", {}).get("tool_call_id", ""),
        )
    return cls(content=data["content"])


class MemoryStore(ABC):
    """存储层抽象基类"""

    @abstractmethod
    def get_messages(self, session_id: str) -> List[BaseMessage]:
        ...

    @abstractmethod
    def save_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        ...

    @abstractmethod
    def touch(self, session_id: str) -> None:
        ...

    @abstractmethod
    def get_last_active(self, session_id: str) -> float:
        ...

    @abstractmethod
    def clear_session(self, session_id: str) -> int:
        ...

    @abstractmethod
    def list_sessions(self) -> List[str]:
        ...

    @abstractmethod
    def cleanup_expired(self, timeout: int) -> int:
        ...


class InMemoryStore(MemoryStore):
    """内存字典存储 (单机演示, 重启丢失)"""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}

    def _ensure(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "last_active": time.time(),
            }
        return self._sessions[session_id]

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        return list(self._ensure(session_id)["messages"])

    def save_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        self._ensure(session_id)["messages"] = list(messages)
        self._ensure(session_id)["last_active"] = time.time()

    def touch(self, session_id: str) -> None:
        self._ensure(session_id)["last_active"] = time.time()

    def get_last_active(self, session_id: str) -> float:
        return self._ensure(session_id)["last_active"]

    def clear_session(self, session_id: str) -> int:
        if session_id in self._sessions:
            count = len(self._sessions[session_id]["messages"])
            del self._sessions[session_id]
            return count
        return 0

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def cleanup_expired(self, timeout: int) -> int:
        now = time.time()
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["last_active"] > timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


class SQLiteStore(MemoryStore):
    """
    SQLite 持久化存储

    表结构:
      sessions(session_id TEXT PRIMARY KEY, last_active REAL)
      messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
               content TEXT, created_at REAL)

    线程安全: 每个 connection 用 threading.local 隔离 (sqlite3 默认禁止跨线程)。
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        log.info(f"SQLite 记忆存储就绪: {db_path}")

    def _conn(self) -> sqlite3.Connection:
        """每个线程独立 connection"""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(self._db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                last_active REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            """
        )
        conn.commit()
        conn.close()

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        rows = self._conn().execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [_deserialize_message({"role": r["role"], "content": r["content"]}) for r in rows]

    def save_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        """整体覆盖: 删除旧消息 + 重新插入 (项目规模可接受)"""
        conn = self._conn()
        now = time.time()
        try:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute(
                "INSERT INTO sessions(session_id, last_active) VALUES(?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET last_active=excluded.last_active",
                (session_id, now),
            )
            for msg in messages:
                data = _serialize_message(msg)
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, created_at) "
                    "VALUES(?, ?, ?, ?)",
                    (session_id, data["role"], data["content"], now),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def touch(self, session_id: str) -> None:
        self._conn().execute(
            "INSERT INTO sessions(session_id, last_active) VALUES(?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET last_active=excluded.last_active",
            (session_id, time.time()),
        )
        self._conn().commit()

    def get_last_active(self, session_id: str) -> float:
        row = self._conn().execute(
            "SELECT last_active FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        return row["last_active"] if row else time.time()

    def clear_session(self, session_id: str) -> int:
        conn = self._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()
        return count

    def list_sessions(self) -> List[str]:
        rows = self._conn().execute("SELECT session_id FROM sessions").fetchall()
        return [r["session_id"] for r in rows]

    def cleanup_expired(self, timeout: int) -> int:
        threshold = time.time() - timeout
        conn = self._conn()
        expired = conn.execute(
            "SELECT session_id FROM sessions WHERE last_active < ?", (threshold,)
        ).fetchall()
        if not expired:
            return 0
        for row in expired:
            conn.execute("DELETE FROM messages WHERE session_id=?", (row["session_id"],))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (row["session_id"],))
        conn.commit()
        return len(expired)


def _create_store() -> MemoryStore:
    """根据配置创建存储后端"""
    backend = settings.memory_backend.lower()
    if backend == "sqlite":
        return SQLiteStore(settings.sqlite_path)
    return InMemoryStore()


class MemoryManager:
    """
    对话记忆管理器

    职责:
    1. 管理多会话对话历史 (委托给 store)
    2. Token 计数与滑动窗口裁剪 (业务逻辑, 与存储无关)
    3. 会话生命周期管理 (创建/获取/过期清理)
    """

    def __init__(self):
        self._store: MemoryStore = _create_store()
        try:
            self._encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        self._cleanup_count = 0
        log.info(f"记忆后端: {type(self._store).__name__}")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _messages_to_text(self, messages: List[BaseMessage]) -> str:
        return " ".join(
            msg.content for msg in messages if isinstance(msg.content, str)
        )

    def _trim_history(self, session_id: str):
        """
        滑动窗口裁剪

        超过 max_history_tokens 时, 分离 system 消息(始终保留),
        从最早的对话消息开始移除, 直到总 token 降到阈值以下。
        """
        messages = self._store.get_messages(session_id)
        total_tokens = self._count_tokens(self._messages_to_text(messages))

        if total_tokens <= settings.max_history_tokens:
            return

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        while other_msgs:
            removed = other_msgs.pop(0)
            total_tokens -= self._count_tokens(str(removed.content))
            if total_tokens <= settings.max_history_tokens:
                break

        self._store.save_messages(session_id, system_msgs + other_msgs)
        log.info(f"会话 {session_id} 历史裁剪完成, 当前 token: {total_tokens}")

    def add_message(self, session_id: str, message: BaseMessage):
        """添加一条消息到会话历史"""
        messages = self._store.get_messages(session_id)
        messages.append(message)
        self._store.save_messages(session_id, messages)
        self._trim_history(session_id)

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """获取会话的完整消息历史"""
        self._store.touch(session_id)
        return self._store.get_messages(session_id)

    def add_user_message(self, session_id: str, content: str):
        self.add_message(session_id, HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str):
        self.add_message(session_id, AIMessage(content=content))

    def set_system_prompt(self, session_id: str, content: str):
        """设置系统提示词 (一个会话只保留一条, 重复设置覆盖)"""
        messages = self._store.get_messages(session_id)
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        messages.insert(0, SystemMessage(content=content))
        self._store.save_messages(session_id, messages)

    def clear_session(self, session_id: str):
        """清空指定会话历史"""
        count = self._store.clear_session(session_id)
        if count:
            log.info(f"清空会话 {session_id}, 移除 {count} 条消息")

    def cleanup_expired(self) -> int:
        """清理过期会话"""
        cleaned = self._store.cleanup_expired(settings.session_timeout)
        if cleaned:
            self._cleanup_count += cleaned
            log.info(f"清理 {cleaned} 个过期会话 (累计清理 {self._cleanup_count})")
        return cleaned

    def get_session_info(self, session_id: str) -> dict:
        """获取会话统计信息"""
        messages = self._store.get_messages(session_id)
        if not messages and session_id not in self._store.list_sessions():
            return {"exists": False, "message_count": 0, "tokens": 0}
        tokens = self._count_tokens(self._messages_to_text(messages))
        return {
            "exists": True,
            "message_count": len(messages),
            "tokens": tokens,
            "last_active": self._store.get_last_active(session_id),
        }

    def list_sessions(self) -> List[str]:
        return self._store.list_sessions()


# 全局单例
memory_manager = MemoryManager()
