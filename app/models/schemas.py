"""
Pydantic 数据模型定义

定义所有 API 请求/响应的数据结构, 提供自动文档生成和类型校验。
"""
from typing import List, Optional

from pydantic import BaseModel, Field


# -------------------------------------------------------
# 对话相关
# -------------------------------------------------------

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., min_length=1, max_length=4000, description="用户消息")
    session_id: str = Field(..., description="会话ID, 用于隔离多轮对话")
    use_rag: bool = Field(True, description="是否启用知识库检索增强")
    use_tools: bool = Field(True, description="是否启用工具调用 (Function Calling)")


class ChatResponse(BaseModel):
    """对话响应 (非流式)"""
    session_id: str
    reply: str
    sources: List[str] = Field(default_factory=list, description="引用来源")
    tool_used: List[str] = Field(default_factory=list, description="使用的工具列表")
    rag_chunks: int = Field(0, description="检索到的知识库文档块数")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    message_count: int
    tokens: int
    last_active: float


# -------------------------------------------------------
# 知识库管理相关
# -------------------------------------------------------

class DocumentIngestResponse(BaseModel):
    """文档入库响应"""
    file: str
    chunks: int
    total: int
    time: float
    message: str = "文档入库成功"


class KnowledgeDocument(BaseModel):
    """知识库文档信息"""
    name: str
    size: str
    type: str


class KnowledgeStats(BaseModel):
    """知识库统计"""
    total_chunks: int
    total_files: int
    documents: List[KnowledgeDocument]


# -------------------------------------------------------
# 工具相关
# -------------------------------------------------------

class ToolInfo(BaseModel):
    """工具信息"""
    name: str
    description: str
    enabled: bool = True


# -------------------------------------------------------
# 通用
# -------------------------------------------------------

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    llm_model: str
    embedding_model: str
    knowledge_chunks: int
    active_sessions: int


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
