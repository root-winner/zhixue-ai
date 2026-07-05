"""
对话接口模块

提供:
1. POST /chat - 流式对话 (Server-Sent Events)
2. POST /chat/sync - 非流式对话
3. GET /chat/sessions - 会话列表
4. DELETE /chat/session/{session_id} - 清空会话
"""
import json
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.agent import agent_manager
from app.core.memory import memory_manager
from app.core.rag import rag_engine
from app.models.schemas import ChatRequest, ChatResponse, SessionInfo
from app.utils.logger import log

router = APIRouter(prefix="/chat", tags=["对话"])

# RAG 系统提示词模板
RAG_SYSTEM_PROMPT = """你是一个智能知识库助手「智学AI」。你的任务是基于提供的参考文档和你的知识, 准确、专业地回答用户问题。

回答规范:
1. 优先使用【参考文档】中的信息回答, 确保准确性
2. 如果参考文档中没有相关信息, 如实告知并基于自身知识回答
3. 回答要条理清晰, 适当使用 Markdown 格式
4. 引用文档内容时, 注明来源文件名"""

# 普通对话系统提示词
CHAT_SYSTEM_PROMPT = """你是一个智能助手「智学AI」。请友好、专业、简洁地回答用户的问题。适当使用 Markdown 格式让回答更清晰。"""


@router.post("")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口 (Server-Sent Events)

    返回 SSE 事件流, 每个事件包含:
    - tool_call: 工具调用通知
    - tool_result: 工具执行结果
    - token: 生成的文本片段
    - done: 完成事件
    - error: 错误事件
    """
    session_id = request.session_id
    log.info(f"对话请求: session={session_id}, rag={request.use_rag}, tools={request.use_tools}")
    log.info(f"用户消息: {request.message[:50]}...")

    # 构建系统提示词
    system_prompt = CHAT_SYSTEM_PROMPT
    sources = []
    rag_chunks = 0

    if request.use_rag:
        # RAG 检索
        rag_result = rag_engine.build_context(request.message)
        if rag_result["context"]:
            system_prompt = RAG_SYSTEM_PROMPT + f"\n\n【参考文档】\n{rag_result['context']}"
            sources = rag_result["sources"]
            rag_chunks = rag_result["chunk_count"]
            log.info(f"RAG 检索到 {rag_chunks} 个文档块, 来源: {sources}")

    # 设置系统提示词并添加用户消息
    memory_manager.set_system_prompt(session_id, system_prompt)
    memory_manager.add_user_message(session_id, request.message)

    # 获取完整消息历史
    messages = memory_manager.get_messages(session_id)

    async def event_generator():
        """SSE 事件生成器"""
        full_response = ""
        tools_used = []

        try:
            # Agent 执行
            async for event in agent_manager.run(messages, enable_tools=request.use_tools):
                event_type = event["type"]

                if event_type == "tool_call":
                    tools_used.append(event["name"])
                    yield {"event": "tool_call", "data": json.dumps({
                        "name": event["name"],
                        "args": event["args"],
                    }, ensure_ascii=False)}

                elif event_type == "tool_result":
                    yield {"event": "tool_result", "data": json.dumps({
                        "name": event["name"],
                        "result": event["result"][:500],  # 截断过长结果
                    }, ensure_ascii=False)}

                elif event_type == "token":
                    full_response += event["content"]
                    yield {"event": "token", "data": json.dumps({
                        "content": event["content"],
                    }, ensure_ascii=False)}

                elif event_type == "done":
                    full_response = event.get("full_response", full_response)
                    # 保存 AI 回复到记忆
                    memory_manager.add_ai_message(session_id, full_response)
                    yield {"event": "done", "data": json.dumps({
                        "session_id": session_id,
                        "sources": sources,
                        "tools_used": tools_used,
                        "rag_chunks": rag_chunks,
                        "full_response": full_response,
                    }, ensure_ascii=False)}

                elif event_type == "error":
                    yield {"event": "error", "data": json.dumps({
                        "message": event["message"],
                    }, ensure_ascii=False)}

        except Exception as e:
            log.error(f"流式对话异常: {e}")
            yield {"event": "error", "data": json.dumps({
                "message": str(e),
            }, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@router.post("/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """非流式对话接口 (等待完整回复后返回)"""
    session_id = request.session_id

    # 构建系统提示词
    system_prompt = CHAT_SYSTEM_PROMPT
    sources = []
    rag_chunks = 0

    if request.use_rag:
        rag_result = rag_engine.build_context(request.message)
        if rag_result["context"]:
            system_prompt = RAG_SYSTEM_PROMPT + f"\n\n【参考文档】\n{rag_result['context']}"
            sources = rag_result["sources"]
            rag_chunks = rag_result["chunk_count"]

    memory_manager.set_system_prompt(session_id, system_prompt)
    memory_manager.add_user_message(session_id, request.message)
    messages = memory_manager.get_messages(session_id)

    full_response = ""
    tools_used = []

    async for event in agent_manager.run(messages, enable_tools=request.use_tools):
        if event["type"] == "tool_call":
            tools_used.append(event["name"])
        elif event["type"] == "token":
            full_response += event["content"]
        elif event["type"] == "done":
            full_response = event.get("full_response", full_response)
            break

    memory_manager.add_ai_message(session_id, full_response)

    return ChatResponse(
        session_id=session_id,
        reply=full_response,
        sources=sources,
        tool_used=tools_used,
        rag_chunks=rag_chunks,
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """列出所有活跃会话"""
    sessions = []
    for sid in memory_manager.list_sessions():
        info = memory_manager.get_session_info(sid)
        sessions.append(SessionInfo(
            session_id=sid,
            message_count=info["message_count"],
            tokens=info["tokens"],
            last_active=info["last_active"],
        ))
    return sessions


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清空指定会话历史"""
    if session_id not in memory_manager.list_sessions():
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    memory_manager.clear_session(session_id)
    return {"message": f"会话 {session_id} 已清空"}


@router.post("/sessions/cleanup")
async def cleanup_sessions():
    """清理过期会话"""
    cleaned = memory_manager.cleanup_expired()
    return {"cleaned": cleaned, "message": f"已清理 {cleaned} 个过期会话"}
