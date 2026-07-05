"""
智学AI - Streamlit 前端界面

启动方式:
  streamlit run frontend/app.py

功能:
1. 流式对话 (打字机效果)
2. 知识库管理 (上传/列表/删除)
3. 会话管理 (新建/切换/清空)
4. 工具开关 (RAG / Function Calling)
"""
import json
import os
import uuid

import httpx
import streamlit as st

# ============================================================
# 配置
# ============================================================
# 支持环境变量覆盖, Docker 部署时通过 API_BASE 注入
API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api")

# 后端根地址 (去掉 /api 后缀)
API_ROOT = API_BASE.replace("/api", "") if API_BASE.endswith("/api") else API_BASE

st.set_page_config(
    page_title="智学AI - 智能知识库对话系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 会话状态初始化
# ============================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True
if "use_tools" not in st.session_state:
    st.session_state.use_tools = True


# ============================================================
# API 调用函数
# ============================================================
def stream_chat(message: str):
    """流式对话, 返回事件生成器"""
    payload = {
        "message": message,
        "session_id": st.session_state.session_id,
        "use_rag": st.session_state.use_rag,
        "use_tools": st.session_state.use_tools,
    }
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{API_BASE}/chat", json=payload) as resp:
            current_event = "message"
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            yield {"event": current_event, "data": data_str}
                        except Exception:
                            continue


def upload_document(file):
    """上传文档到知识库"""
    files = {"file": (file.name, file.getvalue(), "application/octet-stream")}
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{API_BASE}/knowledge/upload", files=files)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}


def get_knowledge_stats():
    """获取知识库统计"""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{API_BASE}/knowledge/stats")
            return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def clear_knowledge():
    """清空知识库"""
    with httpx.Client(timeout=30) as client:
        resp = client.delete(f"{API_BASE}/knowledge/all")
        return resp.json()


def get_system_status():
    """获取系统状态"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{API_ROOT}/")
            return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_tools():
    """获取工具列表"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{API_BASE}/tools")
            return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🤖 智学AI")
    st.caption("智能知识库对话系统")
    st.divider()

    # 系统状态
    st.subheader("📡 系统状态")
    status = get_system_status()
    if status:
        st.success("🟢 服务在线")
        st.text(f"模型: {status.get('llm_model', 'N/A')}")
        st.text(f"知识块: {status.get('knowledge_chunks', 0)}")
        st.text(f"活跃会话: {status.get('active_sessions', 0)}")
    else:
        st.error("🔴 服务离线")
        st.caption("请先启动后端: uvicorn app.main:app --reload")

    st.divider()

    # 会话管理
    st.subheader("💬 会话管理")
    st.text(f"当前会话: {st.session_state.session_id}")

    if st.button("🔄 新建会话", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 功能开关
    st.subheader("⚙️ 功能开关")
    st.session_state.use_rag = st.toggle(
        "📚 知识库问答 (RAG)",
        value=st.session_state.use_rag,
        help="启用后将从知识库检索相关文档增强回答",
    )
    st.session_state.use_tools = st.toggle(
        "🔧 工具调用 (Agent)",
        value=st.session_state.use_tools,
        help="启用后AI可调用天气查询、网络搜索等工具",
    )

    st.divider()

    # 知识库管理
    st.subheader("📚 知识库")
    stats = get_knowledge_stats()
    if stats:
        st.metric("已索引文档块", stats.get("total_chunks", 0))
        st.metric("文档数量", stats.get("total_files", 0))

    uploaded = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "docx", "md"],
        help="支持 PDF / Word / Markdown / TXT",
    )
    if uploaded is not None:
        with st.spinner("文档入库中..."):
            result = upload_document(uploaded)
        if "error" in result:
            st.error(f"上传失败: {result['error']}")
        else:
            st.success(
                f"✅ {result['file']} 入库成功\n\n"
                f"切分: {result['chunks']} 块 | 总计: {result['total']} 块 | 耗时: {result['time']}s"
            )
            st.rerun()

    if st.button("⚠️ 清空知识库", use_container_width=True):
        if st.session_state.get("confirm_clear_kb"):
            result = clear_knowledge()
            st.warning(f"已清空: {result.get('deleted_chunks', 0)} 块, {result.get('deleted_files', 0)} 文件")
            st.session_state.confirm_clear_kb = False
            st.rerun()
        else:
            st.session_state.confirm_clear_kb = True
            st.warning("再次点击确认清空")
            st.rerun()

    st.divider()

    # 工具列表
    st.subheader("🔧 可用工具")
    tools_list = get_tools()
    for t in tools_list:
        with st.expander(t["name"]):
            st.caption(t["description"])

    st.divider()
    st.caption("💡 初级AI应用开发项目演示")


# ============================================================
# 主对话区
# ============================================================
st.title("💬 智学AI 对话")

# 功能状态标签
cols = st.columns(3)
with cols[0]:
    if st.session_state.use_rag:
        st.info("📚 RAG: 开启")
    else:
        st.caption("📚 RAG: 关闭")
with cols[1]:
    if st.session_state.use_tools:
        st.info("🔧 工具: 开启")
    else:
        st.caption("🔧 工具: 关闭")
with cols[2]:
    st.caption(f"🆔 {st.session_state.session_id}")

st.divider()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📎 来源: " + " | ".join(msg["sources"]))
        if msg.get("tools"):
            st.caption("🔧 工具: " + " | ".join(msg["tools"]))

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            full_response = ""
            sources = []
            tools_used = []

            # 占位区, 用于流式更新
            response_placeholder = st.empty()
            tool_info_placeholder = st.empty()
            source_placeholder = st.empty()

            for event in stream_chat(prompt):
                if event.get("event") == "tool_call":
                    data = json.loads(event["data"])
                    tools_used.append(data["name"])
                    tool_info_placeholder.info(f"🔧 调用工具: {data['name']}")

                elif event.get("event") == "tool_result":
                    tool_info_placeholder.empty()

                elif event.get("event") == "token":
                    data = json.loads(event["data"])
                    full_response += data["content"]
                    response_placeholder.markdown(full_response + "▌")

                elif event.get("event") == "done":
                    data = json.loads(event["data"])
                    full_response = data.get("full_response", full_response)
                    sources = data.get("sources", [])
                    tools_used = data.get("tools_used", tools_used)
                    response_placeholder.markdown(full_response)

                    if sources:
                        source_placeholder.caption("📎 来源: " + " | ".join(sources))
                    if tools_used:
                        tool_info_placeholder.caption("🔧 工具: " + " | ".join(tools_used))

                elif event.get("event") == "error":
                    data = json.loads(event["data"])
                    response_placeholder.error(f"错误: {data.get('message', '未知错误')}")
                    full_response = f"[错误] {data.get('message', '')}"

        # 保存到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "tools": tools_used,
        })
