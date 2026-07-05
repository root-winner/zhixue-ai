# 🏗️ 架构设计文档

## 一、整体架构

系统采用 **前后端分离 + 分层架构** 设计：

```
表现层 (Streamlit)  →  接口层 (FastAPI)  →  核心引擎层  →  基础设施层
                                              ├── RAG引擎
                                              ├── 记忆管理
                                              └── Agent
```

### 分层职责

| 层级 | 模块 | 职责 |
|------|------|------|
| 表现层 | `frontend/app.py` | 用户交互、流式渲染、知识库管理界面 |
| 接口层 | `app/api/*` | 请求路由、参数校验、响应封装 |
| 核心引擎层 | `app/core/*` | RAG检索、对话记忆、Agent决策 |
| 基础设施层 | `config.py` `app/utils/*` | 配置管理、日志、LLM/Embedding封装 |

---

## 二、核心模块设计

### 2.1 RAG 引擎 (`app/core/rag.py`)

```
文档入库流程:
  load_document()      → 按扩展名选择 Loader 加载
  split_documents()    → RecursiveCharacterTextSplitter 递归切分
  add_to_vectorstore() → Embedding 向量化 + Chroma 持久化 + BM25语料同步

查询检索流程 (支持三种策略, 由 retrieval_strategy 配置):
  retrieve()           → 分发: vector / bm25 / hybrid(默认)
    _retrieve_vector() → Chroma similarity_search_with_relevance_scores
    _retrieve_bm25()   → BM25Okapi 关键词检索
    _retrieve_hybrid() → 两路召回 + RRF 融合
  rerank()             → Cross-Encoder 精排(可选) / 关键词加权(降级)
  build_context()      → 拼接上下文 + 来源标注
```

**混合检索 RRF 融合算法：**
```python
# 向量召回 Top-K + BM25 召回 Top-K, 按排名倒数融合
score(d) = Σ 1 / (rrf_k + rank_i(d))   # rrf_k=60 平滑常数
# 优点: 无需归一化两路异构分数 (向量相似度 vs BM25 TF-IDF 分)
```

**重排序：**
- 优先 Cross-Encoder (bge-reranker-v2-m3)：拼接 (query, doc) 输入 Transformer 输出相关性分数
- 降级方案：`keyword_hits - length_penalty`（关键词命中加分 + 长度惩罚）

### 2.2 对话记忆 (`app/core/memory.py`)

```
存储层抽象 (Strategy 模式):
  MemoryStore (抽象基类)
    ├── InMemoryStore  内存字典 (轻量, 重启丢失)
    └── SQLiteStore    SQLite 持久化 (默认, 重启恢复)
  通过 settings.memory_backend 切换, 上层 MemoryManager 无感知

SQLite 表结构:
  sessions(session_id TEXT PK, last_active REAL)
  messages(id INT PK, session_id TEXT, role TEXT, content TEXT, created_at REAL)

滑动窗口:
  1. tiktoken 计算全部历史消息 Token
  2. 超过 max_history_tokens 时, 移除最早的非 system 消息
  3. system 消息始终保留 (维持人设)
```

### 2.3 Agent (`app/core/agent.py`)

```
Function Calling 流程:
  1. LLM.bind_tools(tools) 绑定工具
  2. ainvoke(messages) → 判断是否需要工具
  3. 若返回 tool_calls:
     a. 逐个执行工具
     b. ToolMessage 加入历史
     c. 再次 astream 生成最终回答
  4. 若无 tool_calls: 直接流式输出
```

### 2.4 流式输出

```
后端: sse-starlette EventSourceResponse
  → async generator yield 事件
  → 事件类型: tool_call / tool_result / token / done / error

前端: Streamlit empty() placeholder
  → 逐 token markdown 渲染 + "▌" 光标
```

### 2.5 可观测性 (`app/core/metrics.py`)

```
MetricsCollector (单例, 线程安全)
  ├── LLM 指标: prompt/completion token (tiktoken 估算), 调用次数
  ├── 检索指标: 策略分布, 命中文档数, 检索耗时, 重排序保留率
  └── 请求指标: 请求数, 状态码分布, 平均耗时

采集点:
  - llm.stream_chat(): 流式结束后估算 token 上报
  - rag.retrieve(): 记录策略/命中数/耗时
  - rag.rerank(): 记录重排序前后数量
  - main.py 中间件: 记录请求耗时/状态码

暴露: GET /api/metrics → JSON 快照
```

---

## 三、数据流

### 对话请求完整流程

```
1. 用户输入 → Streamlit
2. POST /api/chat {message, session_id, use_rag, use_tools}
3. [若 use_rag] RAG引擎.build_context(message)
   → retrieve() 向量检索
   → rerank() 重排序
   → 返回 context + sources
4. memory_manager.set_system_prompt(拼接context)
5. memory_manager.add_user_message()
6. agent_manager.run(messages)
   → [若 use_tools] LLM决策是否调工具
   → 执行工具 → 工具结果入历史
   → astream 逐 token 生成
7. SSE 推送 token → 前端打字机渲染
8. done 事件 → memory_manager.add_ai_message()
```

---

## 四、扩展点

| 扩展方向 | 实现方式 |
|---------|---------|
| 多用户认证 | 加 JWT 中间件 + 用户表 |
| 持久化记忆 | memory 换 Redis 存储 |
| 大规模知识库 | Chroma 换 Milvus + HNSW索引 |
| Reranker 升级 | 接入 bge-reranker-large |
| 多模态 | 加 OCR 模块处理图片文档 |
| 评估监控 | 加 LangSmith / 自建评估集 |
