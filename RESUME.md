# 📄 简历项目描述

> 以下提供 **精简版**（直接粘贴简历）和 **详细版**（面试讲述用）两个版本，以及 **面试高频问题准备**。

---

## 一、简历项目经历（精简版，直接粘贴）

### 版本 A：详细版（推荐，约 250 字）

**智学AI — 基于 RAG 的智能知识库对话系统** | 个人项目

**技术栈：** Python · FastAPI · LangChain · ChromaDB · Streamlit · OpenAI 兼容接口 · SQLite

- 基于 RAG（检索增强生成）技术构建智能知识库问答系统，支持上传 PDF/Word/Markdown 文档；检索层采用 **BM25 关键词检索 + 向量语义检索的混合方案**，用 **RRF（Reciprocal Rank Fusion）融合两路召回结果**，可选接入 bge-reranker Cross-Encoder 精排，关键词精准查询与语义泛化查询的准确率均优于单一检索
- 设计多会话隔离的对话记忆管理模块，基于 tiktoken 实现 Token 级滑动窗口裁剪；采用 **Strategy 模式抽象存储层**，支持内存与 **SQLite 持久化**两种后端，服务重启后对话历史可恢复，生产环境可平滑替换 Redis
- 实现 Function Calling Agent，LLM 自主决策调用天气查询、网络搜索等工具，将工具结果融入上下文生成回答
- 采用 FastAPI + SSE 实现流式输出，前端 Streamlit 实现打字机效果；接入 **可观测性模块**，采集 LLM Token 用量、检索命中率/耗时、请求耗时，通过 `/api/metrics` 端点暴露监控指标
- 编写 pytest 测试 70+ 用例（单元测试 mock 隔离外部依赖 + `pytest.mark.integration` 端到端集成测试），使用 Docker + docker-compose 实现容器化部署

---

### 版本 B：超精简版（简历空间有限时用，约 120 字）

**智学AI — 基于 RAG 的智能知识库对话系统** | Python · FastAPI · LangChain · ChromaDB

- 基于 RAG 技术构建知识库问答系统，支持 PDF/Word 文档上传，**BM25+向量混合检索（RRF 融合）+ 可选 Cross-Encoder 精排**，回答带来源引用
- 实现多会话隔离的对话记忆管理，Token 级滑动窗口裁剪，**SQLite 持久化**重启可恢复（Strategy 模式抽象存储层）
- 集成 Function Calling Agent，LLM 自主调用天气/搜索工具；FastAPI SSE 流式输出；**可观测性模块**采集 Token/检索/请求指标
- pytest 70+ 用例（单元+集成测试）+ Docker 容器化部署

---

## 二、面试讲述版（详细 STAR 结构）

> 面试时按此逻辑展开，控制在 3-5 分钟。

**【Situation 背景】**
AI 应用开发中，大模型存在知识时效性差、领域知识不足、容易幻觉等问题。企业需要让 AI 基于私有文档精准回答，同时能调用外部工具获取实时信息。

**【Task 任务】**
独立设计并开发一个智能知识库对话系统，要求：
1. 支持上传多种格式文档构建专属知识库
2. 基于知识库精准问答，回答可溯源
3. 支持多轮对话上下文记忆
4. 能自主调用外部工具（天气、搜索）
5. 流式输出，体验流畅

**【Action 行动】**

**技术选型：**
- 后端选 FastAPI（异步高性能 + 自动 API 文档）
- AI 框架选 LangChain（RAG 生态成熟，面试认可度高）
- 向量库选 Chroma（嵌入式零运维，适合项目演示）
- 前端选 Streamlit（快速出交互界面）

**RAG 流程设计：**
1. 文档加载：用 LangChain 的 PyPDFLoader / Docx2txtLoader 等加载器，按扩展名自动选择
2. 文本切分：用 RecursiveCharacterTextSplitter，分隔符按 `\n\n` → `\n` → 句号 → 空格 优先级递归切分，chunk_size=500，overlap=50，保证语义完整性
3. 向量化：调用 OpenAI 兼容 Embedding 接口（也支持本地 bge 模型离线运行）
4. **混合检索**：这是检索层的核心。纯向量检索对精确关键词匹配不敏感，纯 BM25 又缺乏语义理解。我用混合方案——向量召回 Top-K + BM25 召回 Top-K，再用 **RRF（Reciprocal Rank Fusion）** 融合：`score(d) = Σ 1/(k + rank(d))`，k=60 平滑。RRF 的优点是无需归一化不同检索器的异构分数，对两路排序都鲁棒。文档入库时同步维护 BM25 语料索引（rank_bm25），与 Chroma 向量库保持一致
5. **重排序**：可选接入 bge-reranker-v2-m3 Cross-Encoder 精排——它把 (query, doc) 拼接输入 Transformer 输出相关性分数，比 Bi-Encoder 向量更精准；未安装时降级为关键词加权重排序（命中数加分+长度惩罚）。通过 `use_reranker` 配置切换
6. 生成：将重排序后的文档块拼接进 system prompt，LLM 基于参考文档生成带来源引用的回答

**对话记忆设计：**
- 用 Strategy 模式抽象存储层：`MemoryStore` 基类 + `InMemoryStore` / `SQLiteStore` 两个实现，通过配置切换，上层 `MemoryManager` 无感知
- SQLite 持久化：sessions 和 messages 两张表，服务重启后从 SQLite 恢复对话历史，生产环境可平滑替换 Redis
- 用 tiktoken 精确计算历史消息 Token 数，超过 max_history_tokens（2000）时分离出 system 消息（始终保留），从最早对话消息裁剪
- 后台定时清理超时会话（默认 1 小时）

**Function Calling 设计：**
- 用 LangChain 的 @tool 装饰器定义天气（wttr.in 免费API）和搜索（DuckDuckGo）工具
- LLM 绑定工具后，ainvoke 时返回 tool_calls，Agent 执行工具拿到结果，再走一轮 LLM 生成最终回答
- 支持前端开关，用户可控制是否启用工具

**流式输出：**
- 后端用 sse-starlette 实现 SSE，Agent 的 token 逐个推送
- 前端 Streamlit 用 empty placeholder 实现 Markdown 逐字渲染打字机效果

**可观测性：**
- 自建 MetricsCollector 单例，采集三类指标：LLM 调用（tiktoken 估算 prompt/completion token）、检索质量（策略分布、命中数、耗时、重排序保留率）、HTTP 请求（数量、状态码、平均耗时）
- 流式模式下 API 不返回 usage，用 tiktoken 估算 token 数上报
- FastAPI 中间件统一记录请求日志，`/api/metrics` 端点暴露指标快照

**测试与部署：**
- pytest 70+ 用例：单元测试用 unittest.mock 隔离 LLM/向量库/HTTP，不依赖真实服务即可跑；`pytest.mark.integration` 标记的集成测试做端到端验证（真实 LLM 对话、RAG 全链路、SQLite 持久化恢复）
- Docker + docker-compose 编排前后端，Volume 持久化向量库和 SQLite

**【Result 结果】**
- 完成完整可运行项目，覆盖文档入库→混合检索→重排序→对话→工具调用全链路
- 混合检索 + Reranker 让关键词精准查询与语义泛化查询的准确率均优于单一检索
- 对话记忆 SQLite 持久化，服务重启不丢历史
- 可观测性模块提供 Token/检索/请求维度的监控，具备生产可观测基础

---

## 三、面试高频问题准备

> 以下是面试官常问的深挖问题，提前准备。

### Q1：RAG 的完整流程是怎样的？
**答：** 离线索引阶段：文档加载 → 递归切分 → Embedding 向量化 → 存入 Chroma。在线查询阶段：用户问题 → 向量化 → 相似度检索 Top-K → 重排序 → 拼接进 prompt → LLM 生成回答。关键在于切分策略和检索质量。

### Q2：为什么选 Chroma 而不是 Pinecone / Milvus？
**答：** Chroma 是嵌入式向量库，无需部署服务，数据本地持久化，适合项目和中小规模场景。Pinecone 是托管云服务需要 API Key 且收费；Milvus 适合大规模生产但部署重。项目阶段 Chroma 够用且零运维，简历也能体现技术选型的权衡思考。

### Q3：文档切分为什么用 500 字符？overlap 50？
**答：** chunk_size 太小语义不完整，太大检索精度下降且浪费 Token。500 字符对中文约 250-300 字，能覆盖一个完整段落。overlap 50 防止切分时把相关内容截断在两个块之间，保证边界内容的连续性。实际要根据文档类型调参。

### Q4：你的检索和重排序是怎么做的？
**答：** 检索层我用的是**混合检索**：BM25 关键词检索 + 向量语义检索，各路召回 Top-K，再用 **RRF（Reciprocal Rank Fusion）** 融合，公式 `score(d)=Σ 1/(k+rank(d))`，k=60 平滑。RRF 的好处是不用归一化两路异构分数。重排序层可选接入 bge-reranker-v2-m3 Cross-Encoder 精排，它把 query 和 doc 拼接送入 Transformer 输出相关性分数，比 Bi-Encoder 向量更准但更慢，适合对召回结果二次精排；没装时降级为关键词加权重排序。这是个可配置的 trade-off。

### Q5：对话记忆的滑动窗口怎么实现的？记忆怎么存储？
**答：** 用 tiktoken 精确计算历史消息 Token 数。超过阈值时分离 system 消息（始终保留），其余对话消息从最早的开始移除。存储层用 **Strategy 模式**抽象出 `MemoryStore` 基类，实现了 `InMemoryStore` 和 `SQLiteStore` 两个后端，通过配置切换。SQLiteStore 用 sessions 和 messages 两张表持久化，服务重启后从 SQLite 恢复对话历史。上层 `MemoryManager` 只依赖抽象接口，生产环境换成 Redis 只需新增一个 `RedisStore` 实现类，业务代码零改动。

### Q6：Function Calling 和 ReAct Agent 有什么区别？
**答：** Function Calling 是模型原生支持的函数调用能力，LLM 直接输出结构化的工具调用参数，可靠性强。ReAct 是通过 prompt 工程让模型输出 Thought-Action-Observation 的推理链，更灵活但依赖提示词质量、容易出错。我用的是 Function Calling，因为更稳定。

### Q7：如果知识库很大（百万级文档），你的方案有什么瓶颈？
**答：** 1）Chroma 单机内存有限，需换 Milvus/Qdrant 分布式向量库；2）检索速度需加 ANN 索引（HNSW）；3）可加一级关键词检索（BM25）做初筛再向量精排；4）Embedding 可批量预处理；5）可引入缓存层缓存高频问题。

### Q8：流式输出为什么要用 SSE 而不是 WebSocket？
**答：** 对话场景是单向服务器推送，SSE 天然适合且基于 HTTP 无需额外协议升级，浏览器原生支持，实现简单。WebSocket 适合双向实时通信如聊天室。SSE 对流式 LLM 输出是最佳选择。

### Q9：项目有什么不足 / 可以改进的地方？
**答：** 已经做了的：混合检索 + 可选 Reranker、SQLite 记忆持久化、可观测性指标采集。还可以继续改进的方向：1）多模态支持（图片文档 OCR）；2）用户认证和权限管理；3）对话质量评估和 LangSmith 全链路追踪；4）高并发下 SQLite 写锁瓶颈可换 Redis Cluster；5）检索缓存层缓存高频问题。这些是后续迭代方向。

### Q10：你在项目中遇到的最大困难是什么？
**答：** （可根据自己理解调整）混合检索的融合策略。一开始两路检索结果分数不在一个量级——向量是 0~1 的相似度，BM25 是无上界的 TF-IDF 分数，直接加权效果很差。研究后选了 RRF 融合，只用排名倒数求和，绕开了分数归一化难题。但 RRF 的 k 参数（平滑常数）要调——太小则头部结果垄断，太大则区分度不够。最终 k=60 是参考学界常用的经验值，效果平衡。这个过程让我理解了检索质量优化是个系统工程，召回和融合都要调。

### Q11：你的项目有测试吗？怎么做的？
**答：** 有。用 pytest 写了 63 个单元测试用例，覆盖四个模块：记忆管理（测试滑动窗口裁剪、会话隔离、过期清理）、RAG 引擎（测试关键词提取、重排序逻辑、文档切分）、工具调用（mock 掉 httpx 测试天气和搜索工具的各类场景）、API 接口（用 FastAPI TestClient + mock LLM 测试完整请求链路）。测试中通过 unittest.mock 隔离外部依赖，所以单元测试不需要真实 LLM API 也能跑。这是工程化能力的体现。

### Q12：为什么要用 Docker？docker-compose 做了什么？
**答：** Docker 解决环境一致性问题——避免"在我电脑上能跑"的问题。项目用 docker-compose 编排了两个服务：后端 FastAPI 和前端 Streamlit。后端 Dockerfile 基于 python:3.11-slim，分步安装依赖和复制代码利用层缓存；前端独立容器只装 streamlit 和 httpx。通过 Volume 挂载持久化向量数据库和知识库文档，重启不丢数据。后端有 healthcheck，前端 depends_on 等后端健康后才启动，保证启动顺序。

### Q13：BM25 和向量检索各有什么优劣？为什么要混合？
**答：** 向量检索（Bi-Encoder）把 query 和 doc 分别编码成向量算相似度，擅长语义泛化——"怎么部署"能匹配到"安装步骤"，但对生僻的精确关键词（如产品型号、专有名词）反而不敏感，因为 Embedding 把它们模糊化了。BM25 基于词频统计，对精确关键词匹配强，但理解不了同义和语义。混合两者互补：向量管"意思相近"，BM25 管"字面命中"，RRF 融合排名避开分数不可比的问题。这是生产环境 RAG 检索的主流做法。

### Q14：记忆持久化为什么用 SQLite 不直接用 Redis？
**答：** 项目阶段选 SQLite 是因为零运维、单文件、标准库自带，适合演示和中小规模。Redis 性能更好但需要额外部署服务。更重要的是我用了 Strategy 模式抽象存储层——`MemoryStore` 基类定义接口，`InMemoryStore`/`SQLiteStore` 是两个实现，切换后端只改配置不改业务代码。生产环境换 Redis 只需新增一个 `RedisStore` 实现类。这个设计体现的是"面向接口编程"的解耦思维，面试官比较看重。

### Q15：可观测性采集了什么？流式输出怎么统计 token？
**答：** 采集三类指标：LLM 调用（prompt/completion token、调用次数）、检索质量（策略分布、命中文档数、检索耗时、重排序保留率）、HTTP 请求（数量、状态码、平均耗时）。难点是流式输出——SSE 逐 token 推送，OpenAI 的流式接口不返回 usage 字段。我的做法是用 tiktoken 估算：输入侧算 messages 的 token 数，输出侧把流式累积的完整文本再 encode 一次算 token 数，上报给 MetricsCollector。虽然不是 API 精确值，但偏差很小，够用于成本监控。指标通过 `/api/metrics` 端点暴露，生产环境可对接 Prometheus + Grafana。

### Q16：为什么前端用 Streamlit 而不是 React/Vue？
**答：** 这是技术选型的 trade-off。Streamlit 的优势是几行 Python 就能出交互界面，对 AI 应用快速验证想法、做演示非常高效——项目重点是后端 RAG/Agent 能力，前端不是核心考察点。局限是定制性差、不适合复杂交互和大规模用户。所以我的定位很明确：**Streamlit 用于 MVP 快速验证和面试演示**，如果做生产级产品，前端会换成 React + TypeScript（前后端分离，FastAPI 提供 REST/SSE 接口），这样能支持复杂交互、组件化开发和更好的性能。技术选型要匹配场景，不是为了用而用。

---

## 四、技术关键词（简历技能栏参考）

```
大模型应用开发: RAG / 混合检索(BM25+向量) / RRF融合 / Cross-Encoder重排序 / Function Calling / Agent / Prompt Engineering
框架工具: LangChain / FastAPI / Streamlit / ChromaDB / SQLite
模型接入: OpenAI API / 通义千问 / 智谱GLM / 本地Embedding(bge) / bge-reranker
工程能力: SSE流式输出 / 异步编程 / Pydantic数据校验 / Strategy模式 / 可观测性(指标采集)
测试部署: pytest(单元+集成测试) / Mock隔离 / Docker容器化 / docker-compose编排
文档处理: PDF/Word/Markdown解析 / 递归文本切分 / tiktoken计数
```
