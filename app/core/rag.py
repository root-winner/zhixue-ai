"""
RAG (检索增强生成) 核心引擎

完整流程:
  离线: 文档加载 → 文本切分 → 向量化 → 存入 Chroma (同步维护 BM25 语料)
  在线: 用户提问 → 混合检索(向量+BM25, RRF融合) → 重排序(可选Cross-Encoder) → 构建提示词 → LLM 生成

关键设计:
1. 多格式文档加载 (PDF / Word / Markdown / TXT)
2. 递归字符切分, 保留语义完整性
3. 混合检索: 向量语义检索 + BM25 关键词检索, 用 RRF 融合两路结果
4. 重排序: 可选 Cross-Encoder (bge-reranker) 精排, 未安装时降级为关键词加权重排序
5. 带来源引用的回答生成
"""
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.embeddings import embedding_manager
from app.utils.logger import log
from config import settings

# 可选依赖: BM25 关键词检索
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# 可选依赖: Cross-Encoder 重排序
try:
    from FlagEmbedding import FlagReranker
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False


# 文件扩展名 -> Loader 映射
LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}

# 简单停用词表
_STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "什么", "怎么", "可以", "请", "问", "吗",
    "呢", "吧", "the", "a", "an", "is", "are", "of", "to", "in", "and", "or",
}


def _tokenize(text: str) -> List[str]:
    """
    轻量分词: 英文按单词, 中文按单字。

    无需 jieba, 适合项目演示; 生产环境可换 jieba/HanLP 提升中文召回。
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    # 中文逐字切分 (BM25 对短 token 也能工作)
    tokens += [ch for ch in re.findall(r"[\u4e00-\u9fff]", text)]
    return tokens


class RAGEngine:
    """
    RAG 引擎

    职责:
    1. 文档加载与切分
    2. 向量化与持久化存储 (Chroma) + BM25 语料同步维护
    3. 混合检索 (向量 + BM25, RRF 融合)
    4. 重排序 (Cross-Encoder 精排 / 关键词加权降级)
    5. 上下文构建
    """

    def __init__(self):
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )
        self._vectorstore = None
        # BM25 索引 (与向量库文档保持同步)
        self._bm25: Optional["BM25Okapi"] = None
        self._bm25_docs: List[Document] = []
        # Cross-Encoder 重排序器 (懒加载)
        self._reranker = None
        self._init_vectorstore()
        self._rebuild_bm25_index()

    def _init_vectorstore(self):
        """初始化 Chroma 向量数据库"""
        from langchain_chroma import Chroma

        log.info(f"初始化向量数据库: {settings.chroma_path}")
        self._vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding_manager.embeddings,
            persist_directory=str(settings.chroma_path),
        )
        log.success(f"向量数据库就绪, 当前文档块数: {self._count_chunks()}")

    def _count_chunks(self) -> int:
        """统计当前向量库中的文档块数量"""
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0

    # -------------------------------------------------------
    # BM25 索引维护
    # -------------------------------------------------------

    def _rebuild_bm25_index(self):
        """从向量库重建 BM25 语料索引 (启动时 / 文档变更后调用)"""
        if not HAS_BM25:
            log.warning("未安装 rank_bm25, 混合检索降级为纯向量检索")
            return

        try:
            data = self._vectorstore._collection.get(
                include=["documents", "metadatas"]
            )
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(data.get("documents", []), data.get("metadatas", []))
            ]
            self._bm25_docs = docs
            if docs:
                corpus = [_tokenize(d.page_content) for d in docs]
                self._bm25 = BM25Okapi(corpus)
                log.info(f"BM25 索引重建完成, 语料 {len(docs)} 条")
            else:
                self._bm25 = None
                log.info("BM25 索引为空 (知识库无文档)")
        except Exception as e:
            log.warning(f"BM25 索引重建失败: {e}, 混合检索将降级")
            self._bm25 = None
            self._bm25_docs = []

    # -------------------------------------------------------
    # 文档处理流程
    # -------------------------------------------------------

    def load_document(self, file_path: str) -> List[Document]:
        """加载文档, 自动根据扩展名选择 Loader"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            raise ValueError(f"不支持的文件格式: {ext} (支持: {list(LOADER_MAP.keys())})")

        log.info(f"加载文档: {path.name} (类型: {ext})")
        loader = loader_cls(str(path))
        documents = loader.load()

        for doc in documents:
            doc.metadata.update({
                "source_file": path.name,
                "file_type": ext,
                "doc_id": str(uuid.uuid4())[:8],
            })

        log.info(f"文档加载完成: {path.name}, 共 {len(documents)} 页/段")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """递归切分文档为语义块"""
        chunks = self._text_splitter.split_documents(documents)
        log.info(f"文档切分完成: {len(documents)} 段 -> {len(chunks)} 个语义块")
        return chunks

    def add_to_vectorstore(self, chunks: List[Document]) -> int:
        """将文档块向量化并存入 Chroma, 同步更新 BM25 索引"""
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        self._vectorstore.add_documents(documents=chunks, ids=ids)
        log.info(f"向量化完成, 新增 {len(chunks)} 个文档块")

        # 同步重建 BM25 索引 (rank_bm25 不支持增量, 整体重建)
        self._rebuild_bm25_index()
        return len(chunks)

    def ingest_file(self, file_path: str) -> dict:
        """完整文档入库流程: 加载 → 切分 → 向量化 → 存储"""
        start = time.time()
        file_name = Path(file_path).name

        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        added = self.add_to_vectorstore(chunks)

        elapsed = round(time.time() - start, 2)
        total = self._count_chunks()
        log.success(f"文档入库完成: {file_name}, 耗时 {elapsed}s, 总块数 {total}")
        return {"file": file_name, "chunks": added, "total": total, "time": elapsed}

    # -------------------------------------------------------
    # 检索流程
    # -------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        检索: 根据 retrieval_strategy 配置分发
        - vector: 纯向量语义检索
        - bm25:   纯 BM25 关键词检索
        - hybrid: 向量 + BM25 混合检索, RRF 融合 (默认, 生产做法)
        """
        from app.core.metrics import metrics

        strategy = settings.retrieval_strategy.lower()
        top_k = top_k or settings.retrieval_top_k
        start = time.time()

        # 知识库为空时跳过检索, 避免无意义的 Embedding API 调用
        if self._count_chunks() == 0:
            log.info("知识库为空, 跳过检索")
            return []

        if strategy == "vector":
            results = self._retrieve_vector(query, top_k)
        elif strategy == "bm25":
            results = self._retrieve_bm25(query, top_k)
        else:
            results = self._retrieve_hybrid(query)
            strategy = "hybrid"

        metrics.record_retrieval(strategy, len(results), time.time() - start)
        return results

    def _retrieve_vector(self, query: str, top_k: int) -> List[Document]:
        """纯向量相似度检索"""
        docs = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        filtered = [(doc, score) for doc, score in docs if score > 0.3]
        if not filtered:
            filtered = docs
        log.info(f"[向量检索] query='{query[:30]}...', 命中 {len(filtered)} 条")
        return [doc for doc, _ in filtered]

    def _retrieve_bm25(self, query: str, top_k: int) -> List[Document]:
        """纯 BM25 关键词检索"""
        if not self._bm25:
            log.warning("BM25 索引不可用, 回退向量检索")
            return self._retrieve_vector(query, top_k)

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        # 取 Top-K
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results = [self._bm25_docs[i] for i in ranked_idx if scores[i] > 0]
        log.info(f"[BM25检索] query='{query[:30]}...', 命中 {len(results)} 条")
        return results

    def _retrieve_hybrid(self, query: str) -> List[Document]:
        """
        混合检索: 向量召回 + BM25 召回, RRF 融合

        RRF (Reciprocal Rank Fusion):
          score(d) = Σ 1 / (k + rank_i(d))
        对每路检索结果按排名倒数打分求和, k 为平滑常数(默认60)。
        优点: 无需归一化不同检索器的分数, 对异构排序鲁棒。
        """
        k = settings.rrf_k
        v_top = settings.hybrid_vector_top_k
        b_top = settings.hybrid_bm25_top_k

        # 两路召回
        vector_docs = self._retrieve_vector(query, v_top)
        bm25_docs = self._retrieve_bm25(query, b_top) if self._bm25 else []

        # RRF 融合: 以文档内容指纹去重累加分数
        rrf_scores: dict = {}
        doc_map: dict = {}
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

        # 按融合分数排序
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        results = [doc_map[key] for key in sorted_keys]
        log.info(
            f"[混合检索] 向量 {len(vector_docs)} + BM25 {len(bm25_docs)} "
            f"→ RRF 融合 {len(results)} 条"
        )
        return results

    # -------------------------------------------------------
    # 重排序
    # -------------------------------------------------------

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        重排序

        优先级:
        1. use_reranker=True 且 FlagEmbedding 可用 → Cross-Encoder 精排
        2. 否则 → 关键词加权重排序 (轻量降级方案)
        """
        if not documents:
            return []

        top_k = settings.rerank_top_k

        if self._should_use_cross_encoder():
            result = self._rerank_cross_encoder(query, documents, top_k)
        else:
            result = self._rerank_keyword(query, documents, top_k)

        from app.core.metrics import metrics
        metrics.record_rerank(len(documents), len(result))
        log.info(f"重排序完成: {len(documents)} -> {len(result)} 条")
        return result

    def _should_use_cross_encoder(self) -> bool:
        """是否启用 Cross-Encoder 精排"""
        if not settings.use_reranker:
            return False
        if not HAS_RERANKER:
            log.warning("use_reranker=True 但未安装 FlagEmbedding, 降级关键词重排序")
            return False
        return True

    def _get_reranker(self):
        """懒加载 Cross-Encoder 重排序模型"""
        if self._reranker is None:
            log.info(f"加载 Cross-Encoder 重排序模型: {settings.reranker_model}")
            self._reranker = FlagReranker(
                settings.reranker_model, use_fp16=True
            )
            log.success("Cross-Encoder 加载完成")
        return self._reranker

    def _rerank_cross_encoder(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        """
        Cross-Encoder 重排序

        Cross-Encoder 将 (query, doc) 拼接输入 Transformer,
        输出相关性分数, 比 Bi-Encoder(向量) 更精准但更慢。
        适合对召回结果做二次精排。
        """
        reranker = self._get_reranker()
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.compute_score(pairs, normalize=True)

        # 单条输入时 compute_score 返回 float, 统一成 list
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        log.info(f"[Cross-Encoder] 重排序分数: {[round(s, 3) for _, s in scored]}")
        return [doc for doc, _ in scored[:top_k]]

    def _rerank_keyword(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        """
        关键词加权重排序 (降级方案)

        向量相似度为主, 关键词命中数作为加分, 文档长度做惩罚。
        无需额外模型, 适合无 GPU 或快速演示场景。
        """
        query_keywords = self._extract_keywords(query)
        scored_docs = []

        for doc in documents:
            text = doc.page_content.lower()
            keyword_score = sum(1 for kw in query_keywords if kw in text)
            length_penalty = min(len(text) / 2000, 0.5)
            final_score = keyword_score - length_penalty
            scored_docs.append((doc, final_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def _extract_keywords(self, query: str) -> List[str]:
        """简单关键词提取 (去停用词 + 长度过滤)"""
        words = re.findall(r"[\w]+", query.lower())
        keywords = [w for w in words if len(w) > 1 and w not in _STOP_WORDS]
        return keywords

    # -------------------------------------------------------
    # 上下文构建
    # -------------------------------------------------------

    def build_context(self, query: str) -> dict:
        """构建完整的 RAG 上下文"""
        # 检索 + 重排序
        retrieved_docs = self.retrieve(query)
        reranked_docs = self.rerank(query, retrieved_docs)

        if not reranked_docs:
            return {"context": "", "sources": [], "chunk_count": 0}

        context_parts = []
        sources = []
        for i, doc in enumerate(reranked_docs, 1):
            source_file = doc.metadata.get("source_file", "未知来源")
            context_parts.append(f"[参考文档{i} - 来源: {source_file}]\n{doc.page_content}")
            if source_file not in sources:
                sources.append(source_file)

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "chunk_count": len(reranked_docs),
        }

    # -------------------------------------------------------
    # 知识库管理
    # -------------------------------------------------------

    def list_documents(self) -> List[dict]:
        """列出知识库中已入库的文档文件"""
        files = []
        for f in settings.knowledge_base_dir.iterdir():
            if f.is_file() and f.suffix.lower() in LOADER_MAP:
                files.append({
                    "name": f.name,
                    "size": f"{f.stat().st_size / 1024:.1f} KB",
                    "type": f.suffix.lower(),
                })
        return files

    def delete_all(self) -> int:
        """清空向量数据库与 BM25 索引 (返回删除的块数)"""
        count = self._count_chunks()
        self._vectorstore.delete_collection()
        self._init_vectorstore()  # 重新初始化空集合
        self._bm25 = None
        self._bm25_docs = []
        log.warning(f"已清空向量数据库与 BM25 索引, 删除 {count} 个文档块")
        return count


# 全局单例
rag_engine = RAGEngine()
