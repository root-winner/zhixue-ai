"""
RAG 引擎单元测试

覆盖:
1. 关键词提取 (停用词过滤)
2. 重排序逻辑 (关键词加权 + 长度惩罚)
3. 文档切分 (递归字符切分)
4. 上下文构建 (来源标注)
5. 检索流程 (mock 向量库)
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.mark.unit
class TestKeywordExtraction:
    """关键词提取"""

    def test_basic_extraction(self, rag_engine_mock):
        """基本关键词提取"""
        keywords = rag_engine_mock._extract_keywords("智学AI知识库系统")
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_filters_stop_words(self, rag_engine_mock):
        """停用词应被过滤"""
        keywords = rag_engine_mock._extract_keywords("的 了 是 在 我")
        # 停用词应被过滤掉 (这些是停用词)
        assert "的" not in keywords
        assert "了" not in keywords
        assert "是" not in keywords

    def test_preserves_meaningful_words(self, rag_engine_mock):
        """保留有意义的词"""
        keywords = rag_engine_mock._extract_keywords("Python编程 语言")
        assert "python" in keywords  # 转小写
        assert "编程" in keywords or "语言" in keywords

    def test_empty_query(self, rag_engine_mock):
        """空查询返回空列表"""
        keywords = rag_engine_mock._extract_keywords("")
        assert keywords == []

    def test_single_char_filtered(self, rag_engine_mock):
        """单字符被过滤 (长度 <= 1)"""
        keywords = rag_engine_mock._extract_keywords("a b c 测试")
        assert "a" not in keywords
        assert "b" not in keywords
        assert "测试" in keywords


@pytest.mark.unit
class TestRerank:
    """重排序逻辑"""

    def test_rerank_by_keyword_hits(self, rag_engine_mock, sample_documents):
        """关键词命中多的文档应排在前面"""
        query = "智学AI知识库"
        result = rag_engine_mock.rerank(query, sample_documents)

        # 第一条文档包含 "智学AI" 和 "知识库", 应排最前
        assert "智学AI" in result[0].page_content or "知识库" in result[0].page_content

    def test_rerank_returns_top_k(self, rag_engine_mock, sample_documents):
        """重排序后返回数量不超过 rerank_top_k"""
        from config import settings

        result = rag_engine_mock.rerank("测试", sample_documents)
        assert len(result) <= settings.rerank_top_k

    def test_rerank_empty_input(self, rag_engine_mock):
        """空输入返回空列表"""
        result = rag_engine_mock.rerank("query", [])
        assert result == []

    def test_rerank_preserves_document_content(self, rag_engine_mock, sample_documents):
        """重排序不修改文档内容"""
        original_contents = [doc.page_content for doc in sample_documents]
        rag_engine_mock.rerank("测试", sample_documents)

        # 原始文档内容不变
        for orig, doc in zip(original_contents, sample_documents):
            assert orig == doc.page_content

    def test_rerank_order_changes(self, rag_engine_mock):
        """重排序应改变文档顺序 (相关文档上浮)"""
        docs = [
            Document(page_content="Python是一种编程语言", metadata={}),
            Document(page_content="FastAPI是Python的Web框架，用于构建API", metadata={}),
            Document(page_content="今天天气不错", metadata={}),
        ]

        result = rag_engine_mock.rerank("Python FastAPI API", docs)
        # 包含 Python/FastAPI 的文档应排在天气文档前面
        contents = [d.page_content for d in result]
        weather_idx = next((i for i, c in enumerate(contents) if "天气" in c), -1)
        if weather_idx >= 0:
            # 天气文档不应排第一
            assert weather_idx > 0 or len(result) <= 1


@pytest.mark.unit
class TestDocumentSplitting:
    """文档切分"""

    def test_split_short_document(self, rag_engine_mock):
        """短文档切分后仍为一块"""
        doc = Document(
            page_content="这是一段很短的文本。",
            metadata={"source_file": "test.txt"},
        )
        chunks = rag_engine_mock.split_documents([doc])
        assert len(chunks) >= 1
        assert chunks[0].page_content == "这是一段很短的文本。"

    def test_split_long_document(self, rag_engine_mock):
        """长文档被切分为多块"""
        long_text = "这是第一个段落的内容。\n\n" * 200  # 足够长
        doc = Document(page_content=long_text, metadata={"source_file": "long.txt"})
        chunks = rag_engine_mock.split_documents([doc])

        assert len(chunks) > 1, "长文档应被切分为多块"

    def test_split_preserves_metadata(self, rag_engine_mock):
        """切分后元数据保留"""
        doc = Document(
            page_content="A" * 1000,
            metadata={"source_file": "test.pdf", "doc_id": "123"},
        )
        chunks = rag_engine_mock.split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("source_file") == "test.pdf"
            assert chunk.metadata.get("doc_id") == "123"

    def test_chunk_size_within_limit(self, rag_engine_mock):
        """切分块大小不超过 chunk_size + overlap"""
        from config import settings

        long_text = "测试文本内容。" * 500
        doc = Document(page_content=long_text, metadata={})
        chunks = rag_engine_mock.split_documents([doc])

        for chunk in chunks:
            # 允许 overlap 范围内的误差
            assert len(chunk.page_content) <= settings.chunk_size + settings.chunk_overlap + 50


@pytest.mark.unit
class TestContextBuilding:
    """上下文构建"""

    def test_build_context_with_results(self, rag_engine_mock, sample_documents):
        """有检索结果时构建上下文"""
        # mock retrieve 和 rerank
        rag_engine_mock.retrieve = MagicMock(return_value=sample_documents[:2])
        rag_engine_mock.rerank = MagicMock(return_value=sample_documents[:2])

        result = rag_engine_mock.build_context("智学AI")

        assert result["context"] != ""
        assert result["chunk_count"] == 2
        assert len(result["sources"]) > 0
        assert "[参考文档" in result["context"]

    def test_build_context_empty_results(self, rag_engine_mock):
        """无检索结果时返回空上下文"""
        rag_engine_mock.retrieve = MagicMock(return_value=[])
        rag_engine_mock.rerank = MagicMock(return_value=[])

        result = rag_engine_mock.build_context("不存在的查询")

        assert result["context"] == ""
        assert result["sources"] == []
        assert result["chunk_count"] == 0

    def test_context_includes_source_attribution(self, rag_engine_mock, sample_documents):
        """上下文包含来源标注"""
        rag_engine_mock.retrieve = MagicMock(return_value=[sample_documents[0]])
        rag_engine_mock.rerank = MagicMock(return_value=[sample_documents[0]])

        result = rag_engine_mock.build_context("查询")
        assert "来源" in result["context"]
        assert sample_documents[0].metadata["source_file"] in result["sources"]


@pytest.mark.unit
class TestRetrieve:
    """检索流程 (mock 向量库)"""

    def test_retrieve_calls_vectorstore(self, rag_engine_mock, sample_documents):
        """检索调用向量数据库的 similarity_search"""
        rag_engine_mock._vectorstore.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(doc, 0.9) for doc in sample_documents]
        )

        results = rag_engine_mock.retrieve("查询")
        assert len(results) > 0
        rag_engine_mock._vectorstore.similarity_search_with_relevance_scores.assert_called_once()

    def test_retrieve_filters_low_scores(self, rag_engine_mock, sample_documents):
        """相似度低于阈值的结果仍保留 (避免空结果)"""
        # 全部低于 0.3 阈值
        mock_results = [(doc, 0.1) for doc in sample_documents]
        rag_engine_mock._vectorstore.similarity_search_with_relevance_scores = MagicMock(
            return_value=mock_results
        )

        results = rag_engine_mock.retrieve("查询")
        # 全部低于阈值时仍返回 (避免空结果的兜底逻辑)
        assert len(results) == len(sample_documents)

    def test_retrieve_custom_top_k(self, rag_engine_mock, sample_documents):
        """自定义 top_k 参数"""
        rag_engine_mock._vectorstore.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(doc, 0.9) for doc in sample_documents[:2]]
        )

        rag_engine_mock.retrieve("查询", top_k=2)
        call_args = rag_engine_mock._vectorstore.similarity_search_with_relevance_scores.call_args
        assert call_args[1]["k"] == 2
