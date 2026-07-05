"""
Embedding 向量模型封装层

支持两种模式:
1. 在线 API 模式: OpenAI 兼容 Embedding 接口 (text-embedding-3-small 等)
2. 本地离线模式: sentence-transformers 本地模型 (BAAI/bge-small-zh-v1.5)

通过 USE_LOCAL_EMBEDDING 环境变量切换。
"""
from typing import List, Optional

from app.utils.logger import log
from config import settings


class EmbeddingManager:
    """
    Embedding 管理器 (单例模式)

    职责:
    1. 根据配置创建在线或本地 Embedding 实例
    2. 提供统一的文本向量化接口
    """

    _instance: Optional["EmbeddingManager"] = None
    _embeddings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._embeddings is None:
            self._init_embeddings()

    def _init_embeddings(self):
        """根据配置初始化 Embedding 模型"""
        if settings.use_local_embedding:
            log.info(f"使用本地 Embedding 模型: {settings.local_embedding_model}")
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.local_embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                log.success("本地 Embedding 模型加载完成")
            except ImportError:
                log.warning("未安装 langchain-huggingface, 回退到在线模式")
                self._init_online_embeddings()
            except Exception as e:
                log.error(f"本地 Embedding 加载失败: {e}, 回退到在线模式")
                self._init_online_embeddings()
        else:
            self._init_online_embeddings()

    def _init_online_embeddings(self):
        """初始化在线 OpenAI 兼容 Embedding"""
        log.info(
            f"使用在线 Embedding: model={settings.embedding_model_name}, "
            f"base_url={settings.embedding_base_url}"
        )
        from langchain_openai import OpenAIEmbeddings

        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        log.success("在线 Embedding 初始化完成")

    @property
    def embeddings(self):
        """获取 Embedding 实例 (供 Chroma 使用)"""
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """向量化单条查询文本"""
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """向量化多条文档"""
        return self._embeddings.embed_documents(texts)


# 全局单例
embedding_manager = EmbeddingManager()
