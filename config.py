"""
全局配置管理模块

通过 pydantic-settings 从 .env 文件加载配置, 类型校验 + 环境变量覆盖。
所有配置项集中管理, 避免散落在各模块的硬编码。
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录: ai-chatbot/
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """应用配置, 自动从 .env 文件读取"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM 大模型 ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o-mini"

    # --- Embedding 向量模型 ---
    embedding_api_key: str = "sk-placeholder"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model_name: str = "text-embedding-3-small"
    use_local_embedding: bool = False
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # --- 向量数据库 ---
    chroma_persist_dir: str = str(BASE_DIR / "chroma_db")
    chroma_collection_name: str = "knowledge_base"

    # --- RAG 参数 ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 4
    rerank_top_k: int = 2
    retrieval_strategy: str = "hybrid"
    hybrid_vector_top_k: int = 5
    hybrid_bm25_top_k: int = 5
    rrf_k: int = 60
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- 对话记忆 ---
    max_history_tokens: int = 2000
    session_timeout: int = 3600
    memory_backend: str = "sqlite"
    sqlite_path: str = str(BASE_DIR / "data" / "chat_memory.db")

    # --- 服务配置 ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # --- 工具 ---
    weather_api_key: str = ""

    @property
    def knowledge_base_dir(self) -> Path:
        path = BASE_DIR / "data" / "knowledge_base"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
