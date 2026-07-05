"""
可观测性模块 - 指标采集

采集三类指标:
1. LLM 调用: prompt/completion token 用量、调用次数 (tiktoken 估算)
2. 检索质量: 检索次数、策略分布、命中文档数、重排序前后变化、检索耗时
3. HTTP 请求: 请求数、状态码分布、平均耗时

设计:
- 单例 MetricsCollector, 线程安全 (threading.Lock)
- 流式 LLM 无法从 API 拿 usage, 用 tiktoken 估算 token 数
- 暴露 /api/metrics 端点查看快照, 生产环境可对接 Prometheus
"""
import threading
import time
from collections import defaultdict
from typing import List, Optional

import tiktoken

from app.utils.logger import log


def _get_encoder():
    """获取 tiktoken 编码器 (用于 token 估算)"""
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """估算文本 token 数"""
    return len(_get_encoder().encode(text))


def estimate_messages_tokens(messages: list) -> int:
    """估算 LangChain 消息列表的 token 数"""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # 每条消息约 4 token 开销 (角色标记等)
        total += 4
    return total


class MetricsCollector:
    """指标采集器 (单例, 线程安全)"""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self._data_lock = threading.Lock()
        self._start_time = time.time()
        # LLM 指标
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        # 检索指标
        self._retrieval_total = 0
        self._retrieval_by_strategy = defaultdict(int)
        self._retrieval_chunks = 0
        self._retrieval_time_total = 0.0
        self._rerank_calls = 0
        self._rerank_before_total = 0
        self._rerank_after_total = 0
        # 请求指标
        self._request_total = 0
        self._request_by_status = defaultdict(int)
        self._request_latency_total = 0.0

    # ---- LLM ----

    def record_llm_call(
        self, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """记录一次 LLM 调用的 token 用量"""
        with self._data_lock:
            self._llm_calls += 1
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens

    # ---- 检索 ----

    def record_retrieval(
        self, strategy: str, chunks: int, elapsed: float
    ) -> None:
        """记录一次检索"""
        with self._data_lock:
            self._retrieval_total += 1
            self._retrieval_by_strategy[strategy] += 1
            self._retrieval_chunks += chunks
            self._retrieval_time_total += elapsed

    def record_rerank(self, before: int, after: int) -> None:
        """记录一次重排序"""
        with self._data_lock:
            self._rerank_calls += 1
            self._rerank_before_total += before
            self._rerank_after_total += after

    # ---- 请求 ----

    def record_request(self, status_code: int, elapsed: float) -> None:
        """记录一次 HTTP 请求"""
        with self._data_lock:
            self._request_total += 1
            self._request_by_status[status_code] += 1
            self._request_latency_total += elapsed

    # ---- 快照 ----

    def snapshot(self) -> dict:
        """获取当前指标快照"""
        with self._data_lock:
            avg_retrieval_time = (
                self._retrieval_time_total / self._retrieval_total
                if self._retrieval_total
                else 0
            )
            avg_request_latency = (
                self._request_latency_total / self._request_total
                if self._request_total
                else 0
            )
            avg_rerank_ratio = (
                self._rerank_after_total / self._rerank_before_total
                if self._rerank_before_total
                else 0
            )
            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "llm": {
                    "total_calls": self._llm_calls,
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                    "total_tokens": self._prompt_tokens + self._completion_tokens,
                },
                "retrieval": {
                    "total_calls": self._retrieval_total,
                    "by_strategy": dict(self._retrieval_by_strategy),
                    "total_chunks": self._retrieval_chunks,
                    "avg_chunks_per_call": round(
                        self._retrieval_chunks / self._retrieval_total, 2
                    ) if self._retrieval_total else 0,
                    "avg_latency_ms": round(avg_retrieval_time * 1000, 1),
                    "rerank_calls": self._rerank_calls,
                    "rerank_avg_retention": f"{round(avg_rerank_ratio * 100, 1)}%",
                },
                "requests": {
                    "total": self._request_total,
                    "by_status": dict(self._request_by_status),
                    "avg_latency_ms": round(avg_request_latency * 1000, 1),
                },
            }

    def reset(self) -> None:
        """重置所有指标"""
        with self._data_lock:
            self._init_state()
        log.info("指标已重置")


# 全局单例
metrics = MetricsCollector()
