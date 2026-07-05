"""
LLM 大语言模型封装层

统一封装 OpenAI 兼容接口, 支持流式输出和 Function Calling。
通过配置可无缝切换 GPT / 通义千问 / 智谱 / DeepSeek 等模型。
"""
from typing import AsyncIterator, Optional

from langchain_openai import ChatOpenAI

from app.utils.logger import log
from config import settings


class LLMManager:
    """
    LLM 管理器 (单例模式)

    职责:
    1. 创建并缓存 ChatOpenAI 实例
    2. 提供普通对话 / 流式对话接口
    3. 支持 Function Calling (tools 参数)
    """

    _instance: Optional["LLMManager"] = None
    _llm: Optional[ChatOpenAI] = None
    _streaming_llm: Optional[ChatOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._llm is None:
            self._init_llm()

    def _init_llm(self):
        """初始化 LLM 实例"""
        log.info(f"初始化 LLM: model={settings.llm_model_name}, base_url={settings.llm_base_url}")

        # 普通模式 LLM (用于 Function Calling 等需要完整响应的场景)
        self._llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.7,
            streaming=False,
        )

        # 流式模式 LLM (用于打字机效果)
        self._streaming_llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.7,
            streaming=True,
        )

        log.success("LLM 初始化完成")

    @property
    def llm(self) -> ChatOpenAI:
        """获取普通 LLM 实例"""
        return self._llm

    @property
    def streaming_llm(self) -> ChatOpenAI:
        """获取流式 LLM 实例"""
        return self._streaming_llm

    def get_llm_with_tools(self, tools: list):
        """
        获取绑定了工具的 LLM (用于 Function Calling)

        Args:
            tools: LangChain 工具列表

        Returns:
            绑定工具后的 LLM 实例
        """
        return self._llm.bind_tools(tools) if tools else self._llm

    async def stream_chat(self, messages: list) -> AsyncIterator[str]:
        """
        流式对话接口 - 逐 token 返回, 用于前端打字机效果

        流式模式下 API 不返回 usage, 用 tiktoken 估算 prompt/completion token
        并上报到 MetricsCollector, 实现可观测性。

        Args:
            messages: LangChain 消息列表

        Yields:
            每个 token 的文本内容
        """
        from app.core.metrics import estimate_messages_tokens, estimate_tokens, metrics

        prompt_tokens = estimate_messages_tokens(messages)
        completion_chars = []

        try:
            async for chunk in self._streaming_llm.astream(messages):
                if chunk.content:
                    completion_chars.append(chunk.content)
                    yield chunk.content

            # 流式结束, 估算 completion token 并上报
            completion_tokens = estimate_tokens("".join(completion_chars))
            metrics.record_llm_call(prompt_tokens, completion_tokens)
        except Exception as e:
            log.error(f"流式对话异常: {e}")
            yield f"[生成异常] {e}"


# 全局单例
llm_manager = LLMManager()
