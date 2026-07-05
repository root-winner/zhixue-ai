"""
Function Calling Agent 模块

实现基于工具调用的智能 Agent:
1. LLM 自主判断是否需要调用工具
2. 自动执行工具 (天气查询 / 网络搜索)
3. 将工具结果融入对话, 生成最终回答

流程:
  用户输入 → LLM 决策(是否调用工具)
    ├── 需要工具 → 执行工具 → 工具结果 → LLM 生成回答
    └── 不需要 → 直接生成回答
"""
import json
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.core.llm import llm_manager
from app.tools.search import web_search_tool
from app.tools.weather import weather_tool
from app.utils.logger import log


# 注册所有可用工具
AVAILABLE_TOOLS = [weather_tool, web_search_tool]


class AgentManager:
    """
    Agent 管理器

    职责:
    1. 管理工具注册表
    2. 执行 Function Calling 流程
    3. 支持流式输出最终回答
    """

    def __init__(self):
        self._tools = AVAILABLE_TOOLS
        self._tool_map = {tool.name: tool for tool in self._tools}
        log.info(f"Agent 已注册工具: {list(self._tool_map.keys())}")

    @property
    def tools(self):
        return self._tools

    @property
    def tool_names(self) -> list:
        return list(self._tool_map.keys())

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        执行单个工具调用

        Args:
            tool_name: 工具名称
            tool_input: 工具参数

        Returns:
            工具执行结果 (字符串)
        """
        tool = self._tool_map.get(tool_name)
        if tool is None:
            return f"错误: 未知工具 '{tool_name}'"

        try:
            log.info(f"执行工具: {tool_name}, 参数: {tool_input}")
            result = tool.invoke(tool_input)
            log.info(f"工具 {tool_name} 执行完成, 结果长度: {len(str(result))}")
            return str(result)
        except Exception as e:
            log.error(f"工具 {tool_name} 执行失败: {e}")
            return f"工具执行错误: {e}"

    async def run(
        self,
        messages: list,
        enable_tools: bool = True,
    ) -> AsyncIterator[dict]:
        """
        执行 Agent 流程 (流式输出)

        Yields:
            dict, type 取值:
            - {"type": "tool_call", "name": 工具名, "args": 参数}
            - {"type": "tool_result", "name": 工具名, "result": 结果}
            - {"type": "token", "content": 文本片段}
            - {"type": "done", "full_response": 完整回复}
            - {"type": "error", "message": 错误信息}
        """
        try:
            if not enable_tools:
                # 不启用工具, 直接流式对话
                full_response = ""
                async for chunk in self._stream_response(messages):
                    full_response += chunk
                    yield {"type": "token", "content": chunk}

                yield {"type": "done", "full_response": full_response}
                return

            # 绑定工具的 LLM
            llm_with_tools = llm_manager.get_llm_with_tools(self._tools)

            # 第一步: LLM 决策 (是否需要调用工具)
            log.info("Agent 决策中...")
            response: AIMessage = await llm_with_tools.ainvoke(messages)

            # 第二步: 如果有工具调用, 执行工具
            if response.tool_calls:
                messages.append(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    yield {"type": "tool_call", "name": tool_name, "args": tool_args}

                    # 执行工具
                    result = self._execute_tool(tool_name, tool_args)
                    yield {"type": "tool_result", "name": tool_name, "result": result}

                    # 将工具结果加入消息
                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    ))

                # 第三步: 基于工具结果, 流式生成最终回答
                log.info("基于工具结果生成最终回答...")
                full_response = ""
                async for chunk in self._stream_response(messages):
                    full_response += chunk
                    yield {"type": "token", "content": chunk}

                yield {"type": "done", "full_response": full_response}
            else:
                # 不需要工具, 直接返回 LLM 的回答 (流式重新生成)
                log.info("无需工具调用, 直接回复")
                full_response = ""
                async for chunk in self._stream_response(messages):
                    full_response += chunk
                    yield {"type": "token", "content": chunk}

                yield {"type": "done", "full_response": full_response}

        except Exception as e:
            log.error(f"Agent 执行异常: {e}")
            yield {"type": "error", "message": str(e)}

    async def _stream_response(self, messages: list) -> AsyncIterator[str]:
        """流式获取 LLM 回复"""
        async for chunk in llm_manager.stream_chat(messages):
            yield chunk


# 全局单例
agent_manager = AgentManager()
