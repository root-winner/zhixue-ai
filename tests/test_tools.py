"""
工具调用单元测试

覆盖:
1. 天气查询工具 (mock httpx)
2. 网络搜索工具 (mock httpx)
3. 工具注册与调用
"""
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestWeatherTool:
    """天气查询工具"""

    def test_weather_tool_is_registered(self):
        """工具已正确注册为 LangChain Tool"""
        from app.tools.weather import weather_tool

        assert weather_tool.name == "weather_tool"
        assert "天气" in weather_tool.description or "weather" in weather_tool.description.lower()

    @patch("app.tools.weather.httpx.Client")
    def test_weather_success(self, mock_client_cls):
        """成功查询天气"""
        from app.tools.weather import weather_tool

        # 构造 mock 响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current_condition": [{
                "temp_C": "25",
                "FeelsLikeC": "27",
                "humidity": "60",
                "windspeedKmph": "10",
                "winddir16Point": "NE",
                "lang_zh": [{"value": "晴"}],
            }],
            "nearest_area": [{
                "areaName": [{"value": "北京"}],
            }],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = weather_tool.invoke({"city": "北京"})

        assert "北京" in result
        assert "25" in result
        assert "晴" in result
        assert "60%" in result

    @patch("app.tools.weather.httpx.Client")
    def test_weather_http_error(self, mock_client_cls):
        """HTTP 错误时返回错误信息"""
        import httpx as httpx_mod
        from app.tools.weather import weather_tool

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = weather_tool.invoke({"city": "不存在的城市"})

        assert "失败" in result or "错误" in result

    @patch("app.tools.weather.httpx.Client")
    def test_weather_network_error(self, mock_client_cls):
        """网络异常时返回错误信息"""
        from app.tools.weather import weather_tool

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Connection timeout")
        mock_client_cls.return_value = mock_client

        result = weather_tool.invoke({"city": "北京"})

        assert "失败" in result or "错误" in result
        assert "Connection timeout" in result


@pytest.mark.unit
class TestSearchTool:
    """网络搜索工具"""

    def test_search_tool_is_registered(self):
        """工具已正确注册"""
        from app.tools.search import web_search_tool

        assert web_search_tool.name == "web_search_tool"
        assert "搜索" in web_search_tool.description or "search" in web_search_tool.description.lower()

    @patch("app.tools.search.httpx.Client")
    def test_search_success(self, mock_client_cls):
        """成功搜索"""
        from app.tools.search import web_search_tool

        # 构造 DuckDuckGo HTML 响应
        fake_html = """
        <html><body>
            <div class="result">
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com">示例结果标题</a>
                <a class="result__snippet">这是搜索结果摘要内容</a>
            </div>
            <div class="result">
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Ftest.com">第二个结果</a>
                <a class="result__snippet">第二个摘要</a>
            </div>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = fake_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = web_search_tool.invoke({"query": "测试搜索"})

        assert "示例结果标题" in result or "搜索" in result

    @patch("app.tools.search.httpx.Client")
    def test_search_no_results(self, mock_client_cls):
        """无搜索结果时返回提示"""
        from app.tools.search import web_search_tool

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = web_search_tool.invoke({"query": "不存在的关键词"})

        assert "未找到" in result or "无结果" in result

    @patch("app.tools.search.httpx.Client")
    def test_search_network_error(self, mock_client_cls):
        """网络异常时返回错误信息"""
        from app.tools.search import web_search_tool

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Network error")
        mock_client_cls.return_value = mock_client

        result = web_search_tool.invoke({"query": "测试"})

        assert "失败" in result or "错误" in result


@pytest.mark.unit
class TestAgentTools:
    """Agent 工具注册"""

    def test_agent_has_tools(self):
        """Agent 已注册工具"""
        from app.core.agent import agent_manager

        assert len(agent_manager.tools) >= 2
        tool_names = agent_manager.tool_names
        assert "weather_tool" in tool_names
        assert "web_search_tool" in tool_names

    def test_tool_map_correct(self):
        """工具映射表正确"""
        from app.core.agent import agent_manager

        assert "weather_tool" in agent_manager._tool_map
        assert "web_search_tool" in agent_manager._tool_map
