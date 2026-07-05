"""
工具管理接口模块

提供:
1. GET /tools - 列出可用工具
2. POST /tools/weather - 直接调用天气工具
3. POST /tools/search - 直接调用搜索工具
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.agent import agent_manager
from app.models.schemas import ToolInfo

router = APIRouter(prefix="/tools", tags=["工具"])


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    input: str = Field(..., description="工具输入参数")


@router.get("", response_model=list[ToolInfo])
async def list_tools():
    """列出所有可用工具"""
    tools = []
    for tool in agent_manager.tools:
        tools.append(ToolInfo(
            name=tool.name,
            description=tool.description or "",
            enabled=True,
        ))
    return tools


@router.post("/weather")
async def call_weather(request: ToolCallRequest):
    """直接调用天气查询工具"""
    from app.tools.weather import weather_tool
    result = weather_tool.invoke({"city": request.input})
    return {"tool": "weather", "input": request.input, "result": result}


@router.post("/search")
async def call_search(request: ToolCallRequest):
    """直接调用网络搜索工具"""
    from app.tools.search import web_search_tool
    result = web_search_tool.invoke({"query": request.input})
    return {"tool": "search", "input": request.input, "result": result}
