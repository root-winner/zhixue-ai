"""
天气查询工具

使用 wttr.in 免费天气 API (无需 API Key)。
支持中英文城市名查询。

备用方案: 和风天气 (需 API Key, 见 .env.example)
"""
import httpx
from langchain_core.tools import tool

from app.utils.logger import log


@tool
def weather_tool(city: str) -> str:
    """
    查询指定城市的当前天气信息。

    当用户询问天气相关问题时使用此工具。
    支持中文和英文城市名, 例如: "北京", "上海", "Tokyo", "New York"。

    Args:
        city: 城市名称 (中英文均可)

    Returns:
        天气信息文本, 包含温度、天气状况、湿度、风速等
    """
    log.info(f"天气查询工具被调用: city={city}")

    try:
        # wttr.in 免费 API, 格式化输出
        url = f"https://wttr.in/{city}"
        params = {
            "format": "j1",  # JSON 格式
            "lang": "zh",
        }

        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params, headers={"Accept-Language": "zh-CN"})
            resp.raise_for_status()
            data = resp.json()

        # 解析天气数据
        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "N/A")
        feels_like = current.get("FeelsLikeC", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("windspeedKmph", "N/A")
        wind_dir = current.get("winddir16Point", "N/A")
        weather_desc_list = current.get("lang_zh", [{}])
        if not weather_desc_list:
            weather_desc_list = current.get("weatherDesc", [{"value": "未知"}])
        weather_desc = weather_desc_list[0].get("value", "未知")

        # 获取地区名
        area = data.get("nearest_area", [{}])[0]
        area_name_list = area.get("areaName", [{"value": city}])
        area_name = area_name_list[0].get("value", city) if area_name_list else city

        result = (
            f"📍 {area_name} 当前天气:\n"
            f"  🌡️ 温度: {temp_c}°C (体感 {feels_like}°C)\n"
            f"  ☁️ 天气: {weather_desc}\n"
            f"  💧 湿度: {humidity}%\n"
            f"  🌬️ 风速: {wind_speed} km/h ({wind_dir})"
        )

        log.info(f"天气查询成功: {area_name} {temp_c}°C")
        return result

    except httpx.HTTPStatusError as e:
        log.error(f"天气查询 HTTP 错误: {e}")
        return f"天气查询失败: 无法获取 '{city}' 的天气信息 (HTTP {e.response.status_code})"
    except Exception as e:
        log.error(f"天气查询异常: {e}")
        return f"天气查询失败: {e}"
