"""
网络搜索工具

使用 DuckDuckGo HTML 版本进行搜索 (无需 API Key)。
返回搜索结果摘要, 供 LLM 引用。

依赖: httpx, beautifulsoup4
"""
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from app.utils.logger import log


@tool
def web_search_tool(query: str) -> str:
    """
    在互联网上搜索信息。

    当用户询问最新资讯、事实性知识、或知识库中没有的内容时使用此工具。
    返回前 5 条搜索结果的标题和摘要。

    Args:
        query: 搜索关键词

    Returns:
        搜索结果文本 (前5条)
    """
    log.info(f"网络搜索工具被调用: query={query}")

    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query, "kl": "cn-zh"}

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 解析搜索结果
        results = []
        result_blocks = soup.find_all("div", class_="result")[:5]

        for i, block in enumerate(result_blocks, 1):
            # 标题
            title_tag = block.find("a", class_="result__a")
            title = title_tag.get_text(strip=True) if title_tag else "无标题"

            # 摘要
            snippet_tag = block.find("a", class_="result__snippet")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "无摘要"

            # 链接
            href = title_tag.get("href", "") if title_tag else ""
            # DuckDuckGo 的链接是重定向格式, 提取实际 URL
            if "uddg=" in href:
                from urllib.parse import parse_qs, urlparse
                parsed = urlparse(href)
                qs = parse_qs(parsed.query)
                href = qs.get("uddg", [href])[0]

            results.append(f"{i}. {title}\n   {snippet}\n   🔗 {href}")

        if not results:
            return f"未找到与 '{query}' 相关的搜索结果"

        output = f"🔍 搜索 '{query}' 的结果:\n\n" + "\n\n".join(results)
        log.info(f"网络搜索完成, 返回 {len(results)} 条结果")
        return output

    except Exception as e:
        log.error(f"网络搜索异常: {e}")
        return f"网络搜索失败: {e}"
