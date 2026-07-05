"""
日志工具模块

基于 loguru 封装, 统一日志格式, 支持文件轮转和控制台彩色输出。
"""
import sys
from loguru import logger

from config import settings


def setup_logger():
    """初始化全局日志配置"""
    # 清除默认配置
    logger.remove()

    # 控制台输出: 彩色, DEBUG 级别
    logger.add(
        sys.stderr,
        level="DEBUG" if settings.debug else "INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件输出: 按天轮转, 保留 7 天
    log_dir = settings.chroma_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        level="INFO",
        rotation="00:00",       # 每天轮转
        retention="7 days",     # 保留 7 天
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )

    return logger


# 导出已配置的 logger 实例
log = setup_logger()
