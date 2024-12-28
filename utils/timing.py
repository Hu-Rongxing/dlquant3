import time
import functools
import logging
from typing import Callable, Any

from logger import log_manager
logger = log_manager.get_logger(__name__)


def timeit(logger: logging.Logger = logger):
    """
    装饰器：记录函数执行时间

    Args:
        logger (logging.Logger, optional): 日志记录器，默认使用标准输出

    Returns:
        Callable: 装饰后的函数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 选择日志记录方式
            log_func = logger.info if logger else print

            # 记录开始时间
            start_time = time.time()
            log_func(f"开始执行 {func.__name__} 函数")

            try:
                # 执行原始函数
                result = func(*args, **kwargs)

                # 计算执行时间
                end_time = time.time()
                duration = end_time - start_time

                # 格式化时间显示
                if duration < 60:
                    time_str = f"{duration:.2f} 秒"
                elif duration < 3600:
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    time_str = f"{minutes} 分 {seconds} 秒"
                else:
                    hours = int(duration // 3600)
                    minutes = int((duration % 3600) // 60)
                    seconds = int(duration % 60)
                    time_str = f"{hours} 小时 {minutes} 分 {seconds} 秒"

                log_func(f"{func.__name__} 函数执行完成，总耗时：{time_str}")

                return result

            except Exception as e:
                # 记录异常
                log_func(f"{func.__name__} 函数执行出错：{e}")
                raise

        return wrapper

    return decorator