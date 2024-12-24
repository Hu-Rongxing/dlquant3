import time
from functools import wraps
from typing import Tuple, Type
# 自定义
from logger import log_manager
applogger = log_manager.get_logger(__name__)


def retry(max_attempts: int = 3, delay: float = 1,
          exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    重试装饰器

    Args:
        max_attempts (int): 最大重试次数
        delay (float): 重试间隔
        exceptions (Tuple[Type[Exception], ...]): 需要重试的异常类型
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    applogger.warning(f"第 {attempts} 次重试 {func.__name__}，错误：{e}")
                    if attempts == max_attempts:
                        applogger.error(f"{func.__name__} 重试失败")
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator