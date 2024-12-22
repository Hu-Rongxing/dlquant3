import time
from functools import wraps
import logging


def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    重试装饰器

    :param max_attempts: 最大重试次数
    :param delay: 初始重试延迟
    :param backoff: 延迟指数增长因子
    :param exceptions: 需要重试的异常类型
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logging.error(f"重试 {max_attempts} 次后仍失败: {e}")
                        raise

                    logging.warning(f"第 {attempts} 次重试: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator