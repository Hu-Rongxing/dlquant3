import logging
import threading
import time
from queue import Queue
from typing import List

from utils.mail_utils import send_email_notification
from .logging_config import LoggingConfig, TRADER_LEVEL_NUM
from .logging_formatter import EnhancedFormatter


class CustomSMTPHandler(logging.Handler):
    """
    自定义SMTP处理器，支持邮件合并发送和时间间隔控制

    特性：
    1. 合并多条日志消息
    2. 至少间隔15秒发送一次邮件
    3. 线程安全
    4. 可配置最大日志合并数量
    """

    def __init__(
            self,
            config: LoggingConfig,
            min_interval: int = 15,  # 最小发送间隔（秒）
            max_log_count: int = 10  # 最大合并日志数量
    ):
        super().__init__()
        self.config = config
        self.setLevel(TRADER_LEVEL_NUM)

        # 日志队列和控制参数
        self._log_queue = Queue()
        self._last_send_time = 0
        self._min_interval = min_interval
        self._max_log_count = max_log_count

        # 启动后台发送线程
        self._stop_event = threading.Event()
        self._send_thread = threading.Thread(
            target=self._background_send_thread,
            daemon=True
        )
        self._send_thread.start()

    def emit(self, record):
        """
        将日志记录放入队列

        :param record: 日志记录
        """
        try:
            msg = self.format(record)
            self._log_queue.put(msg)
        except Exception:
            self.handleError(record)

    def _background_send_thread(self):
        """
        后台线程：定期检查并发送日志
        """
        while not self._stop_event.is_set():
            try:
                # 收集日志
                logs = self._collect_logs()

                # 发送条件：
                # 1. 有日志
                # 2. 距离上次发送超过最小间隔
                current_time = time.time()
                if logs and (current_time - self._last_send_time >= self._min_interval):
                    self._send_merged_email(logs)
                    self._last_send_time = current_time

                    # 避免过度消耗CPU
                time.sleep(1)

            except Exception as e:
                print(f"后台发送线程异常: {e}")

    def _collect_logs(self) -> List[str]:
        """
        收集待发送的日志

        :return: 日志列表
        """
        logs = []
        while not self._log_queue.empty() and len(logs) < self._max_log_count:
            try:
                logs.append(self._log_queue.get_nowait())
            except Exception:
                break
        return logs

    def _send_merged_email(self, logs: List[str]):
        """
        发送合并的日志邮件

        :param logs: 日志列表
        """
        if not logs:
            return

        try:
            # 合并日志
            merged_log = "<br>".join(logs)

            # 发送邮件通知
            send_email_notification(merged_log)

            print(f"已发送 {len(logs)} 条合并日志")

        except Exception as e:
            print(f"合并日志邮件发送失败: {e}")

    def close(self):
        """
        关闭处理器，确保后台线程安全退出
        """
        self._stop_event.set()
        if self._send_thread.is_alive():
            self._send_thread.join(timeout=5)
        super().close()

    # 使用示例


def create_smtp_handler(config: LoggingConfig = None):
    """
    创建自定义SMTP处理器的工厂方法

    :param config: 日志配置
    :return: CustomSMTPHandler实例
    """
    email_handler = CustomSMTPHandler(
        config,
        min_interval=15,  # 最小发送间隔15秒
        max_log_count=20  # 最多合并10条日志
    )
    email_handler.setFormatter(EnhancedFormatter())

    return email_handler