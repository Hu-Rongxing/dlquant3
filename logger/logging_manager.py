import os
import logging
import tempfile
import random
import datetime
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
from queue import Queue

from .logging_config import LoggingConfig, TRADER_LEVEL_NUM
from .logging_formatter import EnhancedFormatter
from .smtp_handler import create_smtp_handler


class LoggingManager:
    _instance = None

    def __new__(cls, config: LoggingConfig = None):
        """单例模式"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: LoggingConfig = None):
        """初始化日志管理器"""
        if hasattr(self, '_initialized'):
            return

        self.config = config or LoggingConfig()

        if not self.config.validate():
            print("日志配置验证失败，将使用默认配置")
            self.config = LoggingConfig()

        def trader(self, message, *args, **kwargs):
            if self.isEnabledFor(TRADER_LEVEL_NUM):
                self._log(TRADER_LEVEL_NUM, message, args, **kwargs)

        logging.Logger.trader = trader

        self._setup_logging()
        self._initialized = True

    def _setup_logging(self):
        """设置日志系统"""
        log_dir = os.path.abspath(self.config.log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # 使用 Midnight Rotating File Handler
        def make_log_file_path():
            return os.path.join(log_dir, f"{datetime.date.today()}.log")

        class MidnightRotatingFileHandler(TimedRotatingFileHandler):
            def __init__(self, filename, **kwargs):
                super().__init__(filename, **kwargs)
                self.make_log_file_path = make_log_file_path

            def computeRollover(self, currentTime):
                return super().computeRollover(currentTime)

            def doRollover(self):
                # 确保在午夜准确创建新文件
                if self.stream:
                    self.stream.close()
                    self.stream = None

                    # 生成新的日志文件路径
                new_log_path = self.make_log_file_path()
                self.baseFilename = new_log_path

                # 重新打开文件流
                self.stream = self._open()

                # 处理备份文件
                if self.backupCount > 0:
                    for s, _ in self.getFilesToDelete():
                        os.remove(s)

                        # 创建文件处理器

        file_handler = MidnightRotatingFileHandler(
            make_log_file_path(),
            when='midnight',
            interval=1,
            backupCount=15,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(EnhancedFormatter())

        # 创建日志队列
        self.log_queue = Queue()
        logging.root.setLevel(logging.DEBUG)

        # 创建队列处理器
        queue_handler = QueueHandler(self.log_queue)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(EnhancedFormatter())

        # 邮件处理器
        mail_handler = create_smtp_handler(self.config)

        # 清除现有处理器
        logging.root.handlers.clear()
        logging.root.addHandler(queue_handler)

        # 启动队列监听器
        self.listener = QueueListener(self.log_queue, file_handler, console_handler, mail_handler)
        self.listener.start()

    @staticmethod
    def get_logger(name: str):
        """获取特定模块的日志记录器"""
        return logging.getLogger(name)

    def stop(self):
        """停止日志监听器"""
        self.listener.stop()

# 全局日志管理器
log_manager = LoggingManager()

if __name__ == '__main__':
    logger = log_manager.get_logger(__name__)
    logger.info('信息')
    logger.trader("交易")