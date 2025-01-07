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

        # 生成以日期命名的日志文件
        log_file_name = f"{datetime.date.today()}.log"
        log_path = os.path.join(log_dir, log_file_name)

        # 创建日志文件处理器，按天轮转
        try:
            file_handler = TimedRotatingFileHandler(
                log_path,
                when='midnight',  # 每天午夜轮转
                interval=1,       # 每天轮转一次
                backupCount=15,   # 保留15天的日志
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(EnhancedFormatter())
        except PermissionError:
            # 处理权限错误，生成随机数后缀的日志文件
            random_suffix = random.randint(1000, 9999)
            error_log_path = os.path.join(log_dir, f"{datetime.date.today()}_{random_suffix}.log")
            file_handler = TimedRotatingFileHandler(
                error_log_path,
                when='midnight',
                interval=1,
                backupCount=15,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(EnhancedFormatter())
            print(f"权限错误，已切换到临时日志文件: {error_log_path}")

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