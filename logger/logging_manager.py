import os
import logging
import tempfile
from logging.handlers import RotatingFileHandler

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
        """
        初始化日志管理器

        :param config: 日志配置，如果为None则使用默认配置
        """
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return

            # 使用传入配置或创建默认配置
        self.config = config or LoggingConfig()

        # 验证配置
        if not self.config.validate():
            print("日志配置验证失败，将使用默认配置")
            self.config = LoggingConfig()

            # 动态添加 trader 方法到 Logger

        def trader(self, message, *args, **kwargs):
            if self.isEnabledFor(TRADER_LEVEL_NUM):
                self._log(TRADER_LEVEL_NUM, message, args, **kwargs)

        logging.Logger.trader = trader

        # 设置日志系统
        self._setup_logging()

        # 标记已初始化
        self._initialized = True


    def _setup_logging(self):
        """设置日志系统"""
        # 使用配置中的绝对路径
        log_dir = os.path.abspath(self.config.log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # 确保使用绝对路径
        log_path = os.path.abspath(os.path.join(log_dir, self.config.log_file))

        try:
            # 尝试创建日志文件（如果不存在）
            open(log_path, 'a').close()
        except (IOError, PermissionError) as e:
            print(f"无法创建日志文件 {log_path}: {e}")

            # 备选方案：使用系统临时目录
            temp_log_dir = os.path.join(tempfile.gettempdir(), "logs")
            os.makedirs(temp_log_dir, exist_ok=True)
            log_path = os.path.join(temp_log_dir, self.config.log_file)

            try:
                open(log_path, 'a').close()
                print(f"已切换到临时日志文件: {log_path}")
            except Exception as temp_e:
                print(f"无法创建临时日志文件: {temp_e}")
                # 如果仍然失败，使用 NullHandler
                logging.root.addHandler(logging.NullHandler())
                return

                # 打印日志文件路径（调试用）
        print(f"最终日志文件路径: {log_path}")

        # 创建根日志记录器
        logging.root.setLevel(logging.DEBUG)

        # 文件处理器（支持日志轮转）
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=self.config.max_log_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(EnhancedFormatter())

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(EnhancedFormatter())

        # 邮件处理器
        mail_handler = create_smtp_handler(self.config)

        # 清除现有处理器
        logging.root.handlers.clear()

        # 添加处理器
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        logging.root.addHandler(mail_handler)

        # 添加邮件处理器（如果配置了）
        try:
            email_handler = create_smtp_handler(self.config)
            logging.root.addHandler(email_handler)
        except Exception as e:
            print(f"邮件处理器创建失败: {e}")

    @staticmethod
    def get_logger(name: str):
        """获取特定模块的日志记录器"""
        return logging.getLogger(name)

    # 全局日志管理器


log_manager = LoggingManager()

if __name__ == '__main__':
    logger = log_manager.get_logger(__name__)
    logger.info('信息')
    logger.trader("交易")