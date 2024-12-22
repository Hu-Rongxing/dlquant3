from config import settings
from logger.logging_config import LoggingConfig
from logger.logging_manager import log_manager
import os


def test_log_config():
    config = LoggingConfig()
    print(config)


def test_logging():
    logger = log_manager.get_logger(__name__)
    # 多级别日志测试
    print("当前工作目录:", os.getcwd())

    logger.debug("这是一个调试信息")
    logger.info("这是一个普通信息")
    logger.warning("这是一个警告")
    logger.error("这是一个错误")
    logger.trader("这是一个交易日志")

    # 检查日志文件
    log_path = os.path.abspath(
        os.path.join(settings.get("logging.log_dir"),
                     settings.get("logging.log_file"))
    )
    print(f"\n日志文件路径: {log_path}")
    print(f"日志文件是否存在: {os.path.exists(log_path)}")


if __name__ == '__main__':
    test_logging()