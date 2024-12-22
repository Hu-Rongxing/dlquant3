import logging
from .logging_config import TRADER_LEVEL_NUM

class EnhancedFormatter(logging.Formatter):
    """增强型日志格式化器"""
    FORMATS = {
        logging.DEBUG: "%(asctime)s | DEBUG | %(name)s:%(lineno)d | %(message)s",
        logging.INFO: "%(asctime)s | INFO | %(name)s:%(lineno)d | %(message)s",
        logging.WARNING: "%(asctime)s | WARNING | %(name)s:%(lineno)d | %(message)s",
        logging.ERROR: "%(asctime)s | ERROR | %(name)s:%(lineno)d | %(message)s",
        logging.CRITICAL: "%(asctime)s | CRITICAL | %(name)s:%(lineno)d | %(message)s",
        TRADER_LEVEL_NUM: "%(asctime)s | TRADER | %(name)s:%(lineno)d | %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# 示例用法：
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(EnhancedFormatter())
    logger.addHandler(handler)

    logger.debug("这是一个 debug 日志")
    logger.info("这是一个 info 日志")
    logger.warning("这是一个 warning 日志")
    logger.error("这是一个 error 日志")
    logger.critical("这是一个 critical 日志")
    logger.log(TRADER_LEVEL_NUM, "这是一个 trader 日志")