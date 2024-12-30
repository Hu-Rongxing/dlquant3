# logging_config.py
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
# from logging_manager import setup_logging
#
# # 调用日志设置
# setup_logging()

# 自定义日志级别
TRADER_LEVEL_NUM = 35
logging.addLevelName(TRADER_LEVEL_NUM, "TRADER")
from config import settings


def generate_log_filename(base_name: str, timestamp: bool = True) -> str:
    """
    生成带有时间戳的日志文件名

    :param base_name: 基础文件名
    :param timestamp: 是否添加时间戳
    :return: 完整的日志文件名
    """
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(base_name)
        return f"{name}_{timestamp_str}{ext}"
    return base_name


@dataclass
class LoggingConfig:
    """日志系统配置"""
    log_dir: str = settings.get("logging.log_dir", "logger/logs")
    log_file: str = settings.get("logging.log_file", "logger/logs/applog.log")
    max_log_size: int = 5 * 1024 * 1024  # 10MB
    backup_count: int = 5  # 保留的日志文件数量
    log_level: int = settings.get("logging.log_level", logging.INFO)

    email_recipients: List[str] = field(default_factory=lambda: [settings.get("qqmail.ALERT_EMAIL_ADDRESS", "280712999@qq.com")])
    email_sender: Optional[str] = settings.get("qqmail.SMTP_USER_NAME", "hurongxing@vip.qq.com")
    email_smtp_server: Optional[str] = settings.get("qqmail.SMTP_HOST", "smtp.qq.com")
    email_password: Optional[str] = settings.get("qqmail.SMTP_PASSWORD", "")
    email_port: int = 587

    def validate(self) -> bool:
        """
        验证日志配置的有效性

        :return: 配置是否有效
        """
        try:
            # 确保日志目录存在，使用绝对路径
            log_dir = os.path.abspath(self.log_dir)
            os.makedirs(log_dir, exist_ok=True)

            # 构建完整的日志文件路径
            log_path = os.path.join(log_dir, self.log_file)

            # 尝试创建文件
            with open(log_path, 'a') as f:
                pass

            return True
        except (IOError, PermissionError) as e:
            print(f"日志目录或文件创建失败: {e}")
            return False
