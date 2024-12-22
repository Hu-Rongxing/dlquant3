from dataclasses import dataclass
from typing import Callable, List, Optional
# 自定义
from utils.set_system import change_language
from qmt_client import monitor_xt_client, generate_trading_report


@dataclass
class TaskConfig:
    """任务配置类"""
    func: Callable
    cron_expression: str
    task_id: str
    description: str
    check_trading_day: bool = True  # 是否检查交易日
    run_in_separate_process: bool = False  # 是否在单独进程中运行

tasks = [
    TaskConfig(
        func=change_language,
        cron_expression="8:00,9:00,10:00,11:00,13:00,14:00",
        task_id="change_language",
        description="切换输入法为英文。"
    ),
    TaskConfig(
        func=monitor_xt_client,
        cron_expression="8:00,9:20,12:30",
        task_id="monitor_xt_client",
        description="监控迅投客户端。",
        run_in_separate_process=True
    ),
    TaskConfig(
        func=generate_trading_report,
        cron_expression="9:00,11:35,15:05",
        task_id="generate_trading_report",
        description="生成交易报告。",
        run_in_separate_process=True
    ),
]