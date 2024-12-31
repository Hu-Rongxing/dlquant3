import sys
import time
import signal
import schedule
import multiprocessing
from typing import List, Callable
import logging
import traceback
import psutil
from dataclasses import dataclass

from logger import log_manager
logger = log_manager.get_logger(__name__)


# 交易日检查
def is_trading_day() -> bool:
    """  
    检查是否为交易日  

    Returns:  
        bool: 是否为交易日  
    """
    try:
        from data_processing.source_trading_calendar import ChineseMarketCalendar
        return ChineseMarketCalendar().is_trading_day()
    except ImportError:
        # 如果导入失败，默认返回 True  
        return True

    # 任务配置类


@dataclass
class TaskConfig:
    """任务配置类"""
    func: Callable
    cron_expression: str
    task_id: str
    description: str
    check_trading_day: bool = True
    run_in_separate_process: bool = False


# 进程管理器
class ProcessManager:
    """  
    进程管理器，确保同一时刻只有一个进程实例运行  
    """

    def __init__(self, logger=None):
        """  
        初始化进程管理器  

        Args:  
            logger (logging.Logger, optional): 日志记录器  
        """
        self.logger = logger or logging.getLogger(__name__)
        self.running_processes = {}

    def is_process_running(self, task_id: str) -> bool:
        """  
        检查特定任务的进程是否正在运行  

        Args:  
            task_id (str): 任务ID  

        Returns:  
            bool: 进程是否正在运行  
        """
        if task_id in self.running_processes:
            pid = self.running_processes[task_id]
            try:
                # 检查进程是否存在  
                process = psutil.Process(pid)
                return process.is_running()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # 进程已不存在，清理记录  
                del self.running_processes[task_id]
                return False
        return False

    def run_task_in_process(self, task_config: TaskConfig):
        """  
        在独立进程中运行任务，确保同一时刻只有一个实例  

        Args:  
            task_config (TaskConfig): 任务配置  
        """
        # 检查是否已有进程运行  
        if self.is_process_running(task_config.task_id):
            self.logger.info(f"任务 {task_config.task_id} 已在运行，跳过本次调度")
            return

        try:
            # 创建并启动进程  
            process = multiprocessing.Process(
                target=self._safe_run_task,
                args=(task_config,)
            )
            process.start()

            # 记录进程ID  
            self.running_processes[task_config.task_id] = process.pid

            self.logger.info(f"启动任务 {task_config.task_id}，进程ID: {process.pid}")

        except Exception as e:
            self.logger.error(f"创建进程失败: {e}")

    def _safe_run_task(self, task_config: TaskConfig):
        """  
        安全执行任务  

        Args:  
            task_config (TaskConfig): 任务配置  
        """
        try:
            # 检查交易日  
            if task_config.check_trading_day and not is_trading_day():
                self.logger.info(f"非交易日，跳过任务: {task_config.task_id}")
                return

                # 执行任务
            task_config.func()

        except Exception as e:
            self.logger.error(f"任务 {task_config.task_id} 执行失败: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            # 清理进程记录  
            if task_config.task_id in self.running_processes:
                del self.running_processes[task_config.task_id]

            # 任务执行器


class TaskExecutor:
    """  
    任务执行器  
    """

    def __init__(self, logger=None):
        """  
        初始化任务执行器  

        Args:  
            logger (logging.Logger, optional): 日志记录器  
        """
        self.logger = logger or logging.getLogger(__name__)
        self.process_manager = ProcessManager(self.logger)

    def schedule_tasks(self, tasks: List[TaskConfig]):
        """  
        调度任务  

        Args:  
            tasks (List[TaskConfig]): 任务配置列表  
        """
        for task in tasks:
            # 解析时间点  
            for time_point in task.cron_expression.split(','):
                if task.run_in_separate_process:
                    # 独立进程任务  
                    schedule.every().day.at(time_point).do(
                        self.process_manager.run_task_in_process,
                        task_config=task
                    )
                else:
                    # 主进程任务  
                    schedule.every().day.at(time_point).do(
                        self._run_task_in_main_process,
                        task_config=task
                    )

                self.logger.info(f"已调度任务 {task.task_id}: {time_point}")

    def _run_task_in_main_process(self, task_config: TaskConfig):
        """  
        在主进程中运行任务  

        Args:  
            task_config (TaskConfig): 任务配置  
        """
        try:
            # 检查交易日  
            if task_config.check_trading_day and not is_trading_day():
                self.logger.info(f"非交易日，跳过任务: {task_config.task_id}")
                return

                # 执行任务
            task_config.func()

        except Exception as e:
            self.logger.error(f"任务 {task_config.task_id} 执行失败: {e}")
            self.logger.error(traceback.format_exc())

        # 主调度程序


def main(tasks: List[TaskConfig]):
    """  
    主调度程序  

    Args:  
        tasks (List[TaskConfig]): 任务配置列表  
    """

    try:
        # 创建任务执行器  
        executor = TaskExecutor(logger)

        # 调度任务  
        executor.schedule_tasks(tasks)

        logger.info("任务调度器已启动")

        # 信号处理  
        def signal_handler(signum, frame):
            logger.info("收到终止信号，正在关闭...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 持续运行调度器  
        while True:
            schedule.run_pending()
            time.sleep(1)

    except Exception as e:
        logger.error(f"任务调度器发生异常: {e}", exc_info=True)
        sys.exit(1)

    # 导入具体任务函数


from utils.set_system import change_language
from qmt_client import monitor_xt_client, generate_trading_report
from data_processing.sync_stock_data import sync_stock_data_main
from data_processing.source_trading_calendar import update_calendar
from models.Ensemb_model import run_ensemble_training
from models.deploy_model import buying_strategy
from qmt_client.qmt_trader import cancel_all_order_async

# 任务配置  
tasks = [
    TaskConfig(
        func=change_language,
        cron_expression="08:00,09:00,10:00,11:00,13:00,14:00",
        task_id="change_language",
        description="切换输入法为英文。",
        check_trading_day=True,
    ),
    TaskConfig(
        func=monitor_xt_client,
        cron_expression="08:00,09:20,12:30",
        task_id="monitor_xt_client",
        description="监控迅投客户端。",
        run_in_separate_process=True
    ),
    TaskConfig(
        func=generate_trading_report,
        cron_expression="09:00,11:35,15:05",
        task_id="generate_trading_report",
        description="生成交易报告。",
        check_trading_day=True
    ),
    TaskConfig(
        func=sync_stock_data_main,
        cron_expression="08:00,12:00,16:00",
        task_id="sync_stock_data_main",
        description="下载并同步股票数据到数据库。",
        check_trading_day=True
    ),
    TaskConfig(
        func=run_ensemble_training,
        cron_expression="08:10",
        task_id="run_ensemble_training",
        description="训练集成模型。",
        check_trading_day=True
    ),
    TaskConfig(
        func=buying_strategy,
        cron_expression="12:57",
        task_id="buying_strategy",
        description="买入策略。",
        run_in_separate_process=True,
        check_trading_day=True
    ),
    TaskConfig(
        func=cancel_all_order_async(),
        cron_expression="09:25",
        task_id="cancel_all_order_async",
        description="取消未成交订单。",
        run_in_separate_process=False,
        check_trading_day=True
    ),
]

# 主入口  
if __name__ == '__main__':
    # 设置多进程启动方法（Windows兼容）
    # 更新日期
    update_calendar()
    # 切换输入法
    change_language()

    if sys.platform == 'win32':
        multiprocessing.freeze_support()

        # 启动主调度程序
    main(tasks)