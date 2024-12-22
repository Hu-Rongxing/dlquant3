import ctypes
import subprocess
import time
import os
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict, Tuple
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor

import cv2
import psutil
import pyautogui
import win32con
import win32gui
from multiprocessing import Lock, Process, Queue
from pywinauto import Application, findwindows
from pywinauto.findwindows import ElementNotFoundError

# 自定义模块
from config import settings
from logger import log_manager
from utils.multi_process import SingletonMeta

# 日志配置
app_logger = log_manager.get_logger(__name__)


class ProcessStatus(Enum):
    """进程状态枚举"""
    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()


class WindowFinderError(Exception):
    """窗口查找相关的自定义异常"""
    pass


class ProcessMonitorError(Exception):
    """进程监控相关的自定义异常"""
    pass


class WindowRegexFinder:
    """
    窗口查找和交互的高级工具类

    提供基于正则表达式的窗口查找、DPI缩放检测、窗口置顶和交互等功能
    """

    def __init__(
            self,
            regex_pattern: str,
            timeout: int = 10,
            retry_interval: float = 1.0
    ):
        """
        初始化窗口查找器

        :param regex_pattern: 用于匹配窗口标题的正则表达式
        :param timeout: 查找窗口的最大超时时间（秒）
        :param retry_interval: 重试间隔时间（秒）
        """
        self.regex_pattern: str = regex_pattern
        self.timeout: int = timeout
        self.retry_interval: float = retry_interval

        self.app: Optional[Application] = None
        self.window: Optional[Application.WindowSpecification] = None
        self.handle: Optional[int] = None
        self.scaling_factor: float = self._get_scaling_factor()

    def _get_scaling_factor(self) -> float:
        """
        获取Windows的DPI缩放系数

        :return: DPI缩放因子，默认为1.0
        """
        try:
            dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
            app_logger.debug(f"DPI缩放系数: {dpi_scale}")
            return dpi_scale
        except Exception as e:
            app_logger.warning(f"获取DPI缩放系数失败: {e}")
            return 1.0

    def find_window(self) -> Optional[int]:
        """
        查找符合正则表达式的窗口，支持超时重试

        :return: 窗口句柄，如果未找到返回None
        :raises WindowFinderError: 超时未找到窗口
        """
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                windows = findwindows.find_windows(title_re=self.regex_pattern)
                if windows:
                    self.handle = windows[0]
                    app_logger.info(f"找到窗口句柄: {self.handle}")
                    self.app = Application(backend="uia").connect(handle=self.handle)
                    self.window = self.app.window(handle=self.handle)
                    return self.handle
            except ElementNotFoundError as e:
                app_logger.debug(f"查找窗口失败: {e}")
            time.sleep(self.retry_interval)

        raise WindowFinderError(
            f"超时：未找到符合模式 {self.regex_pattern} 的窗口"
        )

    def bring_window_to_top(self) -> None:
        """
        将窗口置顶并聚焦，增加错误处理和重试机制

        :raises WindowFinderError: 窗口置顶失败
        """
        if not self.handle:
            try:
                self.find_window()
            except WindowFinderError as e:
                app_logger.error(e)

        try:

            win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
            win32gui.SetForegroundWindow(self.handle)
            time.sleep(0.5)  # 确保窗口状态稳定

            # 重新连接窗口以确保最新状态
            self.find_window()

            app_logger.debug(f"窗口 {self.handle} 已置顶并聚焦。")
        except Exception as e:
            app_logger.error(f"无法将窗口置顶或连接：{e}")
            raise WindowFinderError(f"窗口置顶失败：{e}")

    def find_and_click_button(self, button_text: str) -> None:
        """
        通过按钮文本查找并点击按钮

        :param button_text: 按钮文本
        :raises WindowFinderError: 未找到按钮
        """
        if not self.window:
            self.find_window()
            # raise WindowFinderError("未设置窗口。请先调用 find_window() 和 bring_window_to_top() 方法。")

        try:
            button = self.window.child_window(title=button_text, control_type="Button")
            if button.exists(timeout=5):
                button.click_input()
                app_logger.debug("按钮点击成功！")
            else:
                raise WindowFinderError(f"未找到文本为 '{button_text}' 的按钮！")
        except ElementNotFoundError as e:
            app_logger.exception(f"未找到文本为 '{button_text}' 的按钮：{e}")
            raise WindowFinderError(f"按钮查找失败：{e}")

    def find_and_click_image_button(self, image_path: str) -> None:
        """通过图像查找并点击按钮"""
        app_logger.debug(f"查找路径 {image_path} 中的按钮图像")

        # self.bring_window_to_top()

        print(Path(image_path).absolute().as_posix())
        try:
            if not Path(image_path).is_file():
                print(Path(image_path).absolute().as_posix())
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            image = cv2.imread(image_path)
            scaling_factor = self._get_scaling_factor()
            scaled_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor)
            cv2.imwrite("tmp.PNG", scaled_image)
            app_logger.debug("保存图片")
            time.sleep(1)
            if image is None:
                raise Exception(f"图像未加载。检查路径: {image_path}")

            button_location = pyautogui.locateOnScreen("tmp.PNG", confidence=0.8)
            time.sleep(0.5)
            app_logger.debug("删除图片")
            os.remove("tmp.PNG")

            if button_location:
                button_point = pyautogui.center(button_location)
                pyautogui.moveTo(button_point.x, button_point.y)  # 移动到按钮位置
                app_logger.debug(f"移动到按钮位置: {button_point}")
                time.sleep(1)  # 减少暂停时间
                pyautogui.click()
                app_logger.debug("图像按钮点击成功！")
            else:

                raise Exception(f"屏幕上未找到按钮。图像路径: {image_path}")
        except Exception as e:
            app_logger.exception(f"点击图像按钮时出错：{e}")
            raise


class ProgramMonitor(metaclass=SingletonMeta):
    """
    程序监控类，负责管理交易客户端的启动、停止和监控
    """
    MINIXT_PROCESS_NAME = "XtMiniQmt.exe"
    LOGIN_PROCESS_NAME = "XtItClient.exe"

    def __init__(
            self,
            program_path: str,
            check_interval: int = 10 * 60
    ):
        """
        初始化程序监控器

        :param program_path: 程序启动路径
        :param check_interval: 检查间隔时间（秒）
        """
        self.program_path = program_path
        self.check_interval = check_interval
        self.lock = Lock()
        self.task_queue: Queue = Queue()

    def check_process_status(self, process_name: str) -> ProcessStatus:
        """
        检查进程状态

        :param process_name: 进程名称
        :return: 进程状态
        """
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == process_name:
                    return ProcessStatus.RUNNING
            return ProcessStatus.STOPPED
        except Exception as e:
            app_logger.error(f"检查进程状态时发生错误: {e}")
            return ProcessStatus.ERROR

    def start_program(self) -> None:
        """
        启动程序并执行自动登录

        :param auto_login: 是否自动登录
        :raises ProcessMonitorError: 启动程序失败
        """
        auto_login = bool(settings.get("xt_client.auto_login"))
        with self.lock:
            if self.check_process_status(self.MINIXT_PROCESS_NAME) == ProcessStatus.RUNNING:
                app_logger.info("迅投程序已运行，无需启动。")
                return

            try:
                process = subprocess.Popen(
                    self.program_path,
                )
                app_logger.info(f"程序 {self.program_path} 已启动。")
            except Exception as e:
                error_msg = f"无法启动程序 {self.program_path}：{e}"
                app_logger.exception(error_msg)
                raise ProcessMonitorError(error_msg)

            time.sleep(20)  # 等待程序启动

            if self.check_process_status(self.LOGIN_PROCESS_NAME) == ProcessStatus.RUNNING:
                try:
                    finder = WindowRegexFinder(r"e海方舟-量化交易版[.\d ]+")
                    finder.find_window()
                    finder.bring_window_to_top()
                    app_logger.debug('暂停')
                    time.sleep(1)
                    print(auto_login)
                    if not auto_login:
                        app_logger.debug("查找登录按钮")
                        image_path = settings.get("xt_client.xt_login_button_png", "config/xt_login_button.PNG")
                        finder.find_and_click_image_button(image_path)
                except Exception as e:

                    app_logger.error(f"登录处理失败：{e}")

            time.sleep(15)

    def stop_program(self) -> None:
        """停止交易客户端进程"""
        with self.lock:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in {self.MINIXT_PROCESS_NAME, self.LOGIN_PROCESS_NAME}:
                    try:
                        proc.terminate()
                        app_logger.info(f"程序 {proc.info['name']} 已停止。")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                        app_logger.error(f"无法停止程序 {proc.info['name']}：{e}")

    def restart_program(self) -> None:
        """重启交易客户端"""
        app_logger.info("正在重启程序...")
        self.stop_program()
        time.sleep(5)  # 等待进程完全结束
        self.start_program()

    def monitor(self) -> None:
        """持续监控程序运行状态"""
        while True:
            if self.check_process_status(self.MINIXT_PROCESS_NAME) != ProcessStatus.RUNNING:
                app_logger.warning(f"检测到 {self.MINIXT_PROCESS_NAME} 未启动，正在启动...")
                self.restart_program()
            else:
                app_logger.info(f"{self.MINIXT_PROCESS_NAME} 正在运行。")
            time.sleep(self.check_interval)


def start_xt_client(
) -> ProgramMonitor:
    """
    启动迅投客户端监控器
    :return: ProgramMonitor实例
    """
    program_path: str = settings.get("xt_client.program_dir")
    check_interval: int = 60
    try:
        xt_client = ProgramMonitor(program_path, check_interval)
        xt_client.start_program()
        return xt_client
    except Exception as e:
        app_logger.exception(f"启动客户端失败：{e}")
        raise


def restart_xt_client(
) -> ProgramMonitor:
    """
    启动迅投客户端监控器

    :return: ProgramMonitor实例
    """
    program_path = settings.get('xt_client.program_dir', "C:/e_trader/bin.x64/XtItClient.exe")
    check_interval = 30
    try:
        xt_client = ProgramMonitor(program_path, check_interval)
        xt_client.restart_program()
        return xt_client
    except Exception as e:
        app_logger.exception(f"启动客户端失败：{e}")
        raise


def monitor_xt_client():
    """
    监控迅投客户端，如果没有启动，则启动客户端。
    注意：该函数会阻塞进程。
    """
    try:
        # 从配置获取程序路径
        program_path = settings.get('xt_client.program_dir', "C:/e_trader/bin.x64/XtItClient.exe")

        if not program_path:
            raise ValueError("未配置程序路径")

        monitor = start_xt_client()

        # 创建并启动工作进程
        worker_process = Process(target=monitor.monitor, daemon=True)
        worker_process.start()
        worker_process.join()

    except Exception as e:
        app_logger.critical(f"程序启动失败：{e}")


if __name__ == "__main__":
    # 放入单独的进程中运行
    monitor_xt_client()
