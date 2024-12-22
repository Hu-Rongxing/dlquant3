from qmt_client import (
    restart_xt_client,
    monitor_xt_client
)
from utils.multi_process import run_in_thread


def test_qmt_monitor():
    restart_xt_client()
    @run_in_thread
    def run_monitor():
        # 会阻塞进程
        monitor_xt_client()
    run_monitor()