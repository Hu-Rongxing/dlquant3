
import secrets
# 导入 time 模块
from datetime import datetime, time

def generate_session_id():
    """
    生成一个基于日期和时间戳的唯一 32 位整数会话 ID，确保在同一天内不重复
    :return: 唯一的会话 ID 整数
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")

    date_int = int(date_str)
    random_number = secrets.randbelow(4096)

    session_id = (date_int << 12) | random_number
    return session_id


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


from datetime import datetime, time

def is_china_market_open():
    # 获取当前时间
    current_time = datetime.now()

    # 定义交易时间
    morning_open = time(9, 30)  # 9:30 AM
    morning_close = time(11, 30)  # 11:30 AM
    afternoon_open = time(13, 0)  # 1:00 PM
    afternoon_close = time(15, 0)  # 3:00 PM

    # 判断当前时间是否在交易时间内
    if current_time.weekday() < 5:  # 周一到周五
        if (morning_open <= current_time.time() <= morning_close) or \
           (afternoon_open <= current_time.time() <= afternoon_close):
            return True
    return False

