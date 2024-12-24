from data_processing.source_trading_calendar import ChineseMarketCalendar, update_calendar
import datetime
from logger import log_manager
logger = log_manager.get_logger(__name__)

# 添加测试函数
def test_trading_day_functions():
    """
    测试交易日相关函数
    """

    calendar = ChineseMarketCalendar()

    update_calendar()

    # 测试今天是否为交易日
    today = datetime.date.today()
    is_today_trading = calendar.is_trading_day(today)
    logger.info(f"{today} 是否为交易日: {is_today_trading}")

    # 测试获取最近的交易日
    last_trading_day = calendar.get_last_trading_day()
    logger.info(f"离{today}最近的交易日: {last_trading_day}")

    # 测试获取下一个交易日
    next_trading_day = calendar.get_next_trading_day()
    logger.info(f"{today}的下一个交易日: {next_trading_day}")

    # 测试特定日期
    test_date = datetime.date(2024, 1, 1)
    is_test_date_trading = calendar.is_trading_day(test_date)
    logger.info(f"{test_date} 是否为交易日: {is_test_date_trading}")

    # 测试获取上一个交易日
    previous_trading_day = calendar.get_previous_trading_day(today)
    logger.info(f"{today} 的上一个交易日: {previous_trading_day}")

    # 测试获取指定范围的交易日
    start_date = datetime.date(datetime.date.today().year, 1, 1)
    end_date = datetime.date(datetime.date.today().year, 12, 31)
    trading_days = calendar.get_trading_days_range(start_date, end_date)
    logger.info(f"{start_date} 到 {end_date} 的交易日数量: {len(trading_days)}")

    # 测试获取交易日数量
    trading_days_count = calendar.get_trading_days_count(start_date, end_date)
    logger.info(f"{start_date} 到 {end_date} 的交易日数量: {trading_days_count}")