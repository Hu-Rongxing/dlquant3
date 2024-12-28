import datetime
import numpy as np
import pandas as pd
from dateutil import parser
from typing import Union, Literal, Any, Literal, Optional
import pytz
from logger import log_manager

logger = log_manager.get_logger(__name__)


def identify_timestamp_unit(timestamp: Union[np.int64, int]) -> Literal['ms', 'us', 'ns', 's']:
    """
    识别时间戳单位的更健壮版本
    """
    try:
        ts = abs(int(timestamp))  # 处理负数时间戳

        if 1_000_000_000_000 < ts < 10_000_000_000_000:
            return 'ms'
        elif 1_000_000_000_000_000 < ts < 10_000_000_000_000_000:
            return 'us'
        elif 1_000_000_000_000_000_000 < ts < 10_000_000_000_000_000_000:
            return 'ns'
        elif 1_000_000_000 < ts < 10_000_000_000:
            return 's'
        else:
            raise ValueError(f"无法识别的时间戳单位: {ts}")
    except Exception as e:
        logger.error(f"时间戳单位识别失败: {e}")
        raise


def convert_to_date(
        date_input: Any,
        to_timezone: str = 'Asia/Shanghai',
        format: str = '%Y-%m-%dT%H:%M:%S%z',
        return_type: Optional[Literal['str', 'datetime', 'date', 'timestamp']] = 'datetime'
) -> Union[str, datetime.datetime, datetime.date, pd.Timestamp]:
    """
    通用日期转换函数，支持多种输入类型和返回类型

    Args:
        date_input: 输入的日期
        to_timezone: 目标时区，默认为上海时区
        format: 输出的日期格式，默认为ISO 8601格式
        return_type: 返回的日期类型，可选 'str', 'datetime', 'date', 'timestamp'

    Returns:
        根据return_type返回不同类型的日期
    """
    try:
        # 处理 None 值
        if date_input is None:
            raise ValueError("输入不能为 None")

            # 获取目标时区
        timezone = pytz.timezone(to_timezone)

        # 存储转换后的本地化日期
        localized_dt = None

        # 类型处理的优先级：从最具体到最通用
        if isinstance(date_input, pd.Timestamp):
            # Pandas Timestamp 处理
            if date_input.tz is None:
                localized_dt = date_input.tz_localize(timezone)
            else:
                localized_dt = date_input.tz_convert(timezone)
            localized_dt = localized_dt.to_pydatetime()

        elif isinstance(date_input, datetime.datetime):
            # datetime 处理
            if date_input.tzinfo is None:
                localized_dt = timezone.localize(date_input)
            else:
                localized_dt = date_input.astimezone(timezone)

        elif isinstance(date_input, np.datetime64):
            # numpy.datetime64 处理
            localized_dt = timezone.localize(pd.Timestamp(date_input).to_pydatetime())

        elif isinstance(date_input, (np.int64, int)):
            # 时间戳处理
            unit = identify_timestamp_unit(date_input)
            localized_dt = pd.to_datetime(date_input, unit=unit).tz_localize('UTC').tz_convert(timezone).to_pydatetime()

        elif isinstance(date_input, datetime.date):
            # date 处理
            dt = datetime.datetime.combine(date_input, datetime.datetime.min.time())
            localized_dt = timezone.localize(dt)

        elif isinstance(date_input, str):
            # 字符串处理
            try:
                parsed_datetime = parser.parse(date_input)
                if parsed_datetime.tzinfo is None:
                    localized_dt = timezone.localize(parsed_datetime)
                else:
                    localized_dt = parsed_datetime.astimezone(timezone)
            except ValueError:
                logger.error(f"无法解析日期字符串: {date_input}")
                raise

        else:
            raise TypeError(f"不支持的日期类型: {type(date_input)}")

            # 根据返回类型转换
        if return_type == 'str':
            return localized_dt.strftime(format)
        elif return_type == 'datetime':
            return localized_dt
        elif return_type == 'date':
            return localized_dt.date()
        elif return_type == 'timestamp':
            return pd.Timestamp(localized_dt)
        else:
            raise ValueError(f"不支持的返回类型: {return_type}")

    except Exception as e:
        logger.error(f"日期转换失败: {e}")
        raise


def demo_convert_to_date():
    """
    演示日期转换的各种场景和返回类型
    """
    # 测试不同时区和返回类型
    test_scenarios = [
        {
            'input': np.datetime64('2024-01-15T23:30:00'),
            'timezone': 'UTC',
            'return_types': ['str', 'datetime', 'date', 'timestamp']
        },
        {
            'input': pd.Timestamp('2024-01-15 23:30:00'),
            'timezone': 'America/New_York',
            'return_types': ['str', 'datetime', 'date', 'timestamp']
        },
        {
            'input': datetime.datetime(2024, 1, 15, 23, 30, 0, tzinfo=pytz.UTC),
            'timezone': 'Europe/London',
            'return_types': ['str', 'datetime', 'date', 'timestamp']
        },
        {
            'input': 1705332600000,  # 毫秒时间戳
            'timezone': 'Asia/Shanghai',
            'return_types': ['str', 'datetime', 'date', 'timestamp']
        },
        {
            'input': '2024-01-15 23:30:00',
            'timezone': 'Asia/Tokyo',
            'return_types': ['str', 'datetime', 'date', 'timestamp']
        }
    ]

    # 遍历测试场景
    for scenario in test_scenarios:
        input_date = scenario['input']
        timezone = scenario['timezone']

        logger.info(f"测试输入: {input_date} (类型: {type(input_date)})")
        logger.info(f"目标时区: {timezone}")

        for return_type in scenario['return_types']:
            try:
                result = convert_to_date(
                    input_date,
                    to_timezone=timezone,
                    return_type=return_type
                )
                logger.info(f"返回类型 {return_type}: {result} (类型: {type(result)})")
            except Exception as e:
                logger.error(f"转换失败 - 返回类型: {return_type}, 错误: {e}")

                # 测试无效输入
    invalid_inputs = [
        'invalid date string',
        '2024-02-30',
        '2024-01-15T25:00:00',
        None,
        12345678901234567890
    ]

    logger.info("测试无效输入:")
    for invalid in invalid_inputs:
        try:
            convert_to_date(invalid)
        except Exception as e:
            logger.error(f"处理无效输入 {invalid} 时出错: {e}")


if __name__ == '__main__':
    demo_convert_to_date()