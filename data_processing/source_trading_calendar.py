import pandas_market_calendars as mcal
import pandas as pd
from sqlalchemy import Column, Date, Integer, String, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import datetime
from typing import List, Optional, Dict

# 导入你的数据库配置
from utils.database import create_sqlalchemy_engine
from utils.date_utils import convert_to_date

# 创建 SQLAlchemy 基类和引擎
Base = declarative_base()
engine = create_sqlalchemy_engine()
SessionLocal = sessionmaker(bind=engine)

from logger import log_manager
logger = log_manager.get_logger(__name__)

# 扩展交易日模型
class TradingDay(Base):
    __tablename__ = 'trading_days'

    trade_date = Column(Date, primary_key=True)
    year = Column(Integer, index=True)
    month = Column(Integer, index=True)
    week_of_year = Column(Integer, index=True)
    week_of_month = Column(Integer)
    day_of_year = Column(Integer)
    day_of_month = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    is_trading_day = Column(Boolean)
    quarter = Column(Integer)
    season = Column(String(10))


class ChineseMarketCalendar:
    def __init__(self, start_year: int = 2000, end_year: Optional[int] = None):
        """
        初始化中国股市交易日历

        Args:
            start_year (int): 开始年份
            end_year (int, optional): 结束年份，默认为当前年份
        """
        self.start_year = start_year
        self.end_year = end_year or datetime.datetime.now().year

    def _convert_to_date(self, date) -> datetime.date:
        """转换输入为日期格式"""
        if not isinstance(date, datetime.date):
            try:
                date = convert_to_date(date)
            except Exception as e:
                logger.error(f"日期转换失败: {e}")
                raise
        return date

    def _get_date_metadata(self, date: datetime.date) -> Dict:
        """获取日期的元数据"""
        return {
            'trade_date': date,
            'year': date.year,
            'month': date.month,
            'week_of_year': date.isocalendar()[1],
            'week_of_month': (date.day - 1) // 7 + 1,
            'day_of_year': date.timetuple().tm_yday,
            'day_of_month': date.day,
            'day_of_week': date.weekday(),
            'is_weekend': date.weekday() >= 5,
            'is_trading_day': False,  # 默认为非交易日
            'quarter': (date.month - 1) // 3 + 1,
            'season': self._get_season(date)
        }

    def _get_season(self, date: datetime.date) -> str:
        """获取季节"""
        month = date.month
        if month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        elif month in [9, 10, 11]:
            return '秋季'
        else:
            return '冬季'

    def get_trading_days(self) -> List[Dict]:
        """获取中国股市交易日历，包含元数据"""
        try:
            sse = mcal.get_calendar('XSHG')
            start_date = f'{self.start_year}-01-01'
            end_date = f'{self.end_year}-12-31'
            trading_days = sse.schedule(start_date=start_date, end_date=end_date)

            return [
                {**self._get_date_metadata(date.date()), 'is_trading_day': True}
                for date in trading_days.index
            ]

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

    def generate_full_calendar(self) -> List[Dict]:
        """生成完整的日历，包括非交易日"""
        full_calendar = []
        start_date = datetime.date(self.start_year, 1, 1)
        end_date = datetime.date(self.end_year, 12, 31)
        trading_days = {day['trade_date'] for day in self.get_trading_days()}

        current_date = start_date
        while current_date <= end_date:
            date_metadata = self._get_date_metadata(current_date)
            date_metadata['is_trading_day'] = current_date in trading_days
            full_calendar.append(date_metadata)
            current_date += datetime.timedelta(days=1)

        return full_calendar

    def is_trading_day(self, check_date: Optional[datetime.date] = None) -> bool:
        """判断指定日期是否为交易日"""
        check_date = self._convert_to_date(check_date or datetime.date.today())

        with SessionLocal() as session:
            result = session.execute(
                text("SELECT is_trading_day FROM trading_days WHERE trade_date = :date"),
                {"date": check_date}
            ).scalar()

            if result is None:
                logger.info(f"未找到 {check_date} 的交易日信息，尝试更新")
                self.save_to_database(calendar_type='full', years_to_update=[check_date.year])
                result = session.execute(
                    text("SELECT is_trading_day FROM trading_days WHERE trade_date = :date"),
                    {"date": check_date}
                ).scalar()

            return result if result is not None else False

    def _get_trading_day(self, reference_date: datetime.date, direction: str) -> Optional[datetime.date]:
        """获取指定日期的最近交易日"""
        with SessionLocal() as session:
            operator = '<' if direction == 'previous' else '>'
            result = session.execute(
                text(f"""  
                SELECT trade_date  
                FROM trading_days  
                WHERE is_trading_day = true AND trade_date {operator} :ref_date  
                ORDER BY trade_date { 'DESC' if direction == 'previous' else 'ASC' }  
                LIMIT 1  
                """),
                {"ref_date": reference_date}
            ).scalar()

            return result

    def get_last_trading_day(self, reference_date: Optional[datetime.date] = None) -> Optional[datetime.date]:
        """获取指定日期之前最近的一个交易日"""
        reference_date = self._convert_to_date(reference_date or datetime.date.today())
        return self._get_trading_day(reference_date, 'previous')

    def get_next_trading_day(self, reference_date: Optional[datetime.date] = None) -> Optional[datetime.date]:
        """获取指定日期之后最近的一个交易日"""
        reference_date = self._convert_to_date(reference_date or datetime.date.today())
        return self._get_trading_day(reference_date, 'next')

    def get_trading_days_range(self, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> List[datetime.date]:
        """获取指定日期范围内的所有交易日"""
        start_date = self._convert_to_date(start_date or datetime.date(datetime.date.today().year, 1, 1))
        end_date = self._convert_to_date(end_date or datetime.date(datetime.date.today().year, 12, 31))

        with SessionLocal() as session:
            result = session.execute(
                text("""  
                SELECT trade_date  
                FROM trading_days  
                WHERE is_trading_day = true  
                  AND trade_date BETWEEN :start_date AND :end_date  
                ORDER BY trade_date  
                """),
                {
                    "start_date": start_date,
                    "end_date": end_date
                }
            ).scalars().all()

            return list(result)

    def get_trading_days_count(self, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> int:
        """获取指定日期范围内的交易日数量"""
        start_date = self._convert_to_date(start_date or datetime.date(datetime.date.today().year, 1, 1))
        end_date = self._convert_to_date(end_date or datetime.date(datetime.date.today().year, 12, 31))

        with SessionLocal() as session:
            result = session.execute(
                text("""  
                SELECT COUNT(*)  
                FROM trading_days  
                WHERE is_trading_day = true  
                  AND trade_date BETWEEN :start_date AND :end_date  
                """),
                {
                    "start_date": start_date,
                    "end_date": end_date
                }
            ).scalar()

            return result or 0

    def save_to_database(self, calendar_type: str = 'trading', years_to_update: Optional[List[int]] = None):
        """将日历保存到数据库"""
        def validate_year(year: int) -> bool:
            """验证年份是否在合理范围内"""
            return 1990 <= year <= 3000

        with SessionLocal() as session:
            if years_to_update is None:
                years_to_update = [datetime.datetime.now().year]

            valid_years = [year for year in years_to_update if validate_year(year)]
            if not valid_years:
                logger.warning(f"提供的年份列表 {years_to_update} 无效")
                return

            invalid_years = set(years_to_update) - set(valid_years)
            if invalid_years:
                logger.warning(f"以下年份被过滤: {invalid_years}")

            calendar_data = []
            for year in valid_years:
                self.start_year = year
                self.end_year = year
                calendar_data.extend(
                    self.get_trading_days() if calendar_type == 'trading'
                    else self.generate_full_calendar()
                )

            if not calendar_data:
                logger.warning("没有生成任何日历数据")
                return

            stmt = insert(TradingDay).values(calendar_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['trade_date'],
                set_={
                    'year': stmt.excluded.year,
                    'month': stmt.excluded.month,
                    'week_of_year': stmt.excluded.week_of_year,
                    'week_of_month': stmt.excluded.week_of_month,
                    'day_of_year': stmt.excluded.day_of_year,
                    'day_of_month': stmt.excluded.day_of_month,
                    'day_of_week': stmt.excluded.day_of_week,
                    'is_weekend': stmt.excluded.is_weekend,
                    'is_trading_day': stmt.excluded.is_trading_day,
                    'quarter': stmt.excluded.quarter,
                    'season': stmt.excluded.season
                }
            )

            result = session.execute(stmt)
            session.commit()
            logger.info(f"成功插入/更新 {result.rowcount} 个日期")

    def analyze_calendar(self):
        """分析日历统计信息"""
        with SessionLocal() as session:
            trading_days_count = session.execute(
                text("SELECT COUNT(*) FROM trading_days WHERE is_trading_day = true")
            ).scalar()

            yearly_trading_days = session.execute(
                text("""  
                SELECT year, COUNT(*) as trading_days  
                FROM trading_days  
                WHERE is_trading_day = true  
                GROUP BY year  
                ORDER BY year  
                """)
            ).fetchall()

            logger.info("交易日统计:")
            logger.info(f"总交易日数: {trading_days_count}")
            logger.info("\n年度交易日数:")
            for year, days in yearly_trading_days:
                logger.info(f"{year}年: {days}天")

    def save_full_calendar(self, start_year: int = 1991, end_year: int = 2025):
        """保存完整的日历到数据库"""
        self.start_year = start_year
        self.end_year = end_year
        self.save_to_database(calendar_type='full')
        logger.info(f"已保存 {start_year}-{end_year} 的完整日历")


    def get_trading_calendar(self) -> Optional[pd.DataFrame]:
        """获取最近的交易日并返回为 DataFrame"""
        try:
            with SessionLocal() as session:
                # 使用 pd.read_sql 直接读取 SQL 查询结果到 DataFrame
                df = pd.read_sql(
                    """  
                    SELECT trade_date  
                    FROM trading_days  
                    WHERE is_trading_day = true  
                    ORDER BY trade_date ASC 
                    """,
                    con=session.bind  # 使用当前会话的数据库连接
                )

                # 处理 None 的情况
                return df if not df.empty else None

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return None  # 在发生异常时返回 None


def main():
    # 创建表（如果不存在）
    Base.metadata.create_all(bind=engine)

    # 初始化日历
    calendar = ChineseMarketCalendar()

    # 写入完整日历
    for i in range(1991, 2030):
        logger.info(f"正在写入完整日历：{i}年")
        calendar.save_to_database(calendar_type='trading', years_to_update=[i])

    # 分析日历
    calendar.analyze_calendar()


def update_calendar(calendar_type='full'):
    """更新当年和下一年的交易日"""
    year = int(datetime.datetime.now().year)
    calendar = ChineseMarketCalendar()
    calendar.save_to_database(calendar_type=calendar_type, years_to_update=[year, year + 1])


if __name__ == '__main__':
    calendar = ChineseMarketCalendar()

    # 获取上一个交易日
    previous_trading_day = calendar.get_last_trading_day()
    print(f"上一个交易日: {previous_trading_day}")

    # 获取下一个交易日
    next_trading_day = calendar.get_next_trading_day()
    print(f"下一个交易日: {next_trading_day}")

    # 获取今年的交易日列表
    trading_days = calendar.get_trading_days_range('20241228','20251227')
    print(f"指定范围的交易日: {trading_days}")
    print(f"今年交易日数量: {len(trading_days)}")

    # 获取今年的交易日数量
    trading_days_count = calendar.get_trading_days_count()
    print(f"今年交易日数量: {trading_days_count}")

    # 获取特定日期范围的交易日
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 6, 30)
    specific_trading_days = calendar.get_trading_days_range(start_date, end_date)
    print(f"{start_date} 到 {end_date} 的交易日数量: {len(specific_trading_days)}")

    print(calendar.get_trading_calendar())