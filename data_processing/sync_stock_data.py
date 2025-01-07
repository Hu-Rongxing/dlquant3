import pandas as pd
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy import text

# 自定义模块
from logger import log_manager
from utils.database import create_sqlalchemy_engine
from utils.date_utils import convert_to_date
from .source_stock_data import (
    get_data_from_local,
    preprocess_market_data,
    batch_upsert_data
)

logger = log_manager.get_logger(__name__)


class StockDataSyncManager:
    """
    股票数据同步管理器，提供增量更新和全量同步功能
    """

    def __init__(self, database: str = 'quant'):
        """
        初始化数据库引擎

        Args:
            database (str): 数据库名称
        """
        self.engine = create_sqlalchemy_engine(database)

    def get_last_sync_date(
            self,
            table_name: str = 'stock_data'
    ) -> Optional[str]:
        """
        获取最后一次同步的日期

        Args:
            table_name (str): 目标表名

        Returns:
            Optional[str]: 最后同步日期，格式为 'YYYYMMDD'
        """
        try:
            query = f"""  
            SELECT MAX(date) as last_date   
            FROM {table_name}  
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                last_date = result.scalar()

            return last_date
        except Exception as e:
            logger.error(f"获取最后同步日期失败：{e}")
            return None

    def sync_stock_data(
            self,
            period: str = '1d',
            stock_list: Optional[List[str]] = None,
            batch_size: int = 5000,
            table_name: str = 'stock_data',
            force_full_sync: bool = False,
            days_to_sync: int = 5  # 默认同步最近一年的数据
    ):
        """
        同步股票数据，支持增量和全量更新

        Args:
            period (str): 数据周期
            stock_list (Optional[List[str]]): 股票列表
            batch_size (int): 批次大小
            table_name (str): 目标表名
            force_full_sync (bool): 是否强制全量同步
            days_to_sync (int): 同步的天数范围
        """
        try:
            # 确定开始时间
            if force_full_sync:
                start_time = '20160101'  # 全量同步从最早时间开始
                incrementally = False
                logger.info(f"强制全量同步更新，开始日期：{start_time}")
            else:
                last_sync_date = self.get_last_sync_date(table_name)
                logger.info(f"增量同步更新，开始日期：{last_sync_date}")

                if last_sync_date:
                    # 转换为日期并计算同步起始日期
                    start_date = convert_to_date(last_sync_date,)
                    start_time = (start_date - timedelta(days=7)).strftime('%Y%m%d')  # 额外同步7天防止数据遗漏
                    incrementally = True
                else:
                    start_time = (datetime.now() - timedelta(days=days_to_sync)).strftime('%Y%m%d')
                    incrementally = False

            end_time = datetime.now().strftime('%Y%m%d')

            logger.info(f"开始同步数据：{start_time} - {end_time}")
            logger.info(f"增量同步模式：{incrementally}")

            # 获取本地数据
            market_data = get_data_from_local(
                period=period,
                start_time=start_time,
                end_time=end_time,
                stock_list=stock_list,
                incrementally=incrementally
            )

            if market_data.empty:
                logger.warning("未获取到新的股票数据")
                return

                # 批量更新数据
            batch_upsert_data(
                engine=self.engine,
                df=market_data,
                batch_size=batch_size,
                table_name=table_name
            )

            logger.info(f"成功同步 {len(market_data)} 条股票数据")

        except Exception as e:
            logger.error(f"股票数据同步失败：{e}")
            raise

    def get_sync_status(
            self,
            table_name: str = 'stock_data'
    ) -> Dict[str, Union[str, int]]:
        """
        获取数据同步状态

        Args:
            table_name (str): 目标表名

        Returns:
            Dict[str, Union[str, int]]: 同步状态信息
        """
        try:
            with self.engine.connect() as conn:
                # 获取最后同步日期
                last_date_query = f"SELECT MAX(date) as last_date FROM {table_name}"
                last_date = conn.execute(text(last_date_query)).scalar()

                # 获取总记录数
                total_records_query = f"SELECT COUNT(*) as total_records FROM {table_name}"
                total_records = conn.execute(text(total_records_query)).scalar()

                # 获取股票代码数量
                unique_stocks_query = f"SELECT COUNT(DISTINCT stock_code) as unique_stocks FROM {table_name}"
                unique_stocks = conn.execute(text(unique_stocks_query)).scalar()

            return {
                'last_sync_date': last_date,
                'total_records': total_records,
                'unique_stocks': unique_stocks
            }

        except Exception as e:
            logger.error(f"获取同步状态失败：{e}")
            return {}


def sync_stock_data_main(force_full_sync=False):

    try:
        # 创建同步管理器
        sync_manager = StockDataSyncManager()

        # 查看当前同步状态
        sync_status = sync_manager.get_sync_status()
        logger.info("当前同步状态：")
        for key, value in sync_status.items():
            logger.info(f"{key}: {value}")

            # 执行数据同步
        sync_manager.sync_stock_data(
            period='1d',  # 日线数据
            force_full_sync=force_full_sync,  # 增量同步
            days_to_sync=5  # 同步最近一年的数据
        )

        # 再次检查同步状态
        updated_sync_status = sync_manager.get_sync_status()
        logger.info("\n同步后状态：")
        for key, value in updated_sync_status.items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.exception(f"主程序执行失败：{e}")


if __name__ == '__main__':
    sync_stock_data_main()