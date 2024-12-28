import pandas as pd
from typing import List, Optional, Callable, Dict
import xtquant.xtdata as xtdata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table, Column, String, BigInteger, DateTime, Float, Date
import numpy as np

# 自定义模块
from logger import log_manager
from utils.retry import retry
from .get_securities import get_investment_target


applogger = log_manager.get_logger(__name__)


# 确保只保留需要的列
COLUMNS_TO_INSERT = [
    'id', 'time', 'stock_code', 'date', 'open', 'close',
    'high', 'low', 'volume', 'amount', 'modify_time'
]


@retry(max_attempts=3, delay=3, exceptions=(Exception,))
def download_stock_data(
        stock: str,
        period: str,
        start_time: str,
        end_time: str,
        incrementally: bool
) -> bool:
    """下载单只股票历史数据"""
    try:
        xtdata.download_history_data(stock, period, start_time, end_time, incrementally=incrementally)
        applogger.info(f"成功下载股票数据：{stock}")
        return True
    except Exception as e:
        applogger.exception(f"下载股票数据失败：{stock}，错误信息：{e}")
        raise


def download_history_data(
        stock_list: Optional[List[str]] = None,
        period: str = '1d',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        callback: Optional[Callable[[str, bool], None]] = None,
        incrementally: bool = False,
        max_workers: int = 10
) -> bool:
    """并行下载历史数据"""
    # 参数预处理
    stock_list = stock_list or get_investment_target().securities.to_list()
    start_time = start_time or '20160101'
    end_time = end_time or datetime.now().strftime('%Y%m%d')

    success = True
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 dict 推导式简化代码
        futures = {
            executor.submit(download_stock_data, stock, period, start_time, end_time, incrementally): stock
            for stock in stock_list
        }

        # 使用生成器表达式优化内存
        results = (
            (futures[future], future.result())
            for future in as_completed(futures)
        )

        for stock, result in results:
            try:
                if callback:
                    callback(stock, result)
                if not result:
                    success = False
            except Exception as e:
                applogger.error(f"下载股票数据异常：{stock}，错误：{e}")
                if callback:
                    callback(stock, False)
                success = False

    return success


def preprocess_market_data(
        market_data: Dict[str, pd.DataFrame],
        columns_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    增强版数据预处理

    新增功能：
    1. 更严格的数据清洗
    2. 异常值处理
    3. 性能优化
    """
    final_mapping = columns_mapping or {}
    processed_dataframes = []
    now = datetime.now()

    # 使用更高效的处理方式
    for stock_code, df in market_data.items():
        if df.empty:
            applogger.warning(f"股票 {stock_code} 无数据")
            continue

        try:
            # 使用更安全的数据处理方式
            df = df.copy()
            df.index.name = 'date'
            df.sort_index(inplace=True)
            df.reset_index(inplace=True)

            # 日期转换和验证
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)

            # 列名标准化
            df.columns = df.columns.str.lower()
            df.rename(columns=final_mapping, inplace=True)

            # 添加额外信息
            df['modify_time'] = now
            df['stock_code'] = stock_code
            df['id'] = df['stock_code'] + '_' + df['date'].astype(str)

            # 严格的数据类型转换和清洗
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # 异常值处理：去除极端值
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower_bound, upper_bound)

            processed_dataframes.append(df)

        except Exception as e:
            applogger.error(f"处理股票 {stock_code} 数据时发生错误: {e}")

            # 使用更高效的合并方式
    return pd.concat(processed_dataframes, ignore_index=True) if processed_dataframes else pd.DataFrame()


def batch_upsert_data(
        engine: sqlalchemy.engine.base.Engine,
        df: pd.DataFrame,
        batch_size: int = 5000,
        table_name: str = 'stock_data'
):
    """
    优化批量 Upsert 数据

    新增功能：
    1. 更好的错误处理
    2. 性能监控
    3. 更详细的日志
    """
    import time

    # 创建表定义（保持不变）
    metadata = MetaData()
    stock_data = Table(table_name, metadata,
                       Column('id', String, primary_key=True),
                       Column('time', BigInteger, primary_key=True),
                       Column('stock_code', String),
                       Column('date', Date),
                       Column('open', Float),
                       Column('close', Float),
                       Column('high', Float),
                       Column('low', Float),
                       Column('volume', Float),
                       Column('amount', Float),
                       Column('modify_time', DateTime)
                       )

    df_to_insert = df[COLUMNS_TO_INSERT].copy()
    df_to_insert.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    start_time = time.time()
    total_rows = len(df_to_insert)
    processed_rows = 0

    with engine.begin() as connection:
        def _upsert_batch(batch):
            nonlocal processed_rows
            try:
                insert_stmt = insert(stock_data).values(batch.to_dict('records'))
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        col: insert_stmt.excluded[col]
                        for col in batch.columns if col != 'id'
                    }
                )
                connection.execute(upsert_stmt)
                processed_rows += len(batch)

                # 实时日志
                applogger.info(f"已处理 {processed_rows}/{total_rows} 行数据")

            except Exception as e:
                applogger.error(f"批次插入失败: {e}")
                raise

        for i in range(0, len(df_to_insert), batch_size):
            batch = df_to_insert.iloc[i:i + batch_size]
            _upsert_batch(batch)

            # 性能日志
    elapsed_time = time.time() - start_time
    applogger.info(f"数据插入完成，总耗时 {elapsed_time:.2f} 秒，处理 {total_rows} 行数据")


def get_data_from_local(
        period: str = '1d',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        stock_list: Optional[List[str]] = None,
        incrementally: bool = False,
        columns_mapping: Optional[Dict[str, str]] = None,
        max_workers: int = 10,
) -> pd.DataFrame:
    """
    增强版本地数据获取函数

    新增功能：
    1. 更详细的异常处理
    2. 性能监控
    3. 数据质量检查
    """
    import time

    start = time.time()

    try:
        # 参数验证
        if not stock_list:
            stock_list = get_investment_target().securities.to_list()

            # 下载数据
        download_history_data(
            stock_list=stock_list,
            period=period,
            start_time=start_time or '20160101',
            end_time=end_time or datetime.now().strftime('%Y%m%d'),
            incrementally=incrementally,
            max_workers=max_workers
        )

        # 获取本地数据
        market_data = xtdata.get_local_data(
            field_list=[],
            stock_list=stock_list,
            period=period,
            start_time=start_time or '20160101',
            end_time=end_time or datetime.now().strftime('%Y%m%d'),
            count=-1,
            dividend_type='front',
            fill_data=True
        )

        # 预处理数据，支持自定义列名映射
        # 预处理数据
        combined_df = preprocess_market_data(market_data, columns_mapping)

        # 数据质量检查
        if combined_df.empty:
            applogger.warning("未获取到任何股票数据")
            return pd.DataFrame()

            # 性能和质量日志
        elapsed_time = time.time() - start
        applogger.info(
            f"数据获取完成，耗时 {elapsed_time:.2f} 秒，"
            f"获取 {len(stock_list)} 只股票，"
            f"总记录数 {len(combined_df)}"
        )

        return combined_df

    except Exception as e:
        applogger.error(f"获取股票数据失败：{e}", exc_info=True)
        raise


if __name__ == '__main__':
    df = get_data_from_local()
    print(df)