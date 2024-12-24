import pandas as pd
from typing import List, Optional, Callable, Dict
import xtquant.xtdata as xtdata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table, Column, String, DateTime, Float
import numpy as np

# 自定义模块
from logger import log_manager
from utils.retry import retry
from .get_securities import get_investment_target


applogger = log_manager.get_logger(__name__)


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
    预处理市场数据，支持自定义列名映射

    Args:
        market_data: 原始市场数据字典
        columns_mapping: 列名映射字典，用于重命名列

    Returns:
        处理后的DataFrame
    """
    # 默认列名映射
    default_mapping = {
        'open': 'open',
        'close': 'close',
        'high': 'high',
        'low': 'low',
        'volume': 'volume',
        'amount': 'amount'
    }

    # 合并默认和自定义映射
    columns_mapping = columns_mapping or {}
    final_mapping = {**default_mapping, **columns_mapping}

    processed_dataframes = []
    now = datetime.now()

    for stock_code, df in market_data.items():
        if not df.empty:
            df = df.copy()  # 创建副本避免修改原始数据

            # 数据预处理
            df.index.name = 'date'
            df.reset_index(inplace=True)
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns=final_mapping, inplace=True)
            df['modify_time'] = datetime.now()
            df['stock_code'] = stock_code
            df['id'] = df[['stock_code', 'date']].agg('_'.join, axis=1)

            # 转换数据类型
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['modify_time'] = now
            processed_dataframes.append(df)
        else:
            applogger.warning(f"未获取到股票数据，股票：{stock_code}")

    return pd.concat(processed_dataframes, ignore_index=True) if processed_dataframes else pd.DataFrame()


def batch_upsert_data(
        engine: sqlalchemy.engine.base.Engine,
        df: pd.DataFrame,
        batch_size: int = 5000,
        table_name: str = 'stock_data'
):
    """
    批量 Upsert 数据，支持自定义表名和更灵活的处理

    Args:
        engine: SQLAlchemy 引擎
        df: 待插入的数据框
        batch_size: 批次大小
        table_name: 目标表名
    """
    # 创建元数据和表定义
    metadata = MetaData()
    stock_data = Table(table_name, metadata,
                       Column('id', String, primary_key=True),
                       Column('stock_code', String),
                       Column('date', String),
                       Column('open', Float),
                       Column('close', Float),
                       Column('high', Float),
                       Column('low', Float),
                       Column('volume', Float),
                       Column('amount', Float),
                       Column('modify_time', DateTime)
                       )

    # 确保只保留需要的列
    columns_to_insert = [
        'id', 'stock_code', 'date', 'open', 'close',
        'high', 'low', 'volume', 'amount', 'modify_time'
    ]
    df_to_insert = df[columns_to_insert].copy()
    df_to_insert.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    # 使用连接池和事务
    with engine.begin() as connection:
        def _upsert_batch(batch):
            insert_stmt = insert(stock_data).values(batch.to_dict('records'))
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                    col: insert_stmt.excluded[col]
                    for col in batch.columns if col != 'id'
                }
            )
            connection.execute(upsert_stmt)
            # 分批处理

        for i in range(0, len(df_to_insert), batch_size):
            applogger.info(f"正在更新：{i}")
            batch = df_to_insert.iloc[i:i + batch_size]
            _upsert_batch(batch)


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
    获取本地股票数据并入库

    Args:
        period: 数据周期
        start_time: 开始时间
        end_time: 结束时间
        stock_list: 股票列表
        incrementally: 是否增量下载
        max_workers: 最大工作线程数
        batch_size: 批次大小
        columns_mapping: 自定义列名映射

    Returns:
        处理后的DataFrame
    """
    # 参数预处理
    stock_list = stock_list or get_investment_target().securities.to_list()
    start_time = start_time or '20160101'
    end_time = end_time or datetime.now().strftime('%Y%m%d')

    # 下载数据
    download_history_data(
        stock_list=stock_list,
        period=period,
        start_time=start_time,
        end_time=end_time,
        incrementally=incrementally,
        max_workers=max_workers
    )

    try:
        # 获取本地数据
        market_data = xtdata.get_local_data(
            field_list=[],
            stock_list=stock_list,
            period=period,
            start_time=start_time,
            end_time=end_time,
            count=-1,
            dividend_type='front',
            fill_data=True
        )

        # 预处理数据，支持自定义列名映射
        combined_df = preprocess_market_data(market_data, columns_mapping)

        if combined_df.empty:
            applogger.error("未获取到任何股票数据")
            return pd.DataFrame()

            # 打印数据框信息
        applogger.info(f"合并后的数据框形状: {combined_df.shape}")



        return combined_df

    except Exception as e:
        applogger.error(f"获取股票数据失败，错误信息：{e}")
        raise



if __name__ == '__main__':
    df = get_data_from_local()
    print(df)