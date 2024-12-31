import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from functools import lru_cache
from .source_stock_data import get_data_from_local
from logger import log_manager
import time  # 新增性能监控模块

logger = log_manager.get_logger(__name__)


class DataProcessor:
    """
    数据处理与技术指标计算的核心类
    提供数据清洗、技术指标计算、宽格式转换等功能
    """

    # 优化1：使用类型注解和常量定义
    FEATURE_COLS: List[str] = [
        'open', 'high', 'low', 'close', 'volume', 'amount',
        'ma_3', 'ma_5', 'ma_10'
    ]

    INDEX_STOCKS: List[str] = ['000001.SH', '399001.SZ', '399006.SZ']

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        优化2：数据清洗更加健壮
        """
        # 使用更安全的数据清洗方法
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 优化：添加更多清洗逻辑
        df.dropna(subset=['close', 'volume'], how='any', inplace=True)

        # 移除异常值
        # numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        # for col in numeric_cols:
        #     df = df[np.abs(df[col] - df[col].mean()) <= (3 * df[col].std())]

        return df

    @classmethod
    def calculate_technical_indicators(cls, group: pd.DataFrame) -> pd.DataFrame:
        """
        优化3：技术指标计算向量化
        """
        # 使用更高效的填充方法
        group = group.ffill().bfill()

        # 向量化计算目标变量
        group['target'] = group['close'].pct_change()

        # 批量计算移动平均线
        ma_windows = [3, 5, 10]
        for window in ma_windows:
            group[f'ma_{window}'] = group['close'].rolling(window=window, center=False, min_periods=1).mean()

            # 优化：同时计算指数移动平均线
        # group[['ema_5', 'ema_10']] = pd.DataFrame({
        #     'ema_5': group['close'].ewm(span=5, adjust=False).mean(),
        #     'ema_10': group['close'].ewm(span=10, adjust=False).mean()
        # })

        # 优化平均价格计算
        # group['ave_price'] = group.apply(
        #     lambda row: row['amount'] / (row['volume'] * 100)
        #     if row['volume'] > 0 else row['close'],
        #     axis=1
        # )

        return group

    @classmethod
    @lru_cache(maxsize=128)  # 优化4：增加缓存大小
    def process_dataframe(cls, count=-1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        优化5：增加性能监控和更详细的错误处理
        """
        start_time = time.time()
        try:
            # 获取数据
            dataframe = get_data_from_local(count=count)

            # 数据清洗
            dataframe = cls.clean_dataframe(dataframe)

            # 优化：使用更高效的时间序列映射
            timeseries = dataframe['date'].drop_duplicates().sort_values()
            timeseries.reset_index(drop=True, inplace=True)
            date_to_int_series = pd.Series(timeseries.index, index=timeseries)
            # int_to_date_series = timeseries.reset_index(drop=True)

            # 排序并生成时间序列
            dataframe.sort_values(by=['stock_code', 'date'], inplace=True)
            dataframe['time_seq'] = dataframe['date'].map(date_to_int_series)

            # 计算技术指标
            dataframe = (
                dataframe.groupby('stock_code', group_keys=False)
                .apply(cls.calculate_technical_indicators)
                .reset_index(drop=True)
            )

            logger.info(f"数据处理耗时: {time.time() - start_time:.2f}秒")
            return dataframe, date_to_int_series

        except Exception as e:
            logger.error(f"数据处理失败 - {str(e)}")
            raise

    @classmethod
    def generate_wide_dataframe(
            cls,
            include_index: bool = True,
            index_stocks: List[str] = None,
            count=-1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        优化6：提高数据转换的健壮性
        """
        # 处理数据
        data_cleaned, date_to_int_series = cls.process_dataframe(count=count)
        logger.trader(f"下载数据的最新日期{data_cleaned['date'].max()}")

        # 创建目标变量数据框
        target_df = data_cleaned.pivot_table(
            index="time_seq",
            columns="stock_code",
            values='target',
            fill_value=0  # 优化：填充缺失值
        )
        target_df.sort_index(inplace=True)

        # 创建协变量数据框
        covariate_df = data_cleaned.pivot_table(
            index="time_seq",
            columns="stock_code",
            values=cls.FEATURE_COLS,
            fill_value=0  # 优化：填充缺失值
        )
        covariate_df.columns.names = ['variable', 'stock_code']
        covariate_df.sort_index(inplace=True)

        # 添加指数数据
        if include_index:
            index_stocks = index_stocks or cls.INDEX_STOCKS
            index_data = get_data_from_local(stock_list=index_stocks, count=count)
            index_data['time_seq'] = index_data['time'].map(date_to_int_series)

            index_data_pivot = index_data.pivot_table(
                index="time_seq",
                columns="stock_code",
                values=['open', 'high', 'low', 'close', 'amount'],
                fill_value=0  # 优化：填充缺失值
            )
            index_data_pivot.columns.names = ['variable', 'stock_code']


            covariate_df = covariate_df.join(index_data_pivot)

            # 规范化列名
        covariate_df.columns = [
            f"{var}_{stock}"
            for (var, stock) in covariate_df.columns
        ]

        return target_df, covariate_df, date_to_int_series


def main():
    try:
        # 生成宽格式数据
        target_df, covariate_df, int_to_date_series = DataProcessor.generate_wide_dataframe()

        logger.info(f"目标变量形状: {target_df.shape}")
        logger.info(f"协变量形状: {covariate_df.shape}")

    except Exception as e:
        logger.error(f"主程序执行失败 - {str(e)}")


if __name__ == '__main__':
    main()