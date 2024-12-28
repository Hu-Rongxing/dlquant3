# coding:utf-8
import numpy as np
import pandas as pd
from typing import Union, Optional
from xtquant import xtdata
from logger import log_manager

logger = log_manager.get_logger(__name__)


class DividendAdjustment:
    """
    股票除权复权处理工具类

    提供多种复权方法和高效的除权比率计算
    """

    @staticmethod
    def validate_input(
            quote_datas: pd.DataFrame,
            divid_datas: Optional[pd.DataFrame] = None
    ) -> None:
        """
        输入数据验证

        Args:
            quote_datas (pd.DataFrame): 行情数据
            divid_datas (pd.DataFrame, optional): 除权数据

        Raises:
            ValueError: 输入数据不合法时
        """
        if not isinstance(quote_datas, pd.DataFrame):
            raise ValueError("quote_datas必须是DataFrame")

        if divid_datas is not None and not isinstance(divid_datas, pd.DataFrame):
            raise ValueError("divid_datas必须是DataFrame")

        if quote_datas.empty:
            raise ValueError("quote_datas不能为空")

    @classmethod
    def gen_divid_ratio(
            cls,
            quote_datas: pd.DataFrame,
            divid_datas: pd.DataFrame
    ) -> pd.DataFrame:
        """
        高效生成除权比率序列

        优化点：
        1. 输入数据验证
        2. 使用更高效的计算方法
        3. 增加错误处理

        Args:
            quote_datas (pd.DataFrame): 原始行情数据
            divid_datas (pd.DataFrame): 除权数据

        Returns:
            pd.DataFrame: 累积除权比率序列
        """
        try:
            # 输入数据验证
            cls.validate_input(quote_datas, divid_datas)

            # 处理空除权数据的特殊情况
            if divid_datas.empty:
                logger.warning("没有除权数据，返回全1的复权比率")
                return pd.DataFrame(
                    np.ones(len(quote_datas)),
                    index=quote_datas.index,
                    columns=quote_datas.columns
                )

                # 排序并去重
            divid_datas = divid_datas.sort_index().drop_duplicates()
            quote_datas = quote_datas.sort_index()

            # 重建索引，确保连续性
            full_index = quote_datas.index.union(divid_datas.index)

            # 使用更高效的重索引方法
            s = divid_datas['dr'].reindex(full_index, method='ffill').fillna(1)

            # 计算累积乘积
            s = s.cumprod()

            # 按行情数据索引提取结果
            result = s.reindex(quote_datas.index).to_frame()
            result.columns = quote_datas.columns

            return result

        except Exception as e:
            logger.error(f"除权比率计算失败: {e}")
            raise

    @classmethod
    def process_forward_ratio(
            cls,
            quote_datas: pd.DataFrame,
            divid_datas: pd.DataFrame
    ) -> pd.DataFrame:
        """
        前复权处理

        优化点：
        1. 增加输入数据验证
        2. 提供更灵活的复权方法
        3. 增加日志记录

        Args:
            quote_datas (pd.DataFrame): 原始行情数据
            divid_datas (pd.DataFrame): 除权数据

        Returns:
            pd.DataFrame: 前复权处理后的数据
        """
        try:
            # 输入数据验证
            cls.validate_input(quote_datas, divid_datas)

            # 计算累积除权比率
            drl = cls.gen_divid_ratio(quote_datas, divid_datas)

            # 归一化处理：使最后一个数据点的复权比率为1
            drlf = drl / drl.iloc[-1]

            # 应用复权比率并四舍五入
            result = (quote_datas * drlf).round(2)

            logger.info(f"前复权处理完成，处理数据点数: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"前复权处理失败: {e}")
            raise

    @classmethod
    def process_backward_ratio(
            cls,
            quote_datas: pd.DataFrame,
            divid_datas: pd.DataFrame
    ) -> pd.DataFrame:
        """
        后复权处理

        Args:
            quote_datas (pd.DataFrame): 原始行情数据
            divid_datas (pd.DataFrame): 除权数据

        Returns:
            pd.DataFrame: 后复权处理后的数据
        """
        try:
            # 输入数据验证
            cls.validate_input(quote_datas, divid_datas)

            # 计算累积除权比率
            drl = cls.gen_divid_ratio(quote_datas, divid_datas)

            # 应用复权比率并四舍五入
            result = (quote_datas * drl).round(2)

            logger.info(f"后复权处理完成，处理数据点数: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"后复权处理失败: {e}")
            raise

        # 主程序示例


def main():
    try:
        # 选择股票代码
        stock_code = '002594.SZ'

        # 获取除权因子数据
        divid_datas = xtdata.get_divid_factors(stock_code)

        # 复权计算用于处理价格字段
        field_list = ['open', 'high', 'low', 'close']
        quote_datas = xtdata.get_market_data(
            field_list,
            [stock_code],
            '1d',
            dividend_type='none'
        )['close'].T

        # 前复权处理
        forward_adjusted = DividendAdjustment.process_forward_ratio(
            quote_datas,
            divid_datas
        )

        print("前复权结果:")
        print(forward_adjusted.head())

    except Exception as e:
        logger.error(f"主程序执行失败: {e}")


if __name__ == '__main__':
    main()