from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle
import os
import tempfile

from config import settings
from .get_stock_data_from_local import DataProcessor
from .source_trading_calendar import ChineseMarketCalendar
from logger import log_manager

logger = log_manager.get_logger(__name__)


def merge_date_series(date_to_int_series, future_dates):
    """
    安全地合并两个日期序列，增加更多健壮性检查
    """
    # 类型检查
    if not isinstance(date_to_int_series, pd.Series):
        raise TypeError("date_to_int_series必须是pandas Series")

        # 确保future_dates是可迭代的
    if not hasattr(future_dates, '__iter__'):
        raise ValueError("future_dates必须是可迭代的日期列表")

        # 转换并合并
    try:
        swapped_series = pd.Series(
            index=date_to_int_series.values,
            data=pd.to_datetime(date_to_int_series.index)
        )

        future_series = pd.Series(pd.to_datetime(future_dates))

        last_index = date_to_int_series.max()
        print(last_index)
        future_series.index = range(last_index + 1, last_index + len(future_series) + 1)

        return pd.concat([swapped_series, future_series], axis=0)

    except Exception as e:
        logger.error(f"日期序列合并失败: {e}")
        raise

class TimeSeriesProcessor:
    def __init__(self, config: Optional[Dict] = None):
        """
        时间序列数据处理器初始化

        Args:
            config (Optional[Dict]): 配置参数，默认使用 settings
        """
        self.config = config or settings
        self.calendar = ChineseMarketCalendar()

        # 使用临时目录作为备选
        self.scaler_dir = self.config.get(
            "data.train_scaler_dir",
            os.path.join(tempfile.gettempdir(), 'scalers')
        )

        # 确保缩放器保存目录存在
        Path(self.scaler_dir).mkdir(parents=True, exist_ok=True)

    # @lru_cache(maxsize=1)
    def _load_scaler(self, filename: str):
        """
        安全地加载缩放器

        Args:
            filename (str): 文件名

        Returns:
            Scaler: 加载的缩放器
        """
        try:
            # 尝试从多个位置加载
            possible_paths = [
                Path(self.scaler_dir) / filename,
                Path(tempfile.gettempdir()) / 'scalers' / filename
            ]

            for file_path in possible_paths:
                logger.info(Path(file_path).absolute().as_posix())
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    logger.warning(f"未找到缩放器文件: {file_path}")
                    raise Exception(f"文件{file_path}不存在")
            return None

        except Exception as e:
            logger.error(f"加载缩放器时发生错误: {e}")
            raise

    def _save_scaler(self, scaler, filename: str):
        """
        安全地保存缩放器，并确保目录存在

        Args:
            scaler: 要保存的缩放器
            file_path (str): 保存路径
        """
        try:
            # 确保目录存在
            filepath = Path(self.scaler_dir) / filename
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # 检查是否有写入权限
            if not os.access(filepath.parent, os.W_OK):
                logger.error(f"没有写入权限: {filepath.parent}")
                raise PermissionError(f"没有写入权限: {filepath.parent}")

            with open(filepath, 'wb') as f:
                pickle.dump(scaler, f)
        except PermissionError:
            # 尝试使用临时目录
            temp_path = os.path.join(tempfile.gettempdir(), 'scalers', Path(filepath).name)
            Path(temp_path).parent.mkdir(parents=True, exist_ok=True)

            with open(temp_path, 'wb') as f:
                pickle.dump(scaler, f)

            logger.warning(f"无法写入原始路径，已保存到临时目录: {temp_path}")
        except Exception as e:
            logger.error(f"保存缩放器时发生错误: {e}")
            raise

    def rbf_encode_time_features(
            self,
            dates: pd.DatetimeIndex,
            num_centers: int = 10,
            width_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        使用径向基函数(RBF)对时间特征进行编码

        Args:
            dates (pd.DatetimeIndex): 日期时间索引
            num_centers (int, optional): RBF核函数的中心数量. Defaults to 10.
            width_factor (float, optional): RBF核函数宽度因子. Defaults to 1.0.

        Returns:
            pd.DataFrame: 编码后的时间特征
        """
        try:
            scalers = {
                'day': MinMaxScaler(),
                'weekday': MinMaxScaler(),
                'month': MinMaxScaler(),
                'week': MinMaxScaler()
            }

            scaled_features = {
                'day': scalers['day'].fit_transform(dates.day.values.reshape(-1, 1)).flatten(),
                'weekday': scalers['weekday'].fit_transform(dates.weekday.values.reshape(-1, 1)).flatten(),
                'month': scalers['month'].fit_transform(dates.month.values.reshape(-1, 1)).flatten(),
                'week': scalers['week'].fit_transform(dates.isocalendar().week.values.reshape(-1, 1)).flatten()
            }

            width = width_factor / num_centers
            centers = np.linspace(0, 1, num_centers)

            rbf_features = {
                feature: np.exp(-((scaled_features[feature][:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
                for feature in scaled_features.keys()
            }

            encoded_matrix = np.hstack(list(rbf_features.values()))
            return pd.DataFrame(encoded_matrix)

        except Exception as e:
            logger.error(f"RBF编码时发生错误: {e}")
            raise

    def generate_processed_series_data(
            self,
            mode: Literal['training', 'predicting'] = 'training',
            train_length: Optional[int] = None,
            test_length: Optional[int] = None
    ) -> Dict[str, Union[TimeSeries, Scaler]]:
        """
        生成处理后的时间序列数据

        Args:
            mode (str): 数据模式 'training' 或 'predicting'
            train_length (Optional[int]): 训练数据长度
            test_length (Optional[int]): 测试数据长度

        Returns:
            Dict[str, Union[TimeSeries, Scaler]]: 处理后的时间序列数据
        """
        try:
            # 获取配置参数
            train_length = train_length or self.config.get("data.train_data_length", 45)
            test_length = test_length or self.config.get("data.test_data_length", 45)
            train_date_end = test_length + train_length

            # 生成目标变量和协变量
            if mode == 'predicting':
                count = 250
            else:
                count = -1
            target_df, covariate_df, date_to_int_series = DataProcessor.generate_wide_dataframe(count=count)
            logger.info(f"目标变量形状: {target_df.shape}")
            logger.info(f"协变量形状: {covariate_df.shape}")

            # 转换为 Darts TimeSeries
            target_time_series = TimeSeries.from_dataframe(
                df=target_df,
                fill_missing_dates=True,
                freq=1,
                fillna_value=0
            ).astype(np.float32)

            past_covariate_time_series = TimeSeries.from_dataframe(
                df=covariate_df,
                fill_missing_dates=True,
                freq=1,
                fillna_value=0
            ).astype(np.float32)

            train_scaler_file = 'train_scaler.pkl'
            logger.info(train_scaler_file)
            covariate_scaler_file = 'covariate_scaler.pkl'

            if mode == 'training':
                train_scaler = Scaler(name='target').fit(target_time_series[:-train_date_end])
                # 使用更安全的文件名
                self._save_scaler(train_scaler, train_scaler_file)
                past_cov_scaler = Scaler(name='covariate').fit(past_covariate_time_series[:-train_date_end])
                self._save_scaler(past_cov_scaler, covariate_scaler_file)

            elif mode == 'predicting':
                train_scaler = Scaler(name='target').fit(target_time_series[:-train_date_end])
                # train_scaler = self._load_scaler(train_scaler_file)
                past_cov_scaler = self._load_scaler(covariate_scaler_file)

                # 数据缩放
            target_scaled = train_scaler.transform(target_time_series)
            covariates_scaled = past_cov_scaler.transform(past_covariate_time_series)

            # 处理未来日期
            latest_date = date_to_int_series.index.max()
            next_start_date = self.calendar.get_next_trading_day(latest_date)
            future_dates = self.calendar.get_trading_days_range(
                start_date= next_start_date,
                end_date=next_start_date + relativedelta(years=1)
            )
            future_series = pd.Series(future_dates)

            # 使用新的合并方法
            final_series = merge_date_series(date_to_int_series, future_dates)

            # 使用合并后的series创建DatetimeIndex
            future_encoded_features = self.rbf_encode_time_features(
                pd.DatetimeIndex(final_series)
            )

            future_covariate_series = TimeSeries.from_dataframe(
                future_encoded_features
            ).astype(np.float32)

            return {
                "target": target_scaled,
                "past_covariates": covariates_scaled,
                "future_covariates": future_covariate_series,
                "scaler_train": train_scaler,
                "scaler_past": past_cov_scaler,
                "final_date_series": final_series  # 可选：返回最终的日期序列
            }

        except Exception as e:
            logger.error(f"处理时间序列数据时发生错误: {e}")
            raise


def main():
    processor = TimeSeriesProcessor()
    result = processor.generate_processed_series_data(mode='training')
    print("处理完成")
    return result


if __name__ == '__main__':
    main()