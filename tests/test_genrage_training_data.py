import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
from data_processing.generate_training_data import merge_date_series, TimeSeriesProcessor


def test_merge_date_series():
    # 创建测试数据
    original_dates = pd.Series({
        datetime(2023, 1, 1):0,
        datetime(2023, 1, 2):1,
        datetime(2023, 1, 3):2
    })

    future_dates = [
        datetime(2023, 1, 4),
        datetime(2023, 1, 5),
        datetime(2023, 1, 6)
    ]

    # 执行合并
    result = merge_date_series(original_dates, future_dates)

    # 断言检查
    assert isinstance(result, pd.Series)
    assert len(result) == 6
    assert result.iloc[0] == datetime(2023, 1, 1)
    assert result.iloc[-1] == datetime(2023, 1, 6)

    # 检查索引是否连续
    assert list(result.index) == list(range(6))


def test_merge_date_series_error_handling():
    # 测试类型错误
    with pytest.raises(TypeError):
        merge_date_series([], [datetime(2023, 1, 1)])

    with pytest.raises(ValueError):
        merge_date_series(pd.Series([1, 2, 3]), None)


def test_rbf_encode_time_features():
    # 创建测试日期索引
    dates = pd.date_range(start='2023-01-01', periods=10)

    # 初始化处理器
    processor = TimeSeriesProcessor()

    # 执行RBF编码
    result = processor.rbf_encode_time_features(dates)

    # 断言检查
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 10  # 行数应该等于输入日期数
    assert result.shape[1] == 40  # 4个特征 * 10个中心点


def test_time_series_processor_initialization():
    # 测试初始化
    processor = TimeSeriesProcessor()

    assert hasattr(processor, 'config')
    assert hasattr(processor, 'calendar')
    assert hasattr(processor, 'scaler_dir')

    # 检查缩放器目录是否存在
    import os
    assert os.path.exists(processor.scaler_dir)


@pytest.mark.parametrize("mode", ['training', 'predicting'])
def test_generate_processed_series_data(mode):
    # 初始化处理器
    processor = TimeSeriesProcessor()

    # 执行数据处理
    result = processor.generate_processed_series_data(mode=mode)

    # 断言检查关键结果
    assert 'target' in result
    assert 'past_covariates' in result
    assert 'future_covariates' in result
    assert 'scaler_train' in result
    assert 'scaler_past' in result

    # 检查返回的对象类型
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler

    assert isinstance(result['target'], TimeSeries)
    assert isinstance(result['past_covariates'], TimeSeries)
    assert isinstance(result['future_covariates'], TimeSeries)
    assert isinstance(result['scaler_train'], Scaler)
    assert isinstance(result['scaler_past'], Scaler)


def test_save_and_load_scaler(tmp_path):
    # 初始化处理器
    processor = TimeSeriesProcessor()

    # 创建测试缩放器
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler

    test_series = TimeSeries.from_dataframe(pd.DataFrame({'value': [1, 2, 3, 4, 5]}))
    test_scaler = Scaler().fit(test_series)

    # 测试保存
    test_file_path = str(tmp_path / 'test_scaler.pkl')
    processor._save_scaler(test_scaler, test_file_path)

    # 测试加载
    loaded_scaler = processor._load_scaler(test_file_path)

    # 断言检查
    assert loaded_scaler is not None
    assert isinstance(loaded_scaler, Scaler)


def test_error_handling():
    # 初始化处理器
    processor = TimeSeriesProcessor()

    # 测试加载不存在的缩放器
    with pytest.raises(Exception):
        processor._load_scaler('non_existent_scaler.pkl')

    # 性能测试


def test_performance_rbf_encode():
    import time

    # 创建大量日期
    dates = pd.date_range(start='2023-01-01', periods=1000)

    # 初始化处理器
    processor = TimeSeriesProcessor()

    # 测量编码时间
    start_time = time.time()
    result = processor.rbf_encode_time_features(dates)
    end_time = time.time()

    # 性能断言
    assert end_time - start_time < 1.0  # 编码应在1秒内完成
    assert result.shape[0] == 1000