from darts.models import RegressionEnsembleModel, NaiveEnsembleModel
from sklearn.metrics import (
    precision_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.linear_model import Ridge
import os
import matplotlib
from pathlib import Path

# 自定义
from data_processing.generate_training_data import TimeSeriesProcessor # 生成训练数据
from .model_factory import (
    ModelFactory,
    TiDEModelParamStrategy,
    TSMixerModelParamStrategy,
    XGBModelModelParamStrategy,
    LightGBMModelParamStrategy,
    TFTModelParamStrategy,
    BaseParamStrategy, LightGBMModelParamStrategy
)
from .train_functions import _calculate_components_precision
from logger import log_manager
from utils.darts_models import load_darts_model, save_darts_model
from utils.gup_manage import GPUMemoryManager
from config import settings

logger = log_manager.get_logger(__name__)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # 例如，设置为 4


def train_and_evaluate_ensemble(data: dict, strategies: list):
    """
    训练集成模型并进行评估。

    Parameters:
        data (dict): 包含训练数据的字典
        strategies (list): 基础模型列表

    Returns:
        tuple: (训练好的模型, 预测结果, 真实数据, 评估指标字典)
    """
    GPUMemoryManager().clear_memory()

    models = []
    for strategy in strategies:
        # 加载模型
        model_name = strategy.model_name
        model_path = Path(settings.get('models.model_path')) / f"{model_name}/{model_name}_final_model.pkl"

        if not model_path.exists():
            logger.error(f"模型路径不存在: {model_path}")
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        model = load_darts_model(model_name, model_path.as_posix())

        # LightBGM存在兼容性问题，不能再次训练。如果需要再次训练，可以只加载模型参数。
        # model.fit(
        #     series=data['train_val'],
        #     past_covariates=data['past_covariates'],
        #     future_covariates=data['future_covariates'],
        #     val_series=data['test'],
        #     val_past_covariates=data['past_covariates'],
        #     val_future_covariates=data['future_covariates'],
        # )

        # 评估模型精度。
        evaluation_model(data, model)

        models.append(model)

    ridge_model = Ridge(
        alpha=1.0,  # 正则化强度
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=0.001,
        solver='auto'
    )

        # 创建并配置集成模型
    ensemble_model = RegressionEnsembleModel(
        forecasting_models=models,
        regression_train_n_points=225,  # 使用整个序列
        regression_model=ridge_model,
        train_using_historical_forecasts=True,
        train_forecasting_models=False   # 使用预训练模型，不用重复训练。
    )
    ensemble_model_name = "RegressionEnsembleModel"
    setattr(ensemble_model, "model_name", ensemble_model_name)

    # 训练集成模型
    logger.info('开始训练集成模型')
    ensemble_model.fit(
        series=data['target'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates']
    )

    # 评估模型
    evaluation_model(data, ensemble_model)

    GPUMemoryManager().clear_memory()
    ensemble_path = Path(
        settings.get('models.model_path')) / f"{ensemble_model_name}/{ensemble_model_name}_final_model.pkl"
    save_darts_model(ensemble_model, ensemble_path.as_posix())

    return ensemble_model


def evaluation_model(data, model):
    # 模型回测
    backtest_series = model.historical_forecasts(
        series=data['target'],
        past_covariates=data.get('past_covariates'),
        future_covariates=data.get('future_covariates'),
        start=data['test'].time_index[0],
        forecast_horizon=1,
        stride=1,
        retrain=False,
        last_points_only=True
    )
    logger.info(backtest_series.time_index)
    logger.info(data['target'].time_index)
    # 数据对齐和反缩放
    common_times = data['target'].time_index.intersection(backtest_series.time_index)
    true_data = data['target'].slice(common_times[0], common_times[-1])
    pred_data = backtest_series.slice(common_times[0], common_times[-1])
    pred_data = data['scaler_train'].inverse_transform(pred_data)
    true_data = data['scaler_train'].inverse_transform(true_data)
    # 标签转换
    true_labels = (true_data.values().flatten() > 0).astype(int)
    pred_labels = (pred_data.values().flatten() > 0).astype(int)
    # 指标计算映射
    metric_functions = {
        'precision': precision_score,
        'f1_score': f1_score
    }
    # 计算评估指标
    evaluation_results = {}
    for metric_name, func in metric_functions.items():
        evaluation_results[metric_name] = func(
            true_labels,
            pred_labels,
            zero_division=0
        )
    logger.info(f"模型 「{model.model_name}」 的精度： {evaluation_results}")
    # 组件级别精确率
    components_precision = _calculate_components_precision(
        true_data,
        pred_data,
        threshold=0
    )
    logger.info(f"模型「{model.model_name}」")
    logger.info(components_precision)


def run_ensemble_training(mode='training') -> tuple:
    """
    运行集成模型训练的主函数。

    Parameters:
        mode (str): 数据加载模式，默认为'training'

    Returns:
        tuple: (ensemble_model, pred_data, true_data, metrics)
            - ensemble_model: 训练好的集成模型
            - pred_data: 预测数据
            - true_data: 真实数据
            - metrics: 评估指标字典
    """

    logger.info("开始集成模型训练流程")

    strategies = [
        LightGBMModelParamStrategy,
        TiDEModelParamStrategy,
        TSMixerModelParamStrategy,
        XGBModelModelParamStrategy,
        # TFTModelParamStrategy,
    ]

    try:
        # 加载数据
        logger.info("正在加载数据...")
        data = TimeSeriesProcessor().generate_processed_series_data(mode=mode)
        model = train_and_evaluate_ensemble(data, strategies)

        logger.info("集成模型训练完成!")
        return model

    except Exception as e:
        logger.error(f"集成模型训练过程发生错误: {e}")
        raise


if __name__ == '__main__':
    try:
        ensemble, pred_data, true_data, metrics = run_ensemble_training()
        logger.info("程序成功完成")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")