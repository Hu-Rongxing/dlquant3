import torch
import darts
from typing import Dict,Any, Optional
import matplotlib.pyplot as plt
import matplotlib
import logging
from sklearn.metrics import (
    precision_score,
    f1_score
)
# 自定义
from logger import log_manager
from utils.gup_manage import GPUMemoryManager

logger = log_manager.get_logger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# 配置 Matplotlib 以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题


def plot_backtest_data(backtest_data: darts.TimeSeries, title: str = "回测数据") -> None:
    """
    绘制回测结果的预测值。

    Parameters:
        backtest_data (torch.Tensor): 回测预测的数据。
        title (str): 图表标题，默认为"回测数据"。
    """
    df = backtest_data.pd_dataframe()
    sample_df = df.sample(n=min(5, df.shape[1]), axis=1, random_state=42)  # 确保不超过列数
    sample_df.plot(lw=2, alpha=0.6)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()
    plt.close()


def plot_components_precision(components_precision: dict, overall_precision: float = None) -> None:
    """
    绘制每个组件的精确率。

    Parameters:
        components_precision (dict): 每个组件的精确率，键为组件索引，值为精确率。
        overall_precision (float, optional): 整体精确率，用于在标题中显示。
    """
    plt.figure(figsize=(12, 6))

    # 提取组件索引和对应的精确率
    indices = list(components_precision.keys())
    precisions = list(components_precision.values())

    # 创建数值位置
    x_positions = range(len(indices))

    # 绘制条形图
    plt.bar(x_positions, precisions, color='skyblue', alpha=0.7)

    # 设置图表标题，包含整体精确率（如果提供）
    title = "每个组件的精确率"
    if overall_precision is not None:
        title += f"（总体精确率 {overall_precision:.2%}）"
    plt.title(title)

    # 设置x轴标签和y轴标签
    plt.xlabel("组件索引")
    plt.ylabel("精确率")
    plt.ylim(0, 1)

    # 设置x轴刻度位置和标签
    plt.xticks(ticks=x_positions, labels=indices, rotation=45)

    # 添加y轴网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 自动调整子图参数，防止标签被截断
    plt.tight_layout()

    # 显示图表
    plt.show()
    plt.close()

class ModelEvaluationError(Exception):
    """自定义模型评估异常"""
    pass


def safe_metric_calculation(
        metric_func,
        y_true,
        y_pred,
        default_value: float = 0.0,
        **kwargs
) -> float:
    """
    安全地计算评估指标

    Args:
        metric_func: 评估指标函数
        y_true: 真实标签
        y_pred: 预测标签
        default_value: 默认返回值
        **kwargs: 传递给指标函数的额外参数

    Returns:
        计算得到的指标值
    """
    try:
        return metric_func(y_true, y_pred, **kwargs)
    except Exception as e:
        logger.warning(f"指标计算失败: {metric_func.__name__}, 错误: {e}")
        return default_value


def train_and_evaluate(model, data, config: Optional[Dict[str, Any]] = None, test: bool=True
) -> Dict[str, float]:
    """
    高级模型训练和评估函数

    Args:
        model: 待训练模型
        data: 包含训练和验证数据的字典
        config: 可选的配置参数

    Returns:
        评估指标字典
    """
    # 默认配置
    default_config = {
        'metrics': [
            'precision', 'f1_score', 'mse',
            'rmse', 'mae', 'mape'
        ],
        'plot_results': True,
        'verbose': True,
        'prediction_threshold': 0,
        'gpu_memory_log': True
    }
    config = {**default_config, **(config or {})}

    # 日志记录GPU内存使用情况
    if config['gpu_memory_log']:
        logger.info(f"训练前GPU内存: {GPUMemoryManager.get_gpu_memory_info()}")

    try:
        # 模型训练前的内存检查和清理
        GPUMemoryManager.clear_memory()

        # 模型训练
        try:
            if test:
                model.fit(
                    series=data['train'],
                    past_covariates=data['past_covariates'],
                    future_covariates=data['future_covariates'],
                    val_series=data['val'],
                    val_past_covariates=data['past_covariates'],
                    val_future_covariates=data['future_covariates'],
                )
            else:
                model.fit(
                    series=data['train_val'],
                    past_covariates=data['past_covariates'],
                    future_covariates=data['future_covariates'],
                    val_series=data['test'],
                    val_past_covariates=data['past_covariates'],
                    val_future_covariates=data['future_covariates'],
                )
        except Exception as train_error:
            logger.error(f"模型训练失败: {train_error}")
            raise ModelEvaluationError(f"模型训练异常: {train_error}")
        # 历史预测（回测）
        try:
            backtest_series = model.historical_forecasts(
                series=data['target'],
                past_covariates=data['past_covariates'],
                future_covariates=data['future_covariates'],
                start=data['test'].time_index[0],
                forecast_horizon=1,
                stride=1,
                retrain=False,
                last_points_only=True
            )
        except Exception as forecast_error:
            logger.error(f"历史预测失败: {forecast_error}")
            raise ModelEvaluationError(f"历史预测异常: {forecast_error}")

        # 数据对齐和反缩放
        common_times = data['target'].time_index.intersection(backtest_series.time_index)
        true_data = data['target'].slice(common_times[0], common_times[-1])
        pred_data = backtest_series.slice(common_times[0], common_times[-1])

        pred_data = data['scaler_train'].inverse_transform(pred_data)
        true_data = data['scaler_train'].inverse_transform(true_data)

        # 标签转换
        true_labels = (true_data.values().flatten() > config['prediction_threshold']).astype(int)
        pred_labels = (pred_data.values().flatten() > config['prediction_threshold']).astype(int)

        # 指标计算映射
        metric_functions = {
            'precision': precision_score,
            'f1_score': f1_score
        }

        # 计算评估指标
        evaluation_results = {}
        for metric_name in config['metrics']:
            if metric_name in metric_functions:
                func = metric_functions[metric_name]
                evaluation_results[metric_name] = safe_metric_calculation(
                    func,
                    true_labels,
                    pred_labels,
                    zero_division=0
                )

        # 组件级别精确率
        components_precision = _calculate_components_precision(
            true_data,
            pred_data,
            threshold=config['prediction_threshold']
        )
        evaluation_results['components_precision'] = components_precision

        # 可视化结果
        if config['plot_results']:
            plot_backtest_data(pred_data)
            plot_components_precision(
                components_precision,
                overall_precision=evaluation_results.get('precision', 0)
            )

        # 日志记录
        if config['verbose']:
            for metric, value in evaluation_results.items():
                logger.info(f"{metric}: {value}")

        return evaluation_results

    except ModelEvaluationError as e:
        logger.error(f"模型评估失败: {e}")
        return {metric: 0.0 for metric in config['metrics']}

    except Exception as unexpected_error:
        logger.critical(f"未预料的评估异常: {unexpected_error}")
        return {metric: 0.0 for metric in config['metrics']}

    finally:
        # 最终清理GPU内存
        GPUMemoryManager.clear_memory()
        if config['gpu_memory_log']:
            logger.info(f"训练后GPU内存: {GPUMemoryManager.get_gpu_memory_info()}")


def _calculate_components_precision(
        true_data,
        pred_data,
        threshold: float = 0
) -> Dict[str, float]:
    """
    计算各组件的精确率

    Args:
        true_data: 真实数据
        pred_data: 预测数据
        threshold: 分类阈值

    Returns:
        组件精确率字典
    """
    components_precision = {}
    for component_idx in pred_data.components:
        try:
            # logger.info(true_data[component_idx].values())
            component_true_labels = (
                    true_data[component_idx].values().flatten() > threshold
            ).astype(int)
            component_pred_labels = (
                    pred_data[component_idx].values().flatten() > threshold
            ).astype(int)

            component_precision = safe_metric_calculation(
                precision_score,
                component_true_labels,
                component_pred_labels,
                zero_division=0
            )
            logger.info(f"{component_idx}: {component_precision}")
            if component_precision > 0:
                components_precision[component_idx] = component_precision

        except Exception as e:
            logger.warning(f"Component {component_idx} 精度计算失败: {e}")

    return components_precision



