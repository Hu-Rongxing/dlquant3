
def train_and_evaluate_ensemble(data: dict, forecasting_models: list) -> tuple:
    """
    训练集成模型并进行评估。

    Parameters:
        data (dict): 包含训练数据的字典
        forecasting_models (list): 基础模型列表

    Returns:
        tuple: (训练好的模型, 预测结果, 真实数据, 评估指标字典)
    """
    # 训练基础模型
    for model in forecasting_models:
        logger.info(f"训练基础模型: {model.model_name}")
        model.fit(
            series=data['full_data'],
            past_covariates=data.get('past_covariates'),
            future_covariates=data.get('future_covariates'),
            val_series=data['test'],
            val_past_covariates=data.get('past_covariates'),
            val_future_covariates=data.get('future_covariates')
        )

    # 创建并配置集成模型
    ensemble = RegressionEnsembleModel(
        forecasting_models=forecasting_models,
        regression_train_n_points=24,
        regression_model=Ridge(),
        train_using_historical_forecasts=False,
        train_forecasting_models=False
    )
    setattr(ensemble, "model_name", model_name)

    # 训练集成模型
    logger.info('开始训练集成模型')
    ensemble.fit(
        series=data['full_data'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates']
    )

    # 模型回测
    backtest_series = ensemble.historical_forecasts(
        series=data['full_data'],
        past_covariates=data.get('past_covariates'),
        future_covariates=data.get('future_covariates'),
        start=data['full_data'].time_index[-PRED_STEPS],
        forecast_horizon=1,
        stride=1,
        retrain=False,
        last_points_only=True
    )

    # 评估预测结果
    metrics, true_data, pred_data = evaluate_predictions(
        data['full_data'], backtest_series, data
    )

    return ensemble, pred_data, true_data, metrics


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
    try:
        # 加载数据
        logger.info("正在加载数据...")
        data = generate_processed_series_data(mode=mode)

        # 加载基础模型
        logger.info("正在加载基础模型...")
        base_models = {
            "TiDEModel": define_TiDEModel_trial,
            "TSMixerModel": define_TSMixerModel_trial,
            "LightGBMModel": define_LightGBMModel_trial,
        }

        forecasting_models = []
        for model_name in base_models.keys():
            try:
                model = load_model(model_name)
                forecasting_models.append(model)
                logger.info(f"成功加载模型: {model_name}")
            except Exception as e:
                logger.error(f"加载模型 {model_name} 失败: {e}")
                raise RuntimeError(f"基础模型加载失败: {model_name}")

        # 训练和评估集成模型
        logger.info("开始训练集成模型...")
        ensemble, pred_data, true_data, metrics = train_and_evaluate_ensemble(
            data, forecasting_models
        )

        # 保存模型
        logger.info("保存集成模型...")
        save_model(ensemble)

        # 绘制结果
        logger.info("绘制评估结果...")
        components_precision = calculate_component_precisions(true_data, pred_data)
        plot_results(pred_data, metrics['precision'], components_precision)

        logger.info("集成模型训练完成!")
        return ensemble, pred_data, true_data, metrics

    except Exception as e:
        logger.error(f"集成模型训练过程发生错误: {e}")
        raise


if __name__ == '__main__':
    try:
        ensemble, pred_data, true_data, metrics = run_ensemble_training()
        logger.info("程序成功完成")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")