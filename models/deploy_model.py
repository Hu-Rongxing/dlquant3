from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed
from config import settings
from logger import log_manager
from utils.darts_models import load_darts_model
from data_processing.generate_training_data import TimeSeriesProcessor
from qmt_client.qmt_trader import buy_stock_async

logger = log_manager.get_logger(__name__)

def predict_market_by_ensemble_model():
    model_name = "RegressionEnsembleModel"
    data = TimeSeriesProcessor().generate_processed_series_data(mode='predicting')
    model_path = Path(settings.get('models.model_path')) / f"{model_name}/{model_name}_final_model.pkl"
    ensemble_model = load_darts_model(model_name, model_path.as_posix())

    predictions = ensemble_model.predict(
        n=1,
        series=data['target'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates']
    )

    # 对预测结果进行反缩放
    predictions = data["scaler_train"].inverse_transform(predictions)
    # 将预测结果转换为DataFrame并排序
    pred_df = predictions.pd_dataframe() * 100
    sorted_pred = pred_df.iloc[0].sort_values(ascending=False)
    # 筛选出大于0的预测结果
    positive_predictions = sorted_pred[sorted_pred > 0.2]
    stock_codes = positive_predictions.index.tolist()

    # buy_stock_async(stock_codes)
    print(positive_predictions)

    return stock_codes

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def buying_strategy():
    """买入策略函数"""
    try:
        to_buy = predict_market_by_ensemble_model()
    except Exception as e:
        logger.error("生成待买入列表错误。")
        raise e
    try:
        if to_buy:
            logger.trader(f"买入股票列表：{'、'.join(to_buy)}")
            buy_stock_async(to_buy, strategy_name='买入策略', order_remark='集成模型。')
        else:
            logger.info("无股票可买入。")
    except Exception as e:
        logger.error(f"买入股票列表{'、'.join(to_buy)}出现错误")
        raise e

if __name__ == '__main__':
    predict_market_by_ensemble_model()