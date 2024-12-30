import joblib
from pathlib import Path
from .training_model import train_and_evaluate
from .model_factory import (
    ModelFactory,
    TiDEModelParamStrategy,
    TSMixerModelParamStrategy,
    XGBModelModelParamStrategy,
    LightGBMModelParamStrategy,
    TFTModelParamStrategy,
    BaseParamStrategy
)
from data_processing.generate_training_data import TimeSeriesProcessor
from config import settings
from logger import log_manager

logger = log_manager.get_logger(__name__)


class ModelDeployment:
    def __init__(self, model_params_strategy):
        self.model_name = model_params_strategy.model_name
        self.study_path = Path(settings.get('models.model_path')) / self.model_name
        self.study_file = self.study_path / f"optuna_study_{self.model_name}.pkl"
        self.model_factory = ModelFactory()
        self.data = None

    def load_best_model(self):
        """
        加载最佳模型
        """
        try:
            # 加载 Optuna 研究对象
            study = joblib.load(self.study_file)
            best_trial = study.best_trial
            best_params = best_trial.params

            # 创建模型并加载最佳参数
            model = self.model_factory.define_common_model(best_params)
            logger.info(f"成功加载最佳模型: {self.model_name}，参数: {best_params}")
            return model
        except Exception as e:
            logger.error(f"加载最佳模型失败: {e}")
            raise

    def prepare_data(self):
        """
        数据准备和预处理
        """
        data_precessor = TimeSeriesProcessor()
        data = data_precessor.generate_processed_series_data()
        # 生成train、test、val
        test_length = settings.get("data.test_data_length", 45)
        buffer_length = settings.get("data.initial_buffer_data", 120)
        data['test'] = data['target'][-(test_length + buffer_length):]
        val_length = settings.get("data.val_data_length", 45)
        data['val'] = data['target'][-(test_length + val_length + buffer_length):-test_length]
        data['train'] = data['target'][:-(test_length + val_length)]
        data['train_val'] = data['target'][:-test_length]
        self.data = data
        logger.info("数据准备完成。")
        return self.data

    def retrain_model(self, model):
        """
        使用加载的模型进行再训练
        """
        try:
            # 准备数据
            self.prepare_data()
            # 进行训练
            evaluation = train_and_evaluate(model, self.data, test=False)
            logger.info(f"模型再训练完成，评估结果: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"模型再训练失败: {e}")
            raise

    def save_model(self, model):
        """
        保存训练好的模型
        """
        model_path = self.study_path / f"{self.model_name}_final_model.pkl"
        try:
            joblib.dump(model, model_path)
            logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")


def main():
    # 选择模型参数策略
    model_params_strategy = TiDEModelParamStrategy  # 选择您需要的模型策略

    # 初始化模型部署
    deployment = ModelDeployment(model_params_strategy)

    # 加载最佳模型
    model = deployment.load_best_model()

    # 再训练模型
    deployment.retrain_model(model)

    # 保存最终模型
    deployment.save_model(model)


if __name__ == "__main__":
    main()