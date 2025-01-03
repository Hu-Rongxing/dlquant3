import optuna
from typing import Optional, Dict, Any
import os
import joblib
from pathlib import Path

# 导入之前定义的模型工厂和策略
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
from .training_model import train_and_evaluate
from config import settings
from logger import log_manager

logger = log_manager.get_logger(__name__)


class TimeSeriesModelTrainer:
    def __init__(self, model_params_strategy:BaseParamStrategy):
        """
        初始化时间序列模型训练器，支持断点续训

        Args:
            data: 输入数据
            target_column: 目标列名
            name: 模型名称
            study_storage_path: Optuna研究存储路径
        """
        self.model_name = model_params_strategy.model_name
        self.data = None
        self.study_path = Path(settings.get('models.model_path')) / self.model_name
        self.study_file = self.study_path / f"optuna_study_{self.model_name}.pkl"
        self.model_params_strategy = model_params_strategy

        # 创建目录
        self.study_file.parent.mkdir(parents=True, exist_ok=True)
        # 准备数据
        self.prepare_data()
        # 初始化模型工厂
        self.model_factory = ModelFactory()


    def prepare_data(self):
        """
        数据准备和预处理
        """
        data_precessor = TimeSeriesProcessor()
        data = data_precessor.generate_processed_series_data()
        # 生成train、test、val
        test_length = settings.get("data.test_data_length", 45)
        buffer_length = settings.get("data.initial_buffer_data = 120", 120)
        data['test'] = data['target'][-(test_length + buffer_length) : ]
        val_length = settings.get("data.val_data_length", 45)
        data['val'] = data['target'][-(test_length + val_length + buffer_length) : -test_length]
        data['train'] = data['target'][:-(test_length + val_length)]
        data['trian_val'] = data['target'][ : -test_length]
        self.data = data

        return self.data

    def _create_study(
            self,
            direction: str = 'minimize',
            load_if_exists: bool = True
    ) -> optuna.Study:
        """
        创建或加载 Optuna 研究对象

        Args:
            direction: 优化方向
            study_name: 研究名称
            load_if_exists: 是否加载已存在的研究

        Returns:
            Optuna 研究对象
        """
        try:
            # 尝试加载已存在的研究
            if load_if_exists and os.path.exists(self.study_file):
                study = joblib.load(self.study_file.as_posix())
                logger.info(f"成功加载现有研究: {self.study_file}")
                return study

                # 创建新的研究
            study = optuna.create_study(
                direction=direction,
                study_name=f"optuna_study_{self.model_name}"
            )
            return study

        except Exception as e:
            logger.error(f"研究创建/加载失败: {e}")
            raise

    def _save_study(self, study: optuna.Study):
        """
        保存 Optuna 研究对象到本地

        Args:
            study: Optuna 研究对象
        """
        try:
            joblib.dump(study, self.study_file)
            logger.info(f"研究已保存到: {self.study_file}")
        except Exception as e:
            logger.error(f"研究保存失败: {e}")

    def objective(self, trial):
        """
        Optuna 目标函数，用于模型超参数优化

        Args:
            trial (optuna.Trial): Optuna 试验对象

        Returns:
            float: 模型性能指标
        """

        # 注册策略
        self.model_factory.register_param_strategy(self.model_params_strategy)

        # 创建模型
        model = self.model_factory.define_trial_model(trial)

        try:
            evaluation = train_and_evaluate(model, self.data)
            return evaluation['precision']

        except Exception as e:
            print(f"模型训练失败: {e}")
            return float('inf')

    @staticmethod
    def print_progress_callback(study, trial):
        """
        打印优化进度的回调函数
        """
        # 计算完成百分比
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        total_trials = len(study.trials)
        progress_percentage = (completed_trials / total_trials) * 100 if total_trials > 0 else 0

        # 打印进度信息
        print(f"Trial {trial.number}: Progress {progress_percentage:.2f}% "
              f"(Completed: {completed_trials}/{total_trials})")

        # 可选：打印当前试验的参数和值
        if trial.value is not None:
            print(f"  Current Value: {trial.value}")
            print(f"  Current Params: {trial.params}")


    def optimize_hyperparameters(
            self,
            n_trials: int = 100,
            timeout: Optional[float] = None,
            callbacks: Optional[list] = None,
            additional_config: Optional[Dict[str, Any]] = None
    ) -> optuna.Study:
        """
        使用 Optuna 进行超参数优化，支持断点续训

        Args:
            n_trials: 优化试验次数
            timeout: 超时时间（秒）
            callbacks: 额外的回调函数
            additional_config: 额外的配置参数

        Returns:
            Optuna 研究对象
        """
        # 默认回调函数
        default_callbacks = [
            # 定期保存研究
            optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,)),
            # 打印进度
            self.print_progress_callback
        ]

        # 合并回调函数
        all_callbacks = (callbacks or []) + default_callbacks

        # 创建或加载研究
        study = self._create_study()

        try:
            # 开始优化
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=all_callbacks
            )

            # 保存研究结果
            self._save_study(study)

            # 打印最佳结果
            self._log_best_trial(study)

            return study

        except Exception as e:
            logger.error(f"超参数优化失败: {e}")
            # 保存当前进度
            self._save_study(study)
            raise

    def _log_best_trial(self, study: optuna.Study):
        """
        记录最佳试验信息

        Args:
            study: Optuna 研究对象
        """
        best_trial = study.best_trial
        logger.info("最佳试验信息:")
        logger.info(f"  值: {best_trial.value}")
        logger.info("  最佳超参数:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

    def resume_optimization(
            self,
            additional_trials: int = 50,
            timeout: Optional[float] = None
    ) -> optuna.Study:
        """
        从上次中断的地方继续优化

        Args:
            additional_trials: 额外的试验次数
            timeout: 超时时间

        Returns:
            更新后的 Optuna 研究对象
        """
        try:
            # 加载现有研究
            study = joblib.load(self.study_file)

            # 继续优化
            study.optimize(
                self.objective,
                n_trials=additional_trials,
                timeout=timeout
            )

            # 保存更新后的研究
            self._save_study(study)

            # 记录最佳试验
            self._log_best_trial(study)

            return study

        except FileNotFoundError:
            logger.warning("未找到现有研究，将创建新的研究")
            return self.optimize_hyperparameters(n_trials=additional_trials)
        except Exception as e:
            logger.error(f"恢复优化失败: {e}")
            raise

        # 使用示例


def main():
    # 初始化训练器

    model_params_strategy = TiDEModelParamStrategy
    trainer = TimeSeriesModelTrainer(model_params_strategy=model_params_strategy)

    # 第一次运行优化
    study = trainer.optimize_hyperparameters(
        n_trials=50,  # 初始试验数
        # timeout=3600,  # 1小时超时
        callbacks=[
            # 可添加自定义回调
        ]
    )

    # 意外中断后恢复优化
    try:
        # 从上次中断处继续
        updated_study = trainer.resume_optimization(
            additional_trials=50,  # 额外的试验数
            # timeout=1800  # 30分钟超时
        )
    except Exception as e:
        logger.error(f"恢复优化失败: {e}")


if __name__ == "__main__":
    main()