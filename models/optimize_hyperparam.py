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
from .train_functions import train_and_evaluate
from config import settings
from logger import log_manager

logger = log_manager.get_logger(__name__)


class TimeSeriesModelTrainer:
    def __init__(self, model_params_strategy: BaseParamStrategy):
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
        self.best_score = float('-inf')  # 初始化最佳得分为负无穷

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
        data_processor = TimeSeriesProcessor()
        self.data = data_processor.generate_processed_series_data()
        return self.data

    def _create_study(
            self,
            direction: str = 'maximize',
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
            current_score = evaluation['precision']

            # 更新最佳得分
            if current_score > self.best_score:
                self.best_score = current_score
                logger.info(f"新的最佳得分: {self.best_score} (Trial {trial.number})")

            return current_score

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
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
        logger.info(f"Trial {trial.number}: Progress {progress_percentage:.2f}% "  
                    f"(Completed: {completed_trials}/{total_trials})")

        # 可选：打印当前试验的参数和值
        if trial.value is not None:
            logger.info(f"  Current Value: {trial.value}")
            logger.info(f"  Current Params: {trial.params}")

    def optimize_hyperparameters(
            self,
            n_trials: int = 100,
            timeout: Optional[float] = None,
            save_interval: int = 3,  # 每10个trials保存一次
            callbacks: Optional[list] = None,
            additional_config: Optional[Dict[str, Any]] = None
    ) -> optuna.Study:
        # 自定义定期保存的回调
        def periodic_save_callback(study, trial):
            # 每隔 save_interval 个完成的试验保存一次
            if trial.number > 0 and trial.number % save_interval == 0:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    try:
                        self._save_study(study)
                        logger.info(f"已在第 {trial.number} 个试验处保存研究进度")
                    except Exception as e:
                        logger.error(f"定期保存失败: {e}")

        # 默认回调函数
        default_callbacks = [
            periodic_save_callback,
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

            # 最终保存
            self._save_study(study)

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


def train_separate_models():
    # 初始化训练器
    for model_params_strategy in [
        TiDEModelParamStrategy,
        TSMixerModelParamStrategy,
        XGBModelModelParamStrategy,
        LightGBMModelParamStrategy,
        TFTModelParamStrategy
    ]:
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


