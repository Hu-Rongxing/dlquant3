from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import optuna

from darts.models import (
    LightGBMModel,
    XGBModel,
    TSMixerModel,
    TFTModel,
    TiDEModel
)

# 假设这些是您的自定义模块
from config import settings
from logger import log_manager

# 定义模型类型的联合类型
ModelType = Union[
    LightGBMModel,
    XGBModel,
    TSMixerModel,
    TFTModel,
    TiDEModel
]


class BaseParamStrategy(ABC):
    model_name:str = None
    model = None
    """
    参数策略的基类，提供通用的参数管理功能
    """

    def __init__(self):
        self.logger = log_manager.get_logger(__name__)
        self.params: Dict[str, Any] = {
            "random_state": 42
        }

    def update_params(self, **params: Any) -> Dict[str, Any]:
        """
        更新模型参数

        Args:
            **params: 待更新的参数

        Returns:
            更新后的完整参数字典
        """
        return {**self.params, **params}

    @abstractmethod
    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """
        为超参数优化生成参数

        Args:
            trial: Optuna 试验对象

        Raises:
            NotImplementedError: 未实现的抽象方法
        """
        raise NotImplementedError("必须实现 generate_trail_params 方法")

    def generate_common_params(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成通用参数

        Args:
            params_dict: 参数字典

        Returns:
            处理后的参数字典
        """
        return params_dict


class ModelFactory:
    """
    提供生成深度学习模型、机器学习模型的统一框架。
    """

    def __init__(self):
        self.logger = log_manager.get_logger(__name__)
        self.model_name: Optional[str] = None
        self.param_strategy: Optional[BaseParamStrategy] = None
        self.model: Optional[ModelType] = None
        self.logger = log_manager.get_logger(__name__)

    def register_param_strategy(
            self,
            param_strategy_cls: Type[BaseParamStrategy]
    ) -> None:
        """
        注册参数策略

        Args:
            param_strategy_cls: 参数策略类
        """
        try:
            self.param_strategy = param_strategy_cls()
            # 使用 getattr 安全地获取 name 属性
            self.model_name = getattr(param_strategy_cls, 'name', None)
        except Exception as e:
            self.logger.error(f"注册参数策略失败: {e}")
            raise

    def define_trial_model(
            self,
            trial: optuna.trial.Trial
    ) -> ModelType:
        """
        为超参数优化创建模型

        Args:
            trial: Optuna 试验对象

        Returns:
            创建的模型实例

        Raises:
            ValueError: 未注册参数策略
        """
        if not self.param_strategy:
            raise ValueError("未注册参数策略")

        try:
            # 获取模型类
            model_cls = getattr(self.param_strategy, 'model', None)
            if not model_cls:
                raise ValueError("未找到模型类")

            # 生成参数
            params = self.param_strategy.generate_trail_params(trial)
            self.model = model_cls(**params)

            # 设置模型名称
            if not hasattr(self.model, 'model_name'):
                self.model.model_name = self.model_name

            return self.model
        except Exception as e:
            self.logger.error(f"创建试验模型失败: {e}")
            raise

    def define_common_model(
            self,
            params_dict: Dict[str, Any]
    ) -> ModelType:
        """
        使用通用参数创建模型

        Args:
            params_dict: 模型参数字典

        Returns:
            创建的模型实例

        Raises:
            ValueError: 未注册参数策略
        """
        if not self.param_strategy:
            raise ValueError("未注册参数策略")

        try:
            # 获取模型类
            model_cls = getattr(self.param_strategy, 'model', None)
            if not model_cls:
                raise ValueError("未找到模型类")

            # 生成参数
            params = self.param_strategy.generate_common_params(params_dict)
            self.model = model_cls(**params)

            # 设置模型名称
            if not hasattr(self.model, 'model_name'):
                self.model.model_name = self.model_name

            return self.model
        except Exception as e:
            self.logger.error(f"创建通用模型失败: {e}")
            raise

class MachineModelParamStrategy(BaseParamStrategy):
    """定义通用darts机器学习模型添加一个通用形参verbose=-1"""

    def __init__(self):
        super().__init__()
        self.params = self.update_params(
            verbose=-1
        )

class LightGBMModelParamStrategy(MachineModelParamStrategy):
    model_name = 'LightGBMModel'
    model = LightGBMModel
    """
    定义LightGBM模型。
    """
    def generate_trail_params(self, trial: optuna.trial.Trial):
        future_past = trial.suggest_int("future_past", 1, 5)
        # b = trial.suggest_int("b", 0, 5)
        params = {
            # 定义过去时间序列数据的滞后步数
            "lags": trial.suggest_int("lags", 1, 60),
            # 定义过去协变量的滞后步数
            "lags_past_covariates": trial.suggest_int("lags_past_covariates", 1, 60),
            # 定义未来协变量的滞后步数，使用元组形式
            "lags_future_covariates": (future_past, 1),
            # 定义输出时间序列的长度
            # "output_chunk_length": trial.suggest_int("output_chunk_length", 1, min(20, PRED_STEPS)),
            # 决策树的数量
            "n_estimators": trial.suggest_int("n_estimators", 5, 120),
            # 每棵树的学习率
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # 决策树的最大深度
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            # 每棵树的子样本比例
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # 每棵树的特征列采样比例
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # L1 正则化项
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            # L2 正则化项
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            # LightGBM 专用参数：最小分割增益
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        return self.update_params(**params)


    def generate_common_params(self, params_dict: Dict):
        future_past = params_dict.pop("future_past", 1)
        params_dict.update({"lags_future_covariates": (future_past, 1)})
        return self.update_params(**params_dict)


class XGBModelModelParamStrategy(MachineModelParamStrategy):
    """
    定义LightGBM模型。
    """
    model_name = 'XGBModel'
    model = XGBModel
    def generate_trail_params(self, trial: optuna.trial.Trial):
        future_past = trial.suggest_int("future_past", 1, 5)
        params = {
            "lags": trial.suggest_int("lags", 1, 60),
            "lags_past_covariates": trial.suggest_int("lags_past_covariates", 1, 60),
            "lags_future_covariates": (future_past, 1),
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
        }

        return self.update_params(**params)

    def generate_common_params(self, params_dict: Dict):
        future_past = params_dict.pop("future_past", 1)
        params_dict.update({"lags_future_covariates": (future_past, 1)})
        return self.update_params(**params_dict)


class DeepLearnModelParamStrategy(BaseParamStrategy):
    """定义通用darts机器学习模型添加一个通用形参verbose=-1"""

    def __init__(self):
        super().__init__()
        self.params = self.update_params(
            # 添加公共参数和默认参数，字类将改写部分参数。
            batch_size = 64,
            n_epochs= 100,
            model_name = None,
            work_dir = settings.get("models.model_path", "models/saved_models"),
            log_tensorboard = False,
            nr_epochs_val_period = 1,
            force_reset = False,
            save_checkpoints = False,
            add_encoders = None,
            random_state = None,
            pl_trainer_kwargs = None,
            show_warnings = False
        )

class TFTModelParamStrategy(DeepLearnModelParamStrategy):
    """
    定义LightGBM模型。
    """
    model_name = 'TFTModel'
    model = TFTModel

    def __init__(self):
        super().__init__()
        work_dir = Path(settings.get("models.model_path", "models/saved_models")) / "TFTModel"

        # 如果目录不存在，则创建
        work_dir.mkdir(parents=True, exist_ok=True)

        self.params = self.update_params(
            model_name='TFTModel',
            work_dir=work_dir.as_posix(),
        )

    def generate_trail_params(self, trial: optuna.trial.Trial):
        params = {
            "input_chunk_length": trial.suggest_int("input_chunk_length", 1, 64),
            "output_chunk_length": trial.suggest_int("output_chunk_length", 1, 20),
            "hidden_size": trial.suggest_int("hidden_size", 8, 128),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 4),
            "num_attention_heads": trial.suggest_int("num_attention_heads", 1, 4),
            "full_attention": trial.suggest_categorical("full_attention", [True, False]),
            "feed_forward": trial.suggest_categorical("feed_forward", ['GatedResidualNetwork', 'ReLU']),
            "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 4, 32)
        }
        return self.update_params(**params)


class TSMixerModelParamStrategy(DeepLearnModelParamStrategy):
    """
    定义LightGBM模型。
    """
    model_name = 'TSMixerModel'
    model = TSMixerModel

    def __init__(self):
        super().__init__()
        work_dir = Path(settings.get("models.model_path", "models/saved_models")) / "TSMixerModel"

        # 如果目录不存在，则创建
        work_dir.mkdir(parents=True, exist_ok=True)

        self.params = self.update_params(
            model_name='TSMixerModel',
            work_dir=work_dir.as_posix(),
        )

    def generate_trail_params(self, trial: optuna.trial.Trial):
        params = {
            "input_chunk_length": trial.suggest_int("input_chunk_length", 4, 64),
            "output_chunk_length": trial.suggest_int("output_chunk_length", 1, 20),
            "hidden_size": trial.suggest_int("hidden_size", 32, 512),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("activation", ["ReLU", "GELU", "SELU"]),
            "num_blocks": trial.suggest_int("num_blocks", 1, 4)
        }
        return self.update_params(**params)

class TiDEModelParamStrategy(DeepLearnModelParamStrategy):
    """
    定义LightGBM模型。
    """
    model_name = 'TiDEModel'
    model = TiDEModel

    def __init__(self):
        super().__init__()
        work_dir = Path(settings.get("models.model_path", "models/saved_models")) / "TiDEModel"

        # 如果目录不存在，则创建
        work_dir.mkdir(parents=True, exist_ok=True)

        self.params = self.update_params(
            model_name='TiDEModel',
            work_dir=work_dir.as_posix(),
        )

    def generate_trail_params(self, trial: optuna.trial.Trial):
        params = {
            "input_chunk_length": trial.suggest_int("input_chunk_length", 1, 64),
            "output_chunk_length": trial.suggest_int("output_chunk_length", 1, 20),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 1, 4),
            "decoder_output_dim": trial.suggest_int("decoder_output_dim", 1, 64),
            "hidden_size": trial.suggest_int("hidden_size", 32, 256),
            "temporal_width_past": trial.suggest_int("temporal_width_past", 1, 32),
            "temporal_decoder_hidden": trial.suggest_int("temporal_decoder_hidden", 8, 64),
            "use_layer_norm": trial.suggest_categorical("use_layer_norm", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
        return self.update_params(**params)


model_factory_dict = {
    TiDEModelParamStrategy,
    TSMixerModelParamStrategy,
    XGBModelModelParamStrategy,
    LightGBMModelParamStrategy,
    TFTModelParamStrategy
}


