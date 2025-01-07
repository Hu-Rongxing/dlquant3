from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import optuna
from pytorch_lightning.callbacks import EarlyStopping
from darts.models import (
    LightGBMModel,
    XGBModel,
    TSMixerModel,
    TFTModel,
    TiDEModel
)
from config import settings
from logger import log_manager

# 定义模型类型的联合类型，以便在后续代码中使用
ModelType = Union[
    LightGBMModel,
    XGBModel,
    TSMixerModel,
    TFTModel,
    TiDEModel
]

class BaseParamStrategy(ABC):
    """
    添加所有darts模型公用的参数。
    """
    model_name: str = None
    model = None

    def __init__(self):
        """初始化参数策略基类"""
        self.logger = log_manager.get_logger(__name__)
        self.params: Dict[str, Any] = {
            "random_state": 42
        }

    def update_params(self, **params: Any) -> Dict[str, Any]:
        """更新模型参数"""
        self.params.update(params)  # 直接更新self.params
        return self.params

    @abstractmethod
    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """为超参数优化生成参数"""
        raise NotImplementedError("必须实现 generate_trail_params 方法")

    def generate_common_params(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """生成通用参数"""
        self.params.update(params_dict)
        return self.params

    def log_error(self, message: str) -> None:
        """统一的错误日志记录方法"""
        self.logger.error(message)


class ModelFactory:
    def __init__(self):
        """初始化模型工厂"""
        self.logger = log_manager.get_logger(__name__)
        self.model_name: Optional[str] = None
        self.param_strategy: Optional[BaseParamStrategy] = None
        self.model: Optional[ModelType] = None

    def register_param_strategy(self, param_strategy_cls: Type[BaseParamStrategy]) -> None:
        """注册参数策略"""
        try:
            self.param_strategy = param_strategy_cls()  # 实例化参数策略
            self.model_name = getattr(param_strategy_cls, 'model_name', None)
        except TypeError as e:
            self.logger.error(f"参数策略类错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"注册参数策略失败: {e}")
            raise

    def define_trial_model(self, trial: optuna.trial.Trial) -> ModelType:
        """为超参数优化创建模型实例"""
        if not self.param_strategy:
            raise ValueError("未注册参数策略")

        model_cls = getattr(self.param_strategy, 'model', None)
        if not model_cls:
            raise ValueError("未找到模型类")

        params = self.param_strategy.generate_trail_params(trial)
        self.model = model_cls(**params)

        self.set_model_name()  # 提取设置模型名称的逻辑

        return self.model

    def define_common_model(self, params_dict: Dict[str, Any]) -> ModelType:
        """使用通用参数创建模型实例"""
        if not self.param_strategy:
            raise ValueError("未注册参数策略")

        model_cls = getattr(self.param_strategy, 'model', None)
        if not model_cls:
            raise ValueError("未找到模型类")

        params = self.param_strategy.generate_common_params(params_dict)
        self.model = model_cls(**params)

        self.set_model_name()  # 提取设置模型名称的逻辑

        return self.model

    def set_model_name(self) -> None:
        """设置模型名称"""
        if not hasattr(self.model, 'model_name'):
            self.model.model_name = self.model_name


class MachineModelParamStrategy(BaseParamStrategy):
    """
    添加机器学习模型共用的参数
    """
    def __init__(self):
        """初始化机器学习模型参数策略"""
        super().__init__()
        self.params = self.update_params(
            early_stopping_rounds=10
        )


class LightGBMModelParamStrategy(MachineModelParamStrategy):
    model_name = 'LightGBMModel'  # 模型名称
    model = LightGBMModel  # 关联的模型类

    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """为LightGBM模型生成超参数"""
        future_past = trial.suggest_int("future_past", 1, 5)  # 未来协变量的滞后步数
        params = {
            "lags": trial.suggest_int("lags", 1, 60),
            "lags_past_covariates": trial.suggest_int("lags_past_covariates", 1, 60),
            "lags_future_covariates": (future_past, 1),
            "n_estimators": trial.suggest_int("n_estimators", 5, 120),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        return self.update_params(**params)

    def generate_common_params(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """为通用模式生成LightGBM模型参数"""
        future_past = params_dict.pop("future_past", 1)  # 获取未来协变量的滞后步数
        if 'verbose' in params_dict:
            params_dict.pop('verbose')
        params_dict.update({
            "lags_future_covariates": (future_past, 1),
            # "verbose": -1,
        })  # 更新参数字典
        return self.update_params(**params_dict)


class XGBModelModelParamStrategy(MachineModelParamStrategy):
    model_name = 'XGBModel'  # 模型名称
    model = XGBModel  # 关联的模型类

    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """为XGBoost模型生成超参数"""
        future_past = trial.suggest_int("future_past", 1, 5)  # 未来协变量的滞后步数
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

    def generate_common_params(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """为通用模式生成XGBoost模型参数"""
        future_past = params_dict.pop("future_past", 1)  # 获取未来协变量的滞后步数
        params_dict.update({"lags_future_covariates": (future_past, 1)})  # 更新参数字典
        return self.update_params(**params_dict)


class DeepLearnModelParamStrategy(BaseParamStrategy):
    """
    添加深度学习模型公用的参数
    """
    def __init__(self):
        """初始化深度学习模型参数策略"""
        super().__init__()
        self.params = self.update_params(
            batch_size=64,
            n_epochs=500,
            model_name=None,
            work_dir=settings.get("models.model_path", "models/saved_models"),
            log_tensorboard=False,
            nr_epochs_val_period=1,
            force_reset=False,
            save_checkpoints=False,
            add_encoders=None,
            random_state=None,
            pl_trainer_kwargs={
                 "callbacks": [
                    EarlyStopping(
                        monitor="val_loss",  # 监控的指标
                        patience=10,         # 早停耐心轮数
                        verbose=True,        # 是否打印早停信息
                        mode="min",           # 最小化监控指标
                        min_delta=0.00001
                    )
                ]
            },
            show_warnings=False
        )


class TFTModelParamStrategy(DeepLearnModelParamStrategy):
    model_name = 'TFTModel'  # 模型名称
    model = TFTModel  # 关联的模型类

    def __init__(self):
        """初始化TFT模型参数策略"""
        super().__init__()
        work_dir = Path(settings.get("models.model_path", "models/saved_models")) / "TFTModel"
        work_dir.mkdir(parents=True, exist_ok=True)  # 确保工作目录存在

        self.params = self.update_params(
            model_name='TFTModel',
            work_dir=work_dir.as_posix(),
        )

    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """为TFT模型生成超参数"""
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
    model_name = 'TSMixerModel'  # 模型名称
    model = TSMixerModel  # 关联的模型类

    def __init__(self):
        """初始化TSMixer模型参数策略"""
        super().__init__()
        work_dir = Path(settings.get("models.model_path", "models/saved_models")) / "TSMixerModel"
        work_dir.mkdir(parents=True, exist_ok=True)  # 确保工作目录存在

        self.params = self.update_params(
            model_name='TSMixerModel',
            work_dir=work_dir.as_posix(),
        )

    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """为TSMixer模型生成超参数"""
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
    model_name = 'TiDEModel'  # 模型名称
    model = TiDEModel  # 关联的模型类

    def __init__(self):
        """初始化TiDE模型参数策略"""
        super().__init__()
        work_dir = Path(settings.get("models.model_path", "models/saved_models")) / "TiDEModel"
        work_dir.mkdir(parents=True, exist_ok=True)  # 确保工作目录存在

        self.params = self.update_params(
            model_name='TiDEModel',
            work_dir=work_dir.as_posix(),
        )

    def generate_trail_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """为TiDE模型生成超参数"""
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