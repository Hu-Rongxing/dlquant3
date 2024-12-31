from pathlib import Path  # 用于跨平台的文件路径操作  
import importlib  # 用于动态导入模块和类  

# 自定义模块导入  
from logger import log_manager  # 从当前包中导入自定义的日志记录器  
logger = log_manager.get_logger(__name__)  

def save_darts_model(model, model_path: str) -> Path:  
    """  
    保存训练好的模型到文件。  

    参数:  
        model: 训练好的模型对象，要求模型具有 `model_name` 属性和 `save` 方法。  
        model_path (str): 模型保存的路径。  

    返回:  
        Path: 保存的模型文件路径。  
    """  
    model_dir = Path(model_path)  
    model_dir.parent.mkdir(parents=True, exist_ok=True)  
    model.save(model_dir.as_posix())
    logger.info(f"模型 {model.model_name} 已保存到 {model_path}")  
    return model_dir  # 返回保存的模型路径

def load_darts_model(model_name: str, model_path: str):  
    """  
    动态导入并加载指定名称的模型。  

    参数:  
        model_name (str): 模型的名称，用于动态导入和指定加载路径。  
        model_path (str): 模型加载的路径。  

    返回:  
        model: 加载的模型对象，已设置 `model_name` 属性。  
    """  
    # 动态导入 'darts.models' 模块  
    try:  
        module = importlib.import_module('darts.models')  # 导入模块  
    except ImportError as e:  
        logger.error(f"无法导入模块 'darts.models': {e}")  
        raise  

    # 从模块中获取指定的模型类  
    try:  
        darts_model_class = getattr(module, model_name)  # 获取类  
    except AttributeError:  
        logger.error(f"模块 'darts.models' 中没有找到类 '{model_name}'")  
        raise  

    # 尝试加载模型参数  
    try:  
        model_instance = darts_model_class.load(str(model_path))  # 假设模型对象具有 `load` 方法  
        logger.info(f"模型 {model_name} 已从 {model_path} 加载")  
    except FileNotFoundError:  
        logger.error(f"模型文件未找到: {model_path}")  
        raise  
    except Exception as e:  
        logger.error(f"加载模型失败: {e}")  
        raise  

    # 动态为模型对象添加 `model_name` 属性，以便后续使用  
    setattr(model_instance, "model_name", model_name)  

    return model_instance  # 返回加载的模型对象