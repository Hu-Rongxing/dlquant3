
import torch
from logger import log_manager

logger = log_manager.get_logger(__name__)

class GPUMemoryManager:
    """GPU内存管理器"""

    @staticmethod
    def clear_memory():
        """清理GPU内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"GPU内存清理失败: {e}")

    @staticmethod
    def get_gpu_memory_info():
        """获取GPU内存使用情况"""
        if not torch.cuda.is_available():
            return "No GPU available"

        try:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            return f"GPU Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB"
        except Exception as e:
            logger.warning(f"获取GPU内存信息失败: {e}")
            return "GPU memory info unavailable"