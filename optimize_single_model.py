from models.optimize_hyperparam import optimize_hyperparam_for_models
from models.train_final_model import train_final_model

if __name__ == '__main__':
    # 训练单个基础模型。
    optimize_hyperparam_for_models()
    train_final_model()