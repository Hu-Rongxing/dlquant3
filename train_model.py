from models.optimize_hyperparam import train_separate_models
from models.load_study import train_final_model

if __name__ == '__main__':
    # 训练单个基础模型。
    train_separate_models()
    train_final_model()