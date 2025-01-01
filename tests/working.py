from data_processing.generate_training_data import main
from xtquant import xtdata
import pandas as pd
from data_processing.get_securities import get_investment_target
from models.optimize_hyperparam import optimize_hyperparam_for_models
from models.train_final_model import train_final_model

stocks = get_investment_target()
if __name__ == '__main__':
    # 获取本地数据
    train_final_model()
    print("*"*35)