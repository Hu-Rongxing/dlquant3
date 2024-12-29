from data_processing.generate_training_data import main
from xtquant import xtdata
import pandas as pd
from data_processing.get_securities import get_investment_target
from models.optimize_hyperparam import main

stocks = get_investment_target()
if __name__ == '__main__':
    # 获取本地数据
    main()
    print("*"*35)