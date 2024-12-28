from data_processing.generate_training_data import main
from xtquant import xtdata
import pandas as pd
from data_processing.get_securities import get_investment_target

stocks = get_investment_target()
if __name__ == '__main__':
    # 获取本地数据
    market_data = xtdata.get_local_data(
        field_list=[],
        stock_list=stocks['securities'].to_list(),
        period='1d',
        start_time='20160101',
        end_time='20241227',
        count=15,
        dividend_type='front',
        fill_data=True
    )
    print(pd.DataFrame(market_data))