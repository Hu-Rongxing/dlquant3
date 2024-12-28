from .get_securities import get_investment_target
from xtquant import xtdata


stocks = get_investment_target()
for stock in stocks.securities:
    dd = xtdata.get_divid_factors(stock)
    dd['time'] = dd['time'].astype('int64')
    # 将dd写入数据库
    # divid_datas.columns =
    # Index(['time', 'interest', 'stockBonus', 'stockGift', 'allotNum', 'allotPrice',
    #        'gugai', 'dr'],
    #       dtype='object')

    # divid_datas.index =
    # Index(['20140813', '20161213', '20170804', '20180817', '20190726', '20200818',
    #        '20210806', '20220729', '20230728', '20240729'],
    #       dtype='object')
    # TODO: 将数据写入数据库， 支持增量更新。

