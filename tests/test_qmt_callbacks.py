import time

from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant
# 自定义
from logger import log_manager
from qmt_client.qmt_callback import MyXtQuantTraderCallback
from config import settings
from utils.others import generate_session_id

app_logger = log_manager.get_logger(__name__)


def test_xtclient_callbacks():
    app_logger.info("demo test")

    # path为mini qmt客户端安装目录下userdata_mini路径
    path = settings.get("xt_client.xt_data_dir")

    # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号
    session_id = generate_session_id()
    print(path)
    user = settings.get("mini_xt.MINI_XT_USER")
    xt_trader = XtQuantTrader(str(path), session_id)
    acc = StockAccount(user)

    callback = MyXtQuantTraderCallback()
    xt_trader.register_callback(callback)
    xt_trader.start()

    connect_result = xt_trader.connect()
    app_logger.info(f"Connect result: {connect_result}")

    subscribe_result = xt_trader.subscribe(acc)
    app_logger.info(f"Subscribe result: {subscribe_result}")

    stock_code = '600000.SH'

    # 使用指定价下单，接口返回订单编号，后续可以用于撤单操作以及查询委托状态
    app_logger.info("order using the fix price:")
    fix_result_order_id = xt_trader.order_stock(acc, stock_code, xtconstant.STOCK_BUY, 200, xtconstant.FIX_PRICE, 10.5,
                                                'strategy_name', 'remark')
    app_logger.info(f"Fix result order ID: {fix_result_order_id}")

    # 使用订单编号撤单
    app_logger.info("cancel order:")
    cancel_order_result = xt_trader.cancel_order_stock(acc, fix_result_order_id)
    app_logger.info(f"Cancel order result: {cancel_order_result}")

    # 使用异步下单接口，接口返回下单请求序号seq，seq可以和on_order_stock_async_response的委托反馈response对应起来
    app_logger.info("order using async api:")
    async_seq = xt_trader.order_stock_async(acc, stock_code, xtconstant.STOCK_BUY, 200, xtconstant.FIX_PRICE, 10.5,
                                            'strategy_name', 'remark')
    app_logger.info(f"Async seq: {async_seq}")

    # 查询证券资产
    app_logger.info("query asset:")
    asset = xt_trader.query_stock_asset(acc)
    if asset:
        app_logger.info(f"Asset cash: {asset.cash}")

        # 根据订单编号查询委托
    app_logger.info("query order:")
    order = xt_trader.query_stock_order(acc, fix_result_order_id)
    if order:
        app_logger.info(f"Order ID: {order.order_id}")

        # 查询当日所有的委托
    app_logger.info("query orders:")
    orders = xt_trader.query_stock_orders(acc)
    app_logger.info(f"Orders count: {len(orders)}")
    if orders:
        last_order = orders[-1]
        app_logger.info(
            f"Last order - stock_code: {last_order.stock_code}, order_volume: {last_order.order_volume}, price: {last_order.price}")

        # 查询当日所有的成交
    app_logger.info("query trade:")
    trades = xt_trader.query_stock_trades(acc)
    app_logger.info(f"Trades count: {len(trades)}")
    if trades:
        last_trade = trades[-1]
        app_logger.info(
            f"Last trade - stock_code: {last_trade.stock_code}, traded_volume: {last_trade.traded_volume}, traded_price: {last_trade.traded_price}")

        # 查询当日所有的持仓
    app_logger.info("query positions:")
    positions = xt_trader.query_stock_positions(acc)
    app_logger.info(f"Positions count: {len(positions)}")
    if positions:
        last_position = positions[-1]
        app_logger.info(
            f"Last position - account_id: {last_position.account_id}, stock_code: {last_position.stock_code}, volume: {last_position.volume}")

        # 根据股票代码查询对应持仓
    app_logger.info("query position:")
    position = xt_trader.query_stock_position(acc, stock_code)
    if position:
        app_logger.info(
            f"Position - account_id: {position.account_id}, stock_code: {position.stock_code}, volume: {position.volume}")

        # 阻塞线程，接收交易推送
    time.sleep(5)
    # xt_trader.run_forever()