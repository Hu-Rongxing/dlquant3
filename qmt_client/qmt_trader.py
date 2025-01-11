# 导入必要的库
import math
import os
import time
from datetime import datetime
from typing import List, Optional

# 迅投API
from xtquant import xtconstant
from xtquant import xtdata
from xtquant.xttype import StockAccount
from xtquant.xttrader import XtQuantTrader

# 自定义模块
from config import settings
from logger import log_manager
from utils.others import generate_session_id
from .qmt_callback import MyXtQuantTraderCallback
from .qmt_monitor import start_xt_client, restart_xt_client
from .data_class import order_type_dic

"""  
执行迅投客户端的账户登录、交易等操作。  
"""

# 提权
os.environ["__COMPAT_LAYER"] = "RunAsInvoker"

# 获取日志记录器
applogger = log_manager.get_logger(__name__)  # 日志系统

# 最大持仓数
MAX_POSITIONS: int = settings.get("strategy.max_positions", 4)

# 用户ID
USER_ID: str = settings.get("mini_xt.MINI_XT_USER")

# 账户信息
ACC: StockAccount = StockAccount(USER_ID)

def setup_xt_trader() -> XtQuantTrader:
    """
    设置并启动XTQuantTrader。

    Args:
        acc (StockAccount): 交易账户信息。默认为ACC。

    Returns:
        XtQuantTrader: 配置好的XTQuantTrader实例。

    Raises:
        RuntimeError: 如果无法连接到XT服务器。
    """
    acc: StockAccount = ACC
    # 启动XT客户端
    start_xt_client()

    # 创建回调函数
    callback = MyXtQuantTraderCallback()

    # 获取用户数据路径
    user_data_path = settings.get("xt_client.xt_data_dir")

    # 生成会话ID
    session_id = generate_session_id()

    # 创建XTQuantTrader实例
    xt_trader = XtQuantTrader(user_data_path, session_id)

    # 注册回调函数
    xt_trader.register_callback(callback)

    # 启动XTQuantTrader
    xt_trader.start()

    # 尝试连接交易服务器
    if xt_trader.connect() < 0:
        # 如果连接失败，重启XT客户端
        restart_xt_client()
        if xt_trader.connect() < 0:
            # 如果仍然连接失败，抛出异常
            raise RuntimeError('无法连接到XT服务器。')

    # 订阅账户
    xt_trader.subscribe(acc)

    return xt_trader

def get_max_ask_price(stock_code: str) -> Optional[float]:
    """
    获取指定股票代码的五档行情最高报价

    Args:
        stock_code (str): 股票代码，例如 "000001.SZ"

    Returns:
        Optional[float]: 最新股价，如果获取失败则返回 None
    """
    # 如果stock_code是字符串，转换为列表
    if isinstance(stock_code, str):
        code_list = [stock_code]
    else:
        code_list = stock_code
        stock_code = code_list[0]

    try:
        # 获取全 tick 数据
        data = xtdata.get_full_tick(code_list)

        # 如果数据存在并且不为空
        if stock_code in data and bool(data[stock_code]):
            # 获取时间
            time = data[stock_code]['timetag']

            # 计算最高卖价
            max_ask_price = max(
                max(data[stock_code]['askPrice']),  # 最高卖价
                max(data[stock_code]['bidPrice']),  # 最高买价
                data[stock_code]['lastPrice'] * 1.01  # 最新价+1%
            )

            # 保留两位小数
            max_ask_price = math.ceil(max_ask_price * 100) / 100

            # 获取合约基础信息
            instrument = xtdata.get_instrument_detail(stock_code, iscomplete=True)

            # 如果成交价等于涨停价
            if data[stock_code]['lastPrice'] == instrument["UpStopPrice"]:
                applogger.warning(f"{stock_code}涨停")
                return 999999

            # 如果合约不可交易
            if instrument["InstrumentStatus"] > 0:
                applogger.error(f"证券{stock_code}处于停牌状态{instrument['InstrumentStatus']}")
                return 999998

            # 不超过涨停价
            if instrument["UpStopPrice"] > 0:
                max_ask_price = min(max_ask_price, instrument["UpStopPrice"])
            else:
                applogger.warning(f"{stock_code}涨停价异常")

            applogger.info(f"股票【{stock_code}】的最高价格: {max_ask_price}")

            # 返回最高卖价
            return max_ask_price
        else:
            # 如果数据不存在或为空，返回 None
            applogger.error(f"未能获取股票 {stock_code} 的数据")
            return None
    except Exception as e:
        # 如果发生错误，返回 None
        applogger.error(f"获取股票 {stock_code} 的数据时发生错误: {e}")
        # raise Exception("获取股票最高价错误")
        return None

def buy_stock_async(stocks: List[str], strategy_name: str = '', order_remark: str = '') -> bool:
    """
    异步买入股票。

    根据股票代码后缀确定所属市场并设置订单类型，然后发出买入指令。

    Args:
        stocks (List[str]): 需要买入的股票代码列表。
        strategy_name (str, optional): 策略名称。默认为空字符串。
        order_remark (str, optional): 订单备注。默认为空字符串。

    Returns:
        bool: 如果成功发出订单返回 True，否则返回 False。
    """
    # 设置交易者
    xt_trader = setup_xt_trader()

    # 查询资产状况
    applogger.info("查询资产状况。")
    for attempt in range(15):
        asset = xt_trader.query_stock_asset(ACC)
        if asset:
            break
        applogger.warning(f"尝试 {attempt + 1}/15: xt_trader.query_stock_asset 返回 None。")
        time.sleep(1)
    else:
        applogger.error("无法获取资产信息：xt_trader.query_stock_asset 返回 None。")
        return False

    # 获取可用现金
    cash: float = asset.cash
    applogger.info(f"可用现金: {cash}。")

    # 查询持仓信息
    positions = xt_trader.query_stock_positions(ACC)
    positions_stocks = {pos.stock_code for pos in positions if pos.volume > 0}

    # 计算可买入的股票
    to_buy_stocks = [s for s in stocks if s not in positions_stocks]
    available_slots = min(MAX_POSITIONS - len(positions_stocks), len(to_buy_stocks), MAX_POSITIONS)
    available_slots = max(available_slots, 0)
    applogger.info(f"可用持仓槽位: {available_slots}")

    # 如果没有可买入的股票，返回 False
    if available_slots == 0:
        applogger.info(f"当前持仓已满: {list(positions_stocks)}。")
        return False

    # 买入股票
    for stock_code in to_buy_stocks[:available_slots]:
        order_type = xtconstant.FIX_PRICE  # 统一使用固定价格
        applogger.info(f"股票【{stock_code}】报价类型: {order_type}")

        # 获取最高卖价
        max_ask_price = get_max_ask_price(stock_code)

        # 如果最高卖价异常，跳过
        if max_ask_price in {999999, 999998}:
            status_message = {
                999999: f"股票已涨停: {stock_code}",
                999998: f"当前合约不可交易: {stock_code}"
            }.get(max_ask_price, "max_ask_price异常")
            applogger.warning(status_message)
            continue

        # 如果最高卖价为空或小于0，跳过
        if not max_ask_price or max_ask_price <= 0:
            applogger.warning(f"无法获取股票 {stock_code} 的有效要价。")
            continue

        # 计算买入数量
        quantity = math.floor(cash / max_ask_price / available_slots / 100) * 100
        if quantity < 100:
            applogger.info(f"股票 {stock_code} 可买数量不足，现金: {cash}, 当前股价: {max_ask_price}")
            continue

        try:
            # 下单
            response = xt_trader.order_stock(
                account=ACC,
                stock_code=stock_code,
                order_type=xtconstant.STOCK_BUY,
                order_volume=quantity,
                price_type=order_type,
                price=max_ask_price,
                strategy_name=strategy_name,
                order_remark=order_remark
            )
            applogger.info("完成提交。")
            if response < 0:
                applogger.trader(
                    f"【提交下单失败！- 买入 - {strategy_name}】股票: {stock_code}, 数量: {quantity}, "  
                    f"单价: {max_ask_price}, 金额: {quantity * max_ask_price}, 返回值: {response}"
                )
            else:
                applogger.trader(
                    f"【提交下单成功！- 买入 - {strategy_name}】股票: {stock_code}, 数量: {quantity}, "  
                    f"单价: {max_ask_price}, 金额: {quantity * max_ask_price}, 返回值: {response}"
                )
        except Exception as e:
            applogger.error(f"下单时发生错误：{e}")
            return False

    return True

def cancel_all_order_async():
    """ 撤销未完全成交的挂单"""
    xt_trader = setup_xt_trader()
    pending_orders = xt_trader.query_stock_orders(ACC)
    for order in pending_orders:
        if order.order_status in [
            xtconstant.ORDER_UNREPORTED,
            xtconstant.ORDER_WAIT_REPORTING,
            xtconstant.ORDER_REPORTED,
            xtconstant.ORDER_REPORTED_CANCEL,
            xtconstant.ORDER_PARTSUCC_CANCEL,
            xtconstant.ORDER_PART_SUCC,
            xtconstant.ORDER_JUNK,
            xtconstant.ORDER_UNKNOWN
        ]:
            cancel_response = xt_trader.cancel_order_stock_async(ACC, order.order_id)
            applogger.info(f"撤销订单 {order.order_id}，响应: {cancel_response}")

def generate_trading_report() -> str:
    """
    生成交易报告。

    查询资产、持仓和当日成交记录，并生成格式化的报告字符串。

    Returns:
        str: 交易报告。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    xt_trader = setup_xt_trader()

    try:
        # 查询资产信息
        asset = xt_trader.query_stock_asset(ACC)
        # 查询持仓信息
        positions = xt_trader.query_stock_positions(ACC)
        # 查询当日成交记录
        trades = xt_trader.query_stock_trades(ACC)
    except Exception as e:
        applogger.error(f"查询交易数据时发生错误：{e}")
        return ""

    # 构造报告行
    report_lines = [
        f"<h1 style='color: #00698f;'>交易报告 - {today}</h1>",
        "<hr style='border: 1px solid #ccc; margin: 10px 0;'>",
        "<h3 style='color: #666;'>资产情况：</h3>",
        f"<p style='font-size: 14px; color: #666;'>--资产代码: {asset.account_id}, 总资产: {asset.total_asset}, "  
        f"股票市值: {asset.market_value}, 可用现金: {asset.cash}</p>",
        "<hr style='border: 1px solid #ccc; margin: 10px 0;'>",
        "<h3 style='color: #666;'>持仓情况：</h3>",
    ]

    # 添加持仓信息
    for position in positions:
        report_lines.append(
            f"<p style='font-size: 14px; color: #666;'>--股票代码: {position.stock_code}, 股票市值: {position.market_value}, "  
            f"持股数量: {position.volume}, 平均成本: {position.avg_price}</p>"
        )

    # 添加当日成交记录
    report_lines.extend([
        "<hr style='border: 1px solid #ccc; margin: 10px 0;'>",
        "<h3 style='color: #666;'>当日成交：</h3>",
    ])

    for trade in trades:
        traded_time = datetime.fromtimestamp(trade.traded_time).strftime("%Y-%m-%d %H:%M:%S")
        order_type = order_type_dic.get(trade.order_type, '未定义')
        report_lines.append(
            f"<p style='font-size: 14px; color: #666;'>--{order_type} - {trade.strategy_name}】股票代码: {trade.stock_code}, "  
            f"成交金额: {trade.traded_amount}, 成交数量: {trade.traded_volume}, "  
            f"成交价格: {trade.traded_price}, 成交时间: {traded_time}, 备注: {trade.order_remark}</p>"
        )

    # 添加分隔线
    report_lines.append("<hr style='border: 1px solid #ccc; margin: 10px 0;'>")

    # 构造报告字符串
    report = ''.join(report_lines)

    # 记录报告
    applogger.trader(report)
    return report