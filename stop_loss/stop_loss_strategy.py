import pickle
import time
import threading
from multiprocessing import Manager
from pathlib import Path
# 自定义
from xtquant import xtconstant, xtdata
from qmt_client.qmt_monitor import start_xt_client
from qmt_client.qmt_trader import ACC, setup_xt_trader
from logger import log_manager
from data_processing.get_securities import get_investment_target
from utils.others import is_trading_day, is_china_market_open
from config import settings


# 锁对象，确保线程安全
max_profit_lock = threading.Lock()
logger = log_manager.get_logger(__name__)
MAX_PROFIT_PATH = settings.get("stop_loss.max_profit_path")

class StopLossProgram:
    def __init__(self):
        self.profit_threshold = 0.01
        self.drawdown_threshold = 0.3333
        self.stop_loss_threshold = -0.01
        self.max_profit = {}
        self.positions = {}
        self.load_max_profit()
        self.positions_lock = threading.Lock()
        self.xt_trader = setup_xt_trader()

    def save_max_profit(self):
        """
        保存 max_profit 字典到文件，并在持仓变动时重置对应股票的最高收益率。
        """
        try:
            # 更新持仓信息
            self.update_positions()

            # 重置已卖出股票的最高收益率
            removed_stocks = set(self.max_profit.keys()) - set(self.positions.keys())
            for stock_code in removed_stocks:
                self.max_profit.pop(stock_code, None)
                logger.info(f"重置 {stock_code} 的最高收益率")

            # 保存更新后的 max_profit 字典到文件
            with open(MAX_PROFIT_PATH, 'wb') as f:
                pickle.dump(self.max_profit, f)
            logger.debug(f"已保存 max_profit 至 {MAX_PROFIT_PATH}")
        except Exception as e:
            logger.exception(f"保存 max_profit 时发生异常: {e}")

    def load_max_profit(self):
        """
        从文件加载 max_profit 字典
        """
        try:
            if Path(MAX_PROFIT_PATH).exists():
                with open(MAX_PROFIT_PATH, 'rb') as f:
                    self.max_profit = pickle.load(f)
                logger.debug(f"从 {MAX_PROFIT_PATH} 成功加载 max_profit{self.max_profit}")
            else:
                self.max_profit = {}
                logger.warning(f"未找到 max_profit 文件 {MAX_PROFIT_PATH}，将使用空字典")
        except Exception as e:
            logger.exception(f"加载 max_profit 时发生异常: {e}")
            self.max_profit = {}

    def update_positions(self):
        """
        更新持仓信息，只保留可用数量大于0的持仓。
        """
        try:
            positions = self.xt_trader.query_stock_positions(ACC)
            self.positions = {pos.stock_code: pos for pos in positions if pos.can_use_volume > 0}
            logger.info(f"已更新持仓信息: {[pos.stock_code for pos in self.positions.values()]}")
        except Exception as e:
            logger.exception(f"更新持仓信息时发生异常: {e}")

    def sell_stock(self, stock_code, quantity, price=0, strategy_name='', order_remark=''):
        """
        卖出股票函数，根据股票代码后缀确定所属市场并设置 order_type 后，异步发出卖出指令。
        """
        try:
            order_type = self.get_order_type(stock_code)
            response = self.xt_trader.order_stock_async(
                ACC, stock_code, xtconstant.STOCK_SELL, quantity,
                order_type, price, strategy_name, order_remark
            )
            logger.info(f'卖出股票【{stock_code}】，数量【{quantity}】，返回值为【{response}】')
            # 更新持仓信息
            self.update_positions()
        except Exception as e:
            logger.exception(f"卖出股票时发生异常: {e}")

    @staticmethod
    def get_order_type(stock_code):
        """
        根据股票代码获取对应的 order_type
        """
        if stock_code.endswith('.SH'):
            return xtconstant.MARKET_SH_CONVERT_5_CANCEL
        elif stock_code.endswith('.SZ'):
            return xtconstant.MARKET_SZ_CONVERT_5_CANCEL
        else:
            return 0  # 默认使用限价单

    def stop_loss_max_profit(self, datas):
        """
        止盈止损逻辑：
        1. 当收益率达到止损阈值（-1%）时，立即卖出止损。
        2. 当收益率超过止盈阈值（如3%）后，若从最高收益率回撤超过设定值（如30%），则卖出止盈。
        """
        stocks_set = set(self.positions.keys()) & set(datas.keys())
        for stock_code in stocks_set:
            position = self.positions[stock_code]
            volume = position.volume
            avg_price = position.avg_price
            last_price = datas[stock_code]['lastPrice']

            if avg_price != 0:

                current_profit = (last_price - avg_price) / avg_price

                # 检查是否达到止损阈值
                if current_profit <= self.stop_loss_threshold:
                    self.sell_stock(stock_code, volume, 0, "止损策略", f"收益率{current_profit:.2%}")
                    logger.warning(f"卖出 {stock_code}，当前收益率 {current_profit:.2%}，达到止损阈值")
                    continue  # 已经卖出，不再检查止盈策略
                # 初始化或更新最高收益率
                # self.load_max_profit()
                max_profit_value = self.max_profit.get(stock_code, -999999)
                # logger.info(f"{current_profit} + {max_profit_value}")
                if current_profit >= max_profit_value:
                    self.max_profit[stock_code] = current_profit
                    self.save_max_profit()
                    logger.info(f"更新 {stock_code} 的最高收益率为 {self.max_profit[stock_code]:.2%}")
                # 当当前收益率超过止盈阈值后，开始监控回撤

                drawdown = (self.max_profit[stock_code] - current_profit) / self.max_profit[stock_code]

                if self.max_profit[stock_code] >= self.profit_threshold:
                    logger.debug(f"{stock_code} 当前盈利为{current_profit:.2%}；回撤幅度为 {drawdown:.2%}; 最大盈利为{max_profit_value:.2%}")

                    if drawdown >= self.drawdown_threshold:
                        self.sell_stock(stock_code, volume, 0, "止盈策略", f"R{current_profit:.2%}_D{drawdown:.2%}")
                        logger.warning(
                            f"卖出 {stock_code}，当前收益率 {current_profit:.2%}，"  
                            f"最高收益率 {self.max_profit[stock_code]:.2%}"
                        )
                        continue
                    continue
                elif self.max_profit[stock_code] >= 0.007:
                    logger.debug(f"{stock_code} 当前盈利为{current_profit:.2%}；回撤幅度为 {drawdown:.2%}; 最大盈利为{max_profit_value:.2%}")

                    if current_profit <= 0.0035:
                        self.sell_stock(stock_code, volume, 0, "止盈策略", f"R{current_profit:.2%}_D{drawdown:.2%}")
                        logger.warning(
                            f"卖出 {stock_code}，当前收益率 {current_profit:.2%}，"
                            f"最高收益率 {self.max_profit[stock_code]:.2%}"
                        )
                        continue
                    continue
                elif self.max_profit[stock_code] >= 0.005:
                    logger.debug(f"{stock_code} 当前盈利为{current_profit:.2%}；回撤幅度为 {drawdown:.2%}; 最大盈利为{max_profit_value:.2%}")

                    if current_profit <= 0.0005:
                        self.sell_stock(stock_code, volume, 0, "止盈策略", f"R{current_profit:.2%}_D{drawdown:.2%}")
                        logger.warning(
                            f"卖出 {stock_code}，当前收益率 {current_profit:.2%}，"
                            f"最高收益率 {self.max_profit[stock_code]:.2%}"
                        )
                        continue
                    continue
                else:
                    logger.debug(
                        f"{stock_code} 当前收益率 {current_profit:.2%}，未达到止盈监控阈值"
                    )
            else:
                logger.warning(f"{stock_code} 的平均买入价格为 {avg_price}")

    def call_back_functions(self, data, last_update_time):
        """
        数据回调函数，每次数据更新时调用，用于执行止损和止盈逻辑。
        """

        if not is_trading_day():
            logger.info("今天不是交易日。")
            return

        if not is_china_market_open():
            logger.info("当前不在交易时间内。")
            return

        current_time = time.time()
        if current_time - last_update_time.value >= 600:
            logger.info("开始更新持仓信息和订单状态")
            try:
                self.load_max_profit()
                self.update_positions()
                logger.info(f"已更新持仓信息: {[pos.stock_code for pos in self.positions.values()]}")

                # 撤销未完全成交的挂单
                pending_orders = self.xt_trader.query_stock_orders(ACC)
                for order in pending_orders:
                    if order.order_status == xtconstant.ORDER_PART_SUCC:
                        cancel_response = self.xt_trader.cancel_order_stock_async(ACC, order.order_id)
                        logger.info(f"撤销订单 {order.order_id}，响应: {cancel_response}")

                self.save_max_profit()
            except Exception as e:
                logger.exception(f"更新持仓信息和订单状态时发生异常: {e}")
            finally:
                last_update_time.value = time.time()

        self.stop_loss_max_profit(data)

    def start(self):
        if not is_trading_day():
            logger.info("今天不是交易日")
            return

        try:
            self.update_positions()
            manager = Manager()
            last_update_time = manager.Value('d', time.time())

            stock_list = get_investment_target()
            stock_list = list(set(stock_list) | set(self.positions.keys()))
            logger.info(f"订阅行情 {stock_list}")

            xtdata.subscribe_whole_quote(
                stock_list,
                callback=lambda data: self.call_back_functions(data, last_update_time)
            )
            logger.info("止损程序启动")
            xtdata.run()

            connect_result = self.xt_trader.connect()
            logger.info(f'建立交易连接，返回值：{connect_result}')
            subscribe_result = self.xt_trader.subscribe(ACC)
            logger.info(f'对交易回调进行订阅，返回值：{subscribe_result}')

            self.xt_trader.run_forever()
        except Exception as e:
            logger.exception(f"止损主程序运行时发生异常: {e}")

def stop_loss_main():
    program = StopLossProgram()
    while True:
        try:
            logger.info('启动 xtdata 订阅服务')
            program.start()
        except Exception as e:
            logger.warning("发生异常，重启 mini 迅投客户端")
            logger.exception(e)
            start_xt_client()
        finally:
            logger.info("等待 3 秒后重试")
            time.sleep(3)


if __name__ == '__main__':
    stop_loss_main()