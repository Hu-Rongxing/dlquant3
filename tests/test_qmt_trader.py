import pytest
from qmt_client.qmt_trader import (
    setup_xt_trader, get_max_ask_price,
    buy_stock_async, generate_trading_report,
    cancel_all_order_async
)

@pytest.fixture
def xt_trader():
    return setup_xt_trader()

def test_setup_xt_trader(xt_trader):
    assert xt_trader is not None

def test_get_max_ask_price():
    stock_code = "000001.SZ"
    max_ask_price = get_max_ask_price(stock_code)
    print('\n')
    print(max_ask_price)
    assert max_ask_price is not None

def test_buy_stock_async(xt_trader):
    stock_code = "000001.SZ"
    result = buy_stock_async([stock_code])
    assert result is not None

def test_cancel_order(xt_trader):
    cancel_all_order_async()

def test_generate_trading_report(xt_trader):
    report = generate_trading_report()
    assert report is not None

def test_get_max_ask_price_invalid_stock_code():
    stock_code = "invalid_stock_code"
    max_ask_price = get_max_ask_price(stock_code)
    assert max_ask_price is None

def test_buy_stock_async_invalid_stock_code(xt_trader):
    stock_code = "invalid_stock_code"
    result = buy_stock_async([stock_code])
    assert result is True

def test_generate_trading_report_empty(xt_trader):
    report = generate_trading_report()
    assert report != ""

