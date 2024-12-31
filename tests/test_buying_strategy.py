from models.deploy_model import buying_strategy
from qmt_client.qmt_trader import cancel_all_order_async

def test_buying_strategy():
    buying_strategy()
    cancel_all_order_async()