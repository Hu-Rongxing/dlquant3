from data_processing.sync_stock_data import sync_stock_data_main

def test_sync_stock_data():
    sync_stock_data_main(force_full_sync=True)