from data_processing.source_stock_data import download_history_data, get_data_from_local
if __name__ == '__main__':
    # download_history_data()
    df = get_data_from_local()
    print(df)