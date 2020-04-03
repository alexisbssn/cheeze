class StockData:
    def __init__(self, ohlcv_histories, technical_indicators, next_day_open_values):
        self.ohlcv_histories = ohlcv_histories
        self.technical_indicators = technical_indicators
        self.next_day_open_values = next_day_open_values

class TransformedData:
    def __init__(self, stock_data: StockData, y_normaliser):
        self.stock_data = stock_data
        self.y_normaliser = y_normaliser

class ModelTestingData(TransformedData):
    def __init__(self, model, train_data: StockData, test_data: StockData, y_normaliser):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.y_normaliser = y_normaliser