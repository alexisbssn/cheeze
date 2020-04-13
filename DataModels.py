class StockData:
    def __init__(self, ohlcv_histories, technical_indicators, next_day_open_values, symbols_count):
        self.ohlcv_histories = ohlcv_histories
        self.technical_indicators = technical_indicators
        self.next_day_open_values = next_day_open_values
        self.symbols_count = symbols_count

class TransformedData:
    def __init__(self, stock_data: StockData, y_normalisers, symbol_encoder):
        self.stock_data = stock_data
        self.y_normalisers = y_normalisers
        self.symbol_encoder = symbol_encoder

class ModelTestingData(TransformedData):
    def __init__(self, model, train_data: StockData, test_data: StockData, y_normalisers):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.y_normalisers = y_normalisers