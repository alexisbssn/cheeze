import pandas as pd
from sklearn import preprocessing
import numpy as np
import statistics
from DataModels import *

def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1) # no need for the date
    data = data.drop(0, axis=0) # drop the first data point because it's unreliable (IPO?)

    data = data.values

    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    history_points = 50 # TODO config
    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv = [] # open, high, low, close, volume
    open_values_next_day = []
    open_values_next_day_normalized = []
    for i in range(len(data_normalized) - history_points):
        ohlcv.append(data_normalized[i:i + history_points].copy())
        open_values_next_day.append(data[:,0][i + history_points].copy()) # 'open' is the 0th column
        open_values_next_day_normalized.append(data_normalized[:,0][i + history_points].copy()) # 'open' is the 0th column

    ohlcv_histories_normalized = np.array(ohlcv)
    next_day_open_values = np.expand_dims(np.array(open_values_next_day), -1)
    next_day_open_values_normalized = np.expand_dims(np.array(open_values_next_day_normalized), -1)

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    technical_indicators = get_technical_indicators(ohlcv_histories_normalized)
    
    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    # pylint: disable=E1136  # pylint/issues/3139
    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalised.shape[0]
    stock_data = StockData(ohlcv_histories_normalized, technical_indicators, next_day_open_values_normalized)
    return TransformedData(stock_data, y_normalizer)

def get_technical_indicators(histories):
    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in histories:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma: float = np.mean(his[:, 3])
        standard_deviation = statistics.stdev(his[:, 3])
        bollingerHigh = sma + 2*standard_deviation
        bollingerLow = sma - 2*standard_deviation
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        ohlcv = his[-1:][0]
        indicators = np.array([ohlcv[1], ohlcv[2], bollingerHigh, bollingerLow, sma, macd])
        technical_indicators.append(indicators)

    return np.array(technical_indicators)