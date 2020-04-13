import pandas as pd
from sklearn import preprocessing
import numpy as np
import statistics
from DataModels import *
import category_encoders as ce

def add_column(arr, index):
    indices = np.zeros((arr.shape[0], 1))
    indices[:,0] = index
    return np.append(arr, indices, axis=1)

def multi_csv_to_dataset(paths: [str]):
    index = 0
    y_normalizers = []
    for path in paths:
        index = index + 1
        transformed_data = csv_to_dataset(path)
        y_normalizers.append(transformed_data.y_normalisers[0])
        symbol_data = transformed_data.stock_data

        # add index column
        indices_ohlcv = np.zeros((symbol_data.ohlcv_histories.shape[0], symbol_data.ohlcv_histories.shape[1], 1))
        indices_ohlcv[:,:,0] = index
        symbol_data.ohlcv_histories = np.append(symbol_data.ohlcv_histories, indices_ohlcv, axis=2)

        indices_tech_ind = np.zeros((symbol_data.technical_indicators.shape[0], 1))
        indices_tech_ind[:,0] = index
        symbol_data.technical_indicators = np.append(symbol_data.technical_indicators, indices_tech_ind, axis=1)

        if index == 1:
            dataset = symbol_data
        else:
            dataset.technical_indicators = np.concatenate([dataset.technical_indicators, symbol_data.technical_indicators])
            dataset.ohlcv_histories = np.concatenate([dataset.ohlcv_histories, symbol_data.ohlcv_histories])
            dataset.next_day_open_values = np.concatenate([dataset.next_day_open_values, symbol_data.next_day_open_values])

    dataset.symbols_count = 5#len(paths)

    # swap the index column to one-hot multiple columns
    ohe = preprocessing.OneHotEncoder()
    ohe.fit(dataset.technical_indicators[:,-1].reshape(-1,1))

    ohlcv_index_cols = np.zeros((dataset.ohlcv_histories.shape[0], dataset.ohlcv_histories.shape[1], 5))
    ohlcv_index_cols[:,:,0] = 1

    #ohlcv_index_cols = ohe.transform(dataset.ohlcv_histories[:,:,-1].reshape(-1,1)).toarray()\
    #    .reshape(dataset.ohlcv_histories.shape[0], dataset.ohlcv_histories.shape[1], -1)

    dataset.ohlcv_histories = np.delete(dataset.ohlcv_histories, -1, axis=2)
    dataset.ohlcv_histories = np.append(dataset.ohlcv_histories, ohlcv_index_cols, axis=2)

    #tech_ind_index_cols = ohe.transform(dataset.technical_indicators[:,-1].reshape(-1,1)).toarray().reshape(dataset.technical_indicators.shape[0],-1)
    tech_ind_index_cols = np.zeros((dataset.technical_indicators.shape[0],5))
    tech_ind_index_cols[:,0] = 1
    dataset.technical_indicators = np.delete(dataset.technical_indicators, -1, axis=1)
    dataset.technical_indicators = np.append(dataset.technical_indicators, tech_ind_index_cols, axis=1)

    return TransformedData(dataset, y_normalizers, ohe)
        

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
    stock_data = StockData(ohlcv_histories_normalized, technical_indicators, next_day_open_values_normalized, 1)
    return TransformedData(stock_data, [y_normalizer], None)

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