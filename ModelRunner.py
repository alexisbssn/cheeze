import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from sklearn.metrics import classification_report
from DataTransformer import csv_to_dataset

# AI Portfolio Manager
class ModelRunner:

    def __init__(self):
        # Data in to test that the saving of weights worked
        ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('BA_daily.csv')
        
        # Read structure from json
        json_file = open('model.json', 'r')
        json = json_file.read()
        json_file.close()
        self.network = model_from_json(json)

        # Read weights from HDF5
        self.network.load_weights("weights.h5")

        # Verify weights and structure are loaded
        y_pred = self.network.predict(X.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y, y_pred))

# ModelRunner()