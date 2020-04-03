
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from DataTransformer import csv_to_dataset
import numpy as np
from DataModels import *
# this makes the random number generation predictable
# https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(4)

class ModelBuilder:

    def build_lstm_branch(self, histories):
        lstm_input = Input(shape=(histories.shape[1], histories.shape[2]), name='lstm_input')
        x = LSTM(histories.shape[1], name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        x = Dense(120, name='lstm_dense_0')(x)
        x = Dense(120, name='lstm_dense_1')(x)
        lstm_branch = Model(inputs=lstm_input, outputs=x)
        return lstm_branch

    def build_technical_indicators_branch(self, technical_indicators):
        dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
        x = Dense(120, name='tech_dense_0')(dense_input)
        x = Dropout(0.25, name='tech_dropout_0')(x)
        x = Dense(90, activation="relu", name='tech_dense_1')(x)
        x = Dropout(0.2, name='tech_dropout_1')(x)
        technical_indicators_branch = Model(inputs=dense_input, outputs=x)
        return technical_indicators_branch

    def single_branch(self, branch):
        dense_output = Dense(1, name='dense_out')(branch.output)
        model = Model(inputs=branch.input, outputs=dense_output)
        return model

    def combine_branches(self, branches):
        combined = concatenate([branch.output for branch in branches], name='concatenate')
        x = Dense(64, name='pooling_dense_0')(combined)
        x = Dropout(0.2, name='pooling_dropout_0')(x)
        x = Dense(64, name='pooling_dense_1')(x)
        dense_output = Dense(1, name='dense_out')(x)
        model = Model(inputs=[branch.input for branch in branches], outputs=dense_output)
        return model

    def split_test_data(self, dataset: TransformedData, test_split: float):
        n = int(dataset.stock_data.ohlcv_histories.shape[0] * test_split) # ohlcv_histories.shape looks like (1168, 50, 5)

        ohlcv_train = dataset.stock_data.ohlcv_histories[:n]
        tech_ind_train = dataset.stock_data.technical_indicators[:n]
        next_day_train = dataset.stock_data.next_day_open_values[:n]
        train_dataset = StockData(ohlcv_train, tech_ind_train, next_day_train)

        ohlcv_test = dataset.stock_data.ohlcv_histories[n:]
        tech_ind_test = dataset.stock_data.technical_indicators[n:]
        next_day_test = dataset.stock_data.next_day_open_values[n:]
        test_dataset = StockData(ohlcv_test, tech_ind_test, next_day_test)

        return ModelTestingData(None, train_dataset, test_dataset, dataset.y_normaliser)

        


    def build_model(self, dataset, epochs: int):
        split_data = self.split_test_data(dataset, 0.7)
                
        lstm_branch = self.build_lstm_branch(dataset.stock_data.ohlcv_histories)
        technical_indicators_branch = self.build_technical_indicators_branch(dataset.stock_data.technical_indicators)
        model = self.combine_branches([lstm_branch, technical_indicators_branch])
        #model = self.single_branch(lstm_branch)

        adam = optimizers.Adam(lr=0.0005)
        model.compile(optimizer=adam, loss='mse')

        from keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)

        model.fit(
            #x=[split_data.train_data.ohlcv_histories],
            x=[split_data.train_data.ohlcv_histories, split_data.train_data.technical_indicators], 
            y=split_data.train_data.next_day_open_values, 
            batch_size=32,
            epochs=epochs,
            shuffle=True)

        split_data.model = model
        return split_data


        