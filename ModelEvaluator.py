import numpy as np
from DataModels import *

class ModelEvaluator:

    def predict(self, model, data: StockData, y_normaliser):
        y_predicted_norm = model.predict([data.ohlcv_histories, data.technical_indicators])
        #y_predicted_norm = model.predict([data.ohlcv_histories])
        y_predicted = y_normaliser.inverse_transform(y_predicted_norm)
        return y_predicted

    def plot(self, dataset: ModelTestingData):
        y_test_predicted = self.predict(dataset.model, dataset.test_data, dataset.y_normalisers[0])
        y_predicted = self.predict(dataset.model, dataset.train_data, dataset.y_normalisers[0])

        unscaled_y_test = dataset.y_normalisers[0].inverse_transform(dataset.test_data.next_day_open_values.reshape(-1,1))
        unscaled_y_train = dataset.y_normalisers[0].inverse_transform(dataset.train_data.next_day_open_values.reshape(-1,1))

        assert unscaled_y_test.shape == y_test_predicted.shape
        real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
        scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
        print("Scaled mean square error (MSE):" + str(scaled_mse))

        import matplotlib.pyplot as plt

        plt.gcf().set_size_inches(22, 15, forward=True)

        real = plt.plot(unscaled_y_test[:-1], label='real')
        pred = plt.plot(y_test_predicted[:-1], label='predicted')

        # real = plt.plot(unscaled_y_train[:-1], label='real')
        # pred = plt.plot(y_predicted[:-1], label='predicted')

        plt.legend(['Real', 'Predicted'])
        plt.show()

    def analyze(self, dataset: ModelTestingData):
        evaluation = dataset.model.evaluate(
            [dataset.test_data.ohlcv_histories, dataset.test_data.technical_indicators],
            #[dataset.test_data.ohlcv_histories],
            dataset.test_data.next_day_open_values)
        print("evaluation: " + str(evaluation))

        y_test_predicted = self.predict(dataset.model, dataset.test_data, dataset.y_normalisers[0])

        unscaled_y_test = dataset.y_normalisers[0].inverse_transform(dataset.test_data.next_day_open_values.reshape(-1,1))

        assert unscaled_y_test.shape == y_test_predicted.shape
        real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
        scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
        print("scaled mean square error (test): " + str(scaled_mse))

        # also getting predictions for the entire dataset, just to see how it performs
        y_predicted = self.predict(dataset.model, dataset.train_data, dataset.y_normalisers[0])

        unscaled_y_train = dataset.y_normalisers[0].inverse_transform(dataset.train_data.next_day_open_values.reshape(-1,1))

        assert unscaled_y_train.shape == y_predicted.shape
        real_mse = np.mean(np.square(unscaled_y_train - y_predicted))
        scaled_mse = real_mse / (np.max(unscaled_y_train) - np.min(unscaled_y_train)) * 100
        print("scaled mean square error (training dataset): " + str(scaled_mse))