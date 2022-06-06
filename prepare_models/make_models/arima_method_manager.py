import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from utils import get_average_error_direction_prediction

def run_arima_method():

    sp500_data = yf.download('^GSPC', start="1980-01-01", end="2021-11-21")
    sp500_data = sp500_data[['Close']]

    # difs = (sp500_data.shift() - sp500_data) / sp500_data
    # difs = difs.dropna()
    # y = difs.Close.values

    sp500_data.dropna()
    y = sp500_data.Close.values

    best_params = None
    best_result = 0.0

    param_list = [(x, y, z) for x in range(5) for y in range(5) for z in range(5)]
    # param_list = [(0, 1, 3)]

    order_index = 0

    for order in param_list:

        order_index = order_index + 1
        print("ORDER NUMBER: ")
        print(order_index)

        mses = []
        all_predictions = []
        all_test_values = []

        tscv = TimeSeriesSplit(n_splits=100,
                               max_train_size = 3*31,
                               test_size=1)

        for train_index, test_index in tscv.split(y):

            try:
                train = y[train_index]
                test = y[test_index]

                # for each ts split do a model
                mod = sm.tsa.ARIMA(train, order=order)
                res = mod.fit()
                pred = res.forecast(1)[0]

                all_test_values.append(test)
                all_predictions.append(pred)

                pred = [pred]
                mse = mean_squared_error(test, pred)
                mses.append(mse)

            except Exception as e:
                # ignore models that error

                # print("ERROR 1")
                # print(e)
                pass

        try:
            average_mse = np.mean(mses)
            std_mse = np.std(mses)
            direction_prediction_result = get_average_error_direction_prediction(all_predictions, all_test_values)

            if (direction_prediction_result > best_result):
                best_result = direction_prediction_result
                best_params = order

            print("average_mse: ")
            print(average_mse)

            print("std_mse: ")
            print(std_mse)

            print("direction_prediction_result: ")
            print(direction_prediction_result)

        except:
            print("ERROR 2")

    print("BEST ORDER: ")
    print(best_params)
    print("BEST direction_prediction_result: ")
    print(best_result)
