# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import fileinput
import sys
from pathlib import Path
import datetime
import os

def get_real_average_change_of_test_period(
    column,
    start_test_date,
    end_test_date,
    days_into_account,
    instrument):

    # 1. Get values.
    # Get the data
    dfTest = web.DataReader(instrument, data_source='yahoo', start=start_test_date, end=end_test_date)
    # Create a new dataframe with only our column
    data_test = dfTest.filter([column])
    # Convert the dataframe to Numpy array
    datasetTest = data_test.values

    # 2. Remove first n days from the beginning.
    datasetTest = datasetTest[days_into_account:]

    # 3. Get their daily abs changes.
    dailyChanges = []
    for i in range(0, len(datasetTest)):
        if i != 0:
            dailyChanges.append(abs(datasetTest[i] - datasetTest[i-1]))

    # 4. Get and print the average of daily abs changes.
    average = np.average(dailyChanges)
    print("")
    print("------------------")
    print("REAL AVERAGE DAILY CHANGE:")
    print(average)
    print("------------------")
    print("")

    # ToDo: Multiple columns!

def save_model(model, output_dir, average, average_required_for_model_to_be_saved = -1):
    if average_required_for_model_to_be_saved > 0:
        if average < average_required_for_model_to_be_saved:
            model.save(output_dir + str(average))
    else:
        model.save(output_dir + str(average))

def get_average_error(predictions, y_test):
    return np.average(np.abs(predictions - y_test))

def print_average_error(average):
    # 0 means perfect prediction
    print("")
    print("------------------")
    print("AVERAGE ERROR:")
    print(average)
    print("------------------")
    print("")

def print_rmse(predictions, y_test):
    # 0 means perfect prediction
    rmse = np.sqrt( np.mean(predictions - y_test)**2 )
    print("")
    print("------------------")
    print("RMSE:")
    print(rmse)
    print("------------------")
    print("")

def check_if_txt_log_exists(output_dir):
    logFilePath = Path(output_dir + 'log.txt')
    if logFilePath.is_file() == False:
        print("")
        print("------------------")
        print("NO TXT LOG CREATING...")
        os.system("mkdir " + output_dir)
        os.system("touch " + str(logFilePath))
        print("TXT LOG CREATED!")
        print("------------------")
        print("")

def check_if_csv_log_exists(output_dir):
    logCsvFilePath = Path(output_dir + 'log.csv')
    if logCsvFilePath.is_file() == False:
        print("")
        print("------------------")
        print("NO CSV LOG CREATING...")
        os.system("mkdir " + output_dir)
        os.system("touch " + str(logCsvFilePath))
        print("CSV LOG CREATED!")
        print("------------------")
        print("")


#
# CSV and TXT logs methods
#

def write_opening_separator_into_txt_log(output_dir):
    with open((output_dir + 'log.txt'),'a') as f:
        f.write("\n")
        f.write("-------------------------\n")

def write_closing_separator_into_txt_log(output_dir):
    with open((output_dir + 'log.txt'),'a') as f:
        f.write("-------------------------")

def write_average_into_txt_log(output_dir, average_error):
    with open((output_dir + 'log.txt'),'a') as f:
        f.write(str(average_error) + '\n')

def write_model_creation_details_into_csv_log(
    modelConcept,
    columns,
    start_train_date,
    end_train_date,
    start_test_date,
    end_test_date,
    instrument,
    output_dir,
    optimizer_type,
    loss_function_type,
    days_into_account,
    epochs_amount,
    average_error,
    random_seed,
    optimizer_learning_rate
    ):
    with open((output_dir + 'log.csv'),'a') as f:

        # Datetime stamp
        f.write(str(datetime.datetime.now()) + ', ')

        # Data Parameters
        f.write(str(instrument) + ', ')
            # Many columns?
        f.write(str(columns) + ', ')
        f.write(str(start_train_date) + ', ')
        f.write(str(end_train_date) + ', ')
        f.write(str(start_test_date) + ', ')
        f.write(str(end_test_date) + ', ')

        # Model Parameters
        f.write(str(modelConcept) + ', ')
        f.write(str(optimizer_type) + ', ')
        f.write(str(loss_function_type) + ', ')
        f.write(str(epochs_amount) + ', ')
        f.write(str(days_into_account) + ', ')

        # Result
        f.write(str(average_error) + ', ')

        # Further Model Parameters
        f.write(str(random_seed) + ', ')
        f.write(str(optimizer_learning_rate) + '\n')

#
# Unused code storage
#

# np.savetxt("predictions.csv", predictions, delimiter=",")
# np.savetxt("real_values.csv", y_test, delimiter=",")

# with open('models/' + instrument + '/' + column0 + '/best.txt','r') as f:
#     output = f.read()
#     if average < float(output):
#         with open('models/' + instrument + '/' + column0 + '/best.txt','w') as f:
#             f.write(str(average))
#             bestMatch = float(output)
