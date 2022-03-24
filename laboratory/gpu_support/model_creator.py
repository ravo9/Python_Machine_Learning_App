# -*- coding: utf-8 -*-

# GPU support stuff
import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# GPU support stuff
import keras
from keras import layers

import math
import pandas_datareader as web
import numpy as np
from numpy import concatenate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, LeakyReLU
from keras.optimizers import Adam
import random
from utils import get_average_error, print_average_error, print_rmse, save_model, write_result_into_txt_log, write_model_creation_details_into_csv_log

# GPU support stuff
import keras
import keras.backend as K
K

def makeMultipleVariableModel(
    columns,
    start_train_date,
    end_train_date,
    start_test_date,
    end_test_date,
    instrument,
    output_dir,
    optimizer_type,
    loss_function_type,
    average_required_for_model_to_be_saved,
    days_into_account,
    epochs_amount,
    random_seed,
    optimizer_learning_rate
    ):

    # Introduce the model
    modelConcept = 'multiple_variable_model'
    # modelConcept = 'multiple_variable_model_lay[0]_and_lay[1]_30neurons_lay[3]_extraDense25'

    # Get the data
    dfTrain = web.DataReader(instrument, data_source='yahoo', start=start_train_date, end=end_train_date)
    df_test = web.DataReader(instrument, data_source='yahoo', start=start_test_date, end=end_test_date)

    # Create a new dataframe with only our column
    dataTrain = dfTrain.filter(columns)
    data_test = df_test.filter(columns)

    # Convert the dataframe to Numpy array
    datasetTrain = dataTrain.values
    dataset_test = data_test.values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_data = scaler.fit_transform(datasetTrain)
    scaled_test_data = scaler.fit_transform(dataset_test)

    # Split the data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(days_into_account, len(scaled_train_data)):
      x_train.append(scaled_train_data[i-days_into_account:i])
      y_train.append(scaled_train_data[i, 0])

    # Convert the x_train and y_train into Numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data for LSTM
    numberOfColumns = len(columns)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], numberOfColumns))

    # Set the seed
    os.environ['PYTHONHASHSEED']=str(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # tf.random.set_seed(random_seed)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # model = Sequential()
    # model.add(Dense(90))
    # model.add(LSTM(50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    # model.add(Dense(1))

    # units = 128
    # learningRate = 0.01
    # optimizer = Adam(lr=learningRate)
    # model = Sequential()
    # model.add(LSTM(units, input_shape=(x_train.shape[1], 1)))
    # model.add(LeakyReLU(alpha=0.5))
    # model.add(Dropout(0.1))
    # model.add(Dense(1))

    # Compile the model
    # GPU support stuff
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=loss_function_type)

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=epochs_amount)

    # Create the data sets x_test and y_test
    x_test = []
    y_test = scaled_test_data[days_into_account:, :]

    for i in range(days_into_account, len(scaled_test_data)):
        x_test.append(scaled_test_data[i-days_into_account:i])

    # Convert the data into Numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape (x_test, (x_test.shape[0], x_test.shape[1], numberOfColumns))

    # Get the models predicted price values
    predictions = model.predict(x_test)

    # predictions = concatenate((predictions, x_test[:, :, 1:]), axis=2)
    predictions = concatenate((predictions, y_test[:, 1:]), axis=1)
    predictions = scaler.inverse_transform(predictions)

    # Fix line
    y_test = dataset_test[days_into_account:, :]

    average_error = get_average_error(predictions[:,0], y_test[:,0])
    print_average_error(average_error)
    print_rmse(predictions, y_test)

    save_model(model, output_dir, average_error)

    write_model_creation_details_into_csv_log(
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
        optimizer_learning_rate)

    write_result_into_txt_log(output_dir, average_error)
