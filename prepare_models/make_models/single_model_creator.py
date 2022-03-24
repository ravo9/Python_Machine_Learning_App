import tensorflow as tf
import pandas_datareader as web
import numpy as np
import random
from numpy import concatenate
# from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from utils import get_average_error, get_average_error_direction_prediction, print_average_error, print_rmse, save_model
from utils_csv_and_txt import write_result_into_txt_log, write_model_creation_details_into_csv_log


def make_multiple_variable_model(
    config_data_parameters,
    config_machine_learning_parameters,
    config_multiple_models_creation_process_parameters
    ):

    # Unpack config_data_parameters
    columns = config_data_parameters["columns"]
    start_train_date = config_data_parameters["start_train_date"]
    end_train_date = config_data_parameters["end_train_date"]
    start_test_date = config_data_parameters["start_test_date"]
    end_test_date = config_data_parameters["end_test_date"]
    instrument = config_data_parameters["instrument"]

    # Unpack config_machine_learning_parameters
    optimizer_type = config_machine_learning_parameters["optimizer_type"]
    loss_function_type = config_machine_learning_parameters["loss_function_type"]
    days_into_account = config_machine_learning_parameters["days_into_account"]
    epochs_amount = config_machine_learning_parameters["epochs_amount"]
    random_seed = config_machine_learning_parameters["random_seed"]
    optimizer_learning_rate = config_machine_learning_parameters["optimizer_learning_rate"]

    # Unpack config_multiple_models_creation_process_parameters
    output_dir = config_multiple_models_creation_process_parameters["output_dir"]
    average_required_for_model_to_be_saved = config_multiple_models_creation_process_parameters["average_required_for_model_to_be_saved"]

    # Introduce the model
    modelConcept = 'multiple_variable_model'

    # Get the data
    dfTrain = web.DataReader(instrument, data_source='yahoo', start=start_train_date, end=end_train_date)
    df_test = web.DataReader(instrument, data_source='yahoo', start=start_test_date, end=end_test_date)

    # Create a new dataframe with only our column (and convert it to Numpy array)
    datasetTrain = dfTrain.filter(columns).values
    dataset_test = df_test.filter(columns).values

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
    tf.random.set_seed(random_seed)

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
    optimizer = Adam(learning_rate=optimizer_learning_rate)
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

    direction_prediction_result = get_average_error_direction_prediction(predictions[:,0], y_test[:,0])
    write_result_into_txt_log(output_dir, direction_prediction_result)
    
    print("Direction predicting results: well predicted values percentage:")
    print(direction_prediction_result)

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