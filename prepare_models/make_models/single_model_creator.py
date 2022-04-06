import tensorflow as tf
import pandas_datareader as web
import pandas as pd
import numpy as np
import random
import datetime
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from utils import get_average_error, get_average_error_direction_prediction, print_average_error, print_rmse, save_model
from utils_csv_and_txt import write_result_into_txt_log, write_model_creation_details_into_csv_log

DATE_FORMAT = "%Y-%m-%d"

def make_multiple_variable_model(
    columns,
    start_train_date,
    start_test_date,
    end_test_date,
    instrument,
    optimizer_type,
    loss_function_type,
    days_into_account,
    epochs_amount,
    batch_size,
    random_seed,
    optimizer_learning_rate,
    output_dir,
    average_required_for_model_to_be_saved,
    layer_1_neurones_number = 50,
    layer_2_neurones_number = 50,
    layer_3_neurones_number = 50
    ):

    # Introduce the model
    modelConcept = 'multiple_variable_model'


    # SECTION 1: PREPARE DATA

    # Get the data
    fetched_dataset = web.DataReader(instrument, data_source='yahoo', start=start_train_date, end=end_test_date)

    # Filter columns that are interesting for us
    fetched_dataset = fetched_dataset.filter(columns)

    # We extract only values (withoud index column) as 2D array here, as 2D is required by the scaler
    fetched_dataset_values = fetched_dataset.values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(fetched_dataset_values)

    # We can transofrm the 2D array into 1D as 2D is not necessary anymore
    scaled_data = scaled_data.flatten()

    # Split the data into datasets
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(days_into_account, len(fetched_dataset)):
        if fetched_dataset.index[i-days_into_account] < pd.to_datetime(start_test_date).date():
            x_train.append(scaled_data[i-days_into_account:i])
            y_train.append(scaled_data[i])
        else:
            x_test.append(scaled_data[i-days_into_account:i])
            y_test.append(scaled_data[i])

    # Convert the X datasets into Numpy arrays (to simplify array in array)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Reshape the data for LSTM and for "predict" method
    numberOfColumns = len(columns)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], numberOfColumns))
    x_test = np.reshape (x_test, (x_test.shape[0], x_test.shape[1], numberOfColumns))


    # SECTION 2: MAKE MODEL AND TRAIN IT

    # Set the seed
    tf.random.set_seed(random_seed)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(layer_1_neurones_number, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(layer_2_neurones_number, return_sequences=False))
    model.add(Dense(layer_3_neurones_number))
    model.add(Dense(1))

    # units = 128
    # model.add(LSTM(units, input_shape=(x_train.shape[1], 1)))
    # model.add(LeakyReLU(alpha=0.5))
    # model.add(Dropout(0.1))

    # Compile the model
    optimizer = Adam(learning_rate=optimizer_learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function_type)

    # # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_amount)


    # SECTION 3: TEST THE MODEL AND PRINT OUT RESULTS

    # Get the models predicted price values
    predictions_2D = model.predict(x_test)

    # Transform y_test 1D array into 2D, as scaler requires that
    y_test_2D = y_test.reshape(-1, 1)

    # Reverse data scalling
    predictions_rescaled_2D = scaler.inverse_transform(predictions_2D)
    y_test_rescaled_2D = scaler.inverse_transform(y_test_2D)

    # We can transofrm the 2D arrays into 1D as 2D is not necessary anymore
    predictions_rescaled = predictions_rescaled_2D.flatten()
    y_test_rescaled = y_test_rescaled_2D.flatten()

    average_error = get_average_error(predictions_rescaled, y_test_rescaled)
    print_average_error(average_error)
    print_rmse(predictions_rescaled, y_test_rescaled)

    direction_prediction_result = get_average_error_direction_prediction(predictions_rescaled, y_test_rescaled)
    # write_result_into_txt_log(output_dir, direction_prediction_result)
    print("Direction predicting results: well predicted values percentage:")
    print(direction_prediction_result)


    # SECTION 4: SAVE RESULTS

    # save_model(model, output_dir, average_error)

    # write_model_creation_details_into_csv_log(
    #     modelConcept,
    #     columns,
    #     start_train_date,
    #     end_train_date,
    #     start_test_date,
    #     end_test_date,
    #     instrument,
    #     output_dir,
    #     optimizer_type,
    #     loss_function_type,
    #     days_into_account,
    #     epochs_amount,
    #     average_error,
    #     random_seed,
    #     optimizer_learning_rate)

    score = direction_prediction_result
    return score
