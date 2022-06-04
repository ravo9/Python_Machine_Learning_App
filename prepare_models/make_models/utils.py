import pandas_datareader as web
import numpy as np


def get_real_average_change_of_test_period(
    column,
    start_test_date,
    end_test_date,
    days_into_account,
    instrument):

    # 1. Get values.
    # Get the data
    df_test = web.DataReader(instrument, data_source='yahoo', start=start_test_date, end=end_test_date)
    # Create a new dataframe with only our column
    data_test = df_test.filter([column])
    # Convert the dataframe to Numpy array
    dataset_test = data_test.values

    # 2. Remove first n days from the beginning.
    dataset_test = dataset_test[days_into_account:]

    # 3. Get their daily abs changes.
    daily_changes = []
    for i in range(0, len(dataset_test)):
        if i != 0:
            daily_changes.append(abs(dataset_test[i] - dataset_test[i-1]))

    # 4. Get and print the average of daily abs changes.
    average = np.average(daily_changes)
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


def get_average_error_direction_prediction(predictions, y_test):
    # - * - = +  -> GOOD
    # - * + = -  -> BAD
    # + * + = +  -> GOOD

    changes_predicted = []
    changes_real = []

    for i in range(0, len(predictions)):
        if i != 0:
            change_predicted = predictions[i] - predictions[i - 1]
            change_real = y_test[i] - y_test[i - 1]
            changes_predicted.append(change_predicted)
            changes_real.append(change_real)

    multiplied_array = np.asarray(changes_predicted) * np.asarray(changes_real)
    all_values_amount = len(multiplied_array)
    well_predicted_values = 0
    for value in multiplied_array:
        if value > 0:
            well_predicted_values += 1

    return well_predicted_values/all_values_amount


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
    return rmse


def print_currently_making_model_info(index, all_models_amount):
    print("Making model number " + str(index) + " out of " + str(all_models_amount))


def print_model_creation_interrupted_error(error, waiting_time):
    print("Error: Model creation interrupted. Trying again after " + str(waiting_time) + " seconds...")
    print(error)
