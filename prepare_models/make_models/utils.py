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


def get_average_error_direction_prediction(predictions, y_test):
    # - * - = +  -> GOOD
    # - * + = -  -> BAD
    # + * + = +  -> GOOD

    changes_predictions = []
    changes_real = []

    for i in range(0, len(predictions)):
        if i != 0:
            change_predicted = predictions[i] - predictions[i - 1]
            change_real = y_test[i] - y_test[i - 1]
            changes_predictions.append(change_predicted)
            changes_real.append(change_real)

    multiplied_array = np.asarray(changes_predictions) * np.asarray(changes_real)
    all_values_amount = len(multiplied_array)
    well_predicted_values = 0
    for value in multiplied_array:
        if value > 0:
            well_predicted_values += 1

    print("TESTO 1")
    print(all_values_amount)
    print("TESTO 2")
    print(well_predicted_values)
    print("TESTO 3")
    print(well_predicted_values/all_values_amount)

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


def print_currently_making_model_info(index, all_models_amount):
    print("Making model number " + str(index+1) + " out of " + str(all_models_amount))


def print_model_creation_interrupted_error(error, waiting_time):
    print("Error: Model creation interrupted. Trying again after " + str(waiting_time) + " seconds...")
    print(error)
