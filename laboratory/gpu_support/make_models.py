
# GPU support stuff
import os
import plaidml.keras
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# GPU support stuff
import keras
from keras import layers

from keras.optimizers import Adam
import fileinput
import sys
from utils import check_if_txt_log_exists, check_if_csv_log_exists, write_opening_separator_into_txt_log, write_closing_separator_into_txt_log, get_real_average_change_of_test_period
from model_creator import makeMultipleVariableModel
import time

# GPU support stuff
import keras.backend as K
K

# Data Parameters
columns = ['Close']
start_train_date='2015-09-15'
end_train_date='2020-07-24'
start_test_date='2020-06-27'
end_test_date='2020-11-25'
instrument = 'AAPL'

# Files System Parameters
output_dir = 'many_models_creation_output/'
average_required_for_model_to_be_saved = 26000.0

# Model Parameters
optimizer_type = 'adam'
# loss_function_type = 'mean_squared_error'
loss_function_type = 'mean_absolute_error'
days_into_account = [15, 30, 45]
epochs_amount = [6, 8]
random_seed = [3131, 23213]
optimizer_learning_rate = [0.001]

# Multiple Build Params
amount_of_models_per_one_setting = 1
sleep_the_computer_when_the_work_is_done = False


# Main function

check_if_csv_log_exists(output_dir)
check_if_txt_log_exists(output_dir)

write_opening_separator_into_txt_log(output_dir)

for days in days_into_account:

    # get_real_average_change_of_test_period(
    #     column,
    #     start_test_date,
    #     end_test_date,
    #     days,
    #     instrument)

    for epochs in epochs_amount:
        for learningRate in optimizer_learning_rate:
            for seed in random_seed:
                for i in range(amount_of_models_per_one_setting):
                    print("Making model number " + str(i+1) + " out of " + str(amount_of_models_per_one_setting))
                    is_making_model_finished_successfully = False

                    while (is_making_model_finished_successfully == False):
                        try:
                            makeMultipleVariableModel(
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
                                days,
                                epochs,
                                seed,
                                learningRate)
                            is_making_model_finished_successfully = True
                        except Exception as e:
                            print("Error: Model creation interrupted. Trying again after 15 seconds...")
                            print(e)
                            time.sleep(15)

write_closing_separator_into_txt_log(output_dir)

# Run the whole script as sudo to let it work.
if (sleep_the_computer_when_the_work_is_done):
    os.system('shutdown -s now')
