import os
import time
from utils_csv_and_txt import write_opening_separator_into_txt_log, write_closing_separator_into_txt_log
from single_model_creator import make_multiple_variable_model
from utils import print_currently_making_model_info, print_model_creation_interrupted_error


def start_creation_of_machine_learning_models(
    config_data_parameters,
    config_machine_learning_parameters,
    config_multiple_models_creation_process_parameters):

    output_dir = config_multiple_models_creation_process_parameters["output_dir"]
    amount_of_models_per_one_setting = config_multiple_models_creation_process_parameters["amount_of_models_per_one_setting"]
    sleep_the_computer_when_the_work_is_done = config_multiple_models_creation_process_parameters["sleep_the_computer_when_the_work_is_done"]
    time_of_waiting_after_unsuccessful_model_creation_in_seconds = config_multiple_models_creation_process_parameters["time_of_waiting_after_unsuccessful_model_creation_in_seconds"]

    all_models_amount = len(config_data_parameters) * len(config_machine_learning_parameters) * amount_of_models_per_one_setting
    current_model = 0

    write_opening_separator_into_txt_log(output_dir)

    for config_data in config_data_parameters:
        for config_machineLearning in config_machine_learning_parameters:
            for i in range(amount_of_models_per_one_setting):

                current_model = current_model + 1
                print_currently_making_model_info(current_model, all_models_amount)

                is_making_model_finished_successfully = False
                while is_making_model_finished_successfully is False:
                    try:

                        # Unpack config_data_parameters
                        columns = config_data["columns"]
                        start_train_date = config_data["start_train_date"]
                        end_train_date = config_data["end_train_date"]
                        start_test_date = config_data["start_test_date"]
                        end_test_date = config_data["end_test_date"]
                        instrument = config_data["instrument"]

                        # Unpack config_machine_learning_parameters
                        optimizer_type = config_machineLearning["optimizer_type"]
                        loss_function_type = config_machineLearning["loss_function_type"]
                        days_into_account = config_machineLearning["days_into_account"]
                        epochs_amount = config_machineLearning["epochs_amount"]
                        random_seed = config_machineLearning["random_seed"]
                        optimizer_learning_rate = config_machineLearning["optimizer_learning_rate"]

                        # Unpack config_multiple_models_creation_process_parameters
                        output_dir = config_multiple_models_creation_process_parameters["output_dir"]
                        average_required_for_model_to_be_saved = config_multiple_models_creation_process_parameters["average_required_for_model_to_be_saved"]

                        make_multiple_variable_model(
                            columns,
                            start_train_date,
                            end_train_date,
                            start_test_date,
                            end_test_date,
                            instrument,
                            optimizer_type,
                            loss_function_type,
                            days_into_account,
                            epochs_amount,
                            random_seed,
                            optimizer_learning_rate,
                            output_dir,
                            average_required_for_model_to_be_saved)

                        is_making_model_finished_successfully = True
                    except Exception as e:
                        print_model_creation_interrupted_error(e, time_of_waiting_after_unsuccessful_model_creation_in_seconds)
                        time.sleep(time_of_waiting_after_unsuccessful_model_creation_in_seconds)


    write_closing_separator_into_txt_log(output_dir)

    # Run the whole script as sudo to let it work.
    if sleep_the_computer_when_the_work_is_done:
        os.system('shutdown -s now')
