import datetime
import os
from pathlib import Path


def check_if_txt_log_exists(output_dir):
    log_file_path = Path(output_dir + 'log.txt')
    if log_file_path.is_file() is False:
        os.system("mkdir " + output_dir)
        os.system("touch " + str(log_file_path))


def check_if_csv_log_exists(output_dir):
    log_csv_file_path = Path(output_dir + 'log.csv')
    if log_csv_file_path.is_file() is False:
        os.system("mkdir " + output_dir)
        os.system("touch " + str(log_csv_file_path))


def write_opening_separator_into_txt_log(output_dir):
    check_if_txt_log_exists(output_dir)
    with open((output_dir + 'log.txt'),'a') as f:
        f.write("\n")
        f.write("-------------------------\n")

def write_closing_separator_into_txt_log(output_dir):
    check_if_txt_log_exists(output_dir)
    with open((output_dir + 'log.txt'),'a') as f:
        f.write("-------------------------")


def write_result_into_txt_log(output_dir, value):
    check_if_txt_log_exists(output_dir)
    with open((output_dir + 'log.txt'),'a') as f:
        f.write(str(value) + '\n')


def write_model_creation_details_into_csv_log(
    output_dir,

    average_error,
    rmse,
    direction_prediction_result,

    epochs_amount,
    random_seed

    # model_concept,
    # columns,
    # start_train_date,
    # end_train_date,
    # start_test_date,
    # end_test_date,
    # instrument,
    # output_dir,
    # optimizer_type,
    # loss_function_type,
    # days_into_account,
    # epochs_amount,
    # average_error,
    # random_seed,
    # optimizer_learning_rate
    ):
    check_if_csv_log_exists(output_dir)
    with open((output_dir + 'log.csv'),'a') as f:

        # Result
        f.write(str(average_error) + ', ')
        f.write(str(rmse) + ', ')
        f.write(str(direction_prediction_result) + ', ')

        # Separator
        f.write(', ')

        # Model Parameters
        f.write(str(epochs_amount) + ', ')
        f.write(str(random_seed) + '\n')


        # # Datetime stamp
        # f.write(str(datetime.datetime.now()) + ', ')
        #
        # # Data Parameters
        # f.write(str(instrument) + ', ')
        #     # Many columns?
        # f.write(str(columns) + ', ')
        # f.write(str(start_train_date) + ', ')
        # f.write(str(end_train_date) + ', ')
        # f.write(str(start_test_date) + ', ')
        # f.write(str(end_test_date) + ', ')
        #
        # # Model Parameters
        # f.write(str(model_concept) + ', ')
        # f.write(str(optimizer_type) + ', ')
        # f.write(str(loss_function_type) + ', ')
        # f.write(str(epochs_amount) + ', ')
        # f.write(str(days_into_account) + ', ')
        #
        # # Result
        # f.write(str(average_error) + ', ')
        #
        # # Further Model Parameters
        # f.write(str(random_seed) + ', ')
        # f.write(str(optimizer_learning_rate) + '\n')
