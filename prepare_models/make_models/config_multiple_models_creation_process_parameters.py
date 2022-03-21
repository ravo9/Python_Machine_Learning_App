# Multiple Models Creation Process Parameters
output_dir = 'models_creation_output/'
average_required_for_model_to_be_saved = 0.0060
amount_of_models_per_one_setting = 1
sleep_the_computer_when_the_work_is_done = False
time_of_waiting_after_unsuccessful_model_creation_in_seconds = 15


def get_config_multiple_models_creation_process_parameters():
    parameters = {
        "output_dir": output_dir,
        "average_required_for_model_to_be_saved": average_required_for_model_to_be_saved,
        "amount_of_models_per_one_setting": amount_of_models_per_one_setting,
        "sleep_the_computer_when_the_work_is_done": sleep_the_computer_when_the_work_is_done,
        "time_of_waiting_after_unsuccessful_model_creation_in_seconds": time_of_waiting_after_unsuccessful_model_creation_in_seconds
    }
    return parameters
