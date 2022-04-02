# from multiple_models_creation_manager import start_creation_of_machine_learning_models
# from config_machine_learning_parameters import get_config_machine_learning_parameters_many
# from config_data_parameters import get_config_data_parameters_many
# from config_multiple_models_creation_process_parameters import get_config_multiple_models_creation_process_parameters
# from bayesian_optimisation_manager import start_bayesian_optimisation_analysis

# How to update the dates:
# The start_test_date doesn't have to be next day after end_train_date. It may be
# set as day 'next day after end_train_date' - days_into_account + 1 .

# Todo 1. Automate setting testing period.


# Main function
# config_data_parameters = get_config_data_parameters_many()
# config_machine_learning_parameters = get_config_machine_learning_parameters_many()
# config_multiple_models_creation_process_parameters = get_config_multiple_models_creation_process_parameters()

# start_creation_of_machine_learning_models(
#     config_data_parameters,
#     config_machine_learning_parameters,
#     config_multiple_models_creation_process_parameters
# )

start_bayesian_optimisation_analysis()
