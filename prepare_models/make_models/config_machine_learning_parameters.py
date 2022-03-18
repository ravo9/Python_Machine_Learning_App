# Constants
LOSS_FUNCTION_TYPE_MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
LOSS_FUNCTION_TYPE_MEAN_SQUARED_ERROR = 'mean_squared_error'
OPTIMIZER_TYPE_ADAM = 'adam'


# Machine Learning Selected Parameters
optimizer_type = OPTIMIZER_TYPE_ADAM
loss_function_type = LOSS_FUNCTION_TYPE_MEAN_ABSOLUTE_ERROR
days_into_account = [25]
epochs_amount = [2]
random_seed = [2131]
optimizer_learning_rate = [0.001]


def get_config_machine_learning_parameters_many():
    configs = []
    for days in days_into_account:
        for epochs in epochs_amount:
            for seed in random_seed:
                for rate in optimizer_learning_rate:
                    config = get_config_machine_learning_parameters_single(
                        optimizer_type, loss_function_type, days, epochs, seed, rate
                    )
                    configs.append(config)
    return configs


def get_config_machine_learning_parameters_single(
    optimizer_type,
    loss_function_type,
    days_into_account,
    epochs_amount,
    random_seed,
    optimizer_learning_rate
):
    parameters = {
        "optimizer_type": optimizer_type,
        "loss_function_type": loss_function_type,
        "days_into_account": days_into_account,
        "epochs_amount": epochs_amount,
        "random_seed": random_seed,
        "optimizer_learning_rate": optimizer_learning_rate
    }
    return parameters
