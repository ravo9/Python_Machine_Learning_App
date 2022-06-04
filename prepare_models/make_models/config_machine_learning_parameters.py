# Constants
LOSS_FUNCTION_TYPE_MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
LOSS_FUNCTION_TYPE_MEAN_SQUARED_ERROR = 'mean_squared_error'
OPTIMIZER_TYPE_ADAM = 'adam'


# Machine Learning Selected Parameters
optimizer_type = OPTIMIZER_TYPE_ADAM
loss_function_type = LOSS_FUNCTION_TYPE_MEAN_ABSOLUTE_ERROR
days_into_account = [24]
epochs_amount = [10]
batch_size = 1
random_seed = [2323]
optimizer_learning_rate = [0.001]
layer_1_neurones_number = 18
layer_2_neurones_number = 99
layer_3_neurones_number = 37


def get_config_machine_learning_parameters_many():
    configs = []
    for days in days_into_account:
        for epochs in epochs_amount:
            for seed in random_seed:
                for rate in optimizer_learning_rate:
                    config = get_config_machine_learning_parameters_single(
                        optimizer_type, loss_function_type, days, epochs, batch_size, seed, rate,
                        layer_1_neurones_number, layer_2_neurones_number, layer_3_neurones_number
                    )
                    configs.append(config)
    return configs


def get_config_machine_learning_parameters_single(
    optimizer_type,
    loss_function_type,
    days_into_account,
    epochs_amount,
    batch_size,
    random_seed,
    optimizer_learning_rate,
    layer_1_neurones_number,
    layer_2_neurones_number,
    layer_3_neurones_number
):
    parameters = {
        "optimizer_type": optimizer_type,
        "loss_function_type": loss_function_type,
        "days_into_account": days_into_account,
        "epochs_amount": epochs_amount,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "optimizer_learning_rate": optimizer_learning_rate,
        "layer_1_neurones_number": layer_1_neurones_number,
        "layer_2_neurones_number": layer_2_neurones_number,
        "layer_3_neurones_number": layer_3_neurones_number
    }
    return parameters
