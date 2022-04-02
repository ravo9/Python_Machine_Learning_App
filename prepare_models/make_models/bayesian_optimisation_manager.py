# from single_model_creator import make_multiple_variable_model
from bayes_opt import BayesianOptimization

def start_bayesian_optimisation_analysis():

    # Set paramaters
    params_nn ={
        # 'days_into_account':(5, 45),
        # 'epochs_amount':(2, 30),
        'layer_1_neurones_number':(0, 150),
        'layer_2_neurones_number':(0, 150),
        'layer_3_neurones_number':(0, 150),
        'layer_4_neurones_number':(0, 150)
        # 'optimizer_learning_rate':(0.0001, 0.01)
    }

    # Run Bayesian Optimization
    nn_bo = BayesianOptimization(runModelMaking, params_nn, random_state=111)
    nn_bo.maximize(init_points=30, n_iter=15)

    params_nn_ = nn_bo.max['params']
    print(params_nn_)


def runModelMaking(
    layer_1_neurones_number,
    layer_2_neurones_number,
    layer_3_neurones_number,
    layer_4_neurones_number
    # days_into_account,
    # epochs_amount
    # optimizer_learning_rate
    ):

    # Prepare config_data_parameters
    columns = ['Close']
    start_train_date='2015-09-15'
    end_train_date='2021-07-24'
    start_test_date='2021-06-27'
    end_test_date='2021-11-25'
    instrument = 'AAPL'

    # Prepare config_machine_learning_parameters
    optimizer_type = 'adam'
    loss_function_type = 'mean_absolute_error'
    days_into_account = 10
    epochs_amount = 6
    random_seed = 2323
    optimizer_learning_rate = 0.0009

    # Prepare config_multiple_models_creation_process_parameters
    output_dir = 'models_creation_output/'
    average_required_for_model_to_be_saved = 0.0060

    # epochs_amount = int(epochs_amount)
    # days_into_account = int(days_into_account)
    layer_1_neurones_number = int(layer_1_neurones_number)
    layer_2_neurones_number = int(layer_2_neurones_number)
    layer_3_neurones_number = int(layer_3_neurones_number)
    layer_4_neurones_number = int(layer_4_neurones_number)

    # layer_1_neurones_number = 18
    # layer_2_neurones_number = 99
    # layer_3_neurones_number = 37
    # layer_4_neurones_number = ??

    score = make_multiple_variable_model(
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
        average_required_for_model_to_be_saved,
        layer_1_neurones_number,
        layer_2_neurones_number,
        layer_3_neurones_number,
        layer_4_neurones_number
        )

    print("SCORE: " + str(score))
    return score
