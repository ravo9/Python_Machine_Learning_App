# Data Selected Parameters
columns = ['Close']
start_train_date='2012-09-15'
start_test_date='2022-01-01'
end_test_date='2022-06-01'
instruments = ['NDX', 'CL=F']


def get_config_data_parameters_many():
    configs = []
    config = get_config_data_parameters_single(
        columns, start_train_date, start_test_date, end_test_date, instruments
    )
    configs.append(config)
    return configs


def get_config_data_parameters_single(
    columns,
    start_train_date,
    start_test_date,
    end_test_date,
    instruments
):
    parameters = {
        "columns": columns,
        "start_train_date": start_train_date,
        "start_test_date": start_test_date,
        "end_test_date": end_test_date,
        "instruments": instruments
    }
    return parameters
