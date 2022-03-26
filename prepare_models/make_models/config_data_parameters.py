# Data Selected Parameters
columns = ['Close']
start_train_date='2015-09-15'
end_train_date='2021-07-24'
start_test_date='2021-06-27'
end_test_date='2021-11-25'
instrument = 'AAPL'


def get_config_data_parameters_many():
    configs = []
    config = get_config_data_parameters_single(
        columns, start_train_date, end_train_date, start_test_date, end_test_date, instrument
    )
    configs.append(config)
    return configs


def get_config_data_parameters_single(
    columns,
    start_train_date,
    end_train_date,
    start_test_date,
    end_test_date,
    instrument
):
    parameters = {
        "columns": columns,
        "start_train_date": start_train_date,
        "end_train_date": end_train_date,
        "start_test_date": start_test_date,
        "end_test_date": end_test_date,
        "instrument": instrument
    }
    return parameters
