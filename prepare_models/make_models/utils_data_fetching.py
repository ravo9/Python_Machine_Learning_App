import pandas_datareader as web
import pandas as pd

def fetch_data_from_multiple_instruments(
    instruments,
    start_train_date,
    end_test_date,
    columns
    ):
    concatenated_dataset = None
    for instrument in instruments:
        fetched_dataset = web.DataReader(instrument, data_source='yahoo', start=start_train_date, end=end_test_date)
        # Filter columns that are interesting for us
        fetched_dataset = fetched_dataset.filter(columns)
        if concatenated_dataset is None:
            concatenated_dataset = fetched_dataset
        else:
            concatenated_dataset = pd.concat([concatenated_dataset, fetched_dataset], axis=1, join="inner")
    return concatenated_dataset
