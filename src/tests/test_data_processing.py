import pandas as pd

from make_data_processing.data_processing import split_data


def test_split_data():
    # Function to test if splitting was well done.
    # This is just an example of a test that can be made.
    # More elaborate test can and will be added later.
    dataset_file = "tests/data.csv"
    df = pd.read_csv(dataset_file)
    X_train, X_test, y_train, y_test = split_data(df=df)

    assert X_train.shape[0] > X_test.shape[1]
    assert y_train.shape[0] == X_train.shape[0] and y_test.shape[0] == X_test.shape[0]
