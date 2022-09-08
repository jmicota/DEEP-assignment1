import torch
import torch.nn as nn
import pandas as pd

LABELS_STOCK_PRICES = ['company', 'year', 'day', 'quarter', 'stock_price']
LABELS_MARKET_ANALYSIS = ['segment', 'year', 'quarter', 'trend']
LABELS_MARKET_SEGMENTS = ['company', 'segment']
LABELS_INFO = ['company', 'year', 'day', 'quarter', 'expert1_prediction', 'expert2_prediction', 'sentiment_analysis',
               'm1', 'm2', 'm3', 'm4']
KEYS_1 = 'company'
KEYS_2 = ['segment', 'year', 'quarter']
KEYS_3 = ['company', 'year', 'day', 'quarter']


def int_to_bool_tuple(num):
    bin_string = format(num, '03b')
    return tuple([x == '1' for x in bin_string[::-1]])


def standardize_data(data):
    # Map IT and BIO to 0 or 1 variabels
    data = data.replace('IT', 0)
    data = data.replace('BIO', 1)
    # Group by day (for each day we have data for each company)
    data = data.groupby(['year', 'day', 'quarter'])
    return data


def process_values(data):
    X = list()
    for _, group in data:
        # Slicing [2:] excludes company id and year
        X.append([list(v[2:]) for v in group.values])
    return X


def process_values_and_derive_labels(data):
    # The stock value of each company of the first day listed is 100.0
    last_sp = [100.0, 100.0, 100.0]
    X = list()
    y = list()

    # For each day
    for _, group in data:
        # Slicing [2:-1] excludes company id, year and stock_price
        X.append([list(v[2:-1]) for v in group.values])

        # Label data
        stock_improvement = 0
        for i in range(3):
            if list(group['stock_price'])[i] > last_sp[i]:
                stock_improvement = (stock_improvement << 1) | 1
            else:
                stock_improvement = (stock_improvement << 1) | 0
        y.append(stock_improvement)
        last_sp = list(group['stock_price'])

    # Shift the labeles mean "after this information the stock went up" (True/False)
    X = X[:-1]
    y = y[1:]
    return X, y


def create_training_dataloaders(info_company, info_quarter, info_daily, info_prices, batch_size, split=80):
    # Create dataframes
    df_stock_prices = pd.DataFrame(info_prices, columns=LABELS_STOCK_PRICES)
    df_market_analysis = pd.DataFrame(info_quarter, columns=LABELS_MARKET_ANALYSIS)
    df_market_segments = pd.DataFrame(info_company, columns=LABELS_MARKET_SEGMENTS)
    df_info = pd.DataFrame(info_daily, columns=LABELS_INFO)

    ### i changed input to input_data, because pytorch said 'input' is a built-in name for smth
    # Merge data into the input data frame
    input_data = pd.merge(left=df_info, right=df_market_segments, left_on=KEYS_1, right_on=KEYS_1)
    input_data = pd.merge(left=input_data, right=df_market_analysis, left_on=KEYS_2, right_on=KEYS_2)
    input_data = pd.merge(left=input_data, right=df_stock_prices, left_on=KEYS_3, right_on=KEYS_3)

    # Fix non-numerical values and group
    input_data = standardize_data(input_data)

    # Derive binary labels from stock prices
    X, y = process_values_and_derive_labels(input_data)

    ### why first 100? too slow otherwise?
    X = torch.tensor(X[:100], dtype=torch.float)
    y = torch.tensor(y[:100], dtype=torch.int64)

    ### here i changed split for train_ds, to take 80% of data, then y to take another 20%
    train_ds = torch.utils.data.TensorDataset(X[len(X) * split // 100:], y[len(X) * split // 100:])
    test_ds = torch.utils.data.TensorDataset(X[:len(X) * (100 - split) // 100], y[:len(X) * (100 - split) // 100])

    # TODO check sizes:
    # print("Train ds size: ", train_ds.size())
    # print("Test ds size: ", test_ds.size())

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


# Method for reading data in Predictor.predict()
def prediction_dataloaders(info_company, info_quarter, info_daily):
    df_market_analysis = pd.DataFrame(info_quarter, columns=LABELS_MARKET_ANALYSIS)
    df_market_segments = pd.DataFrame(info_company, columns=LABELS_MARKET_SEGMENTS)
    df_info = pd.DataFrame(info_daily, columns=LABELS_INFO)

    input_data = pd.merge(left=df_info, right=df_market_segments, left_on=KEYS_1, right_on=KEYS_1)
    input_data = pd.merge(left=input_data, right=df_market_analysis, left_on=KEYS_2, right_on=KEYS_2)

    input_data = standardize_data(input_data)
    X = process_values(input_data)

    ### also :100, probably to change??
    X = torch.tensor(X[:100], dtype=torch.float)
    return X
