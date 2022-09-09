import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

LABELS_STOCK_PRICES = ['company', 'year', 'day', 'quarter', 'stock_price']
LABELS_MARKET_ANALYSIS = ['segment', 'year', 'quarter', 'trend']
LABELS_MARKET_SEGMENTS = ['company', 'segment']
LABELS_INFO = ['company', 'year', 'day', 'quarter', 'expert1_prediction', 'expert2_prediction', 'sentiment_analysis',
               'm1', 'm2', 'm3', 'm4']
KEYS_1 = 'company'
KEYS_2 = ['segment', 'year', 'quarter']
KEYS_3 = ['company', 'year', 'day', 'quarter']

# Column ranges to exclude from data
EXCLUDE_ID_YEAR_PRICE = (2, -1)
EXCLUDE_ID_YEAR = (2, 0)


def int_to_bool_tuple(num):
    bin_string = format(num, '03b')
    return tuple(reversed([x == '1' for x in bin_string[::-1]]))


# Map segment names and group data
def standardize_data(data):
    data = data.replace('IT', 0)
    data = data.replace('BIO', 1)
    data = data.groupby(['year', 'day', 'quarter'])
    return data


# Exclude chosen number of first and last columns from data
def exclude_column_range(data, column_range):
    X = list()
    for _, group in data:
        if column_range[1] != 0:
            X.append([list(v[column_range[0]:column_range[1]]) for v in group.values])
        else:
            X.append([list(v[column_range[0]:]) for v in group.values])
    return X


# Create label array based on previous day stock prices
def derive_labels_from_stock_prices(data):
    # The stock value of each company of the first day listed is 100.0
    last_sp = [100.0, 100.0, 100.0]
    y = list()
    for _, group in data:
        stock_improvement = 0
        for i in range(3):
            # Save binary value for each company
            if list(group['stock_price'])[i] > last_sp[i]:
                stock_improvement = (stock_improvement << 1) | 1
            else:
                stock_improvement = (stock_improvement << 1) | 0
        # Save in form of integer
        y.append(stock_improvement)
        last_sp = list(group['stock_price'])
    return y


# Preprocessing data for network training and testing
def create_training_dataloaders(info_company, info_quarter, info_daily, info_prices, batch_size, split=80):
    # Create dataframes
    df_stock_prices = pd.DataFrame(info_prices, columns=LABELS_STOCK_PRICES)
    df_market_analysis = pd.DataFrame(info_quarter, columns=LABELS_MARKET_ANALYSIS)
    df_market_segments = pd.DataFrame(info_company, columns=LABELS_MARKET_SEGMENTS)
    df_info = pd.DataFrame(info_daily, columns=LABELS_INFO)

    # Merge data into the input data frame
    input_data = pd.merge(left=df_info, right=df_market_segments, left_on=KEYS_1, right_on=KEYS_1)
    input_data = pd.merge(left=input_data, right=df_market_analysis, left_on=KEYS_2, right_on=KEYS_2)
    input_data = pd.merge(left=input_data, right=df_stock_prices, left_on=KEYS_3, right_on=KEYS_3)

    # Fix non-numerical values and group
    input_data = standardize_data(input_data)

    X = exclude_column_range(input_data, EXCLUDE_ID_YEAR_PRICE)
    y = derive_labels_from_stock_prices(input_data)
    # Shift the labels to "after this information the stock went up" (True/False)
    X = X[:-1]
    y = y[1:]

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.int64)

    # Calculate train and test dataset sizes (default split 80 / 20)
    train_len = X.__len__() * split // 100
    test_len = X.__len__() - train_len

    train_ds = TensorDataset(X[:train_len], y[:train_len])
    test_ds = TensorDataset(X[test_len:], y[test_len:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


# Preprocessing data for Predictor.predict()
def prepare_prediction_data(info_company, info_quarter, info_daily):
    # Create dataframes
    df_market_analysis = pd.DataFrame(info_quarter, columns=LABELS_MARKET_ANALYSIS)
    df_market_segments = pd.DataFrame(info_company, columns=LABELS_MARKET_SEGMENTS)
    df_info = pd.DataFrame(info_daily, columns=LABELS_INFO)

    # Merge data into the input data frame
    input_data = pd.merge(left=df_info, right=df_market_segments, left_on=KEYS_1, right_on=KEYS_1)
    input_data = pd.merge(left=input_data, right=df_market_analysis, left_on=KEYS_2, right_on=KEYS_2)

    # Fix non-numerical values and group
    input_data = standardize_data(input_data)

    X = exclude_column_range(input_data, EXCLUDE_ID_YEAR)
    X = torch.tensor(X, dtype=torch.float)
    return X
