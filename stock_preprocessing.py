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


def int_to_bool_tuple(num):
    bin_string = format(num, '03b')
    return tuple(reversed([x == '1' for x in bin_string[::-1]]))


# Map segment names and group data
def standardize_data(data):
    data = data.replace('IT', 0)
    data = data.replace('BIO', 1)
    data = data.groupby(['year', 'day', 'quarter'])
    return data


def exclude_id_and_year_columns(data):
    X = list()
    for _, group in data:
            X.append([list(v[2:]) for v in group.values])
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


def merge_and_standardize_data(info_company, info_quarter, info_daily, info_prices):
    # Create dataframes
    df_stock_prices = pd.DataFrame(info_prices, columns=LABELS_STOCK_PRICES)
    df_market_analysis = pd.DataFrame(info_quarter, columns=LABELS_MARKET_ANALYSIS)
    df_market_segments = pd.DataFrame(info_company, columns=LABELS_MARKET_SEGMENTS)
    df_info = pd.DataFrame(info_daily, columns=LABELS_INFO)

    # Merge data into the input data frame
    data = pd.merge(left=df_info, right=df_market_segments, left_on=KEYS_1, right_on=KEYS_1)
    data = pd.merge(left=data, right=df_market_analysis, left_on=KEYS_2, right_on=KEYS_2)
    data = pd.merge(left=data, right=df_stock_prices, left_on=KEYS_3, right_on=KEYS_3)

    # Fix non-numerical values and group data
    data = standardize_data(data)
    return data


# Preprocessing data for network training and testing
def create_training_dataloaders(info_company, info_quarter, info_daily, info_prices, batch_size, split=80):
    input_data = merge_and_standardize_data(info_company, info_quarter, info_daily, info_prices)

    X = exclude_id_and_year_columns(input_data)
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
def prepare_prediction_data(info_company, info_quarter, info_daily, current_prices):
    # Process prices to match expected format
    info_prices = []
    for i in range(len(info_company)):
        info_prices.append((i, info_daily[0][1], info_daily[0][2], info_daily[0][3], current_prices[i]))

    input_data = merge_and_standardize_data(info_company, info_quarter, info_daily, info_prices)

    X = exclude_id_and_year_columns(input_data)
    X = torch.tensor(X, dtype=torch.float)
    return X
