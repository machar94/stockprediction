import utils
import numpy as np

from sklearn.model_selection import train_test_split as tts

def cleanData(df_prices, technical_indicators, params, predict):
    """
    Feature extraction for yahoo finance pandasreader stock input
    """

    tickers = df_prices['Close'].columns.to_list()

    multiStockFeatures = {}
    multiStockLabels = {}

    for tick in tickers:
        print('Creating features for ' + tick)
        criteria = df_prices.columns.get_level_values(level=1).isin([tick])
        prices = df_prices[df_prices.columns[criteria]]
        
        # Magic 1 because yahoo finance has the following column organization
        # Level 0: High, Open, Low, Close, Adj Close etc. (not sure of order)
        # Level 1: <Stock Name 1>, <Stock Name 2> etc.
        # We seek to ignore the stock name and just have the level 0 cols
        prices = prices.droplevel(1, axis=1)

        prices = prices.filter(['High', 'Low', 'Volume', 'Adj Close'],
                                  axis=1)

        names = {
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume',
            'Adj Close': 'close'
        }

        prices.rename(columns=names, inplace=True)

        # Prepare signals dictionary layout for uploading features
        signals = utils.resetSignals(technical_indicators, params)

        features, labels = utils.cleanData(prices,
                                           signals,
                                           predict=predict,
                                           headlines=None,
                                           name=tick,
                                           verbose=False)

        multiStockFeatures[tick] = features
        multiStockLabels[tick] = labels

    return multiStockFeatures, multiStockLabels


def train_test_split(features, labels, scaler, seq_len, n_features,
                     test_type='all_stocks', test_size=0.25):
    
    if test_type not in ['all_stocks', 'new_stocks']:
        raise ValueError('invalid test_type provided')
    
    X_train = np.empty(shape=(0, seq_len, n_features))
    X_test = np.empty(shape=(0, seq_len, n_features))
    y_train = np.empty(shape=(0, 1))
    y_test = np.empty(shape=(0, 1))
    
    if test_type == 'new_stocks':
        n_train = len(features.keys()) - \
            int(len(features.keys()) * test_size)
        
        train_ticks = ", ".join(list(features.keys())[n_train:])
        print('Using ' + train_ticks + ' data for testing')

    for i, tick in enumerate(features.keys()):
        scaled = scaler.transform(features[tick])
        x, y = utils.create_sequences(scaled, labels[tick], seq_len)
        
        if test_type == 'new_stocks':
            if i < n_train:
                X_train = np.concatenate((X_train, x), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)
            else:
                X_test = np.concatenate((X_test, x), axis=0)
                y_test = np.concatenate((y_test, y), axis=0)
                
        elif test_type == 'all_stocks':
            # Split the data into training and testing sets
            xt, xv, yt, yv = tts(x, y, test_size=test_size, shuffle=False)

            X_train = np.concatenate((X_train, xt), axis=0)
            X_test = np.concatenate((X_test, xv), axis=0)
            y_train = np.concatenate((y_train, yt), axis=0)
            y_test = np.concatenate((y_test, yv), axis=0)
            
    return X_train, X_test, y_train, y_test