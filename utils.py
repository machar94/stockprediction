import os
import warnings
import numpy as np
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Other Helper Functions

def resetSignals(indicators):
    """
    Creates a signals structure given list of functions
    """
    signals = {}

    for func in indicators:
        signals[func.__name__] = {}
        signals[func.__name__]['func'] = func
        signals[func.__name__]['params'] = {}
        signals[func.__name__]['data'] = None
        signals[func.__name__]['file'] = None

    return signals


def getHeadlineScores(headlines):
    analyser = SentimentIntensityAnalyzer()
    getscore = lambda x: analyser.polarity_scores(x)['compound']
    sentiment = headlines.applymap(getscore)
    return sentiment


def extractFeatures(prices, headlines, signals):
    """
    Calculate signals based on items in dictionary
    """
    for _, signal in signals.items():
        signal['func'](prices, signal, plot=False)

    sentiment = None
    if headlines is not None:
        sentiment = getHeadlineScores(headlines)

    return signals, sentiment


def prepareLabels(prices, predict):
    """
    Calculates label depending on future price
    0 - price predict days ahead is down (trend down)
    1 - price predict days ahead is up (trend up)
    """

    # Because np.nan is represented as a large negative number,
    # wait to change labels to int type until after cleaning data
    close = prices['close']
    labels = np.full(close.shape, np.nan)
    for i in range(0, len(close) - predict):
        labels[i] = close[i + predict] - close[i]

    with warnings.catch_warnings():
        # Comparison with nan values creates warnings
        warnings.filterwarnings('ignore', 'invalid')
        labels[labels >= 0] = 1
        labels[labels < 0] = 0

    return labels


def writeToFile(labels, dates, sentiment, signals, predict, dropRows, name):

    # Create an empty pandas frame
    df = pd.DataFrame(index=dates)

    # Add all indicators to df
    for column in signals.keys():
        if signals[column]['data'].shape[1] > 1:
            for i in range(signals[column]['data'].shape[1]):
                df[column + str(i)] = pd.Series(signals[column]['data'][:, i],
                                                index=df.index)
        else:
            df[column] = pd.Series(signals[column]['data'].flatten(),
                                   index=df.index)

    # Add all headlines to df
    if sentiment is not None:
        df = pd.concat([df, sentiment], axis=1)

    # Add labels to df
    df['labels'] = pd.Series(labels.flatten(), index=df.index)

    # Drop rows with nan's
    df = df.drop(df.index[dropRows])

    # Save to a file features/all_features.csv
    directory = 'features/'
    os.makedirs(directory, exist_ok=True)
    fn = directory + name + 'all_features_predict_' + str(predict) + '.csv'
    df.to_csv(fn)


def getCleanData(prices,
                 signals,
                 predict=1,
                 headlines=None,
                 name=None,
                 verbose=True):
    """
    Removes rows 
    """
    signals, sentiment = extractFeatures(prices, headlines, signals)
    labels = prepareLabels(prices, predict)

    # Create a N x D matrix of samples vs features
    features = [v['data'] for _, v in signals.items()]
    if sentiment is not None:
        features.append(sentiment.values)
    features = np.hstack(features)

    # Remove rows with nans
    rowsbool = np.isnan(features).any(axis=1)
    rowsbool = np.logical_or(rowsbool, np.isnan(labels))
    rows = np.arange(features.shape[0])[rowsbool]

    # Checks if all rows are continuous
    diffs = np.diff(np.arange(features.shape[0])[~rowsbool])
    if ~((diffs == 1).all()):
        print('Warning, time series data is not continuous!')

    if verbose:
        print(f'Removing the following times')
        print(f'============================')
        fmtr = lambda x: x.strftime('%Y/%m/%d')
        for row, time in zip(rows, prices.index[rows].format(formatter=fmtr)):
            print(f'row: {row:>4}   time: {time}')
    else:
        print(f'Removing {len(rows)} rows from data')

    # Write to file only rows with all valid values
    writeToFile(labels, prices.index, sentiment, signals, predict, rowsbool,
                name)

    features = features[~rowsbool, :]
    labels = labels[~rowsbool]

    return features, labels


def create_sequences(features, labels, seq_length):
    """
    Returns a (T,H,D) numpy array
    T - number of data points
    H - history size of lstm
    D - dimension of data X
    """

    N, D = features.shape
    T = N - seq_length + 1
    xs = np.empty((T, seq_length, D))
    ys = np.empty((T, 1), dtype=np.int)

    for i in range(T):
        xs[i] = np.copy(features[i:(i + seq_length), :])
        ys[i] = np.copy(labels[i])

    return xs, ys


def getMultiStockData(df_prices, technical_indicators, predict):

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
        signals = resetSignals(technical_indicators)

        # Set all signal parameters
        signals['sma']['params']['window'] = 14
        signals['wma']['params']['window'] = 14
        signals['mom']['params']['window'] = 14
        signals['macd']['params']['fastperiod'] = 12
        signals['macd']['params']['slowperiod'] = 26
        signals['macd']['params']['signalperiod'] = 9
        signals['rsi']['params']['window'] = 14
        signals['willr']['params']['window'] = 14

        #         signals['arima_sma']['params']['sma_window'] = 3
        #         signals['arima_sma']['params']['arima_window'] = 30
        #         signals['arima_sma']['file'] = 'features/djia_arima_sma.npy'

        #         signals['arima_wma']['params']['wma_window'] = 3
        #         signals['arima_wma']['params']['arima_window'] = 30
        #         signals['arima_wma']['file'] = 'features/djia_arima_wma.npy'

        #         signals['arima_ema']['params']['ema_window'] = 3
        #         signals['arima_ema']['params']['arima_window'] = 30
        #         signals['arima_ema']['file'] = 'features/djia_arima_ema.npy'

        features, labels = getCleanData(prices,
                                        signals,
                                        predict=predict,
                                        headlines=None,
                                        name=tick,
                                        verbose=False)

        multiStockFeatures[tick] = features
        multiStockLabels[tick] = labels

    return multiStockFeatures, multiStockLabels


def getBaselineAcc(preds):
    baseline = np.ones_like(preds)
    n = np.equal(preds, baseline).sum()
    return 100 * n / baseline.size