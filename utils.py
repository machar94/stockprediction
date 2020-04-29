import os
import warnings
import numpy as np
import pandas as pd

from solver import Solver
from forecaster import Forecaster
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Other Helper Functions

def initSignals(signals, params):

    for name, signal in signals.items():
        
        if params[name]['params'] is None:
            continue

        # Fill in parameters
        for k, v in params[name]['params'].items():
            signal['params'][k] = v

        signal['file'] = params[name]['file']


def resetSignals(indicators, params):
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

    initSignals(signals, params)

    return signals


def headlineScores(headlines):
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
        sentiment = headlineScores(headlines)

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


def cleanData(prices,
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

    return features, labels, ~rowsbool


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
        ys[i] = np.copy(labels[i+seq_length-1])

    return xs, ys


def classify(y_true, threshold):
    y_true[y_true > threshold] = 1
    y_true[y_true < threshold] = 0
    return y_true


def calcMetrics(modelparams, solverparams, dataloaders, dataset_sizes, iters,
                threshold):
    """
    Calculate metrics averaged over iters
    """

    precision = np.empty(shape=(iters, ))
    accuracy = np.empty(shape=(iters, ))
    recall = np.empty(shape=(iters, ))

    for i in range(iters):
        print(f'Model Evaluation #{i+1}')

        # Train
        model = Forecaster(n_features=modelparams['n_features'],
                           n_hidden=modelparams['n_hidden'],
                           n_layers=modelparams['n_layers'],
                           dropout=modelparams['dropout'])
        
        solver = Solver(model, 
                        num_epochs=solverparams['num_epochs'], 
                        verbose=solverparams['verbose'], 
                        plot=solverparams['plot'])
        
        model = solver.train(dataloaders, dataset_sizes)

        # Evaluate
        y_test, y_pred = solver.eval(model, dataloaders['val'])

        # Metrics
        ml_classifications = classify(y_pred, threshold)
        accuracy[i] = accuracy_score(y_test, ml_classifications)
        precision[i] = precision_score(y_test, ml_classifications)
        recall[i] = recall_score(y_test, ml_classifications)

    return np.mean(accuracy), np.mean(precision), np.mean(recall)