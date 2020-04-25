import os
import talib
import warnings
import pmdarima as pm
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pmdarima import auto_arima
from functools import wraps
from os.path import isfile, join
from tqdm.notebook import tnrange

def plotSignal(func):
    @wraps(func)
    def wrapper(prices, signal, plot=False):
        func(prices, signal)

        if plot:
            plt.rcParams['font.size'] = 16
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_axes([0, 0, 1, 1])
            plt.plot(prices.index, signal['data'])
            ax.set_title(func.__name__)
            ax.set_xlabel('Dates')
            fig.autofmt_xdate(rotation=30)
            ax.grid()
            plt.show()

    return wrapper


def useFile(func):
    @wraps(func)
    def wrapper(prices, signal):

        # Set the filename to search for loading
        directory = 'features/'
        if signal['file'] is not None:
            fn = signal['file']
        else:
            fn = directory + func.__name__ + '.npy'

        # Load features if numpy file exists
        if isfile(fn):
            print(f'Loading {func.__name__} from {fn} ...')
            signal['data'] = np.load(fn)
        else:
            func(prices, signal, func.__name__)
            os.makedirs(directory, exist_ok=True)
            np.save(fn, signal['data'])

    return wrapper


def arima(price, window, desc):

    pred = np.full(price.shape, np.nan)
    for i in tnrange(window, price.shape[0], desc=desc):

        train = price[i - window:i]

        if np.any(np.isnan(train)):
            continue

        with warnings.catch_warnings():
            # Uninvertible hessian
            warnings.filterwarnings('ignore', 'Inverting')
            # RuntimeWarning: invalid value encountered in true_divide
            warnings.filterwarnings('ignore', 'invalid')
            # RuntimeWarning: overflow encountered in exp
            warnings.filterwarnings('ignore', 'overflow')
            # ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
            # warnings.filterwarnings('ignore', 'Maximum')
            # RuntimeWarning: divide by zero encountered in true_divide
            warnings.filterwarnings('ignore', 'divide')

            # Initialize model
            model = auto_arima(train,
                               max_p=3,
                               max_q=3,
                               seasonal=False,
                               trace=False,
                               error_action='ignore',
                               suppress_warnings=True)

            # Determine model parameters
            model.fit(train)
            order = model.get_params()['order']

            # Fit and predict
            model = pm.ARIMA(order=order)
            model.fit(train)
            pred[i] = model.predict(1)

    return pred


@plotSignal
def sma(prices, signal):
    """
    Simple Moving Average
    """

    window = signal['params']['window']
    signal['data'] = talib.SMA(prices['close'], window).to_numpy()[:, None]


@plotSignal
def wma(prices, signal):
    """
    Weighted Moving Average
    """

    window = signal['params']['window']
    signal['data'] = talib.WMA(prices['close'], window).to_numpy()[:, None]


@plotSignal
def mom(prices, signal):
    """
    Momentum
    """

    window = signal['params']['window']
    signal['data'] = talib.MOM(prices['close'], window).to_numpy()[:, None]


@plotSignal
def macd(prices, signal):
    """
    Moving Average Convergence Divergence
    """

    fast = signal['params']['fastperiod']
    slow = signal['params']['slowperiod']
    mavg = signal['params']['signalperiod']

    macd, macdsignal, _ = talib.MACD(prices['close'], fast, slow, mavg)

    signal['data'] = np.hstack(
        [macd.to_numpy()[:, None],
         macdsignal.to_numpy()[:, None]])


@plotSignal
def rsi(prices, signal):
    """
    Relative Strength Index
    """

    window = signal['params']['window']
    signal['data'] = talib.RSI(prices['close'], window).to_numpy()[:, None]


@plotSignal
def stoch(prices, signal):
    """
    Stochastics
    """

    slowk, slowd = talib.STOCH(prices['high'], prices['low'], prices['close'])

    signal['data'] = np.hstack(
        [slowk.to_numpy()[:, None],
         slowd.to_numpy()[:, None]])


@plotSignal
def willr(prices, signal):
    """
    Williams R Oscillator
    """

    window = signal['params']['window']
    signal['data'] = talib.WILLR(prices['high'], prices['low'],
                                 prices['close'], window).to_numpy()[:, None]


@plotSignal
def adosc(prices, signal):
    """
    Accumulation / Distribution Oscillator
    """

    signal['data'] = talib.ADOSC(prices['high'],
                                 prices['low'],
                                 prices['close'],
                                 prices['volume'],
                                 fastperiod=3,
                                 slowperiod=10).to_numpy()[:, None]


@plotSignal
def ema(prices, signal):
    """
    Exponential Moving Average
    """

    window = signal['params']['window']
    signal['data'] = talib.EMA(prices['close'], window).to_numpy()[:, None]


@plotSignal
@useFile
def arima_sma(prices, signal, name):
    """
    ARIMA on Simple Moving Average
    """

    sma_window = signal['params']['sma_window']
    sma_close = talib.SMA(prices['close'], sma_window).to_numpy()[:, None]
    signal['data'] = arima(sma_close, signal['params']['arima_window'], name)


@plotSignal
@useFile
def arima_wma(prices, signal, name):
    """
    ARIMA on Weighted Moving Average
    """

    wma_window = signal['params']['wma_window']
    wma_close = talib.WMA(prices['close'], wma_window).to_numpy()[:, None]
    signal['data'] = arima(wma_close, signal['params']['arima_window'], name)


@plotSignal
@useFile
def arima_ema(prices, signal, name):
    """
    ARIMA on Exponential Moving Average
    """

    ema_window = signal['params']['ema_window']
    ema_close = talib.EMA(prices['close'], ema_window).to_numpy()[:, None]
    signal['data'] = arima(ema_close, signal['params']['arima_window'], name)