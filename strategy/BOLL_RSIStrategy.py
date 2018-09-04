import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os

import sys
backtest_dir = 'C://backtest/backtest/'
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)
    
from Backtest import *
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

class BOLL_RSIStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window_BOLL = 10, a = 2,
                 window_RSI = 10, s=70, b=30):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.window_BOLL = window_BOLL
        self.a = a

        self.window_RSI = (window_RSI - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")
        self.s = s
        self.b = b

        self.updown = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars_BOLL = self.data_handler.get_latest_bars_values(
                ticker, "close", N=self.window_BOLL
            )
            bars_RSI = self.data_handler.get_latest_bars_values(
                ticker, "close", N=2
            )
            bar_date = event.timestamp
            if bars_BOLL is not None and bars_BOLL != [] and len(bars_RSI) > 1:
                bars_mean = np.mean(bars_BOLL)
                bars_std = np.std(bars_BOLL)
                upperbound = bars_mean + self.a * bars_std
                lowerbound = bars_mean - self.a * bars_std

                self.updown[bar_date] = bars_RSI[-1] - bars_RSI[-2]
                updown = self.updown[bar_date - self.window_RSI: bar_date]
                up = np.sum(updown.loc[updown > 0])
                down = -1 * np.sum(updown.loc[updown < 0])
                if down == 0:
                    RSI = 100
                else:
                    RSI = 100 - 100 / (1 + up / down)

                if (event.close > upperbound and RSI < self.b) and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "LONG"
                elif (event.close < lowerbound or RSI > self.s) and self.holdinds[ticker] == "LONG":
                    self.generate_sell_signals(ticker, bar_date, "CLOSE")
                    self.holdinds[ticker] = "EMPTY"


def run_backtest(config, trading_data, ohlc_data, window_BOLL, a, window_RSI, s, b):
    config['title'] = "BOLL_RSIStrategy" + "_" + str(window_BOLL) + "_" + str(a) + "_" + str(window_RSI) + "_" + str(s) + "_" + str(b)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")


    events_queue = queue.Queue()
    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )

    strategy = BOLL_RSIStrategy(config, events_queue, data_handler,
                            window_BOLL = window_BOLL, a = a,
                            window_RSI=window_RSI, s=s, b=b)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/BOLL_RSIStrategy",
        "title": "BOLL_RSIStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 1.0,
        "freq": 60,  # min
        "commission_ratio": 0.001,
        "suggested_quantity": None,     # None or a value
        "max_quantity": None,           # None or a value, Maximum purchase quantity
        "min_quantity": None,           # None or a value, Minimum purchase quantity
        "min_handheld_cash": None,      # None or a value, Minimum handheld funds
        "exchange": "Binance",
        "tickers": ['BTCUSDT']
    }

    # trading_data = {}
    # for ticker in config['tickers']:
    #     # trading_data[ticker] = open_gz_files(config['csv_dir'], ticker)
    #     trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker + '.h5', key=ticker)

    ohlc_data = {}
    for ticker in config['tickers']:
        # ohlc_data[ticker] = generate_bars(trading_data, ticker, config['freq'])
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    backtest, results = run_backtest(config, trading_data, ohlc_data, window_BOLL = 30, a = 0.1, window_RSI = 7, s=76, b=45)

