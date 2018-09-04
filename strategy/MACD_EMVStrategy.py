
from multiprocessing import Pool
import numpy as np
import pandas as pd
import queue
import os

import sys
backtest_dir = 'C://backtest/backtest/'
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)
    
from Backtest import *
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars



class MACD_EMVStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 short_window = 10, long_window = 40,
                 window = 60, n = 30, m = 10):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.short_window = short_window
        self.long_window = long_window

        self.window = window
        self.n = (n - 1) * pd.to_timedelta(str(data_handler.freq) + "Min") 
        self.m = (m - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")

        self.em = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])
        self.emv = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings

    def _get_em(self, bars_high, bars_low, bars_amount, bar_date):
        roll_max_t = np.max(bars_high[-self.window:])
        roll_min_t = np.min(bars_low[-self.window:])
        roll_max_2t = np.max(bars_high[-2 * self.window: -self.window])
        roll_min_2t = np.min(bars_low[-2 * self.window: -self.window])
        roll_amount_t = np.sum(bars_amount)
        roll_t = roll_min_t + roll_max_t
        roll_2t = roll_min_2t + roll_max_2t

        em = (roll_t - roll_2t) / 2 * roll_t / roll_amount_t
        self.em[bar_date] = em
        emv = np.sum(self.em[bar_date - self.n: bar_date])
        self.emv[bar_date] = emv
        maemv = np.mean(self.emv[bar_date - self.m: bar_date])
        return em, emv, maemv

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars = self.data_handler.get_latest_bars_values(ticker, "close", N=self.long_window)
            bars_high = self.data_handler.get_latest_bars_values(ticker, "high", N = 2 * self.window)
            bars_low = self.data_handler.get_latest_bars_values(ticker, "low", N = 2 * self.window)
            bars_amount = self.data_handler.get_latest_bars_values(ticker, "amount", N = self.window)
            bar_date = event.timestamp
            if bars is not None and bars != [] and len(bars_high) > self.window:
                short_ma = np.mean(bars[-self.short_window:])
                long_ma  = np.mean(bars[-self.long_window:])
                em, emv, maemv = self._get_em(bars_high, bars_low, bars_amount, bar_date)

                if (short_ma > long_ma and emv > maemv) and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif (short_ma < long_ma or emv < maemv) and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run_backtest(config, trading_data, ohlc_data, short_window = 10, long_window = 40, window=40, n=10, m=10):
    config['title'] = "MACD_EMVStrategy" + "_" + str(short_window) + "_" + str(long_window) + "_" + str(window) + "_" + str(n) + "_" + str(m)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = MACD_EMVStrategy(config, events_queue, data_handler,
                            short_window=short_window, long_window=long_window,
                           window=window, n=n, m=m)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results




if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/MACD_EMVStrategy",
        "title": "MACD_EMVStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq="60" + "T"),
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

    backtest, results = run_backtest(config, trading_data, ohlc_data, short_window = 16, long_window = 45, window=31, n=113, m=41)


