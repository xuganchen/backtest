import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

class BOLL_EMVStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window_BOLL = 10, a = 2,
                 window_EMV = 60, n = 30, m = 10):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.window_BOLL = window_BOLL
        self.a = a

        self.window_EMV = window_EMV
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
        roll_max_t = np.max(bars_high[-self.window_EMV:])
        roll_min_t = np.min(bars_low[-self.window_EMV:])
        roll_max_2t = np.max(bars_high[-2 * self.window_EMV: -self.window_EMV])
        roll_min_2t = np.min(bars_low[-2 * self.window_EMV: -self.window_EMV])
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
            bars = self.data_handler.get_latest_bars_values(
                ticker, "close", N=self.window_BOLL
            )
            bars_high = self.data_handler.get_latest_bars_values(ticker, "high", N = 2 * self.window_EMV)
            bars_low = self.data_handler.get_latest_bars_values(ticker, "low", N = 2 * self.window_EMV)
            bars_amount = self.data_handler.get_latest_bars_values(ticker, "amount", N = self.window_EMV)
            bar_date = event.timestamp
            if bars is not None and bars != [] and len(bars_high) > self.window_EMV:
                bars_mean = np.mean(bars)
                bars_std = np.std(bars)
                upperbound = bars_mean + self.a * bars_std
                lowerbound = bars_mean - self.a * bars_std
                em, emv, maemv = self._get_em(bars_high, bars_low, bars_amount, bar_date)

                if (event.close > upperbound and emv > maemv) and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "LONG"
                elif (event.close < lowerbound or emv < maemv) and self.holdinds[ticker] == "LONG":
                    self.generate_sell_signals(ticker, bar_date, "CLOSE")
                    self.holdinds[ticker] = "EMPTY"


def run_backtest(config, trading_data, ohlc_data, window_BOLL, a, window_EMV=40, n=10, m=10):
    config['title'] = "BOLL_EMVStrategy" + "_" + str(window_BOLL) + "_" + str(a) + "_" + str(window_EMV) + "_" + str(n) + "_" + str(m)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")


    events_queue = queue.Queue()
    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )

    strategy = BOLL_EMVStrategy(config, events_queue, data_handler,
                            window_BOLL = window_BOLL, a = a,
                           window_EMV=window_EMV, n=n, m=m)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/BOLL_EMVStrategy",
        "title": "BOLL_EMVStrategy",
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

    backtest, results = run_backtest(config, trading_data, ohlc_data, window_BOLL = 22, a = 1.28, window_EMV=65, n=14, m=9)

