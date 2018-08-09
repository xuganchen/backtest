
from __future__ import print_function

import datetime
import numpy as np
import pandas as pd

from Backtest.strategy import Strategy
from Backtest.event import SignalEvent
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import JSONDataHandler
import queue



class KDJStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

        self.window = window
        self.sK = sK
        self.sD = sD
        self.sJ = sJ
        self.bK = bK
        self.bD = bD
        self.bJ = bJ
        self.K = 0


    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def _get_RSV(self, event, bars_high, bars_low, bar_date):
        high = np.max(bars_high)
        low = np.min(bars_low)
        RSV = (event.close - low) / (high - low) * 100
        K = 2/3 * self.K + 1/3 * RSV
        D = 2/3 * self.K + 1/3 * K
        J = 3 * K - 2 * D
        self.K = K
        return RSV, K, D, J

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars_high = self.bars.get_latest_bars_values(ticker, "high", N = self.window)
            bars_low = self.bars.get_latest_bars_values(ticker, "low", N = self.window)

            if len(bars_high) > 1:
                RSV, K, D, J = self._get_RSV(event, bars_high, bars_low, bar_date)
                LONG = sum([K > D, K < self.bK, D < self.bD, J < self.bJ])
                SHORT = sum([K < D, K > self.sK, D > self.sD, J > self.sJ])
                if LONG >= 3 and self.holdinds[ticker] == "EMPTY":
                    print("LONG: %s" % bar_date)
                    signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "HOLD"
                elif SHORT >= 3 and self.holdinds[ticker] == "HOLD":
                    print("SHORT: %s" % bar_date)
                    signal = SignalEvent(ticker, "SHORT", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "EMPTY"

def run(config, freq, tickers):
    equity = 500.0
    start_date = datetime.datetime(2018, 7, 25, 0, 0, 0)
    end_date = datetime.datetime(2018, 7, 25, 6, 20, 0)
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], freq, events_queue, tickers,
        start_date=start_date, end_date=end_date
    )
    strategy = KDJStrategy(data_handler, events_queue, suggested_quantity = 1,
                           window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90)

    backtest = Backtest(config, freq, strategy, tickers, equity, start_date, end_date, events_queue,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "C:\\Users\\user\\out\\",
        "title": "KDJStrategy",
        "save_plot": True,
        "save_tradelog": True
    }
    freq = 1    # min
    tickers = ['ETHUSDT']
    run(config, freq, tickers)


