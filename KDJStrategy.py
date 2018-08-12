
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
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif SHORT >= 3 and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date']
    )
    strategy = KDJStrategy(data_handler, events_queue, suggested_quantity = 1,
                           window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "F:/Python/backtest/backtest/results/KDJStrategy",
        "title": "KDJStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-07-25T00:00:00", tz = "UTC"),
        "end_date": pd.Timestamp("2018-07-25T06:20:00", tz = "UTC"),
        "equity": 500.0,
        "freq": 1,      # min
        "commission": 0.001, 
        "tickers": ['ETHUSDT']
    }
    run(config)


