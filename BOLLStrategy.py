
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

class BOLLStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 10, a = 2):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

        self.window = window
        self.a = a

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars = self.bars.get_latest_bars_values(
                ticker, "close", N=self.window
            )
            bar_date = event.timestamp
            if bars is not None and bars != []:
                bars_mean = np.mean(bars)
                bars_std = np.std(bars)
                upperbound = bars_mean + self.a * bars_std
                lowerbound = bars_mean - self.a * bars_std

                if event.close > upperbound and self.holdinds[ticker] == "EMPTY":
                    print("LONG: %s" % bar_date)
                    signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "LONG"
                if event.close < bars_mean and self.holdinds[ticker] == "LONG":
                    print("CLOSE: %s" % bar_date)
                    signal = SignalEvent(ticker, "SHORT", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "EMPTY"

                if event.close < lowerbound and self.holdinds[ticker] == "EMPTY":
                    print("SHORT: %s" % bar_date)
                    signal = SignalEvent(ticker, "SHORT", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "SHORT"
                if event.close > bars_mean and self.holdinds[ticker] == "SHORT":
                    print("CLOSE: %s" % bar_date)
                    signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date']
    )
    strategy = BOLLStrategy(data_handler, events_queue,
                            suggested_quantity = 1, window = 10, a = 2)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "C:\\Users\\user\\out\\",
        "title": "BOLLStrategy",
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-07-25T00:00:00", tz = "UTC"),
        "end_date": pd.Timestamp("2018-07-25T06:20:00", tz = "UTC"),
        "equity": 500.0,
        "freq": 1,      # min
        "tickers": ['ETHUSDT']
    }
    run(config)
