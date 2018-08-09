
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



class MACDStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()


    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp

            if self.holdinds[ticker] == "EMPTY":
                print("LONG: %s" % bar_date)
                signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                self.events.put(signal)
                self.holdinds[ticker] = "HOLD"
            elif self.holdinds[ticker] == "HOLD":
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
        config["csv_dir"], freq, events_queue, tickers,
        start_date=start_date, end_date=end_date
    )
    strategy = MACDStrategy(data_handler, events_queue)

    backtest = Backtest(config, freq, strategy, tickers, equity, start_date, end_date, events_queue,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "C:\\Users\\user\\out\\",
        "title": "xxxStrategy",
        "save_plot": True,
        "save_tradelog": True
    }
    freq = 1    # min
    tickers = ['ETHUSDT']
    run(config, freq, tickers)
