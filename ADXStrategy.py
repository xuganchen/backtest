
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



class ADXStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 10):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

        self.window = (window - 1) * pd.to_timedelta("1min")
        self.hd = pd.Series(0, index = bars.times)
        self.ld = pd.Series(0, index = bars.times)


    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def _get_hdld(self, bars_high, bars_low, bar_date):
        a = bars_high[-1] - bars_high[-2]
        b = bars_low[-2] - bars_low[-1]
        if a > 0 and a > b:
            self.hd[bar_date] = a
        else:
            self.hd[bar_date] = 0
        if b > 0 and b > a:
            self.ld[bar_date] = b
        else:
            self.ld[bar_date] = 0

        hd = np.mean(self.hd[bar_date - self.window: bar_date])
        ld = np.mean(self.ld[bar_date - self.window: bar_date])
        return hd, ld

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars_high = self.bars.get_latest_bars_values(ticker, "high", N = 2)
            bars_low = self.bars.get_latest_bars_values(ticker, "low", N = 2)

            if len(bars_high) > 1:
                hd, ld = self._get_hdld(bars_high, bars_low, bar_date)
                if hd - ld > 0 and self.holdinds[ticker] == "EMPTY":
                    print("LONG: %s" % bar_date)
                    signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "HOLD"
                elif hd - ld < 0 and self.holdinds[ticker] == "HOLD":
                    print("SHORT: %s" % bar_date)
                    signal = SignalEvent(ticker, "SHORT", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date']
    )
    strategy = ADXStrategy(data_handler, events_queue, suggested_quantity = 1,
                           window = 10)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "F:/Python/backtest/backtest/results/ADXStrategy",
        "title": "ADXStrategy",
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-07-25T00:00:00", tz = "UTC"),
        "end_date": pd.Timestamp("2018-07-25T06:20:00", tz = "UTC"),
        "equity": 500.0,
        "freq": 1,      # min
        "tickers": ['ETHUSDT']
    }
    run(config)


