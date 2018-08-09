
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



class EMVStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 10, n = 10, m = 10):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

        self.window = window
        self.n = (n - 1) * pd.to_timedelta("1min")
        self.m = (m - 1) * pd.to_timedelta("1min")

        self.em = pd.Series(0.0, index = bars.times)
        self.emv = pd.Series(0.0, index = bars.times)

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def _get_em(self, bars_high, bars_low, bars_volume, bar_date):
        roll_max_t = np.max(bars_high[-self.window:])
        roll_min_t = np.min(bars_low[-self.window:])
        roll_max_2t = np.max(bars_high[-2 * self.window: -self.window])
        roll_min_2t = np.min(bars_low[-2 * self.window: -self.window])
        roll_volume_t = np.sum(bars_volume)
        roll_t = roll_min_t + roll_max_t
        roll_2t = roll_min_2t + roll_max_2t

        em = (roll_t - roll_2t) / 2 * roll_t / roll_volume_t
        self.em[bar_date] = em
        emv = np.sum(self.em[bar_date - self.n: bar_date])
        self.emv[bar_date] = emv
        maemv = np.mean(self.emv[bar_date - self.m: bar_date])
        return em, emv, maemv

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars_high = self.bars.get_latest_bars_values(ticker, "high", N = 2 * self.window)
            bars_low = self.bars.get_latest_bars_values(ticker, "low", N = 2 * self.window)
            bars_volume = self.bars.get_latest_bars_values(ticker, "volume", N = self.window)
            bar_date = event.timestamp

            if len(bars_high) > self.window:
                em, emv, maemv = self._get_em(bars_high, bars_low, bars_volume, bar_date)
                if emv > maemv and self.holdinds[ticker] == "EMPTY":
                    print("LONG: %s" % bar_date)
                    signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "HOLD"
                elif emv < maemv and self.holdinds[ticker] == "HOLD":
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
    strategy = EMVStrategy(data_handler, events_queue,
                           suggested_quantity=1, window=40, n=10, m=10)

    backtest = Backtest(config, freq, strategy, tickers, equity, start_date, end_date, events_queue,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "C:\\Users\\user\\out\\",
        "title": "EMVStrategy",
        "save_plot": True,
        "save_tradelog": True
    }
    freq = 1    # min
    tickers = ['ETHUSDT']
    run(config, freq, tickers)
