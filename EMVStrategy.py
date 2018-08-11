
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
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif emv < maemv and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date']
    )
    strategy = EMVStrategy(data_handler, events_queue,
                           suggested_quantity=1, window=40, n=10, m=10)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    backtest.start_trading(config)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "F:/Python/backtest/backtest/results/EMVStrategy",
        "title": "EMVStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-07-25T00:00:00", tz = "UTC"),
        "end_date": pd.Timestamp("2018-07-25T06:20:00", tz = "UTC"),
        "equity": 500.0,
        "freq": 1,      # min
        "tickers": ['ETHUSDT']
    }
    run(config)
