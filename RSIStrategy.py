
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

class RSIStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 10, s=70, b=30):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

        self.window = (window - 1) * pd.to_timedelta("1min")
        self.s = s
        self.b = b

        self.updown = pd.Series(0.0, index = bars.times)

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars = self.bars.get_latest_bars_values(
                ticker, "close", N=2
            )
            bar_date = event.timestamp
            if len(bars) > 1:
                self.updown[bar_date] = bars[-1] - bars[-2]
                updown = self.updown[bar_date - self.window: bar_date]
                up = np.sum(updown.loc[updown > 0])
                down = -1 * np.sum(updown.loc[updown < 0])
                if down == 0:
                    RSI = 100
                else:
                    RSI = 100 - 100 / (1 + up / down)

                if RSI < self.b and self.holdinds[ticker] == "EMPTY":
                    print("LONG: %s" % bar_date)
                    signal = SignalEvent(ticker, "LONG", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "HOLD"
                elif RSI > self.s and self.holdinds[ticker] == "HOLD":
                    print("SHORT: %s" % bar_date)
                    signal = SignalEvent(ticker, "SHORT", self.suggested_quantity)
                    self.events.put(signal)
                    self.holdinds[ticker] = "EMPTY"

def run(config, freq, save_plot, tickers):
    csv_dir = config["csv_dir"]
    out_dir = config["out_dir"]
    title = config["title"]
    equity = 500.0
    start_date = datetime.datetime(2018, 7, 25, 0, 0, 0)
    end_date = datetime.datetime(2018, 7, 25, 6, 20, 0)
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        csv_dir, freq, events_queue, tickers,
        start_date=start_date, end_date=end_date
    )
    strategy = RSIStrategy(data_handler, events_queue, suggested_quantity = 1,
                            window=10, s=70, b=30)

    backtest = Backtest(csv_dir, freq, strategy, tickers, equity, start_date, end_date, events_queue,
                        data_handler= data_handler)

    backtest.start_trading(out_dir = out_dir, title = title, save_plot = save_plot)


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/GOOD/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "C:\\Users\\user\\out\\",
        "title": "RSIStrategy"
    }
    freq = 1    # min
    save_plot = True
    tickers = ['ETHUSDT']
    run(config, freq, save_plot, tickers)



