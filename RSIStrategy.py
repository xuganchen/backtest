
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
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif RSI > self.s and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date']
    )
    strategy = RSIStrategy(data_handler, events_queue, suggested_quantity = 1,
                            window=10, s=70, b=30)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    backtest.start_trading()


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "F:/Python/backtest/backtest/results/RSIStrategy",
        "title": "RSIStrategy",
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
