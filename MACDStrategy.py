import numpy as np
import pandas as pd
import queue

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import JSONDataHandler



class MACDStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 short_window = 10, long_window = 40):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

        self.short_window = short_window
        self.long_window = long_window

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars = self.bars.get_latest_bars_values(
                ticker, "close", N=self.long_window
            )
            bar_date = event.timestamp
            if bars is not None and bars != []:
                short_ma = np.mean(bars[-self.short_window:])
                long_ma  = np.mean(bars[-self.long_window:])

                if short_ma > long_ma and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif short_ma < long_ma and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()
    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date']
    )
    strategy = MACDStrategy(data_handler, events_queue, suggested_quantity = 1,
                            short_window = 10, long_window = 40)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


if __name__ == "__main__":
    config = {
        "csv_dir": "F:/Python/backtest/ethusdt-trade.csv.2018-07-25.formatted",
        "out_dir": "F:/Python/backtest/backtest/results/MACDStrategy",
        "title": "MACDStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-07-25T04:20:00", tz = "UTC"),
        "end_date": pd.Timestamp("2018-07-25T06:20:00", tz = "UTC"),
        "equity": 500.0,
        "freq": 1,      # min
        "tickers": ['ETHUSDT']
    }
    backtest, results = run(config)


