import numpy as np
import pandas as pd
import queue

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

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

                # 版本1
                if event.close > upperbound and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "LONG"
                elif event.close < lowerbound and self.holdinds[ticker] == "LONG":
                    self.generate_sell_signals(ticker, bar_date, "CLOSE")
                    self.holdinds[ticker] = "EMPTY"

                # # 版本2
                # if event.close > upperbound and self.holdinds[ticker] == "EMPTY":
                #     self.generate_buy_signals(ticker, bar_date, "LONG")
                #     self.holdinds[ticker] = "LONG"
                # elif event.close < bars_mean and self.holdinds[ticker] == "LONG":
                #     self.generate_sell_signals(ticker, bar_date, "CLOSE")
                #     self.holdinds[ticker] = "EMPTY"
                #
                # elif event.close < lowerbound and self.holdinds[ticker] == "EMPTY":
                #     self.generate_sell_signals(ticker, bar_date, "SHORT")
                #     self.holdinds[ticker] = "SHORT"
                # elif event.close > bars_mean and self.holdinds[ticker] == "SHORT":
                #     self.generate_buy_signals(ticker, bar_date, "CLOSE")
                #     self.holdinds[ticker] = "EMPTY"

def run(config):
    events_queue = queue.Queue()

    # trading_data = {}
    # for ticker in config['tickers']:
    #     # trading_data[ticker] = open_gz_files(config['csv_dir'], ticker)
    #     trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker + '.h5', key=ticker)

    ohlc_data = {}
    for ticker in config['tickers']:
        # ohlc_data[ticker] = generate_bars(trading_data, ticker, config['freq'])
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    data_handler = OHLCDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date'],
        trading_data = trading_data, ohlc_data = ohlc_data
    )

    strategy = BOLLStrategy(data_handler, events_queue,
                            suggested_quantity = 1, window = 10, a = 2)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/BOLLStrategy",
        "title": "MACDStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-01-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 500.0,
        "freq": 60,  # min
        "commission_ratio": 0.001,
        "exchange": "Binance",
        "tickers": ['BTCUSDT']
    }
    backtest, results = run(config)