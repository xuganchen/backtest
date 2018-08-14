import numpy as np
import pandas as pd
import queue

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.open_gz_files import open_gz_files


class KDJStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 60, sK=30, sD=30, sJ=20, bK=70, bD=70, bJ=80):
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
        self.K = 50
        self.D = 50


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
        D = 2/3 * self.D + 1/3 * K
        J = 3 * K - 2 * D
        self.K = K
        self.D = D
        return K, D, J

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars_high = self.bars.get_latest_bars_values(ticker, "high", N = self.window)
            bars_low = self.bars.get_latest_bars_values(ticker, "low", N = self.window)

            K, D, J = self._get_RSV(event, bars_high, bars_low, bar_date)
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
    strategy = KDJStrategy(data_handler, events_queue, suggested_quantity = 1,
                           window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/KDJStrategy",
        "title": "KDJStrategy",
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

