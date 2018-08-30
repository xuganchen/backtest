import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars


class ADX_RSIStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window_ADX = 10,
                 window_RSI = 10, s=70, b=30):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.window_ADX = (window_ADX - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")
        self.hd = pd.Series(0, index = data_handler.times[self.start_date: self.end_date])
        self.ld = pd.Series(0, index = data_handler.times[self.start_date: self.end_date])

        self.window_RSI = (window_RSI - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")
        self.s = s
        self.b = b

        self.updown = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
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

        hd_mean = np.mean(self.hd[bar_date - self.window_ADX: bar_date])
        ld_mean = np.mean(self.ld[bar_date - self.window_ADX: bar_date])
        return hd_mean, ld_mean

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars_high = self.data_handler.get_latest_bars_values(ticker, "high", N = 2)
            bars_low = self.data_handler.get_latest_bars_values(ticker, "low", N = 2)
            bars = self.data_handler.get_latest_bars_values(
                ticker, "close", N=2
            )

            if len(bars_high) > 1 and len(bars) > 1:
                hd_mean, ld_mean = self._get_hdld(bars_high, bars_low, bar_date)

                self.updown[bar_date] = bars[-1] - bars[-2]
                updown = self.updown[bar_date - self.window_RSI: bar_date]
                up = np.sum(updown.loc[updown > 0])
                down = -1 * np.sum(updown.loc[updown < 0])
                if down == 0:
                    RSI = 100
                else:
                    RSI = 100 - 100 / (1 + up / down)

                if (hd_mean - ld_mean > 0 and RSI < self.b) and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif (hd_mean - ld_mean < 0 or RSI > self.s) and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"
            else:
                self.hd[bar_date] = 0
                self.ld[bar_date] = 0

def run_backtest(config, trading_data, ohlc_data, window_ADX = 10, window_RSI = 10, s=70, b=30):
    config['title'] = "ADX_RSIStrategy" + "_" + str(window_ADX) + "_" + str(window_RSI) + "_" + str(s) + "_" + str(b)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")
    
    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = ADX_RSIStrategy(config, events_queue, data_handler,
                           window_ADX = window_ADX,
                            window_RSI=window_RSI, s=s, b=b)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results
    

if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/ADX_RSIStrategy",
        "title": "ADX_RSIStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-04-01T00:0:00", freq = "60" + "T"),    # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq = "60" + "T"),
        "equity": 1.0,
        "freq": 60,      # min
        "commission_ratio": 0.001,
        "suggested_quantity": None,     # None or a value
        "max_quantity": None,           # None or a value, Maximum purchase quantity
        "min_quantity": None,           # None or a value, Minimum purchase quantity
        "min_handheld_cash": None,      # None or a value, Minimum handheld funds
        "exchange": "Binance",
        "tickers": ['BTCUSDT']
    }

    # trading_data = {}
    # for ticker in config['tickers']:
    #     # trading_data[ticker] = open_gz_files(config['csv_dir'], ticker)
    #     trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker + '.h5', key=ticker)

    ohlc_data = {}
    for ticker in config['tickers']:
        # ohlc_data[ticker] = generate_bars(trading_data, ticker, config['freq'])
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    backtest, results = run_backtest(config, trading_data, ohlc_data, window_ADX = 36, window_RSI = 96, s=55, b=45)


