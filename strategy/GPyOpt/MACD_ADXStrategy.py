import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os


from setting import set_env
global back_config
back_config = set_env()
    
from Backtest import *
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars


class MACD_ADXStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 short_window = 10, long_window = 40,
                 window = 10):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.short_window = short_window
        self.long_window = long_window

        self.window = (window - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")
        self.hd = pd.Series(0, index = data_handler.times[self.start_date: self.end_date])
        self.ld = pd.Series(0, index = data_handler.times[self.start_date: self.end_date])

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

        hd_mean = np.mean(self.hd[bar_date - self.window: bar_date])
        ld_mean = np.mean(self.ld[bar_date - self.window: bar_date])
        return hd_mean, ld_mean

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars = self.data_handler.get_latest_bars_values(
                ticker, "close", N=self.long_window
            )
            bars_high = self.data_handler.get_latest_bars_values(ticker, "high", N = 2)
            bars_low = self.data_handler.get_latest_bars_values(ticker, "low", N = 2)
            if bars is not None and bars != [] and len(bars_high) > 1:
                short_ma = np.mean(bars[-self.short_window:])
                long_ma  = np.mean(bars[-self.long_window:])
                hd_mean, ld_mean = self._get_hdld(bars_high, bars_low, bar_date)

                if (short_ma > long_ma and hd_mean - ld_mean > 0) and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif (short_ma < long_ma or hd_mean - ld_mean < 0) and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"
            else:
                self.hd[bar_date] = 0
                self.ld[bar_date] = 0


def run_backtest(config, trading_data, ohlc_data, short_window = 10, long_window = 40, window = 10):
    config['title'] = "MACD_ADXStrategy" + "_" + str(short_window) + "_" + str(long_window) + "_" + str(window)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = MACD_ADXStrategy(config, events_queue, data_handler,
                            short_window=short_window, long_window=long_window,
                            window = window)
    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results



if __name__ == "__main__":
    config = {
        "csv_dir": back_config['data_folder'],
        "out_dir": os.path.join(back_config['output_folder'], "MACD_ADXStrategy"),
        "title": "MACD_ADXStrategy",
        "is_plot": False,
        "save_plot": True,
        "save_tradelog": True,
        # "start_date": pd.Timestamp("2018-02-01T00:0:00", freq = "60" + "T"),    # str(freq) + "T"
        # "end_date": pd.Timestamp("2018-06-01T00:00:00", freq = "60" + "T"),
        "start_date": pd.Timestamp("2018-06-01T00:0:00", freq = "60" + "T"),    # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq = "60" + "T"),
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
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '/' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    backtest, results = run_backtest(config, trading_data, ohlc_data, short_window = 20, long_window = 156, window = 20)


