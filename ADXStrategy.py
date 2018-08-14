import numpy as np
import pandas as pd
import queue

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars


class ADXStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,
                 window = 10):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()
        
        self.window = (window - 1) * pd.to_timedelta(str(bars.freq) + "Min")
        self.hd = pd.Series(0, index = bars.times[bars.start_date: bars.end_date])
        self.ld = pd.Series(0, index = bars.times[bars.start_date: bars.end_date])

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

        hd_mean = np.mean(self.hd[bar_date - self.window: bar_date])
        ld_mean = np.mean(self.ld[bar_date - self.window: bar_date])
        return hd_mean, ld_mean

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars_high = self.bars.get_latest_bars_values(ticker, "high", N = 2)
            bars_low = self.bars.get_latest_bars_values(ticker, "low", N = 2)

            if len(bars_high) > 1:
                hd_mean, ld_mean = self._get_hdld(bars_high, bars_low, bar_date)
                if hd_mean - ld_mean > 0 and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif hd_mean - ld_mean < 0 and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"
            else:
                self.hd[bar_date] = 0
                self.ld[bar_date] = 0

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
    strategy = ADXStrategy(data_handler, events_queue, suggested_quantity = 1,
                           window = 10)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/ADXStrategy",
        "title": "ADXStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-01-01T00:0:00", freq = "60" + "T"),    # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq = "60" + "T"),
        "equity": 500.0,
        "freq": 60,      # min
        "commission_ratio": 0.001,
        "exchange": "Binance",
        "tickers": ['BTCUSDT']
    }
    backtest, results = run(config)

