from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

from pandas.tseries.offsets import Minute
from Backtest.event import MarketEvent
from Backtest.open_json_files import open_json_files
from Backtest.open_gz_files import open_gz_files


class DataHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, ticker):
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, ticker, N = 1):
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, ticker):
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, ticker, val_type):
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, ticker, val_type, N = 1):
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        raise NotImplementedError("Should implement update_bars()")



class JSONDataHandler(DataHandler):
    def __init__(self, csv_dir, freq, events_queue, tickers, start_date = None, end_date = None, trading_data = None, data = None):
        self.csv_dir = csv_dir
        self.freq = freq
        self.events_queue = events_queue
        self.continue_backtest = True
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.trading_data = {}
        self.data = {}
        self.data_iter = {}
        self.latest_data = {}
        self.times = pd.Series()
        
        if trading_data is None:
            self._open_gz_files()
            # self._open_json_files()
        else:
            self.trading_data = trading_data

        if data is None:
            self.generate_bars()
        else:
            self.data = data

        for ticker in self.tickers:
            self.data_iter[ticker] = self.data[ticker].loc[self.start_date: self.end_date].iterrows()
            self.latest_data[ticker] = []

            times = self.data[ticker].index
            print("Data Time Interval for %s:" % (ticker))
            if self.start_date < times[0]:
                print("\tStart Date\t: %s" % times[0])
            else:
                print("\tStart Date\t: %s" % self.start_date)
            if self.end_date > times[-1]:
                print("\tEnd Date\t: %s" % times[-1])
            else:
                print("\tEnd Date\t: %s" % self.end_date)

            times = times.to_series()
            self.times = pd.concat([self.times, times]).drop_duplicates()
            self.times = self.times.sort_index()




    def _open_json_files(self):
        for ticker in self.tickers:
            self.trading_data[ticker] = open_json_files(self.csv_dir, ticker)

    def _open_gz_files(self):
        for ticker in self.tickers:
            self.trading_data[ticker] = open_gz_files(self.csv_dir, ticker)


    def generate_bars(self):
        for ticker in self.tickers:
            trading_data = self.trading_data[ticker]
            unit = str(self.freq) + "Min"

            trading_data["open"] = trading_data["last"]
            trading_data["high"] = trading_data["last"]
            trading_data["low"] = trading_data["last"]
            trading_data["close"] = trading_data["last"]
            data = trading_data.resample(unit).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            })
            data.loc[data['volume'] == 0, 'volume'] = np.nan
            data.fillna(method = 'backfill', inplace = True)

            self.data[ticker] = data
            # self.data_iter[ticker] = data.loc[self.start_date: self.end_date].iterrows()
            # self.latest_data[ticker] = []

            # times = data.index
            # print("Data Time Interval for %s:" % (ticker))
            # if self.start_date < times[0]:
            #     print("\tStart Date\t: %s" % self.times[0])
            # else:
            #     print("\tStart Date\t: %s" % self.start_date)
            # if self.end_date > times[-1]:
            #     print("\tEnd Date\t: %s" % self.times[-1])
            # else:
            #     print("\tEnd Date\t: %s" % self.end_date)

            # times = times.to_series()
            # self.times = pd.concat([self.times, times]).drop_duplicates()
            # self.times = self.times.sort_index()



    def _get_new_bar(self, ticker):
        for row in self.data_iter[ticker]:
            yield row

    def get_latest_bar(self, ticker):
        latest_data = self.latest_data[ticker]
        return latest_data[ticker][-1]

    def get_latest_bars(self, ticker, N = 1):
        latest_data = self.latest_data[ticker]
        return latest_data[-N:]

    def get_latest_bar_datetime(self, ticker):
        latest_data = self.latest_data[ticker]
        return latest_data[-1][0]

    def get_latest_bar_value(self, ticker, val_type):
        latest_data = self.latest_data[ticker]
        return getattr(latest_data[-1][1], val_type)

    def get_latest_bars_values(self, ticker, val_type, N = 1):
        bars_list = self.get_latest_bars(ticker, N)
        return np.array([getattr(bar[1], val_type) for bar in bars_list])


    def update_bars(self):
        for ticker in self.tickers:
            try:
                bar = next(self._get_new_bar(ticker))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_data[ticker].append(bar)
                timestamp = bar[0]
                open = getattr(bar[1], "open")
                high = getattr(bar[1], "high")
                close = getattr(bar[1], "close")
                low = getattr(bar[1], "low")
                volume = getattr(bar[1], "volume")
                freq = self.freq
                market_event = MarketEvent(ticker, timestamp, open, high, close, low, volume, freq)
                self.events_queue.put(market_event)