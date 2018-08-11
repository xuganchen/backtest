from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

from pandas.tseries.offsets import Minute
from Backtest.event import MarketEvent
from Backtest.open_json_files import open_json_files


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
    def __init__(self, csv_dir, freq, events_queue, tickers, start_date = None, end_date = None):
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
        self._open_json_files()
        self.generate_bars()

    def _open_json_files(self):
        for ticker in self.tickers:
            self.trading_data[ticker] = open_json_files(self.csv_dir, ticker)


    def generate_bars(self):
        for ticker in self.tickers:
            unit = self.freq * Minute()
            trading_data = self.trading_data[ticker]
            time_counter = trading_data.head(1).index[0].floor('1min')
            records = []
            times = []
            while True:
                if time_counter > trading_data.tail(1).index[0]:
                    break

                data = trading_data[(trading_data.index < time_counter + unit) & (trading_data.index > time_counter)]
                if data.empty:
                    continue

                times.append(time_counter)
                record = []
                record.append(data['last'].head(1)[0])  # open
                record.append(data['last'].max())  # high
                record.append(data['last'].min())  # low
                record.append(data['last'].tail(1)[0])  # close
                record.append(data['volume'].sum())  # volume
                records.append(record)

                time_counter = time_counter + unit

            self.times = times
            print("Data Time Interval:")
            if self.start_date < times[0]:
                print("Start Date: %s" % times[0])
            else:
                print("Start Date: %s" % self.start_date)
            if self.end_date > times[-1]:
                print("End Date: %s" % times[-1])
            else:
                print("End Date: %s" % self.end_date)

            self.data = pd.DataFrame.from_records(records, columns=['open', 'high', 'low', 'close', 'volume'],
                                                  index=times)
            self.data_iter[ticker] = self.data.loc[self.start_date: self.end_date].iterrows()
            self.latest_data[ticker] = []


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