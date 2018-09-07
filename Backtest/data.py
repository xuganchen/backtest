from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

from pandas.tseries.offsets import Minute
from Backtest.event import MarketEvent


class DataHandler(object):
    '''
    Handling the work related to DATA.
    DataHandler is base class providing an interface for all subsequent data handler.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, ticker):
        '''
        Get the latest bar information for ticker,
        '''
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, ticker, N = 1):
        '''
        Get the latest N bar information for ticker
        '''
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, ticker):
        '''
        Get the latest datetime
        '''
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, ticker, val_type):
        '''
        Get the lastest bar's val_type
        '''
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, ticker, val_type, N = 1):
        '''
        Get the lastest N bar's val_type
        '''
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        '''
        Update timeline for data 
        and generate MARKET event for each ticker
        '''
        raise NotImplementedError("Should implement update_bars()")



class OHLCDataHandler(DataHandler):
    '''
    Handling the work related to DATA.
    The data format is "ticker timestamp open high low close"(OHLC)
    '''
    def __init__(self, config, events_queue, trading_data = None, ohlc_data = None):
        '''
        Parameters:
        csv_dir: input data path,
        freq: the frequency in config, timedelta between every two bar
        events_queue: the event queue
        tickers: the list of trading digital currency
        start_date: strat datetime of backtesting
        end_date: end datetime of backtesting
        trading_data: transaction data
            dict - trading_data[ticker] = df_ticker
                df_ticker = pd.DataFrame(index = "pd.timestamp", 
                                        columns = ["volume", "last"])
        ohlc_data: ohlc data
            dict - ohlc_data[ticker] = df_ticker
                df_ticker = pd.DataFrame(index = "pd.timestamp", 
                    columns = ["open", "high", "low", "close", "volume", "amount"])
        '''
        '''
        self.continue_backtest: the condition whether can continue backtesting,
                                determined by timeline
        self.trading_data = trading_data
        self.data = ohlc_data
        self.data_iter: for each ticker, the iterator of data DataFrame
        self.latest_data: for each ticker, past time data
        self.times: the time series of all time
        '''
        self.config = config
        self.csv_dir = config['csv_dir']
        self.freq = config['freq']
        self.tickers = config['tickers']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.events_queue = events_queue

        self.continue_backtest = {ticker: True for ticker in self.tickers}

        self.trading_data = {}
        self.data = {}
        self.data_iter = {}
        self.latest_data = {}
        self.times = pd.Series()
        
        if trading_data is None and ohlc_data is None:
            raise ValueError("Should input trading_data or ohlc_data")

        if trading_data is not None:
            self.trading_data = trading_data

        if ohlc_data is None:
            self.generate_bars()
        else:
            self.data = ohlc_data

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
            self.times = self.times.loc[self.start_date: self.end_date]

    def generate_bars(self):
        '''
        for each ticker, organize transaction data into OHLC data
        '''
        for ticker in self.tickers:
            trading_data = self.trading_data[ticker]
            unit = str(self.freq) + "Min"

            trading_data["open"] = trading_data["last"]
            trading_data["high"] = trading_data["last"]
            trading_data["low"] = trading_data["last"]
            trading_data["close"] = trading_data["last"]
            trading_data['amount'] = trading_data['last'] * trading_data['volume']
            data = trading_data.resample(unit).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum"
            })
            data.loc[data['volume'] == 0, 'volume'] = np.nan
            data.loc[data['amount'] == 0, 'amount'] = np.nan
            data.fillna(method = 'backfill', inplace = True)

            self.data[ticker] = data

    def _get_new_bar(self, ticker):
        """
        iterate the data for ticker
        """
        for row in self.data_iter[ticker]:
            yield row

    def get_latest_bar(self, ticker):
        '''
        Get the latest bar information for ticker,
        return a N-row dict:
            {"timestamp",
            ["open", "high", "low", "close", "volume", "amount"]]}
        '''
        latest_data = self.latest_data[ticker]
        return latest_data[ticker][-1]

    def get_latest_bars(self, ticker, N = 1):
        '''
        Get the latest N bar information for ticker,
        return a 1-row dict:
            {"timestamp",
            ["open", "high", "low", "close", "volume", "amount"]]}
        '''
        latest_data = self.latest_data[ticker]
        return latest_data[-N:]


    def get_latest_bar_datetime(self, ticker):
        '''
        Get the latest datetime
        return timestamp
        '''
        latest_data = self.latest_data[ticker]
        return latest_data[-1][0]

    def get_latest_bar_value(self, ticker, val_type):
        '''
        Get the lastest bar's val_type
        return float
        
        Parameters:
        val_type: in ["open", "high", "low", "close", "volume", "amount"]]
        '''
        latest_data = self.latest_data[ticker]
        return getattr(latest_data[-1][1], val_type)

    def get_latest_bars_values(self, ticker, val_type, N = 1):
        '''
        Get the lastest N bar's val_type
        return np.array
        
        Parameters:
        val_type: in ["open", "high", "low", "close", "volume", "amount"]]
        '''
        bars_list = self.get_latest_bars(ticker, N)
        return np.array([getattr(bar[1], val_type) for bar in bars_list])


    def update_bars(self):
        '''
        Update timeline for data 
        and generate MARKET event for each ticker

        If time is up, set continue_backtest = False
        and finish the backtest
        '''
        now_time = None
        for ticker in self.tickers:
            try:
                bar = next(self._get_new_bar(ticker))
            except StopIteration:
                self.continue_backtest[ticker] = False
            else:
                if bar is not None:
                    self.latest_data[ticker].append(bar)
                    timestamp = bar[0]
                    open = getattr(bar[1], "open", np.nan)
                    high = getattr(bar[1], "high", np.nan)
                    close = getattr(bar[1], "close", np.nan)
                    low = getattr(bar[1], "low", np.nan)
                    volume = getattr(bar[1], "volume", np.nan)
                    amount = getattr(bar[1], "amount", np.nan)
                    freq = self.freq
                    market_event = MarketEvent(ticker, timestamp, open, high, low, close, volume, amount, freq)
                    self.events_queue.put(market_event)
                    now_time = timestamp
        if now_time is None:
            return None
        else:
            return now_time           


        