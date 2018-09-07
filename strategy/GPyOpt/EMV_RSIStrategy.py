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


class EMV_RSIStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window_EMV = 60, n = 30, m = 10,
                 window_RSI = 10, s=70, b=30):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.window_EMV = window_EMV
        self.n = (n - 1) * pd.to_timedelta(str(data_handler.freq) + "Min") 
        self.m = (m - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")

        self.em = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])
        self.emv = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])

        self.window_RSI = (window_RSI - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")
        self.s = s
        self.b = b

        self.updown = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings

    def _get_em(self, bars_high, bars_low, bars_amount, bar_date):
        roll_max_t = np.max(bars_high[-self.window_EMV:])
        roll_min_t = np.min(bars_low[-self.window_EMV:])
        roll_max_2t = np.max(bars_high[-2 * self.window_EMV: -self.window_EMV])
        roll_min_2t = np.min(bars_low[-2 * self.window_EMV: -self.window_EMV])
        roll_amount_t = np.sum(bars_amount)
        roll_t = roll_min_t + roll_max_t
        roll_2t = roll_min_2t + roll_max_2t

        em = (roll_t - roll_2t) / 2 * roll_t / roll_amount_t
        self.em[bar_date] = em
        emv = np.sum(self.em[bar_date - self.n: bar_date])
        self.emv[bar_date] = emv
        maemv = np.mean(self.emv[bar_date - self.m: bar_date])
        return em, emv, maemv

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars_high = self.data_handler.get_latest_bars_values(ticker, "high", N = 2 * self.window_EMV)
            bars_low = self.data_handler.get_latest_bars_values(ticker, "low", N = 2 * self.window_EMV)
            bars_amount = self.data_handler.get_latest_bars_values(ticker, "amount", N = self.window_EMV)
            bars = self.data_handler.get_latest_bars_values(
                ticker, "close", N=2
            )
            bar_date = event.timestamp

            if len(bars_high) > self.window_EMV:
                em, emv, maemv = self._get_em(bars_high, bars_low, bars_amount, bar_date)
                self.updown[bar_date] = bars[-1] - bars[-2]
                updown = self.updown[bar_date - self.window_RSI: bar_date]
                up = np.sum(updown.loc[updown > 0])
                down = -1 * np.sum(updown.loc[updown < 0])
                if down == 0:
                    RSI = 100
                else:
                    RSI = 100 - 100 / (1 + up / down)

                if (emv > maemv and RSI < self.b) and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif (emv < maemv or RSI > self.s) and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"
                    

def run_backtest(config, trading_data, ohlc_data, window_EMV=40, n=10, m=10, window_RSI = 10, s=70, b=30):
    config['title'] = "EMV_RSIStrategy" + "_" + str(window_EMV) + "_" + str(n) + "_" + str(m) + "_" + str(window_RSI) + "_" + str(s) + "_" + str(b)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = EMV_RSIStrategy(config, events_queue, data_handler,
                           window_EMV=window_EMV, n=n, m=m,
                            window_RSI=window_RSI, s=s, b=b)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results



if __name__ == "__main__":
    config = {
        "csv_dir": back_config['data_folder'],
        "out_dir": os.path.join(back_config['output_folder'], "EMV_RSIStrategy"),
        "title": "EMV_RSIStrategy",
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

    backtest, results = run_backtest(config, trading_data, ohlc_data, window_EMV=146, n=37, m=78, window_RSI = 4, s=76, b=39)


