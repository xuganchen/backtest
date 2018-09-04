
from multiprocessing import Pool
import numpy as np
import pandas as pd
import queue
import os

import sys
backtest_dir = '..'
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)
    
from Backtest import *
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

import talib as ta

FUNCTION_MAP = {
    'DEMA': lambda data, N: ta.DEMA(data, timeperiod=N),
    'EMA': lambda data, N: ta.EMA(data, timeperiod=N),
    'MA': lambda data, N: ta.MA(data, timeperiod=N),
    'KAMA': lambda data, N: ta.KAMA(data, timeperiod=N),
    'SMA': lambda data, N: ta.SMA(data, timeperiod=N),
    'TRIMA': lambda data, N: ta.TRIMA(data, timeperiod=N),
    'WMA': lambda data, N: ta.WMA(data, timeperiod=N),
    'TEMA': lambda data, N: ta.TEMA(data, timeperiod=N),
}

class TAStrategy(Strategy):
    def __init__(self, config, events, data_handler, index = 'EMA',
                 short_window = 10, long_window = 40):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()

        self.short_window = short_window
        self.long_window = long_window
        self.F = FUNCTION_MAP[index]

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            long = self.data_handler.get_latest_bars_values(
                ticker, "close", N=self.long_window+1
            )
            short = self.data_handler.get_latest_bars_values(
                ticker, "close", N=self.short_window+1
            )
            bar_date = event.timestamp
            if long is not None and long != []:
                F = self.F
                S = F(short, self.short_window)[-2:]
                L = F(long, self.long_window)[-2:]
                flag = (S[-1] > L[-1]) & (S[-2] < L[-2]) # cross

                if S.sum() == np.nan or L.sum() == np.nan:
                    return

                if flag and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif not flag and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run_backtest(config, trading_data, ohlc_data, index = 'EMA', short_window = 10, long_window = 40):
    config['title'] = "TA_" + index + "_" + str(short_window) + "_" + str(long_window)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = TAStrategy(config, events_queue, data_handler, index = index,
                            short_window=short_window, long_window=long_window)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results


    # dict_ans = {
    #     "short_window": [short_window],
    #     "long_window": [long_window],
    #     "Sharpe Ratio": [results['sharpe']],
    #     "Total Returns": [(results['cum_returns'][-1] - 1)],
    #     "Max Drawdown": [(results["max_drawdown"] * 100.0)],
    #     "Max Drawdown Duration": [(results['max_drawdown_duration'])],
    #     "Trades": [results['trade_info']['trading_num']],
    #     "Trade Winning": [results['trade_info']['win_pct']],
    #     "Average Trade": [results['trade_info']['avg_trd_pct']],
    #     "Average Win": [results['trade_info']['avg_win_pct']],
    #     "Average Loss": [results['trade_info']['avg_loss_pct']],
    #     "Best Trade": [results['trade_info']['max_win_pct']],
    #     "Worst Trade": [results['trade_info']['max_loss_pct']],
    #     "Worst Trade Date": [results['trade_info']['max_loss_dt']],
    #     "Avg Days in Trade": [results['trade_info']['avg_dit']]
    # }
    # return pd.DataFrame(dict_ans)


if __name__ == "__main__":
    config = {
        "csv_dir": "../Binance",
        "out_dir": "../MACDStrategy",
        "title": "MACDStrategy",
        "is_plot": False,
        "save_plot": False,
        "save_tradelog": False,
        "start_date": pd.Timestamp("2018-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 1.0,
        "freq": 60,  # min
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

    backtest, results = run_backtest(config, trading_data, ohlc_data, index = 'EMA', short_window = 20, long_window = 75)




    # interval = np.array([5, 10, 12, 26, 30, 35, 45, 60, 72, 84, 96, 120, 252])
    # # interval = np.array([5, 12, 26, 45, 60, 96, 120, 252])
    # # interval = np.array([5, 30, 120])

    # pool = Pool(4)
    # results = []
    # for i in range(len(interval)):
    #     for j in range(i + 1, len(interval)):
    #         short_window = interval[i]
    #         long_window = interval[j]
    #         result = pool.apply_async(run, args=(config, trading_data, ohlc_data, short_window, long_window,))
    #         results.append(result)

    # ans = pd.DataFrame()
    # for results in results:
    #     df = results.get()
    #     ans = pd.concat([ans, df], ignore_index=True)
    # pool.close()

    # if not os.path.exists(config['out_dir']):
    #     os.makedirs(config['out_dir'])
    # ans = ans.sort_values(by="Total Returns", ascending=False)
    # ans.to_csv(config['out_dir'] + "/result_MACDStrategy.csv")
