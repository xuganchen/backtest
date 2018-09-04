import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os


import sys
backtest_dir = 'C://backtest/backtest/'
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)
    
from Backtest import *
from Backtest.open_gz_files import open_gz_files


class KDJStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
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
        for s in self.tickers:
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
            bars_high = self.data_handler.get_latest_bars_values(ticker, "high", N = self.window)
            bars_low = self.data_handler.get_latest_bars_values(ticker, "low", N = self.window)

            K, D, J = self._get_RSV(event, bars_high, bars_low, bar_date)
            LONG = sum([K > D, K > self.bK, D > self.bD, J > self.bJ])
            SHORT = sum([K < D, K < self.sK, D < self.sD, J < self.sJ])
            if LONG >= 3 and self.holdinds[ticker] == "EMPTY":
                self.generate_buy_signals(ticker, bar_date, "LONG")
                self.holdinds[ticker] = "HOLD"
            elif SHORT >= 3 and self.holdinds[ticker] == "HOLD":
                self.generate_sell_signals(ticker, bar_date, "SHORT")
                self.holdinds[ticker] = "EMPTY"


def run_backtest(config, trading_data, ohlc_data, window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90):
    config['title'] = "KDJStrategy" + "_" + str(window) + "_" + str(sK) + "_" + str(bK)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = KDJStrategy(config, events_queue, data_handler,
                           window = window, sK=sK, sD=sD, sJ=sJ, bK=bK, bD=bD, bJ=bJ)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

    # dict_ans = {
    #     "window": [window],
    #     "s": [sK],
    #     "b": [bK],
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
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/KDJStrategy",
        "title": "KDJStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq="60" + "T"),
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

    backtest, results = run_backtest(config, trading_data, ohlc_data,
                                     window = 25, sK=30, sD=30, sJ=13, bK=72, bD=72, bJ=89)



    # # interval = np.array([5, 10, 12, 26, 30, 35, 45, 60])
    # # interval_s = np.array([10, 20, 30])
    # # interval_b = np.array([90, 80, 70])
    # interval = np.array([12, 26])
    # interval_s = np.array([30])
    # interval_b = np.array([70])

    # pool = Pool(4)
    # results = []
    # for i in range(len(interval)):
    #     for j in range(len(interval_s)):
    #         for k in range(len(interval_s)):
    #             window = interval[i]
    #             s = interval_s[j]
    #             b = interval_b[k]
    #             result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, s, s, s, b, b, b,))
    #             results.append(result)

    # ans = pd.DataFrame()
    # for results in results:
    #     df = results.get()
    #     ans = pd.concat([ans, df], ignore_index=True)
    # pool.close()

    # if not os.path.exists(config['out_dir']):
    #     os.makedirs(config['out_dir'])
    # ans = ans.sort_values(by="Total Returns", ascending=False)
    # ans.to_csv(config['out_dir'] + "/result_KDJStrategy.csv")

