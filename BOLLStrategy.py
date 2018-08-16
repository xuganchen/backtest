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

class BOLLStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window = 10, a = 2):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()

        self.window = window
        self.a = a

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars = self.data_handler.get_latest_bars_values(
                ticker, "close", N=self.window
            )
            bar_date = event.timestamp
            if bars is not None and bars != []:
                bars_mean = np.mean(bars)
                bars_std = np.std(bars)
                upperbound = bars_mean + self.a * bars_std
                lowerbound = bars_mean - self.a * bars_std

                # 版本1
                if event.close > upperbound and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "LONG"
                elif event.close < lowerbound and self.holdinds[ticker] == "LONG":
                    self.generate_sell_signals(ticker, bar_date, "CLOSE")
                    self.holdinds[ticker] = "EMPTY"

                # # 版本2
                # if event.close > upperbound and self.holdinds[ticker] == "EMPTY":
                #     self.generate_buy_signals(ticker, bar_date, "LONG")
                #     self.holdinds[ticker] = "LONG"
                # elif event.close < bars_mean and self.holdinds[ticker] == "LONG":
                #     self.generate_sell_signals(ticker, bar_date, "CLOSE")
                #     self.holdinds[ticker] = "EMPTY"
                #
                # elif event.close < lowerbound and self.holdinds[ticker] == "EMPTY":
                #     self.generate_sell_signals(ticker, bar_date, "SHORT")
                #     self.holdinds[ticker] = "SHORT"
                # elif event.close > bars_mean and self.holdinds[ticker] == "SHORT":
                #     self.generate_buy_signals(ticker, bar_date, "CLOSE")
                #     self.holdinds[ticker] = "EMPTY"

def run_backtest(config, trading_data, ohlc_data, window, a):
    config['title'] = "BOLLStrategy" + "_" + str(window) + "_" + str(a)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")


    events_queue = queue.Queue()
    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )

    strategy = BOLLStrategy(config, events_queue, data_handler,
                            window = window, a = a)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

    # dict_ans = {
    #     "window": [window],
    #     "a": [a],
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
        "out_dir": "C:/backtest/results/BOLLStrategy",
        "title": "MACDStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-01-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 500.0,
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

    backtest, results = run_backtest(config, trading_data, ohlc_data, window, a)



    # window_interval = np.array([5, 10, 12, 14, 20, 26, 30, 35, 45, 60, 72, 84, 96, 120, 252])
    # a_interval = np.array([1, 1.2, 1.5, 1.8, 2, 2.2, 2.5, 2.8, 3])
    # # window_interval = np.array([12, 26])
    # # a_interval = np.array([1.5, 2])

    # pool = Pool(4)
    # results = []
    # for i in range(len(window_interval)):
    #     for j in range(len(a_interval)):
    #         window = window_interval[i]
    #         a = a_interval[j]
    #         result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, a,))
    #         results.append(result)

    # ans = pd.DataFrame()
    # for results in results:
    #     df = results.get()
    #     ans = pd.concat([ans, df], ignore_index=True)
    # pool.close()

    # if not os.path.exists(config['out_dir']):
    #     os.makedirs(config['out_dir'])
    # ans = ans.sort_values(by="Total Returns", ascending=False)
    # ans.to_csv(config['out_dir'] + "/result_BOLLStrategy.csv")