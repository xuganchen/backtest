
from multiprocessing import Pool
import numpy as np
import pandas as pd
import queue
import os

from Backtest.strategy import Strategy
from Backtest.event import EventType
from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

class RSIStrategy(Strategy):
    def __init__(self, config, events, data_handler,
                 window = 10, s=70, b=30):
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']

        self.window = (window - 1) * pd.to_timedelta(str(data_handler.freq) + "Min")
        self.s = s
        self.b = b

        self.updown = pd.Series(0.0, index = data_handler.times[self.start_date: self.end_date])

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bars = self.data_handler.get_latest_bars_values(
                ticker, "close", N=2
            )
            bar_date = event.timestamp
            if len(bars) > 1:
                self.updown[bar_date] = bars[-1] - bars[-2]
                updown = self.updown[bar_date - self.window: bar_date]
                up = np.sum(updown.loc[updown > 0])
                down = -1 * np.sum(updown.loc[updown < 0])
                if down == 0:
                    RSI = 100
                else:
                    RSI = 100 - 100 / (1 + up / down)

                if RSI < self.b and self.holdinds[ticker] == "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")
                    self.holdinds[ticker] = "HOLD"
                elif RSI > self.s and self.holdinds[ticker] == "HOLD":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")
                    self.holdinds[ticker] = "EMPTY"

def run_backtest(config, trading_data, ohlc_data, window = 10, s=70, b=30):
    config['title'] = "RSIStrategy" + "_" + str(window) + "_" + str(s) + "_" + str(b)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = RSIStrategy(config, events_queue, data_handler,
                            window=window, s=s, b=b)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

    # dict_ans = {
    #     "window": [window],
    #     "s": [s],
    #     "b": [b],
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
        "out_dir": "C:/backtest/results/RSIStrategy",
        "title": "RSIStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-04-01T00:00:00", freq = "60" + "T"),    # str(freq) + "T"
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

    backtest, results = run_backtest(config, trading_data, ohlc_data, window = 2, s=78, b=15)




    # interval = np.array([5, 10, 12, 26, 30, 35, 45, 60, 72, 84, 96, 120, 252])
    # interval_s = np.array([55, 60, 70, 80, 90])
    # interval_b = np.array([10, 20, 30, 40, 45])
    # # interval = np.array([12, 26])
    # # interval_s = np.array([70])
    # # interval_b = np.array([30])

    # pool = Pool(4)
    # results = []
    # for i in range(len(interval)):
    #     for j in range(len(interval_s)):
    #         window = interval[i]
    #         s = interval_s[j]
    #         b = interval_b[j]
    #         result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, s, b, ))
    #         results.append(result)

    # ans = pd.DataFrame()
    # for results in results:
    #     df = results.get()
    #     ans = pd.concat([ans, df], ignore_index=True)
    # pool.close()

    # if not os.path.exists(config['out_dir']):
    #     os.makedirs(config['out_dir'])
    # ans = ans.sort_values(by="Total Returns", ascending=False)
    # ans.to_csv(config['out_dir'] + "/result_RSIStrategy.csv")