
from multiprocessing import Pool
import numpy as np
import pandas as pd
import queue
import os

from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from MACDStrategy import MACDStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

def run_backtest(config, trading_data, ohlc_data, short_window = 10, long_window = 40):
    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date'],
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = MACDStrategy(data_handler, events_queue, suggested_quantity = 100,
                            short_window = short_window, long_window = long_window)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

def run(config, trading_data, ohlc_data, short_window, long_window):
    config['title'] = "MACDStrategy" + "_" + str(short_window) + "_" + str(long_window)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")
    backtest, results = run_backtest(config, trading_data, ohlc_data, short_window=short_window, long_window=long_window)
    dict_ans = {
        "short_window": [short_window],
        "long_window": [long_window],
        "Sharpe Ratio": [results['sharpe']],
        "Total Returns": [(results['cum_returns'][-1] - 1)],
        "Max Drawdown": [(results["max_drawdown"] * 100.0)],
        "Max Drawdown Duration": [(results['max_drawdown_duration'])],
        "Trades": [results['trade_info']['trading_num']],
        "Trade Winning": [results['trade_info']['win_pct']],
        "Average Trade": [results['trade_info']['avg_trd_pct']],
        "Average Win": [results['trade_info']['avg_win_pct']],
        "Average Loss": [results['trade_info']['avg_loss_pct']],
        "Best Trade": [results['trade_info']['max_win_pct']],
        "Worst Trade": [results['trade_info']['max_loss_pct']],
        "Worst Trade Date": [results['trade_info']['max_loss_dt']],
        "Avg Days in Trade": [results['trade_info']['avg_dit']]
    }
    return pd.DataFrame(dict_ans)


if __name__ == '__main__':

## 样本内

    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/MACDStrategy/in_sample",
        "title": "MACDStrategy",
        "is_plot": False,
        "save_plot": False,
        "save_tradelog": False,
        "start_date": pd.Timestamp("2017-05-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq="60" + "T"),
        "equity": 100000.0,
        "freq": 60,  # min
        "commission_ratio": 0.001,
        "exchange": "Binance",
        "tickers": ['BTCUSDT']
    }

    # trading_data = {}
    # for ticker in config['tickers']:
    #     # trading_data[ticker] = open_json_gz_files(config_in['csv_dir'], ticker
    #     #                           config['start_date'], config['end_date'])
    #     trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker + '.h5', key=ticker)
    #     # trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + 'BTCUSDT2018-05-01to2018-09-01.h5', key=ticker)

    ohlc_data = {}
    for ticker in config['tickers']:
        # ohlc_data[ticker] = generate_bars(trading_data, ticker, config['freq'])
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    interval = np.array([5, 10, 12, 26, 30, 35, 45, 60, 72, 84, 96, 120, 252])
    # interval = np.array([5, 12, 26, 45, 60, 96, 120, 252])
    # interval = np.array([5, 30, 120])

    pool = Pool(4)
    results = []
    for i in range(len(interval)):
        for j in range(i + 1, len(interval)):
            short_window = interval[i]
            long_window = interval[j]
            result = pool.apply_async(run, args=(config, trading_data, ohlc_data, short_window, long_window,))
            results.append(result)

    ans = pd.DataFrame()
    for results in results:
        df = results.get()
        ans = pd.concat([ans, df], ignore_index=True)
    pool.close()

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    ans = ans.sort_values(by="Total Returns", ascending=False)
    ans.to_csv(config['out_dir'] + "/result_MACDStrategy_in_sample.csv")

    config["is_plot"] = True
    config["save_plot"] = True
    config["save_tradelog"] = True

    best_short_window = ans["short_window"].head(10)
    best_long_window = ans["long_window"].head(10)

    pool = Pool(4)
    results = []
    for i in range(len(best_short_window)):
        short_window = best_short_window.iloc[i]
        long_window = best_long_window.iloc[i]
        result = pool.apply_async(run, args=(config, trading_data, ohlc_data, short_window, long_window,))
        results.append(result)

    for results in results:
        df = results.get()
    pool.close()
    
## 样本外
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/MACDStrategy/out_sample",
        "title": "MACDStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 100000.0,
        "freq": 60,  # min
        "commission_ratio": 0.001,
        "exchange": "Binance",
        "tickers": ['BTCUSDT']
    }

    # trading_data = {}
    # for ticker in config['tickers']:
    #     # trading_data[ticker] = open_json_gz_files(config_in['csv_dir'], ticker
    #     #                           config['start_date'], config['end_date'])
    #     trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker + '.h5', key=ticker)
    #     # trading_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + 'BTCUSDT2018-05-01to2018-09-01.h5', key=ticker)

    ohlc_data = {}
    for ticker in config['tickers']:
        # ohlc_data[ticker] = generate_bars(trading_data, ticker, config['freq'])
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    pool = Pool(4)
    results = []
    for i in range(len(best_short_window)):
        short_window = best_short_window.iloc[i]
        long_window = best_long_window.iloc[i]
        result = pool.apply_async(run, args=(config, trading_data, ohlc_data, short_window, long_window,))
        results.append(result)

    ans = pd.DataFrame()
    for results in results:
        df = results.get()
        ans = pd.concat([ans, df], ignore_index=True)
    pool.close()

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    ans = ans.sort_values(by="Total Returns", ascending=False)
    ans.to_csv(config['out_dir'] + "/result_MACDStrategy_out_sample.csv")