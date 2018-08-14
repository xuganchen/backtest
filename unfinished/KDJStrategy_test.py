
from multiprocessing import Pool
import numpy as np
import pandas as pd
import queue
import os

from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from KDJStrategy import KDJStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

def run_backtest(config, trading_data, ohlc_data, window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90):
    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date'],
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = KDJStrategy(data_handler, events_queue, suggested_quantity = 100,
                           window = window, sK=sK, sD=sD, sJ=sJ, bK=bK, bD=bD, bJ=bJ)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

def run(config, trading_data, ohlc_data, window = 10, sK=20, sD=20, sJ=10, bK=80, bD=80, bJ=90):
    config['title'] = "KDJStrategy" + "_" + str(window) + "_" + str(sK) + "_" + str(bK)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")
    backtest, results = run_backtest(config, trading_data, ohlc_data,
                                     window = window, sK=sK, sD=sD, sJ=sJ, bK=bK, bD=bD, bJ=bJ)
    dict_ans = {
        "window": window,
        "s": sK,
        "b": bK,
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
        "out_dir": "C:/backtest/results/KDJStrategy/in_sample",
        "title": "KDJStrategy",
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

    # interval = np.array([5, 10, 12, 26, 30, 35, 45, 60])
    # interval_s = np.array([10, 20, 30])
    # interval_b = np.array([90, 80, 70])
    interval = np.array([12, 26])
    interval_s = np.array([30])
    interval_b = np.array([70])

    pool = Pool(4)
    results = []
    for i in range(len(interval)):
        for j in range(len(interval_s)):
            for k in range(len(interval_s)):
                window = interval[i]
                s = interval_s[j]
                b = interval_b[k]
                result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, s, s, s, b, b, b,))
                results.append(result)

    ans = pd.DataFrame()
    for results in results:
        df = results.get()
        ans = pd.concat([ans, df], ignore_index=True)
    pool.close()

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    ans = ans.sort_values(by="Total Returns", ascending=False)
    ans.to_csv(config['out_dir'] + "/result_KDJStrategy_in_sample.csv")

    config["is_plot"] = True
    config["save_plot"] = True
    config["save_tradelog"] = True

    best_window = ans["window"].head(10)
    best_s = ans["s"].head(10)
    best_b = ans["b"].head(10)

    pool = Pool(4)
    results = []
    for i in range(len(best_window)):
        window = best_window.iloc[i]
        s = best_s.iloc[i]
        b = best_b.iloc[i]
        result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, s, s, s, b, b, b,))
        results.append(result)

    for results in results:
        df = results.get()
    pool.close()
    
## 样本外
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/KDJStrategy/out_sample",
        "title": "KDJStrategy",
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
    for i in range(len(best_window)):
        window = best_window.iloc[i]
        s = best_s.iloc[i]
        b = best_b.iloc[i]
        result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, s, s, s, b, b, b,))
        results.append(result)

    ans = pd.DataFrame()
    for results in results:
        df = results.get()
        ans = pd.concat([ans, df], ignore_index=True)
    pool.close()

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    ans = ans.sort_values(by="Total Returns", ascending=False)
    ans.to_csv(config['out_dir'] + "/result_KDJStrategy_out_sample.csv")