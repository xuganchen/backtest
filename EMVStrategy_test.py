import numpy as np
import pandas as pd
import queue
import os

from Backtest.backtest import Backtest
from Backtest.data import JSONDataHandler
from EMVStrategy import EMVStrategy
from Backtest.open_gz_files import open_gz_files


def run(config, trading_data, window=40, n=10, m=10):
    events_queue = queue.Queue()

    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date'], trading_data = trading_data
    )
    strategy = EMVStrategy(data_handler, events_queue, suggested_quantity = 100,
                            window=window, n=n, m=m)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

# 1min
def run_1min():

    ## 样本内

    config_in = {
        "csv_dir": "F:/Python/backtest/trades_Bitfinex_folder",
        "out_dir": "F:/Python/backtest/backtest/results/EMVStrategy/1min/in_sample",
        "title": "EMVStrategy",
        "is_plot": False,
        "save_plot": False,
        "save_tradelog": False,
        "start_date": pd.Timestamp("2017-07-01T00:0:00", freq="1" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq="1" + "T"),
        "equity": 100000.0,
        "freq": 1,  # min
        "tickers": ['ETHUSD']
    }

    trading_data = {}
    for ticker in config_in['tickers']:
        trading_data[ticker] = open_gz_files(config_in['csv_dir'], ticker)

    # interval = np.array([5, 10, 26, 30, 45])
    interval = np.array([10, 26])
    ans = []


    for i in range(len(interval)):
        for j in range(len(interval)):
            for k in range(len(interval)):
                window = interval[i]
                n = interval[j]
                m = interval[k]
                config_in['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(n) + "_" + str(m)
                print("---------------------------------")
                print(config_in['title'])
                print("---------------------------------")
                backtest, results = run(config_in, trading_data, window=window, n=n, m=m)
                dict_ans = {
                    "window": window,
                    "n": n,
                    "m": m,
                    "Sharpe Ratio": results['sharpe'],
                    "Total Returns": (results['cum_returns'][-1] - 1),
                    "Max Drawdown": (results["max_drawdown"] * 100.0),
                    "Max Drawdown Duration": (results['max_drawdown_duration']),
                    "Trades": results['trade_info']['trading_num'],
                    "Trade Winning": results['trade_info']['win_pct'],
                    "Average Trade": results['trade_info']['avg_trd_pct'],
                    "Average Win": results['trade_info']['avg_win_pct'],
                    "Average Loss": results['trade_info']['avg_loss_pct'],
                    "Best Trade": results['trade_info']['max_win_pct'],
                    "Worst Trade": results['trade_info']['max_loss_pct'],
                    "Worst Trade Date": results['trade_info']['max_loss_dt'],
                    "Avg Days in Trade": results['trade_info']['avg_dit']

                }
                ans.append(dict_ans)
    ans = pd.DataFrame(ans)

    if not os.path.exists(config_in['out_dir']):
        os.makedirs(config_in['out_dir'])
    ans.to_csv("F:/Python/backtest/backtest/results/EMVStrategy/1min" + "/result_EMVStrategy_in_sample.csv")
    ans = ans.sort_values(by="Total Returns", ascending=False)

    config_in["is_plot"] = True
    config_in["save_plot"] = True
    config_in["save_tradelog"] = True

    best_window = ans["window"].head(10)
    best_n = ans["n"].head(10)
    best_m = ans["m"].head(10)

    for i in range(len(best_window)):
        window = best_window.iloc[i]
        n = best_n.iloc[i]
        m = best_m.iloc[i]
        config_in['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(n) + "_" + str(m)
        backtest, results = run(config_in, trading_data, window=window, n=n, m=m)

    ## 样本外
    config_out = {
        "csv_dir": "F:/Python/backtest/trades_Bitfinex_folder",
        "out_dir": "F:/Python/backtest/backtest/results/EMVStrategy/1min/out_sample",
        "title": "EMVStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-04-01T00:0:00", freq="1" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="1" + "T"),
        "equity": 100000.0,
        "freq": 1,  # min
        "tickers": ['ETHUSD']
    }
    
    best = []
    for i in range(len(best_window)):
        window = best_window.iloc[i]
        n = best_n.iloc[i]
        m = best_m.iloc[i]
        config_out['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(n) + "_" + str(m)
        print("---------------------------------")
        print(config_out['title'])
        print("---------------------------------")
        backtest, results = run(config_out, trading_data, window=window, n=n, m=m)
        dict_ans = {        
            "window": window,
            "n": n,
            "m": m,
            "Sharpe Ratio": results['sharpe'],
            "Total Returns": (results['cum_returns'][-1] - 1),
            "Max Drawdown": (results["max_drawdown"] * 100.0),
            "Max Drawdown Duration": (results['max_drawdown_duration']),
            "Trades": results['trade_info']['trading_num'],
            "Trade Winning": results['trade_info']['win_pct'],
            "Average Trade": results['trade_info']['avg_trd_pct'],
            "Average Win": results['trade_info']['avg_win_pct'],
            "Average Loss": results['trade_info']['avg_loss_pct'],
            "Best Trade": results['trade_info']['max_win_pct'],
            "Worst Trade": results['trade_info']['max_loss_pct'],
            "Worst Trade Date": results['trade_info']['max_loss_dt'],
            "Avg Days in Trade": results['trade_info']['avg_dit']

        }
        best.append(dict_ans)
    best = pd.DataFrame(best)
    best = best.sort_values(by="Total Returns", ascending=False)

    if not os.path.exists(config_out['out_dir']):
        os.makedirs(config_out['out_dir'])
    best.to_csv("F:/Python/backtest/backtest/results/EMVStrategy/1min" + "/result_EMVStrategy_out_sample.csv")


# 60min
def run_60min():

    ## 样本内

    config_in = {
        "csv_dir": "F:/Python/backtest/trades_Bitfinex_folder",
        "out_dir": "F:/Python/backtest/backtest/results/EMVStrategy/60min/in_sample",
        "title": "EMVStrategy",
        "is_plot": False,
        "save_plot": False,
        "save_tradelog": False,
        "start_date": pd.Timestamp("2017-07-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq="60" + "T"),
        "equity": 100000.0,
        "freq": 60,  # min
        "tickers": ['ETHUSD']
    }

    trading_data = {}
    for ticker in config_in['tickers']:
        trading_data[ticker] = open_gz_files(config_in['csv_dir'], ticker)

    # interval = np.array([5, 10, 26, 30, 45])
    interval = np.array([26])

    ans = []


    for i in range(len(interval)):
        for j in range(len(interval)):
            for k in range(len(interval)):
                window = interval[i]
                n = interval[j]
                m = interval[k]
                config_in['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(n) + "_" + str(m)
                print("---------------------------------")
                print(config_in['title'])
                print("---------------------------------")
                backtest, results = run(config_in, trading_data, window=window, n=n, m=m)
                dict_ans = {
                    "window": window,
                    "n": n,
                    "m": m,
                    "Sharpe Ratio": results['sharpe'],
                    "Total Returns": (results['cum_returns'][-1] - 1),
                    "Max Drawdown": (results["max_drawdown"] * 100.0),
                    "Max Drawdown Duration": (results['max_drawdown_duration']),
                    "Trades": results['trade_info']['trading_num'],
                    "Trade Winning": results['trade_info']['win_pct'],
                    "Average Trade": results['trade_info']['avg_trd_pct'],
                    "Average Win": results['trade_info']['avg_win_pct'],
                    "Average Loss": results['trade_info']['avg_loss_pct'],
                    "Best Trade": results['trade_info']['max_win_pct'],
                    "Worst Trade": results['trade_info']['max_loss_pct'],
                    "Worst Trade Date": results['trade_info']['max_loss_dt'],
                    "Avg Days in Trade": results['trade_info']['avg_dit']

                }
                ans.append(dict_ans)
    ans = pd.DataFrame(ans)

    if not os.path.exists(config_in['out_dir']):
        os.makedirs(config_in['out_dir'])
    ans.to_csv("F:/Python/backtest/backtest/results/EMVStrategy/60min" + "/result_EMVStrategy_in_sample.csv")
    ans = ans.sort_values(by="Total Returns", ascending=False)

    config_in["is_plot"] = True
    config_in["save_plot"] = True
    config_in["save_tradelog"] = True

    best_window = ans["window"].head(10)
    best_n = ans["n"].head(10)
    best_m = ans["m"].head(10)

    for i in range(len(best_window)):
        window = best_window.iloc[i]
        n = best_n.iloc[i]
        m = best_m.iloc[i]
        config_in['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(n) + "_" + str(m)
        backtest, results = run(config_in, trading_data, window=window, n=n, m=m)

    ## 样本外
    config_out = {
        "csv_dir": "F:/Python/backtest/trades_Bitfinex_folder",
        "out_dir": "F:/Python/backtest/backtest/results/EMVStrategy/60min/out_sample",
        "title": "EMVStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 100000.0,
        "freq": 60,  # min
        "tickers": ['ETHUSD']
    }
    
    best = []
    for i in range(len(best_window)):
        window = best_window.iloc[i]
        n = best_n.iloc[i]
        m = best_m.iloc[i]
        config_out['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(n) + "_" + str(m)
        print("---------------------------------")
        print(config_out['title'])
        print("---------------------------------")
        backtest, results = run(config_out, trading_data, window=window, n=n, m=m)
        dict_ans = {        
            "window": window,
            "n": n,
            "m": m,
            "Sharpe Ratio": results['sharpe'],
            "Total Returns": (results['cum_returns'][-1] - 1),
            "Max Drawdown": (results["max_drawdown"] * 100.0),
            "Max Drawdown Duration": (results['max_drawdown_duration']),
            "Trades": results['trade_info']['trading_num'],
            "Trade Winning": results['trade_info']['win_pct'],
            "Average Trade": results['trade_info']['avg_trd_pct'],
            "Average Win": results['trade_info']['avg_win_pct'],
            "Average Loss": results['trade_info']['avg_loss_pct'],
            "Best Trade": results['trade_info']['max_win_pct'],
            "Worst Trade": results['trade_info']['max_loss_pct'],
            "Worst Trade Date": results['trade_info']['max_loss_dt'],
            "Avg Days in Trade": results['trade_info']['avg_dit']

        }
        best.append(dict_ans)
    best = pd.DataFrame(best)
    best = best.sort_values(by="Total Returns", ascending=False)

    if not os.path.exists(config_out['out_dir']):
        os.makedirs(config_out['out_dir'])
    best.to_csv("F:/Python/backtest/backtest/results/EMVStrategy/60min" + "/result_EMVStrategy_out_sample.csv")


# run

# run_1min()

run_60min()