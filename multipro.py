
from multiprocessing import Pool
import numpy as np
import pandas as pd
import queue
import os

from Backtest.backtest import Backtest
from Backtest.data import JSONDataHandler
from MACDStrategy import MACDStrategy
from Backtest.open_gz_files import open_gz_files

def run(config, trading_data, short_window = 10, long_window = 40):
    events_queue = queue.Queue()

    data_handler = JSONDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date'], trading_data = trading_data
    )
    strategy = MACDStrategy(data_handler, events_queue, suggested_quantity = 100,
                            short_window = short_window, long_window = long_window)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return backtest, results

def Multipro(config, trading_data, short_window, long_window):
    config['title'] = "MACDStrategy" + "_" + str(short_window) + "_" + str(long_window)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")
    backtest, results = run(config, trading_data, short_window=short_window, long_window=long_window)
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
    pd.DataFrame(dict_ans).to_csv(config['out_dir'] + "/" + 'result_' + config['title'] + '.csv')


if __name__ == '__main__':
    config_in = {
        "csv_dir": "C:/backtest/trades_Bitfinex_folder",
        "out_dir": "C:/backtest/backtest/results/MACDStrategy/multipro",
        "title": "MACDStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-01-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 100000.0,
        "freq": 1,  # min
        "commission_ratio": 0.001,
        "exchange": the exchange
        "tickers": ['ETHUSD']  # , 'BCHUSD', 'BCHBTC', 'BCHETH', 'EOSBTC']
        # "tickers": ['BCCBTC', 'BCCUSD', 'BCHBTC', 'BCHETH', 'BCHUSD',
        #             'ELFBTC', 'ELFETH', 'ELFUSD', 'EOSBTC', 'EOSETH',
        #             'EOSUSD', 'ETCBTC', 'ETCUSD', 'ETHBTC', 'ETHUSD',
        #             'IOSBTC', 'IOSETH', 'IOSUSD', 'LTCBTC', 'LTCUSD',
        #             'XRPBTC', 'XRPUSD']
    }

    trading_data = {}
    for ticker in config_in['tickers']:
        # trading_data[ticker] = open_gz_files(config_in['csv_dir'], ticker)
        trading_data[ticker] = pd.read_hdf(config_in['csv_dir'] + '\\' + ticker + '.h5', key=ticker)


    interval = np.array([5, 10, 12, 26, 30, 35, 45, 60, 72, 84, 96, 120, 252])
    # interval = np.array([5, 30, 120])

    pool = Pool(4)
    window = [(config_in, trading_data, interval[i], interval[j])
              for i in range(len(interval)) for j in range(i + 1, len(interval))]
    pool.starmap(Multipro, window)
    pool.close()
