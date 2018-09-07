import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os

from setting import set_env
global back_config
back_config = set_env()
from Backtest import *
from EMV_RSIStrategy import EMV_RSIStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

import GPyOpt

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

    return (results['cum_returns'][-1] - 1)


def running(X):
    config = {
        "csv_dir": back_config['data_folder'],
        "out_dir": os.path.join(back_config['output_folder'], "EMV_RSIStrategy"),
        "title": "EMV_RSIStrategy",
        "is_plot": False,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-02-01T00:0:00", freq = "60" + "T"),    # str(freq) + "T"
        "end_date": pd.Timestamp("2018-06-01T00:00:00", freq = "60" + "T"),
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


    window_EMV = int(X[0,0])
    n = X[0,1]
    m = X[0,2]
    window_RSI = int(X[0,3])
    s = int(X[0,4])
    b = X[0,5]

    return run_backtest(config, trading_data, ohlc_data, window_EMV, n, m, window_RSI, s, b)

if __name__ == "__main__":
    domain = [{'name': 'window_EMV', 'type': 'discrete', 'domain': tuple(range(1,240))},
              {'name': 'n', 'type': 'discrete', 'domain': tuple(range(1,120))},
              {'name': 'm', 'type': 'discrete', 'domain': tuple(range(1,240))},
              {'name': 'window_RSI', 'type': 'discrete', 'domain': tuple(range(1,240))},
              {'name': 's', 'type': 'discrete', 'domain': tuple(range(55,85))},
              {'name': 'b', 'type': 'discrete', 'domain': tuple(range(15,45))},
              ]



    batch_size = back_config['GPyOpt']['batch_size']
    num_cores = back_config['GPyOpt']['num_cores']

    from numpy.random import seed
    seed(123)

    BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=running,
                                                           domain=domain,
                                                           model_type='GP',
                                                           acquisition_type='EI',
                                                           normalize_Y=True,
                                                           initial_design_numdata=30,
                                                           initial_design_type='random',  # or 'grid',
                                                           evaluator_type='local_penalization',
                                                           batch_size=batch_size,
                                                           num_cores=num_cores,
                                                           acquisition_jitter=0)

    max_iter = 60
    BO_demo_parallel.run_optimization(max_iter)


