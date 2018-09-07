import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os
import GPyOpt

from setting import set_env
global back_config
back_config = set_env()
from Backtest import *
from MACD_EMVStrategy import MACD_EMVStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars


def run_backtest(config, trading_data, ohlc_data, short_window = 10, long_window = 40, window=40, n=10, m=10):
    config['title'] = "MACD_EMVStrategy" + "_" + str(short_window) + "_" + str(long_window) + "_" + str(window) + "_" + str(n) + "_" + str(m)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = MACD_EMVStrategy(config, events_queue, data_handler,
                            short_window=short_window, long_window=long_window,
                           window=window, n=n, m=m)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return (results['cum_returns'][-1] - 1)


def running(X):
    config = {
        "csv_dir": back_config['data_folder'],
        "out_dir": os.path.join(back_config['output_folder'], "MACD_EMVStrategy"),
        "title": "MACD_EMVStrategy",
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

    short_window = int(X[0,0])
    long_window = int(X[0,1])
    window = int(X[0,2])
    n = int(X[0,3])
    m = int(X[0,4])

    return run_backtest(config, trading_data, ohlc_data, short_window, long_window , window, n, m)

if __name__ == "__main__":
    domain = [{'name': 'short_window', 'type': 'discrete', 'domain': tuple(range(1,120))},
              {'name': 'long_window', 'type': 'discrete', 'domain': tuple(range(1,240))},
              {'name': 'window', 'type': 'discrete', 'domain': tuple(range(1,240))},
              {'name': 'n', 'type': 'discrete', 'domain': tuple(range(1,120))},
              {'name': 'm', 'type': 'discrete', 'domain': tuple(range(1,240))},
              ]
    constraints = [{'name': 'constr_1', 'constraint': 'x[:,0] - x[:,1]'},
              ]



    batch_size = back_config['GPyOpt']['batch_size']
    num_cores = back_config['GPyOpt']['num_cores']

    from numpy.random import seed
    seed(123)

    BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=running,
                                                           domain=domain,
                                                           constraints=constraints, 
                                                           model_type='GP',
                                                           acquisition_type='EI',
                                                           normalize_Y=True,
                                                           initial_design_numdata=25,
                                                           initial_design_type='random',  # or 'gird',
                                                           evaluator_type='local_penalization',
                                                           batch_size=batch_size,
                                                           num_cores=num_cores,
                                                           acquisition_jitter=0)

    max_iter = 45
    BO_demo_parallel.run_optimization(max_iter)

