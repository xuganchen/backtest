import numpy as np
import pandas as pd
import queue
import matplotlib.pyplot as plt

from BayesianOptimization.bayesian_optimization import BayesianOptimization

from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from KDJStrategy import KDJStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars


def run_backtest(config, trading_data, ohlc_data,
                 window=10, sK=20, bK=80, delta=10):
    window = int(window)
    sK = int(sK)
    sD = sK
    sJ = sK - delta
    bK = int(bK)
    bD = bK
    bJ = bK + delta
    config['title'] = "EMVStrategy" + "_" + str(window) + "_" + str(sK) + "_" + str(bK) + "_" + str(delta)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data=trading_data, ohlc_data=ohlc_data
    )
    strategy = KDJStrategy(config, events_queue, data_handler,
                           window=window, sK=sK, sD=sD, sJ=sJ, bK=bK, bD=bD, bJ=bJ)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler=data_handler)

    results = backtest.start_trading()

    # dict_ans = {
    #     "window": [window],
    #     "sK/sD/sJ": [(sK, sD, sJ)],
    #     "bK/bD/bJ": [(bK, bD, bJ)],
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
    return (results['cum_returns'][-1] - 1)


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/KDJStrategy",
        "title": "KDJStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-07-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
        "equity": 1.0,
        "freq": 60,  # min
        "commission_ratio": 0.001,
        "suggested_quantity": None,  # None or a value
        "max_quantity": None,  # None or a value, Maximum purchase quantity
        "min_quantity": None,  # None or a value, Minimum purchase quantity
        "min_handheld_cash": None,  # None or a value, Minimum handheld funds
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
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker + '_OHLC_60min.h5', key=ticker)

    trading_data = None


    gp_params = {"alpha": 1e-5}

    BO = BayesianOptimization(
        run_backtest,
        pbounds={'window': (1, 120),
                 'sK': (10, 30),
                 'bK': (70, 90),
                 'delta': (0, 20)},
        is_int=[1, 1, 1, 1],
        invariant={
            'config': config,
            'trading_data': trading_data,
            'ohlc_data': ohlc_data
        },
        random_state=1
    )
    BO.explore({
        'window': np.arange(1, 120, 12),
        'sK': np.arange(10, 30, 2),
        'bK': np.arange(90, 70, -2),
        'delta': np.repeat(10, 10)
    },
        eager=True)
    BO.maximize(init_points=0, n_iter=120, acq="ei", xi=0.1, **gp_params)
    BO.maximize(init_points=0, n_iter=120, acq="ei", xi=0.0001, **gp_params)

    print(BO.res['max'])

    Target = pd.DataFrame({'Parameters': BO.X.tolist(), 'Target': BO.Y})
    Target.to_csv(config['out_dir'] + "/target_ei.csv")
    Target.sort_values(by="Target")

