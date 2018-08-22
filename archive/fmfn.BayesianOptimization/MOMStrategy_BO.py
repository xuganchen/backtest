import numpy as np
import pandas as pd
import queue
import matplotlib.pyplot as plt

from BayesianOptimization.bayesian_optimization import BayesianOptimization

from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from Backtest.portfolio import PortfolioHandler
from MOMStrategy import MOMStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

def run_backtest(config, trading_data, ohlc_data, window, num, rev):
    window = int(window)
    num = int(num)
    config['title'] = "MOMStrategy" + "_" + str(window) + "_" + str(num) + '_' + str(rev)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    portfolio_handler = PortfolioHandler(
        config, data_handler, events_queue
    )
    strategy = MOMStrategy(config, events_queue, data_handler, portfolio_handler,
                         window = window, num = num, rev = rev)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler, portfolio_handler = portfolio_handler)

    results = backtest.start_trading()

    # dict_ans = {
    #     "window": [window],
    #     "num": [num],
    #     "rev": [rev],
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
        "out_dir": "C:/backtest/results/MOMStrategy",
        "title": "MOMStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2018-01-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-07-01T00:00:00", freq="60" + "T"),
        "equity": 1.0,
        "freq": 60,  # min
        "commission_ratio": 0.002,
        "suggested_quantity": None,     # None or a value
        "max_quantity": None,           # None or a value, Maximum purchase quantity
        "min_quantity": None,           # None or a value, Minimum purchase quantity
        "min_handheld_cash": None,      # None or a value, Minimum handheld funds
        "exchange": "Binance",
        "tickers": ['BTCUSDT', 'CMTBNB', 'CMTBTC', 'CMTETH',
                    'ETHUSDT', 'LTCUSDT', 'VENBNB', #  'EOSUSDT' 'XRPUSDT'
                   'VENBTC', 'VENETH']
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
        {'window': (1, 120),
         'num': (1, 5)},
        is_int=[1, 1],
        invariant={
            'config': config,
            'trading_data': trading_data,
            'ohlc_data': ohlc_data,
            'rev': 1
        },
        random_state=1
    )
    print(np.arange(1, 120, 10))
    BO.explore({
        'window': np.arange(1, 120, 10),
        'num': np.repeat(3, 12)
    },
        eager=True)
    BO.maximize(init_points=0, n_iter=50, acq="ei", xi=0.01, **gp_params)
    BO.maximize(init_points=0, n_iter=50, acq="ei", xi=0.0001, **gp_params)

    print(BO.res['max'])

    Target = pd.DataFrame({'Parameters': BO.X.tolist(), 'Target': BO.Y})
    Target.to_csv(config['out_dir'] + "/target_ei.csv")
    Target.sort_values(by="Target")