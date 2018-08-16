import numpy as np
import pandas as pd
import queue
import matplotlib.pyplot as plt

from BayesianOptimization.bayesian_optimization import BayesianOptimization

from Backtest.backtest import Backtest
from Backtest.data import OHLCDataHandler
from ADXStrategy import ADXStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars


def run_backtest(config, trading_data, ohlc_data, window):
    window = int(window)
    config['title'] = "ADXStrategy" + "_" + str(window)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")

    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data=trading_data, ohlc_data=ohlc_data
    )
    strategy = ADXStrategy(config, events_queue, data_handler,
                           window=window)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler=data_handler)

    results = backtest.start_trading()

    # dict_ans = {
    #     "window": [window],
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

def plot_bo(window, bo, filename):
    xs = [x["window"] for x in bo.res["all"]["params"]]
    ys = bo.res["all"]["values"]

    length = window[1]
    mean, sigma = bo.gp.predict(np.arange(length).reshape(-1, 1), return_std=True)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(np.arange(length), mean)
    plt.fill_between(np.arange(length), mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, length)
    plt.hlines(0, 0, 240, hold=None,color='black',linestyle="--")
    plt.hlines(-1, 0, 240, hold=None,color='black',linestyle="--")
    plt.hlines(1, 0, 240, hold=None,color='black',linestyle="--")
    plt.show()
    fig.savefig(filename, dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    config = {
        "csv_dir": "C:/backtest/Binance",
        "out_dir": "C:/backtest/results/ADXStrategy",
        "title": "ADXStrategy",
        "is_plot": False,
        "save_plot": False,
        "save_tradelog": False,
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
        {'window': (1, 240)},
        is_int=[1],
        invariant={
            'config': config,
            'trading_data': trading_data,
            'ohlc_data': ohlc_data
        },
        random_state=1
    )

    BO.explore({
        'window': np.arange(1, 240, 20)
    },
        eager=True)

    # BO.maximize(init_points=0, n_iter=50, acq='ucb', kappa=5, **gp_params)
    BO.maximize(init_points=0, n_iter=50, acq="ei", xi=0.01, **gp_params)
    print(BO.res['max'])

    Target = pd.DataFrame({'Parameters': BO.X.tolist(), 'Target': BO.Y})
    Target.to_csv(config['out_dir'] + "/target_ucb2.csv")

    filename = config['out_dir'] + "/target_ucb2.png"
    plot_bo((0, 240), BO, filename)

