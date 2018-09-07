import numpy as np
import pandas as pd
import queue
from multiprocessing import Pool
import os


from setting import set_env
global back_config
back_config = set_env()
from Backtest import *
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

class MOMStrategy(Strategy):
    def __init__(self, config, events, data_handler, portfolio_handler,
                 window = 10, num = 3, rev = 0):  
        # rev = 0: buy first num, rev = 1: buy last num
        self.config = config
        self.data_handler = data_handler
        self.portfolio_handler = portfolio_handler
        self.tickers = self.config['tickers']
        self.events = events
        self.holdinds = self._calculate_initial_holdings()
        
        self.bar_date = None
        self.window = window
        self.num = num
        self.rev = rev

    def _calculate_initial_holdings(self):
        holdings = {}
        for s in self.tickers:
            holdings[s] = "EMPTY"
        return holdings
    
    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            bar_date = event.timestamp
            if self.bar_date == bar_date:
                pass
            else:
                self.bar_date = bar_date
                bars = {}
                for ticker in self.tickers:
                    bars[ticker] = self.data_handler.get_latest_bars_values(
                                    ticker, "close", N=self.window
                                    )
                bars = pd.DataFrame(bars).dropna(axis=1)

                if len(bars) > 0:
                    change = (bars.iloc[-1,:] - bars.iloc[0,:]) / bars.iloc[0,:]
                    change = change.sort_values(ascending= False)

                    current_tickers = self.portfolio_handler.current_tickers
                    cash_for_order = self.portfolio_handler.cash_for_order
                    
                    if self.rev == 0:
                        MOM_tickers = change.iloc[:self.num].keys()
                    elif self.rev == 1:
                        MOM_tickers = change.iloc[-self.num:].keys()
                    
                    for ticker in current_tickers:
                        if ticker not in MOM_tickers:
                            self.generate_sell_signals(ticker, bar_date, "SHORT")
                            quantity = self.portfolio_handler.current_tickers_info[ticker]['quantity']
                            event_price = bars.loc[len(bars) - 1, ticker]
                            cash_for_order += (event_price * quantity) * (1-self.portfolio_handler.commission_ratio)

                    MOM_num = sum([ticker not in current_tickers for ticker in MOM_tickers])
                    if MOM_num != 0:
                        suggested_cash = cash_for_order / MOM_num
                        for ticker in MOM_tickers:
                            if ticker not in current_tickers:
                                self.generate_buy_signals(ticker, bar_date, "LONG", suggested_cash = suggested_cash)



def run_backtest(config, trading_data, ohlc_data, window = 10, num = 4, rev = 1):
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
    return backtest, results


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


if __name__ == "__main__":

    config = {
        "csv_dir": back_config['data_folder'],
        "out_dir": os.path.join(back_config['output_folder'], "MOMStrategy"),
        "title": "MOMStrategy",
        "is_plot": True,
        "save_plot": True,
        "save_tradelog": True,
        "start_date": pd.Timestamp("2017-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
        "end_date": pd.Timestamp("2018-04-01T00:00:00", freq="60" + "T"),
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
        ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '/' + ticker +'_OHLC_60min.h5', key=ticker)

    trading_data = None

    backtest, results = run_backtest(config, trading_data, ohlc_data, window = 97, num = 3, rev = 1)




    # interval = np.array([5, 10, 12, 26, 30, 35, 45, 60, 72, 84, 96, 120, 252])
    # # interval = np.array([5, 12, 26, 45, 60, 96, 120, 252])
    # # interval = np.array([5, 30, 120])
    # num_inter = np.array([1, 2, 3, 4])
    # rev = 1

    # pool = Pool(4)
    # results = []
    # for i in range(len(interval)):
    #     for j in range(len(num_inter)):
    #         window = interval[i]
    #         num = num_inter[j]
    #         result = pool.apply_async(run, args=(config, trading_data, ohlc_data, window, num, rev,))
    #         results.append(result)

    # ans = pd.DataFrame()
    # for results in results:
    #     df = results.get()
    #     ans = pd.concat([ans, df], ignore_index=True)
    # pool.close()

    # if not os.path.exists(config['out_dir']):
    #     os.makedirs(config['out_dir'])
    # ans = ans.sort_values(by="Total Returns", ascending=False)
    # ans.to_csv(config['out_dir'] + "/result_MOMStrategy.csv")